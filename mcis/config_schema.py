from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic import BaseModel


class DataValidityMode(str, Enum):
    empirical = "empirical"
    synthetic = "synthetic"
    mixed = "mixed"


class ClaimLevel(str, Enum):
    engineering_demo = "engineering_demo"
    descriptive_evidence = "descriptive_evidence"
    inferential_evidence = "inferential_evidence"
    predictive_prototype = "predictive_prototype"


ALLOWED_INFERENTIAL_MODES: set[DataValidityMode] = {DataValidityMode.empirical}


class ProjectConfig(BaseModel):
    name: str = "mcis"
    random_seed: int = Field(default=42, ge=0)
    data_validity_mode: DataValidityMode = DataValidityMode.mixed
    claim_level: ClaimLevel = ClaimLevel.descriptive_evidence

    @model_validator(mode="after")
    def validate_claim_level_against_mode(self) -> ProjectConfig:
        if self.claim_level in (
            ClaimLevel.inferential_evidence,
            ClaimLevel.predictive_prototype,
        ) and self.data_validity_mode not in ALLOWED_INFERENTIAL_MODES:
            raise ValueError(
                f"claim_level={self.claim_level.value!r} requires "
                f"data_validity_mode='empirical', but mode is "
                f"{self.data_validity_mode.value!r}"
            )
        return self


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    aggregated_dir: str = "data/aggregated"
    primary_file: str = "ais_blacksea_12m.csv"
    dev_file: str = "ais_blacksea_6m.csv"


class ConflictConfig(BaseModel):
    t0: date
    zone_name: str = "Black Sea / Sea of Azov"
    pre_window_days: int = Field(default=90, ge=1)
    post_window_days: int = Field(default=90, ge=1)
    event_study_windows: list[int]


class SpatialConfig(BaseModel):
    lon_min: float = Field(default=27.5, ge=-180, le=180)
    lon_max: float = Field(default=41.5, ge=-180, le=180)
    lat_min: float = Field(default=40.5, ge=-90, le=90)
    lat_max: float = Field(default=46.8, ge=-90, le=90)
    grid_resolution_deg: float = Field(default=0.5, gt=0, le=10)
    grid_resolution_fine_deg: float = Field(default=0.25, gt=0, le=10)

    @model_validator(mode="after")
    def validate_bounds(self) -> SpatialConfig:
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max})")
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max})")
        return self


class TemporalConfig(BaseModel):
    time_bucket: str = "1D"
    time_bucket_fine: str = "6h"
    timestamp_col: str = "posDt"


class CleaningConfig(BaseModel):
    sog_max: float = Field(default=50.0, gt=0, le=120)
    sog_min: float = Field(default=0.0, ge=0)
    rot_clamp: float = Field(default=127.0, gt=0)
    rot_no_info_value: float = 128.0
    cog_valid_max: float = 360.0
    lon_bounds: list[float]
    lat_bounds: list[float]
    heading_no_info: int = 511
    navstatus_unknown: list[int]
    length_zero_as_null: bool = True
    width_zero_as_null: bool = True
    min_vessel_obs: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> CleaningConfig:
        if len(self.lon_bounds) != 2:
            raise ValueError(f"lon_bounds must have 2 elements, got {len(self.lon_bounds)}")
        if self.lon_bounds[0] >= self.lon_bounds[1]:
            raise ValueError(
                f"lon_bounds[0] ({self.lon_bounds[0]}) must be < lon_bounds[1] ({self.lon_bounds[1]})"
            )
        if len(self.lat_bounds) != 2:
            raise ValueError(f"lat_bounds must have 2 elements, got {len(self.lat_bounds)}")
        if self.lat_bounds[0] >= self.lat_bounds[1]:
            raise ValueError(
                f"lat_bounds[0] ({self.lat_bounds[0]}) must be < lat_bounds[1] ({self.lat_bounds[1]})"
            )
        return self

    @model_validator(mode="after")
    def validate_sog_range(self) -> CleaningConfig:
        if self.sog_min > self.sog_max:
            raise ValueError(
                f"sog_min ({self.sog_min}) must be <= sog_max ({self.sog_max})"
            )
        return self


class FeaturesConfig(BaseModel):
    ais_silence_gap_hours: float = Field(default=24, gt=0)
    rot_spike_threshold: float = Field(default=20.0, ge=0)
    speed_anomaly_zscore: float = Field(default=3.0, ge=0)
    entropy_time_window: str = "7D"
    military_vessel_types: list[str]
    cargo_vessel_types: list[str]
    tanker_types: list[str]
    flag_risk_groups: dict[str, list[str]]

    @field_validator("flag_risk_groups")
    @classmethod
    def validate_flag_groups(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        expected = {"russia", "ukraine", "nato", "convenience"}
        actual = set(v.keys())
        missing = expected - actual
        if missing:
            raise ValueError(f"flag_risk_groups missing required keys: {sorted(missing)}")
        return v


class AggregationConfig(BaseModel):
    metrics: list[str]


class AnalysisConfig(BaseModel):
    significance_level: float = Field(default=0.05, ge=0, le=1)
    granger_max_lag: int = Field(default=14, ge=1)
    its_polynomial_degree: int = Field(default=2, ge=0)
    did_control_zone: str = "mediterranean"
    event_study_metric: str = "vessel_count"


class ModelConfig(BaseModel):
    formulation: str = "early_warning_anomaly"
    target_col: str = "conflict_onset"
    lookahead_days: int = Field(default=7, ge=1)
    early_warning_window_days: int = Field(default=30, ge=1)
    train_normal_start: date
    train_normal_end: date
    calibration_start: date
    calibration_end: date
    event_eval_start: date
    event_eval_end: date
    post_event_start: date
    post_event_end: date
    features_to_use: list[str]

    @model_validator(mode="after")
    def validate_date_ordering(self) -> ModelConfig:
        dates = [
            ("train_normal_start", self.train_normal_start),
            ("train_normal_end", self.train_normal_end),
            ("calibration_start", self.calibration_start),
            ("calibration_end", self.calibration_end),
            ("event_eval_start", self.event_eval_start),
            ("event_eval_end", self.event_eval_end),
            ("post_event_start", self.post_event_start),
            ("post_event_end", self.post_event_end),
        ]
        for i in range(len(dates) - 1):
            name_a, val_a = dates[i]
            name_b, val_b = dates[i + 1]
            if val_a > val_b:
                raise ValueError(
                    f"Model date ordering violation: {name_a}={val_a} > {name_b}={val_b}. "
                    f"Dates must be in chronological order."
                )
        return self


class ValidationConfig(BaseModel):
    forbidden_features: list[str]
    min_rows_per_day: int = Field(default=10, ge=0)
    min_unique_mmsi_per_day: int = Field(default=3, ge=0)


class OutputConfig(BaseModel):
    figures_dir: str = "outputs/figures"
    tables_dir: str = "outputs/tables"
    models_dir: str = "outputs/models"
    model_registry_dir: str = "outputs/models/registry"
    dpi: int = Field(default=300, ge=72, le=1200)
    figure_format: str = "png"

    @field_validator("figure_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"png", "pdf", "svg", "jpg", "jpeg", "eps"}
        if v.lower() not in allowed:
            raise ValueError(f"figure_format must be one of {sorted(allowed)}, got {v!r}")
        return v.lower()


class Settings(BaseModel):
    project: ProjectConfig
    data: DataConfig
    conflict: ConflictConfig
    spatial: SpatialConfig
    temporal: TemporalConfig
    cleaning: CleaningConfig
    features: FeaturesConfig
    aggregation: AggregationConfig
    analysis: AnalysisConfig
    model: ModelConfig
    validation: ValidationConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_spatial_and_cleaning_bounds_align(self) -> Settings:
        if (
            abs(self.spatial.lon_min - self.cleaning.lon_bounds[0]) > 0.01
            or abs(self.spatial.lon_max - self.cleaning.lon_bounds[1]) > 0.01
            or abs(self.spatial.lat_min - self.cleaning.lat_bounds[0]) > 0.01
            or abs(self.spatial.lat_max - self.cleaning.lat_bounds[1]) > 0.01
        ):
            import warnings
            warnings.warn(
                "spatial bounds differ from cleaning.lon_bounds/lat_bounds. "
                "Consider keeping them in sync."
            )
        return self

    @model_validator(mode="after")
    def validate_features_subset_of_metrics(self) -> Settings:
        model_features = set(self.model.features_to_use)
        agg_metrics = set(self.aggregation.metrics)
        unknown = model_features - agg_metrics
        if unknown:
            raise ValueError(
                f"model.features_to_use contains metrics not in aggregation.metrics: "
                f"{sorted(unknown)}"
            )
        return self


ROOT_SECTION_ORDER: ClassVar[list[str]] = [
    "project", "data", "conflict", "spatial", "temporal", "cleaning",
    "features", "aggregation", "analysis", "model", "validation", "output",
]

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, preserving base values for missing keys."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def validate_config(
    config_or_path: dict[str, Any] | str | Path,
    *,
    defaults_path: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        raw = config_or_path

    defaults_path = Path(defaults_path) if defaults_path else _DEFAULT_CONFIG_PATH
    if defaults_path and defaults_path.exists():
        with open(defaults_path, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f)
        merged = _deep_merge(defaults, raw)
    else:
        merged = raw

    settings = Settings(**merged)

    return settings.model_dump(mode="json")
