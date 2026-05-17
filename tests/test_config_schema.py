from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from mcis.config_schema import (
    DataValidityMode,
    ClaimLevel,
    FeaturesConfig,
    validate_config,
    _deep_merge,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture
def full_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class TestDeepMerge:
    def test_override_scalar(self):
        base = {"a": 1, "b": 2}
        override = {"a": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 99, "b": 2}

    def test_nested_merge(self):
        base = {"x": {"y": 1, "z": 2}, "w": 3}
        override = {"x": {"y": 99}}
        result = _deep_merge(base, override)
        assert result == {"x": {"y": 99, "z": 2}, "w": 3}

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_empty_override(self):
        base = {"a": 1, "b": {"c": 2}}
        result = _deep_merge(base, {})
        assert result == base

    def test_empty_base(self):
        override = {"a": 1}
        result = _deep_merge({}, override)
        assert result == override


class TestValidateConfig:
    def test_valid_from_file(self):
        result = validate_config(str(CONFIG_PATH))
        assert result["project"]["data_validity_mode"] == "empirical"
        assert result["conflict"]["t0"] == "2022-02-24"
        assert len(result["aggregation"]["metrics"]) == 17

    def test_valid_from_dict(self, full_config):
        result = validate_config(full_config)
        assert result["project"]["name"] == "mcis"
        assert result["spatial"]["lon_min"] == 27.5

    def test_partial_config_merged_with_defaults(self):
        partial = {
            "project": {"data_validity_mode": "synthetic",
                        "claim_level": "engineering_demo",
            },
            "conflict": {
                "t0": "2022-02-24",
                "event_study_windows": [-7, 0, 7],
            },
            "cleaning": {
                "lon_bounds": [27, 42],
                "lat_bounds": [40, 47],
                "navstatus_unknown": [95],
            },
            "features": {
                "military_vessel_types": ["SAR"],
                "cargo_vessel_types": ["CARGO"],
                "tanker_types": ["TANKER"],
                "flag_risk_groups": {
                    "russia": ["RU"],
                    "ukraine": ["UA"],
                    "nato": ["TR"],
                    "convenience": ["PA"],
                },
            },
            "aggregation": {"metrics": ["vessel_count", "mean_sog"]},
            "model": {
                "train_normal_start": "2021-08-24",
                "train_normal_end": "2021-12-25",
                "calibration_start": "2021-12-26",
                "calibration_end": "2022-01-24",
                "event_eval_start": "2022-01-25",
                "event_eval_end": "2022-02-23",
                "post_event_start": "2022-02-24",
                "post_event_end": "2022-08-24",
                "features_to_use": ["vessel_count"],
            },
            "validation": {"forbidden_features": []},
        }
        result = validate_config(partial)
        assert result["project"]["data_validity_mode"] == "synthetic"
        assert result["project"]["random_seed"] == 42
        assert result["temporal"]["time_bucket"] == "1D"
        assert result["output"]["dpi"] == 300
        assert result["validation"]["min_rows_per_day"] == 10

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_config("nonexistent_config.yaml")


MINIMAL: dict = {
    "conflict": {"t0": "2022-02-24", "event_study_windows": [-7, 0, 7]},
    "cleaning": {"lon_bounds": [27, 42], "lat_bounds": [40, 47], "navstatus_unknown": [95]},
    "features": {
        "military_vessel_types": ["SAR"], "cargo_vessel_types": ["CARGO"],
        "tanker_types": ["TANKER"],
        "flag_risk_groups": {"russia": ["RU"], "ukraine": ["UA"], "nato": ["TR"], "convenience": ["PA"]},
    },
    "aggregation": {"metrics": ["vessel_count", "mean_sog"]},
    "model": {
        "train_normal_start": "2021-08-24", "train_normal_end": "2021-12-25",
        "calibration_start": "2021-12-26", "calibration_end": "2022-01-24",
        "event_eval_start": "2022-01-25", "event_eval_end": "2022-02-23",
        "post_event_start": "2022-02-24", "post_event_end": "2022-08-24",
        "features_to_use": ["vessel_count"],
    },
    "validation": {"forbidden_features": []},
}


class TestProjectValidation:
    def test_synthetic_inferential_blocked(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "synthetic", "claim_level": "inferential_evidence"}
        with pytest.raises(ValueError, match="claim_level"):
            validate_config(d)

    def test_synthetic_predictive_blocked(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "synthetic", "claim_level": "predictive_prototype"}
        with pytest.raises(ValueError, match="claim_level"):
            validate_config(d)

    def test_empirical_inferential_allowed(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "empirical", "claim_level": "inferential_evidence"}
        result = validate_config(d)
        assert result["project"]["data_validity_mode"] == "empirical"
        assert result["project"]["claim_level"] == "inferential_evidence"

    def test_empirical_predictive_allowed(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "empirical", "claim_level": "predictive_prototype"}
        result = validate_config(d)
        assert result["project"]["claim_level"] == "predictive_prototype"

    def test_mixed_engineering_ok(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "mixed", "claim_level": "engineering_demo"}
        result = validate_config(d)
        assert result["project"]["claim_level"] == "engineering_demo"

    def test_mixed_inferential_blocked(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "mixed", "claim_level": "inferential_evidence"}
        with pytest.raises(ValueError, match="claim_level"):
            validate_config(d)

    def test_random_seed_negative(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "synthetic", "random_seed": -1}
        with pytest.raises(ValueError, match="random_seed"):
            validate_config(d)

    def test_invalid_mode(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "bogus"}
        with pytest.raises(ValueError, match="data_validity_mode"):
            validate_config(d)

    def test_invalid_claim_level(self):
        d = copy.deepcopy(MINIMAL)
        d["project"] = {"data_validity_mode": "synthetic", "claim_level": "fake_level"}
        with pytest.raises(ValueError, match="claim_level"):
            validate_config(d)


class TestSpatialValidation:
    def test_lon_min_gte_lon_max(self):
        d = copy.deepcopy(MINIMAL)
        d["spatial"] = {"lon_min": 50, "lon_max": 40, "lat_min": 30, "lat_max": 40}
        with pytest.raises(ValueError, match="lon_min"):
            validate_config(d)

    def test_lat_min_gte_lat_max(self):
        d = copy.deepcopy(MINIMAL)
        d["spatial"] = {"lon_min": 27, "lon_max": 42, "lat_min": 50, "lat_max": 40}
        with pytest.raises(ValueError, match="lat_min"):
            validate_config(d)

    def test_grid_resolution_zero(self):
        d = copy.deepcopy(MINIMAL)
        d["spatial"] = {"lon_min": 27, "lon_max": 42, "lat_min": 40, "lat_max": 47, "grid_resolution_deg": 0}
        with pytest.raises(ValueError, match="grid_resolution_deg"):
            validate_config(d)


class TestCleaningValidation:
    def test_lon_bounds_wrong_length(self):
        d = copy.deepcopy(MINIMAL)
        d["cleaning"]["lon_bounds"] = [1, 2, 3]
        with pytest.raises(ValueError, match="lon_bounds"):
            validate_config(d)

    def test_lat_bounds_reversed(self):
        d = copy.deepcopy(MINIMAL)
        d["cleaning"]["lat_bounds"] = [50, 40]
        with pytest.raises(ValueError, match="lat_bounds"):
            validate_config(d)

    def test_sog_min_gt_max(self):
        d = copy.deepcopy(MINIMAL)
        d["cleaning"]["sog_min"] = 80
        d["cleaning"]["sog_max"] = 30
        with pytest.raises(ValueError, match="sog_min"):
            validate_config(d)

    def test_min_vessel_obs_zero(self):
        d = copy.deepcopy(MINIMAL)
        d["cleaning"]["min_vessel_obs"] = 0
        with pytest.raises(ValueError, match="min_vessel_obs"):
            validate_config(d)


class TestModelDateOrdering:
    def test_reversed_dates(self):
        d = copy.deepcopy(MINIMAL)
        d["model"]["train_normal_start"] = "2021-12-25"
        d["model"]["train_normal_end"] = "2021-08-24"
        with pytest.raises(ValueError, match="train_normal_start"):
            validate_config(d)

    def test_calibration_before_train(self):
        d = copy.deepcopy(MINIMAL)
        d["model"]["train_normal_end"] = "2022-01-15"
        d["model"]["calibration_start"] = "2022-01-10"
        d["model"]["calibration_end"] = "2022-01-20"
        with pytest.raises(ValueError, match="train_normal_end"):
            validate_config(d)


class TestFeaturesInMetrics:
    def test_unknown_feature(self):
        d = copy.deepcopy(MINIMAL)
        d["aggregation"]["metrics"] = ["vessel_count"]
        d["model"]["features_to_use"] = ["nonexistent_metric"]
        with pytest.raises(ValueError, match="features_to_use"):
            validate_config(d)

    def test_valid_subset(self):
        d = copy.deepcopy(MINIMAL)
        d["aggregation"]["metrics"] = ["vessel_count", "mean_sog", "cog_variance"]
        d["model"]["features_to_use"] = ["vessel_count", "cog_variance"]
        result = validate_config(d)
        assert result["model"]["features_to_use"] == ["vessel_count", "cog_variance"]


class TestFlagRiskGroups:
    def test_missing_key(self):
        with pytest.raises(ValueError, match="flag_risk_groups"):
            FeaturesConfig(
                ais_silence_gap_hours=24,
                rot_spike_threshold=20.0,
                speed_anomaly_zscore=3.0,
                entropy_time_window="7D",
                military_vessel_types=["SAR"],
                cargo_vessel_types=["CARGO"],
                tanker_types=["TANKER"],
                flag_risk_groups={"russia": ["RU"], "ukraine": ["UA"]},
            )


class TestOutputValidation:
    def test_invalid_figure_format(self):
        d = copy.deepcopy(MINIMAL)
        d["output"] = {"figure_format": "gif"}
        with pytest.raises(ValueError, match="figure_format"):
            validate_config(d)

    def test_dpi_too_low(self):
        d = copy.deepcopy(MINIMAL)
        d["output"] = {"dpi": 10}
        with pytest.raises(ValueError, match="dpi"):
            validate_config(d)

    def test_dpi_too_high(self):
        d = copy.deepcopy(MINIMAL)
        d["output"] = {"dpi": 9999}
        with pytest.raises(ValueError, match="dpi"):
            validate_config(d)

    def test_case_insensitive_format(self):
        d = copy.deepcopy(MINIMAL)
        d["output"] = {"figure_format": "PDF"}
        result = validate_config(d)
        assert result["output"]["figure_format"] == "pdf"


class TestEnumValues:
    def test_data_validity_mode_values(self):
        assert DataValidityMode("empirical").value == "empirical"
        assert DataValidityMode("synthetic").value == "synthetic"
        assert DataValidityMode("mixed").value == "mixed"

    def test_claim_level_values(self):
        assert ClaimLevel("engineering_demo").value == "engineering_demo"
        assert ClaimLevel("descriptive_evidence").value == "descriptive_evidence"
        assert ClaimLevel("inferential_evidence").value == "inferential_evidence"
        assert ClaimLevel("predictive_prototype").value == "predictive_prototype"
