from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mcis.utils.io import ensure_dir


_VALID_VALIDITY_MODES = {"empirical", "synthetic", "mixed"}
_VALID_CLAIM_LEVELS = {
    "engineering_demo",
    "descriptive_evidence",
    "inferential_evidence",
    "predictive_prototype",
}
_ALLOWED_INFERENTIAL_MODES = {"empirical"}


def validate_data_validity_mode(config: dict[str, Any]) -> None:
    mode = config.get("project", {}).get("data_validity_mode")
    if mode is None:
        raise ValueError("config.project.data_validity_mode is required")
    if mode not in _VALID_VALIDITY_MODES:
        raise ValueError(
            f"Invalid data_validity_mode: {mode!r}. "
            f"Must be one of {sorted(_VALID_VALIDITY_MODES)}"
        )


def validate_claim_level(config: dict[str, Any]) -> None:
    mode = config.get("project", {}).get("data_validity_mode", "synthetic")
    level = config.get("project", {}).get("claim_level")
    if level is None:
        raise ValueError("config.project.claim_level is required")
    if level not in _VALID_CLAIM_LEVELS:
        raise ValueError(
            f"Invalid claim_level: {level!r}. "
            f"Must be one of {sorted(_VALID_CLAIM_LEVELS)}"
        )
    if (
        level in ("inferential_evidence", "predictive_prototype")
        and mode not in _ALLOWED_INFERENTIAL_MODES
    ):
        raise ValueError(
            f"claim_level={level!r} requires data_validity_mode='empirical', "
            f"but mode is {mode!r}. Inferential claims are not allowed for "
            f"non-empirical data."
        )


def assert_no_leakage(
    feature_cols: list[str],
    config: dict[str, Any],
) -> None:
    forbidden = set(config.get("validation", {}).get("forbidden_features", []))
    leakage = sorted(set(feature_cols) & forbidden)
    if leakage:
        raise ValueError(
            f"Leakage features detected in model input: {leakage}. "
            f"These features must not be used as predictors."
        )


def validate_temporal_split(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    date_col = "date"
    if date_col not in panel.columns:
        if isinstance(panel.index, pd.MultiIndex):
            date_level = panel.index.get_level_values("date")
            if date_level is not None:
                dates = pd.Series(date_level.unique()).sort_values()
            else:
                raise ValueError("Panel must have a 'date' column or index level")
        else:
            raise ValueError(f"Panel must have a '{date_col}' column")
    else:
        dates = panel[date_col].dropna().unique()
        dates = pd.Series(dates).sort_values()

    model_cfg = config.get("model", {})
    splits = {
        "train": (model_cfg.get("train_normal_start"), model_cfg.get("train_normal_end")),
        "calibration": (model_cfg.get("calibration_start"), model_cfg.get("calibration_end")),
        "event_eval": (model_cfg.get("event_eval_start"), model_cfg.get("event_eval_end")),
        "post_event": (model_cfg.get("post_event_start"), model_cfg.get("post_event_end")),
    }

    result: dict[str, Any] = {
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "total_dates": int(len(dates)),
        "splits": {},
    }

    for name, (start, end) in splits.items():
        if start and end:
            mask = (dates >= start) & (dates <= end)
            n_dates = int(mask.sum())
            result["splits"][name] = {
                "start": start,
                "end": end,
                "n_dates": n_dates,
            }
        else:
            result["splits"][name] = None

    return result


def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"Required columns missing from DataFrame: {missing}"
        )


def write_run_metadata(
    config: dict[str, Any],
    output_dir: str | Path,
    stage_name: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{stage_name}_{timestamp}.json"
    path = output_dir / filename

    metadata: dict[str, Any] = {
        "stage": stage_name,
        "timestamp": timestamp,
        "data_validity_mode": config.get("project", {}).get("data_validity_mode"),
        "claim_level": config.get("project", {}).get("claim_level"),
        "conflict_t0": config.get("conflict", {}).get("t0"),
        "conflict_zone": config.get("conflict", {}).get("zone_name"),
    }
    if extra:
        metadata.update(extra)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return path
