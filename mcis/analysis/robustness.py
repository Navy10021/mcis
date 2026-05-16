from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from mcis.analysis.event_study import run_event_study


def run_placebo_cutdates(
    panel: pd.DataFrame,
    metric: str,
    candidate_offsets: list[int],
    estimation_window: tuple[int, int] = (-90, -31),
    event_window: tuple[int, int] = (-30, 30),
) -> dict[str, Any]:
    p_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for offset in candidate_offsets:
        shifted = panel.copy()
        if "days_to_t0" not in shifted.columns:
            raise ValueError("Panel must contain 'days_to_t0' column")
        shifted["days_to_t0"] = shifted["days_to_t0"] - offset
        result = run_event_study(
            shifted,
            metric=metric,
            estimation_window=estimation_window,
            event_window=event_window,
        )
        day0_p = result.get("p_values", {}).get("0", np.nan)
        p = float(day0_p) if np.isfinite(day0_p) else 1.0
        p_values.append(p)
        rows.append({"offset_days": int(offset), "day0_p_value": p, "status": result.get("status")})

    p_arr = np.array(p_values, dtype=float)
    return {
        "n_placebos": int(len(rows)),
        "min_p_value": float(np.nanmin(p_arr)) if len(p_arr) else None,
        "median_p_value": float(np.nanmedian(p_arr)) if len(p_arr) else None,
        "rows": rows,
    }


def apply_multiple_testing(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    keys = list(p_values.keys())
    vals = np.array([float(p_values[k]) for k in keys], dtype=float)
    reject_fdr, p_fdr, _, _ = multipletests(vals, alpha=alpha, method="fdr_bh")
    reject_holm, p_holm, _, _ = multipletests(vals, alpha=alpha, method="holm")
    adjusted = {}
    for i, key in enumerate(keys):
        adjusted[key] = {
            "raw_p_value": float(vals[i]),
            "fdr_bh_p_value": float(p_fdr[i]),
            "fdr_bh_reject": bool(reject_fdr[i]),
            "holm_p_value": float(p_holm[i]),
            "holm_reject": bool(reject_holm[i]),
        }
    return {"alpha": alpha, "adjusted_p_values": adjusted}
