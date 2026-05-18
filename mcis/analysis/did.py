from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_did(
    panel: pd.DataFrame,
    treated_grid_ids: list[str],
    control_grid_ids: list[str],
    event_date: str = "2022-02-24",
    metric: str = "vessel_count",
    config: dict | None = None,
) -> dict[str, Any]:
    if "grid_id" not in panel.index.names:
        raise ValueError("Panel must have MultiIndex with 'grid_id' level")
    if "time_bucket" not in panel.index.names:
        raise ValueError("Panel must have MultiIndex with 'time_bucket' level")
    if metric not in panel.columns:
        raise ValueError(f"Metric '{metric}' not found in panel columns")

    t0 = pd.Timestamp(event_date).tz_localize(None)

    df = panel.reset_index()
    df["treated"] = df["grid_id"].isin(treated_grid_ids).astype(int)
    df["post"] = (df["time_bucket"] >= t0).astype(int)
    df["did"] = df["treated"] * df["post"]

    df_valid = df[df[metric].notna()].copy()

    treated_pre = df_valid.query("treated == 1 and post == 0")[metric]
    treated_post = df_valid.query("treated == 1 and post == 1")[metric]
    control_pre = df_valid.query("treated == 0 and post == 0")[metric]
    control_post = df_valid.query("treated == 0 and post == 1")[metric]

    y_treat_pre = treated_pre.mean()
    y_treat_post = treated_post.mean()
    y_ctrl_pre = control_pre.mean()
    y_ctrl_post = control_post.mean()

    delta_treat = y_treat_post - y_treat_pre
    delta_ctrl = y_ctrl_post - y_ctrl_pre
    delta = delta_treat - delta_ctrl

    try:
        import statsmodels.api as sm
        X = df_valid[["treated", "post", "did"]].copy()
        X = sm.add_constant(X)
        y = df_valid[metric]
        model = sm.OLS(y, X).fit(cov_type="HC1")
        did_coef = model.params.get("did", np.nan)
        did_pval = model.pvalues.get("did", np.nan)
        did_ci = model.conf_int().loc["did"].tolist() if "did" in model.params.index else [np.nan, np.nan]
        model_r2 = float(model.rsquared)
    except Exception as e:
        did_coef = np.nan
        did_pval = np.nan
        did_ci = [np.nan, np.nan]
        model_r2 = np.nan

    parallel_trends = _check_parallel_trends(df_valid, metric, t0)

    return {
        "metric": metric,
        "event_date": event_date,
        "n_treated_grids": len(treated_grid_ids),
        "n_control_grids": len(control_grid_ids),
        "n_treated_pre": int(len(treated_pre)),
        "n_treated_post": int(len(treated_post)),
        "n_control_pre": int(len(control_pre)),
        "n_control_post": int(len(control_post)),
        "y_treat_pre": float(y_treat_pre),
        "y_treat_post": float(y_treat_post),
        "y_ctrl_pre": float(y_ctrl_pre),
        "y_ctrl_post": float(y_ctrl_post),
        "delta_treat": float(delta_treat),
        "delta_ctrl": float(delta_ctrl),
        "delta": float(delta),
        "did_coef": float(did_coef) if not np.isnan(did_coef) else None,
        "did_p_value": float(did_pval) if not np.isnan(did_pval) else None,
        "did_ci": [float(did_ci[0]) if not np.isnan(did_ci[0]) else None,
                   float(did_ci[1]) if not np.isnan(did_ci[1]) else None],
        "model_r_squared": None if np.isnan(model_r2) else model_r2,
        "parallel_trends": parallel_trends,
        "status": "ok",
    }


def _check_parallel_trends(
    df: pd.DataFrame, metric: str, t0: pd.Timestamp, n_windows: int = 3
) -> dict[str, Any]:
    pre_df = df[df["post"] == 0].copy()
    if len(pre_df) < n_windows * 2:
        return {"testable": False, "reason": "insufficient_pre_data"}

    treated_means = pre_df[pre_df["treated"] == 1].groupby("time_bucket")[metric].mean()
    control_means = pre_df[pre_df["treated"] == 0].groupby("time_bucket")[metric].mean()

    if len(treated_means) < n_windows or len(control_means) < n_windows:
        return {"testable": False, "reason": "insufficient_pre_periods"}

    try:
        treated_trend = np.polyfit(np.arange(len(treated_means)), treated_means.values, 1)
        control_trend = np.polyfit(np.arange(len(control_means)), control_means.values, 1)
        slope_diff = abs(treated_trend[0] - control_trend[0])
        return {
            "testable": True,
            "treated_slope": float(treated_trend[0]),
            "control_slope": float(control_trend[0]),
            "slope_difference": float(slope_diff),
            "parallel_trends_assumption": slope_diff < 0.1,
        }
    except Exception as e:
        return {"testable": False, "reason": str(e)}
