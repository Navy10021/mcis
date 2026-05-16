from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def _check_stationarity(series: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    series = series.dropna()
    if len(series) < 10:
        return {"stationary": False, "reason": "too_few_observations", "n": len(series)}
    try:
        result = adfuller(series, autolag="AIC")
        return {
            "stationary": bool(result[1] < alpha),
            "adf_statistic": float(result[0]),
            "p_value": float(result[1]),
            "n_lags": int(result[2]),
            "n_observations": int(result[3]),
            "critical_values": {k: float(v) for k, v in result[4].items()},
        }
    except Exception as e:
        return {"stationary": False, "reason": str(e)}


def run_granger(
    panel: pd.DataFrame,
    predictor_col: str,
    target_col: str,
    max_lag: int = 14,
    alpha: float = 0.05,
    config: dict | None = None,
) -> dict[str, Any]:
    if predictor_col not in panel.columns:
        raise ValueError(f"Predictor '{predictor_col}' not found in panel columns")
    if target_col not in panel.columns:
        raise ValueError(f"Target '{target_col}' not found in panel columns")

    panel = panel.sort_index()
    data = panel[[predictor_col, target_col]].dropna()

    if len(data) < max_lag + 5:
        return {
            "predictor": predictor_col,
            "target": target_col,
            "status": "insufficient_data",
            "n_observations": int(len(data)),
            "max_lag": max_lag,
        }

    predictor_stationarity = _check_stationarity(data[predictor_col], alpha)
    target_stationarity = _check_stationarity(data[target_col], alpha)

    test_data = data.copy()
    transforms_applied: list[str] = []

    if not predictor_stationarity.get("stationary", False):
        test_data[predictor_col] = test_data[predictor_col].diff().dropna()
        transforms_applied.append(f"{predictor_col}_differenced")

    if not target_stationarity.get("stationary", False):
        test_data[target_col] = test_data[target_col].diff().dropna()
        transforms_applied.append(f"{target_col}_differenced")

    test_data = test_data.dropna()
    if len(test_data) < max_lag + 5:
        return {
            "predictor": predictor_col,
            "target": target_col,
            "status": "insufficient_data_after_transform",
            "n_observations": int(len(test_data)),
            "transforms_applied": transforms_applied,
        }

    try:
        gc_result = grangercausalitytests(
            test_data[[target_col, predictor_col]].values,
            maxlag=max_lag,
            verbose=False,
        )
    except Exception as e:
        return {
            "predictor": predictor_col,
            "target": target_col,
            "status": "error",
            "error": str(e),
        }

    lags = {}
    for lag, result in gc_result.items():
        ssr_ftest = result[0]["ssr_ftest"]
        params_ftest = result[0]["params_ftest"]
        lags[str(lag)] = {
            "ssr_f_stat": float(ssr_ftest[0]),
            "ssr_p_value": float(ssr_ftest[1]),
            "ssr_df": [int(ssr_ftest[2]), int(ssr_ftest[3])],
            "params_f_stat": float(params_ftest[0]),
            "params_p_value": float(params_ftest[1]),
            "reject_h0": bool(ssr_ftest[1] < alpha),
        }

    significant_lags = [
        int(k) for k, v in lags.items() if v["reject_h0"]
    ]

    best_lag = None
    best_p = 1.0
    for k, v in lags.items():
        p = v["ssr_p_value"]
        if p < best_p:
            best_p = p
            best_lag = int(k)

    return {
        "predictor": predictor_col,
        "target": target_col,
        "max_lag": max_lag,
        "alpha": alpha,
        "n_observations": int(len(test_data)),
        "status": "ok",
        "transforms_applied": transforms_applied,
        "predictor_stationarity": predictor_stationarity,
        "target_stationarity": target_stationarity,
        "lags": lags,
        "significant_lags": significant_lags,
        "best_lag": best_lag,
        "best_p_value": float(best_p),
    }
