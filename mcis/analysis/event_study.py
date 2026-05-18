from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def run_event_study(
    panel: pd.DataFrame,
    metric: str,
    event_date: str = "2022-02-24",
    estimation_window: tuple[int, int] = (-90, -31),
    event_window: tuple[int, int] = (-30, 30),
    config: dict | None = None,
) -> dict[str, Any]:
    if metric not in panel.columns:
        raise ValueError(f"Metric '{metric}' not found in panel columns")

    if "days_to_t0" not in panel.columns:
        raise ValueError("Panel must contain 'days_to_t0' column")

    est_start, est_end = estimation_window
    ev_start, ev_end = event_window

    est_mask = (panel["days_to_t0"] >= est_start) & (panel["days_to_t0"] <= est_end)
    ev_mask = (panel["days_to_t0"] >= ev_start) & (panel["days_to_t0"] <= ev_end)

    est_series = panel.loc[est_mask, metric].dropna()
    ev_series = panel.loc[ev_mask, metric].dropna()

    if len(est_series) < 2:
        return {
            "metric": metric,
            "status": "insufficient_estimation_data",
            "n_estimation": len(est_series),
            "n_event": len(ev_series),
        }

    baseline_mean = est_series.mean()
    baseline_std = est_series.std()

    ev_dates = panel.loc[ev_mask & panel[metric].notna(), "days_to_t0"].sort_values()

    abnormal_values = {}
    cumulative = 0.0
    cumulative_series: list[float] = []
    significant_dates: list[str] = []
    t_stats_list: list[float] = []
    p_values_list: list[float] = []
    date_labels: list[str] = []

    for d in sorted(ev_dates.unique()):
        day_data = panel.loc[panel["days_to_t0"] == d, metric].dropna()
        if len(day_data) == 0:
            continue

        actual = day_data.mean()
        abnormal = actual - baseline_mean

        if baseline_std > 0 and len(est_series) > 1:
            se = baseline_std * np.sqrt(1 + 1 / len(est_series))
            t_stat = abnormal / se if se > 0 else 0.0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(est_series) - 1))
        else:
            t_stat = 0.0
            p_val = 1.0

        cumulative += abnormal
        cumulative_series.append(cumulative)
        abnormal_values[int(d)] = abnormal
        t_stats_list.append(float(t_stat))
        p_values_list.append(float(p_val))
        date_labels.append(str(int(d)))

        if p_val < 0.05:
            significant_dates.append(str(int(d)))

    return {
        "metric": metric,
        "event_date": event_date,
        "estimation_window": list(estimation_window),
        "event_window": list(event_window),
        "n_estimation": int(len(est_series)),
        "n_event": int(len(ev_series)),
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "abnormal_values": abnormal_values,
        "cumulative_abnormal": {
            str(k): float(v) for k, v in zip(sorted(ev_dates.unique()), cumulative_series)
        },
        "t_stats": dict(zip(date_labels, [float(t) for t in t_stats_list])),
        "p_values": dict(zip(date_labels, [float(p) for p in p_values_list])),
        "significant_dates": significant_dates,
        "status": "ok",
    }
