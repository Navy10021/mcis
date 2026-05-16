from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mcis.validation import assert_no_leakage

FORBIDDEN_FEATURES = {
    "days_to_t0",
    "post_conflict",
    "conflict_onset",
    "warning_window",
    "event_window",
    "date",
    "time_bucket",
}

EARLY_WARNING_METRICS = [
    "first_alert_lead_days",
    "mean_alert_lead_days",
    "false_alarms_per_30_days",
    "alert_precision_in_warning_window",
    "alert_recall_in_warning_window",
    "alert_stability",
    "auc_pr_warning_window",
    "brier_score",
]


def compute_anomaly_metrics(
    scores: pd.Series,
    dates: pd.Series,
    event_date: str,
    warning_window_days: int = 30,
    threshold: float | None = None,
    threshold_percentile: float = 95.0,
) -> dict[str, Any]:
    """Compute early-warning metrics for anomaly scores.

    Parameters
    ----------
    scores:
        Anomaly scores indexed by date or aligned with dates.
    dates:
        Date values for each score, as datetime-like Series.
    event_date:
        Conflict onset date (T0).
    warning_window_days:
        Days before T0 to consider as the warning window.
    threshold:
        Fixed threshold. If None, computed from threshold_percentile.
    threshold_percentile:
        Percentile of training scores to use as threshold.

    Returns
    -------
    dict with keys: first_alert_lead_days, mean_alert_lead_days,
        false_alarms_per_30_days, alert_precision, alert_recall,
        alert_stability, n_alerts_pre, n_alerts_total.
    """
    dates = pd.to_datetime(dates)
    t0 = pd.Timestamp(event_date)

    if threshold is None:
        threshold = np.percentile(scores.dropna(), threshold_percentile)

    pre_mask = dates < t0
    warning_start = t0 - pd.Timedelta(days=warning_window_days)
    warning_mask = (dates >= warning_start) & (dates < t0)
    post_mask = dates >= t0

    alerts = scores > threshold

    n_pre = int(alerts[pre_mask].sum())
    n_warning = int(alerts[warning_mask].sum())
    n_post = int(alerts[post_mask].sum())
    n_total = int(alerts.sum())

    warning_dates = dates[warning_mask & alerts]

    if len(warning_dates) > 0:
        first_alert_lead = int((t0 - warning_dates.min()).days)
        mean_alert_lead = float((t0 - warning_dates).mean().days)
    else:
        first_alert_lead = None
        mean_alert_lead = None

    pre_period_days = max(int((t0 - dates[pre_mask].min()).days), 1)
    false_alarm_rate = n_pre / pre_period_days * 30.0 if n_pre > 0 else 0.0

    stability = _compute_alert_stability(dates, alerts, warning_mask)

    return {
        "threshold": float(threshold),
        "first_alert_lead_days": first_alert_lead,
        "mean_alert_lead_days": mean_alert_lead,
        "false_alarms_per_30_days": round(false_alarm_rate, 2),
        "n_alerts_pre": n_pre,
        "n_alerts_warning_window": n_warning,
        "n_alerts_post": n_post,
        "n_alerts_total": n_total,
        "alert_stability": round(stability, 4),
        "n_dates_pre": int(pre_mask.sum()),
        "n_dates_warning": int(warning_mask.sum()),
        "n_dates_post": int(post_mask.sum()),
    }


def _compute_alert_stability(
    dates: pd.Series,
    alerts: pd.Series,
    warning_mask: pd.Series,
) -> float:
    """Stability = consecutive alert days / total alert days.

    Higher values mean sustained warnings rather than isolated spikes.
    """
    warning_alerts = alerts[warning_mask]
    if warning_alerts.sum() < 2:
        return 0.0
    runs = (warning_alerts.diff() != 0).cumsum()
    run_sizes = warning_alerts.groupby(runs).sum()
    consecutive = run_sizes[run_sizes > 0].sum()
    total = warning_alerts.sum()
    return consecutive / total if total > 0 else 0.0


def run_placebo_dates(
    scores: pd.Series,
    dates: pd.Series,
    true_event_date: str,
    candidate_dates: list[str],
    warning_window_days: int = 30,
    exclusion_days: int = 30,
) -> dict[str, Any]:
    """Run placebo test against alternative event dates.

    Parameters
    ----------
    scores:
        Anomaly scores.
    dates:
        Date values for each score.
    true_event_date:
        The real T0.
    candidate_dates:
        Placebo event dates to test.
    warning_window_days:
        Window before each candidate to check max score.
    exclusion_days:
        Days around true event to exclude from placebo candidates
        (not applied if candidate_dates is passed directly).

    Returns
    -------
    dict with placebo_max_scores, true_event_max_score, p_value.
    """
    t0 = pd.Timestamp(true_event_date)
    true_window_start = t0 - pd.Timedelta(days=warning_window_days)

    true_max = float(scores[(dates >= true_window_start) & (dates < t0)].max())

    placebo_max_scores = {}
    for cand_str in candidate_dates:
        cand = pd.Timestamp(cand_str)
        ws = cand - pd.Timedelta(days=warning_window_days)
        mask = (dates >= ws) & (dates < cand) & (dates < t0 - pd.Timedelta(days=exclusion_days))
        if mask.sum() > 0:
            placebo_max_scores[cand_str] = float(scores[mask].max())
        else:
            placebo_max_scores[cand_str] = None

    placebo_vals = [v for v in placebo_max_scores.values() if v is not None]
    p_value = sum(1 for v in placebo_vals if v >= true_max) / max(len(placebo_vals), 1)

    return {
        "true_event_date": true_event_date,
        "true_event_max_score": true_max,
        "placebo_candidate_dates": candidate_dates,
        "placebo_max_scores": placebo_max_scores,
        "placebo_p_value": p_value,
        "n_placebo_valid": len(placebo_vals),
    }


def compute_forecast_error_anomaly(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    method: str = "zscore",
) -> pd.Series:
    """Compute anomaly score from multi-step forecast errors.

    Parameters
    ----------
    y_true:
        Actual values (T x F).
    y_pred:
        Predicted values (T x F).
    method:
        'zscore' — z-score each feature's error then mean.
        'mae'    — mean absolute error across features.

    Returns
    -------
    Series of anomaly scores indexed like y_true.
    """
    errors = (y_true - y_pred).abs()

    if method == "zscore":
        mu = errors.mean()
        sigma = errors.std().replace(0, np.nan)
        scores = errors.subtract(mu).divide(sigma).mean(axis=1)
    elif method == "mae":
        scores = errors.mean(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return scores.clip(0, None)


def compute_classification_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
) -> dict[str, float]:
    """Compute threshold-free classification metrics.

    Parameters
    ----------
    y_true:
        Binary labels (0/1).
    y_score:
        Continuous anomaly/confidence scores.

    Returns
    -------
    dict with auc_roc, auc_pr, brier_score.
    """
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        brier_score_loss,
    )

    valid = y_true.notna() & y_score.notna()
    y_true_v = y_true[valid].astype(int)
    y_score_v = y_score[valid]

    if y_true_v.nunique() < 2:
        return {
            "auc_roc": None,
            "auc_pr": None,
            "brier_score": None,
            "n_pos": int(y_true_v.sum()),
            "n_neg": int((1 - y_true_v).sum()),
        }

    try:
        auc_roc = float(roc_auc_score(y_true_v, y_score_v))
    except Exception:
        auc_roc = None

    try:
        precision, recall, _ = precision_recall_curve(y_true_v, y_score_v)
        auc_pr = float(auc(recall, precision))
    except Exception:
        auc_pr = None

    try:
        brier = float(brier_score_loss(y_true_v, y_score_v))
    except Exception:
        brier = None

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier_score": brier,
        "n_pos": int(y_true_v.sum()),
        "n_neg": int((1 - y_true_v).sum()),
    }
