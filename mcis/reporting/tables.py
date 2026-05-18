from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcis.utils.io import ensure_dir


def dataset_statistics_table(
    loader_report: dict[str, Any],
    cleaner_report: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build Table 1: Dataset statistics.

    Parameters
    ----------
    loader_report:
        Must include 'total_rows', 'n_unique_vessels', 'date_min', 'date_max',
        and optionally 'vessel_types', 'flag_distribution', 'n_unique_mmsi'.
    cleaner_report:
        Optional cleaning report with 'rows_before', 'rows_after'.

    Returns
    -------
    DataFrame with rows as statistics, columns as values.
    """
    rows = []

    rows.append(("Total Rows", str(loader_report.get("total_rows", "N/A"))))
    rows.append(("Date Range (min)", str(loader_report.get("date_min", "N/A"))))
    rows.append(("Date Range (max)", str(loader_report.get("date_max", "N/A"))))
    rows.append(("Unique Vessels (MMSI)", str(loader_report.get("n_unique_mmsi", "N/A"))))
    rows.append(("Unique Vessels (vesselUid)", str(loader_report.get("n_unique_vessels", "N/A"))))
    rows.append(("Vessel Types (distinct)", str(loader_report.get("n_vessel_types", "N/A"))))
    rows.append(("Flags (distinct)", str(loader_report.get("n_flags", "N/A"))))

    pos_src_counts = loader_report.get("pos_src_distribution")
    if pos_src_counts is not None:
        for src, count in pos_src_counts.items():
            rows.append((f"  posSrc = {src}", str(count)))
    else:
        rows.append(("posSrc Distribution", "N/A"))

    if cleaner_report is not None:
        rows.append(("Cleaning: Rows Before", str(cleaner_report.get("rows_before", "N/A"))))
        rows.append(("Cleaning: Rows After", str(cleaner_report.get("rows_after", "N/A"))))
        rows.append(("Cleaning: Dropped", str(cleaner_report.get("rows_dropped", "N/A"))))

    flag_dist = loader_report.get("flag_distribution")
    if flag_dist is not None:
        flag_str = ", ".join(f"{k}: {v}" for k, v in sorted(flag_dist.items(), key=lambda x: -x[1])[:10])
        rows.append(("Top Flags", flag_str))

    return pd.DataFrame(rows, columns=["Statistic", "Value"])


def feature_descriptive_table(
    df: pd.DataFrame,
    config: dict[str, Any],
    feature_cols: list[str] | None = None,
    group_col: str = "post_conflict",
) -> pd.DataFrame:
    """Build Table 2: Feature descriptive statistics before/after conflict onset.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame or aggregated panel.
        Must contain the group_col column if grouping desired.
    config:
        Project config. Used to get t0 if group_col not in df.
    feature_cols:
        Columns to summarize. If None, uses config's model.features_to_use.
    group_col:
        Column to group by for pre/post comparison. If not in df, splits by t0.

    Returns
    -------
    DataFrame with mean/std/pre/post difference for each feature.
    """
    if feature_cols is None:
        feature_cols = config.get("model", {}).get("features_to_use", [])
    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in ("days_to_t0", "post_conflict")][:10]

    if group_col not in df.columns:
        t0 = pd.Timestamp(config.get("conflict", {}).get("t0", "2022-02-24"))
        if "date" in df.columns:
            df = df.copy()
            df[group_col] = (pd.to_datetime(df["date"]) >= t0).astype(int)
        elif isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            dates = df.index.get_level_values("date")
            df = df.copy()
            df[group_col] = (dates >= t0).astype(int)
        else:
            raise ValueError(f"Cannot determine pre/post split: {group_col} not in df")

    groups = df.groupby(group_col)
    rows = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        pre = groups.get_group(0)[col].dropna()
        post = groups.get_group(1)[col].dropna()
        pre_mean = pre.mean() if len(pre) > 0 else np.nan
        pre_std = pre.std() if len(pre) > 0 else np.nan
        post_mean = post.mean() if len(post) > 0 else np.nan
        post_std = post.std() if len(post) > 0 else np.nan
        diff = post_mean - pre_mean
        rows.append({
            "Feature": col,
            "Pre_Mean": pre_mean,
            "Pre_Std": pre_std,
            "Post_Mean": post_mean,
            "Post_Std": post_std,
            "Diff": diff,
            "N_Pre": len(pre),
            "N_Post": len(post),
        })

    result = pd.DataFrame(rows)
    return result


def event_study_results_table(
    event_study_result: dict[str, Any],
    metric: str,
    config: dict[str, Any],
    windows: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Build Table 3: Event study abnormal values for specified windows.

    Parameters
    ----------
    event_study_result:
        Return value from mcis.analysis.event_study.run_event_study.
    metric:
        Metric name (for column labeling).
    config:
        Project config. Uses conflict.event_study_windows if windows not given.
    windows:
        List of (start, end) day offsets relative to t0.

    Returns
    -------
    DataFrame with one row per day, including abnormal value, t-stat, p-value.
    """
    if windows is None:
        windows_raw = config.get("conflict", {}).get("event_study_windows", [])
        windows = [(w, w) for w in windows_raw]

    abnormal = event_study_result.get("abnormal_values", pd.Series(dtype=float))
    t_stats = event_study_result.get("t_stats", pd.Series(dtype=float))
    p_values = event_study_result.get("p_values", pd.Series(dtype=float))
    significant = event_study_result.get("significant_dates", [])

    results = []
    for start, end in windows:
        for day in range(start, end + 1):
            if day in abnormal.index:
                ab = abnormal.get(day, np.nan)
                t = t_stats.get(day, np.nan)
                p = p_values.get(day, np.nan)
                sig = str(day) in significant
                results.append({
                    "Event Day": day,
                    "Abnormal Value": round(ab, 6) if not np.isnan(ab) else np.nan,
                    "t-stat": round(t, 4) if not np.isnan(t) else np.nan,
                    "p-value": round(p, 4) if not np.isnan(p) else np.nan,
                    "Significant (p<0.05)": "Yes" if sig else "No",
                })

    return pd.DataFrame(results)


def its_results_table(
    its_result: dict[str, Any],
    metric: str,
) -> pd.DataFrame:
    """Build Table 4: ITS regression coefficients.

    Parameters
    ----------
    its_result:
        Return value from mcis.analysis.its.run_its.
    metric:
        Metric name.

    Returns
    -------
    DataFrame with coefficients, SE, t-stat, p-value, CI bounds.
    """
    params = its_result.get("params", {})
    se = its_result.get("std_errors", {})
    pvals = its_result.get("pvalues", {})
    rsquared = its_result.get("rsquared")
    rsquared_adj = its_result.get("rsquared_adj")

    rows = []
    for coef_name in params:
        coef_val = params[coef_name]
        se_val = se.get(coef_name, np.nan)
        pval = pvals.get(coef_name, np.nan)

        ci_lower = coef_val - 1.96 * se_val if not np.isnan(se_val) else np.nan
        ci_upper = coef_val + 1.96 * se_val if not np.isnan(se_val) else np.nan

        rows.append({
            "Coefficient": coef_name,
            "Estimate": round(coef_val, 6),
            "Std. Error": round(se_val, 6) if not np.isnan(se_val) else np.nan,
            "t-stat": round(coef_val / se_val, 4) if not np.isnan(se_val) and se_val != 0 else np.nan,
            "p-value": round(pval, 4) if not np.isnan(pval) else np.nan,
            "CI Lower (95%)": round(ci_lower, 6) if not np.isnan(ci_lower) else np.nan,
            "CI Upper (95%)": round(ci_upper, 6) if not np.isnan(ci_upper) else np.nan,
        })

    result = pd.DataFrame(rows)

    if rsquared is not None:
        result.loc["R²"] = {
            "Coefficient": "R²",
            "Estimate": round(rsquared, 4),
            "Std. Error": "",
            "t-stat": "",
            "p-value": "",
            "CI Lower (95%)": "",
            "CI Upper (95%)": "",
        }
    if rsquared_adj is not None:
        result.loc["Adj. R²"] = {
            "Coefficient": "Adj. R²",
            "Estimate": round(rsquared_adj, 4),
            "Std. Error": "",
            "t-stat": "",
            "p-value": "",
            "CI Lower (95%)": "",
            "CI Upper (95%)": "",
        }

    return result


def granger_results_table(
    granger_result: dict[str, Any],
    predictor: str,
    target: str = "conflict_onset",
) -> pd.DataFrame:
    """Build Table 5: Granger causality results by lag.

    Parameters
    ----------
    granger_result:
        Return value from mcis.analysis.granger.run_granger.
    predictor:
        Predictor column name.
    target:
        Target column name.

    Returns
    -------
    DataFrame with one row per lag, F-stat, p-value, reject H0.
    """
    # Accept either "results" (old) or "lags" (actual) key
    results = granger_result.get("results") or granger_result.get("lags", {})
    best_lag = granger_result.get("best_lag")
    n_obs = granger_result.get("n_observations") or granger_result.get("n_obs")

    rows = []
    for lag_str, lag_result in sorted(results.items(), key=lambda x: int(x[0])):
        lag_int = int(lag_str)
        f_stat = lag_result.get("ssr_f_stat") or lag_result.get("ssr_ftest", (np.nan, np.nan))[0]
        p_val = lag_result.get("ssr_p_value") or lag_result.get("ssr_ftest", (np.nan, np.nan))[1]
        reject = p_val < 0.05 if not np.isnan(p_val) else False

        rows.append({
            "Lag": lag_int,
            "F-statistic": round(f_stat, 4) if not np.isnan(f_stat) else np.nan,
            "p-value": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "Reject H0 (p<0.05)": "Yes" if reject else "No",
            "Best Lag": "★" if best_lag is not None and lag_int == best_lag else "",
        })

    df = pd.DataFrame(rows)
    if n_obs is not None:
        df.attrs["n_obs"] = n_obs

    return df


def save_table(
    table: pd.DataFrame,
    path: str | Path,
    fmt: str = "csv",
    caption: str | None = None,
) -> Path:
    """Save a table to disk in the requested format.

    Parameters
    ----------
    table:
        DataFrame to save.
    path:
        Output file path (extension can override format).
    fmt:
        'csv', 'latex', or 'markdown'. Ignored if path extension is recognized.
    caption:
        Caption for latex/markdown output.

    Returns
    -------
    Path to saved file.
    """
    path = Path(path)
    ensure_dir(path.parent)

    ext = path.suffix.lower()

    if ext == ".md":
        fmt = "markdown"
    elif ext == ".tex":
        fmt = "latex"
    elif ext == ".csv":
        fmt = "csv"

    if fmt == "csv":
        table.to_csv(path, index=False)
    elif fmt == "latex":
        latex_str = table.to_latex(index=False, caption=caption, label=f"tab:{path.stem}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex_str)
    elif ext == ".md" or fmt == "markdown":
        md = f"**{caption}**\n\n" if caption else ""
        try:
            md += table.to_markdown(index=False)
        except ImportError:
            md += "```\n" + table.to_csv(index=False, sep="|").replace("|", " | ") + "\n```"
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        table.to_csv(path, index=False)

    return path.absolute()
