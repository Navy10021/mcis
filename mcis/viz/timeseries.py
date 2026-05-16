from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_before_after(
    panel: pd.DataFrame,
    metric: str,
    save_path: str | Path,
    t0: str = "2022-02-24",
    rolling_window: int = 7,
) -> plt.Figure:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    t0_ts = pd.Timestamp(t0)

    series = panel[metric].dropna()
    dates = series.index

    ax.plot(dates, series.values, color="#3498db", alpha=0.5, linewidth=0.8, label="Daily")

    rolling_mean = series.rolling(rolling_window, min_periods=1).mean()
    rolling_std = series.rolling(rolling_window, min_periods=1).std()
    ax.plot(dates, rolling_mean.values, color="#2c3e50", linewidth=1.5, label=f"{rolling_window}d MA")
    ax.fill_between(
        dates,
        (rolling_mean - 1.96 * rolling_std).values,
        (rolling_mean + 1.96 * rolling_std).values,
        color="#2c3e50",
        alpha=0.1,
        label="95% CI",
    )

    ax.axvline(x=t0_ts, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"T₀ ({t0})")

    pre_mean = series[dates < t0_ts].mean() if (dates < t0_ts).any() else None
    if pre_mean is not None:
        ax.axhline(y=pre_mean, color="#95a5a6", linestyle=":", linewidth=1, alpha=0.7, label="Pre-event mean")

    ax.set_xlabel("Date")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} — Before/After {t0}")
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_multi_metric_panel(
    panel: pd.DataFrame,
    metrics: list[str],
    save_path: str | Path,
    t0: str = "2022-02-24",
) -> plt.Figure:
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten() if n_metrics > 1 else [axes]

    t0_ts = pd.Timestamp(t0)

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        if metric not in panel.columns:
            ax.text(0.5, 0.5, f"Missing: {metric}", ha="center", va="center", transform=ax.transAxes)
            continue

        series = panel[metric].dropna()
        ax.plot(series.index, series.values, color="#3498db", alpha=0.6, linewidth=0.7)
        ax.axvline(x=t0_ts, color="#e74c3c", linestyle="--", linewidth=1)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_flag_composition(
    panel: pd.DataFrame,
    save_path: str | Path,
    t0: str = "2022-02-24",
    flag_cols: list[str] | None = None,
) -> plt.Figure:
    if flag_cols is None:
        flag_cols = ["russian_flag_fraction", "ukrainian_flag_fraction", "nato_flag_fraction"]

    available = [c for c in flag_cols if c in panel.columns]
    if not available:
        raise ValueError(f"None of the flag columns {flag_cols} found in panel")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    t0_ts = pd.Timestamp(t0)
    labels = {
        "russian_flag_fraction": "Russia",
        "ukrainian_flag_fraction": "Ukraine",
        "nato_flag_fraction": "NATO",
    }
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, col in enumerate(available):
        series = panel[col].dropna() * 100
        ax.plot(series.index, series.values, color=colors[i], linewidth=1.5, label=labels.get(col, col))

    ax.axvline(x=t0_ts, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"T₀ ({t0})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Flag Fraction (%)")
    ax.set_title("Daily Flag Composition")
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
