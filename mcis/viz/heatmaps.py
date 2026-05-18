from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    panel: pd.DataFrame,
    save_path: str | Path,
    metrics: list[str] | None = None,
    pre_event_only: bool = True,
    t0: str = "2022-02-24",
) -> plt.Figure:
    if metrics is None:
        exclude = {"days_to_t0", "post_conflict"}
        metrics = [c for c in panel.columns if c not in exclude and panel[c].dtype in (float, int)]
    else:
        metrics = [c for c in metrics if c in panel.columns]

    if len(metrics) < 2:
        raise ValueError(f"Need at least 2 metrics for correlation heatmap, got {len(metrics)}")

    df = panel[metrics].copy()
    if pre_event_only and "days_to_t0" in panel.columns:
        t0_ts = pd.Timestamp(t0)
        df = df[panel.index < t0_ts] if not isinstance(panel.index, pd.MultiIndex) else df

    df = df.dropna(how="any")
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.8), max(6, len(metrics) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix (Pre-Event Period)" if pre_event_only else "Feature Correlation Matrix")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_time_heatmap(
    panel: pd.DataFrame,
    save_path: str | Path,
    metrics: list[str] | None = None,
    t0: str = "2022-02-24",
    window: tuple[int, int] = (-60, 60),
) -> plt.Figure:
    if "days_to_t0" not in panel.columns:
        raise ValueError("Panel must contain 'days_to_t0' column")

    if metrics is None:
        exclude = {"days_to_t0", "post_conflict"}
        metrics = [c for c in panel.columns if c not in exclude and panel[c].dtype in (float, int)]
    else:
        metrics = [c for c in metrics if c in panel.columns]

    if len(metrics) < 2:
        raise ValueError(f"Need at least 2 metrics, got {len(metrics)}")

    w_start, w_end = window
    df = panel[(panel["days_to_t0"] >= w_start) & (panel["days_to_t0"] <= w_end)].copy()

    if len(df) == 0:
        raise ValueError(f"No data in window [{w_start}, {w_end}] days around T₀")

    z_scores = df[metrics].apply(lambda x: (x - x.mean()) / x.std()).dropna(how="all")

    fig, ax = plt.subplots(figsize=(14, max(5, len(metrics) * 0.5)))
    sns.heatmap(
        z_scores.T,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={"label": "Z-Score"},
        ax=ax,
    )
    ax.set_xlabel(f"Days Relative to T₀ ({t0})")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Values Over Time (Z-Score Normalized)")

    t0_idx = None
    days_values = df["days_to_t0"].values if "days_to_t0" in df.columns else None
    if days_values is not None:
        for j, d in enumerate(days_values):
            if d == 0:
                t0_idx = j
                break
    if t0_idx is not None:
        ax.axvline(x=t0_idx + 0.5, color="black", linestyle="--", linewidth=1)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
