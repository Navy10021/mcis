from __future__ import annotations

from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.express as px


def plot_traffic_density(
    panel: pd.DataFrame,
    save_path: str | Path,
    metric: str = "vessel_count",
    t0: str = "2022-02-24",
    m: folium.Map | None = None,
) -> folium.Map:
    if "grid_id" not in panel.index.names:
        raise ValueError("Panel must have MultiIndex with 'grid_id' level")

    df = panel.reset_index()
    if "centroid_lon" not in df.columns or "centroid_lat" not in df.columns:
        raise ValueError("Panel must contain centroid_lon, centroid_lat columns")

    if m is None:
        center_lat = df["centroid_lat"].mean()
        center_lon = df["centroid_lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    df = df.dropna(subset=[metric, "centroid_lon", "centroid_lat"])
    max_val = df[metric].max()
    if max_val == 0:
        max_val = 1

    for _, row in df.iterrows():
        radius = max(2, (row[metric] / max_val) * 15)
        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=radius,
            color="red",
            fill=True,
            fill_opacity=0.5,
            popup=f"{row['grid_id']}: {row[metric]:.1f}",
        ).add_to(m)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(save_path))
    return m


def plot_vessel_tracks(
    df: pd.DataFrame,
    save_path: str | Path,
    color_col: str = "flag",
    sample: int = 10000,
) -> px.scatter_mapbox:
    plot_df = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=42)

    fig = px.scatter_map(
        plot_df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        hover_name="mmsi" if "mmsi" in plot_df.columns else None,
        zoom=6,
        height=600,
        map_style="open-street-map",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    return fig


def plot_grid_anomaly(
    grid_panel: pd.DataFrame,
    metric: str,
    save_path: str | Path,
    t0: str = "2022-02-24",
    window: tuple[int, int] = (-7, 7),
) -> folium.Map:
    if "grid_id" not in grid_panel.index.names:
        raise ValueError("Panel must have MultiIndex with 'grid_id' level")

    df = grid_panel.reset_index()
    if "centroid_lon" not in df.columns or "centroid_lat" not in df.columns:
        raise ValueError("Panel must contain centroid_lon, centroid_lat columns")
    if "days_to_t0" not in df.columns:
        raise ValueError("Panel must contain days_to_t0 column")

    w_start, w_end = window
    window_df = df[(df["days_to_t0"] >= w_start) & (df["days_to_t0"] <= w_end)]
    pre_df = df[df["days_to_t0"] < 0]

    grid_means = pre_df.groupby("grid_id")[metric].mean().rename("baseline")
    window_means = window_df.groupby("grid_id")[metric].mean().rename("window_mean")
    comparison = pd.concat([grid_means, window_means], axis=1).dropna()
    comparison["anomaly"] = comparison["window_mean"] - comparison["baseline"]

    centroids = df[["grid_id", "centroid_lon", "centroid_lat"]].drop_duplicates()
    comparison = comparison.merge(centroids, on="grid_id")

    center_lat = comparison["centroid_lat"].mean()
    center_lon = comparison["centroid_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    max_abs = max(comparison["anomaly"].abs().max(), 0.01)

    for _, row in comparison.iterrows():
        ratio = row["anomaly"] / max_abs
        color = "#e74c3c" if ratio > 0 else "#3498db" if ratio < 0 else "#95a5a6"
        opacity = min(abs(ratio) * 0.8 + 0.2, 1.0)

        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=10,
            color=color,
            fill=True,
            fill_opacity=opacity,
            popup=(
                f"Grid: {row['grid_id']}<br>"
                f"Baseline: {row['baseline']:.2f}<br>"
                f"Window: {row['window_mean']:.2f}<br>"
                f"Anomaly: {row['anomaly']:.2f}"
            ),
        ).add_to(m)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(save_path))
    return m
