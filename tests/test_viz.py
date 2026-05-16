from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcis.viz.heatmaps import plot_correlation_heatmap, plot_feature_time_heatmap
from mcis.viz.maps import plot_grid_anomaly, plot_traffic_density, plot_vessel_tracks
from mcis.viz.timeseries import (
    plot_before_after,
    plot_flag_composition,
    plot_multi_metric_panel,
)


@pytest.fixture(scope="session")
def blacksea_panel():
    np.random.seed(42)
    dates = pd.date_range("2021-09-01", "2022-06-01", freq="D")
    n = len(dates)
    df = pd.DataFrame(index=dates)
    df["days_to_t0"] = (dates - pd.Timestamp("2022-02-24")).days
    df["post_conflict"] = (dates >= "2022-02-24").astype(int)
    df["vessel_count"] = np.random.poisson(80, n).astype(float)
    df["mean_sog"] = np.random.uniform(8, 12, n)
    df["std_sog"] = np.random.uniform(2, 5, n)
    df["rot_spike_count"] = np.random.poisson(3, n).astype(float)
    df["ais_silence_count"] = np.random.poisson(5, n).astype(float)
    df["cargo_fraction"] = np.random.uniform(0.3, 0.6, n)
    df["tanker_fraction"] = np.random.uniform(0.1, 0.3, n)
    df["russian_flag_fraction"] = np.random.uniform(0.1, 0.3, n)
    df["ukrainian_flag_fraction"] = np.random.uniform(0.1, 0.3, n)
    df["nato_flag_fraction"] = np.random.uniform(0.2, 0.4, n)
    df["route_entropy"] = np.random.uniform(2, 4, n)
    df["cog_variance"] = np.random.uniform(100, 500, n)
    return df


@pytest.fixture(scope="session")
def grid_panel():
    np.random.seed(42)
    dates = pd.date_range("2022-02-01", "2022-03-31", freq="D")
    grid_ids = [f"{i}_{j}" for i in range(3) for j in range(3)]

    rows = []
    for gid in grid_ids:
        for d in dates:
            rows.append({
                "grid_id": gid,
                "time_bucket": d,
                "vessel_count": np.random.poisson(10),
                "mean_sog": np.random.uniform(8, 12),
            })

    df = pd.DataFrame(rows)
    df["post_conflict"] = (df["time_bucket"] >= "2022-02-24").astype(int)
    df["days_to_t0"] = (df["time_bucket"] - pd.Timestamp("2022-02-24")).dt.days

    centroids = {
        gid: (27.5 + (int(gid.split("_")[0]) + 0.5) * 0.5,
              40.5 + (int(gid.split("_")[1]) + 0.5) * 0.5)
        for gid in grid_ids
    }
    df["centroid_lon"] = df["grid_id"].map(lambda x: centroids[x][0])
    df["centroid_lat"] = df["grid_id"].map(lambda x: centroids[x][1])

    return df.set_index(["grid_id", "time_bucket"])


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestMaps:
    def test_plot_traffic_density_saves_html(self, grid_panel, tmp_path):
        save = tmp_path / "density.html"
        result = plot_traffic_density(grid_panel, save)
        assert save.exists()
        assert save.stat().st_size > 100

    def test_plot_traffic_density_returns_folium_map(self, grid_panel, tmp_path):
        save = tmp_path / "density2.html"
        result = plot_traffic_density(grid_panel, save)
        assert "folium" in type(result).__module__

    def test_plot_traffic_density_missing_coords_raises(self, tmp_path):
        panel = pd.DataFrame({"vessel_count": [1]})
        with pytest.raises(ValueError, match="MultiIndex"):
            plot_traffic_density(panel, tmp_path / "x.html", metric="vessel_count")

    def test_plot_vessel_tracks_saves_html(self, tmp_path):
        df = pd.DataFrame({
            "mmsi": [1, 2, 3],
            "latitude": [43.0, 44.0, 45.0],
            "longitude": [30.0, 31.0, 32.0],
            "flag": ["RU", "UA", "TR"],
        })
        save = tmp_path / "tracks.html"
        result = plot_vessel_tracks(df, save, sample=100)
        assert save.exists()
        assert save.stat().st_size > 100

    def test_plot_grid_anomaly_saves_html(self, grid_panel, tmp_path):
        save = tmp_path / "anomaly.html"
        result = plot_grid_anomaly(grid_panel, "vessel_count", save)
        assert save.exists()
        assert save.stat().st_size > 100

    def test_plot_grid_anomaly_missing_days_raises(self, tmp_path):
        panel = pd.DataFrame({"vessel_count": [1], "centroid_lon": [30.0], "centroid_lat": [43.0]})
        panel.index = pd.MultiIndex.from_tuples([("0_0", "2022-02-24")], names=["grid_id", "time_bucket"])
        with pytest.raises(ValueError, match="days_to_t0"):
            plot_grid_anomaly(panel, "vessel_count", tmp_path / "x.html")


class TestTimeseries:
    def test_plot_before_after_saves_png(self, blacksea_panel, tmp_path):
        save = tmp_path / "ba.png"
        result = plot_before_after(blacksea_panel, "vessel_count", save)
        assert save.exists()
        assert save.stat().st_size > 1000

    def test_plot_before_after_returns_figure(self, blacksea_panel, tmp_path):
        save = tmp_path / "ba2.png"
        result = plot_before_after(blacksea_panel, "vessel_count", save)
        assert hasattr(result, "savefig")

    def test_plot_before_after_with_rolling(self, blacksea_panel, tmp_path):
        save = tmp_path / "ba3.png"
        result = plot_before_after(blacksea_panel, "vessel_count", save, rolling_window=14)
        assert save.exists()

    def test_plot_multi_metric_panel_saves_png(self, blacksea_panel, tmp_path):
        save = tmp_path / "multi.png"
        metrics = ["vessel_count", "mean_sog", "std_sog"]
        result = plot_multi_metric_panel(blacksea_panel, metrics, save)
        assert save.exists()
        assert save.stat().st_size > 1000

    def test_plot_multi_metric_panel_with_missing_metric(self, blacksea_panel, tmp_path):
        save = tmp_path / "multi2.png"
        result = plot_multi_metric_panel(blacksea_panel, ["vessel_count", "nonexistent"], save)
        assert save.exists()

    def test_plot_flag_composition_saves_png(self, blacksea_panel, tmp_path):
        save = tmp_path / "flags.png"
        result = plot_flag_composition(blacksea_panel, save)
        assert save.exists()
        assert save.stat().st_size > 1000

    def test_plot_flag_composition_missing_cols_raises(self, tmp_path):
        panel = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="flag columns"):
            plot_flag_composition(panel, tmp_path / "f.png")

    def test_plot_flag_composition_custom_cols(self, blacksea_panel, tmp_path):
        save = tmp_path / "flags2.png"
        result = plot_flag_composition(blacksea_panel, save, flag_cols=["russian_flag_fraction"])
        assert save.exists()


class TestHeatmaps:
    def test_plot_correlation_heatmap_saves_png(self, blacksea_panel, tmp_path):
        save = tmp_path / "corr.png"
        result = plot_correlation_heatmap(blacksea_panel, save)
        assert save.exists()
        assert save.stat().st_size > 1000

    def test_plot_correlation_heatmap_returns_figure(self, blacksea_panel, tmp_path):
        save = tmp_path / "corr2.png"
        result = plot_correlation_heatmap(blacksea_panel, save)
        assert hasattr(result, "savefig")

    def test_plot_correlation_heatmap_fewer_than_2_raises(self, tmp_path):
        panel = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="at least 2"):
            plot_correlation_heatmap(panel, tmp_path / "x.png", metrics=["a"])

    def test_plot_feature_time_heatmap_saves_png(self, blacksea_panel, tmp_path):
        save = tmp_path / "ftime.png"
        result = plot_feature_time_heatmap(blacksea_panel, save)
        assert save.exists()
        assert save.stat().st_size > 1000

    def test_plot_feature_time_heatmap_missing_days_raises(self, tmp_path):
        panel = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="days_to_t0"):
            plot_feature_time_heatmap(panel, tmp_path / "x.png")

    def test_plot_feature_time_heatmap_fewer_than_2_metrics_raises(self, tmp_path):
        panel = pd.DataFrame({"days_to_t0": [1, 2], "a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="at least 2"):
            plot_feature_time_heatmap(panel, tmp_path / "x.png", metrics=["a"])

    def test_plot_feature_time_heatmap_window_filtering(self, blacksea_panel, tmp_path):
        save = tmp_path / "ftime2.png"
        result = plot_feature_time_heatmap(blacksea_panel, save, window=(-30, 30))
        assert save.exists()
