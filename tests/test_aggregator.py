from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcis.aggregator import (
    AISAggregator,
    AGG_SPEC,
    assign_grid_cell,
    build_grid,
    build_panels,
    compute_aggregation,
)
from mcis.utils.io import load_yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture(scope="session")
def config():
    return load_yaml(CONFIG_PATH)


@pytest.fixture
def aggregator(config):
    return AISAggregator(config)


@pytest.fixture
def sample_features_df():
    base = pd.Timestamp("2022-02-20", tz="UTC")
    n = 30
    np.random.seed(42)
    return pd.DataFrame({
        "mmsi": np.random.choice([111, 222, 333], n),
        "posDt": [base + pd.Timedelta(hours=i) for i in range(n)],
        "longitude": np.random.uniform(28, 41, n),
        "latitude": np.random.uniform(41, 46, n),
        "sog": np.random.uniform(0, 20, n),
        "cog": np.random.uniform(0, 360, n),
        "rot": np.random.uniform(-20, 20, n),
        "rot_abs": np.abs(np.random.uniform(-20, 20, n)),
        "rot_spike": np.random.choice([True, False], n, p=[0.1, 0.9]),
        "heading": np.random.choice([90.0, 180.0, np.nan], n),
        "navStatus": np.random.choice([0, 1, 5], n),
        "vesselType": np.random.choice(["CARGO", "TANKER", "FISHING"], n),
        "vessel_category": np.random.choice(["cargo", "tanker", "fishing"], n),
        "flag": np.random.choice(["RU", "UA", "TR"], n),
        "flag_group": np.random.choice(["russia", "ukraine", "nato"], n),
        "length": np.random.randint(50, 350, n),
        "width": np.random.randint(10, 50, n),
        "destination": np.random.choice(["PORT_A", "PORT_B", None], n),
        "draught": np.random.uniform(3, 15, n),
        "draught_fraction": np.random.uniform(0.2, 1.0, n),
        "posSrc": np.random.choice(["TER", "SAT"], n, p=[0.7, 0.3]),
        "flag_sat_src": np.random.choice([True, False], n, p=[0.3, 0.7]),
        "flag_no_destination": np.random.choice([True, False], n, p=[0.4, 0.6]),
        "ais_silence": np.random.choice([True, False], n, p=[0.05, 0.95]),
        "turn_event": np.random.choice([True, False], n, p=[0.1, 0.9]),
        "time_gap_hours": np.random.uniform(0.5, 6, n),
        "step_dist_km": np.random.uniform(0, 20, n),
        "implied_speed_kt": np.random.uniform(0, 25, n),
        "speed_discrepancy": np.random.uniform(0, 5, n),
        "cog_change": np.random.uniform(0, 30, n),
        "dest_changed": np.random.choice([True, False], n),
        "speed_state": pd.Categorical(np.random.choice(["stopped", "drifting", "slow", "normal", "fast"], n)),
        "sog_rolling_mean_7d": np.random.uniform(0, 20, n),
        "sog_rolling_std_7d": np.random.uniform(0, 5, n),
        "rot_spike_rolling_count_7d": np.random.randint(0, 5, n),
        "ais_silence_rolling_count_7d": np.random.randint(0, 3, n),
        "cog_change_rolling_std_7d": np.random.uniform(0, 20, n),
    })


class TestBuildGrid:
    def test_returns_dataframe(self, config):
        grid = build_grid(config)
        assert isinstance(grid, pd.DataFrame)

    def test_grid_columns(self, config):
        grid = build_grid(config)
        expected = {"grid_id", "grid_lon_idx", "grid_lat_idx",
                     "centroid_lon", "centroid_lat",
                     "lon_min", "lon_max", "lat_min", "lat_max"}
        assert expected.issubset(set(grid.columns))

    def test_expected_number_of_cells(self, config):
        grid = build_grid(config)
        spatial = config.get("spatial", {})
        res = spatial.get("grid_resolution_deg", 0.5)
        lon_min, lon_max = spatial["lon_min"], spatial["lon_max"]
        lat_min, lat_max = spatial["lat_min"], spatial["lat_max"]
        n_lons = len(np.arange(lon_min, lon_max, res))
        n_lats = len(np.arange(lat_min, lat_max, res))
        assert len(grid) == n_lons * n_lats

    def test_centroid_within_cell(self, config):
        grid = build_grid(config)
        for _, row in grid.head(10).iterrows():
            assert row["lon_min"] < row["centroid_lon"] < row["lon_max"]
            assert row["lat_min"] < row["centroid_lat"] < row["lat_max"]

    def test_grid_ids_unique(self, config):
        grid = build_grid(config)
        assert grid["grid_id"].is_unique

    def test_default_config(self):
        grid = build_grid({})
        assert len(grid) > 0
        assert "grid_id" in grid.columns


class TestAssignGridCell:
    def test_grid_lon_lat_idx_computed(self, config):
        df = pd.DataFrame({
            "longitude": [30.5, 35.0],
            "latitude": [43.0, 44.5],
            "posDt": pd.to_datetime(["2022-02-24", "2022-02-25"], utc=True),
        })
        result = assign_grid_cell(df, config)
        assert "grid_lon_idx" in result.columns
        assert "grid_lat_idx" in result.columns
        assert "grid_id" in result.columns

    def test_grid_id_format(self, config):
        df = pd.DataFrame({
            "longitude": [30.0],
            "latitude": [43.0],
            "posDt": pd.to_datetime(["2022-02-24"], utc=True),
        })
        result = assign_grid_cell(df, config)
        assert "_" in result["grid_id"].iloc[0]

    def test_time_bucket_created(self, config):
        df = pd.DataFrame({
            "longitude": [30.0],
            "latitude": [43.0],
            "posDt": pd.to_datetime(["2022-02-24T14:30:00"], utc=True),
        })
        result = assign_grid_cell(df, config)
        assert result["time_bucket"].iloc[0] == pd.Timestamp("2022-02-24", tz="UTC")

    def test_6h_bucket_created(self, config):
        df = pd.DataFrame({
            "longitude": [30.0],
            "latitude": [43.0],
            "posDt": pd.to_datetime(["2022-02-24T14:30:00"], utc=True),
        })
        result = assign_grid_cell(df, config)
        assert result["time_bucket_6h"].iloc[0] == pd.Timestamp("2022-02-24T12:00:00", tz="UTC")

    def test_preserves_input_columns(self, config):
        df = pd.DataFrame({
            "longitude": [30.0],
            "latitude": [43.0],
            "posDt": pd.to_datetime(["2022-02-24"], utc=True),
            "mmsi": [111],
            "sog": [10.0],
        })
        result = assign_grid_cell(df, config)
        for col in df.columns:
            assert col in result.columns


class TestComputeAggregation:
    def test_returns_dataframe(self, config):
        df = pd.DataFrame({
            "longitude": [30.0, 30.1],
            "latitude": [43.0, 43.1],
            "posDt": pd.to_datetime(["2022-02-24", "2022-02-24"], utc=True),
            "mmsi": [111, 222],
            "sog": [10.0, 12.0],
            "cog": [90.0, 180.0],
            "rot": [0.0, 0.0],
            "rot_abs": [5.0, 3.0],
            "rot_spike": [False, False],
            "ais_silence": [False, False],
            "turn_event": [False, False],
            "vessel_category": ["cargo", "tanker"],
            "flag_group": ["russia", "ukraine"],
            "flag_sat_src": [False, True],
            "flag_no_destination": [True, False],
            "draught": [8.0, 6.0],
            "draught_fraction": [0.5, 0.4],
            "speed_discrepancy": [0.5, 1.0],
        })
        with_bucket = assign_grid_cell(df, config)
        panel = compute_aggregation(with_bucket, config)
        assert isinstance(panel, pd.DataFrame)

    def test_expected_metric_columns(self, config):
        df = pd.DataFrame({
            "longitude": [30.0, 30.1],
            "latitude": [43.0, 43.1],
            "posDt": pd.to_datetime(["2022-02-24", "2022-02-24"], utc=True),
            "mmsi": [111, 222],
            "sog": [10.0, 12.0],
            "cog": [90.0, 180.0],
            "rot": [0.0, 0.0],
            "rot_abs": [5.0, 3.0],
            "rot_spike": [False, False],
            "ais_silence": [False, False],
            "turn_event": [False, False],
            "vessel_category": ["cargo", "tanker"],
            "flag_group": ["russia", "ukraine"],
            "flag_sat_src": [False, True],
            "flag_no_destination": [True, False],
            "draught": [8.0, 6.0],
            "draught_fraction": [0.5, 0.4],
            "speed_discrepancy": [0.5, 1.0],
        })
        with_bucket = assign_grid_cell(df, config)
        panel = compute_aggregation(with_bucket, config)
        expected_metrics = config.get("aggregation", {}).get("metrics", [])
        for m in expected_metrics:
            assert m in panel.columns, f"Missing metric column: {m}"

    def test_post_conflict_and_days_to_t0(self, config):
        df = pd.DataFrame({
            "longitude": [30.0, 30.1],
            "latitude": [43.0, 43.1],
            "posDt": pd.to_datetime(["2022-02-20", "2022-03-01"], utc=True),
            "mmsi": [111, 222],
            "sog": [10.0, 12.0],
            "cog": [90.0, 180.0],
            "rot": [0.0, 0.0],
            "rot_abs": [5.0, 3.0],
            "rot_spike": [False, False],
            "ais_silence": [False, False],
            "turn_event": [False, False],
            "vessel_category": ["cargo", "tanker"],
            "flag_group": ["russia", "ukraine"],
            "flag_sat_src": [False, True],
            "flag_no_destination": [True, False],
            "draught": [8.0, 6.0],
            "draught_fraction": [0.5, 0.4],
            "speed_discrepancy": [0.5, 1.0],
        })
        with_bucket = assign_grid_cell(df, config)
        panel = compute_aggregation(with_bucket, config)
        assert "post_conflict" in panel.columns
        assert "days_to_t0" in panel.columns

    def test_missing_column_raises(self, config):
        df = pd.DataFrame({
            "longitude": [30.0],
            "latitude": [43.0],
            "posDt": pd.to_datetime(["2022-02-24"], utc=True),
            "mmsi": [111],
        })
        with_bucket = assign_grid_cell(df, config)
        with pytest.raises(ValueError, match="Missing required source columns"):
            compute_aggregation(with_bucket, config)


class TestBuildPanels:
    def test_returns_dict_with_two_keys(self, config, sample_features_df):
        panels = build_panels(sample_features_df, config)
        assert isinstance(panels, dict)
        assert "grid_daily" in panels
        assert "blacksea_daily" in panels

    def test_grid_panel_has_multiindex(self, config, sample_features_df):
        panels = build_panels(sample_features_df, config)
        grid_panel = panels["grid_daily"]
        assert isinstance(grid_panel.index, pd.MultiIndex)
        assert "grid_id" in grid_panel.index.names
        assert "time_bucket" in grid_panel.index.names

    def test_blacksea_panel_has_date_index(self, config, sample_features_df):
        panels = build_panels(sample_features_df, config)
        bs_panel = panels["blacksea_daily"]
        assert isinstance(bs_panel.index, pd.DatetimeIndex)

    def test_blacksea_has_post_conflict(self, config, sample_features_df):
        panels = build_panels(sample_features_df, config)
        assert "post_conflict" in panels["blacksea_daily"].columns
        assert "days_to_t0" in panels["blacksea_daily"].columns

    def test_grid_panel_is_sorted(self, config, sample_features_df):
        panels = build_panels(sample_features_df, config)
        grid_panel = panels["grid_daily"]
        assert grid_panel.index.is_monotonic_increasing

    def test_blacksea_panel_is_sorted(self, config, sample_features_df):
        panels = build_panels(sample_features_df, config)
        assert panels["blacksea_daily"].index.is_monotonic_increasing


class TestAISAggregator:
    def test_init_with_config(self, config):
        agg = AISAggregator(config)
        assert agg.config is not None

    def test_grid_property_lazy_loads(self, config):
        agg = AISAggregator(config)
        assert agg._grid is None
        g = agg.grid
        assert agg._grid is not None
        assert isinstance(g, pd.DataFrame)

    def test_grid_property_cached(self, config):
        agg = AISAggregator(config)
        g1 = agg.grid
        g2 = agg.grid
        assert g1 is g2

    def test_transform_returns_dict(self, config, sample_features_df):
        agg = AISAggregator(config)
        panels = agg.transform(sample_features_df)
        assert isinstance(panels, dict)
        assert "grid_daily" in panels
        assert "blacksea_daily" in panels

    def test_transform_handles_utc_parsing(self, config):
        agg = AISAggregator(config)
        df = pd.DataFrame({
            "mmsi": [111],
            "posDt": ["2022-02-24T12:00:00Z"],
            "longitude": [30.0],
            "latitude": [43.0],
            "sog": [10.0],
            "cog": [90.0],
            "rot": [0.0],
            "rot_abs": [0.0],
            "rot_spike": [False],
            "heading": [90.0],
            "navStatus": [0],
            "vesselType": ["CARGO"],
            "vessel_category": ["cargo"],
            "flag": ["RU"],
            "flag_group": ["russia"],
            "length": [100],
            "width": [20],
            "destination": ["PORT"],
            "draught": [8.0],
            "draught_fraction": [0.5],
            "posSrc": ["TER"],
            "flag_sat_src": [False],
            "flag_no_destination": [False],
            "ais_silence": [False],
            "turn_event": [False],
            "time_gap_hours": [1.0],
            "step_dist_km": [0.0],
            "implied_speed_kt": [0.0],
            "speed_discrepancy": [0.0],
            "cog_change": [0.0],
            "dest_changed": [False],
            "speed_state": pd.Categorical(["normal"]),
            "sog_rolling_mean_7d": [10.0],
            "sog_rolling_std_7d": [1.0],
            "rot_spike_rolling_count_7d": [0],
            "ais_silence_rolling_count_7d": [0],
            "cog_change_rolling_std_7d": [0.0],
        })
        panels = agg.transform(df)
        assert "grid_daily" in panels


class TestAGGSPEC:
    def test_all_metrics_have_valid_spec(self):
        for metric_name, (source_col, agg_fn) in AGG_SPEC.items():
            assert isinstance(metric_name, str)
            assert isinstance(source_col, str) or callable(source_col)
            assert isinstance(agg_fn, str) or callable(agg_fn)

    def test_source_columns_are_strings(self):
        for _, (source_col, _) in AGG_SPEC.items():
            if not callable(source_col):
                assert isinstance(source_col, str)

    def test_count_metrics_produce_integers(self, config):
        df = pd.DataFrame({
            "longitude": [30.1, 30.2],
            "latitude": [43.1, 43.2],
            "posDt": pd.to_datetime(["2022-02-24", "2022-02-24"], utc=True),
            "mmsi": [111, 222],
            "sog": [10.0, 12.0],
            "cog": [90.0, 180.0],
            "rot": [0.0, 0.0],
            "rot_abs": [5.0, 3.0],
            "rot_spike": [False, False],
            "ais_silence": [False, False],
            "turn_event": [False, False],
            "vessel_category": ["cargo", "tanker"],
            "flag_group": ["russia", "ukraine"],
            "flag_sat_src": [False, True],
            "flag_no_destination": [True, False],
            "draught": [8.0, 6.0],
            "draught_fraction": [0.5, 0.4],
            "speed_discrepancy": [0.5, 1.0],
        })
        with_bucket = assign_grid_cell(df, config)
        panel = compute_aggregation(with_bucket, config)
        assert panel["vessel_count"].iloc[0] == 2

    def test_mean_sog_expected_value(self, config):
        df = pd.DataFrame({
            "longitude": [30.1, 30.2, 30.3],
            "latitude": [43.1, 43.2, 43.3],
            "posDt": pd.to_datetime(["2022-02-24", "2022-02-24", "2022-02-24"], utc=True),
            "mmsi": [111, 222, 333],
            "sog": [10.0, 20.0, 30.0],
            "cog": [90.0, 180.0, 270.0],
            "rot": [0.0, 5.0, -5.0],
            "rot_abs": [5.0, 3.0, 1.0],
            "rot_spike": [False, False, False],
            "ais_silence": [False, False, False],
            "turn_event": [False, False, False],
            "vessel_category": ["cargo", "tanker", "cargo"],
            "flag_group": ["russia", "ukraine", "nato"],
            "flag_sat_src": [False, True, False],
            "flag_no_destination": [True, False, False],
            "draught": [8.0, 6.0, 10.0],
            "draught_fraction": [0.5, 0.4, 0.6],
            "speed_discrepancy": [0.5, 1.0, 0.3],
        })
        with_bucket = assign_grid_cell(df, config)
        panel = compute_aggregation(with_bucket, config)
        assert panel["mean_sog"].iloc[0] == pytest.approx(20.0)
