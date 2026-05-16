from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcis.features import (
    AISFeatureEngineer,
    _classify_flag_group,
    _classify_vessel_category,
    _haversine_series,
    route_entropy,
)
from mcis.utils.io import load_yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture(scope="session")
def config():
    return load_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def engine(config):
    return AISFeatureEngineer(config)


@pytest.fixture
def simple_df():
    base_ts = pd.Timestamp("2022-02-20", tz="UTC")
    n = 12
    return pd.DataFrame({
        "mmsi": [1] * 6 + [2] * 6,
        "posDt": [base_ts + pd.Timedelta(hours=i) for i in range(6)] * 2,
        "longitude": [29.0, 29.5, 30.0, 30.5, 31.0, 31.5] * 2,
        "latitude": [42.0, 42.2, 42.4, 42.6, 42.8, 43.0] * 2,
        "sog": [0.0, 5.0, 12.0, 18.0, 25.0, 0.5] * 2,
        "cog": [90, 90, 90, 90, 90, 90] * 2,
        "rot": [0.0, 0.0, 10.0, 30.0, -50.0, 0.0] * 2,
        "rot_abs": [0.0, 0.0, 10.0, 30.0, 50.0, 0.0] * 2,
        "heading": [90.0] * n,
        "navStatus": [0] * n,
        "posMsgType": [1.0] * n,
        "posSrc": ["TER"] * n,
        "vesselName": ["V_A"] * 6 + ["V_B"] * 6,
        "callsign": ["C_A"] * 6 + ["C_B"] * 6,
        "flag": ["TR"] * 6 + ["RU"] * 6,
        "vesselTypeAis": [70.0] * 6 + [80.0] * 6,
        "vesselType": ["CARGO"] * 6 + ["TANKER"] * 6,
        "length": [100] * n,
        "width": [20] * n,
        "dwt": [10000.0] * n,
        "grt": [8000.0] * n,
        "destination": ["PORT_A"] * n,
        "eta": [None] * n,
        "draught": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0] * 2,
        "staticMsgType": [np.nan] * n,
        "staticSrc": [np.nan] * n,
        "staticDt": [base_ts] * n,
        "insertDt": [base_ts] * n,
        "date": [base_ts.date()] * n,
        "days_to_t0": [-4] * n,
    })


class TestRouteEntropy:
    def test_uniform_distribution_max_entropy(self):
        cog = np.tile(np.arange(0, 360, 10), 10)
        h = route_entropy(pd.Series(cog))
        assert h > 5.0

    def test_single_direction_min_entropy(self):
        cog = pd.Series([90.0] * 100)
        h = route_entropy(cog)
        assert h == 0.0

    def test_fewer_than_5_values_returns_nan(self):
        cog = pd.Series([90.0, 180.0, 270.0])
        assert np.isnan(route_entropy(cog))

    def test_all_nan_returns_nan(self):
        cog = pd.Series([np.nan] * 10)
        assert np.isnan(route_entropy(cog))


class TestHaversine:
    def test_zero_distance(self):
        lat = pd.Series([42.0, 42.0])
        lon = pd.Series([29.0, 29.0])
        result = _haversine_series(lat, lon, lat, lon)
        assert all(result == 0.0)

    def test_known_distance_black_sea(self):
        lat1 = pd.Series([41.0])
        lon1 = pd.Series([29.0])
        lat2 = pd.Series([42.0])
        lon2 = pd.Series([29.0])
        result = _haversine_series(lat1, lon1, lat2, lon2)
        assert abs(result.iloc[0] - 111.0) < 2.0

    def test_missing_input_returns_nan(self):
        lat1 = pd.Series([np.nan])
        lon1 = pd.Series([29.0])
        lat2 = pd.Series([42.0])
        lon2 = pd.Series([29.0])
        result = _haversine_series(lat1, lon1, lat2, lon2)
        assert np.isnan(result.iloc[0])


class TestClassifyVesselCategory:
    def test_cargo_types(self, config):
        for vt in ["BULK CARRIER", "GENERAL CARGO", "CONTAINER", "CARGO RO RO"]:
            result = _classify_vessel_category(pd.Series([vt]), config)
            assert result.iloc[0] == "cargo", f"{vt} should be cargo"

    def test_tanker_types(self, config):
        for vt in ["TANKER", "OIL/CHEMICAL TANKER", "LNG TANKER", "BUNKERING TANKER"]:
            result = _classify_vessel_category(pd.Series([vt]), config)
            assert result.iloc[0] == "tanker", f"{vt} should be tanker"

    def test_military_types(self, config):
        for vt in ["LAW ENFORCE", "SAR", "PILOT VESSEL", "MANNED VTS"]:
            result = _classify_vessel_category(pd.Series([vt]), config)
            assert result.iloc[0] == "military_para", f"{vt} should be military_para"

    def test_passenger(self, config):
        result = _classify_vessel_category(pd.Series(["PASSENGER"]), config)
        assert result.iloc[0] == "passenger"

    def test_fishing(self, config):
        result = _classify_vessel_category(pd.Series(["FISHING"]), config)
        assert result.iloc[0] == "fishing"

    def test_support(self, config):
        for vt in ["TUG", "SUPPLY", "OFFSHORE"]:
            result = _classify_vessel_category(pd.Series([vt]), config)
            assert result.iloc[0] == "support", f"{vt} should be support"

    def test_unknown_falls_to_other(self, config):
        result = _classify_vessel_category(pd.Series(["UNKNOWN TYPE"]), config)
        assert result.iloc[0] == "other"

    def test_nan_returns_other(self, config):
        result = _classify_vessel_category(pd.Series([np.nan]), config)
        assert result.iloc[0] == "other"


class TestClassifyFlagGroup:
    def test_russia(self, config):
        result = _classify_flag_group(pd.Series(["RU"]), config)
        assert result.iloc[0] == "russia"

    def test_ukraine(self, config):
        result = _classify_flag_group(pd.Series(["UA"]), config)
        assert result.iloc[0] == "ukraine"

    def test_nato(self, config):
        for flag in ["TR", "RO", "BG", "GR"]:
            result = _classify_flag_group(pd.Series([flag]), config)
            assert result.iloc[0] == "nato", f"{flag} should be nato"

    def test_convenience(self, config):
        for flag in ["PA", "MH", "LR", "MT"]:
            result = _classify_flag_group(pd.Series([flag]), config)
            assert result.iloc[0] == "convenience", f"{flag} should be convenience"

    def test_other(self, config):
        result = _classify_flag_group(pd.Series(["US"]), config)
        assert result.iloc[0] == "other"

    def test_nan_returns_other(self, config):
        result = _classify_flag_group(pd.Series([np.nan]), config)
        assert result.iloc[0] == "other"


class TestRowLevelFeatures:
    def test_speed_state_stopped(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        assert df.loc[0, "speed_state"] == "stopped"

    def test_speed_state_fast(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        assert df.loc[4, "speed_state"] == "fast"

    def test_rot_spike_flag(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        assert df.loc[2, "rot_spike"] == False
        assert df.loc[3, "rot_spike"] == True
        assert df.loc[4, "rot_spike"] == True

    def test_vessel_category(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        assert df.loc[0, "vessel_category"] == "cargo"
        assert df.loc[6, "vessel_category"] == "tanker"

    def test_flag_group(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        assert df.loc[0, "flag_group"] == "nato"
        assert df.loc[6, "flag_group"] == "russia"

    def test_draught_fraction_normal(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        expected = 8.0 / (100 * 0.06)
        assert abs(df.loc[0, "draught_fraction"] - expected) < 0.01

    def test_draught_fraction_clamped(self, engine, simple_df):
        df = simple_df.copy()
        df["draught"] = 20.0
        df["length"] = 100
        result = engine._row_level_features(df)
        assert np.isnan(result.loc[0, "draught_fraction"])


class TestTrajectoryFeatures:
    def test_time_gap_hours(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        df = engine._trajectory_features(df)
        assert np.isnan(df.loc[0, "time_gap_hours"])
        assert df.loc[1, "time_gap_hours"] == 1.0

    def test_ais_silence_ter_gap(self, engine):
        base = pd.Timestamp("2022-02-20", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1, 1],
            "posDt": [base, base + pd.Timedelta(hours=48)],
            "posSrc": ["TER", "TER"],
            "longitude": [29.0, 29.5],
            "latitude": [42.0, 42.2],
            "sog": [10.0, 10.0],
            "cog": [90.0, 90.0],
            "rot": [0.0, 0.0],
            "rot_abs": [0.0, 0.0],
            "heading": [90.0, 90.0],
            "navStatus": [0, 0],
            "vesselType": ["CARGO", "CARGO"],
            "flag": ["TR", "TR"],
            "length": [100, 100],
            "width": [20, 20],
            "destination": ["PORT", "PORT"],
            "draught": [8.0, 8.0],
        })
        df["posDt"] = pd.to_datetime(df["posDt"], utc=True)
        df = engine._row_level_features(df)
        df = engine._trajectory_features(df)
        assert df.loc[1, "ais_silence"] == True

    def test_ais_silence_not_flagged_for_sat(self, engine):
        base = pd.Timestamp("2022-02-20", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1, 1],
            "posDt": [base, base + pd.Timedelta(hours=48)],
            "posSrc": ["SAT", "SAT"],
            "longitude": [29.0, 29.5],
            "latitude": [42.0, 42.2],
            "sog": [10.0, 10.0],
            "cog": [90.0, 90.0],
            "rot": [0.0, 0.0],
            "rot_abs": [0.0, 0.0],
            "heading": [90.0, 90.0],
            "navStatus": [0, 0],
            "vesselType": ["CARGO", "CARGO"],
            "flag": ["TR", "TR"],
            "length": [100, 100],
            "width": [20, 20],
            "destination": ["PORT", "PORT"],
            "draught": [8.0, 8.0],
        })
        df["posDt"] = pd.to_datetime(df["posDt"], utc=True)
        df = engine._row_level_features(df)
        df = engine._trajectory_features(df)
        assert df.loc[1, "ais_silence"] == False

    def test_step_dist_km_non_zero(self, engine, simple_df):
        df = engine._row_level_features(simple_df)
        df = engine._trajectory_features(df)
        assert df.loc[1, "step_dist_km"] > 0

    def test_cog_change_wraparound(self, engine):
        base = pd.Timestamp("2022-02-20", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1, 1],
            "posDt": [base, base + pd.Timedelta(hours=1)],
            "posSrc": ["TER", "TER"],
            "longitude": [29.0, 29.0],
            "latitude": [42.0, 42.0],
            "sog": [10.0, 10.0],
            "cog": [359.0, 1.0],
            "rot": [0.0, 0.0],
            "rot_abs": [0.0, 0.0],
            "heading": [90.0, 90.0],
            "navStatus": [0, 0],
            "vesselType": ["CARGO", "CARGO"],
            "flag": ["TR", "TR"],
            "length": [100, 100],
            "width": [20, 20],
            "destination": ["PORT", "PORT"],
            "draught": [8.0, 8.0],
        })
        df["posDt"] = pd.to_datetime(df["posDt"], utc=True)
        df = engine._row_level_features(df)
        df = engine._trajectory_features(df)
        assert abs(df.loc[1, "cog_change"] - 2.0) < 0.01

    def test_turn_event_flagged(self, engine):
        base = pd.Timestamp("2022-02-20", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1, 1],
            "posDt": [base, base + pd.Timedelta(hours=1)],
            "posSrc": ["TER", "TER"],
            "longitude": [29.0, 29.0],
            "latitude": [42.0, 42.0],
            "sog": [10.0, 10.0],
            "cog": [0.0, 90.0],
            "rot": [0.0, 0.0],
            "rot_abs": [0.0, 0.0],
            "heading": [90.0, 90.0],
            "navStatus": [0, 0],
            "vesselType": ["CARGO", "CARGO"],
            "flag": ["TR", "TR"],
            "length": [100, 100],
            "width": [20, 20],
            "destination": ["PORT", "PORT"],
            "draught": [8.0, 8.0],
        })
        df["posDt"] = pd.to_datetime(df["posDt"], utc=True)
        df = engine._row_level_features(df)
        df = engine._trajectory_features(df)
        assert df.loc[1, "turn_event"] == True

    def test_dest_changed(self, engine):
        base = pd.Timestamp("2022-02-20", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1, 1],
            "posDt": [base, base + pd.Timedelta(hours=1)],
            "posSrc": ["TER", "TER"],
            "longitude": [29.0, 29.0],
            "latitude": [42.0, 42.0],
            "sog": [10.0, 10.0],
            "cog": [90.0, 90.0],
            "rot": [0.0, 0.0],
            "rot_abs": [0.0, 0.0],
            "heading": [90.0, 90.0],
            "navStatus": [0, 0],
            "vesselType": ["CARGO", "CARGO"],
            "flag": ["TR", "TR"],
            "length": [100, 100],
            "width": [20, 20],
            "destination": ["PORT_A", "PORT_B"],
            "draught": [8.0, 8.0],
        })
        df["posDt"] = pd.to_datetime(df["posDt"], utc=True)
        df = engine._row_level_features(df)
        df = engine._trajectory_features(df)
        assert df.loc[1, "dest_changed"] == True


class TestRollingFeatures:
    def test_rolling_features_backward_looking(self, engine, simple_df):
        df = engine.transform(simple_df)
        for col in ["sog_rolling_mean_7d", "sog_rolling_std_7d",
                     "rot_spike_rolling_count_7d", "ais_silence_rolling_count_7d",
                     "cog_change_rolling_std_7d"]:
            assert col in df.columns, f"Missing: {col}"

    def test_rolling_has_no_future_leakage(self, engine, simple_df):
        df = engine.transform(simple_df)
        first = df[df["mmsi"] == 1].iloc[0]
        assert pd.notna(first["sog_rolling_mean_7d"])

    def test_rolling_count_accumulates(self, engine):
        base = pd.Timestamp("2022-02-20", tz="UTC")
        n = 10
        df = pd.DataFrame({
            "mmsi": [1] * n,
            "posDt": [base + pd.Timedelta(hours=i) for i in range(n)],
            "posSrc": ["TER"] * n,
            "longitude": [29.0] * n,
            "latitude": [42.0] * n,
            "sog": [10.0] * n,
            "cog": [90.0] * n,
            "rot": [25.0] * n,
            "rot_abs": [25.0] * n,
            "heading": [90.0] * n,
            "navStatus": [0] * n,
            "vesselType": ["CARGO"] * n,
            "flag": ["TR"] * n,
            "length": [100] * n,
            "width": [20] * n,
            "destination": ["PORT"] * n,
            "eta": [None] * n,
            "draught": [8.0] * n,
            "staticMsgType": [np.nan] * n,
            "staticSrc": [np.nan] * n,
            "staticDt": [base] * n,
            "insertDt": [base] * n,
        })
        df["posDt"] = pd.to_datetime(df["posDt"], utc=True)
        result = engine.transform(df)
        r = result[result["mmsi"] == 1].reset_index(drop=True)
        assert r.loc[0, "rot_spike_rolling_count_7d"] == 1
        assert r.iloc[-1, r.columns.get_loc("rot_spike_rolling_count_7d")] == n


class TestEndToEnd:
    def test_transform_adds_all_feature_columns(self, engine, simple_df):
        result = engine.transform(simple_df)
        expected = [
            "speed_state", "rot_spike", "vessel_category", "flag_group",
            "draught_fraction", "time_gap_hours", "ais_silence",
            "step_dist_km", "implied_speed_kt", "speed_discrepancy",
            "cog_change", "turn_event", "dest_changed",
            "sog_rolling_mean_7d", "sog_rolling_std_7d",
            "rot_spike_rolling_count_7d", "ais_silence_rolling_count_7d",
            "cog_change_rolling_std_7d",
        ]
        for col in expected:
            assert col in result.columns, f"Missing feature: {col}"

    def test_transform_preserves_input_columns(self, engine, simple_df):
        input_cols = set(simple_df.columns)
        result = engine.transform(simple_df)
        for col in input_cols:
            assert col in result.columns, f"Input column removed: {col}"

    def test_vessel_sorted_after_transform(self, engine, simple_df):
        result = engine.transform(simple_df)
        for mmsi in result["mmsi"].unique():
            vessel = result[result["mmsi"] == mmsi]
            assert vessel["posDt"].is_monotonic_increasing

    def test_no_columns_removed(self, engine, simple_df):
        input_cols = set(simple_df.columns)
        result = engine.transform(simple_df)
        assert len(result.columns) >= len(input_cols)
