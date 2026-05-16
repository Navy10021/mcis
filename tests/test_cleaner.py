from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcis.cleaner import AISCleaner
from mcis.utils.io import load_yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture(scope="session")
def config():
    return load_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def raw_df():
    np.random.seed(42)
    n = 200
    base_ts = pd.Timestamp("2022-02-20", tz="UTC")
    return pd.DataFrame({
        "vesselUid": range(1, n + 1),
        "mmsi": np.random.choice([111111111, 222222222, 333333333, 444444444], n),
        "imo": np.where(np.random.random(n) < 0.4, np.nan, np.random.uniform(1000000, 9999999, n)),
        "longitude": np.where(np.random.random(n) < 0.05, 50.0, np.random.uniform(28, 41, n)),
        "latitude": np.where(np.random.random(n) < 0.05, 10.0, np.random.uniform(41, 46, n)),
        "sog": np.where(np.random.random(n) < 0.03, 99.0, np.random.uniform(0, 25, n)),
        "cog": np.where(np.random.random(n) < 0.05, 360.0, np.random.uniform(0, 359, n)),
        "rot": np.random.choice(
            [0.0, -5.0, 10.0, -128.0, 128.0, -200.0, 300.0, np.nan],
            n,
            p=[0.4, 0.2, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05],
        ),
        "heading": np.where(np.random.random(n) < 0.05, 511.0, np.random.uniform(0, 359, n)),
        "navStatus": np.random.choice([0, 1, 5, 7, 15, 95, 98, 99], n),
        "posMsgType": np.random.choice([1.0, 3.0, 18.0, np.nan], n),
        "posSrc": np.random.choice(["TER", "SAT", "ROAM"], n, p=[0.6, 0.3, 0.1]),
        "vesselName": [f"VESSEL_{i}" for i in range(n)],
        "callsign": [None if np.random.random() < 0.1 else f"CALL_{i}" for i in range(n)],
        "flag": np.random.choice(["TR", "RU", "UA", "PA", "RO"], n),
        "vesselTypeAis": np.random.choice([70.0, 80.0, 30.0, np.nan], n),
        "vesselType": np.random.choice(["CARGO", "TANKER", "PASSENGER", "FISHING"], n),
        "length": np.where(np.random.random(n) < 0.05, 0, np.random.randint(50, 350, n)),
        "width": np.where(np.random.random(n) < 0.05, 0, np.random.randint(10, 50, n)),
        "dwt": np.where(np.random.random(n) < 0.4, np.nan, np.random.uniform(1000, 100000, n)),
        "grt": np.where(np.random.random(n) < 0.4, np.nan, np.random.uniform(500, 80000, n)),
        "destination": [
            None if np.random.random() < 0.4 else np.random.choice(["PORT_A", "PORT_B", "PORT_C", ""])
            for _ in range(n)
        ],
        "eta": [None if np.random.random() < 0.45 else "2022-02-25T12:00:00Z" for _ in range(n)],
        "draught": np.where(np.random.random(n) < 0.06, np.nan, np.random.uniform(3, 15, n)),
        "staticMsgType": np.full(n, np.nan),
        "staticSrc": np.full(n, np.nan),
        "posDt": [base_ts + pd.Timedelta(hours=i) for i in range(n)],
        "staticDt": [base_ts + pd.Timedelta(hours=i) for i in range(n)],
        "insertDt": [base_ts + pd.Timedelta(hours=i) for i in range(n)],
    })


@pytest.fixture
def cleaner(config):
    return AISCleaner(config)


class TestCoordinateFilter:
    def test_removes_outside_bounds(self, cleaner):
        df = pd.DataFrame({
            "longitude": [30.0, 50.0, 28.0, 42.0],
            "latitude": [44.0, 45.0, 10.0, 47.0],
            "mmsi": [1, 2, 3, 4],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_coordinate_filter(df)
        assert len(result) == 1

    def test_keeps_valid_bounds(self, cleaner):
        df = pd.DataFrame({
            "longitude": [30.0, 35.0],
            "latitude": [44.0, 42.0],
            "mmsi": [1, 2],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_coordinate_filter(df)
        assert len(result) == 2


class TestSOGFilter:
    def test_removes_above_max(self, cleaner):
        df = pd.DataFrame({
            "sog": [10.0, 55.0, 0.0, -5.0],
            "mmsi": [1, 2, 3, 4],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_sog_filter(df)
        assert len(result) == 2

    def test_keeps_valid_sog(self, cleaner):
        df = pd.DataFrame({
            "sog": [0.0, 12.5, 50.0],
            "mmsi": [1, 2, 3],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_sog_filter(df)
        assert len(result) == 3


class TestROTNormalize:
    def test_rot_128_becomes_nan(self, cleaner):
        df = pd.DataFrame({
            "rot": [128.0, -128.0, 0.0, np.nan],
            "mmsi": [1, 2, 3, 4],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_rot_normalize(df)
        assert result.loc[0, "rot"] != 128.0
        assert np.isnan(result.loc[0, "rot"])
        assert np.isnan(result.loc[1, "rot"])

    def test_rot_clipped_to_127(self, cleaner):
        df = pd.DataFrame({
            "rot": [200.0, -300.0, 50.0],
            "mmsi": [1, 2, 3],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_rot_normalize(df)
        assert result.loc[0, "rot"] == 127.0
        assert result.loc[1, "rot"] == -127.0

    def test_rot_abs_added(self, cleaner):
        df = pd.DataFrame({
            "rot": [-50.0, 30.0],
            "mmsi": [1, 2],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_rot_normalize(df)
        assert "rot_abs" in result.columns
        assert result.loc[0, "rot_abs"] == 50.0
        assert result.loc[1, "rot_abs"] == 30.0

    def test_rot_nan_preserved(self, cleaner):
        df = pd.DataFrame({
            "rot": [np.nan, 10.0],
            "mmsi": [1, 2],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_rot_normalize(df)
        assert np.isnan(result.loc[0, "rot"])
        assert np.isnan(result.loc[0, "rot_abs"])


class TestHeadingNormalize:
    def test_511_becomes_nan(self, cleaner):
        df = pd.DataFrame({
            "heading": [511.0, 90.0, 511.0],
            "mmsi": [1, 2, 3],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_heading_normalize(df)
        assert np.isnan(result.loc[0, "heading"])
        assert result.loc[1, "heading"] == 90.0
        assert np.isnan(result.loc[2, "heading"])

    def test_heading_normal_preserved(self, cleaner):
        df = pd.DataFrame({
            "heading": [0.0, 180.0, 359.0, np.nan],
            "mmsi": [1, 2, 3, 4],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_heading_normalize(df)
        assert result.loc[0, "heading"] == 0.0
        assert result.loc[1, "heading"] == 180.0
        assert result.loc[2, "heading"] == 359.0
        assert np.isnan(result.loc[3, "heading"])


class TestCOGNormalize:
    def test_360_becomes_nan(self, cleaner):
        df = pd.DataFrame({
            "cog": [360.0, 180.0, 0.0],
            "mmsi": [1, 2, 3],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_cog_normalize(df)
        assert np.isnan(result.loc[0, "cog"])
        assert result.loc[1, "cog"] == 180.0
        assert result.loc[2, "cog"] == 0.0


class TestNavStatusNormalize:
    def test_unknown_codes_mapped_to_neg1(self, cleaner):
        df = pd.DataFrame({
            "navStatus": [0, 95, 98, 99, 5, 15],
            "mmsi": [1, 2, 3, 4, 5, 6],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_navstatus_normalize(df)
        assert result.loc[0, "navStatus"] == 0
        assert result.loc[1, "navStatus"] == -1
        assert result.loc[2, "navStatus"] == -1
        assert result.loc[3, "navStatus"] == -1
        assert result.loc[4, "navStatus"] == 5
        assert result.loc[5, "navStatus"] == 15


class TestDimensionNormalize:
    def test_zero_length_width_to_nan(self, cleaner):
        df = pd.DataFrame({
            "length": [0, 100, 200],
            "width": [50, 0, 30],
            "mmsi": [1, 2, 3],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_dimension_normalize(df)
        assert np.isnan(result.loc[0, "length"])
        assert result.loc[1, "length"] == 100
        assert result.loc[2, "length"] == 200
        assert result.loc[0, "width"] == 50
        assert np.isnan(result.loc[1, "width"])
        assert result.loc[2, "width"] == 30


class TestDedup:
    def test_removes_duplicate_mmsi_posdt(self, cleaner):
        ts = pd.Timestamp("2022-02-24", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1, 1, 2, 2],
            "posDt": [ts, ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=1)],
            "sog": [10.0, 12.0, 8.0, 9.0],
        })
        result = cleaner._step_dedup(df)
        assert len(result) == 2

    def test_keeps_first_occurrence(self, cleaner):
        ts = pd.Timestamp("2022-02-24", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1, 1],
            "posDt": [ts, ts],
            "sog": [10.0, 12.0],
        })
        result = cleaner._step_dedup(df)
        assert result["sog"].iloc[0] == 10.0


class TestMinObsFilter:
    def test_drops_sparse_vessels(self, cleaner):
        df = pd.DataFrame({
            "mmsi": [1, 1, 1, 2, 2],
            "posDt": pd.date_range("2022-02-24", periods=5, freq="h", tz="UTC"),
        })
        result = cleaner._step_min_obs_filter(df)
        assert 1 in result["mmsi"].values
        assert 2 not in result["mmsi"].values

    def test_keeps_frequent_vessels(self, cleaner):
        df = pd.DataFrame({
            "mmsi": [1, 1, 1, 1],
            "posDt": pd.date_range("2022-02-24", periods=4, freq="h", tz="UTC"),
        })
        result = cleaner._step_min_obs_filter(df)
        assert len(result) == 4


class TestQualityFlags:
    def test_quality_flags_created(self, cleaner):
        df = pd.DataFrame({
            "posSrc": ["SAT", "TER", "ROAM"],
            "heading": [np.nan, 90.0, np.nan],
            "imo": [np.nan, 1000000.0, np.nan],
            "destination": ["PORT_A", np.nan, np.nan],
            "length": [np.nan, 100.0, 0.0],
            "width": [50.0, np.nan, 30.0],
            "mmsi": [1, 2, 3],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner._step_quality_flags(df)
        assert list(result["flag_sat_src"]) == [True, False, False]
        assert list(result["flag_ter_src"]) == [False, True, False]
        assert list(result["flag_roam_src"]) == [False, False, True]
        assert list(result["flag_no_heading"]) == [True, False, True]
        assert list(result["flag_no_imo"]) == [True, False, True]
        assert list(result["flag_no_destination"]) == [False, True, True]


class TestCleaningReport:
    def test_report_after_clean(self, cleaner, raw_df):
        result = cleaner.clean(raw_df)
        report = cleaner.cleaning_report()
        assert "coordinate_filter" in report
        assert "sog_filter" in report
        assert "rot_normalize" in report
        assert "heading_normalize" in report
        assert "cog_normalize" in report
        assert "navstatus_normalize" in report
        assert "dimension_normalize" in report
        assert "dedup" in report
        assert "min_obs_filter" in report
        assert "quality_flags" in report
        assert "total_initial" in report
        assert "total_final" in report

    def test_report_before_after_counts(self, cleaner):
        ts = pd.Timestamp("2022-02-24", tz="UTC")
        df = pd.DataFrame({
            "mmsi": [1] * 10,
            "longitude": [30.0] * 10,
            "latitude": [44.0] * 10,
            "sog": [10.0] * 10,
            "rot": [0.0] * 10,
            "heading": [90.0] * 10,
            "cog": [180.0] * 10,
            "navStatus": [0] * 10,
            "length": [100] * 10,
            "width": [20] * 10,
            "posSrc": ["TER"] * 10,
            "imo": [1000000.0] * 10,
            "destination": ["PORT"] * 10,
            "posDt": pd.date_range("2022-02-24", periods=10, freq="h", tz="UTC"),
        })
        result = cleaner.clean(df)
        report = cleaner.cleaning_report()
        assert report["total_final"]["before"] == 10
        assert report["total_final"]["after"] == 10


class TestEndToEnd:
    def test_clean_pipeline_removes_records(self, cleaner, raw_df):
        result = cleaner.clean(raw_df)
        assert len(result) < len(raw_df)
        assert len(result) > 0

    def test_clean_pipeline_no_nulls_in_quality_flags(self, cleaner, raw_df):
        result = cleaner.clean(raw_df)
        for col in ["flag_sat_src", "flag_ter_src", "flag_roam_src",
                     "flag_no_heading", "flag_no_imo", "flag_no_destination",
                     "flag_invalid_dimension", "flag_sparse_vessel"]:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].notna().all(), f"Nulls in {col}"

    def test_clean_does_not_remove_columns(self, cleaner, raw_df):
        result = cleaner.clean(raw_df)
        for col in raw_df.columns:
            assert col in result.columns, f"Column removed: {col}"

    def test_clean_no_valid_records_returns_empty(self, cleaner):
        df = pd.DataFrame({
            "mmsi": [1],
            "longitude": [100.0],
            "latitude": [100.0],
            "sog": [99.0],
            "rot": [0.0],
            "heading": [511.0],
            "cog": [360.0],
            "navStatus": [0],
            "length": [0],
            "width": [0],
            "posSrc": ["TER"],
            "imo": [np.nan],
            "destination": [np.nan],
            "posDt": pd.Timestamp("2022-02-24", tz="UTC"),
        })
        result = cleaner.clean(df)
        assert len(result) == 0

    def test_clean_with_real_data_schema(self, cleaner, raw_df):
        result = cleaner.clean(raw_df)
        expected_types = {
            "mmsi": np.integer,
            "longitude": np.floating,
            "latitude": np.floating,
            "sog": np.floating,
            "rot_abs": np.floating,
            "flag_sat_src": np.bool_,
            "flag_ter_src": np.bool_,
        }
        for col, expected in expected_types.items():
            assert np.issubdtype(result[col].dtype, expected), (
                f"{col}: expected {expected}, got {result[col].dtype}"
            )
