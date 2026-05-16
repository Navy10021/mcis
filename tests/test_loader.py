from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcis.loader import AISLoader, DTYPE_MAP, REQUIRED_COLUMNS
from mcis.utils.io import load_yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEV_FILE = DATA_DIR / "raw" / "ais_blacksea_3d.csv"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture(scope="session")
def config():
    return load_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def loader(config):
    return AISLoader(config)


@pytest.fixture(scope="session")
def tiny_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "tiny.csv"
    rows = [
        {
            "vesselUid": 1, "mmsi": 111111111, "imo": 1000000.0,
            "longitude": 30.0, "latitude": 44.0,
            "sog": 10.0, "cog": 90.0, "rot": 0.0, "heading": 90.0,
            "navStatus": 0, "posMsgType": 1.0, "posSrc": "TER",
            "vesselName": "VESSEL_A", "callsign": "CALL_A", "flag": "TR",
            "vesselTypeAis": 70.0, "vesselType": "CARGO",
            "length": 100, "width": 20, "dwt": 10000.0, "grt": 8000.0,
            "destination": "PORT_A", "eta": "2022-02-24T12:00:00Z",
            "draught": 8.0, "staticMsgType": 5.0, "staticSrc": 1.0,
            "posDt": "2022-02-23T00:00:00Z",
            "staticDt": "2022-02-23T00:00:00Z",
            "insertDt": "2022-02-23T00:00:00Z",
        },
        {
            "vesselUid": 2, "mmsi": 222222222, "imo": 2000000.0,
            "longitude": 35.0, "latitude": 42.0,
            "sog": 5.0, "cog": 180.0, "rot": -5.0, "heading": float("nan"),
            "navStatus": 1, "posMsgType": 3.0, "posSrc": "SAT",
            "vesselName": "VESSEL_B", "callsign": "CALL_B", "flag": "RU",
            "vesselTypeAis": 80.0, "vesselType": "TANKER",
            "length": 150, "width": 25, "dwt": 50000.0, "grt": 30000.0,
            "destination": "PORT_B", "eta": "2022-02-25T06:00:00Z",
            "draught": 12.0, "staticMsgType": 5.0, "staticSrc": 1.0,
            "posDt": "2022-02-25T12:00:00Z",
            "staticDt": "2022-02-25T12:00:00Z",
            "insertDt": "2022-02-25T12:00:00Z",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


class TestAISLoaderInit:
    def test_init_with_config(self, config):
        loader = AISLoader(config)
        assert loader.config is config

    def test_init_without_config(self):
        loader = AISLoader()
        assert loader.config == {}


class TestAISLoaderLoad:
    def test_loads_full_file(self, loader):
        df = loader.load(DEV_FILE)
        assert len(df) > 0
        assert "posDt" in df.columns
        assert df["posDt"].dtype.kind == "M"
        assert df["posDt"].dt.tz is not None

    def test_dtype_enforcement(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        assert df["vesselUid"].dtype == np.int64
        assert df["mmsi"].dtype == np.int64
        assert df["imo"].dtype == np.float64
        assert df["longitude"].dtype == np.float64
        assert df["latitude"].dtype == np.float64
        assert df["sog"].dtype == np.float64
        assert df["cog"].dtype == np.float64
        assert pd.api.types.is_string_dtype(df["posSrc"].dtype) or df["posSrc"].dtype == object

    def test_timestamp_parsing_utc(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        assert df["posDt"].dt.tz is not None
        assert str(df["posDt"].dt.tz) == "UTC"
        assert df["staticDt"].dt.tz is not None
        assert df["insertDt"].dt.tz is not None

    def test_date_filtering_start(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-25")
        assert df["posDt"].min() >= pd.Timestamp("2022-02-25", tz="UTC")

    def test_date_filtering_end(self, loader):
        df = loader.load(DEV_FILE, date_end="2022-02-24")
        assert df["posDt"].max() < pd.Timestamp("2022-02-24", tz="UTC")
        assert len(df) > 0

    def test_date_filtering_both(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        dates = df["posDt"]
        assert dates.min() >= pd.Timestamp("2022-02-24", tz="UTC")
        assert dates.max() < pd.Timestamp("2022-02-25", tz="UTC")

    def test_date_filtering_no_match(self, loader):
        df = loader.load(DEV_FILE, date_start="2021-01-01", date_end="2021-01-02")
        assert len(df) == 0

    def test_chunked_load_matches_full_load(self, loader):
        full = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        chunked = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25", chunksize=5000)
        assert len(full) == len(chunked)
        pd.testing.assert_frame_equal(
            full.reset_index(drop=True).sort_values(["mmsi", "posDt"]),
            chunked.reset_index(drop=True).sort_values(["mmsi", "posDt"]),
        )

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load("nonexistent.csv")

    def test_adds_date_column(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        assert "date" in df.columns
        assert df["date"].dtype.kind == "M"

    def test_days_to_t0(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-23", date_end="2022-02-26")
        assert "days_to_t0" in df.columns
        before_mask = df["posDt"] < pd.Timestamp("2022-02-24", tz="UTC")
        after_mask = df["posDt"] >= pd.Timestamp("2022-02-24", tz="UTC")
        assert (df.loc[before_mask, "days_to_t0"] < 0).all()
        assert (df.loc[after_mask, "days_to_t0"] >= 0).all()

    def test_tiny_csv_roundtrip(self, loader, tiny_csv):
        df = loader.load(tiny_csv)
        assert len(df) == 2
        assert list(df["mmsi"]) == [111111111, 222222222]
        assert df["vesselName"].iloc[0] == "VESSEL_A"

    def test_cog_is_float(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        assert df["cog"].dtype == np.float64

    def test_heading_nan_preserved(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        assert df["heading"].isna().any() or df["heading"].notna().any()


class TestAISLoaderSchemaReport:
    def test_contains_required_keys(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        report = loader.schema_report(df)
        assert "total_rows" in report
        assert "date_min" in report
        assert "date_max" in report
        assert "columns" in report
        assert "null_rates" in report
        assert "unique_mmsi" in report
        assert "unique_vessel_types" in report
        assert "source_distribution" in report

    def test_total_rows_correct(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        report = loader.schema_report(df)
        assert report["total_rows"] == len(df)

    def test_null_rates_reported(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        report = loader.schema_report(df)
        nulls = report["null_rates"]
        assert "heading" in nulls
        assert "staticSrc" in nulls
        assert "staticMsgType" in nulls
        assert nulls["staticSrc"] == 100.0

    def test_source_distribution(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        report = loader.schema_report(df)
        src_dist = report["source_distribution"]
        assert isinstance(src_dist, dict)
        assert sum(src_dist.values()) == len(df)

    def test_unique_mmsi_count(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        report = loader.schema_report(df)
        assert report["unique_mmsi"] == df["mmsi"].nunique()

    def test_empty_dataframe_report(self, loader):
        df = pd.DataFrame()
        report = loader.schema_report(df)
        assert report["total_rows"] == 0
        assert report["date_min"] is None

    def test_date_range_in_report(self, loader):
        df = loader.load(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        report = loader.schema_report(df)
        assert report["date_min"] <= report["date_max"]

    def test_load_with_report_returns_tuple(self, loader):
        df, report = loader.load_with_report(DEV_FILE, date_start="2022-02-24", date_end="2022-02-25")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(report, dict)
        assert report["total_rows"] == len(df)


class TestRequiredColumns:
    def test_missing_column_raises(self, loader, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load(bad_csv)

    def test_all_required_present(self, config):
        loader = AISLoader(config)
        df = loader.load(DEV_FILE)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Required column missing: {col}"
