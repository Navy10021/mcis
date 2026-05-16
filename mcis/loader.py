from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mcis.utils.io import load_yaml

DTYPE_MAP: dict[str, str] = {
    "vesselUid": "int64",
    "mmsi": "int64",
    "imo": "float64",
    "longitude": "float64",
    "latitude": "float64",
    "sog": "float64",
    "cog": "float64",
    "rot": "float64",
    "heading": "float64",
    "navStatus": "int64",
    "posMsgType": "float64",
    "posSrc": "str",
    "vesselName": "str",
    "callsign": "str",
    "flag": "str",
    "vesselTypeAis": "float64",
    "vesselType": "str",
    "length": "int64",
    "width": "int64",
    "dwt": "float64",
    "grt": "float64",
    "destination": "str",
    "eta": "str",
    "draught": "float64",
    "staticMsgType": "float64",
    "staticSrc": "float64",
    "posDt": "str",
    "staticDt": "str",
    "insertDt": "str",
}

REQUIRED_COLUMNS = list(DTYPE_MAP.keys())


class AISLoader:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def load(
        self,
        filepath: str | Path,
        date_start: str | None = None,
        date_end: str | None = None,
        chunksize: int | None = None,
    ) -> pd.DataFrame:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"AIS file not found: {filepath}")

        if chunksize is not None:
            chunks = []
            for chunk in pd.read_csv(
                filepath,
                dtype=DTYPE_MAP,
                chunksize=chunksize,
                low_memory=False,
            ):
                chunk = self._parse_and_filter(chunk, date_start, date_end)
                if not chunk.empty:
                    chunks.append(chunk)
            if not chunks:
                return pd.DataFrame()
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(filepath, dtype=DTYPE_MAP, low_memory=False)
            df = self._parse_and_filter(df, date_start, date_end)

        return df

    def _parse_and_filter(
        self,
        df: pd.DataFrame,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        df["posDt"] = pd.to_datetime(df["posDt"], utc=True)
        df["staticDt"] = pd.to_datetime(df["staticDt"], utc=True)
        df["insertDt"] = pd.to_datetime(df["insertDt"], utc=True)

        t0_str = self.config.get("conflict", {}).get("t0", "2022-02-24")
        t0_ts = pd.Timestamp(t0_str, tz="UTC")
        df["date"] = df["posDt"].dt.floor("1D")
        df["days_to_t0"] = (df["date"] - t0_ts).dt.days

        if date_start is not None:
            df = df[df["posDt"] >= pd.Timestamp(date_start, tz="UTC")]
        if date_end is not None:
            df = df[df["posDt"] < pd.Timestamp(date_end, tz="UTC")]

        return df

    def load_with_report(
        self,
        filepath: str | Path,
        date_start: str | None = None,
        date_end: str | None = None,
        chunksize: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = self.load(filepath, date_start=date_start, date_end=date_end, chunksize=chunksize)
        report = self.schema_report(df)
        return df, report

    @staticmethod
    def schema_report(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {
                "total_rows": 0,
                "date_min": None,
                "date_max": None,
                "columns": [],
                "null_rates": {},
                "unique_mmsi": 0,
                "unique_vessel_types": 0,
                "source_distribution": {},
            }

        null_rates = (df.isnull().sum() / len(df) * 100).round(4).to_dict()
        null_rates = {k: float(v) for k, v in null_rates.items()}

        src_dist = {}
        if "posSrc" in df.columns:
            src_dist = df["posSrc"].value_counts().to_dict()
            src_dist = {k: int(v) for k, v in src_dist.items()}

        return {
            "total_rows": int(len(df)),
            "date_min": str(df["posDt"].min()),
            "date_max": str(df["posDt"].max()),
            "columns": list(df.columns),
            "null_rates": null_rates,
            "unique_mmsi": int(df["mmsi"].nunique()),
            "unique_vessel_types": int(df["vesselType"].nunique()) if "vesselType" in df.columns else 0,
            "source_distribution": src_dist,
        }
