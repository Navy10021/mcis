from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AISCleaner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._report: dict[str, dict[str, int]] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        self._report = {}
        initial = len(df)
        df = df.copy()

        df = self._step_coordinate_filter(df)
        df = self._step_sog_filter(df)
        df = self._step_rot_normalize(df)
        df = self._step_heading_normalize(df)
        df = self._step_cog_normalize(df)
        df = self._step_navstatus_normalize(df)
        df = self._step_dimension_normalize(df)
        df = self._step_dedup(df)
        df = self._step_min_obs_filter(df)
        df = self._step_quality_flags(df)

        self._report["total_initial"] = {"before": initial, "after": initial}
        self._report["total_final"] = {"before": initial, "after": len(df)}
        return df

    def cleaning_report(self) -> dict[str, Any]:
        return dict(self._report)

    def _record(self, step: str, before: int, after: int) -> None:
        self._report[step] = {"before": before, "after": after}

    def _step_coordinate_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        cfg = self.config.get("spatial", {})
        lon_min = cfg.get("lon_min", -180)
        lon_max = cfg.get("lon_max", 180)
        lat_min = cfg.get("lat_min", -90)
        lat_max = cfg.get("lat_max", 90)
        mask = (
            (df["longitude"] >= lon_min)
            & (df["longitude"] <= lon_max)
            & (df["latitude"] >= lat_min)
            & (df["latitude"] <= lat_max)
        )
        df = df[mask].copy()
        self._record("coordinate_filter", before, len(df))
        return df

    def _step_sog_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        sog_max = self.config.get("cleaning", {}).get("sog_max", 50.0)
        df = df[df["sog"].between(0.0, sog_max, inclusive="both")].copy()
        self._record("sog_filter", before, len(df))
        return df

    def _step_rot_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        rot_no_info = self.config.get("cleaning", {}).get("rot_no_info_value", 128)
        rot_clamp = self.config.get("cleaning", {}).get("rot_clamp", 127.0)
        df.loc[df["rot"].abs() == rot_no_info, "rot"] = np.nan
        df["rot"] = df["rot"].clip(-rot_clamp, rot_clamp)
        df["rot_abs"] = df["rot"].abs()
        self._record("rot_normalize", before, len(df))
        return df

    def _step_heading_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        heading_no_info = self.config.get("cleaning", {}).get("heading_no_info", 511)
        df.loc[df["heading"] == heading_no_info, "heading"] = np.nan
        self._record("heading_normalize", before, len(df))
        return df

    def _step_cog_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        cog_invalid = self.config.get("cleaning", {}).get("cog_valid_max", 360.0)
        df.loc[df["cog"] == cog_invalid, "cog"] = np.nan
        self._record("cog_normalize", before, len(df))
        return df

    def _step_navstatus_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        unknown_codes = self.config.get("cleaning", {}).get("navstatus_unknown", [95, 98, 99])
        mask = df["navStatus"].isin(unknown_codes)
        df.loc[mask, "navStatus"] = -1
        self._record("navstatus_normalize", before, len(df))
        return df

    def _step_dimension_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        zero_as_null = self.config.get("cleaning", {}).get("length_zero_as_null", True)
        if zero_as_null:
            df.loc[df["length"] == 0, "length"] = np.nan
            df.loc[df["width"] == 0, "width"] = np.nan
        self._record("dimension_normalize", before, len(df))
        return df

    def _step_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["mmsi", "posDt"], keep="first").copy()
        self._record("dedup", before, len(df))
        return df

    def _step_min_obs_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        min_obs = self.config.get("cleaning", {}).get("min_vessel_obs", 3)
        counts = df.groupby("mmsi").size()
        valid = counts[counts >= min_obs].index
        df = df[df["mmsi"].isin(valid)].copy()
        self._record("min_obs_filter", before, len(df))
        return df

    def _step_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df["flag_sat_src"] = df["posSrc"] == "SAT"
        df["flag_ter_src"] = df["posSrc"] == "TER"
        df["flag_roam_src"] = df["posSrc"] == "ROAM"
        df["flag_no_heading"] = df["heading"].isna()
        df["flag_no_imo"] = df["imo"].isna()
        df["flag_no_destination"] = df["destination"].isna()
        df["flag_invalid_dimension"] = df["length"].isna() | df["width"].isna()
        min_obs = self.config.get("cleaning", {}).get("min_vessel_obs", 3)
        counts = df.groupby("mmsi")["mmsi"].transform("count")
        df["flag_sparse_vessel"] = counts < (min_obs * 2)
        self._record("quality_flags", before, len(df))
        return df
