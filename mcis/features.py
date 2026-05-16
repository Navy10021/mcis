from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def route_entropy(cog_series: pd.Series, n_bins: int = 36) -> float:
    if len(cog_series.dropna()) < 5:
        return np.nan
    counts, _ = np.histogram(cog_series.dropna(), bins=n_bins, range=(0, 360))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _haversine_series(
    lat1: pd.Series, lon1: pd.Series,
    lat2: pd.Series, lon2: pd.Series,
) -> pd.Series:
    R = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return pd.Series(R * c, index=lat1.index)


def _classify_vessel_category(vessel_type: pd.Series, config: dict) -> pd.Series:
    vt_lower = vessel_type.fillna("").str.lower().str.strip()

    cargo_keywords = [t.lower() for t in config.get("features", {}).get("cargo_vessel_types", [])]
    tanker_keywords = [t.lower() for t in config.get("features", {}).get("tanker_types", [])]
    military_keywords = [t.lower() for t in config.get("features", {}).get("military_vessel_types", [])]

    def _classify(vt: str) -> str:
        if not vt:
            return "other"
        if any(kw in vt for kw in cargo_keywords):
            return "cargo"
        if any(kw in vt for kw in tanker_keywords):
            return "tanker"
        if any(kw in vt for kw in military_keywords):
            return "military_para"
        if any(kw in vt for kw in ["passenger", "passeng"]):
            return "passenger"
        if any(kw in vt for kw in ["fish", "fishing"]):
            return "fishing"
        if any(kw in vt for kw in ["tug", "supply", "offshore", "dredge", "pilot"]):
            return "support"
        return "other"

    return vessel_type.apply(lambda x: _classify(str(x).lower().strip()))


def _classify_flag_group(flag: pd.Series, config: dict) -> pd.Series:
    groups = config.get("features", {}).get("flag_risk_groups", {})
    ru_flags = set(groups.get("russia", []))
    ua_flags = set(groups.get("ukraine", []))
    nato_flags = set(groups.get("nato", []))
    conv_flags = set(groups.get("convenience", []))

    def _group(f: object) -> str:
        if pd.isna(f) or not str(f).strip() or str(f).strip().upper() == "NAN":
            return "other"
        f_upper = str(f).strip().upper()
        if f_upper in ru_flags:
            return "russia"
        if f_upper in ua_flags:
            return "ukraine"
        if f_upper in nato_flags:
            return "nato"
        if f_upper in conv_flags:
            return "convenience"
        return "other"

    return flag.apply(_group)


class AISFeatureEngineer:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df["posDt"]):
            df["posDt"] = pd.to_datetime(df["posDt"], utc=True)

        df = self._row_level_features(df)
        df = df.sort_values(["mmsi", "posDt"]).reset_index(drop=True)
        df = self._trajectory_features(df)
        df = self._rolling_features(df)

        return df

    def _row_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.get("features", {})

        df["speed_state"] = pd.cut(
            df["sog"],
            bins=[-0.1, 0.3, 3.0, 10.0, 20.0, np.inf],
            labels=["stopped", "drifting", "slow", "normal", "fast"],
        )

        rot_thresh = cfg.get("rot_spike_threshold", 20.0)
        df["rot_spike"] = df["rot_abs"] > rot_thresh

        df["vessel_category"] = _classify_vessel_category(df["vesselType"], self.config)
        df["flag_group"] = _classify_flag_group(df["flag"], self.config)

        safe_length = df["length"].fillna(0).clip(lower=1)
        df["draught_fraction"] = df["draught"] / (safe_length * 0.06)
        df.loc[df["draught_fraction"] > 1.5, "draught_fraction"] = np.nan

        return df

    def _trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.get("features", {})
        silence_gap = cfg.get("ais_silence_gap_hours", 24)

        def compute_vessel_features(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("posDt").copy()

            g["time_gap_hours"] = g["posDt"].diff().dt.total_seconds() / 3600

            g["ais_silence"] = g["time_gap_hours"] > silence_gap
            g.loc[g["posSrc"] == "SAT", "ais_silence"] = False

            g["step_dist_km"] = _haversine_series(
                g["latitude"].shift(), g["longitude"].shift(),
                g["latitude"], g["longitude"],
            )
            g.loc[g["step_dist_km"].isna(), "step_dist_km"] = 0.0

            g["implied_speed_kt"] = (
                g["step_dist_km"] / g["time_gap_hours"].replace(0, np.nan) / 1.852
            ).clip(0, 60)
            g["speed_discrepancy"] = (g["sog"] - g["implied_speed_kt"]).abs()

            cog_change = g["cog"].diff().abs()
            g["cog_change"] = cog_change.apply(
                lambda x: min(x, 360 - x) if pd.notna(x) else np.nan
            )
            g["turn_event"] = g["cog_change"] > 45.0

            g["dest_changed"] = g["destination"] != g["destination"].shift()

            return g

        pieces = []
        for _, group in df.groupby("mmsi"):
            pieces.append(compute_vessel_features(group))
        df = pd.concat(pieces, ignore_index=True)
        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("posDt")

        rolling_cols = {
            "sog_rolling_mean_7d": ("sog", "mean"),
            "sog_rolling_std_7d": ("sog", "std"),
            "rot_spike_rolling_count_7d": ("rot_spike", "sum"),
            "ais_silence_rolling_count_7d": ("ais_silence", "sum"),
            "cog_change_rolling_std_7d": ("cog_change", "std"),
        }

        for new_col, (source_col, agg) in rolling_cols.items():
            df[new_col] = (
                df.groupby("mmsi")[source_col]
                .transform(lambda x: x.rolling("7D", min_periods=1).agg(agg))
            )

        df = df.reset_index()
        return df
