from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mcis.features import route_entropy


AGG_SPEC: dict[str, tuple[str, str | Any]] = {
    "vessel_count": ("mmsi", "count"),
    "unique_mmsi": ("mmsi", "nunique"),
    "mean_sog": ("sog", "mean"),
    "std_sog": ("sog", "std"),
    "median_sog": ("sog", "median"),
    "mean_rot": ("rot", "mean"),
    "mean_rot_abs": ("rot_abs", "mean"),
    "max_abs_rot": ("rot_abs", "max"),
    "rot_spike_count": ("rot_spike", "sum"),
    "rot_spike_fraction": ("rot_spike", "mean"),
    "ais_silence_count": ("ais_silence", "sum"),
    "ais_silence_fraction": ("ais_silence", "mean"),
    "cog_variance": ("cog", lambda x: x.var()),
    "route_entropy": ("cog", route_entropy),
    "mean_turn_events": ("turn_event", "mean"),
    "cargo_count": ("vessel_category", lambda x: (x == "cargo").sum()),
    "tanker_count": ("vessel_category", lambda x: (x == "tanker").sum()),
    "cargo_fraction": ("vessel_category", lambda x: (x == "cargo").mean()),
    "tanker_fraction": ("vessel_category", lambda x: (x == "tanker").mean()),
    "military_para_fraction": ("vessel_category", lambda x: (x == "military_para").mean()),
    "russian_flag_count": ("flag_group", lambda x: (x == "russia").sum()),
    "ukrainian_flag_count": ("flag_group", lambda x: (x == "ukraine").sum()),
    "russian_flag_fraction": ("flag_group", lambda x: (x == "russia").mean()),
    "ukrainian_flag_fraction": ("flag_group", lambda x: (x == "ukraine").mean()),
    "nato_flag_fraction": ("flag_group", lambda x: (x == "nato").mean()),
    "sat_src_fraction": ("flag_sat_src", "mean"),
    "no_destination_fraction": ("flag_no_destination", "mean"),
    "mean_draught": ("draught", "mean"),
    "mean_draught_fraction": ("draught_fraction", "mean"),
    "mean_speed_discrepancy": ("speed_discrepancy", "mean"),
    "max_speed_discrepancy": ("speed_discrepancy", "max"),
}


def build_grid(config: dict) -> pd.DataFrame:
    spatial = config.get("spatial", {})
    res = spatial.get("grid_resolution_deg", 0.5)
    lon_min = spatial.get("lon_min", 27.5)
    lon_max = spatial.get("lon_max", 41.5)
    lat_min = spatial.get("lat_min", 40.5)
    lat_max = spatial.get("lat_max", 46.8)

    lons = np.arange(lon_min, lon_max, res)
    lats = np.arange(lat_min, lat_max, res)

    rows = []
    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            rows.append({
                "grid_id": f"{i}_{j}",
                "grid_lon_idx": i,
                "grid_lat_idx": j,
                "centroid_lon": lon + res / 2,
                "centroid_lat": lat + res / 2,
                "lon_min": lon,
                "lon_max": lon + res,
                "lat_min": lat,
                "lat_max": lat + res,
            })

    return pd.DataFrame(rows)


def assign_grid_cell(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    spatial = config.get("spatial", {})
    res = spatial.get("grid_resolution_deg", 0.5)
    lon_min = spatial.get("lon_min", 27.5)
    lat_min = spatial.get("lat_min", 40.5)

    result = df.copy()
    result["grid_lon_idx"] = ((result["longitude"] - lon_min) / res).astype(int)
    result["grid_lat_idx"] = ((result["latitude"] - lat_min) / res).astype(int)
    result["grid_id"] = (
        result["grid_lon_idx"].astype(str) + "_" + result["grid_lat_idx"].astype(str)
    )

    result["time_bucket"] = result["posDt"].dt.floor("1D")
    result["time_bucket_6h"] = result["posDt"].dt.floor("6h")

    return result


def compute_aggregation(df: pd.DataFrame, config: dict, bucket_col: str = "time_bucket") -> pd.DataFrame:
    metrics = config.get("aggregation", {}).get("metrics", list(AGG_SPEC.keys()))

    spec_subset = {k: v for k, v in AGG_SPEC.items() if k in metrics}

    required_source_cols = set()
    for _, (source_col, _) in spec_subset.items():
        if isinstance(source_col, str):
            required_source_cols.add(source_col)

    missing = required_source_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source columns for aggregation: {sorted(missing)}")

    grouping_cols = ["grid_id", bucket_col]

    groupby = df.groupby(grouping_cols)
    agg_dict = {}
    for new_col, (source_col, agg_fn) in spec_subset.items():
        if isinstance(agg_fn, str):
            agg_dict[new_col] = (source_col, agg_fn)
        else:
            agg_dict[new_col] = (source_col, agg_fn)

    panel = groupby.agg(**agg_dict).reset_index()

    conflict = config.get("conflict", {})
    t0 = pd.Timestamp(conflict.get("t0", "2022-02-24")).tz_localize(None)

    bucket_dates = panel[bucket_col]
    if bucket_dates.dt.tz is not None:
        bucket_dates = bucket_dates.dt.tz_localize(None)

    panel["post_conflict"] = (bucket_dates >= t0).astype(int)
    panel["days_to_t0"] = (bucket_dates - t0).dt.days

    if bucket_col in panel.columns and hasattr(panel[bucket_col].dt, "tz") and panel[bucket_col].dt.tz is not None:
        panel[bucket_col] = panel[bucket_col].dt.tz_localize(None)

    return panel


def build_panels(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    with_bucket = assign_grid_cell(df, config)

    daily_grid_panel = compute_aggregation(with_bucket, config, bucket_col="time_bucket")

    daily_grid_panel = daily_grid_panel.set_index(["grid_id", "time_bucket"]).sort_index()

    blacksea_agg = with_bucket.groupby("time_bucket").agg({
        "mmsi": ["count", "nunique"],
        "sog": ["mean", "std"],
        "rot_abs": ["mean", "max"],
        "rot_spike": ["sum", "mean"],
        "ais_silence": ["sum", "mean"],
        "cog": lambda x: x.var(),
        "turn_event": "mean",
        "draught": "mean",
        "draught_fraction": "mean",
        "speed_discrepancy": ["mean", "max"],
        "flag_sat_src": "mean",
        "flag_no_destination": "mean",
    })

    old_cols = blacksea_agg.columns.tolist()
    new_cols = []
    for col in old_cols:
        name = f"{col[0]}_{col[1]}" if col[1] else col[0]
        name = (name.replace("mmsi_count", "vessel_count")
                    .replace("mmsi_nunique", "unique_mmsi")
                    .replace("sog_mean", "mean_sog")
                    .replace("sog_std", "std_sog")
                    .replace("rot_abs_mean", "mean_rot_abs")
                    .replace("rot_abs_max", "max_abs_rot")
                    .replace("rot_spike_sum", "rot_spike_count")
                    .replace("rot_spike_mean", "rot_spike_fraction")
                    .replace("ais_silence_sum", "ais_silence_count")
                    .replace("ais_silence_mean", "ais_silence_fraction")
                    .replace("cog_<lambda>", "cog_variance")
                    .replace("turn_event_mean", "mean_turn_events")
                    .replace("draught_mean", "mean_draught")
                    .replace("draught_fraction_mean", "mean_draught_fraction")
                    .replace("speed_discrepancy_mean", "mean_speed_discrepancy")
                    .replace("speed_discrepancy_max", "max_speed_discrepancy")
                    .replace("flag_sat_src_mean", "sat_src_fraction")
                    .replace("flag_no_destination_mean", "no_destination_fraction"))
        new_cols.append(name)
    blacksea_agg.columns = new_cols

    vessel_cat_agg = with_bucket.groupby("time_bucket")["vessel_category"].value_counts()
    vessel_cat_pivot = vessel_cat_agg.unstack(fill_value=0)
    total = vessel_cat_pivot.sum(axis=1)
    for cat in ["cargo", "tanker", "military_para"]:
        if cat in vessel_cat_pivot.columns:
            blacksea_agg[f"{cat}_fraction"] = vessel_cat_pivot[cat] / total.replace(0, np.nan)

    flag_agg = with_bucket.groupby("time_bucket")["flag_group"].value_counts()
    flag_pivot = flag_agg.unstack(fill_value=0)
    flag_total = flag_pivot.sum(axis=1)
    rename_flag = {"russia": "russian", "ukraine": "ukrainian", "nato": "nato"}
    for group_name in ["russia", "ukraine", "nato"]:
        if group_name in flag_pivot.columns:
            col_name = f"{rename_flag[group_name]}_flag_fraction"
            blacksea_agg[col_name] = flag_pivot[group_name] / flag_total.replace(0, np.nan)

    route_entropy_daily = with_bucket.groupby("time_bucket")["cog"].apply(
        lambda x: route_entropy(x.dropna())
    ).rename("route_entropy")
    blacksea_agg = blacksea_agg.join(route_entropy_daily)

    conflict = config.get("conflict", {})
    t0 = pd.Timestamp(conflict.get("t0", "2022-02-24")).tz_localize(None)

    index_dates = blacksea_agg.index
    if index_dates.tz is not None:
        index_dates = index_dates.tz_localize(None)

    blacksea_agg["post_conflict"] = (index_dates >= t0).astype(int)
    blacksea_agg["days_to_t0"] = (index_dates - t0).days

    blacksea_agg.index = index_dates
    blacksea_panel = blacksea_agg.sort_index()

    return {
        "grid_daily": daily_grid_panel,
        "blacksea_daily": blacksea_panel,
    }


class AISAggregator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._grid: pd.DataFrame | None = None

    @property
    def grid(self) -> pd.DataFrame:
        if self._grid is None:
            self._grid = build_grid(self.config)
        return self._grid

    def transform(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not pd.api.types.is_datetime64_any_dtype(df["posDt"]):
            df = df.copy()
            df["posDt"] = pd.to_datetime(df["posDt"], utc=True)

        panels = build_panels(df, self.config)
        return panels
