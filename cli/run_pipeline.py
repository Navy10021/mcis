from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import click
import yaml

import mcis.compat  # noqa: F401  (apply NumPy/statsmodels compat patches)

from mcis.aggregator import AISAggregator
from mcis.cleaner import AISCleaner
from mcis.config_schema import validate_config
from mcis.features import AISFeatureEngineer
from mcis.loader import AISLoader
from mcis.validation import compute_file_hash, validate_required_columns, write_run_metadata

logger = logging.getLogger("mcis.cli.run_pipeline")


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _log_step(step: str, elapsed: float, n_rows: int | None = None) -> None:
    msg = f"[{step}] completed in {elapsed:.1f}s"
    if n_rows is not None:
        msg += f" ({n_rows} rows)"
    click.echo(msg)


@click.command()
@click.option("--config", "-c", default="config/settings.yaml", help="Path to config YAML")
@click.option("--file", "-f", default=None, help="Input CSV file path")
@click.option(
    "--steps",
    default="all",
    help="Pipeline steps: all, load, clean, features, aggregate (comma-separated)",
)
@click.option("--date-start", default=None, help="Filter start date (ISO 8601)")
@click.option("--date-end", default=None, help="Filter end date (ISO 8601)")
@click.option("--limit", default=None, type=int, help="Max rows to load (dev use)")
@click.option("--output-dir", default=None, help="Override output directory")
def run_pipeline(
    config: str,
    file: str | None,
    steps: str,
    date_start: str | None,
    date_end: str | None,
    limit: int | None,
    output_dir: str | None,
) -> None:
    """Run the AIS data pipeline: load → clean → features → aggregate."""
    raw_cfg = load_config(config)
    cfg = validate_config(raw_cfg, defaults_path="config/settings.yaml")
    click.echo(f"Pipeline config: {config} (validated)")

    if output_dir:
        cfg["data"]["interim_dir"] = str(Path(output_dir) / "interim")
        cfg["data"]["processed_dir"] = str(Path(output_dir) / "processed")
        cfg["data"]["aggregated_dir"] = str(Path(output_dir) / "aggregated")

    if file is None:
        file = cfg["data"]["raw_dir"] + "/" + cfg["data"]["primary_file"]
    input_hash = compute_file_hash(file) if Path(file).exists() else None

    step_list = [s.strip() for s in steps.split(",")]

    if "all" in step_list:
        step_list = ["load", "clean", "features", "aggregate"]

    df = None
    loader = AISLoader(cfg)

    # ---------- Load ----------
    if "load" in step_list:
        t0 = time.time()
        click.echo(f"Loading {file} ...")
        df, loader_report = loader.load_with_report(
            file,
            date_start=date_start,
            date_end=date_end,
        )
        _log_step("load", time.time() - t0, len(df))
        write_run_metadata(cfg, Path(cfg["data"]["raw_dir"]), "load", extra={**loader_report, "input_file": str(file), "input_file_hash": input_hash})
        click.echo(f"  Schema: {len(df.columns)} cols, {len(df)} rows")
        click.echo(f"  Date range: {loader_report.get('date_min')} to {loader_report.get('date_max')}")
        click.echo(f"  Unique MMSI: {loader_report.get('unique_mmsi')}")

    # ---------- Clean ----------
    if "clean" in step_list and df is not None:
        t0 = time.time()
        click.echo("Cleaning ...")
        cleaner = AISCleaner(cfg)
        df = cleaner.clean(df)
        report = cleaner.cleaning_report()
        _log_step("clean", time.time() - t0, len(df))
        rows_dropped = report.get("total_initial", {}).get("after", 0) - report.get("total_final", {}).get("after", 0)
        click.echo(f"  Dropped: {rows_dropped} records")
        write_run_metadata(cfg, Path(cfg["data"]["interim_dir"]), "clean", extra=report)

        out_path = Path(cfg["data"]["interim_dir"]) / "ais_blacksea_cleaned.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        click.echo(f"  Saved: {out_path}")

    # ---------- Features ----------
    if "features" in step_list and df is not None:
        t0 = time.time()
        click.echo("Engineering features ...")
        engineer = AISFeatureEngineer(cfg)
        df = engineer.transform(df)
        _log_step("features", time.time() - t0, len(df))
        click.echo(f"  Feature columns: {len(df.columns)}")
        write_run_metadata(cfg, Path(cfg["data"]["processed_dir"]), "features")

        out_path = Path(cfg["data"]["processed_dir"]) / "ais_blacksea_features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        click.echo(f"  Saved: {out_path}")

    # ---------- Aggregate ----------
    if "aggregate" in step_list and df is not None:
        t0 = time.time()
        click.echo("Aggregating ...")
        aggregator = AISAggregator(cfg)
        panels = aggregator.transform(df)
        _log_step("aggregate", time.time() - t0)
        click.echo(f"  Grid panel: {len(panels['grid_daily'])} rows")
        click.echo(f"  Black Sea panel: {len(panels['blacksea_daily'])} rows")
        write_run_metadata(cfg, Path(cfg["data"]["aggregated_dir"]), "aggregate")

        agg_dir = Path(cfg["data"]["aggregated_dir"])
        agg_dir.mkdir(parents=True, exist_ok=True)
        panels["grid_daily"].to_parquet(agg_dir / "panel_daily.parquet")
        panels["blacksea_daily"].to_parquet(agg_dir / "panel_blacksea.parquet")
        click.echo(f"  Saved panels to {agg_dir}")

    click.echo("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
