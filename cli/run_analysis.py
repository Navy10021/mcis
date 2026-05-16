from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import click
import pandas as pd
import yaml

from mcis.analysis import (
    apply_multiple_testing,
    run_did,
    run_event_study,
    run_granger,
    run_its,
    run_placebo_cutdates,
)
from mcis.config_schema import validate_config
from mcis.validation import compute_file_hash, write_run_metadata

logger = logging.getLogger("mcis.cli.run_analysis")


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


ANALYSIS_MAP = {
    "event_study": run_event_study,
    "its": run_its,
    "granger": run_granger,
    "did": run_did,
}


@click.command()
@click.option("--config", "-c", default="config/settings.yaml", help="Path to config YAML")
@click.option("--panel", "-p", default=None, help="Path to aggregated panel parquet")
@click.option(
    "--analyses",
    default="event_study",
    help="Comma-separated analyses to run: event_study, its, granger, did",
)
@click.option(
    "--metrics",
    default=None,
    help="Comma-separated metrics to analyze (default: config analysis metrics)",
)
@click.option("--output-dir", "-o", default=None, help="Output directory for results")
@click.option("--grid-panel", default=None, help="Path to grid-level panel (for DiD)")
def run_analysis(
    config: str,
    panel: str | None,
    analyses: str,
    metrics: str | None,
    output_dir: str | None,
    grid_panel: str | None,
) -> None:
    """Run statistical analyses on aggregated AIS panel."""
    raw_cfg = load_config(config)
    cfg = validate_config(raw_cfg, defaults_path="config/settings.yaml")

    if output_dir:
        tables_dir = Path(output_dir) / "tables"
        figures_dir = Path(output_dir) / "figures"
    else:
        tables_dir = Path(cfg["output"]["tables_dir"])
        figures_dir = Path(cfg["output"]["figures_dir"])

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if panel is None:
        panel = str(Path(cfg["data"]["aggregated_dir"]) / "panel_blacksea.parquet")
    click.echo(f"Loading panel: {panel}")
    panel_hash = compute_file_hash(panel) if Path(panel).exists() else None
    panel_df = pd.read_parquet(panel)

    analysis_list = [a.strip() for a in analyses.split(",")]
    metric_list = None
    if metrics:
        metric_list = [m.strip() for m in metrics.split(",")]
    if metric_list is None:
        metric_list = cfg.get("analysis", {}).get("event_study_metric", [])
        if isinstance(metric_list, str):
            metric_list = [metric_list]

    t0_date = cfg["conflict"]["t0"]

    for analysis_name in analysis_list:
        if analysis_name not in ANALYSIS_MAP:
            click.echo(f"  Skipping unknown analysis: {analysis_name}", err=True)
            continue

        click.echo(f"\nRunning {analysis_name} ...")

        for metric in metric_list:
            click.echo(f"  Metric: {metric}")
            t_start = time.time()

            try:
                if analysis_name == "did":
                    if grid_panel is None:
                        click.echo("    Skipping DiD: --grid-panel required")
                        continue
                    grid_df = pd.read_parquet(grid_panel)
                    treated = cfg.get("analysis", {}).get("did_treated_grids", [])
                    control = cfg.get("analysis", {}).get("did_control_grids", [])
                    result = run_did(
                        grid_df,
                        treated_grid_ids=treated,
                        control_grid_ids=control,
                        event_date=t0_date,
                        metric=metric,
                    )
                elif analysis_name == "granger":
                    target = cfg.get("analysis", {}).get("granger_target", "post_conflict")
                    result = run_granger(
                        panel_df,
                        predictor_col=metric,
                        target_col=target,
                        max_lag=cfg.get("analysis", {}).get("granger_max_lag", 14),
                        alpha=cfg.get("analysis", {}).get("significance_level", 0.05),
                    )
                elif analysis_name == "event_study":
                    result = run_event_study(
                        panel_df,
                        metric=metric,
                        event_date=t0_date,
                        estimation_window=cfg.get("analysis", {}).get(
                            "estimation_window", (-90, -31)
                        ),
                        event_window=cfg.get("analysis", {}).get(
                            "event_window", (-30, 30)
                        ),
                        config=cfg,
                    )
                elif analysis_name == "its":
                    result = run_its(
                        panel_df,
                        metric=metric,
                        event_date=t0_date,
                        polynomial_degree=cfg.get("analysis", {}).get(
                            "its_polynomial_degree", 2
                        ),
                    )
                else:
                    click.echo(f"    Unhandled analysis: {analysis_name}", err=True)
                    continue

                elapsed = time.time() - t_start
                click.echo(f"    Done in {elapsed:.1f}s")

                result_meta = {
                    "analysis": analysis_name,
                    "metric": metric,
                    "status": result.get("status"),
                    "input_file": str(panel),
                    "input_file_hash": panel_hash,
                }
                result_filename = f"{analysis_name}_{metric}_{t0_date}.json"
                result_path = tables_dir / result_filename
                if analysis_name == "event_study" and result.get("status") == "ok":
                    placebo_offsets = cfg.get("analysis", {}).get(
                        "placebo_cutdate_offsets",
                        [-60, -45, -30, -15, 15, 30, 45, 60],
                    )
                    placebo_summary = run_placebo_cutdates(
                        panel_df,
                        metric=metric,
                        candidate_offsets=placebo_offsets,
                        estimation_window=tuple(cfg.get("analysis", {}).get("estimation_window", (-90, -31))),
                        event_window=tuple(cfg.get("analysis", {}).get("event_window", (-30, 30))),
                    )
                    result["placebo_cutdates"] = placebo_summary
                    result["multiple_testing"] = apply_multiple_testing(
                        result.get("p_values", {}),
                        alpha=cfg.get("analysis", {}).get("significance_level", 0.05),
                    )

                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, default=str)
                click.echo(f"    Saved: {result_path}")

                write_run_metadata(
                    cfg, tables_dir, f"{analysis_name}_{metric}",
                    extra=result_meta,
                )

            except Exception as e:
                click.echo(f"    FAILED: {e}", err=True)
                logger.exception(f"{analysis_name}({metric}) failed")

    click.echo("\nAll analyses complete.")


if __name__ == "__main__":
    run_analysis()
