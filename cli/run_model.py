from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml

from mcis.config_schema import validate_config
from mcis.models.anomaly import DEFAULT_ANOMALY_MODELS
from mcis.models.evaluate import (
    compute_anomaly_metrics,
    compute_classification_metrics,
    compute_forecast_error_anomaly,
    run_placebo_dates,
)
from mcis.models.forecasting import DEFAULT_FORECASTING_MODELS
from mcis.models.model_card import generate_model_card
from mcis.models.registry import ModelCardRegistry
from mcis.validation import assert_no_leakage, write_run_metadata

logger = logging.getLogger("mcis.cli.run_model")

FORBIDDEN_FEATURES: set[str] = {
    "days_to_t0",
    "post_conflict",
    "conflict_onset",
    "warning_window",
    "event_window",
    "date",
    "time_bucket",
}

ALL_MODELS: dict[str, Any] = {}
ALL_MODELS.update(DEFAULT_ANOMALY_MODELS)
ALL_MODELS.update(DEFAULT_FORECASTING_MODELS)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_shap_analysis(
    model_name: str,
    model_obj: Any,
    train_X: pd.DataFrame,
    eval_X: pd.DataFrame,
    figures_dir: Path,
    feature_cols: list[str],
) -> None:
    """Run SHAP explainability on a fitted model and save plots."""
    try:
        import shap
    except ImportError:
        click.echo("  SHAP not installed. Install with: pip install shap>=0.44.0")
        return

    click.echo(f"  Running SHAP analysis for {model_name}...")

    background = shap.sample(train_X.values, min(100, len(train_X)))

    if hasattr(model_obj, "_fitted") and model_obj._fitted is not None:
        def predict_fn(x: np.ndarray) -> np.ndarray:
            cols = feature_cols[: x.shape[1]]
            dummy_idx = pd.RangeIndex(len(x))
            x_df = pd.DataFrame(x, index=dummy_idx, columns=cols)
            pred = model_obj.predict(x_df, horizon=1)
            if isinstance(pred, pd.DataFrame):
                return pred.mean(axis=1).values
            return np.asarray(pred).ravel()
    else:
        def predict_fn(x: np.ndarray) -> np.ndarray:
            cols = feature_cols[: x.shape[1]]
            dummy_idx = pd.RangeIndex(len(x))
            x_df = pd.DataFrame(x, index=dummy_idx, columns=cols)
            pred = model_obj.predict(x_df)
            if isinstance(pred, pd.DataFrame):
                return pred.mean(axis=1).values
            return np.asarray(pred).ravel()

    try:
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(eval_X.values[:min(100, len(eval_X))])
    except Exception as e:
        click.echo(f"  SHAP explainer failed: {e}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        shap.summary_plot(
            shap_values, eval_X.values[:min(100, len(eval_X))],
            feature_names=feature_cols, show=False, plot_size=None,
        )
        fig_path = figures_dir / f"{model_name}_shap_summary.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        click.echo(f"  SHAP summary: {fig_path}")

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, eval_X.values[:min(100, len(eval_X))],
            feature_names=feature_cols, show=False, plot_type="bar",
            plot_size=None,
        )
        bar_path = figures_dir / f"{model_name}_shap_feature_importance.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        click.echo(f"  SHAP feature importance: {bar_path}")
    except Exception as e:
        click.echo(f"  SHAP plotting failed: {e}")


@click.command()
@click.option("--config", "-c", default="config/settings.yaml", help="Path to config YAML")
@click.option("--panel", "-p", default=None, help="Path to aggregated panel parquet")
@click.option(
    "--models",
    default="rolling_zscore",
    help="Comma-separated models: rolling_zscore, ewma, robust_mahalanobis, var, lstm, tcn",
)
@click.option("--output-dir", "-o", default=None, help="Output directory")
@click.option("--shap-only", is_flag=True, help="Skip evaluation, run SHAP explainability")
@click.option(
    "--eval-metrics",
    default=None,
    help="Comma-separated extra metrics: auc,brier,calibration_error",
)
def run_model(
    config: str,
    panel: str | None,
    models: str,
    output_dir: str | None,
    shap_only: bool,
    eval_metrics: str | None,
) -> None:
    """Train and evaluate early-warning models.

    Supports both anomaly detection models (rolling_zscore, ewma,
    robust_mahalanobis) and forecasting models (var, lstm, tcn).
    """
    raw_cfg = load_config(config)
    cfg = validate_config(raw_cfg, defaults_path="config/settings.yaml")

    if output_dir:
        models_dir = Path(output_dir) / "models"
        tables_dir = Path(output_dir) / "tables"
        figures_dir = Path(output_dir) / "figures"
    else:
        models_dir = Path(cfg["output"]["models_dir"])
        tables_dir = Path(cfg["output"]["tables_dir"])
        figures_dir = Path(cfg["output"]["figures_dir"])

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if panel is None:
        panel = str(Path(cfg["data"]["aggregated_dir"]) / "panel_blacksea.parquet")
    click.echo(f"Loading panel: {panel}")
    panel_df = pd.read_parquet(panel)

    model_cfg = cfg.get("model", {})
    feature_cols = model_cfg.get("features_to_use", [])

    if not feature_cols:
        click.echo("No features_to_use in config.model.", err=True)
        return

    assert_no_leakage(feature_cols, cfg)

    t0 = pd.Timestamp(cfg["conflict"]["t0"])
    warning_days = model_cfg.get("early_warning_window_days", 30)

    panel_df["date"] = panel_df.index if "date" not in panel_df.columns else panel_df["date"]
    panel_df = panel_df.sort_values("date")

    train_end = pd.Timestamp(model_cfg.get("train_normal_end", "2021-12-25"))
    calib_start = pd.Timestamp(model_cfg.get("calibration_start", "2021-12-26"))
    calib_end = pd.Timestamp(model_cfg.get("calibration_end", "2022-01-24"))
    eval_start = pd.Timestamp(model_cfg.get("event_eval_start", "2022-01-25"))
    eval_end = pd.Timestamp(model_cfg.get("event_eval_end", "2022-02-23"))

    train_df = panel_df[panel_df["date"] <= train_end].copy()
    calib_df = panel_df[(panel_df["date"] >= calib_start) & (panel_df["date"] <= calib_end)].copy()
    eval_df = panel_df[(panel_df["date"] >= eval_start) & (panel_df["date"] <= eval_end)].copy()
    post_df = panel_df[panel_df["date"] >= t0].copy()

    click.echo(f"  Train: {train_df['date'].min().date()} \u2192 {train_df['date'].max().date()} ({len(train_df)} days)")
    click.echo(f"  Calibration: {calib_df['date'].min().date()} \u2192 {calib_df['date'].max().date()} ({len(calib_df)} days)")
    click.echo(f"  Event eval: {eval_df['date'].min().date()} \u2192 {eval_df['date'].max().date()} ({len(eval_df)} days)")
    click.echo(f"  Post-event: {post_df['date'].min().date()} \u2192 {post_df['date'].max().date()} ({len(post_df)} days)")

    train_X = train_df[feature_cols].fillna(0)
    calib_X = calib_df[feature_cols].fillna(0)
    eval_X = eval_df[feature_cols].fillna(0)
    post_X = post_df[feature_cols].fillna(0)

    model_names = [m.strip() for m in models.split(",")]
    parsed_eval_metrics: list[str] = [m.strip().lower() for m in (eval_metrics or "").split(",") if m.strip()] if eval_metrics else []

    anomaly_model_names = [m for m in model_names if m in DEFAULT_ANOMALY_MODELS]
    forecasting_model_names = [m for m in model_names if m in DEFAULT_FORECASTING_MODELS]
    unknown_model_names = [m for m in model_names if m not in ALL_MODELS]

    for name in unknown_model_names:
        click.echo(f"  Skipping unknown model: {name}", err=True)

    results_summary: list[dict[str, Any]] = []
    registry = ModelCardRegistry(models_dir / "registry")

    # --- Anomaly Detection Models ---
    for model_name in anomaly_model_names:
        click.echo(f"\n--- Training: {model_name} (anomaly) ---")
        t_start = time.time()

        detector = DEFAULT_ANOMALY_MODELS[model_name]
        detector.fit(train_X)

        train_scores = detector.predict(train_X).mean(axis=1)
        calib_scores = detector.predict(calib_X).mean(axis=1)
        eval_scores = detector.predict(eval_X).mean(axis=1)
        post_scores = detector.predict(post_X).mean(axis=1)

        all_scores = pd.concat([train_scores, calib_scores, eval_scores, post_scores])
        all_dates = pd.concat([
            train_df["date"], calib_df["date"], eval_df["date"], post_df["date"],
        ])

        if shap_only:
            run_shap_analysis(
                model_name, detector, train_X, eval_X,
                figures_dir, feature_cols,
            )
            continue

        metrics_result = compute_anomaly_metrics(
            scores=eval_scores,
            dates=eval_df["date"],
            event_date=cfg["conflict"]["t0"],
            warning_window_days=warning_days,
            threshold_percentile=95.0,
        )

        candidate_dates = [str(d.date()) for d in train_df["date"].iloc[::14]]
        placebo_result = run_placebo_dates(
            scores=all_scores,
            dates=all_dates,
            true_event_date=cfg["conflict"]["t0"],
            candidate_dates=candidate_dates,
            warning_window_days=warning_days,
        )

        elapsed = time.time() - t_start
        click.echo(f"  Done in {elapsed:.1f}s")
        click.echo(f"  First alert lead: {metrics_result.get('first_alert_lead_days')} days")
        click.echo(f"  False alarms/30d: {metrics_result.get('false_alarms_per_30_days')}")
        click.echo(f"  Placebo p-value: {placebo_result.get('placebo_p_value')}")

        extra_metrics: dict[str, Any] = {}
        if "auc" in parsed_eval_metrics or "brier" in parsed_eval_metrics:
            y_true = (eval_df["date"] >= t0).astype(int).reset_index(drop=True)
            y_score = eval_scores.reset_index(drop=True)
            cls_metrics = compute_classification_metrics(y_true, y_score)
            extra_metrics["auc_roc"] = cls_metrics.get("auc_roc")
            extra_metrics["auc_pr"] = cls_metrics.get("auc_pr")
            extra_metrics["brier_score"] = cls_metrics.get("brier_score")
            click.echo(f"  AUC-ROC: {extra_metrics['auc_roc']}")
            click.echo(f"  AUC-PR: {extra_metrics['auc_pr']}")
            click.echo(f"  Brier: {extra_metrics['brier_score']}")

        if "calibration_error" in parsed_eval_metrics:
            try:
                y_true_ece = (eval_df["date"] >= t0).astype(int)
                y_score_ece = eval_scores
                from sklearn.isotonic import IsotonicRegression
                reg = IsotonicRegression(out_of_bounds="clip")
                y_prob = reg.fit_transform(y_score_ece, y_true_ece)
                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_ids = np.digitize(y_prob, bin_edges[1:-1])
                ece = 0.0
                for bin_id in range(n_bins):
                    in_bin = bin_ids == bin_id
                    if in_bin.sum() > 0:
                        bin_acc = y_true_ece[in_bin].mean()
                        bin_conf = y_prob[in_bin].mean()
                        ece += abs(bin_acc - bin_conf) * in_bin.sum()
                ece = ece / len(y_true_ece)
                extra_metrics["calibration_error"] = round(ece, 4)
                click.echo(f"  Calibration error: {extra_metrics['calibration_error']}")
            except Exception as e:
                click.echo(f"  Calibration error computation failed: {e}")

        result: dict[str, Any] = {
            "model_name": model_name,
            "formulation": "anomaly",
            "data_validity_mode": cfg.get("project", {}).get("data_validity_mode", "synthetic"),
            "train_period": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
            "calibration_period": [str(calib_df["date"].min().date()), str(calib_df["date"].max().date())],
            "evaluation_period": [str(eval_df["date"].min().date()), str(eval_df["date"].max().date())],
            "feature_cols": feature_cols,
            "metrics": metrics_result,
            "extra_metrics": extra_metrics,
            "alert_dates": [str(d.date()) for d in eval_df["date"][eval_scores > metrics_result["threshold"]].tolist()],
            "first_alert_lead_days": metrics_result.get("first_alert_lead_days"),
            "placebo_p_value": placebo_result.get("placebo_p_value"),
            "caveats": [
                "Single-event limitation: only one conflict onset date (2022-02-24).",
                "Placebo test uses internal dates; results may not generalize.",
                "Anomaly thresholds calibrated on 'normal' period only.",
            ],
        }

        result_path = tables_dir / f"{model_name}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"  Saved: {result_path}")

        card_path = generate_model_card(result, models_dir)
        click.echo(f"  Model card: {card_path}")

        registry.register_run(result)
        write_run_metadata(cfg, models_dir, f"model_{model_name}", extra={"model": model_name})
        results_summary.append(result)

    # --- Forecasting Models ---
    for model_name in forecasting_model_names:
        click.echo(f"\n--- Training: {model_name} (forecasting) ---")
        t_start = time.time()

        forecaster = DEFAULT_FORECASTING_MODELS[model_name]

        if len(train_X) < 5:
            click.echo(f"  Not enough training data for {model_name}", err=True)
            continue

        forecaster.fit(train_X)

        if shap_only:
            run_shap_analysis(
                model_name, forecaster, train_X, eval_X,
                figures_dir, feature_cols,
            )
            continue

        horizon = min(len(eval_X), 30)
        y_pred = forecaster.predict(train_X, horizon=horizon).reset_index(drop=True)
        y_true = eval_X.iloc[:horizon].reset_index(drop=True)

        eval_scores = compute_forecast_error_anomaly(y_true, y_pred, method="zscore")
        eval_scores = eval_scores.reset_index(drop=True)
        eval_scores_dates = eval_df["date"].iloc[:horizon].reset_index(drop=True)

        all_scores = eval_scores
        all_dates = eval_scores_dates

        metrics_result = compute_anomaly_metrics(
            scores=eval_scores,
            dates=eval_scores_dates,
            event_date=cfg["conflict"]["t0"],
            warning_window_days=warning_days,
            threshold_percentile=95.0,
        )

        candidate_dates = [str(d.date()) for d in train_df["date"].iloc[::14]]
        placebo_result = run_placebo_dates(
            scores=all_scores,
            dates=all_dates,
            true_event_date=cfg["conflict"]["t0"],
            candidate_dates=candidate_dates,
            warning_window_days=warning_days,
        )

        elapsed = time.time() - t_start
        click.echo(f"  Done in {elapsed:.1f}s")
        click.echo(f"  First alert lead: {metrics_result.get('first_alert_lead_days')} days")
        click.echo(f"  False alarms/30d: {metrics_result.get('false_alarms_per_30_days')}")
        click.echo(f"  Placebo p-value: {placebo_result.get('placebo_p_value')}")

        extra_metrics = {}
        if "auc" in parsed_eval_metrics or "brier" in parsed_eval_metrics:
            y_true_bin = (eval_scores_dates >= t0).astype(int)
            cls_metrics = compute_classification_metrics(y_true_bin, eval_scores)
            extra_metrics["auc_roc"] = cls_metrics.get("auc_roc")
            extra_metrics["auc_pr"] = cls_metrics.get("auc_pr")
            extra_metrics["brier_score"] = cls_metrics.get("brier_score")
            click.echo(f"  AUC-ROC: {extra_metrics['auc_roc']}")
            click.echo(f"  AUC-PR: {extra_metrics['auc_pr']}")
            click.echo(f"  Brier: {extra_metrics['brier_score']}")

        result: dict[str, Any] = {
            "model_name": model_name,
            "formulation": "forecasting",
            "data_validity_mode": cfg.get("project", {}).get("data_validity_mode", "synthetic"),
            "train_period": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
            "calibration_period": [str(calib_df["date"].min().date()), str(calib_df["date"].max().date())],
            "evaluation_period": [str(eval_df["date"].min().date()), str(eval_df["date"].max().date())],
            "feature_cols": feature_cols,
            "metrics": metrics_result,
            "extra_metrics": extra_metrics,
            "alert_dates": [str(d.date()) for d in eval_scores_dates[eval_scores > metrics_result["threshold"]].tolist()],
            "first_alert_lead_days": metrics_result.get("first_alert_lead_days"),
            "placebo_p_value": placebo_result.get("placebo_p_value"),
            "caveats": [
                "Forecast-based anomaly: scores derived from prediction errors.",
                "Single-event limitation: only one conflict onset date.",
                f"Forecast horizon: {horizon} days.",
            ],
        }

        result_path = tables_dir / f"{model_name}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"  Saved: {result_path}")

        card_path = generate_model_card(result, models_dir)
        click.echo(f"  Model card: {card_path}")

        registry.register_run(result)
        write_run_metadata(cfg, models_dir, f"model_{model_name}", extra={"model": model_name})
        results_summary.append(result)

    click.echo("\nAll models complete.")
    click.echo(f"Results saved to {tables_dir}")
    click.echo(f"Model cards saved to {models_dir}")

    if results_summary:
        dashboard_path = registry.generate_dashboard(figures_dir)
        click.echo(f"Registry dashboard: {dashboard_path}")


if __name__ == "__main__":
    run_model()
