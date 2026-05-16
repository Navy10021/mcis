from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from cli.run_pipeline import run_pipeline
from cli.run_analysis import run_analysis
from cli.run_model import run_model


class TestRunPipeline:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help_flag(self, runner):
        result = runner.invoke(run_pipeline, ["--help"])
        assert result.exit_code == 0
        assert "Pipeline" in result.output

    def test_no_file_produces_message(self, runner, tmp_path):
        fake_config = tmp_path / "test_config.yaml"
        fake_config.write_text(
            "data:\n  raw_dir: .\n  primary_file: nonexistent.csv\n"
            "  interim_dir: .\n  processed_dir: .\n  aggregated_dir: .\n"
            "project:\n  data_validity_mode: synthetic\n  claim_level: engineering_demo\n"
            "conflict:\n  t0: '2022-02-24'\n"
        )
        result = runner.invoke(run_pipeline, [
            "--config", str(fake_config),
            "--file", str(tmp_path / "nonexistent.csv"),
            "--steps", "load",
        ])
        assert result.exit_code != 0 or "Error" in result.output or "complete" not in result.output


class TestRunAnalysis:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help_flag(self, runner):
        result = runner.invoke(run_analysis, ["--help"])
        assert result.exit_code == 0
        assert "analyses" in result.output

    def test_no_panel_produces_error(self, runner, tmp_path):
        fake_config = tmp_path / "test_config.yaml"
        fake_config.write_text(
            "data:\n  aggregated_dir: .\n"
            "output:\n  tables_dir: .\n  figures_dir: .\n"
            "conflict:\n  t0: '2022-02-24'\n"
            "analysis:\n  event_study_metric: vessel_count\n"
            "  estimation_window: [-90,-31]\n  event_window: [-30,30]\n"
        )
        result = runner.invoke(run_analysis, [
            "--config", str(fake_config),
            "--panel", str(tmp_path / "nonexistent.parquet"),
        ])
        assert result.exit_code != 0 or "Error" in result.output

    def test_empty_analysis_list(self, runner, tmp_path):
        fake_config = tmp_path / "test_config.yaml"
        fake_config.write_text(
            "data:\n  aggregated_dir: .\n"
            "output:\n  tables_dir: .\n  figures_dir: .\n"
            "conflict:\n  t0: '2022-02-24'\n"
        )
        result = runner.invoke(run_analysis, [
            "--config", str(fake_config),
            "--analyses", "",
        ])
        assert result.exit_code != 0 or "running" not in result.output.lower()


class TestRunModel:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help_flag(self, runner):
        result = runner.invoke(run_model, ["--help"])
        assert result.exit_code == 0
        assert "models" in result.output

    def test_no_panel(self, runner, tmp_path):
        fake_config = tmp_path / "test_config.yaml"
        fake_config.write_text(
            "data:\n  aggregated_dir: .\n"
            "output:\n  models_dir: .\n  tables_dir: .\n  figures_dir: .\n"
            "conflict:\n  t0: '2022-02-24'\n"
            "model:\n  features_to_use: [mean_sog]\n"
            "project:\n  data_validity_mode: synthetic\n"
        )
        result = runner.invoke(run_model, [
            "--config", str(fake_config),
            "--panel", str(tmp_path / "nonexistent.parquet"),
        ])
        assert result.exit_code != 0 or "Error" in result.output

    def test_empty_models_list(self, runner, tmp_path):
        fake_config = tmp_path / "test_config.yaml"
        fake_config.write_text(
            "data:\n  aggregated_dir: .\n"
            "output:\n  models_dir: .\n  tables_dir: .\n  figures_dir: .\n"
            "conflict:\n  t0: '2022-02-24'\n"
            "model:\n  features_to_use: [mean_sog]\n"
            "project:\n  data_validity_mode: synthetic\n"
        )
        result = runner.invoke(run_model, [
            "--config", str(fake_config),
            "--models", "",
        ])
        assert result.exit_code != 0 or "Error" in result.output or "rolling_zscore" not in result.output


class TestCLI_EndToEnd:
    """End-to-end tests that create small parquet panels and run CLI commands."""

    @pytest.fixture
    def small_panel(self, tmp_path):
        dates = pd.date_range("2021-08-24", periods=200, freq="D")
        np.random.seed(42)
        t0 = pd.Timestamp("2022-02-24")
        df = pd.DataFrame({
            "date": dates,
            "days_to_t0": (dates - t0).days,
            "vessel_count": np.random.poisson(50, 200),
            "mean_sog": np.random.uniform(8, 14, 200),
            "std_sog": np.random.uniform(1, 4, 200),
            "max_abs_rot": np.random.uniform(0, 30, 200),
            "rot_spike_count": np.random.poisson(2, 200),
            "ais_silence_count": np.random.poisson(3, 200),
            "cargo_fraction": np.random.beta(5, 5, 200),
            "tanker_fraction": np.random.beta(3, 7, 200),
            "russian_flag_fraction": np.random.beta(2, 8, 200),
            "route_entropy": np.random.uniform(2, 5, 200),
            "cog_variance": np.random.uniform(10, 100, 200),
            "post_conflict": (dates >= t0).astype(int),
        })
        path = tmp_path / "panel_small.parquet"
        df.to_parquet(path, index=False)
        return str(path), str(tmp_path)

    @pytest.fixture
    def cli_config(self, tmp_path):
        cfg = {
            "data": {
                "aggregated_dir": str(tmp_path),
            },
            "output": {
                "tables_dir": str(tmp_path / "tables"),
                "figures_dir": str(tmp_path / "figures"),
                "models_dir": str(tmp_path / "models"),
            },
            "conflict": {
                "t0": "2022-02-24",
                "event_study_windows": [-7, 0, 7],
                "estimation_window": [-90, -31],
                "event_window": [-30, 30],
            },
            "analysis": {
                "event_study_metric": "vessel_count",
                "granger_max_lag": 3,
                "its_polynomial_degree": 1,
                "significance_level": 0.05,
            },
            "model": {
                "features_to_use": [
                    "vessel_count", "mean_sog", "max_abs_rot",
                ],
                "lookahead_days": 7,
                "early_warning_window_days": 30,
                "train_normal_end": "2022-01-15",
                "calibration_start": "2022-01-16",
                "calibration_end": "2022-01-31",
                "event_eval_start": "2022-02-01",
                "event_eval_end": "2022-02-23",
                "post_event_start": "2022-02-24",
                "post_event_end": "2022-03-15",
            },
            "project": {
                "data_validity_mode": "synthetic",
                "claim_level": "engineering_demo",
            },
            "validation": {
                "forbidden_features": [
                    "days_to_t0", "post_conflict", "conflict_onset",
                    "warning_window", "event_window", "date", "time_bucket",
                ],
            },
        }
        path = tmp_path / "cli_test_config.yaml"
        import yaml
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        return str(path)

    def test_analysis_e2e(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_analysis, [
            "--config", cli_config,
            "--panel", panel_path,
            "--analyses", "event_study,its,granger",
            "--metrics", "vessel_count,mean_sog",
        ])
        assert result.exit_code == 0, result.output

    def test_model_e2e(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "rolling_zscore,ewma",
        ])
        assert result.exit_code == 0, result.output

    def test_model_with_eval_metrics(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "rolling_zscore",
            "--eval-metrics", "auc,brier",
        ])
        assert result.exit_code == 0, result.output
        assert "AUC-ROC" in result.output
        assert "Brier" in result.output

    def test_model_with_calibration_error(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "rolling_zscore",
            "--eval-metrics", "calibration_error",
        ])
        assert result.exit_code == 0, result.output
        assert "Calibration error" in result.output

    def test_shap_only_flag(self, small_panel, cli_config):
        """--shap-only should run without error (SHAP may not be installed)."""
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "rolling_zscore",
            "--shap-only",
        ])
        assert result.exit_code == 0, result.output

    def test_unknown_model_shows_warning(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "nonexistent_model",
        ])
        assert result.exit_code == 0, result.output
        assert "unknown" in result.output.lower()

    def test_model_registry_created(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "rolling_zscore",
        ])
        assert result.exit_code == 0, result.output
        registry_path = Path(tmp) / "models" / "registry" / "registry_entries.json"
        assert registry_path.exists()
        import json
        with open(registry_path) as f:
            entries = json.load(f)
        assert len(entries) >= 1
        assert entries[0]["model_name"] == "rolling_zscore"

    def test_registry_dashboard_generated(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "rolling_zscore",
        ])
        assert result.exit_code == 0, result.output
        dashboard_files = list(Path(tmp).glob("**/*dashboard*.md"))
        assert len(dashboard_files) >= 1

    def test_var_forecasting_model_e2e(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "var",
        ])
        assert result.exit_code == 0, result.output
        assert "forecasting" in result.output

    def test_shap_only_with_var(self, small_panel, cli_config):
        panel_path, tmp = small_panel
        runner = CliRunner()
        result = runner.invoke(run_model, [
            "--config", cli_config,
            "--panel", panel_path,
            "--models", "var",
            "--shap-only",
        ])
        assert result.exit_code == 0, result.output
