from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mcis.models.registry import ModelCardRegistry


@pytest.fixture
def sample_result() -> dict:
    return {
        "model_name": "rolling_zscore",
        "formulation": "anomaly",
        "data_validity_mode": "synthetic",
        "train_period": ["2021-08-24", "2021-12-25"],
        "calibration_period": ["2021-12-26", "2022-01-24"],
        "evaluation_period": ["2022-01-25", "2022-02-23"],
        "feature_cols": ["vessel_count", "mean_sog"],
        "metrics": {
            "first_alert_lead_days": 14,
            "false_alarms_per_30_days": 2.5,
            "alert_stability": 0.85,
            "n_alerts_warning_window": 5,
            "threshold": 1.5,
        },
        "alert_dates": ["2022-02-10", "2022-02-14"],
        "first_alert_lead_days": 14,
        "placebo_p_value": 0.03,
        "caveats": ["Test caveat"],
    }


class TestModelCardRegistry:
    def test_init_creates_dir(self, tmp_path):
        registry_dir = tmp_path / "registry"
        registry = ModelCardRegistry(str(registry_dir))
        assert registry_dir.exists()
        assert registry._entries == []

    def test_register_run_adds_entry(self, tmp_path, sample_result):
        registry = ModelCardRegistry(str(tmp_path / "registry"))
        entry = registry.register_run(sample_result)
        assert entry["model_name"] == "rolling_zscore"
        assert entry["first_alert_lead_days"] == 14
        assert entry["placebo_p_value"] == 0.03
        assert entry["n_alerts_warning_window"] == 5
        assert entry["false_alarms_per_30_days"] == 2.5
        assert "timestamp" in entry

    def test_register_run_adds_extra_metrics(self, tmp_path):
        result = {
            "model_name": "test",
            "formulation": "anomaly",
            "data_validity_mode": "synthetic",
            "metrics": {"n_alerts_warning_window": 3},
            "extra_metrics": {"auc_roc": 0.85, "auc_pr": 0.72, "brier_score": 0.12},
            "feature_cols": ["a"],
        }
        registry = ModelCardRegistry(str(tmp_path / "registry"))
        entry = registry.register_run(result)
        assert entry["auc_roc"] == 0.85
        assert entry["auc_pr"] == 0.72
        assert entry["brier_score"] == 0.12

    def test_register_run_missing_metrics(self, tmp_path):
        result = {"model_name": "test", "feature_cols": []}
        registry = ModelCardRegistry(str(tmp_path / "registry"))
        entry = registry.register_run(result)
        assert entry["n_alerts_warning_window"] is None

    def test_registry_persists_across_instances(self, tmp_path, sample_result):
        registry_dir = tmp_path / "registry"
        r1 = ModelCardRegistry(str(registry_dir))
        r1.register_run(sample_result)

        r2 = ModelCardRegistry(str(registry_dir))
        assert len(r2._entries) == 1
        assert r2._entries[0]["model_name"] == "rolling_zscore"

    def test_build_registry(self, tmp_path, sample_result):
        registry = ModelCardRegistry(str(tmp_path / "registry"))
        registry.register_run(sample_result)
        result2 = dict(sample_result)
        result2["model_name"] = "ewma"
        result2["placebo_p_value"] = 0.12
        registry.register_run(result2)

        df = registry.build_registry()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "model_name" in df.columns
        assert "placebo_p_value" in df.columns
        assert df["model_name"].tolist() == ["rolling_zscore", "ewma"]

    def test_build_registry_empty(self):
        registry = ModelCardRegistry(Path(__file__).parent / "nonexistent_registry")
        df = registry.build_registry()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_generate_dashboard(self, tmp_path, sample_result):
        registry = ModelCardRegistry(str(tmp_path / "registry"))
        registry.register_run(sample_result)
        out_dir = tmp_path / "dashboard"
        path = registry.generate_dashboard(str(out_dir))
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Model Registry Dashboard" in content
        assert "rolling_zscore" in content
        assert "Summary Table" in content

    def test_generate_dashboard_empty(self, tmp_path):
        registry = ModelCardRegistry(str(tmp_path / "empty_registry"))
        out_dir = tmp_path / "dashboard"
        path = registry.generate_dashboard(str(out_dir))
        content = Path(path).read_text()
        assert "_No entries in registry._" in content

    def test_register_multiple_runs(self, tmp_path, sample_result):
        registry = ModelCardRegistry(str(tmp_path / "registry"))
        for i in range(5):
            r = dict(sample_result)
            r["model_name"] = f"model_{i}"
            registry.register_run(r)
        assert len(registry._entries) == 5
        df = registry.build_registry()
        assert len(df) == 5

    def test_entry_file_is_valid_json(self, tmp_path, sample_result):
        registry = ModelCardRegistry(str(tmp_path / "registry"))
        registry.register_run(sample_result)
        with open(registry._entries_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["model_name"] == "rolling_zscore"
