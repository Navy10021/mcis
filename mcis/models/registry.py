from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mcis.utils.io import ensure_dir


class ModelCardRegistry:
    """Registry of model run summaries across multiple experiments.

    Each run is stored as a JSON entry in the registry directory.
    The registry can be queried as a DataFrame and rendered as
    a comparison dashboard.
    """

    ENTRY_FILENAME = "registry_entries.json"

    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        ensure_dir(self.registry_dir)
        self._entries_path = self.registry_dir / self.ENTRY_FILENAME
        self._entries: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if self._entries_path.exists():
            with open(self._entries_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        return []

    def _save(self) -> None:
        with open(self._entries_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2, default=str)

    def register_run(self, result: dict[str, Any]) -> dict[str, Any]:
        """Register a model run result and persist the registry."""
        entry = {
            "model_name": result.get("model_name"),
            "formulation": result.get("formulation"),
            "data_validity_mode": result.get("data_validity_mode"),
            "train_period": result.get("train_period"),
            "calibration_period": result.get("calibration_period"),
            "evaluation_period": result.get("evaluation_period"),
            "feature_count": len(result.get("feature_cols", [])),
            "first_alert_lead_days": result.get("first_alert_lead_days"),
            "placebo_p_value": result.get("placebo_p_value"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metrics = result.get("metrics", {})
        if isinstance(metrics, dict):
            entry["n_alerts_warning_window"] = metrics.get("n_alerts_warning_window")
            entry["false_alarms_per_30_days"] = metrics.get("false_alarms_per_30_days")
            entry["alert_stability"] = metrics.get("alert_stability")

        extra_metrics = result.get("extra_metrics", {})
        if isinstance(extra_metrics, dict):
            entry["auc_roc"] = extra_metrics.get("auc_roc")
            entry["auc_pr"] = extra_metrics.get("auc_pr")
            entry["brier_score"] = extra_metrics.get("brier_score")

        self._entries.append(entry)
        self._save()
        return entry

    def build_registry(self) -> pd.DataFrame:
        """Return all registry entries as a DataFrame."""
        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame(self._entries)

    def generate_dashboard(
        self,
        output_dir: str | Path,
        title: str = "Model Registry Dashboard",
    ) -> Path:
        """Generate a Markdown dashboard comparing all registered runs."""
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"registry_dashboard_{timestamp}.md"

        df = self.build_registry()

        lines: list[str] = []
        _add = lines.append

        _add(f"# {title}")
        _add("")
        _add(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        _add("")
        _add(f"Total runs: {len(self._entries)}")
        _add("")

        if df.empty:
            _add("_No entries in registry._")
        else:
            _add("## Summary Table")
            _add("")
            summary_cols = [
                "model_name", "formulation", "data_validity_mode",
                "first_alert_lead_days", "placebo_p_value",
                "false_alarms_per_30_days", "alert_stability",
            ]
            display_df = df[[c for c in summary_cols if c in df.columns]].copy()
            _add(display_df.to_markdown(index=False))
            _add("")

            _add("## Evaluation Metrics by Run")
            _add("")
            metrics_cols = [
                "model_name", "n_alerts_warning_window", "auc_roc",
                "auc_pr", "brier_score",
            ]
            metrics_df = df[[c for c in metrics_cols if c in df.columns]].copy()
            if not metrics_df.empty and metrics_df.dropna(how="all", subset=metrics_df.columns.difference(["model_name"])).shape[0] > 0:
                _add(metrics_df.to_markdown(index=False))
                _add("")

            _add("## Per-Model Detail")
            _add("")
            for _, row in df.iterrows():
                _add(f"### {row.get('model_name', 'unknown')}")
                _add("")
                _add(f"- **Formulation:** {row.get('formulation', 'N/A')}")
                _add(f"- **Data Mode:** {row.get('data_validity_mode', 'N/A')}")
                _add(f"- **Features:** {row.get('feature_count', 'N/A')}")
                _add(f"- **First Alert Lead:** {row.get('first_alert_lead_days', 'N/A')} days")
                _add(f"- **Placebo p-value:** {row.get('placebo_p_value', 'N/A')}")
                _add(f"- **Timestamp:** {row.get('timestamp', 'N/A')}")
                _add("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path
