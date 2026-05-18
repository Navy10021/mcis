from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcis.utils.io import ensure_dir


def generate_model_card(
    result: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Generate a model card markdown file from a model result dict.

    The result dict must follow the standard model output schema:
        model_name, formulation, data_validity_mode,
        train_period, calibration_period, evaluation_period,
        feature_cols, metrics, alert_dates, first_alert_lead_days,
        placebo_p_value, caveats.

    Returns path to generated markdown file.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    model_name = result.get("model_name", "unnamed_model")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_model_card_{timestamp}.md"
    path = output_dir / filename

    lines: list[str] = []
    _add = lines.append

    _add(f"# Model Card — {model_name}")
    _add("")

    # Intended use
    _add("## Intended Use")
    _add("")
    _add(
        "Early-warning research prototype for detecting abnormal "
        "maritime behavioral signals."
    )
    _add("")

    # Not intended for
    _add("## Not Intended For")
    _add("")
    _add(
        "Operational military decision-making, vessel interdiction, "
        "attribution, or standalone conflict prediction."
    )
    _add("")

    # Model details
    _add("## Model Details")
    _add("")
    _add(f"- **Model Name:** {model_name}")
    _add(f"- **Formulation:** {result.get('formulation', 'N/A')}")
    _add(f"- **Data Validity Mode:** {result.get('data_validity_mode', 'N/A')}")
    _add(f"- **Git Commit Hash:** {result.get('git_commit_hash', 'N/A')}")
    _add(f"- **Config Snapshot Hash:** {result.get('config_snapshot_hash', 'N/A')}")
    _add(f"- **Input File Hash:** {result.get('input_file_hash', 'N/A')}")

    # Training data
    _add("")
    _add("## Training Data")
    _add("")
    train = result.get("train_period", [])
    if train:
        _add(f"- **Train Period:** {train[0]} to {train[1]}")
    cal = result.get("calibration_period", [])
    if cal:
        _add(f"- **Calibration Period:** {cal[0]} to {cal[1]}")
    eval_p = result.get("evaluation_period", [])
    if eval_p:
        _add(f"- **Evaluation Period:** {eval_p[0]} to {eval_p[1]}")
    _add(f"- **Feature Count:** {len(result.get('feature_cols', []))}")
    _add(f"- **Features:** {', '.join(result.get('feature_cols', []))}")

    # Metrics
    _add("")
    _add("## Evaluation Metrics")
    _add("")
    metrics = result.get("metrics", {})
    if metrics:
        for k, v in metrics.items():
            _add(f"- **{k}:** {v}")
    else:
        _add("(No metrics reported)")

    # Early warning
    _add("")
    _add("## Early Warning Performance")
    _add("")
    _add(f"- **First Alert Lead Days:** {result.get('first_alert_lead_days', 'N/A')}")
    _add(f"- **Alert Dates:** {', '.join(result.get('alert_dates', [])) or 'None'}")
    _add(f"- **Placebo p-value:** {result.get('placebo_p_value', 'N/A')}")

    # Limitations
    _add("")
    _add("## Limitations")
    _add("")
    caveats = result.get("caveats", [])
    if caveats:
        for c in caveats:
            _add(f"- {c}")
    else:
        _add("- Single-event limitation: only one conflict onset date.")
        _add("- AIS coverage bias: SAT vs TER heterogeneity.")
        _add("- Synthetic data caveat: results may not reflect real conflict precursors.")

    # Output schema info
    _add("")
    _add("## Output Schema")
    _add("")
    _add("```yaml")
    _add(f"model_name: {model_name}")
    _add(f"formulation: {result.get('formulation', 'N/A')}")
    _add(f"data_validity_mode: {result.get('data_validity_mode', 'N/A')}")
    _add("```")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path
