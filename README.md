# MCIS (Maritime Conflict Intelligence System)

MCIS is a research-oriented Python framework for detecting maritime behavioral anomalies and supporting early-warning analysis using AIS (Automatic Identification System) data.

It is designed for reproducible conflict-adjacent maritime analytics, with an initial focus on Black Sea traffic dynamics around **February 24, 2022 (T0)**.

---

## Key Objectives

- Quantify changes in maritime traffic and vessel behavior before and after major events.
- Build a reproducible end-to-end pipeline from raw AIS records to analysis-ready panels.
- Prioritize interpretable baselines (statistical methods and anomaly detectors) before complex deep-learning models.

---

## Repository Structure

```text
mcis/
├── cli/                      # Command-line entrypoints
│   ├── run_pipeline.py       # Load → clean → feature-engineer → aggregate
│   ├── run_analysis.py       # Event-study / ITS / DiD / Granger workflows
│   └── run_model.py          # Anomaly & forecasting model workflows
├── config/
│   └── settings.yaml         # Central experiment/pipeline configuration
├── data/
│   ├── raw/                  # Source AIS CSV files
│   ├── interim/              # Cleaned intermediate artifacts
│   ├── processed/            # Feature-engineered artifacts
│   └── aggregated/           # Panel-level Parquet outputs
├── mcis/
│   ├── loader.py
│   ├── cleaner.py
│   ├── features.py
│   ├── aggregator.py
│   ├── validation.py
│   ├── analysis/
│   ├── models/
│   ├── reporting/
│   ├── viz/
│   └── utils/
├── outputs/                  # Tables, metadata, model artifacts, reports
├── tests/                    # Pytest suite
├── ROADMAP.md                # Detailed technical roadmap and guardrails
├── pyproject.toml
└── requirements.txt
```

---

## Installation

### Requirements

- Python **3.11+**
- `pip`

### Base install

```bash
pip install -e .
```

### Optional extras

```bash
# Development dependencies (pytest, coverage, notebooks)
pip install -e ".[dev]"

# ML dependencies (xgboost, torch, shap)
pip install -e ".[ml]"

# Geo / visualization dependencies (geopandas, shapely, folium, plotly)
pip install -e ".[geo]"

# Everything
pip install -e ".[all]"
```

> Note: Some optional packages (e.g., `torch`, `geopandas`) may take significant time or system libraries to install.

---

## Quick Start

### 1) Validate configuration and core guardrails

```bash
pytest tests/test_validation.py -v
```

### 2) Run the data pipeline

```bash
python cli/run_pipeline.py \
  --config config/settings.yaml \
  --file data/raw/ais_blacksea_6m.csv \
  --steps all
```

### 3) Run statistical analysis

```bash
python cli/run_analysis.py \
  --config config/settings.yaml \
  --analyses event_study its \
  --metrics vessel_count mean_sog
```

### 4) Run anomaly models

```bash
python cli/run_model.py \
  --config config/settings.yaml \
  --panel data/aggregated/panel_blacksea.parquet \
  --models rolling_zscore ewma robust_mahalanobis
```

---

## CLI Reference

## `run_pipeline.py`

Purpose:
- Load raw AIS CSV data.
- Apply cleaning and quality-flagging.
- Engineer vessel-level features.
- Aggregate to panel datasets (grid/day and Black Sea/day).

Example:

```bash
python cli/run_pipeline.py \
  --config config/settings.yaml \
  --file data/raw/ais_blacksea_12m.csv \
  --steps load,clean,features,aggregate \
  --date-start 2021-08-24 \
  --date-end 2022-08-24
```

## `run_analysis.py`

Purpose:
- Execute selected analyses (event study, ITS, DiD, Granger) by metric.
- Save result artifacts as JSON/structured tables.

Example:

```bash
python cli/run_analysis.py \
  --config config/settings.yaml \
  --panel data/aggregated/panel_blacksea.parquet \
  --analyses event_study its granger \
  --metrics vessel_count unique_mmsi mean_sog
```

## `run_model.py`

Purpose:
- Train/evaluate temporal anomaly and forecasting-error models.
- Generate model artifacts, cards, and registry outputs.

Example:

```bash
python cli/run_model.py \
  --config config/settings.yaml \
  --panel data/aggregated/panel_blacksea.parquet \
  --models rolling_zscore ewma robust_mahalanobis var_residual
```

---

## Typical Output Artifacts

After successful runs, you should see artifacts such as:

- `data/interim/ais_blacksea_cleaned.parquet`
- `data/processed/ais_blacksea_features.parquet`
- `data/aggregated/panel_daily.parquet`
- `data/aggregated/panel_blacksea.parquet`
- `outputs/tables/*.json`
- `outputs/metadata/*.json`
- `outputs/models/*.json`
- `outputs/models/*_model_card_*.md`
- `outputs/models/registry/registry_entries.json`

---

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=mcis --cov-report=term-missing
```

Run core pipeline-module tests only:

```bash
pytest tests/test_loader.py tests/test_cleaner.py tests/test_features.py tests/test_aggregator.py -v
```

---

## Research Guardrails (Important)

MCIS intentionally enforces methodological constraints for research validity:

1. **No random train/test splits** — use temporal splits only.
2. **No leakage features in model inputs** — e.g., `days_to_t0`, `post_conflict` must not be used as features.
3. **Validity/claim consistency** — inferential claims are restricted under non-empirical validity modes.
4. **Baseline-first modeling** — interpretable statistical/anomaly baselines before complex deep models.

For full policy details, refer to:
- `ROADMAP.md`
- `config/settings.yaml`
- `mcis/validation.py`

---

## Recommended Development Workflow

1. Update `config/settings.yaml`.
2. Run `pytest tests/test_validation.py -v`.
3. Run `cli/run_pipeline.py` to generate data artifacts.
4. Run `cli/run_analysis.py` for statistical outputs.
5. Run `cli/run_model.py` for model outputs and model cards.
6. Run full regression checks with `pytest tests/ --cov=mcis`.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'mcis'`**  
  Ensure editable install was completed from repository root: `pip install -e .`

- **`SHAP not installed` message in model CLI**  
  Install optional dependency: `pip install shap>=0.44.0`

- **Panel file not found**  
  Run pipeline first with `--steps all` to create aggregated panel artifacts.

- **Missing model feature configuration**  
  Check `model.features_to_use` in `config/settings.yaml`.

- **Memory pressure on large CSV files**  
  Use date-range filtering (`--date-start`, `--date-end`) and/or `--limit` for development runs.

---

## License

MIT License.
