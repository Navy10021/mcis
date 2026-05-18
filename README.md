# 🚢 MCIS — Maritime Conflict Intelligence System

<p align="center">
  <strong>A research-grade AIS analytics framework for maritime anomaly detection and early-warning studies.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20Pipeline-6A5ACD">
</p>

MCIS is a reproducible Python framework for analyzing AIS (Automatic Identification System) data to detect maritime behavioral anomalies around conflict-relevant events.

The current primary case study is Black Sea maritime dynamics around **February 24, 2022 (T0)**.

---

## ✨ Why MCIS?

- **Event-aware Maritime Analytics** for pre/post event behavior shifts.
- **End-to-end Reproducibility** from raw CSV to model-ready panels.
- **Methodological Guardrails** to reduce leakage and unsupported claims.
- **Baseline-first Modeling** before moving to heavier deep-learning approaches.

---

## 🧭 Table of Contents

- [Project Layout](#-project-layout)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [CLI Guide](#-cli-guide)
- [Output Artifacts](#-output-artifacts)
- [Testing](#-testing)
- [Research Guardrails](#-research-guardrails)
- [Recommended Workflow](#-recommended-workflow)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🗂 Project Layout

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
│   ├── compat.py             # Dependency compatibility shims (NumPy, SciPy, statsmodels)
│   ├── loader.py
│   ├── cleaner.py
│   ├── features.py
│   ├── aggregator.py
│   ├── validation.py
│   ├── analysis/
│   ├── models/
│   ├── viz/
│   └── utils/
├── notebook/
│   └── mcis_pipeline.ipynb   # End-to-end Jupyter notebook
├── outputs/                  # Tables, metadata, model artifacts, reports
├── tests/                    # Pytest suite (376 tests)
├── ROADMAP.md                # Detailed technical roadmap and guardrails
├── pyproject.toml
└── requirements.txt
```

---

## ⚙️ Installation

### Requirements

- Python **3.11+**
- `pip`

### Base Install

```bash
pip install -e .
```

### Optional Extras

| Extra | Includes | Command |
|---|---|---|
| `dev` | pytest, coverage, notebooks | `pip install -e ".[dev]"` |
| `ml` | xgboost, torch, shap | `pip install -e ".[ml]"` |
| `geo` | geopandas, shapely, folium, plotly | `pip install -e ".[geo]"` |
| `all` | all optional groups | `pip install -e ".[all]"` |

> [!NOTE]
> Some optional packages (especially `torch`, `geopandas`) may require longer installs and additional system libraries.

---

## 🚀 Quick Start

### 1) Validate config + guardrails

```bash
pytest tests/test_validation.py -v
```

### 2) Run pipeline (load → clean → features → aggregate)

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

### 5) Explore via Jupyter notebook

```bash
jupyter notebook notebook/mcis_pipeline.ipynb
```

The notebook covers the full pipeline: data loading, cleaning, feature engineering, aggregation, event studies,
ITS/Granger/DiD analysis, anomaly detection, forecasting, model cards, and interactive maps.

---

## 🧰 CLI Guide

### `cli/run_pipeline.py`

**Purpose**
- Load raw AIS CSV data
- Apply cleaning and quality-flagging
- Engineer vessel-level features
- Aggregate to grid/day and Black Sea/day panels

**Example**

```bash
python cli/run_pipeline.py \
  --config config/settings.yaml \
  --file data/raw/ais_blacksea_12m.csv \
  --steps load,clean,features,aggregate \
  --date-start 2021-08-24 \
  --date-end 2022-08-24
```

### `cli/run_analysis.py`

**Purpose**
- Run selected analyses (event study, ITS, DiD, Granger) by metric
- Save outputs as structured JSON/table artifacts

**Example**

```bash
python cli/run_analysis.py \
  --config config/settings.yaml \
  --panel data/aggregated/panel_blacksea.parquet \
  --analyses event_study its granger \
  --metrics vessel_count unique_mmsi mean_sog
```

### `cli/run_model.py`

**Purpose**
- Train/evaluate temporal anomaly and forecasting-error models
- Generate model artifacts, model cards, and registry records

**Example**

```bash
python cli/run_model.py \
  --config config/settings.yaml \
  --panel data/aggregated/panel_blacksea.parquet \
  --models rolling_zscore ewma robust_mahalanobis var_residual
```

---

## 📦 Output Artifacts

Typical outputs after successful execution:

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

## ✅ Testing

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

## 🔧 Dependency Compatibility

MCIS includes a compatibility shim (`mcis/compat.py`) that patches breaking changes in commonly paired
versions of NumPy, SciPy, and statsmodels:

| Issue | Symptom | Fix |
|---|---|---|
| `np.MachAr` removed in NumPy ≥2.0 | `AttributeError` in statsmodels internals | `compat.py` restores `np.MachAr` |
| `scipy.signal.signaltools._centered` moved in SciPy ≥1.17 | `ImportError` in statsmodels internals | `compat.py` restores the import path |

The compat module is imported automatically at the top of all CLI entrypoints and package `__init__.py` files
so patches apply before any statsmodels-dependent code runs.

**If you encounter missing-attribute errors from statsmodels**, ensure `import mcis.compat` runs before
any statsmodels imports (lazy imports are used in `mcis/analysis/its.py`, `mcis/analysis/did.py`,
`mcis/analysis/granger.py` and optional geo deps in `mcis/viz/maps.py`).

---

## 🛡 Research Guardrails

MCIS intentionally enforces methodological constraints for research validity:

1. **No random train/test split** — temporal split only.
2. **No leakage features in model inputs** — e.g., `days_to_t0`, `post_conflict`.
3. **Validity/claim consistency** — inferential claims are restricted under non-empirical modes.
4. **Baseline-first strategy** — interpretable methods before complex deep models.

For full policy details, see:
- `ROADMAP.md`
- `config/settings.yaml`
- `mcis/validation.py`

---

## 🔁 Recommended Workflow

1. Update `config/settings.yaml`.
2. Run `pytest tests/test_validation.py -v`.
3. Run `cli/run_pipeline.py` to generate data artifacts.
4. Run `cli/run_analysis.py` for statistical outputs.
5. Run `cli/run_model.py` for model outputs and model cards.
6. Run regression checks with `pytest tests/ --cov=mcis`.

---

## 🧯 Troubleshooting

- **`ModuleNotFoundError: No module named 'mcis'`**  
  Run editable install from repo root: `pip install -e .`

- **`SHAP not installed` in model CLI**  
  Install optional dependency: `pip install shap>=0.44.0`

- **Panel file not found**  
  Run the pipeline first with `--steps all`.

- **Missing model feature configuration**  
  Check `model.features_to_use` in `config/settings.yaml`.

- **Memory pressure on large CSV files**  
  Use `--date-start`, `--date-end`, and/or `--limit` for development runs.

- **`AttributeError: module 'numpy' has no attribute 'MachAr'`**  
  Upgrade mismatch between NumPy ≥2.0 and statsmodels 0.14.x. Fixed automatically by `mcis/compat.py` — ensure you import `mcis.compat` before any statsmodels calls.

- **`ImportError: cannot import name '_centered' from 'scipy.signal.signaltools'`**  
  SciPy ≥1.17 moved this private function. Fixed automatically by `mcis/compat.py`.

- **`ModuleNotFoundError: No module named 'folium'` or `'plotly'`**  
  Install geo extras: `pip install -e ".[geo]"`. The notebook and code gracefully degrade with clear error messages when these are missing.

- **Notebook cells fail with stale imports after code changes**  
  Restart the kernel (Kernel → Restart & Run All) to pick up updated `.py` files.

---

## 📄 License

MIT License.
