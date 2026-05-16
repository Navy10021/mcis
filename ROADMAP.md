# ROADMAP.md — MCIS AI Model Development Roadmap

> **Project:** Maritime Conflict Intelligence System (MCIS)  
> **Goal:** Develop a research-valid AIS-based maritime behavioral anomaly detection and early-warning pipeline.  
> **Primary case:** Black Sea / Russia–Ukraine War onset, `T₀ = 2022-02-24`  
> **Development principle:** Build from data validity → reproducible pipeline → interpretable statistical baselines → early-warning anomaly models → forecasting-error temporal models → optional deep learning.

---

## 0. Core Development Philosophy

This project must be developed as a **research-grade early-warning anomaly detection system**, not as a generic supervised conflict classifier.

The current dataset is centered on a single conflict onset date. Therefore, the model must first learn **normal maritime behavior** from pre-conflict AIS panels and then detect whether abnormal behavior emerges before `T₀`. Supervised binary classification is allowed only as a prototype unless additional conflict events, control regions, or continuous conflict-intensity targets are added.

### Non-negotiable rules

1. **No random train/test split.** All splits must be temporal.
2. **No leakage features in model input.** `days_to_t0`, `post_conflict`, `warning_window`, `conflict_onset`, `date`, and `time_bucket` must never enter model features.
3. **No inferential claim from synthetic/augmented data.** Synthetic data is for engineering validation only.
4. **No deep learning before baselines.** Implement rolling z-score, EWMA, and robust Mahalanobis before TCN/PatchTST/TFT.
5. **No statistical analysis on raw AIS rows.** Statistical tests must use aggregated panel data.
6. **No hidden preprocessing.** All scalers, imputers, thresholds, and feature selections must be fitted on training data only.
7. **Every stage must produce artifacts.** Logs, config snapshots, row-count reports, output paths, and validation metadata are mandatory.

---

## 1. Roadmap Overview

| Phase | Objective | Primary Output | Gate to Proceed |
|---|---|---|---|
| Phase 0 | Validity and project initialization | `config/settings.yaml`, `mcis/validation.py`, metadata schema | Validity mode and claim level are enforced |
| Phase 1 | Repository scaffold and utilities | package skeleton, logging, seeds, I/O helpers | CLI can load config and write logs |
| Phase 2 | AIS ingestion | `AISLoader`, raw schema report | timestamps parsed, dtypes enforced, row counts logged |
| Phase 3 | Cleaning and quality flags | `AISCleaner`, cleaned Parquet | all physical rules tested |
| Phase 4 | Vessel-level feature engineering | `AISFeatureEngineer`, feature Parquet | trajectory features and rolling features pass tests |
| Phase 5 | Grid/time aggregation | daily and 6-hour panels | panel schema validated, no missing key metrics |
| Phase 6 | EDA and descriptive analysis | dataset stats, pre/post summaries, figures | Table 1–2 draft-ready |
| Phase 7 | Statistical event analysis | event study, ITS, placebo tests | HAC/bootstrap/FDR outputs saved |
| Phase 8 | Tier 0 anomaly baselines | z-score, EWMA, Mahalanobis detectors | first alert metrics and placebo p-values available |
| Phase 9 | Tier 1 forecasting-error models | VAR/ARIMA, LSTM/TCN/PatchTST residual models | forecast residual scores beat Tier 0 or explain failure |
| Phase 10 | Evaluation and robustness | early-warning metrics, robustness matrix | false-alarm and lead-time results stable |
| Phase 11 | Visualization and reporting | figures, tables, model cards | paper-ready artifacts generated |
| Phase 12 | Optional supervised prototype | classifier experiments only with caveats | clearly labeled as prototype or expanded-data study |

---

## 2. Recommended Development Sequence for AI Coding Agents

When Claude, Codex, or another AI coding agent modifies the repository, use this order strictly:

```text
1. config/settings.yaml
2. mcis/utils/io.py
3. mcis/utils/logging.py
4. mcis/utils/seeds.py
5. mcis/validation.py
6. mcis/loader.py
7. mcis/cleaner.py
8. mcis/features.py
9. mcis/aggregator.py
10. mcis/analysis/event_study.py
11. mcis/analysis/its.py
12. mcis/models/anomaly.py
13. mcis/models/evaluate.py
14. mcis/models/forecasting.py
15. mcis/viz/timeseries.py
16. mcis/viz/maps.py
17. mcis/reporting/tables.py
18. mcis/models/model_card.py
19. cli/run_pipeline.py
20. cli/run_analysis.py
21. cli/run_model.py
```

Do **not** implement `TFT`, `PatchTST`, or supervised `XGBoost` until Phases 2–8 are stable and tested.

---

## 3. Phase 0 — Validity, Configuration, and Claim Control

### 3.1 Objective

Establish project-level guardrails so the system cannot accidentally produce invalid empirical claims, data leakage, or unreproducible outputs.

### 3.2 Files to create or update

```text
config/settings.yaml
mcis/validation.py
mcis/utils/io.py
mcis/utils/logging.py
mcis/utils/seeds.py
outputs/metadata/
```

### 3.3 Required configuration fields

Add or confirm the following fields in `config/settings.yaml`:

```yaml
project:
  name: mcis
  random_seed: 42
  data_validity_mode: empirical     # empirical | synthetic | mixed
  claim_level: descriptive_evidence # engineering_demo | descriptive_evidence | inferential_evidence | predictive_prototype

conflict:
  t0: "2022-02-24"
  zone_name: "Black Sea / Sea of Azov"

model:
  formulation: "early_warning_anomaly"
  lookahead_days: 7
  early_warning_window_days: 30
  train_normal_start: "2021-08-24"
  train_normal_end: "2021-12-25"
  calibration_start: "2021-12-26"
  calibration_end: "2022-01-24"
  event_eval_start: "2022-01-25"
  event_eval_end: "2022-02-23"
  post_event_start: "2022-02-24"
  post_event_end: "2022-08-24"

validation:
  forbidden_features:
    - days_to_t0
    - post_conflict
    - conflict_onset
    - warning_window
    - event_window
    - date
    - time_bucket
  min_rows_per_day: 10
  min_unique_mmsi_per_day: 3
```

### 3.4 Required functions

Implement in `mcis/validation.py`:

```python
def validate_data_validity_mode(config: dict) -> None: ...
def validate_claim_level(config: dict) -> None: ...
def assert_no_leakage(feature_cols: list[str], config: dict) -> None: ...
def validate_temporal_split(panel: pd.DataFrame, config: dict) -> dict: ...
def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None: ...
def write_run_metadata(config: dict, output_dir: str, stage_name: str) -> Path: ...
```

### 3.5 Acceptance criteria

- `assert_no_leakage(['mean_sog', 'days_to_t0'])` raises `ValueError`.
- `data_validity_mode = synthetic` and `claim_level = inferential_evidence` raises `ValueError`.
- Every CLI run writes `outputs/metadata/<stage>_<timestamp>.json`.

### 3.6 Test files

```text
tests/test_validation.py
```

---

## 4. Phase 1 — Repository Scaffold and Reproducibility Utilities

### 4.1 Objective

Create a clean Python package structure with reproducible execution and consistent output handling.

### 4.2 Required utilities

#### `mcis/utils/io.py`

Functions:

```python
def load_yaml(path: str | Path) -> dict: ...
def save_json(obj: dict, path: str | Path) -> None: ...
def save_dataframe(df: pd.DataFrame, path: str | Path) -> None: ...
def ensure_dir(path: str | Path) -> Path: ...
def snapshot_config(config: dict, output_dir: str | Path) -> Path: ...
```

#### `mcis/utils/logging.py`

Functions:

```python
def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger: ...
def log_row_count(logger, stage: str, before: int, after: int) -> None: ...
def log_output_path(logger, path: str | Path) -> None: ...
```

#### `mcis/utils/seeds.py`

Function:

```python
def set_global_seed(seed: int = 42) -> None: ...
```

### 4.3 Acceptance criteria

- `python -m pytest tests/test_utils.py -v` passes.
- Running any CLI command creates a log file under `outputs/logs/`.
- Config snapshot is saved for every run.

---

## 5. Phase 2 — AIS Data Loading

### 5.1 Objective

Load raw AIS CSV files safely and reproducibly while preserving schema, timestamps, and memory efficiency.

### 5.2 Module

```text
mcis/loader.py
```

### 5.3 Class design

```python
class AISLoader:
    def __init__(self, config: dict): ...

    def load(
        self,
        filepath: str | Path,
        date_start: str | None = None,
        date_end: str | None = None,
        chunksize: int | None = None,
    ) -> pd.DataFrame: ...

    def schema_report(self, df: pd.DataFrame) -> dict: ...
```

### 5.4 Implementation requirements

- Use explicit `DTYPE_MAP`.
- Parse `posDt`, `staticDt`, and `insertDt` with `utc=True`.
- Add `date = posDt.dt.floor('1D')`.
- Add `days_to_t0` for analysis only, not model input.
- Support chunked loading for future large files.
- Write a schema report containing:
  - total rows,
  - date min/max,
  - null rate per column,
  - unique MMSI count,
  - unique vessel type count,
  - source distribution for `posSrc`.

### 5.5 Output artifacts

```text
outputs/tables/schema_report_raw.csv
outputs/metadata/load_raw_<timestamp>.json
```

### 5.6 Acceptance criteria

- `posDt` is timezone-aware UTC datetime.
- Raw row count equals source CSV row count.
- Date filtering works with inclusive start and exclusive end semantics.
- Chunked and non-chunked loads return identical rows for the same date range.

### 5.7 Tests

```text
tests/test_loader.py
```

Test cases:

```python
def test_timestamp_parsing_utc(): ...
def test_dtype_enforcement(): ...
def test_date_filtering(): ...
def test_chunked_load_matches_full_load(): ...
def test_schema_report_contains_required_keys(): ...
```

---

## 6. Phase 3 — Cleaning and Data Quality Flags

### 6.1 Objective

Remove physically invalid records, normalize AIS sentinel values, and create quality flags without imputing missing values.

### 6.2 Module

```text
mcis/cleaner.py
```

### 6.3 Class design

```python
class AISCleaner:
    def __init__(self, config: dict): ...
    def clean(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def cleaning_report(self) -> dict: ...
```

### 6.4 Cleaning order

1. Coordinate bounds filter.
2. SOG physical filter.
3. ROT normalization.
4. Heading normalization.
5. COG normalization.
6. `navStatus` normalization.
7. Vessel dimension sentinel handling.
8. Duplicate removal on `(mmsi, posDt)`.
9. Minimum observation filter by MMSI.
10. Quality flag creation.

### 6.5 Required quality flags

```python
flag_sat_src
flag_ter_src
flag_roam_src
flag_no_heading
flag_no_imo
flag_no_destination
flag_invalid_dimension
flag_sparse_vessel
```

### 6.6 Output artifacts

```text
data/interim/ais_blacksea_cleaned.parquet
outputs/tables/cleaning_report.csv
outputs/metadata/clean_<timestamp>.json
```

### 6.7 Acceptance criteria

- No longitude/latitude outside Black Sea bounding box.
- No `sog < 0` or `sog > sog_max`.
- `heading == 511` becomes `NaN`.
- `cog == 360.0` becomes `NaN`.
- `abs(rot) == 128` becomes `NaN`.
- `abs(rot) > 127` is clipped to 127.
- `length == 0` and `width == 0` become `NaN`.
- Cleaning report records rows before/after each rule.

### 6.8 Tests

```text
tests/test_cleaner.py
```

---

## 7. Phase 4 — Vessel-Level Feature Engineering

### 7.1 Objective

Transform cleaned AIS messages into interpretable vessel behavior features for anomaly detection and aggregation.

### 7.2 Module

```text
mcis/features.py
```

### 7.3 Class design

```python
class AISFeatureEngineer:
    def __init__(self, config: dict): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### 7.4 Feature groups

#### Group A — Row-level behavior

```text
speed_state
rot_abs
rot_spike
vessel_category
flag_group
draught_fraction
```

#### Group B — Trajectory behavior

```text
time_gap_hours
ais_silence
step_dist_km
implied_speed_kt
speed_discrepancy
cog_change
turn_event
dest_changed
```

#### Group C — Backward-looking rolling features

```text
sog_rolling_mean_7d
sog_rolling_std_7d
rot_spike_rolling_count_7d
ais_silence_rolling_count_7d
cog_change_rolling_std_7d
```

Important: rolling windows must be **backward-looking only**. Do not use centered windows.

### 7.5 Route entropy

Implement:

```python
def route_entropy(cog_series: pd.Series, n_bins: int = 36) -> float: ...
```

Rules:

- Return `NaN` if fewer than 5 non-null COG values.
- Use 36 bins of 10 degrees.
- Use Shannon entropy with base 2.

### 7.6 Output artifacts

```text
data/processed/ais_blacksea_features.parquet
outputs/tables/feature_missingness_report.csv
outputs/metadata/features_<timestamp>.json
```

### 7.7 Acceptance criteria

- Features are computed after sorting by `mmsi, posDt`.
- AIS silence is not flagged for SAT-only gaps unless explicitly configured.
- `implied_speed_kt` handles zero or missing time gaps safely.
- COG wraparound is handled correctly: `359° → 1°` change = `2°`, not `358°`.
- No rolling feature uses future data.

### 7.8 Tests

```text
tests/test_features.py
```

---

## 8. Phase 5 — Spatial-Temporal Aggregation

### 8.1 Objective

Aggregate vessel-level behavioral features into grid-by-time and whole-region panels for statistical analysis and model development.

### 8.2 Module

```text
mcis/aggregator.py
```

### 8.3 Class design

```python
class AISAggregator:
    def __init__(self, config: dict): ...
    def assign_grid(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def aggregate_grid_daily(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def aggregate_blacksea_daily(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def aggregate_grid_6h(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### 8.4 Required panels

```text
data/aggregated/panel_daily.parquet
data/aggregated/panel_blacksea.parquet
data/aggregated/panel_grid_6h.parquet
```

### 8.5 Required metrics

Traffic:

```text
vessel_count
unique_mmsi
```

Speed:

```text
mean_sog
std_sog
median_sog
```

Maneuvering:

```text
mean_rot_abs
max_rot_abs
rot_spike_count
rot_spike_fraction
```

AIS silence:

```text
ais_silence_count
ais_silence_fraction
```

Route behavior:

```text
cog_variance
route_entropy
mean_turn_events
```

Composition:

```text
cargo_count
tanker_count
cargo_fraction
tanker_fraction
military_para_fraction
russian_flag_count
ukrainian_flag_count
russian_flag_fraction
ukrainian_flag_fraction
nato_flag_fraction
```

Source/quality:

```text
sat_src_fraction
ter_src_fraction
no_destination_fraction
```

Spoofing proxy:

```text
mean_speed_discrepancy
max_speed_discrepancy
```

### 8.6 Panel metadata

Add only for analysis, not for model input:

```text
days_to_t0
post_conflict
warning_window
```

### 8.7 Acceptance criteria

- `panel_blacksea` has one row per date.
- `panel_daily` has index or columns equivalent to `(grid_id, date)`.
- Grid IDs are stable across runs.
- Aggregation handles empty grid-days without silent failure.
- Panel validation report shows missingness and sample size.

### 8.8 Tests

```text
tests/test_aggregator.py
```

---

## 9. Phase 6 — EDA and Descriptive Analysis

### 9.1 Objective

Create the descriptive evidence base needed before statistical tests or modeling.

### 9.2 Files

```text
notebooks/01_eda.ipynb
mcis/reporting/tables.py
mcis/viz/timeseries.py
mcis/viz/maps.py
```

### 9.3 Required tables

```text
outputs/tables/table1_dataset_statistics.csv
outputs/tables/table2_feature_descriptive_stats_pre_post.csv
outputs/tables/source_distribution_by_period.csv
outputs/tables/vessel_type_distribution_by_period.csv
outputs/tables/flag_distribution_by_period.csv
```

### 9.4 Required figures

```text
outputs/figures/fig_dataset_timeline.png
outputs/figures/fig_daily_message_count.png
outputs/figures/fig_vessel_count_pre_post.png
outputs/figures/fig_possrc_composition.png
outputs/figures/fig_blacksea_density_pre_post.html
```

### 9.5 Acceptance criteria

- EDA separates `TER`, `SAT`, and `ROAM` where relevant.
- Pre/post summaries include row count and unique MMSI count.
- Figures include `T₀` marker and date range labels.
- No inferential wording is used in EDA tables.

---

## 10. Phase 7 — Statistical Event Analysis

### 10.1 Objective

Evaluate whether maritime behavioral metrics deviate around `T₀` while accounting for autocorrelation, multiple testing, and placebo dates.

### 10.2 Modules

```text
mcis/analysis/event_study.py
mcis/analysis/its.py
mcis/analysis/placebo.py
```

### 10.3 Event study requirements

Function:

```python
def run_event_study(
    panel: pd.DataFrame,
    metric: str,
    event_date: str,
    estimation_window: tuple[int, int],
    event_window: tuple[int, int],
    config: dict,
) -> dict: ...
```

Required outputs:

```text
abnormal_values
cumulative_abnormal_values
standard_errors
p_values
p_values_fdr
significant_dates_raw
significant_dates_fdr
```

### 10.4 ITS requirements

Function:

```python
def run_its(
    panel: pd.DataFrame,
    metric: str,
    event_date: str,
    config: dict,
) -> dict: ...
```

Model:

```text
Y_t = β₀ + β₁T + β₂D + β₃(T - T₀)D + ε_t
```

Requirements:

- Use HAC/Newey-West standard errors.
- Include optional day-of-week/month controls.
- Save counterfactual trajectory.

### 10.5 Placebo tests

Function:

```python
def generate_placebo_dates(
    panel: pd.DataFrame,
    true_event_date: str,
    exclusion_days: int = 30,
) -> list[pd.Timestamp]: ...
```

Function:

```python
def run_placebo_event_study(...): ...
```

### 10.6 Output artifacts

```text
outputs/tables/event_study_results.csv
outputs/tables/its_results.csv
outputs/tables/placebo_event_study_results.csv
outputs/figures/fig_event_study_car_top_metrics.png
outputs/figures/fig_its_counterfactual_vessel_count.png
outputs/figures/fig_placebo_distribution.png
```

### 10.7 Acceptance criteria

- Every p-value table has FDR-adjusted p-values.
- Placebo results are reported next to true event results.
- Result dicts are JSON-serializable.
- The module refuses to run inferential analysis if `data_validity_mode != empirical` and `claim_level == inferential_evidence`.

---

## 11. Phase 8 — Tier 0 Early-Warning Anomaly Baselines

### 11.1 Objective

Create interpretable early-warning anomaly scores before any complex temporal model.

### 11.2 Module

```text
mcis/models/anomaly.py
```

### 11.3 Required models

```python
class RollingZScoreDetector: ...
class EWMADetector: ...
class RobustMahalanobisDetector: ...
```

### 11.4 Feature set

Primary feature columns:

```text
unique_mmsi
mean_sog
std_sog
rot_spike_fraction
ais_silence_fraction
cargo_fraction
tanker_fraction
russian_flag_fraction
ukrainian_flag_fraction
route_entropy
cog_variance
sat_src_fraction
```

### 11.5 Split usage

```text
Train normal:       2021-08-24 → 2021-12-25
Calibration:        2021-12-26 → 2022-01-24
Event evaluation:   2022-01-25 → 2022-02-23
Post-event analysis:2022-02-24 → 2022-08-24
```

Rules:

- Fit scaler/imputer only on train normal.
- Fit anomaly baseline only on train normal.
- Choose threshold only on calibration.
- Evaluate lead time only on event evaluation.
- Use post-event only for regime-shift description.

### 11.6 Required standardized output

```python
{
    "model_name": str,
    "formulation": "anomaly",
    "data_validity_mode": str,
    "train_period": [str, str],
    "calibration_period": [str, str],
    "evaluation_period": [str, str],
    "feature_cols": list[str],
    "metrics": dict,
    "alert_dates": list[str],
    "first_alert_lead_days": int | None,
    "placebo_p_value": float | None,
    "caveats": list[str],
}
```

### 11.7 Early-warning metrics

```text
first_alert_lead_days
mean_alert_lead_days
false_alarms_per_30_days
alert_precision_in_warning_window
alert_recall_in_warning_window
alert_stability
max_score_in_warning_window
placebo_p_value
```

### 11.8 Output artifacts

```text
outputs/models/rolling_zscore.joblib
outputs/models/ewma.joblib
outputs/models/robust_mahalanobis.joblib
outputs/tables/tier0_anomaly_metrics.csv
outputs/figures/rolling_zscore_anomaly_score.png
outputs/figures/ewma_anomaly_score.png
outputs/figures/robust_mahalanobis_anomaly_score.png
outputs/reports/rolling_zscore_model_card.md
outputs/reports/ewma_model_card.md
outputs/reports/robust_mahalanobis_model_card.md
```

### 11.9 Acceptance criteria

- Each model returns standardized result object.
- Alert thresholds are not selected on event evaluation days.
- Anomaly score plot includes `T₀`, warning window, and threshold.
- Feature contribution for Mahalanobis top anomaly days is saved.

### 11.10 Tests

```text
tests/test_anomaly_models.py
```

---

## 12. Phase 9 — Tier 1 Forecasting-Error Models

### 12.1 Objective

Train models to forecast expected maritime behavior and use forecast residuals as early-warning anomaly scores.

### 12.2 Module

```text
mcis/models/forecasting.py
```

### 12.3 Model order

Implement in this order:

1. Naive persistence forecast.
2. VAR or ARIMA residual baseline.
3. TCN forecaster.
4. LSTM forecaster.
5. PatchTST forecaster.
6. TFT forecaster only if sufficient data and interpretability are required.

### 12.4 Dataset class

```python
class PanelSequenceDataset(Dataset):
    def __init__(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        lookback: int,
        horizon: int,
        split: str,
    ): ...
```

Requirements:

- Input sequence shape: `(lookback, n_features)`.
- Target shape: `(horizon, n_features)` or `(n_features,)` depending on model.
- No sequence can cross from train into calibration/evaluation without explicit split control.
- No future values in input.

### 12.5 Forecasting anomaly score

For each date `t`:

```text
forecast_error_t = |X_t - X_hat_t|
standardized_error_t = zscore(forecast_error_t using train-normal residual distribution)
anomaly_score_t = weighted_mean(standardized_error_t)
```

Weighting options:

```text
equal_weight
inverse_train_std
expert_weight
learned_validation_weight
```

Default: `equal_weight`.

### 12.6 Output artifacts

```text
outputs/models/var_residual.joblib
outputs/models/tcn_forecaster.pt
outputs/models/lstm_forecaster.pt
outputs/models/patchtst_forecaster.pt
outputs/tables/tier1_forecasting_metrics.csv
outputs/figures/tcn_forecast_residual_score.png
outputs/figures/patchtst_forecast_residual_score.png
outputs/reports/tcn_forecaster_model_card.md
outputs/reports/patchtst_forecaster_model_card.md
```

### 12.7 Acceptance criteria

- Forecasting models are compared against naive persistence.
- Residual scores are calibrated on train/calibration only.
- TCN uses causal convolutions only.
- PatchTST does not use any post-target context.
- Performance is reported with lead-time and false-alarm metrics, not only MSE.

### 12.8 Tests

```text
tests/test_forecasting_models.py
```

---

## 13. Phase 10 — Evaluation, Robustness, and Placebo Framework

### 13.1 Objective

Evaluate whether warning scores are meaningful, stable, and robust to modeling choices.

### 13.2 Module

```text
mcis/models/evaluate.py
mcis/analysis/placebo.py
```

### 13.3 Required functions

```python
def compute_early_warning_metrics(
    scores: pd.Series,
    threshold: float,
    event_date: str,
    warning_window_days: int,
) -> dict: ...


def compute_false_alarm_rate(
    scores: pd.Series,
    threshold: float,
    exclusion_window: tuple[str, str],
) -> float: ...


def run_model_placebo_dates(
    panel: pd.DataFrame,
    model,
    candidate_dates: list[pd.Timestamp],
    config: dict,
) -> dict: ...
```

### 13.4 Robustness matrix

Run each major model under the following variations:

| Robustness Test | Values |
|---|---|
| Warning window | 7, 14, 30 days |
| Aggregation level | whole Black Sea, grid-level, vessel-category-level |
| Source filter | all, TER-only, non-SAT |
| Grid resolution | 0.5°, 0.25° |
| Feature group | traffic only, behavior only, composition only, all |
| Threshold rule | 95th percentile, 97.5th percentile, 3-sigma |

### 13.5 Output artifacts

```text
outputs/tables/model_comparison_early_warning.csv
outputs/tables/robustness_matrix.csv
outputs/tables/placebo_model_results.csv
outputs/figures/model_score_comparison.png
outputs/figures/robustness_heatmap.png
outputs/figures/placebo_score_distribution.png
```

### 13.6 Acceptance criteria

- Every model has a false-alarm metric.
- Every primary result has a placebo comparison.
- Robustness matrix clearly identifies unstable results.
- If the true event score is not stronger than placebo, the report states this directly.

---

## 14. Phase 11 — Visualization, Reporting, and Model Cards

### 14.1 Objective

Generate reproducible, paper-ready figures, tables, and model documentation.

### 14.2 Modules

```text
mcis/viz/timeseries.py
mcis/viz/maps.py
mcis/viz/heatmaps.py
mcis/reporting/tables.py
mcis/reporting/report_guardrails.py
mcis/models/model_card.py
```

### 14.3 Required figures

```text
Fig 1: Black Sea grid map with traffic density before/after T₀
Fig 2: Multi-metric time series panel with T₀ marker
Fig 3: Event-study cumulative abnormal values for top metrics
Fig 4: ITS fitted vs counterfactual trajectory
Fig 5: Early-warning anomaly score comparison
Fig 6: Feature contribution heatmap for top anomaly days
Fig 7: Placebo distribution of warning scores
Fig 8: Flag/vessel-type composition stacked area chart
```

### 14.4 Required tables

```text
Table 1: Dataset statistics
Table 2: Feature definitions and descriptive statistics
Table 3: Event-study results with FDR correction
Table 4: ITS regression coefficients with HAC standard errors
Table 5: Early-warning model comparison
Table 6: Robustness and placebo test summary
```

### 14.5 Model card sections

Every trained model must produce:

```markdown
# Model Card — <model_name>

## Intended Use
## Not Intended For
## Training Data
## Data Validity Mode
## Features
## Preprocessing
## Evaluation
## Placebo and Robustness Tests
## Limitations
## Ethical and Operational Caveats
```

### 14.6 Report guardrails

The report generator must block language such as:

```text
proved
caused
reliably predicts war
generalizable conflict predictor
statistically significant
```

when `data_validity_mode != empirical` or when placebo/robustness tests fail.

### 14.7 Acceptance criteria

- All figures save to `outputs/figures/` with 300 DPI for PNG outputs.
- No library function calls `plt.show()`.
- Tables are CSV and Markdown.
- Model cards are generated automatically after training.

---

## 15. Phase 12 — Optional Supervised Prototype

### 15.1 Objective

Support supervised models only as explicitly labeled prototypes, or as valid models after adding multiple events/control regions.

### 15.2 When supervised modeling is allowed

Allowed if at least one condition is true:

1. Multiple conflict events are added.
2. Control regions are added.
3. A daily conflict-intensity target is added.
4. The experiment is labeled as `predictive_prototype` or `engineering_demo`.

### 15.3 Candidate models

```text
Logistic Regression
Random Forest
XGBoost
TCN classifier
PatchTST classifier
TFT classifier
```

### 15.4 Required warning label

Every supervised result table must include:

```text
CAUTION: This is a single-event prototype. Performance metrics are not evidence of general conflict-prediction capability.
```

### 15.5 Acceptance criteria

- Supervised model code calls `assert_supervised_allowed(config)`.
- Metrics are not presented as empirical proof.
- Class imbalance and temporal split are explicitly reported.

---

## 16. CLI Roadmap

### 16.1 Pipeline CLI

```bash
python cli/run_pipeline.py \
  --config config/settings.yaml \
  --file data/raw/ais_blacksea_12m.csv \
  --steps all
```

Expected outputs:

```text
data/interim/ais_blacksea_cleaned.parquet
data/processed/ais_blacksea_features.parquet
data/aggregated/panel_daily.parquet
data/aggregated/panel_blacksea.parquet
outputs/logs/run_pipeline_<timestamp>.log
outputs/metadata/run_pipeline_<timestamp>.json
```

### 16.2 Analysis CLI

```bash
python cli/run_analysis.py \
  --config config/settings.yaml \
  --analyses event_study its placebo \
  --metrics unique_mmsi mean_sog rot_spike_fraction ais_silence_fraction route_entropy
```

Expected outputs:

```text
outputs/tables/event_study_results.csv
outputs/tables/its_results.csv
outputs/tables/placebo_event_study_results.csv
outputs/figures/
```

### 16.3 Model CLI

```bash
python cli/run_model.py \
  --config config/settings.yaml \
  --models rolling_zscore ewma robust_mahalanobis \
  --panel data/aggregated/panel_blacksea.parquet
```

Then:

```bash
python cli/run_model.py \
  --config config/settings.yaml \
  --models var_residual tcn_forecaster patchtst_forecaster \
  --panel data/aggregated/panel_blacksea.parquet
```

Expected outputs:

```text
outputs/models/
outputs/tables/tier0_anomaly_metrics.csv
outputs/tables/tier1_forecasting_metrics.csv
outputs/tables/model_comparison_early_warning.csv
outputs/reports/*_model_card.md
```

---

## 17. Testing Roadmap

### 17.1 Required test execution order

```bash
pytest tests/test_validation.py -v
pytest tests/test_utils.py -v
pytest tests/test_loader.py -v
pytest tests/test_cleaner.py -v
pytest tests/test_features.py -v
pytest tests/test_aggregator.py -v
pytest tests/test_analysis.py -v
pytest tests/test_anomaly_models.py -v
pytest tests/test_forecasting_models.py -v
pytest tests/ --cov=mcis --cov-report=html
```

### 17.2 Minimum coverage targets

| Module | Minimum Coverage |
|---|---:|
| validation.py | 95% |
| loader.py | 90% |
| cleaner.py | 90% |
| features.py | 85% |
| aggregator.py | 85% |
| analysis/ | 80% |
| models/anomaly.py | 85% |
| models/forecasting.py | 75% |
| CLI | smoke tests required |

### 17.3 Synthetic unit-test fixtures

Use tiny hand-built fixtures to verify mathematical correctness:

```text
3 vessels × 5 timestamps for loader/cleaner tests
1 vessel with known COG wraparound for trajectory tests
known route entropy distribution for entropy tests
known pre/post break for ITS tests
known injected anomaly for anomaly detector tests
```

---

## 18. Suggested 4-Week Execution Plan

### Week 1 — Reproducible Data Pipeline

Goal: raw CSV → cleaned Parquet → feature Parquet → aggregated panels.

Tasks:

1. Implement config, utilities, validation.
2. Implement loader.
3. Implement cleaner.
4. Implement feature engineering.
5. Implement aggregator.
6. Write tests through `test_aggregator.py`.

Deliverables:

```text
data/interim/ais_blacksea_cleaned.parquet
data/processed/ais_blacksea_features.parquet
data/aggregated/panel_blacksea.parquet
data/aggregated/panel_daily.parquet
outputs/tables/schema_report_raw.csv
outputs/tables/cleaning_report.csv
```

### Week 2 — EDA and Statistical Analysis

Goal: produce descriptive and event-study evidence.

Tasks:

1. Generate Table 1 and Table 2.
2. Implement event study with FDR correction.
3. Implement ITS with HAC standard errors.
4. Implement placebo date generator.
5. Generate first paper figures.

Deliverables:

```text
outputs/tables/table1_dataset_statistics.csv
outputs/tables/table2_feature_descriptive_stats_pre_post.csv
outputs/tables/event_study_results.csv
outputs/tables/its_results.csv
outputs/figures/fig_multi_metric_timeseries.png
outputs/figures/fig_event_study_car_top_metrics.png
```

### Week 3 — Early-Warning Baselines

Goal: build interpretable anomaly detectors and evaluate lead-time performance.

Tasks:

1. Implement Rolling Z-score detector.
2. Implement EWMA detector.
3. Implement Robust Mahalanobis detector.
4. Implement early-warning metric functions.
5. Implement model placebo framework.
6. Generate model cards.

Deliverables:

```text
outputs/tables/tier0_anomaly_metrics.csv
outputs/tables/placebo_model_results.csv
outputs/figures/rolling_zscore_anomaly_score.png
outputs/figures/robust_mahalanobis_anomaly_score.png
outputs/reports/robust_mahalanobis_model_card.md
```

### Week 4 — Forecasting Models and Final Reporting

Goal: compare forecasting-error models against baselines and finalize reproducible outputs.

Tasks:

1. Implement naive persistence forecast.
2. Implement VAR/ARIMA residual baseline.
3. Implement TCN forecaster.
4. Implement PatchTST forecaster only if TCN pipeline is stable.
5. Run robustness matrix.
6. Generate final model comparison tables and paper-ready figures.

Deliverables:

```text
outputs/tables/tier1_forecasting_metrics.csv
outputs/tables/model_comparison_early_warning.csv
outputs/tables/robustness_matrix.csv
outputs/figures/model_score_comparison.png
outputs/figures/robustness_heatmap.png
outputs/reports/final_experiment_summary.md
```

---

## 19. AI Agent Task Cards

Use the following task cards when asking an AI coding agent to implement each unit.

### Task Card 1 — Validation module

```text
Implement mcis/validation.py.
Requirements:
- validate data_validity_mode and claim_level compatibility
- assert_no_leakage(feature_cols, config)
- validate_temporal_split(panel, config)
- validate_required_columns(df, required)
- write_run_metadata(config, output_dir, stage_name)
Add tests in tests/test_validation.py.
Do not implement any model code.
```

### Task Card 2 — Loader

```text
Implement mcis/loader.py with AISLoader.
Requirements:
- explicit dtype map
- parse posDt/staticDt/insertDt as UTC datetime
- support date_start/date_end filtering
- support chunksize
- generate schema_report
- no cleaning logic here
Add tests in tests/test_loader.py.
```

### Task Card 3 — Cleaner

```text
Implement mcis/cleaner.py with AISCleaner.
Requirements:
- coordinate filter
- SOG filter
- ROT normalization
- heading/cog sentinel handling
- navStatus unknown mapping
- length/width zero to NaN
- duplicate removal
- min observation filter
- quality flags
- cleaning report
Add tests in tests/test_cleaner.py.
```

### Task Card 4 — Feature engineering

```text
Implement mcis/features.py.
Requirements:
- row-level behavior features
- per-vessel trajectory features
- backward-looking rolling features only
- route_entropy function
- safe handling of time gaps and COG wraparound
Add tests in tests/test_features.py.
```

### Task Card 5 — Aggregation

```text
Implement mcis/aggregator.py.
Requirements:
- vectorized grid assignment
- daily grid panel
- whole-Black-Sea daily panel
- 6-hour grid panel
- all metrics listed in ROADMAP.md
- panel validation report
Add tests in tests/test_aggregator.py.
```

### Task Card 6 — Statistical analysis

```text
Implement event_study.py, its.py, and placebo.py.
Requirements:
- event study with abnormal values and cumulative abnormal values
- FDR correction
- ITS with HAC/Newey-West standard errors
- placebo event-date framework
- JSON-serializable result dicts
Add tests in tests/test_analysis.py.
```

### Task Card 7 — Tier 0 anomaly models

```text
Implement mcis/models/anomaly.py and mcis/models/evaluate.py.
Requirements:
- RollingZScoreDetector
- EWMADetector
- RobustMahalanobisDetector
- fit only on train_normal
- threshold only on calibration
- compute early-warning metrics
- save standardized result objects
Add tests in tests/test_anomaly_models.py.
```

### Task Card 8 — Forecasting models

```text
Implement mcis/models/forecasting.py.
Requirements:
- naive persistence baseline
- VAR/ARIMA residual baseline
- PanelSequenceDataset
- TCN forecaster with causal convolutions
- forecast residual anomaly scoring
- compare against Tier 0 models
Add tests in tests/test_forecasting_models.py.
Do not implement TFT until this passes.
```

---

## 20. Final Definition of Done

The project is considered development-complete when all of the following are true:

1. `python cli/run_pipeline.py --steps all` completes on the 12-month dataset.
2. `panel_blacksea.parquet` and `panel_daily.parquet` are generated and validated.
3. Table 1 and Table 2 are generated.
4. Event study, ITS, and placebo analysis run without leakage or validity errors.
5. Rolling z-score, EWMA, and robust Mahalanobis models run successfully.
6. At least one forecasting-error model is compared against Tier 0 baselines.
7. Every model reports lead time, false-alarm rate, alert stability, and placebo p-value.
8. Every trained model has a model card.
9. All figures and tables are saved under `outputs/`.
10. No report uses inferential language when data validity does not support it.
11. `pytest tests/ --cov=mcis` passes.
12. A final `outputs/reports/final_experiment_summary.md` exists.

---

## 21. Immediate Next Step

Start with the following command sequence after repository setup:

```bash
# 1. Install editable package
pip install -e ".[dev]"

# 2. Validate configuration
python -m pytest tests/test_validation.py -v

# 3. Run the development pipeline on the 6-month file
python cli/run_pipeline.py \
  --config config/settings.yaml \
  --file data/raw/ais_blacksea_6m.csv \
  --steps all

# 4. Run full pipeline on the 12-month file only after 6-month run passes
python cli/run_pipeline.py \
  --config config/settings.yaml \
  --file data/raw/ais_blacksea_12m.csv \
  --steps all
```

If Phase 0–5 do not pass, do not proceed to modeling.
