# Model Card — rolling_zscore

## Intended Use

Early-warning research prototype for detecting abnormal maritime behavioral signals.

## Not Intended For

Operational military decision-making, vessel interdiction, attribution, or standalone conflict prediction.

## Model Details

- **Model Name:** rolling_zscore
- **Formulation:** anomaly
- **Data Validity Mode:** empirical
- **Git Commit Hash:** N/A
- **Config Snapshot Hash:** N/A
- **Input File Hash:** N/A

## Training Data

- **Train Period:** N/A to N/A
- **Feature Count:** 5
- **Features:** vessel_count, mean_sog, std_sog, ais_silence_count, cog_variance

## Evaluation Metrics

- **n_alerts_warning_window:** 0

## Early Warning Performance

- **First Alert Lead Days:** None
- **Alert Dates:** None
- **Placebo p-value:** None

## Limitations

- Single-event limitation: only one conflict onset date.

## Output Schema

```yaml
model_name: rolling_zscore
formulation: anomaly
data_validity_mode: empirical
```