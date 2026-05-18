# Model Registry Dashboard

Generated: 2026-05-18 13:06:14 UTC

Total runs: 10

## Summary Table

| model_name     | formulation   | data_validity_mode   |   first_alert_lead_days |   placebo_p_value |   false_alarms_per_30_days |   alert_stability |
|:---------------|:--------------|:---------------------|------------------------:|------------------:|---------------------------:|------------------:|
| rolling_zscore | anomaly       | synthetic            |                      19 |             0.6   |                       2.61 |                 1 |
| ewma           | anomaly       | synthetic            |                      17 |             0.6   |                       2.61 |                 1 |
| rolling_zscore | anomaly       | synthetic            |                      19 |             0.6   |                       2.61 |                 1 |
| rolling_zscore | anomaly       | synthetic            |                      19 |             0.6   |                       2.61 |                 1 |
| rolling_zscore | anomaly       | synthetic            |                      19 |             0.6   |                       2.61 |                 1 |
| rolling_zscore | anomaly       | synthetic            |                      19 |             0.6   |                       2.61 |                 1 |
| rolling_zscore | anomaly       | empirical            |                      29 |             0.375 |                       2    |                 1 |
| ewma           | anomaly       | empirical            |                      29 |             0.375 |                       2    |                 1 |
| var            | forecasting   | empirical            |                      23 |             0     |                       2    |                 1 |
| rolling_zscore | anomaly       | empirical            |                     nan |           nan     |                     nan    |               nan |

## Evaluation Metrics by Run

| model_name     |   n_alerts_warning_window | auc_roc   | auc_pr   | brier_score   |
|:---------------|--------------------------:|:----------|:---------|:--------------|
| rolling_zscore |                         2 |           |          |               |
| ewma           |                         2 |           |          |               |
| rolling_zscore |                         2 |           |          |               |
| rolling_zscore |                         2 |           |          |               |
| rolling_zscore |                         2 |           |          |               |
| rolling_zscore |                         2 |           |          |               |
| rolling_zscore |                         2 |           |          |               |
| ewma           |                         2 |           |          |               |
| var            |                         2 |           |          |               |
| rolling_zscore |                         0 |           |          |               |

## Per-Model Detail

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19.0 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.474577+00:00

### ewma

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 17.0 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.496132+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19.0 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.609698+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19.0 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.721489+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19.0 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:11.375381+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19.0 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:11.871586+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** empirical
- **Features:** 11
- **First Alert Lead:** 29.0 days
- **Placebo p-value:** 0.375
- **Timestamp:** 2026-05-17T11:34:50.839376+00:00

### ewma

- **Formulation:** anomaly
- **Data Mode:** empirical
- **Features:** 11
- **First Alert Lead:** 29.0 days
- **Placebo p-value:** 0.375
- **Timestamp:** 2026-05-17T11:34:51.005366+00:00

### var

- **Formulation:** forecasting
- **Data Mode:** empirical
- **Features:** 5
- **First Alert Lead:** 23.0 days
- **Placebo p-value:** 0.0
- **Timestamp:** 2026-05-17T12:05:04.797203+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** empirical
- **Features:** 5
- **First Alert Lead:** nan days
- **Placebo p-value:** nan
- **Timestamp:** 2026-05-18T13:06:14.351648+00:00
