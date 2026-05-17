# Model Registry Dashboard

Generated: 2026-05-17 12:05:04 UTC

Total runs: 9

## Summary Table

    model_name formulation data_validity_mode  first_alert_lead_days  placebo_p_value  false_alarms_per_30_days  alert_stability
rolling_zscore     anomaly          synthetic                     19            0.600                      2.61              1.0
          ewma     anomaly          synthetic                     17            0.600                      2.61              1.0
rolling_zscore     anomaly          synthetic                     19            0.600                      2.61              1.0
rolling_zscore     anomaly          synthetic                     19            0.600                      2.61              1.0
rolling_zscore     anomaly          synthetic                     19            0.600                      2.61              1.0
rolling_zscore     anomaly          synthetic                     19            0.600                      2.61              1.0
rolling_zscore     anomaly          empirical                     29            0.375                      2.00              1.0
          ewma     anomaly          empirical                     29            0.375                      2.00              1.0
           var forecasting          empirical                     23            0.000                      2.00              1.0

## Evaluation Metrics by Run

    model_name  n_alerts_warning_window auc_roc auc_pr brier_score
rolling_zscore                        2    None   None        None
          ewma                        2    None   None        None
rolling_zscore                        2    None   None        None
rolling_zscore                        2    None   None        None
rolling_zscore                        2    None   None        None
rolling_zscore                        2    None   None        None
rolling_zscore                        2    None   None        None
          ewma                        2    None   None        None
           var                        2    None   None        None

## Per-Model Detail

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.474577+00:00

### ewma

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 17 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.496132+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.609698+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:08.721489+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:11.375381+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** synthetic
- **Features:** 3
- **First Alert Lead:** 19 days
- **Placebo p-value:** 0.6
- **Timestamp:** 2026-05-16T15:02:11.871586+00:00

### rolling_zscore

- **Formulation:** anomaly
- **Data Mode:** empirical
- **Features:** 11
- **First Alert Lead:** 29 days
- **Placebo p-value:** 0.375
- **Timestamp:** 2026-05-17T11:34:50.839376+00:00

### ewma

- **Formulation:** anomaly
- **Data Mode:** empirical
- **Features:** 11
- **First Alert Lead:** 29 days
- **Placebo p-value:** 0.375
- **Timestamp:** 2026-05-17T11:34:51.005366+00:00

### var

- **Formulation:** forecasting
- **Data Mode:** empirical
- **Features:** 5
- **First Alert Lead:** 23 days
- **Placebo p-value:** 0.0
- **Timestamp:** 2026-05-17T12:05:04.797203+00:00
