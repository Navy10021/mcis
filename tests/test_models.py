from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mcis.models.anomaly import (
    DEFAULT_ANOMALY_MODELS,
    EWMADetector,
    RobustMahalanobisDetector,
    RollingZScoreDetector,
)
from mcis.models.evaluate import (
    compute_anomaly_metrics,
    compute_classification_metrics,
    compute_forecast_error_anomaly,
    run_placebo_dates,
)
from mcis.models.model_card import generate_model_card


class TestRollingZScoreDetector:
    @pytest.fixture
    def df(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "mean_sog": np.random.normal(10, 2, 100),
            "vessel_count": np.random.poisson(50, 100).astype(float),
        }, index=dates)

    def test_fit_predict_returns_dataframe(self, df):
        d = RollingZScoreDetector(window=10, threshold=3.0)
        scores = d.fit_predict(df)
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape == df.shape

    def test_anomaly_flags_detected(self, df):
        df = df.copy()
        df.loc[df.index[50], "mean_sog"] = 50.0
        d = RollingZScoreDetector(window=30, threshold=2.0)
        d.fit(df.iloc[:40])
        flags = d.predict_anomaly_flags(df)
        assert flags.iloc[50]["mean_sog"]

    def test_fit_before_predict_required(self, df):
        d = RollingZScoreDetector()
        with pytest.raises(RuntimeError, match="fit"):
            d.predict(df)

    def test_min_periods_respected(self, df):
        d = RollingZScoreDetector(window=100, min_periods=50)
        d.fit(df)
        scores = d.predict(df)
        assert not scores.isna().all().all()


class TestEWMADetector:
    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "mean_sog": np.random.normal(10, 1, 80),
        })

    def test_fit_predict_returns_dataframe(self, df):
        d = EWMADetector(span=14, threshold=3.0)
        d.fit(df)
        scores = d.predict(df)
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape == df.shape

    def test_anomaly_flags(self, df):
        df = df.copy()
        df.loc[df.index[60], "mean_sog"] = 30.0
        d = EWMADetector(span=7, threshold=2.5)
        d.fit(df.iloc[:40])
        flags = d.predict_anomaly_flags(df)
        assert flags.iloc[60]["mean_sog"]

    def test_fit_before_predict_required(self, df):
        d = EWMADetector()
        with pytest.raises(RuntimeError, match="fit"):
            d.predict(df)


class TestRobustMahalanobisDetector:
    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "mean_sog": np.random.normal(10, 2, 100),
            "vessel_count": np.random.poisson(50, 100).astype(float),
        })

    def test_fit_predict_returns_series(self, df):
        d = RobustMahalanobisDetector()
        d.fit(df)
        scores = d.predict(df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == 100

    def test_anomaly_flagged(self, df):
        df.loc[0, "mean_sog"] = 100.0
        d = RobustMahalanobisDetector(contamination=0.01)
        d.fit(df)
        flags = d.predict_anomaly_flags(df)
        assert flags.iloc[0]

    def test_fit_before_predict_required(self, df):
        d = RobustMahalanobisDetector()
        with pytest.raises(RuntimeError, match="fit"):
            d.predict(df)

    def test_too_few_observations(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        d = RobustMahalanobisDetector()
        with pytest.raises(ValueError, match="5"):
            d.fit(df)

    def test_nan_handling(self, df):
        df.loc[0, "mean_sog"] = np.nan
        d = RobustMahalanobisDetector()
        d.fit(df)
        scores = d.predict(df)
        assert np.isnan(scores.iloc[0])
        assert not scores.iloc[1:].isna().all()


class TestDEFAULT_ANOMALY_MODELS:
    def test_has_expected_keys(self):
        assert "rolling_zscore" in DEFAULT_ANOMALY_MODELS
        assert "ewma" in DEFAULT_ANOMALY_MODELS
        assert "robust_mahalanobis" in DEFAULT_ANOMALY_MODELS

    def test_all_are_instantiated(self):
        for name, model in DEFAULT_ANOMALY_MODELS.items():
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")


class TestComputeAnomalyMetrics:
    @pytest.fixture
    def scores_and_dates(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=120, freq="D")
        scores = pd.Series(np.random.exponential(1, 120))
        scores.iloc[100:110] = 5.0
        return scores, dates

    def test_returns_expected_keys(self, scores_and_dates):
        scores, dates = scores_and_dates
        result = compute_anomaly_metrics(scores, dates, "2022-02-24", warning_window_days=30)
        assert "first_alert_lead_days" in result
        assert "mean_alert_lead_days" in result
        assert "false_alarms_per_30_days" in result
        assert "alert_stability" in result

    def test_lead_days_nonnegative(self, scores_and_dates):
        scores, dates = scores_and_dates
        result = compute_anomaly_metrics(scores, dates, "2022-02-24", warning_window_days=30)
        lead = result.get("first_alert_lead_days")
        if lead is not None:
            assert lead >= 0

    def test_no_alerts_in_warning(self):
        dates = pd.date_range("2022-01-01", periods=60, freq="D")
        scores = pd.Series(np.zeros(60))
        result = compute_anomaly_metrics(scores, dates, "2022-02-24", threshold=3.0)
        assert result["first_alert_lead_days"] is None


class TestRunPlaceboDates:
    @pytest.fixture
    def scores_and_dates(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=120, freq="D")
        scores = pd.Series(np.random.exponential(1, 120))
        scores.iloc[50:55] = 8.0
        return scores, dates

    def test_returns_expected_keys(self, scores_and_dates):
        scores, dates = scores_and_dates
        candidates = ["2021-12-01", "2021-11-01", "2021-10-01"]
        result = run_placebo_dates(scores, dates, "2022-02-24", candidates)
        assert "placebo_p_value" in result
        assert "true_event_max_score" in result
        assert "placebo_max_scores" in result

    def test_p_value_between_0_and_1(self, scores_and_dates):
        scores, dates = scores_and_dates
        candidates = ["2021-12-01", "2021-11-01"]
        result = run_placebo_dates(scores, dates, "2022-02-24", candidates)
        assert 0 <= result["placebo_p_value"] <= 1


class TestComputeForecastErrorAnomaly:
    def test_zscore_method(self):
        y_true = pd.DataFrame({"a": [1, 2, 3, 10], "b": [2, 4, 6, 8]})
        y_pred = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 6]})
        scores = compute_forecast_error_anomaly(y_true, y_pred, method="zscore")
        assert len(scores) == 4
        assert scores.iloc[3] > scores.iloc[0]

    def test_mae_method(self):
        y_true = pd.DataFrame({"a": [1, 10], "b": [2, 8]})
        y_pred = pd.DataFrame({"a": [1, 5], "b": [2, 5]})
        scores = compute_forecast_error_anomaly(y_true, y_pred, method="mae")
        assert scores.iloc[1] > scores.iloc[0]

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            compute_forecast_error_anomaly(
                pd.DataFrame(), pd.DataFrame(), method="invalid"
            )


class TestComputeClassificationMetrics:
    def test_returns_keys(self):
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_score = pd.Series([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        result = compute_classification_metrics(y_true, y_score)
        assert "auc_roc" in result
        assert "auc_pr" in result
        assert "brier_score" in result

    def test_single_class_returns_none(self):
        y_true = pd.Series([0, 0, 0])
        y_score = pd.Series([0.1, 0.2, 0.3])
        result = compute_classification_metrics(y_true, y_score)
        assert result["auc_roc"] is None

    def test_perfect_separation(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_score = pd.Series([0.0, 0.0, 1.0, 1.0])
        result = compute_classification_metrics(y_true, y_score)
        assert result["auc_roc"] == 1.0
        assert result["auc_pr"] == 1.0

    def test_counts_included(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_score = pd.Series([0.1, 0.2, 0.8, 0.9])
        result = compute_classification_metrics(y_true, y_score)
        assert result["n_pos"] == 2
        assert result["n_neg"] == 2


class TestGenerateModelCard:
    @pytest.fixture
    def result(self, tmp_path):
        return {
            "model_name": "test_model",
            "formulation": "anomaly",
            "data_validity_mode": "synthetic",
            "train_period": ["2021-01-01", "2021-06-30"],
            "calibration_period": ["2021-07-01", "2021-08-31"],
            "evaluation_period": ["2021-09-01", "2021-12-31"],
            "feature_cols": ["mean_sog", "vessel_count"],
            "metrics": {"first_alert_lead_days": 7, "false_alarms_per_30_days": 1.5},
            "alert_dates": ["2021-12-20", "2021-12-21"],
            "first_alert_lead_days": 7,
            "placebo_p_value": 0.05,
            "caveats": ["Single-event limitation"],
        }

    def test_writes_markdown_file(self, result, tmp_path):
        path = generate_model_card(result, str(tmp_path))
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "test_model" in content
        assert "anomaly" in content
        assert "synthetic" in content
        assert "mean_sog" in content

    def test_model_name_in_filename(self, result, tmp_path):
        path = generate_model_card(result, str(tmp_path))
        assert "test_model" in path.name

    def test_creates_output_dir(self, result, tmp_path):
        nested = tmp_path / "a" / "b"
        path = generate_model_card(result, str(nested))
        assert path.exists()
