from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mcis.models.forecasting import (
    DEFAULT_FORECASTING_MODELS,
    LSTMForecaster,
    TCNForecaster,
    VARForecaster,
)


@pytest.fixture
def sample_ts() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "vessel_count": np.random.poisson(50, n).cumsum(),
        "mean_sog": np.random.uniform(8, 14, n).cumsum() * 0.1,
        "cargo_fraction": np.clip(np.random.beta(5, 5, n) * 100, 0, 100),
    })


class TestVARForecaster:
    def test_fit_and_predict(self, sample_ts):
        forecaster = VARForecaster(maxlags=5)
        train = sample_ts.iloc[:150]
        forecaster.fit(train)
        pred = forecaster.predict(train, horizon=10)
        assert isinstance(pred, pd.DataFrame)
        assert pred.shape == (10, 3)
        assert list(pred.columns) == ["vessel_count", "mean_sog", "cargo_fraction"]

    def test_fit_returns_self(self, sample_ts):
        forecaster = VARForecaster(maxlags=5)
        result = forecaster.fit(sample_ts.iloc[:100])
        assert result is forecaster

    def test_predict_without_fit_raises(self):
        forecaster = VARForecaster(maxlags=5)
        with pytest.raises(RuntimeError, match="Call fit"):
            forecaster.predict(pd.DataFrame({"a": [1, 2, 3]}))

    def test_get_residuals(self, sample_ts):
        forecaster = VARForecaster(maxlags=5)
        forecaster.fit(sample_ts.iloc[:100])
        resid = forecaster.get_residuals()
        assert resid is not None
        assert isinstance(resid, pd.DataFrame)
        assert list(resid.columns) == ["vessel_count", "mean_sog", "cargo_fraction"]

    def test_get_residuals_before_fit(self):
        forecaster = VARForecaster(maxlags=5)
        assert forecaster.get_residuals() is None

    def test_insufficient_data_raises(self):
        forecaster = VARForecaster(maxlags=10)
        tiny = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least maxlags"):
            forecaster.fit(tiny)

    def test_default_maxlags(self):
        forecaster = VARForecaster()
        assert forecaster.maxlags == 14
        assert forecaster.trend == "c"

    def test_in_registry(self):
        assert "var" in DEFAULT_FORECASTING_MODELS
        assert isinstance(DEFAULT_FORECASTING_MODELS["var"], VARForecaster)

    def test_different_horizons(self, sample_ts):
        forecaster = VARForecaster(maxlags=5)
        forecaster.fit(sample_ts.iloc[:100])
        for h in [1, 5, 20]:
            pred = forecaster.predict(sample_ts.iloc[:100], horizon=h)
            assert pred.shape[0] == h

    def test_predict_output_values(self, sample_ts):
        forecaster = VARForecaster(maxlags=5)
        train = sample_ts.iloc[:100]
        forecaster.fit(train)
        pred = forecaster.predict(train, horizon=5)
        assert not pred.isna().any().any()
        assert (pred > -1e6).all().all()
        assert (pred < 1e6).all().all()

    def test_predict_on_different_length_input(self, sample_ts):
        forecaster = VARForecaster(maxlags=5)
        forecaster.fit(sample_ts.iloc[:100])
        pred = forecaster.predict(sample_ts.iloc[:80], horizon=3)
        assert pred.shape == (3, 3)


class TestLSTMForecaster:
    def test_import_error_if_no_torch(self):
        forecaster = LSTMForecaster(input_size=3, epochs=1)
        with pytest.raises(ImportError, match="PyTorch"):
            forecaster.fit(pd.DataFrame({"a": [1, 2, 3]}))

    def test_default_constructor(self):
        forecaster = LSTMForecaster()
        assert forecaster.hidden_size == 64
        assert forecaster.num_layers == 2
        assert forecaster.seq_len == 30


class TestTCNForecaster:
    def test_import_error_if_no_torch(self):
        forecaster = TCNForecaster(input_size=3, epochs=1)
        with pytest.raises(ImportError, match="PyTorch"):
            forecaster.fit(pd.DataFrame({"a": [1, 2, 3]}))

    def test_default_constructor(self):
        forecaster = TCNForecaster()
        assert forecaster.channels == [64, 64, 64]
        assert forecaster.kernel_size == 3
        assert forecaster.seq_len == 30

    def test_build_tcn_channels(self):
        tcn = TCNForecaster(input_size=3, channels=[32, 32], epochs=1)
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not available")
        model = tcn._build_tcn(nn, 3)
        assert model is not None


class TestDEFAULT_FORECASTING_MODELS:
    def test_has_var(self):
        assert "var" in DEFAULT_FORECASTING_MODELS

    def test_all_are_instantiated(self):
        for name, model in DEFAULT_FORECASTING_MODELS.items():
            assert model is not None, f"{name} is None"
