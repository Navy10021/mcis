from __future__ import annotations

from typing import Any
import os

import numpy as np
import pandas as pd


class VARForecaster:
    """Tier 1 forecasting model using Vector Autoregression (VAR).

    Wraps statsmodels VAR, following the sklearn-style fit/predict pattern.
    The predict method returns multi-step forecasts suitable for feeding
    into compute_forecast_error_anomaly.

    Parameters
    ----------
    maxlags : int
        Maximum number of lags for the VAR model.
    trend : str
        Trend specification ('c' for constant, 'n' for none, 'ct' for
        constant + linear trend, 'ctt' for constant + quadratic trend).
    """

    def __init__(self, maxlags: int = 14, trend: str = "c") -> None:
        self.maxlags = maxlags
        self.trend = trend
        self._fitted: Any = None
        self._final_lags: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> VARForecaster:
        from statsmodels.tsa.api import VAR

        if len(X) < self.maxlags + 5:
            raise ValueError(
                f"VAR requires at least maxlags+5 ({self.maxlags + 5}) "
                f"observations, got {len(X)}"
            )

        model = VAR(X, freq=pd.infer_freq(X.index) if hasattr(X, "index") and X.index.dtype.kind == "M" else None)
        results = model.fit(maxlags=self.maxlags, trend=self.trend, ic="aic")
        self._fitted = results
        self._final_lags = results.k_ar
        return self

    def predict(self, X: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        if self._fitted is None:
            raise RuntimeError("Call fit() before predict()")

        if self._final_lags is None:
            raise RuntimeError("Model not fitted")

        if self._final_lags == 0:
            cols = X.columns.tolist()
            index = pd.RangeIndex(start=len(X), stop=len(X) + horizon)
            if hasattr(self._fitted, "intercept") and self._fitted.intercept is not None:
                trend = self._fitted.intercept
            else:
                trend = self._fitted.params[0] if self._fitted.params.shape[0] == 1 else np.zeros(len(cols))
            return pd.DataFrame(
                np.tile(trend, (horizon, 1)),
                index=index, columns=cols,
            )

        if len(X) < self._final_lags:
            raise ValueError(
                f"Need at least {self._final_lags} lags for prediction, "
                f"got {len(X)} rows"
            )

        last_obs = X.iloc[-self._final_lags:]
        raw = self._fitted.forecast(y=last_obs.values, steps=horizon)
        cols = X.columns
        index = pd.RangeIndex(start=len(X), stop=len(X) + horizon)
        return pd.DataFrame(raw, index=index, columns=cols)

    def get_residuals(self) -> pd.DataFrame | None:
        if self._fitted is None:
            return None
        return pd.DataFrame(self._fitted.resid)


class LSTMForecaster:
    """Tier 1 forecasting model using LSTM neural network.

    Wraps PyTorch nn.LSTM in a sklearn-style interface. Requires
    torch >= 2.1.0 (optional dependency).

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of hidden units per LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    seq_len : int
        Lookback window length.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_len: int = 30,
        epochs: int = 50,
        lr: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self._model: Any = None
        self._fitted_cols: list[str] | None = None

    def _import_torch(self) -> tuple[Any, Any, Any]:
        if os.getenv("MCIS_ENABLE_TORCH", "0") != "1":
            raise ImportError(
                "LSTMForecaster requires PyTorch and is disabled by default. "
                "Set MCIS_ENABLE_TORCH=1 after installing torch>=2.1.0."
            )
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "LSTMForecaster requires PyTorch. "
                "Install it with: pip install torch>=2.1.0"
            )
        return torch, nn, optim

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> LSTMForecaster:
        torch, nn, optim = self._import_torch()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if self.input_size is None:
            self.input_size = X.shape[1]

        self._fitted_cols = X.columns.tolist()

        sequences, targets = self._create_sequences(X.values)
        if len(sequences) < 2:
            raise ValueError(
                f"Need at least 2 sequences (got {len(sequences)}). "
                f"Try smaller seq_len or more data."
            )

        class _LSTMModel(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
                self.regressor = nn.Linear(hidden_dim, input_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                return self.regressor(out[:, -1, :])

        model = _LSTMModel(self.input_size, self.hidden_size, self.num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        seq_t = torch.tensor(sequences, dtype=torch.float32)
        target_t = torch.tensor(targets, dtype=torch.float32)

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = model(seq_t)
            loss = criterion(pred, target_t)
            loss.backward()
            optimizer.step()

        self._model = model
        return self

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(data) - self.seq_len):
            X_seq.append(data[i : i + self.seq_len])
            y_seq.append(data[i + self.seq_len])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, X: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")

        torch, _, _ = self._import_torch()

        last_seq = X.values[-self.seq_len:]
        if len(last_seq) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} rows for prediction, "
                f"got {len(X)}"
            )

        last_seq_t = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)

        predictions = []
        current_seq = last_seq_t

        self._model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                pred = self._model(current_seq)
                predictions.append(pred.squeeze(0).numpy())
                new_seq = torch.cat(
                    [current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1
                )
                current_seq = new_seq

        cols = self._fitted_cols if self._fitted_cols else X.columns.tolist()
        index = pd.RangeIndex(start=len(X), stop=len(X) + horizon)
        return pd.DataFrame(np.array(predictions), index=index, columns=cols)


class TCNForecaster:
    """Tier 1 forecasting model using Temporal Convolutional Network (TCN).

    Uses causal dilated convolutions for sequence forecasting.
    Requires torch >= 2.1.0 (optional dependency).

    Parameters
    ----------
    input_size : int
        Number of input features.
    channels : list[int]
        Channel sizes for each convolutional layer.
    kernel_size : int
        Kernel size for causal convolutions.
    seq_len : int
        Lookback window length.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    dropout : float
        Dropout rate.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_size: int | None = None,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        seq_len: int = 30,
        epochs: int = 50,
        lr: float = 0.001,
        dropout: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.channels = channels or [64, 64, 64]
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.random_state = random_state
        self._model: Any = None
        self._fitted_cols: list[str] | None = None

    def _import_torch(self) -> tuple[Any, Any, Any]:
        if os.getenv("MCIS_ENABLE_TORCH", "0") != "1":
            raise ImportError(
                "TCNForecaster requires PyTorch and is disabled by default. "
                "Set MCIS_ENABLE_TORCH=1 after installing torch>=2.1.0."
            )
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "TCNForecaster requires PyTorch. "
                "Install it with: pip install torch>=2.1.0"
            )
        return torch, nn, optim

    def _build_tcn(self, nn: Any, input_dim: int) -> Any:
        class _CausalConv1d(nn.Module):
            def __init__(self, in_c: int, out_c: int, k: int, dilation: int):
                super().__init__()
                self.pad = (k - 1) * dilation
                self.conv = nn.Conv1d(
                    in_c, out_c, k,
                    padding=self.pad,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.conv(x)[:, :, :-self.pad] if self.pad > 0 else self.conv(x)

        class _TCNBlock(nn.Module):
            def __init__(self, in_c: int, out_c: int, k: int, dilation: int, drop: float):
                super().__init__()
                self.conv1 = _CausalConv1d(in_c, out_c, k, dilation)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(drop)
                self.resample = (
                    nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
                )

            def forward(self, x):
                residual = self.resample(x)
                out = self.dropout(self.relu(self.conv1(x)))
                return self.relu(out + residual)

        class _TCNModel(nn.Module):
            def __init__(self, in_dim: int, chans: list[int], k: int, drop: float):
                super().__init__()
                layers = []
                prev = in_dim
                for i, c in enumerate(chans):
                    layers.append(_TCNBlock(prev, c, k, dilation=2**i, drop=drop))
                    prev = c
                self.tcn = nn.Sequential(*layers)
                self.linear = nn.Linear(prev, in_dim)

            def forward(self, x):
                x_t = x.transpose(1, 2)
                out = self.tcn(x_t)
                out = out.transpose(1, 2)
                return self.linear(out[:, -1, :])

        return _TCNModel(input_dim, self.channels, self.kernel_size, self.dropout)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> TCNForecaster:
        torch, nn, optim = self._import_torch()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if self.input_size is None:
            self.input_size = X.shape[1]

        self._fitted_cols = X.columns.tolist()

        sequences, targets = self._create_sequences(X.values)
        if len(sequences) < 2:
            raise ValueError(
                f"Need at least 2 sequences (got {len(sequences)}). "
                f"Try smaller seq_len or more data."
            )

        model = self._build_tcn(nn, self.input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        seq_t = torch.tensor(sequences, dtype=torch.float32)
        target_t = torch.tensor(targets, dtype=torch.float32)

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = model(seq_t)
            loss = criterion(pred, target_t)
            loss.backward()
            optimizer.step()

        self._model = model
        return self

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(data) - self.seq_len):
            X_seq.append(data[i : i + self.seq_len])
            y_seq.append(data[i + self.seq_len])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, X: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")

        torch, _, _ = self._import_torch()

        last_seq = X.values[-self.seq_len:]
        if len(last_seq) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} rows for prediction, "
                f"got {len(X)}"
            )

        predictions = []
        current_seq = last_seq.copy()

        self._model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                seq_t = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
                pred = self._model(seq_t)
                pred_np = pred.squeeze(0).numpy()
                predictions.append(pred_np)
                current_seq = np.vstack([current_seq[1:], pred_np])

        cols = self._fitted_cols if self._fitted_cols else X.columns.tolist()
        index = pd.RangeIndex(start=len(X), stop=len(X) + horizon)
        return pd.DataFrame(np.array(predictions), index=index, columns=cols)


DEFAULT_FORECASTING_MODELS: dict[str, Any] = {
    "var": VARForecaster(maxlags=14),
}
