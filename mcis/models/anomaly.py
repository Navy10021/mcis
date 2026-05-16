from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet

from mcis.validation import assert_no_leakage


class RollingZScoreDetector:
    """Tier 0 anomaly detector using rolling z-score thresholding.

    Computes (X_t - mu_t) / sigma_t where mu and sigma are estimated
    from a backward-looking rolling window. Flags days where |z| > threshold.

    Parameters
    ----------
    window : int
        Rolling window size in days.
    threshold : float
        Z-score threshold for flagging anomalies.
    min_periods : int
        Minimum observations required before emitting a score.
    """

    def __init__(
        self,
        window: int = 30,
        threshold: float = 3.0,
        min_periods: int = 10,
    ) -> None:
        self.window = window
        self.threshold = threshold
        self.min_periods = min_periods
        self._mu: pd.Series | None = None
        self._sigma: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> RollingZScoreDetector:
        """Compute rolling mean and std on training data."""
        self._mu = X.rolling(window=self.window, min_periods=self.min_periods).mean().iloc[-1]
        self._sigma = X.rolling(window=self.window, min_periods=self.min_periods).std().iloc[-1]
        self._sigma = self._sigma.replace(0, np.nan)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute z-score anomaly scores.

        Returns DataFrame with same index as X, same columns.
        """
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Call fit() before predict()")

        mu = self._mu
        sigma = self._sigma

        scores = X.subtract(mu).divide(sigma)
        scores = scores.clip(-10, 10)
        return scores

    def predict_anomaly_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return boolean DataFrame: True where |score| > threshold."""
        scores = self.predict(X)
        return scores.abs() > self.threshold

    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).predict(X)


class EWMADetector:
    """Tier 0 anomaly detector using EWMA residual.

    Compares actual values to EWMA forecast. Flags when absolute
    standardized residual exceeds threshold.

    Parameters
    ----------
    span : int
        EWMA span (controls decay factor alpha = 2/(span+1)).
    threshold : float
        Number of MADs (median absolute deviation) above which to flag.
    """

    def __init__(self, span: int = 14, threshold: float = 3.0) -> None:
        self.span = span
        self.threshold = threshold
        self._ewma: pd.Series | None = None
        self._scale: float | None = None

    def fit(self, X: pd.DataFrame) -> EWMADetector:
        """Fit EWMA on training data."""
        self._ewma = X.ewm(span=self.span, adjust=False).mean().iloc[-1]
        residuals = X - X.ewm(span=self.span, adjust=False).mean()
        mad = residuals.abs().median()
        self._scale = mad.replace(0, np.nan).iloc[-1] if isinstance(mad, pd.DataFrame) else mad
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute standardized residuals (actual - EWMA) / scale."""
        if self._ewma is None:
            raise RuntimeError("Call fit() before predict()")
        residual = X.subtract(self._ewma)
        if self._scale is None:
            return residual
        return residual.divide(self._scale).clip(-10, 10)

    def predict_anomaly_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        scores = self.predict(X)
        return scores.abs() > self.threshold


class RobustMahalanobisDetector:
    """Tier 0 anomaly detector using robust Mahalanobis distance.

    Estimates location and covariance with Minimum Covariance Determinant (MCD),
    then computes Mahalanobis distance for each observation.

    Parameters
    ----------
    contamination : float
        Expected proportion of outliers (for EllipticEnvelope).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self._center: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> RobustMahalanobisDetector:
        valid = X.dropna()
        if len(valid) < 5:
            raise ValueError("Need at least 5 observations for MCD estimation")

        mcd = MinCovDet(random_state=self.random_state).fit(valid.values)
        self._center = mcd.location_
        self._cov = mcd.covariance_
        self._cov_inv = np.linalg.inv(self._cov)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Compute Mahalanobis distance for each row.

        Returns Series with same index as X, NaN for rows with missing values.
        """
        if self._center is None or self._cov_inv is None:
            raise RuntimeError("Call fit() before predict()")

        def _md(row: np.ndarray) -> float:
            if np.any(np.isnan(row)):
                return np.nan
            return float(np.sqrt(mahalanobis(row, self._center, self._cov_inv)))

        scores = X.apply(_md, axis=1)
        return scores

    def predict_anomaly_flags(self, X: pd.DataFrame, threshold: float | None = None) -> pd.Series:
        scores = self.predict(X)
        if threshold is None:
            chi2_cut = np.percentile(scores.dropna(), 100 * (1 - self.contamination))
            return scores > chi2_cut
        return scores > threshold


DEFAULT_ANOMALY_MODELS: dict[str, Any] = {
    "rolling_zscore": RollingZScoreDetector(window=30, threshold=3.0),
    "ewma": EWMADetector(span=14, threshold=3.0),
    "robust_mahalanobis": RobustMahalanobisDetector(contamination=0.05),
}
