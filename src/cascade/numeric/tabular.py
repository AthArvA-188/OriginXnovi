"""scikit-learn backends (runtime env, no torch): the HGB day-ahead forecaster with a split-conformal 80% band, and
the IsolationForest / supervised HGB anomaly scorers.

The forecaster is direct day-ahead: every feature is a lag of at least 24 h plus hour and day of week, so a forecast
for any hour up to 24 h past the origin uses only readings from before the origin. The band comes from the same
fitted model: residual quantiles on a calibration window that the model never trained on.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest

from .base import AnomalyResult, DataCard, ForecastResult, Provenance, causal_fill, hourly
from .baselines import SKLEARN_LICENCE, RobustZ

LAGS = (24, 48, 72, 168, 336)
FEATURES = [f"lag{L}" for L in LAGS] + ["hour", "dow"]
ANOMALY_FEATURES = ["y_n", "lag1_n", "lag24_n", "lag168_n", "resid_n", "diff1_n", "is_zero", "hour", "dow"]


def lag_features(y: pd.Series, index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """Features at each timestamp of `index` (default: y's index). Lags come from the causally filled series."""
    y = hourly(y)
    if index is not None:
        full = y.index.union(index)
        y = y.reindex(pd.date_range(full.min(), full.max(), freq="h"))
    ctx = causal_fill(y)
    df = pd.DataFrame(index=y.index)
    for L in LAGS:
        df[f"lag{L}"] = ctx.shift(L)
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    return df if index is None else df.loc[index]


def conformal_offsets(residuals: np.ndarray, alpha: float = 0.2) -> tuple:
    """Split-conformal lower/upper residual offsets for a (1 - alpha) two-sided band, finite-sample corrected:
    lower = the floor((n+1)*alpha/2)-th smallest residual, upper = the ceil((n+1)*(1-alpha/2))-th smallest."""
    r = np.sort(np.asarray(residuals, dtype=float)[np.isfinite(residuals)])
    n = len(r)
    if n == 0:
        raise ValueError("no calibration residuals")
    k_lo = math.floor((n + 1) * alpha / 2)
    k_hi = math.ceil((n + 1) * (1 - alpha / 2))
    lo = r[k_lo - 1] if k_lo >= 1 else -np.inf
    hi = r[k_hi - 1] if k_hi <= n else np.inf
    return float(lo), float(hi)


class HGBForecaster:
    """HistGradientBoostingRegressor on lags >= 24 h + hour + day of week, with a split-conformal band."""

    name = "hgb_conformal"
    licence = SKLEARN_LICENCE

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.05, random_state: int = 0, alpha: float = 0.2,
                 data: Optional[DataCard] = None):
        self.params = {"max_iter": max_iter, "learning_rate": learning_rate, "random_state": random_state}
        self.alpha = float(alpha)
        self.data = data or DataCard("unspecified", "SYNTHETIC")
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.lo: Optional[float] = None
        self.hi: Optional[float] = None
        self.fit_info: Dict[str, object] = {}

    def fit(self, y: pd.Series, train_end: Optional[pd.Timestamp] = None, calib_end: Optional[pd.Timestamp] = None,
            X=None) -> "HGBForecaster":
        """Train on [start, train_end); calibrate the band on [train_end, calib_end). Without dates the last 20% of
        the series is used for calibration. With calib_end == train_end there is no band (point model only)."""
        y = hourly(y)
        feats = lag_features(y)
        if train_end is None:
            cut = y.index[int(len(y) * 0.8)]
            train_end, calib_end = cut, y.index[-1] + pd.Timedelta(hours=1)
        train_end = pd.Timestamp(train_end)
        calib_end = pd.Timestamp(calib_end) if calib_end is not None else train_end
        tr = (y.index < train_end) & y.notna().to_numpy() & feats[FEATURES].notna().all(axis=1).to_numpy()
        self.model = HistGradientBoostingRegressor(**self.params).fit(feats.loc[tr, FEATURES], y[tr])
        cal = (y.index >= train_end) & (y.index < calib_end) & y.notna().to_numpy()
        if cal.sum() > 0:
            r = y[cal].to_numpy() - self.model.predict(feats.loc[cal, FEATURES])
            self.lo, self.hi = conformal_offsets(r, self.alpha)
        else:
            self.lo = self.hi = None
        self.fit_info = {"n_train": int(tr.sum()), "n_calib": int(cal.sum()), "train_end": str(train_end),
                         "calib_end": str(calib_end), "band_lo": self.lo, "band_hi": self.hi}
        return self

    def predict_at(self, y: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
        """q10/q50/q90 at `index` from features built on y. Every lag is >= 24 h, so values of y at or after
        index.min() + 0 h are never used for hours up to 24 h past the origin."""
        if self.model is None:
            raise RuntimeError("fit() first")
        feats = lag_features(y, index)
        p = self.model.predict(feats[FEATURES])
        lo = p + self.lo if self.lo is not None else np.full(len(p), np.nan)
        hi = p + self.hi if self.hi is not None else np.full(len(p), np.nan)
        return pd.DataFrame({"q10": lo, "q50": p, "q90": hi}, index=index)

    def provenance(self) -> Provenance:
        return Provenance(self.name, "sklearn", None, self.licence, self.data,
                          {**self.params, "alpha": self.alpha, "lags": list(LAGS), **self.fit_info})

    def forecast(self, context: pd.Series, horizon: int = 24, X_future=None) -> ForecastResult:
        if horizon > min(LAGS):
            raise ValueError(f"direct day-ahead model: horizon must be <= {min(LAGS)} h")
        ctx = hourly(context)
        idx = pd.date_range(ctx.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
        f = self.predict_at(ctx, idx)
        return ForecastResult(idx, f.q10.to_numpy(), f.q50.to_numpy(), f.q90.to_numpy(), self.provenance())


def anomaly_features(y: pd.Series, window_days: int = 28) -> pd.DataFrame:
    """Per-series features normalised by a trailing median and IQR (shifted one step), plus the robust z score."""
    y = hourly(y)
    w = 24 * window_days
    roll = y.rolling(w, min_periods=24)
    med = roll.median().shift(1)
    iqr = (roll.quantile(0.75) - roll.quantile(0.25)).shift(1) + 1e-6
    rz = RobustZ(window_days=window_days)
    resid = rz.residual(y)
    df = pd.DataFrame(index=y.index)
    df["y"] = y
    df["y_n"] = (y - med) / iqr
    df["lag1_n"] = (y.shift(1) - med) / iqr
    df["lag24_n"] = (y.shift(24) - med) / iqr
    df["lag168_n"] = (y.shift(168) - med) / iqr
    df["resid_n"] = resid / iqr
    df["diff1_n"] = y.diff() / iqr
    df["is_zero"] = (y == 0).astype(float)
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["robust_z"] = rz.scores(y)
    return df.replace([np.inf, -np.inf], np.nan)


class IsoForestScorer:
    name = "isolation_forest"
    licence = SKLEARN_LICENCE

    def __init__(self, n_estimators: int = 200, max_samples: int = 4096, random_state: int = 0,
                 data: Optional[DataCard] = None):
        self.params = {"n_estimators": n_estimators, "max_samples": max_samples, "random_state": random_state}
        self.data = data or DataCard("unspecified", "SYNTHETIC")
        self.model: Optional[IsolationForest] = None

    def fit(self, X: pd.DataFrame, y=None) -> "IsoForestScorer":
        self.model = IsolationForest(**self.params, n_jobs=1).fit(X[ANOMALY_FEATURES].fillna(0).to_numpy())
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X[ANOMALY_FEATURES].fillna(0).to_numpy())

    def anomaly_score(self, y: pd.Series, threshold: Optional[float] = None) -> AnomalyResult:
        X = anomaly_features(y)
        if self.model is None:
            self.fit(X)
        s = self.score(X)
        thr = float(np.quantile(s, 0.99)) if threshold is None else float(threshold)
        prov = Provenance(self.name, "sklearn", None, self.licence, self.data,
                          {**self.params, "threshold": "top 1% of this series" if threshold is None else thr})
        return AnomalyResult(X.index, s, thr, self.name, prov)


class HGBAnomalyClassifier:
    """Supervised: needs human anomaly labels from other series (split by building, never by row)."""

    name = "hgb_supervised"
    licence = SKLEARN_LICENCE

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.05, random_state: int = 0,
                 data: Optional[DataCard] = None):
        self.params = {"max_iter": max_iter, "learning_rate": learning_rate, "random_state": random_state}
        self.data = data or DataCard("unspecified", "SYNTHETIC")
        self.model: Optional[HistGradientBoostingClassifier] = None

    def fit(self, X: pd.DataFrame, labels: Sequence[int]) -> "HGBAnomalyClassifier":
        self.model = HistGradientBoostingClassifier(**self.params).fit(X[ANOMALY_FEATURES], np.asarray(labels))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[ANOMALY_FEATURES])[:, 1]

    def anomaly_score(self, y: pd.Series, threshold: float = 0.5) -> AnomalyResult:
        if self.model is None:
            raise RuntimeError("fit() on labelled series first")
        X = anomaly_features(y)
        prov = Provenance(self.name, "sklearn", None, self.licence, self.data, {**self.params, "threshold": threshold})
        return AnomalyResult(X.index, self.score(X), float(threshold), self.name, prov)
