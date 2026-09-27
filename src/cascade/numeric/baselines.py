"""Baselines every model is compared against: seasonal naive (same hour last week), naive (same hour yesterday),
and a robust z-score anomaly indicator (residual against the median of the same hour over the previous 4 weeks,
divided by 1.4826 x the trailing median absolute residual, shifted one step so a reading never scores against itself).

The 1.4826 factor (MAD_TO_SD) is the same consistency constant cascade.building.electrical.baseline_anomalies uses, so
a threshold K here and ANOMALY_MAD_K there are on the same standard-deviation-equivalent scale. The BASELINE still
differs: electrical uses a fixed hour-of-week median from the first 28 days and a pooled hour-of-day/day-type MAD;
RobustZ uses a trailing same-hour median over the previous 4 weeks and a trailing 28-day MAD."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import AnomalyResult, DataCard, ForecastResult, Provenance, causal_fill, hourly

SKLEARN_LICENCE = "BSD-3-Clause (scikit-learn)"
CODE_LICENCE = "project code (no third-party weights)"
# 1.4826 x MAD estimates the sd of normally distributed data; same value as cascade.building.electrical.MAD_TO_SD
# (kept here so this package does not need the building layer; a test checks the two agree).
MAD_TO_SD = 1.4826


def _future_index(context: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(context.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")


class SeasonalNaive:
    """Forecast = the value one period earlier (period 168 h = same hour last week). No band."""

    name = "seasonal_naive"
    licence = CODE_LICENCE

    def __init__(self, period: int = 168, data: Optional[DataCard] = None):
        self.period = int(period)
        self.data = data or DataCard("unspecified", "SYNTHETIC")

    def fit(self, y: pd.Series, X=None) -> "SeasonalNaive":  # nothing to learn
        return self

    def forecast(self, context: pd.Series, horizon: int = 24, X_future=None) -> ForecastResult:
        ctx = causal_fill(hourly(context))
        if len(ctx) < self.period:
            raise ValueError(f"context has {len(ctx)} hours; seasonal naive needs {self.period}")
        last = ctx.to_numpy()[-self.period:]
        reps = int(np.ceil(horizon / self.period))
        q50 = np.tile(last, reps)[:horizon]
        nan = np.full(horizon, np.nan)
        prov = Provenance(f"{self.name}(period={self.period})", "baseline", None, self.licence, self.data,
                          {"period": self.period})
        return ForecastResult(_future_index(ctx, horizon), nan, q50, nan.copy(), prov)

    def anomaly_score(self, y: pd.Series) -> AnomalyResult:
        return RobustZ(data=self.data).anomaly_score(y)


class Naive(SeasonalNaive):
    name = "naive"

    def __init__(self, period: int = 24, data: Optional[DataCard] = None):
        super().__init__(period=period, data=data)


class RobustZ:
    """Score = |y - median of the same hour over the previous `weeks` weeks| / (mad_to_sd x trailing median |residual|).

    The scale uses `window_days` of residuals ending one hour before the scored reading (shift(1)), so no reading
    contributes to its own scale. Units: with the default mad_to_sd = 1.4826, standard-deviation equivalents
    (1.4826 x MAD), the same scale as ANOMALY_MAD_K in cascade.building.electrical. mad_to_sd = 1 gives raw MADs."""

    name = "robust_z"
    licence = CODE_LICENCE

    def __init__(self, weeks: int = 4, window_days: int = 28, min_days: int = 7, threshold: float = 5.0,
                 data: Optional[DataCard] = None, mad_to_sd: float = MAD_TO_SD):
        self.weeks, self.window_days, self.min_days = int(weeks), int(window_days), int(min_days)
        self.threshold = float(threshold)
        self.mad_to_sd = float(mad_to_sd)
        self.data = data or DataCard("unspecified", "SYNTHETIC")

    def unit(self) -> str:
        return "raw MADs" if self.mad_to_sd == 1.0 else f"{self.mad_to_sd:g} x MAD (sd-equivalents)"

    def residual(self, y: pd.Series) -> pd.Series:
        y = hourly(y)
        lags = pd.concat([y.shift(168 * k) for k in range(1, self.weeks + 1)], axis=1)
        return y - lags.median(axis=1)

    def scores(self, y: pd.Series) -> pd.Series:
        r = self.residual(y)
        scale = r.abs().rolling(24 * self.window_days, min_periods=24 * self.min_days).median().shift(1)
        return r.abs() / (self.mad_to_sd * scale + 1e-6)

    def anomaly_score(self, y: pd.Series) -> AnomalyResult:
        s = self.scores(y)
        prov = Provenance(f"robust_z(weeks={self.weeks},window={self.window_days}d)", "baseline", None, self.licence,
                          self.data, {"weeks": self.weeks, "window_days": self.window_days,
                                      "threshold": self.threshold, "threshold_unit": self.unit(),
                                      "mad_to_sd": self.mad_to_sd})
        return AnomalyResult(s.index, s.to_numpy(), self.threshold, self.name, prov)
