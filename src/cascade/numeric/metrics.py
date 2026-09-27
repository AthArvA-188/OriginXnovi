"""Forecast and interval metrics used by the backtest. Conventions are stated here and shown on the page.

- WAPE % = 100 * sum|p - y| / sum|y|
- CV(RMSE) % = 100 * sqrt(mean((p - y)^2)) / mean(y)
- NMBE % = 100 * mean(p - y) / mean(y); positive means the model over-forecasts on average
- MASE = mean|p - y| / scale, scale = in-sample mean |y_t - y_(t-period)| over the training period
- pinball(tau) = mean(max(tau * (y - q), (tau - 1) * (y - q)))
- coverage = share of hours with q10 <= y <= q90
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _arr(x: Iterable[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def wape(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(100.0 * np.abs(p - y).sum() / np.abs(y).sum())


def mae(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(np.abs(p - y).mean())


def cvrmse(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(100.0 * np.sqrt(np.mean((p - y) ** 2)) / np.mean(y))


def nmbe(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(100.0 * np.mean(p - y) / np.mean(y))


def mase_scale(train: Iterable[float], period: int = 168) -> float:
    """In-sample mean absolute seasonal difference; NaN pairs are skipped."""
    t = _arr(train)
    if len(t) <= period:
        raise ValueError(f"need more than {period} training values for the MASE scale")
    d = np.abs(t[period:] - t[:-period])
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise ValueError("no finite seasonal differences in the training window")
    return float(d.mean())


def mase(y, p, scale: float) -> float:
    return mae(y, p) / scale


def pinball(y, q, tau: float) -> float:
    y, q = _arr(y), _arr(q)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def coverage(y, lo, hi) -> float:
    y, lo, hi = _arr(y), _arr(lo), _arr(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(_arr(hi) - _arr(lo)))
