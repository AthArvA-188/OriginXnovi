"""Baselines and metrics for the numeric layer (SYNTHETIC series only; no accuracy claims)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cascade.numeric import metrics as M
from cascade.numeric.base import DataCard, causal_fill
from cascade.numeric.baselines import Naive, RobustZ, SeasonalNaive

SYN = DataCard("test series", "SYNTHETIC")


def periodic(weeks: int = 4, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=168 * weeks, freq="h")
    pattern = np.random.default_rng(seed).uniform(10, 50, 168)
    return pd.Series(np.tile(pattern, weeks), index=idx)


def test_seasonal_naive_exact_on_period_168_signal():
    y = periodic(4)
    ctx, fut = y.iloc[: 168 * 3], y.iloc[168 * 3: 168 * 3 + 24]
    f = SeasonalNaive(data=SYN).forecast(ctx, 24)
    np.testing.assert_allclose(f.q50, fut.to_numpy())
    assert list(f.index) == list(fut.index)
    assert not f.has_band and f.provenance.data.label == "SYNTHETIC"


def test_naive_uses_same_hour_yesterday():
    y = periodic(2)
    f = Naive(data=SYN).forecast(y, 24)
    np.testing.assert_allclose(f.q50, y.to_numpy()[-24:])


def test_seasonal_naive_rejects_short_context():
    with pytest.raises(ValueError):
        SeasonalNaive().forecast(periodic(1).iloc[:100], 24)


def test_metrics_match_hand_computed_values():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([12.0, 18.0, 33.0, 40.0])
    # |e| = 2, 2, 3, 0 -> sum 7; sum|y| = 100; e = 2, -2, 3, 0 -> mean 0.75; mean y = 25; mse = 17/4
    assert M.wape(y, p) == pytest.approx(7.0)
    assert M.mae(y, p) == pytest.approx(1.75)
    assert M.nmbe(y, p) == pytest.approx(100 * 0.75 / 25)
    assert M.cvrmse(y, p) == pytest.approx(100 * np.sqrt(17 / 4) / 25)
    assert M.mase(y, p, scale=3.5) == pytest.approx(0.5)
    assert M.coverage(y, p - 1, p + 1) == pytest.approx(0.25)  # only the exact hit is inside
    assert M.mean_width(p - 1, p + 1) == pytest.approx(2.0)
    # pinball at tau=0.9: d = y - q = -2, 2, -3, 0 -> max(0.9d, -0.1d) = 0.2, 1.8, 0.3, 0 -> mean 0.575
    assert M.pinball(y, p, 0.9) == pytest.approx(0.575)


def test_mase_scale_is_mean_seasonal_difference_and_skips_nan():
    t = np.r_[np.zeros(168), np.full(168, 2.0)]
    assert M.mase_scale(t, 168) == pytest.approx(2.0)
    t2 = t.copy()
    t2[200] = np.nan
    assert M.mase_scale(t2, 168) == pytest.approx(2.0)


def test_causal_fill_never_borrows_the_future():
    y = pd.Series([1.0, np.nan, np.nan, 4.0], index=pd.date_range("2024-01-01", periods=4, freq="h"))
    f = causal_fill(y)
    assert f.tolist() == [1.0, 1.0, 1.0, 4.0]
    lead_nan = pd.Series([np.nan, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="h"))
    assert np.isnan(causal_fill(lead_nan).iloc[0])  # no backfill


def test_robust_z_flags_an_injected_spike_and_does_not_score_against_itself():
    y = periodic(8, seed=1) + np.random.default_rng(2).normal(0, 0.5, 168 * 8)
    t = y.index[168 * 7 + 30]
    y_spiked = y.copy()
    y_spiked[t] += 40.0  # INJECTED spike (test only)
    r = RobustZ(threshold=5.0, data=SYN).anomaly_score(y_spiked)
    s = pd.Series(r.scores, index=r.index)
    assert s[t] > 5.0 and bool(r.frame().flag[t])
    # the scale before t is unchanged by the spike (shift(1) keeps a reading out of its own scale)
    s0 = pd.Series(RobustZ(threshold=5.0).anomaly_score(y).scores, index=y.index)
    assert s0[t] < s[t]
    # the seasonal median needs at least one weekly lag, and the scale needs 7 days of residuals, shifted one hour
    assert np.isnan(r.scores[:336]).all() and np.isfinite(r.scores[336])
    assert not r.flags[:336].any()


def test_datacard_rejects_unknown_label():
    with pytest.raises(ValueError):
        DataCard("x", "MADE_UP")
