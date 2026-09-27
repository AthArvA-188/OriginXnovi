"""scikit-learn backends: HGB day-ahead forecaster (no leakage, conformal band) and the anomaly scorers.
All series here are SYNTHETIC (tests/fixtures/numeric/synthetic_meter.csv or generated in the test)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.numeric.backtest import BacktestConfig, baseline_frame, hgb_frame, score
from cascade.numeric.base import DataCard
from cascade.numeric.tabular import (ANOMALY_FEATURES, HGBAnomalyClassifier, HGBForecaster, IsoForestScorer,
                                     anomaly_features, conformal_offsets, lag_features)

FIX = Path(__file__).parent / "fixtures" / "numeric" / "synthetic_meter.csv"
SYN = DataCard("synthetic_meter.csv", "SYNTHETIC")


@pytest.fixture(scope="module")
def meter() -> pd.Series:
    df = pd.read_csv(FIX, parse_dates=["timestamp"])
    assert (df.label == "SYNTHETIC").all()
    return df.set_index("timestamp").kwh.astype(float)


@pytest.fixture(scope="module")
def fitted(meter):
    return HGBForecaster(max_iter=60, data=SYN).fit(meter, meter.index[24 * 7 * 5], meter.index[24 * 7 * 7])


def test_features_are_lags_of_at_least_24h(meter):
    f = lag_features(meter)
    t = meter.index[500]
    assert f.loc[t, "lag24"] == meter.iloc[500 - 24]
    assert f.loc[t, "lag336"] == meter.iloc[500 - 336]


def test_forecast_does_not_use_values_at_or_after_the_origin(meter, fitted):
    origin = meter.index[24 * 7 * 7]
    ctx = meter[meter.index < origin]
    a = fitted.forecast(ctx, 24)
    # appending wild "future" values to the series must not change forecasts for the next 24 h
    fut = pd.Series(1e6, index=pd.date_range(origin, periods=48, freq="h"))
    b = fitted.predict_at(pd.concat([ctx, fut]), a.index)
    np.testing.assert_allclose(a.q50, b.q50.to_numpy())
    assert a.has_band and (a.q10 <= a.q90).all()
    assert a.provenance.data.label == "SYNTHETIC" and a.provenance.params["n_calib"] > 0


def test_forecast_horizon_is_capped_at_the_shortest_lag(meter, fitted):
    with pytest.raises(ValueError):
        fitted.forecast(meter, 25)


def test_conformal_offsets_finite_sample_rule():
    r = np.arange(1, 10, dtype=float)  # n = 9, alpha 0.2: k_lo = floor(1.0) = 1, k_hi = ceil(9.0) = 9
    assert conformal_offsets(r, 0.2) == (1.0, 9.0)
    lo, hi = conformal_offsets(np.arange(1, 5, dtype=float), 0.2)  # n = 4: k_lo 0 -> -inf; k_hi 5 > n -> +inf
    assert lo == -np.inf and hi == np.inf


def test_conformal_band_covers_about_80pct_on_iid_noise():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=2000), rng.normal(size=20000)
    lo, hi = conformal_offsets(cal, 0.2)
    cov = np.mean((test >= lo) & (test <= hi))
    assert 0.77 <= cov <= 0.83


def test_backtest_frames_and_scores_on_synthetic_meter(meter):
    cfg = BacktestConfig(test_start="2024-02-19", last_origin="2024-02-24", hgb_train_end="2024-02-05",
                         hgb_calib_end="2024-02-19")
    bf = baseline_frame(meter, "m1", cfg)
    hf, models = hgb_frame(meter, "m1", cfg, max_iter=60, point_only_refit=True)
    fr = bf.merge(hf, on=["meter", "ts"])
    assert len(fr) == 6 * 24 and {"hgb_q10", "hgb_q50", "hgb_q90", "hgb_oct_q50"} <= set(fr.columns)
    res = score(fr, {"m1": 2.0}, n_boot=50)
    assert res["n_scored_hours"] == 144 and res["baseline"] == "snaive168"
    assert res["models"]["hgb"]["coverage_80"] is not None and res["models"]["snaive168"]["coverage_80"] is None
    lo, hi = res["models"]["hgb"]["WAPE_pct_ci95"]
    assert lo <= res["models"]["hgb"]["WAPE_pct"] <= hi or lo == hi


def test_anomaly_scorers_rank_an_injected_fault_high(meter):
    y = meter.copy()
    rng = np.random.default_rng(3)
    idx = rng.choice(np.arange(24 * 7 * 5, len(y)), size=10, replace=False)
    y.iloc[idx] += 80.0  # INJECTED faults (test only)
    X = anomaly_features(y).dropna(subset=["y"])
    labels = X.index.isin(y.index[idx]).astype(int)
    iso = IsoForestScorer(n_estimators=50, max_samples=256, data=SYN).fit(X)
    s = iso.score(X)
    assert np.median(s[labels == 1]) > np.quantile(s, 0.9)
    clf = HGBAnomalyClassifier(max_iter=30, data=SYN).fit(X, labels)
    p = clf.score(X)
    assert p.shape == (len(X),) and set(ANOMALY_FEATURES) <= set(X.columns)
    assert np.median(p[labels == 1]) > np.median(p[labels == 0])
