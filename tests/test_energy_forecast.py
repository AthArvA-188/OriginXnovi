"""M3 garage forecaster metrics and evaluation on a synthetic two-year hourly series (unit test input only)."""

import numpy as np
import pandas as pd
import pytest

from cascade.building.energy import forecast as F


def test_cvrmse_nmbe():
    y = np.array([10.0, 12.0, 8.0, 10.0])
    p = np.array([11.0, 11.0, 9.0, 9.0])
    assert F.cvrmse(y, p) == pytest.approx(1.0 / 10.0)
    assert F.nmbe(y, p) == pytest.approx(0.0)
    assert F.nmbe(y, y - 1) == pytest.approx(0.1)


def test_eval_meter_and_night_day():
    idx = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    rng = np.random.default_rng(0)
    y = 50 + 10 * np.sin(2 * np.pi * idx.hour / 24) + 5 * (idx.dayofweek < 5) + rng.normal(0, 1, len(idx))
    s = pd.Series(y, index=idx)
    temp = pd.Series(15 + 5 * np.sin(2 * np.pi * idx.dayofyear / 365), index=idx)
    r = F.eval_meter(s, temp, max_iter=50)
    assert r["n_test_h"] > 8000 and r["n_train_h"] > 8000
    lag24 = s.shift(24)[idx.year == 2017]
    assert r["lag24"]["cvrmse"] == pytest.approx(F.cvrmse(s[idx.year == 2017].to_numpy(), lag24.to_numpy()))
    for k in F.VARIANTS:
        assert np.isfinite(r[k]["cvrmse"])
    assert r["best_naive"] in ("lag24", "lag168")
    flat = pd.Series(5.0, index=idx)
    assert F.night_day_ratio(flat) == pytest.approx(1.0)


def test_history_contexts_never_look_ahead():
    """A gap just before the 00:00 issue time must be filled from the past only (no test-day values)."""
    idx = pd.date_range("2017-01-01", periods=24 * 12, freq="h")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    d = pd.Timestamp("2017-01-10")
    s[(s.index >= d - pd.Timedelta(hours=5)) & (s.index < d)] = np.nan  # last 5 h before the issue time missing
    s[s.index >= d] = 1e6  # future values that must never appear in the context
    starts, ctxs = F.history_contexts(s, pd.DatetimeIndex([d]), context_h=48, min_h=24)
    assert starts == [d] and len(ctxs[0]) == 48
    assert ctxs[0].max() < 1e6
    last_past = s[s.index < d].dropna().iloc[-1]
    assert np.allclose(ctxs[0][-5:], last_past)  # carried forward, not interpolated towards the future


def test_forecast_module_has_no_torch_import():
    import inspect

    src = inspect.getsource(F)
    assert "import torch" not in src and "from chronos" not in src
