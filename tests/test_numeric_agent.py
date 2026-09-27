"""NumericAgent: backend selection and fallback, precomputed-forecast pinning, indicator-only Observations.
SYNTHETIC fixture only; the fake precomputed parquet is written to tmp_path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.numeric import HF_MODELS, NumericAgent
from cascade.numeric.base import DataCard

FIX = Path(__file__).parent / "fixtures" / "numeric" / "synthetic_meter.csv"
SYN = DataCard("synthetic_meter.csv", "SYNTHETIC")


@pytest.fixture(scope="module")
def meter() -> pd.Series:
    df = pd.read_csv(FIX, parse_dates=["timestamp"])
    return df.set_index("timestamp").kwh.astype(float)


def write_fake_chronos(d: Path, meter: pd.Series, revision: str) -> pd.Timestamp:
    spec = HF_MODELS["chronos-2"]
    origin = meter.index[-24]
    ts = pd.date_range(origin, periods=24, freq="h")
    pd.DataFrame({"meter": "m1", "origin": origin, "ts": ts, "h": np.arange(1, 25), "q10": 1.0, "q50": 2.0,
                  "q90": 3.0}).to_parquet(d / f"forecasts_{spec.tag}.parquet", index=False)
    (d / f"forecasts_{spec.tag}.provenance.json").write_text(json.dumps(
        {"model_id": spec.model_id, "revision": revision, "context": 1024, "horizon": 24}), encoding="utf-8")
    return origin


def test_falls_back_to_seasonal_naive_without_artifacts(tmp_path, meter):
    agent = NumericAgent(tmp_path, tmp_path, data=SYN)
    assert agent.select("forecast", "m1") == "seasonal_naive"
    assert agent.available() == ["hgb", "seasonal_naive"]
    f = agent.forecast(meter.iloc[:-24], 24, series_id="m1")
    assert f.provenance.backend == "baseline"


def test_uses_precomputed_chronos_only_when_revision_pin_matches(tmp_path, meter):
    origin = write_fake_chronos(tmp_path, meter, HF_MODELS["chronos-2"].revision)
    agent = NumericAgent(tmp_path, tmp_path, data=SYN)
    assert agent.select("forecast", "m1") == "chronos-2"
    f = agent.forecast(meter[meter.index < origin], 24, series_id="m1")
    assert f.provenance.model_id == "amazon/chronos-2" and f.provenance.revision == HF_MODELS["chronos-2"].revision
    assert f.provenance.licence == "Apache-2.0"
    np.testing.assert_allclose(f.q50, 2.0)
    # a series the file does not cover goes to the next backend
    assert agent.select("forecast", "other") == "seasonal_naive"


def test_revision_mismatch_falls_back(tmp_path, meter):
    origin = write_fake_chronos(tmp_path, meter, "0" * 40)
    agent = NumericAgent(tmp_path, tmp_path, data=SYN)
    assert not agent.precomputed_ok("chronos-2")
    assert agent.select("forecast", "m1") == "seasonal_naive"
    f = agent.forecast(meter[meter.index < origin], 24, backend="chronos-2", series_id="m1")
    assert f.provenance.backend == "baseline"


def test_fit_then_forecast_with_hgb(tmp_path, meter):
    agent = NumericAgent(tmp_path, tmp_path, data=SYN)
    prov = agent.fit(meter.iloc[:-24], backend="hgb", series_id="m1", max_iter=40)
    assert prov.backend == "sklearn" and prov.data.label == "SYNTHETIC"
    assert agent.select("forecast", "m1") == "hgb"
    f = agent.forecast(meter.iloc[:-24], 24, series_id="m1")
    assert len(f.q50) == 24 and f.has_band
    with pytest.raises(ValueError):
        agent.fit(meter, backend="chronos-2")


def test_anomaly_observations_are_indicators_with_provenance(tmp_path, meter):
    y = meter.copy()
    t = y.index[24 * 7 * 6 + 10]
    y[t] += 100.0  # INJECTED spike (test only)
    agent = NumericAgent(tmp_path, tmp_path, data=SYN)
    r = agent.anomaly_score(y)
    assert r.method == "robust_z" and r.flags[list(r.index).index(t)]
    obs = agent.to_observations(r, source="C-F01-1-A", element_id="C-F01-1", kind="load_event")
    assert obs and all(o.level is None for o in obs)
    assert all(o.synthetic for o in obs)  # SYNTHETIC data card -> synthetic observations
    o = [o for o in obs if o.ts == t.strftime("%Y-%m-%dT%H:%M:%SZ")][0]
    assert "robust_z" in o.text and "human review" in o.text and "SYNTHETIC" in o.text
    assert o.source == "numeric:C-F01-1-A"


def test_unknown_methods_raise(tmp_path, meter):
    agent = NumericAgent(tmp_path, tmp_path, data=SYN)
    with pytest.raises(ValueError):
        agent.anomaly_score(meter, method="nope")
    with pytest.raises(ValueError):
        agent.anomaly_score(meter, method="hgb_supervised")
    with pytest.raises(ValueError):
        agent.forecast(meter, 24, backend="nope")


def test_select_without_series_id_never_picks_a_backend_it_cannot_serve(tmp_path, meter):
    write_fake_chronos(tmp_path, meter, HF_MODELS["chronos-2"].revision)
    agent = NumericAgent(tmp_path, tmp_path, data=SYN)
    assert agent.select("forecast") == "seasonal_naive"  # no stored HGB and no id to look up
    f = agent.forecast(meter.iloc[:-24], 24)
    assert f.provenance.backend == "baseline"
    agent.fit(meter.iloc[:-24], backend="hgb", max_iter=30)  # default id "series"
    assert agent.select("forecast") == "hgb"
    assert agent.forecast(meter.iloc[:-24], 24).provenance.backend == "sklearn"
