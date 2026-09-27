"""Interior NDT strength estimator and field-reading schema.

Numbers generated inside these tests are SYNTHETIC-FOR-TEST (they check arithmetic, not accuracy).
The measured accuracy lives in eval/interior/ndt_strength_v1.json (REAL data).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.interior.ndt import (PowerLaw, StrengthModel, calibration_shift, conformal_q, metrics)
from cascade.interior.readings import (FieldReading, TEMPLATE_PATH, COLUMNS, grade_moisture, pair_strength_locations,
                                       parse_csv, route)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "interior"


def _synthetic(n=400, seed=0, a=-8.0, b=1.1, c=0.9, noise=0.0):
    rng = np.random.default_rng(seed)
    rn = rng.uniform(20, 55, n)
    vp = rng.uniform(3000, 5000, n)
    fc = np.exp(a + b * np.log(rn) + c * np.log(vp) + rng.normal(0, noise, n))
    return pd.DataFrame({"rn": rn, "vp": vp, "fc": fc})


def test_power_law_recovers_known_coefficients():
    df = _synthetic(noise=0.0)
    law = PowerLaw(("rn", "vp")).fit(df)
    assert law.intercept == pytest.approx(-8.0, abs=1e-6)
    assert law.coef[0] == pytest.approx(1.1, abs=1e-6)
    assert law.coef[1] == pytest.approx(0.9, abs=1e-6)
    np.testing.assert_allclose(law.predict(df), df.fc, rtol=1e-8)
    back = PowerLaw.from_dict(law.as_dict())
    np.testing.assert_allclose(back.predict(df), df.fc, rtol=1e-8)


def test_calibration_shift_is_mean_log_residual():
    lp = np.log([20.0, 30.0, 40.0])
    fc = [22.0, 33.0, 44.0]  # every core 10 % above prediction
    assert calibration_shift(lp, fc) == pytest.approx(math.log(1.1))
    assert calibration_shift([], []) == 0.0
    with pytest.raises(ValueError):
        calibration_shift([1.0], [0.0])


def test_conformal_quantile_rank_and_disjoint_coverage():
    r = np.arange(1, 11, dtype=float)  # n=10 -> ceil(11*0.9)=10th smallest
    assert conformal_q(r, 0.9) == 10.0
    assert conformal_q(np.arange(1, 101, dtype=float), 0.9) == 91.0
    # quantile from one sample, coverage measured on a DISJOINT sample of the same distribution
    rng = np.random.default_rng(1)
    q = conformal_q(np.abs(rng.normal(0, 1, 2000)), 0.9)
    held = np.abs(rng.normal(0, 1, 20000))
    assert 0.88 <= float(np.mean(held <= q)) <= 0.92


def test_metrics_basic():
    m = metrics(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert m["mae"] == 2.0 and m["n"] == 2
    assert m["mape_pct"] == pytest.approx(15.0)


def _fixture_spec():
    df = _synthetic(noise=0.0)
    law = PowerLaw(("rn", "vp")).fit(df)
    rn = PowerLaw(("rn",)).fit(df)
    vp = PowerLaw(("vp",)).fit(df)
    mk = lambda l: {"law": l.as_dict(), "q_log_k0": 0.5, "q_log_by_k": {"1": 0.4, "3": 0.3, "5": 0.25}}  # noqa: E731
    return {"coverage": 0.9, "models": {"sonreb": mk(law), "rn_only": mk(rn), "vp_only": mk(vp)}}


def test_estimate_interval_calibration_and_design_flag():
    sm = StrengthModel(_fixture_spec())
    e = sm.estimate(rn=40, vp=4000)
    expect = math.exp(-8.0 + 1.1 * math.log(40) + 0.9 * math.log(4000))
    assert e.model == "sonreb" and e.k_cores == 0
    assert e.fc_mpa == pytest.approx(expect)
    assert e.lo_mpa == pytest.approx(expect * math.exp(-0.5))
    assert e.hi_mpa == pytest.approx(expect * math.exp(0.5))
    # three cores all 20 % above the law -> shift ln(1.2), post-calibration quantile for k=3
    cores = [(r, v, 1.2 * math.exp(-8.0 + 1.1 * math.log(r) + 0.9 * math.log(v))) for r, v in ((30, 3500), (45, 4200), (50, 4800))]
    c = sm.estimate(rn=40, vp=4000, cores=cores)
    assert c.k_cores == 3
    assert c.fc_mpa == pytest.approx(expect * 1.2)
    assert c.hi_mpa / c.fc_mpa == pytest.approx(math.exp(0.3))
    # far below design strength -> schedule cores (never a structural verdict)
    low = sm.estimate(rn=40, vp=4000, design_fc_mpa=expect * 10)
    assert low.below_design is True and low.action == "schedule"
    mid = sm.estimate(rn=40, vp=4000, design_fc_mpa=expect)
    assert mid.action == "calibrate_with_cores"
    assert sm.estimate(rn=40).model == "rn_only"
    assert sm.estimate(vp=4000).model == "vp_only"
    with pytest.raises(ValueError):
        sm.estimate()
    with pytest.raises(ValueError):
        sm.estimate(rn=2)


def test_shipped_model_file_loads_and_estimates():
    path = ROOT / "models" / "interior" / "ndt_strength_v1.json"
    if not path.exists():
        pytest.skip("run scripts/interior/ndt_strength.py first")
    sm = StrengthModel.load(path)
    e = sm.estimate(rn=35, vp=4000)
    assert 0 < e.lo_mpa < e.fc_mpa < e.hi_mpa
    spec = json.loads(path.read_text())
    assert set(spec["models"]) == {"sonreb", "rn_only", "vp_only"}


def test_template_has_every_schema_column_and_parses():
    assert TEMPLATE_PATH.exists()
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert text.splitlines()[0].split(",") == COLUMNS
    ok, errs = parse_csv(text)  # EXAMPLE rows are guidance only
    assert ok == [] and errs == []


def test_fixture_readings_route_and_grade():
    ok, errs = parse_csv((FIX / "readings_small.csv").read_text(encoding="utf-8"))
    assert [e["reading_id"] for e in errs] == ["T-7"]
    parts = route(ok)
    assert len(parts["strength"]) == 2 and len(parts["moisture"]) == 4
    locs = pair_strength_locations(parts["strength"])
    assert len(locs) == 1
    assert locs[0]["rn"] == 40 and locs[0]["vp"] == pytest.approx(200 / 50 * 1000)
    grades = {r.reading_id: grade_moisture(r) for r in parts["moisture"]}
    assert grades["T-4"]["level"] == "S0" and grades["T-4"]["value_label"] == "RH ok"
    assert grades["T-5"]["level"] == "U"  # spot RH >= 60 is not 'RH elevated' without 72 h
    assert grades["T-6"]["level"] == "S2"  # wood MC above WOOD_MC_MAX
    assert grades["T-3"]["level"] == "U" and "2.00 x the dry reference" in grades["T-3"]["reason"]


def test_reading_validation():
    with pytest.raises(Exception):
        FieldReading(reading_id="x", building_id="b", method="upv")
    r = FieldReading(reading_id="x", building_id="b", method="upv", path_mm=150, transit_us=37.5)
    assert r.vp_ms == pytest.approx(4000.0)
