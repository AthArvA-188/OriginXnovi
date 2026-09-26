"""Water intrusion tracking and prediction (cascade.building.water) on the SYNTHETIC demo tower.

The tower in data/demo/building is generated data with injected, labelled scenarios. These tests check that the
water scenarios (S1 facade crack under east wind-driven rain, S2 blocked roof drain, S3 riser leak) are found, as a
behaviour check against ground_truth.json. They are never an accuracy claim. No test calls a model API.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from building_fixtures import small_tower  # noqa: F401  (session fixture)

from cascade import drift
from cascade.building import problems as P
from cascade.building import water as W
from cascade.building.blueprint import load_building_dir
from cascade.building.model import BuildingData, ElectricalResult, WaterResult
from cascade.building.rules import THRESHOLDS, t
from cascade.building.synthetic import TowerConfig, generate_tower
from cascade.grade import load_rubric

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo" / "building"
ORDER = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Tuple[BuildingData, dict]:
    """The demo tower (read only). When it is missing, the same default config is generated into a temp folder."""
    root = DEMO
    if not (root / "building.json").exists() or not (root / "ground_truth.json").exists():
        root = tmp_path_factory.mktemp("demo_tower")
        generate_tower(root, TowerConfig())
    data = load_building_dir(root)
    truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    return data, truth


@pytest.fixture(scope="module")
def result(demo) -> WaterResult:
    data, _ = demo
    return W.analyze_water(data, P.default_now(data))


def _scenario(truth: dict, sid: str) -> dict:
    return next(s for s in truth["scenarios"] if s["scenario_id"] == sid)


def _s1_room(data: BuildingData, truth: dict) -> str:
    return next(n for n in _scenario(truth, "S1")["affected_nodes"] if data.building.zone(n).kind == "room")


# ------------------------------------------------------------------------------------------------ storms

def test_detect_storms_matches_generator_count(demo, result):
    data, truth = demo
    assert abs(len(result.storms) - truth["storm_count"]) <= 1
    for st in result.storms:
        assert st.total_mm >= t("STORM_MIN_MM")
        assert set(st.wdr) == {"N", "E", "S", "W"}  # orientations used by the facade drops
        assert st.start <= st.end


def test_wdr_index_follows_wind_from_direction():
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    w = pd.DataFrame({"rain_mm": [2.0, 2.0, np.nan], "wind_speed_ms": [5.0, 5.0, 5.0], "wind_dir_deg": [90.0, 270.0, 90.0]}, index=idx)
    east = W.wdr_index(w, "E")
    assert math.isclose(east.iloc[0], 10.0) and east.iloc[1] == 0.0
    assert math.isnan(east.iloc[2])  # missing rain stays missing, never 0


# ------------------------------------------------------------------------------------------------ rain response (S1, S2)

def test_s1_east_facade_driver_is_significant(demo, result):
    data, truth = demo
    room = _s1_room(data, truth)
    fit = result.fits[room]
    assert fit.driver == "wdr_E" and fit.significant
    assert fit.r >= t("RESPONSE_MIN_R") and fit.n_events >= t("RESPONSE_MIN_EVENTS")
    stats = W.driver_stats(W.all_responses(data, result.storms)[fit.sensor_id], result.storms, W.drivers_for(data.building))
    r_w = stats.set_index("driver").loc["wdr_W", "r"]
    assert r_w is not None and r_w < t("RESPONSE_MIN_R")


def test_s2_roof_drain_driver_is_rain_after_onset(demo, result):
    data, truth = demo
    s2 = _scenario(truth, "S2")
    rooms = [n for n in s2["affected_nodes"] if data.building.zone(n) is not None and data.building.zone(n).kind == "room"]
    assert rooms
    for room in rooms:
        fit = result.fits[room]
        assert fit.driver == "rain" and fit.significant, fit
        assert pd.Timestamp(fit.onset_ts) >= pd.Timestamp(s2["onset_ts"])


def test_only_truth_zones_have_significant_fits(demo, result):
    """SYNTHETIC sanity check, not an accuracy claim: significant rain responses appear only in injected zones."""
    data, truth = demo
    nodes = set()
    for s in truth["scenarios"]:
        nodes |= set(s["affected_nodes"])
    assert {z for z, f in result.fits.items() if f.significant} <= nodes


def test_unsensored_zone_is_unmeasured_not_zero(demo, result):
    data, truth = demo
    drop = _scenario(truth, "S1")["root_nodes"][0]  # a facade drop: no humidity sensor
    assert not [s for s in data.building.sensors_in(drop) if s.type == "humidity"]
    fit = W.fit_response(drop, "", W.storm_responses(pd.Series(dtype=float), result.storms), result.storms, ["rain", "wdr_E"])
    assert fit.driver == "none" and fit.gain is None and fit.r is None and not fit.significant
    assert drop not in result.fits
    factors = {f.name: f for f in result.risks[drop]}
    assert factors["rain_response"].value is None and factors["rain_response"].contribution == 0.0
    assert factors["trend"].value is None
    assert "not measured" in factors["rain_response"].explanation
    assert W.risk_score([f for f in factors.values() if f.value is None]) is None  # all unmeasured -> None, not 0


def test_risk_factors_are_explained_and_weighted(demo, result):
    data, truth = demo
    room = _s1_room(data, truth)
    fs = result.risks[room]
    assert [f.name for f in fs] == list(t("RISK_WEIGHTS"))
    for f in fs:
        assert f.basis.startswith("RISK_WEIGHTS") and f.explanation
        assert f.value is None or 0.0 <= f.value <= 1.0
    rr = next(f for f in fs if f.name == "rain_response")
    assert rr.value > 0 and "wdr_E" in rr.explanation
    grid = W.risk_grid(data.building, result.risks)
    assert set(grid["side"]) <= {"N", "E", "S", "W"} and len(grid) == 4 * data.building.floors
    top = grid.sort_values("risk", ascending=False).head(3)["zone_id"].tolist()
    assert room in top


# ------------------------------------------------------------------------------------------------ forecast

def test_forecast_east_storm_crosses_west_does_not(demo, result):
    data, truth = demo
    room = _s1_room(data, truth)
    east = {f.zone_id: f for f in W.forecast(data.building, data, result.fits, rain_mm=30.0, duration_h=8.0, wind_dir_deg=90.0, wind_speed_ms=10.0)}
    fc = east[room]
    assert fc.crosses and fc.driver == "wdr_E" and fc.threshold_key == "EPA_RH_MAX"
    assert fc.hours_to_threshold is not None and math.isfinite(fc.hours_to_threshold)
    lo_h, hi_h = W.forecast_hours_band(fc, result.fits[room], 8.0)
    assert lo_h is not None and hi_h is not None and lo_h <= fc.hours_to_threshold <= hi_h
    assert fc.lo <= fc.peak <= fc.hi
    assert THRESHOLDS[fc.threshold_key].label().startswith("60.0 %RH")
    west = {f.zone_id: f for f in W.forecast(data.building, data, result.fits, rain_mm=30.0, duration_h=8.0, wind_dir_deg=270.0, wind_speed_ms=10.0)}
    assert not west[room].crosses and west[room].hours_to_threshold is None


# ------------------------------------------------------------------------------------------------ moisture grading (S3)

def test_s3_riser_leak_is_active_and_past_drying_window(demo, result):
    data, truth = demo
    s3 = _scenario(truth, "S3")
    onset = pd.Timestamp(s3["onset_ts"])
    shaft = s3["affected_nodes"][0]
    leak = [o for o in result.observations if o.zone_id == shaft and o.unit == "wet"]
    assert leak and all(pd.Timestamp(o.ts) >= onset for o in leak)
    assert all(o.level == "S3" and o.finding.native_scale.value == "Active leak" for o in leak)
    values = {f.native_scale.value for f in result.findings if f.evidence.signal_id == f"{shaft}-LK"}
    assert {"Active leak", "Drying window exceeded"} <= values
    wet_zones = {o.zone_id for o in result.observations if o.unit == "wet"}
    assert wet_zones <= set(s3["affected_nodes"])  # SYNTHETIC sanity: no leak runs away from the injected leak


def test_all_nan_window_gives_u_never_rh_ok(demo):
    data, truth = demo
    room = _s1_room(data, truth)
    sid = f"{room}-RH"
    now = P.default_now(data)
    sensors = data.sensors.copy()
    sensors.loc[sensors.index > pd.Timestamp(now) - pd.Timedelta(days=int(t("ASHRAE160_WINDOW_D"))), sid] = np.nan
    blind = dataclasses.replace(data, sensors=sensors)
    graded = [g for g in W.grade_moisture(blind, now) if g.sensor_id == sid]
    assert len(graded) == 1
    f = graded[0].finding
    assert f.unified.level == "U" and f.native_scale.value == "U" and "not_measurable" in f.unified.flags
    assert all(g.finding.native_scale.value != "RH ok" for g in graded)
    counts = P.level_counts([])  # U has its own bucket and is never added to S0
    assert "U" in counts and counts["S0"] == 0
    obs = W.water_observations(blind, [], {}, [f], graded=graded)
    assert [o.level for o in obs if o.source == f"sensor:{sid}"] == ["U"]


def test_findings_quote_interior_water_rows(result):
    rub = load_rubric("interior_zone")
    criteria = [r["criterion"] for r in rub["rows"]]
    assert result.findings
    for f in result.findings:
        assert f.asset_class == "interior_zone" and f.modality == "sensor" and f.evidence.signal_id
        assert f.native_scale.value in rub["allowed_values"]
        assert f.native_scale.criteria_matched
        for c in f.native_scale.criteria_matched:
            assert any(c in full for full in criteria), c
        hard, _, _ = drift._audit_one(f, rub)
        assert hard == [], (f.finding_id, hard)
        assert f.unified.level != "S0" or f.native_scale.value == "RH ok"


def test_observations_are_labelled_and_placed(demo, result):
    data, _ = demo
    b = data.building
    assert result.observations
    for o in result.observations:
        assert o.synthetic is True  # the demo tower is synthetic
        assert b.zone(o.zone_id) is not None and o.source.startswith("sensor:")
        assert o.level in ("S1", "S2", "S3", "S4", "U")  # S0 findings are context only, never observations
    assert len({o.obs_id for o in result.observations}) == len(result.observations)


# ------------------------------------------------------------------------------------------------ into the problem manager

def _problems(data: BuildingData, water: WaterResult):
    elec = ElectricalResult(load_table=pd.DataFrame(), anomalies=pd.DataFrame())
    return P.build_problems(data.building, data, water, elec, P.default_now(data))


def _check_water_scenarios(data: BuildingData, water: WaterResult, truth: dict) -> Dict[str, str]:
    probs = _problems(data, water)
    found = {}
    for sid in ("S1", "S2", "S3"):
        s = _scenario(truth, sid)
        hits = [p for p in probs if p.root_cause.node_id in s["root_nodes"]]
        assert hits, (sid, [p.problem_id for p in probs][:10])
        assert ORDER[hits[0].level] >= ORDER[s["expected_level_min"]], (sid, hits[0].level)
        assert hits[0].priority is not None
        found[sid] = hits[0].problem_id
    s3 = next(p for p in probs if p.root_cause.node_id in _scenario(truth, "S3")["root_nodes"])
    assert "EPA" in s3.escalation.what and "drying window" in s3.escalation.what
    s1 = next(p for p in probs if p.root_cause.node_id in _scenario(truth, "S1")["root_nodes"])
    assert s1.escalation is not None and s1.escalation.basis.split(",")[0].strip() in THRESHOLDS
    truth_nodes = set()
    for s in truth["scenarios"]:
        truth_nodes |= set(s["affected_nodes"]) | set(s["root_nodes"])
    for p in probs:  # SYNTHETIC sanity check, not an accuracy claim
        if p.level in ("S3", "S4"):
            assert set(p.affected_nodes) & truth_nodes, p.problem_id
    return found


def test_water_scenarios_become_problems_with_true_roots(demo, result):
    data, truth = demo
    found = _check_water_scenarios(data, result, truth)
    assert found["S2"].endswith("ROOF-NE") or found["S2"].endswith("RD-2")


def test_small_tower_water_scenarios(small_tower):
    """The same checks on the 10-floor, 120-day SYNTHETIC tower (different scenario floors and onsets)."""
    truth = json.loads((small_tower.root / "ground_truth.json").read_text(encoding="utf-8"))
    water = W.analyze_water(small_tower, P.default_now(small_tower))
    assert abs(len(water.storms) - truth["storm_count"]) <= 1
    _check_water_scenarios(small_tower, water, truth)
