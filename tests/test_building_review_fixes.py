"""Regression tests for the building review fixes: wet clocks, clustering, stable problem ids, as-of analysis,
work-order due dates, load and thermal clocks, storm windows, tickets, time zones and synthetic labels.

Everything runs on the labelled SYNTHETIC small tower or on hand-built inputs. These are behaviour checks, never
accuracy claims. No test calls a model API.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from building_fixtures import small_tower  # noqa: F401  (session fixture, SYNTHETIC)

from cascade.building import blueprint as bp
from cascade.building import electrical as E
from cascade.building import grading as G
from cascade.building import problems as P
from cascade.building import water as W
from cascade.building.model import (ElectricalResult, Escalation, Observation, Problem, ResponseFit, RootCause, StormEvent,
                                    WaterResult)
from cascade.building.rules import find_band_row, finding_from_row, t
from cascade.grade import load_rubric
from cascade.schema import Evidence, ImageRecord

TS = "%Y-%m-%dT%H:%M:%SZ"


def _obs(oid: str, ts: str, rubric_class: str, family: str, value: float, *, zone=None, element=None, source="x",
         kind="sensor_event", obs_value=None, **extra) -> Observation:
    rub = load_rubric(rubric_class)
    row = find_band_row(rub, family, value)
    f = finding_from_row(row, rub, finding_id=oid, asset_class=rubric_class, modality="sensor", evidence=Evidence(signal_id=source),
                         justification="hand-built test input")
    return Observation(obs_id=oid, kind=kind, ts=ts, zone_id=zone, element_id=element, source=source, level=f.unified.level,
                       value=value if obs_value is None else obs_value, finding=f, synthetic=True, **extra)


def _problem_for(group, root="F04-SHAFT") -> Problem:
    rc = RootCause(node_id=root, label="x", score=0.5, factors=[], explanation="")
    return Problem(problem_id=f"P-{root}", domain="water", title="t", level="S3", priority=None, root_cause=rc,
                   observation_ids=[o.obs_id for o in group], affected_nodes=[root], created_ts=group[0].ts, updated_ts=group[-1].ts)


def _with_sensors(data, sensors: pd.DataFrame):
    return dataclasses.replace(data, sensors=sensors)


# ------------------------------------------------------------------------------------------------ wet clocks

def test_ended_leak_run_gives_no_drying_clock(small_tower):
    data = small_tower
    now = P.default_now(data)
    sens = data.sensors.copy()
    for sid in ("F04-SHAFT-LK", "F03-SHAFT-LK"):
        wet = sens.index[sens[sid] >= 1]
        if len(wet):
            sens.loc[sens.index >= wet[0] + pd.Timedelta(hours=4), sid] = 0.0
    d2 = _with_sensors(data, sens)
    w = W.analyze_water(d2, now)
    leak = [o for o in w.observations if o.obs_id.startswith("leak:F04-SHAFT-LK") or o.obs_id.startswith("leak:F03-SHAFT-LK")]
    assert leak and all(o.end_ts is not None for o in leak)
    assert all(o.finding is None or o.finding.native_scale.value != "Active leak" for o in leak)
    assert not any(P._is_wet(d2.building, o) for o in leak)
    e = E.analyze_electrical(d2, now)
    for p in P.build_problems(d2.building, d2, w, e, now):
        if any(i.startswith("leak:F04-SHAFT-LK") for i in p.observation_ids):
            assert p.escalation is None or "drying window" not in p.escalation.what


def test_unmeasured_leak_sensor_is_never_wet(small_tower):
    b = small_tower.building
    rub = load_rubric("interior_zone")
    row = find_band_row(rub, "leak_sensor", 1.0)
    u = next(r for r in rub["rows"] if r["value"] == "U")
    f = finding_from_row(u, rub, finding_id="water:F04-SHAFT-LK:not_assessable", asset_class="interior_zone", modality="sensor",
                         evidence=Evidence(signal_id="F04-SHAFT-LK"), justification="no data")
    o = Observation(obs_id="f:u", kind="sensor_event", ts="2026-01-01T00:00:00Z", zone_id="F04-SHAFT", source="sensor:F04-SHAFT-LK",
                    level="U", value=None, unit="wet", finding=f, synthetic=True)
    assert row is not None and not P._is_wet(b, o)
    esc = P.estimate_escalation(_problem_for([o]), [o], small_tower, WaterResult(), P.parse_ts("2026-01-02T00:00:00Z"))
    assert esc is None or "drying window" not in esc.what


def test_humidity_drying_window_is_stamped_at_storm_end(small_tower):
    data = small_tower
    now0 = P.default_now(data)
    storms = W.detect_storms(data.weather[data.weather.index <= pd.Timestamp(now0)], data.building)
    st = storms[-3]
    end = pd.Timestamp(st.end)
    sid = "F08-S-RH"
    sens = data.sensors.copy()
    sens.loc[(sens.index >= pd.Timestamp(st.start)) & (sens.index <= end + pd.Timedelta(hours=60)), sid] = 75.0
    now = (end + pd.Timedelta(hours=60)).to_pydatetime()
    d2 = _with_sensors(data, sens)
    g = next(x for x in W.grade_moisture(d2, now) if x.sensor_id == sid and x.finding.defect_type == "wet_duration_h")
    assert g.ts == end.strftime(TS) and g.end_ts is None
    w = W.analyze_water(d2, now)
    o = next(x for x in w.observations if x.obs_id == f"f:water:{sid}:wet_duration_h")
    esc = P.estimate_escalation(_problem_for([o], "F08-FD-S"), [o], d2, WaterResult(), now)
    assert esc is not None and esc.hours == 0.0 and "exceeded" in esc.what


# ------------------------------------------------------------------------------------------------ clustering

def test_off_stack_room_does_not_chain_into_the_roof_problem(small_tower):
    b = small_tower.building
    ts = "2026-01-10T12:00:00Z"
    obs = [Observation(obs_id=f"ev:{z}-RH:ST", kind="sensor_event", ts=ts, zone_id=z, source=f"sensor:{z}-RH", level="S2",
                       value=15.0, synthetic=True) for z in ("F10-E", "F10-N", "F08-S")]
    fits = {z: ResponseFit(z, f"{z}-RH", "rain", 5, 0.5, 0.9, 8.0, 1.0, None, True) for z in ("F10-E", "F10-N")}
    fits["F08-S"] = ResponseFit("F08-S", "F08-S-RH", "wdr_W", 5, 0.1, 0.3, 8.0, 1.0, None, False)
    groups = P.cluster(b, obs, fits=fits)
    by_obs = {o.zone_id: i for i, g in enumerate(groups) for o in g}
    assert by_obs["F10-E"] == by_obs["F10-N"]
    assert by_obs["F08-S"] != by_obs["F10-E"]


def test_electrical_observations_group_per_circuit_and_panel_only_on_same_signal(small_tower):
    b = small_tower.building
    th = _obs("th2", "2026-01-09T10:00:00Z", "electrical_equipment", "delta_t_similar_k", 19.3, element="C-F03-2", kind="thermal_reading")
    ld = _obs("ld3", "2026-04-09T10:00:00Z", "electrical_equipment", "load_pct_continuous", 90.0, element="C-F03-3", kind="load_event")
    th1 = _obs("th1", "2026-01-09T10:00:00Z", "electrical_equipment", "delta_t_similar_k", 18.0, element="C-F03-1", kind="thermal_reading")
    g1 = P.cluster(b, [th, ld])
    assert len(g1) == 2  # an overload months later on another circuit is its own problem
    g2 = P.cluster(b, [th, th1])
    assert len(g2) == 1  # two Band 3 on one panel in one window: a panel-level cause
    assert P.electrical_root(b, g2[0]).node_id == "P-F03"


# ------------------------------------------------------------------------------------------------ ids and store

def test_problem_id_has_no_domain_and_store_carries_status_to_a_rekeyed_row(tmp_path):
    rc = RootCause(node_id="R1@F04", label="x", score=0.5, factors=[], explanation="")
    old = Problem(problem_id="P-water-R1@F04", domain="water", title="t", level="S3", priority=3.0, root_cause=rc,
                  observation_ids=["a", "b"], affected_nodes=["F04-SHAFT"], created_ts="2026-01-01T00:00:00Z", updated_ts="2026-01-02T00:00:00Z")
    store = P.ProblemStore(tmp_path / "problems.json")
    store.merge([old], "2026-01-02T00:00:00Z")
    for to in ("triaged", "work_order"):
        store.transition(old.problem_id, to, "ops", now="2026-01-03T00:00:00Z")
    new = old.model_copy(update={"problem_id": "P-R1@F04", "domain": "mixed", "level": "S4", "observation_ids": ["a", "b", "c"]})
    rows = {p.problem_id: p for p in store.merge([new], "2026-01-20T00:00:00Z")}
    assert rows["P-R1@F04"].status == "work_order" and rows["P-R1@F04"].active
    assert "re-keyed from P-water-R1@F04" in rows["P-R1@F04"].history[-1].note
    assert not rows["P-water-R1@F04"].active
    later = {p.problem_id: p for p in store.merge([], "2026-01-21T00:00:00Z")}
    assert not later["P-R1@F04"].active  # not detected again: history, not open


def test_work_order_due_counts_from_the_analysis_time():
    rc = RootCause(node_id="F06-FD-E", label="x", score=0.5, factors=[], explanation="")
    base = dict(domain="water", title="t", level="S2", priority=3.0, root_cause=rc, observation_ids=["a"], affected_nodes=["F06-E"],
                created_ts="2025-11-01T00:00:00Z", updated_ts="2025-11-03T00:00:00Z")
    p1 = Problem(problem_id="P-a", escalation=Escalation(hours=10.0, lo_hours=None, hi_hours=None, what="w", basis="b",
                                                         as_of="2026-01-28T00:00:00Z"), **base)
    p2 = Problem(problem_id="P-b", escalation=Escalation(hours=10.0, lo_hours=None, hi_hours=None, what="w", basis="b"), **base)
    rows = {r.split(",")[0]: r.split(",") for r in P.work_orders_csv([p1, p2]).strip().splitlines()[1:]}
    assert rows["P-a"][6] == "2026-01-28T10:00:00Z"
    assert rows["P-b"][6] == ""  # no analysis time recorded: no due date is invented


def test_as_of_drops_future_evidence_and_places_undated_photos_at_now():
    now = P.parse_ts("2026-01-10T00:00:00Z")
    past = Observation(obs_id="t1", kind="ticket", ts="2026-01-01T00:00:00Z", zone_id="F01-N", source="ticket:T-1")
    future = Observation(obs_id="t2", kind="ticket", ts="2026-02-01T00:00:00Z", zone_id="F01-N", source="ticket:T-2")
    undated = Observation(obs_id="i1", kind="image_finding", ts=G.UNKNOWN_TS, zone_id="F01-N", source="grader", ts_unknown=True)
    got = {o.obs_id: o for o in P.as_of_observations([past, future, undated, past], now)}
    assert set(got) == {"t1", "i1"} and got["i1"].ts == "2026-01-10T00:00:00Z" and got["i1"].ts_unknown


def test_undated_unsafe_photo_runs_no_clock(small_tower):
    b = small_tower.building
    rub = load_rubric("facade_element")
    row = next(r for r in rub["rows"] if r["value"] == "Unsafe")
    f = finding_from_row(row, rub, finding_id="img1", asset_class="facade_element", modality="rgb",
                         evidence=Evidence(image_ids=["im1"]), justification="test")
    rec = ImageRecord(image_id="im1", path="x.jpg", width=1200, height=900, sha256="0" * 64, asset_class="facade_element",
                      modality="rgb", captured_on=None, asset_id="F06-FD-E")
    o = G.finding_to_observation(f, rec, b)
    assert o.ts_unknown and "Capture date unknown" in o.text
    now = P.default_now(small_tower)
    [o2] = P.as_of_observations([o], now)
    esc = P.estimate_escalation(_problem_for([o2], "F06-FD-E"), [o2], small_tower, WaterResult(), now)
    assert esc is None or esc.hours is None  # no running legal clock from a model pre-classification
    assert esc is not None and "QEWI" in esc.what


# ------------------------------------------------------------------------------------------------ electrical clocks

def test_past_overload_has_no_ongoing_clock(small_tower):
    now = P.parse_ts("2026-01-28T00:00:00Z")
    for recent, expect in ((40.0, None), (90.0, 0.0)):
        o = _obs("ld", "2025-10-05T00:00:00Z", "electrical_equipment", "load_pct_continuous", 85.0, element="C-F08-3",
                 kind="load_event", obs_value=recent)
        esc = P.estimate_escalation(_problem_for([o], "C-F08-3"), [o], small_tower, WaterResult(), now)
        if expect is None:
            assert esc is None or "overload ongoing" not in esc.what
        else:
            assert esc is not None and esc.hours == 0.0 and "overload ongoing" in esc.what


@pytest.mark.parametrize("delta,sd,slope", [(1.5, 1.0, 0.1), (3.5, 1.0, 0.1), (0.5, 0.2, 0.1), (2.0, 0.0, 0.5)])
def test_thermal_band_window_is_ordered(delta, sd, slope):
    d, lo, hi = E.next_band_window(delta, sd, slope)
    assert lo <= d <= hi


def test_low_load_rescan_does_not_hide_an_earlier_band3(small_tower):
    b = small_tower.building
    frame = pd.DataFrame([
        {"ts": "2026-01-01T10:00:00Z", "element_id": "C-F01-1", "t_element_c": 48.0, "t_reference_c": 30.0, "t_ambient_c": 22.0, "load_pct": 70.0},
        {"ts": "2026-03-01T10:00:00Z", "element_id": "C-F01-1", "t_element_c": 48.0, "t_reference_c": 30.0, "t_ambient_c": 22.0, "load_pct": 25.0},
    ])
    fs = E.classify_thermal(b, frame)
    vals = sorted((f.native_scale.value, f.unified.level) for f in fs)
    assert ("Band 3", "S3") in vals and ("U", "U") in vals
    u = next(f for f in fs if f.unified.level == "U")
    assert "cool reading" not in u.justification and "lower bound" in u.justification


def test_hour_of_week_baseline_follows_local_time():
    idx = pd.date_range("2025-09-20", "2025-12-20", freq="h", tz="UTC")
    local = idx.tz_convert("America/New_York")
    on = (local.dayofweek < 5) & (local.hour >= 8) & (local.hour < 19)
    rng = np.random.default_rng(3)
    cur = pd.DataFrame({"A": 4.0 + 10.0 * on + rng.normal(0, 0.3, len(idx))}, index=idx)
    n_local = len(E.baseline_anomalies(cur, tz="America/New_York"))
    n_utc = len(E.baseline_anomalies(cur))
    assert n_local * 5 < n_utc  # the UTC key shifts every weekday ramp after the DST change


# ------------------------------------------------------------------------------------------------ water details

def test_overlapping_storm_windows_are_not_double_counted():
    idx = pd.date_range("2026-01-01", periods=96, freq="h", tz="UTC")
    rh = pd.Series(40.0, index=idx)
    rh[(idx >= "2026-01-03T08:00:00Z") & (idx <= "2026-01-03T20:00:00Z")] = 60.0
    e_st = StormEvent("ST-001", "2026-01-03T00:00:00Z", "2026-01-03T06:00:00Z", 20.0, 5.0, 90.0, 8.0, {"E": 100.0, "W": 0.0})
    w_st = StormEvent("ST-002", "2026-01-03T13:00:00Z", "2026-01-03T16:00:00Z", 20.0, 5.0, 270.0, 8.0, {"E": 0.0, "W": 100.0})
    resp = W.storm_responses(rh, [e_st, w_st]).set_index("storm_id")
    assert resp.loc["ST-001", "rise"] == pytest.approx(20.0)
    assert np.isnan(resp.loc["ST-002", "rise"])  # its baseline sits inside the first storm's response


def test_closed_tickets_do_not_count_as_open(small_tower):
    b = small_tower.building
    now = P.parse_ts("2026-01-20T00:00:00Z")
    tks = [Observation(obs_id=f"ticket:T-{i}", kind="ticket", ts="2026-01-10T00:00:00Z", zone_id="F02-N", source=f"ticket:T-{i}",
                       text="water leak", category="water", status=st) for i, st in enumerate(("closed", "resolved", "closed"))]
    fs = {f.name: f for f in W.zone_risk(b, "F02-N", {}, tks, [], now)}
    assert fs["open_tickets"].value == 0.0
    fs2 = {f.name: f for f in W.zone_risk(b, "F02-N", {}, tks + [tks[0].model_copy(update={"obs_id": "ticket:T-9", "status": "open"})], [], now)}
    assert fs2["open_tickets"].value > 0.0


def test_rain_response_says_too_few_storms_when_data_exists(small_tower):
    b = small_tower.building
    fit = ResponseFit("F02-N", "F02-N-RH", "none", 2, None, None, 2.5, None, "2026-01-01T00:00:00Z", False)
    fs = {f.name: f for f in W.zone_risk(b, "F02-N", {"F02-N": fit}, [], [], P.parse_ts("2026-01-20T00:00:00Z"))}
    assert fs["rain_response"].value is None and "fewer than RESPONSE_MIN_EVENTS" in fs["rain_response"].explanation


def test_storm_event_ids_name_the_sensor(small_tower):
    w = W.analyze_water(small_tower, P.default_now(small_tower))
    evs = [o for o in w.observations if o.obs_id.startswith("ev:")]
    assert evs and all(o.obs_id.split(":")[1] == o.source.split(":", 1)[1] for o in evs)


# ------------------------------------------------------------------------------------------------ synthetic labels

def test_synthetic_input_files_label_a_real_building(small_tower, tmp_path):
    root = tmp_path / "b"
    shutil.copytree(small_tower.root, root)
    raw = json.loads((root / "building.json").read_text(encoding="utf-8"))
    raw["synthetic"] = False
    for s in raw["sensors"]:
        s["synthetic"] = False
    (root / "building.json").write_text(json.dumps(raw), encoding="utf-8")
    data = bp.load_building_dir(root)
    assert data.synthetic_sources["sensors"] and data.synthetic_sources["weather"]
    assert data.building.synthetic and data.any_synthetic
    w = W.analyze_water(data, P.default_now(data))
    assert w.observations and all(o.synthetic for o in w.observations)
