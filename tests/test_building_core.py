"""Core building layer: model, rules, graph, blueprint store, SYNTHETIC tower and problem manager.

Everything here runs on the labelled SYNTHETIC tower. These are behaviour and sanity checks, never accuracy claims.
No test calls a model API.
"""

from __future__ import annotations

import gzip
import io
import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest
from building_fixtures import small_tower  # noqa: F401  (session fixture)
from PIL import Image

from cascade import drift
from cascade.building import blueprint as bp
from cascade.building import graph as g
from cascade.building import problems as P
from cascade.building.model import (Building, Edge, ElectricalResult, Escalation, Observation, Problem, ResponseFit, RootCause,
                                    StormEvent, WaterResult, Zone)
from cascade.building.rules import THRESHOLDS, find_band_row, finding_from_row, t, u_finding
from cascade.building.synthetic import TowerConfig, build_tower_model, generate_tower, load_truth, positions
from cascade.grade import RUBRIC_FOR_CLASS, load_rubric
from cascade.schema import Evidence

TEN = TowerConfig(floors=10, days=120, seed=7)


@pytest.fixture(scope="module")
def model10() -> Building:
    return build_tower_model(TEN)


# ------------------------------------------------------------------------------------------------ model

def test_building_round_trip(tmp_path, model10):
    bp.save_building(model10, tmp_path)
    b2 = bp.load_building(tmp_path)
    assert b2 == model10
    assert b2.floor("F01").plan_image == "plans/F01.png"  # plan paths stay relative


def test_dangling_edge_raises(model10):
    raw = model10.model_dump()
    raw["edges"].append({"src": "F01-N", "dst": "F99-N", "kind": "above"})
    with pytest.raises(ValueError, match="F99-N"):
        Building.model_validate(raw)


def test_facade_drop_needs_orientation():
    with pytest.raises(ValueError, match="orientation"):
        Zone(zone_id="F01-FD-X", floor_id="F01", kind="facade_drop", polygon=[(0, 0), (10, 0), (10, 10)])
    with pytest.raises(ValueError, match="3 points"):
        Zone(zone_id="F01-N", floor_id="F01", kind="room", polygon=[(0, 0), (10, 0)])


def test_observation_needs_zone_or_element():
    with pytest.raises(ValueError, match="zone_id or an element_id"):
        Observation(obs_id="o", kind="ticket", ts="2026-01-01T00:00:00Z", source="ticket:T-1")
    o = Observation(obs_id="o", kind="ticket", ts="2026-01-01T00:00:00Z", zone_id="F01-N", source="ticket:T-1")
    assert o.level is None  # tickets are not graded


def _problem(pid: str = "P-water-F06-FD-E", level: str = "S2", updated: str = "2026-01-02T00:00:00Z", priority=3.0) -> Problem:
    rc = RootCause(node_id="F06-FD-E", label="x", score=0.5, factors=[], explanation="")
    return Problem(problem_id=pid, domain="water", title="t", level=level, priority=priority, root_cause=rc,
                   observation_ids=["a"], affected_nodes=["F06-E"], created_ts="2026-01-01T00:00:00Z", updated_ts=updated,
                   escalation=Escalation(hours=10.0, lo_hours=5.0, hi_hours=20.0, what="w", basis="EPA_DRY_WINDOW_H"))


def test_problem_json_round_trip():
    p = _problem()
    assert Problem.model_validate_json(p.model_dump_json()) == p


# ------------------------------------------------------------------------------------------------ rules

def test_every_threshold_has_url_or_tag():
    for key, th in THRESHOLDS.items():
        assert th.key == key
        ok_url = bool(th.url) and th.url.startswith("https://")
        assert ok_url or th.tag in ("team-proposed, validate", "Assumption"), key
        if th.tag == "PUBLIC":
            assert ok_url, key
        assert th.label()
    with pytest.raises(KeyError):
        t("NOT_A_KEY")


def test_band_rows_and_findings_follow_the_rubric():
    rub = load_rubric("electrical_equipment")
    assert find_band_row(rub, "delta_t_similar_k", 0.5)["value"] == "Band 0"
    assert find_band_row(rub, "delta_t_similar_k", 4.0)["value"] == "Band 2"  # min inclusive, max exclusive
    assert find_band_row(rub, "delta_t_similar_k", 15.0)["value"] == "Band 3"
    assert find_band_row(rub, "delta_t_similar_k", float("nan")) is None
    iw = load_rubric("interior_zone")
    row = find_band_row(iw, "leak_sensor", 1.0)
    f = finding_from_row(row, iw, finding_id="lk", asset_class="interior_zone", modality="sensor",
                         evidence=Evidence(signal_id="F04-SHAFT-LK"), justification="leak sensor wet")
    assert f.unified.level == "S3" and f.native_scale.criteria_matched == [row["criterion"]]
    assert f.action.basis.startswith("interior_water.json row 'Active leak'")
    hard, _, _ = drift._audit_one(f, iw)
    assert hard == []
    u = u_finding(iw, finding_id="u", asset_class="interior_zone", modality="sensor", evidence=Evidence(), reason="no data in the window")
    assert u.unified.level == "U" and u.modality == "sensor" and "not_measurable" in u.unified.flags


def test_new_rubrics_load_with_u_rows():
    for ac in ("facade_element", "interior_zone", "electrical_equipment"):
        rub = load_rubric(ac)
        assert RUBRIC_FOR_CLASS[ac].endswith(".json")
        assert {r["value"] for r in rub["rows"]} <= set(rub["allowed_values"])
        assert any(r["unified"] == "U" for r in rub["rows"])
    assert load_rubric("electrical_equipment")["allowed_values"][-1] == "Band 3"  # S4-eligible value last


# ------------------------------------------------------------------------------------------------ graph

def test_reach_is_max_product_and_water_flows_down(model10):
    adj = {"a": [("b", 0.5, "drains_to"), ("c", 0.9, "drains_to")], "c": [("b", 0.9, "drains_to")]}
    r = g.reach(adj, {"a": 1.0})
    assert math.isclose(r["b"].score, 0.81) and r["b"].path == ("a", "c", "b") and r["b"].hops == 2
    down = g.downstream(model10, "F06-FD-E")
    assert "F06-E" in down and "F05-E" in down and "F07-E" not in down
    assert math.isclose(down["F06-E"].score, t("EDGE_WEIGHT")["drains_to"])


def test_upstream_of_room(model10):
    up = g.upstream(model10, "F05-E")
    assert "F05-FD-E" in up and "F06-E" in up
    assert "F04-E" not in up  # water does not climb


def test_electrical_tree(model10):
    assert set(g.subtree(model10, "P-F03")) == {"P-F03", "C-F03-1", "C-F03-2", "C-F03-3", "C-F03-4",
                                                "LD-F03-1", "LD-F03-2", "LD-F03-3", "LD-F03-4"}
    assert g.path_to_root(model10, "LD-F03-2") == ["LD-F03-2", "C-F03-2", "P-F03", "SWB-1"]
    assert g.zone_of(model10, "RD-2") == "ROOF-NE" and g.zone_of(model10, "F02-N") == "F02-N"
    assert set(g.water_sources(model10)) >= {"ROOF-NE", "RD-2", "R1@F04", "F06-FD-E"}


# ------------------------------------------------------------------------------------------------ blueprint

def test_point_in_polygon_and_zone_at(model10):
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert bp.point_in_polygon((5, 5), sq) and not bp.point_in_polygon((15, 5), sq)
    assert bp.centroid(sq) == (5, 5)
    assert bp.zone_at(model10, "F03", (400, 100)) == "F03-N"
    assert bp.zone_at(model10, "F03", (400, 300)) == "F03-CORE"
    assert bp.zone_at(model10, "F03", (795, 300)) == "F03-FD-E"


def test_render_plan_is_plan_size(small_tower):
    b = small_tower.building
    obs = [Observation(obs_id="x", kind="sensor_event", ts="2026-01-01T00:00:00Z", zone_id="F06-E", source="sensor:F06-E-RH", level="S2"),
           Observation(obs_id="u", kind="sensor_event", ts="2026-01-01T00:00:00Z", zone_id="F06-N", source="sensor:F06-N-RH", level="U"),
           Observation(obs_id="t", kind="ticket", ts="2026-01-01T00:00:00Z", zone_id="F06-W", source="ticket:T-9")]
    img = bp.render_plan(b, small_tower.root, "F06", obs, risk_by_zone={"F06-E": 0.8}, highlight=["F06-FD-E"])
    assert img.size == tuple(b.floor("F06").plan_size_px)
    assert bp.place(b, obs[0])[0] == "F06"
    assert bp.render_plan(b, small_tower.root, "RF", []).size == (800, 600)


def test_tickets_and_zones_csv(tmp_path, model10):
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"], utc=True),
                       "ticket_id": ["T-1", "T-2", "T-3"], "zone_id": ["F02-N", "F02-E", "NOPE"],
                       "text": ["drip from ceiling", "breaker keeps tripping", "x"], "category": [None, None, None],
                       "status": ["open"] * 3, "synthetic": [True] * 3})
    obs = bp.tickets_to_observations(model10, df)
    assert [(o.zone_id, o.category, o.level) for o in obs] == [("F02-N", "water", None), ("F02-E", "electrical", None)]
    p = tmp_path / "zones.csv"
    p.write_text("floor_id,zone_id,name,kind,orientation,polygon\nF01,F01-FD-X,x,facade_drop,E,0 0;10 0;10 10\n", encoding="utf-8")
    zs = bp.load_zones_csv(p, ["F01"])
    assert zs[0].orientation == "E" and zs[0].drop_id == "FD-E" and len(zs[0].polygon) == 3
    with pytest.raises(ValueError):
        bp.load_zones_csv(p, ["F02"])
    assert bp.ifc_supported() is False


# ------------------------------------------------------------------------------------------------ synthetic

def test_default_tower_floors_in_range():
    b = build_tower_model(TowerConfig())
    assert 30 <= b.floors <= 40 and b.synthetic
    assert all(s.synthetic for s in b.sensors)
    assert b.floor("RF").level == b.floors + 1


def test_synthetic_files_exist_and_are_labelled(small_tower):
    root = small_tower.root
    b = small_tower.building
    assert b.synthetic is True
    for f in b.floor_list:
        assert (root / f.plan_image).exists()
    for name in ("weather.csv", "thermal.csv", "tickets.csv"):
        df = pd.read_csv(root / name)
        assert len(df) > 0 and df["synthetic"].astype(str).eq("True").all(), name
    with gzip.open(root / "sensors.csv.gz", "rt", encoding="utf-8") as fh:
        sens = pd.read_csv(fh)
    assert sens["synthetic"].astype(str).eq("True").all()
    assert set(s.sensor_id for s in b.sensors) <= set(sens.columns)
    assert (root / "ground_truth.json").read_text(encoding="utf-8") == (root / "truth.json").read_text(encoding="utf-8")


def test_truth_scenarios_exist_in_building(small_tower):
    truth = load_truth(small_tower.root)
    assert truth["synthetic"] is True and len(truth["scenarios"]) == 5
    nodes = small_tower.building.node_ids()
    for s in truth["scenarios"]:
        assert s["root_nodes"] and set(s["root_nodes"]) <= nodes, s["scenario_id"]
        assert set(s["affected_nodes"]) <= nodes
    assert truth["storm_count"] == len(truth["storms"]) > 10


def test_facade_images_are_large_and_synthetic(small_tower):
    assert len(small_tower.images) == 6
    for r in small_tower.images:
        assert r.width >= 800 and r.height >= 600 and r.labels.get("synthetic") is True
        assert r.asset_class == "facade_element" and small_tower.building.zone(r.asset_id) is not None
        with Image.open(r.path) as im:
            assert im.size == (r.width, r.height)
    assert sum(1 for r in small_tower.images if r.labels.get("scenario") == "S1") == 1


def test_generation_is_deterministic(tmp_path, small_tower):
    again = generate_tower(tmp_path / "again", TEN)
    for name in ("sensors.csv.gz", "weather.csv", "thermal.csv", "tickets.csv", "panel_schedule.csv"):
        assert (again.root / name).read_bytes() == (small_tower.root / name).read_bytes(), name


def test_injected_signals_are_present(small_tower):
    """Generator sanity on SYNTHETIC data (not an accuracy claim): the S1 room rises after east storms, not west ones."""
    p = positions(TEN)
    s = small_tower.sensors[f"F{p['c']:02d}-E-RH"]
    truth = load_truth(small_tower.root)

    def rise(st: dict) -> float:
        start, end = pd.Timestamp(st["start"]), pd.Timestamp(st["end"])
        pre = s[(s.index >= start - pd.Timedelta(hours=24)) & (s.index < start)].median()
        post = s[(s.index >= start) & (s.index <= end + pd.Timedelta(hours=48))].max()
        return float(post - pre)

    east = [rise(st) for st in truth["storms"] if st["wind_from"] == "E"]
    west = [rise(st) for st in truth["storms"] if st["wind_from"] == "W"]
    assert max(east) > t("RESPONSE_MIN_RISE") and max(west) < t("RESPONSE_MIN_RISE")
    lk = small_tower.sensors[f"F{p['m']:02d}-SHAFT-LK"]
    assert lk.max() == 1.0 and lk.iloc[0] == 0.0


# ------------------------------------------------------------------------------------------------ problems (hand-built inputs)

def _obs_from_row(oid: str, ts: str, rubric_class: str, family: str, value: float, *, zone=None, element=None, source="x",
                  kind="sensor_event", modality="sensor") -> Observation:
    rub = load_rubric(rubric_class)
    row = find_band_row(rub, family, value)
    f = finding_from_row(row, rub, finding_id=oid, asset_class=rubric_class, modality=modality, evidence=Evidence(signal_id=source),
                         justification="hand-built test input")
    return Observation(obs_id=oid, kind=kind, ts=ts, zone_id=zone, element_id=element, source=source, level=f.unified.level,
                       value=value, finding=f, synthetic=True)


def _storm_events(truth: dict) -> List[StormEvent]:
    out = []
    for st in truth["storms"]:
        wdr = {o: st["total_mm"] * st["wind_speed_ms"] * max(0.0, math.cos(math.radians(st["wind_dir_deg"] - az)))
               for o, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270))}
        out.append(StormEvent(st["storm_id"], st["start"], st["end"], st["total_mm"], 1.0, st["wind_dir_deg"], st["wind_speed_ms"], wdr))
    return out


def _hand_built(small_tower) -> tuple:
    """Observations and fits shaped like the water and electrical outputs, built from the SYNTHETIC truth."""
    truth = load_truth(small_tower.root)
    p = positions(TEN)
    c, top, m, l, o = (f"F{p[k]:02d}" for k in ("c", "top", "m", "l", "o"))
    m1, m2 = f"F{p['m'] - 1:02d}", f"F{p['m'] - 2:02d}"
    s2_on, s3_on = truth["scenarios"][1]["onset_ts"], truth["scenarios"][2]["onset_ts"]
    obs: List[Observation] = []
    for st in truth["storms"]:
        if st["wind_from"] == "E":
            obs.append(Observation(obs_id=f"ev:{c}-E:{st['storm_id']}", kind="sensor_event", ts=st["end"], zone_id=f"{c}-E",
                                   source=f"sensor:{c}-E-RH", level="S2", value=20.0, synthetic=True))
        if st["start"] >= s2_on:
            for z in (f"{top}-N", f"{top}-E"):
                obs.append(Observation(obs_id=f"ev:{z}:{st['storm_id']}", kind="sensor_event", ts=st["end"], zone_id=z,
                                       source=f"sensor:{z}-RH", level="S1", value=15.0, synthetic=True))
    on = pd.Timestamp(s3_on)
    fmt = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    obs.append(_obs_from_row("lk1", fmt(on + pd.Timedelta(hours=2)), "interior_zone", "leak_sensor", 1.0, zone=f"{m}-SHAFT", source=f"sensor:{m}-SHAFT-LK"))
    obs.append(_obs_from_row("lk2", fmt(on + pd.Timedelta(hours=36)), "interior_zone", "leak_sensor", 1.0, zone=f"{m1}-SHAFT", source=f"sensor:{m1}-SHAFT-LK"))
    obs.append(_obs_from_row("rh3", fmt(on + pd.Timedelta(hours=100)), "interior_zone", "rh_sustained_h", 80.0, zone=f"{m2}-CORE", source=f"sensor:{m2}-CORE-RH"))
    obs.append(_obs_from_row("th", "2026-01-09T10:00:00Z", "electrical_equipment", "delta_t_similar_k", 19.3, element=f"C-{l}-2",
                             kind="thermal_reading", modality="thermal", source="thermal"))
    obs.append(_obs_from_row("ld", "2026-01-28T00:00:00Z", "electrical_equipment", "load_pct_continuous", 90.0, element=f"C-{o}-3",
                             kind="load_event", source=f"load:C-{o}-3"))
    obs.append(_obs_from_row("ok", "2026-01-28T00:00:00Z", "interior_zone", "rh_hourly", 45.0, zone="F05-W", source="sensor:F05-W-RH"))
    fits: Dict[str, ResponseFit] = {
        f"{c}-E": ResponseFit(f"{c}-E", f"{c}-E-RH", "wdr_E", 6, 0.1, 0.9, 8.0, 1.0, None, True),
        f"{top}-N": ResponseFit(f"{top}-N", f"{top}-N-RH", "rain", 5, 0.8, 0.9, 8.0, 1.0, s2_on, True),
        f"{top}-E": ResponseFit(f"{top}-E", f"{top}-E-RH", "rain", 5, 0.5, 0.9, 8.0, 1.0, s2_on, True),
        f"{m2}-CORE": ResponseFit(f"{m2}-CORE", f"{m2}-CORE-RH", "none", 0, None, None, None, None, None, False),
    }
    water = WaterResult(storms=_storm_events(truth), fits=fits, observations=obs)
    elec = ElectricalResult(load_table=pd.DataFrame(), anomalies=pd.DataFrame())
    return truth, water, elec


@pytest.fixture(scope="module")
def hand_problems(small_tower):
    truth, water, elec = _hand_built(small_tower)
    now = P.default_now(small_tower)
    return truth, P.build_problems(small_tower.building, small_tower, water, elec, now)


def test_problem_roots_on_hand_built_observations(hand_problems):
    """Root ranking and grouping logic on inputs shaped like the water/electrical outputs (SYNTHETIC, not accuracy)."""
    truth, probs = hand_problems
    order = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}
    for s in truth["scenarios"]:
        hits = [p for p in probs if p.root_cause.node_id in s["root_nodes"]]
        assert hits, (s["scenario_id"], [p.problem_id for p in probs])
        assert order[hits[0].level] >= order["S1"]
    s1 = next(p for p in probs if p.root_cause.node_id == truth["scenarios"][0]["root_nodes"][0])
    assert all(n.endswith("-E") for n in s1.affected_nodes)  # east-driven events are not folded into the roof problem
    s2 = next(p for p in probs if p.root_cause.node_id in truth["scenarios"][1]["root_nodes"])
    assert "RD-2" in s2.root_cause.label
    s3 = next(p for p in probs if p.root_cause.node_id == truth["scenarios"][2]["root_nodes"][0])
    m = positions(TEN)["m"]
    assert s3.affected_nodes.index(f"F{m - 1:02d}-SHAFT") < s3.affected_nodes.index(f"F{m - 2:02d}-CORE")
    assert "EPA" in s3.escalation.what and "drying window" in s3.escalation.what and s3.escalation.basis == "EPA_DRY_WINDOW_H"
    assert s1.escalation is not None and all(k.strip() in THRESHOLDS for k in s1.escalation.basis.split(","))
    for p in probs:  # every factor names its basis; unmeasured factors contribute nothing
        for f in p.root_cause.factors:
            assert f.basis and (f.value is not None or f.contribution == 0.0)
    assert not any("F05-W" in p.affected_nodes for p in probs)  # S0 is context only


def test_problem_levels_never_count_u_as_s0(hand_problems):
    _, probs = hand_problems
    u = _problem("P-water-U", level="U", priority=None)
    counts = P.level_counts(list(probs) + [u])
    assert counts["U"] == 1 and counts["S0"] == 0
    assert P.priority("U", [], None) is None
    ranked = P.sort_problems([u] + list(probs))
    assert ranked[-1].problem_id == "P-water-U"
    assert P.priority("S2", ["fire_shock_pathway"], None) == 2 * P.priority("S2", [], None)
    assert P.urgency(0.0) == t("URGENCY")["max"] and P.urgency(None) == t("URGENCY")["min"]
    csv_text = P.work_orders_csv(list(probs) + [u])
    assert csv_text.splitlines()[0] == "id,title,level,priority,status,root,due,basis"
    assert csv_text.strip().splitlines()[-1].startswith("P-water-U,")


def test_lifecycle_and_store(tmp_path):
    store = P.ProblemStore(tmp_path / "problems.json")
    p = _problem(updated="2026-01-02T00:00:00Z")
    merged = store.merge([p], "2026-01-03T00:00:00Z")
    assert merged[0].status == "detected" and merged[0].history[0].to_status == "detected"
    assert store.load() == merged  # save/load round trip
    with pytest.raises(ValueError):
        store.transition(p.problem_id, "resolved", "ops")  # detected -> resolved skips triage
    for to in ("triaged", "work_order", "resolved"):
        store.transition(p.problem_id, to, "ops", now="2026-01-05T00:00:00Z")
    done = store.transition(p.problem_id, "verified", "qa", now="2026-01-06T00:00:00Z")
    assert done.status == "verified" and [h.to_status for h in done.history][-4:] == ["triaged", "work_order", "resolved", "verified"]
    with pytest.raises(ValueError):
        store.transition(p.problem_id, "detected", "ops")  # verified is terminal
    # recurrence reopens, with a note; nothing is deleted
    again = store.merge([_problem(updated="2026-02-01T00:00:00Z")], "2026-02-01T01:00:00Z")
    row = next(x for x in again if x.problem_id == p.problem_id)
    assert row.status == "detected" and row.history[-1].note == "recurred"
    other = store.merge([], "2026-02-02T00:00:00Z")
    assert [x.problem_id for x in other] == [p.problem_id]


def test_verify_refused_when_observations_are_newer(tmp_path):
    store = P.ProblemStore(tmp_path / "problems.json")
    p = _problem(updated="2026-01-10T00:00:00Z")
    store.merge([p], "2026-01-01T00:00:00Z")
    for to in ("triaged", "resolved"):
        store.transition(p.problem_id, to, "ops", now="2026-01-05T00:00:00Z")
    with pytest.raises(ValueError, match="newer"):
        store.transition(p.problem_id, "verified", "qa", now="2026-01-11T00:00:00Z")


# ------------------------------------------------------------------------------------------------ problems end to end

@pytest.fixture(scope="module")
def analyzed(small_tower):
    return P.analyze(small_tower.root)


def test_analyze_runs_end_to_end(small_tower, analyzed):
    data, water, elec, probs = analyzed
    root = small_tower.root
    assert (root / "problems.json").exists() and (root / "analysis" / "findings.json").exists()
    assert (root / "analysis" / "queue.csv").exists() and (root / "analysis" / "observations.jsonl").exists()
    assert P.ProblemStore(root / "problems.json").load() == probs
    assert all(p.synthetic for p in probs)


def test_no_severe_problem_away_from_truth(small_tower, analyzed):
    """SYNTHETIC sanity check, not an accuracy claim: no S3+ problem whose affected nodes miss every truth node."""
    truth_nodes = set()
    for s in load_truth(small_tower.root)["scenarios"]:
        truth_nodes |= set(s["affected_nodes"]) | set(s["root_nodes"])
    for p in analyzed[3]:
        if p.level in ("S3", "S4"):
            assert set(p.affected_nodes) & truth_nodes, p.problem_id


@pytest.mark.xfail(strict=False, reason="needs the water and electrical implementations (stubs until they land)")
def test_scenarios_detected_end_to_end(small_tower, analyzed):
    order = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "U": -1}
    probs = analyzed[3]
    for s in load_truth(small_tower.root)["scenarios"]:
        hits = [p for p in probs if p.root_cause.node_id in s["root_nodes"]]
        assert hits, s["scenario_id"]
        assert order[hits[0].level] >= order[s["expected_level_min"]], (s["scenario_id"], hits[0].level)


@pytest.mark.xfail(strict=False, reason="needs the water and electrical implementations (stubs until they land)")
def test_escalation_clocks_end_to_end(small_tower, analyzed):
    truth = load_truth(small_tower.root)["scenarios"]
    probs = analyzed[3]
    s3 = next(p for p in probs if p.root_cause.node_id in truth[2]["root_nodes"])
    assert "EPA" in s3.escalation.what and "drying window" in s3.escalation.what
    s1 = next(p for p in probs if p.root_cause.node_id in truth[0]["root_nodes"])
    assert s1.escalation is not None and s1.escalation.basis.split(",")[0].strip() in THRESHOLDS
