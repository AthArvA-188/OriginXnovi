"""Rule-conformance tests for the fire de-escalation planner (cascade.building.fire) on the SYNTHETIC tower.

The planner has no accuracy metric: it is verified by these rule checks on every floor x zone of the 32-floor tower
and on a 10-floor tower, plus escalation, low-floor, drawing and round-trip cases.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cascade.building import fire
from cascade.building.synthetic import TowerConfig, build_tower_model

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("room", "core", "shaft", "electrical_room")


@pytest.fixture(scope="module")
def tower():
    b = build_tower_model(TowerConfig())
    return b, fire.synthetic_fire_layer(b)


@pytest.fixture(scope="module")
def small():
    b = build_tower_model(TowerConfig(floors=10, days=10))
    return b, fire.synthetic_fire_layer(b)


def _zones(b):
    return [z for f in fire.occupied_floors(b) for z in b.zones_on(f.floor_id) if z.kind in KINDS]


@pytest.mark.parametrize("which", ["tower", "small"])
def test_every_zone_card_passes_rules(which, request):
    b, layer = request.getfixturevalue(which)
    zones = _zones(b)
    assert len(zones) == 7 * b.floors
    bad = {z.zone_id: fire.check_card(b, layer, fire.plan_for_incident(b, layer, z.zone_id)) for z in zones}
    assert {k: v for k, v in bad.items() if v} == {}


def test_card_f19_east(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F19-E")
    assert c.alert_floors == [18, 19, 20]
    assert c.staging_floor == 17
    assert c.relocation_floors == [16, 15, 14]
    assert c.smoke_watch_floors == [21, 22, 23, 24]
    assert (c.evac_stair, c.attack_stair) == ("A", "B")  # east office: attack from the east stair, evacuate west
    assert c.standpipe_landing == 18 and c.fd_lift_exit_floor == 17
    assert c.passenger_banks == ["HIGH"]
    assert c.panel == "P-F19" and c.feeder_path == ["P-F19", "SWB-1"]
    assert c.zone_circuits == ["C-F19-2"]
    assert set(c.risers) == {"R1@F19", "L1@F19"}
    assert c.elec_reached and all(n.endswith("-ELEC") for n in c.elec_reached)
    assert c.occupants["total_to_relocate"] == 3 * layer.occupants["F19"]["design_load"]
    assert c.occupants["assistance_count"] == 2
    assert c.synthetic is True and "DECISION SUPPORT ONLY" in c.banner


def test_west_zone_swaps_stairs(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F10-W")
    assert (c.evac_stair, c.attack_stair) == ("B", "A")


def test_stairs_separated_and_inside_offices(tower):
    b, layer = tower
    a, bb = layer.stair("A"), layer.stair("B")
    assert fire.stair_separation_m(a, bb) >= fire.ft("STAIR_MIN_SEP_FT") * fire.M_PER_FT
    assert (a.zone_side, bb.zone_side) == ("W", "E")


def test_design_load_from_plate_area(tower):
    b, layer = tower
    area = sum(z.area_m2 for z in b.zones_on("F05"))
    assert area == pytest.approx(1200, rel=0.01)
    assert layer.occupants["F05"]["design_load"] == int(area * fire.SQFT_PER_M2 // 150)


def test_escalation_upward_spread_and_impairment_first(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F19-E", detections=["F23"])
    assert set(range(18, 25)) <= set(c.alert_floors)
    status = {t.id: t.status for t in c.triggers}
    assert status["E1"] == "fired" and status["E2"] == "fired" and status["E4"] == "fired"
    assert c.hazards[0].kind == "impairment" and c.hazards[0].floor_ids == ["F23"]
    assert not set(c.relocation_floors) & set(c.alert_floors)


def test_impairment_on_alert_floor_is_first_hazard(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F23-N")  # F22-F24 alerted, F23 impaired
    assert c.hazards[0].kind == "impairment" and "ALERT OR RELOCATION FLOOR" in c.hazards[0].text


def test_low_floors(tower):
    b, layer = tower
    c2 = fire.plan_for_incident(b, layer, "F02-N")
    assert c2.staging_floor is None and c2.relocation_floors == [] and "outside assembly" in c2.relocation_note
    assert c2.fd_lift_exit_floor is None
    assert any("Stairs expected" in ln.item for ln in c2.lines)
    c1 = fire.plan_for_incident(b, layer, "F01-CORE")
    assert c1.alert_floors == [1, 2] and c1.standpipe_landing == 1
    ess = [h for h in c1.hazards if h.kind == "ess"]
    assert ess and ess[0].near


def test_ess_near_only_within_two_floors(tower):
    b, layer = tower
    for n in range(1, 33):
        c = fire.plan_for_incident(b, layer, f"F{n:02d}-S")
        ess = [h for h in c.hazards if h.kind == "ess"]
        assert len(ess) == 1 and ess[0].near == (n <= 3)


def test_nothing_executes_and_humans_decide(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F27-N")
    assert all(not ln.executes for ln in c.lines + c.automatic_sequence)
    assert {d.owner for d in c.decisions} <= {"FSD", "IC"}
    assert all(d.ai_role == "evidence only" for d in c.decisions)
    rec = [a for a in c.automatic_sequence if a.item.startswith("[ ] A2")]
    assert rec and "lobby or hoistway detector" in rec[0].detail


def test_water_watch_below_and_ranked(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F19-E")
    assert c.water_watch and all(w.level < 19 for w in c.water_watch)
    s = [w.score for w in c.water_watch]
    assert s == sorted(s, reverse=True)
    assert min(s) >= 0.05


def test_roof_zone_rejected(tower):
    b, layer = tower
    with pytest.raises(ValueError):
        fire.plan_for_incident(b, layer, "ROOF-NE")


def test_json_round_trips(tower, tmp_path):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F19-E", detections=["F22"])
    assert fire.PlanCard.model_validate_json(c.model_dump_json()) == c
    p = fire.save_fire_layer(layer, tmp_path)
    assert fire.load_fire_layer(p) == layer


def test_rule_table_has_basis_for_every_threshold():
    rules = fire.load_rules()
    ids = {s["n"] for s in rules["sources"]}
    for th in fire.fire_thresholds().values():
        assert th.tag in ("PUBLIC", "PUBLIC, secondary", "team-proposed, validate")
        if th.tag.startswith("PUBLIC"):
            assert th.url and th.url.startswith("https://")
    for sec in ("automatic_sequence", "decisions", "triggers", "first_interstate"):
        for row in rules[sec]:
            assert set(row["src"]) <= ids
    assert all(s["url"].startswith("https://") and s["accessed"] for s in rules["sources"])


def test_svgs_are_wellformed_and_labelled(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F19-E")
    for svg, must in ((fire.svg_section(c, layer), ("DECISION SUPPORT ONLY", "SYNTHETIC", "Stair A", "Stair B")),
                      (fire.svg_flowchart(c), ("evidence only", "D1 FSD calls 911", "D4 IC widens zone")),
                      (fire.svg_swimlane(c), ("evidence only", "Fire department IC", "no clock times"))):
        ET.fromstring(svg)
        assert "<script" not in svg.lower()
        for m in must:
            assert m in svg


def test_markdown_card(tower):
    b, layer = tower
    md = fire.plan_card_markdown(fire.plan_for_incident(b, layer, "F19-E"))
    assert "DECISION SUPPORT ONLY" in md and "SYNTHETIC" in md and "| Item |" in md


def test_artifacts_match_code(tower):
    b, layer = tower
    p = ROOT / "eval" / "fire" / "fire_layer.json"
    if not p.exists():
        pytest.skip("run scripts/fire_build_plans.py")
    assert fire.load_fire_layer(p) == layer
    chk = json.loads((ROOT / "eval" / "fire" / "planner_checks.json").read_text(encoding="utf-8"))
    assert chk["cards_checked"] == chk["cards_passing"] == 224 and chk["synthetic"] is True


def test_escalated_staging_moves_below_widened_alert_and_is_tagged(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F07-E", detections=["F06"])  # alert F05-F08 covers the sourced staging floor F05
    assert c.alert_floors == [5, 6, 7, 8] and c.staging_floor == 4
    ln = next(x for x in c.lines if x.item.startswith("Staging floor"))
    assert fire.STAGING_MOVED_TAG in ln.detail and fire.STAGING_MOVED_TAG in ln.basis
    assert fire.check_card(b, layer, c) == []
    c2 = fire.plan_for_incident(b, layer, "F16-E", detections=["F01", "F32"])  # whole tower alerted: stage outside
    assert c2.staging_floor is None
    ln2 = next(x for x in c2.lines if x.item.startswith("Staging floor"))
    assert "exterior" in ln2.item and fire.STAGING_MOVED_TAG in ln2.basis
    assert fire.check_card(b, layer, c2) == []
    plain = fire.plan_for_incident(b, layer, "F19-E", detections=["F22"])  # alarm above: staging stays at the sourced floor
    ln3 = next(x for x in plain.lines if x.item.startswith("Staging floor"))
    assert plain.staging_floor == 17 and fire.STAGING_MOVED_TAG not in ln3.basis


def test_check_card_catches_untagged_staging_move(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F07-E", detections=["F06"])
    lines = [x.model_copy(update={"basis": fire.basis("FIRE_STAGING_BELOW"), "detail": "staging at least two floors below the fire"})
             if x.item.startswith("Staging floor") else x for x in c.lines]
    bad = fire.check_card(b, layer, c.model_copy(update={"lines": lines}))
    assert any("team-proposed" in v for v in bad)
    bad2 = fire.check_card(b, layer, c.model_copy(update={"staging_floor": 6}))
    assert any("staging" in v for v in bad2)


def test_every_escalated_card_passes_rules(small):
    b, layer = small
    floors = [f.floor_id for f in fire.occupied_floors(b)]
    bad = {}
    for f in floors:
        for g in floors:
            if g != f:
                c = fire.plan_for_incident(b, layer, f"{f}-E", detections=[g])
                v = fire.check_card(b, layer, c)
                if v:
                    bad[f"{f}+{g}"] = v
    assert bad == {}


def test_unbuilt_ai_items_are_marked_in_drawings_and_rules(tower):
    b, layer = tower
    c = fire.plan_for_incident(b, layer, "F19-E")
    flow, swim = fire.svg_flowchart(c), fire.svg_swimlane(c)
    assert "camera near the device: not built" in flow and "did not transfer to a new room" in flow
    assert "timeline: not built" in flow and "after-action timeline (not built)" in swim
    phases = fire.load_rules()["phases"]
    assert all(ph.get("built") for ph in phases)
    assert "not built" in phases[1]["built"] and "did not transfer" in phases[1]["built"]
    assert "not built" in phases[6]["built"]


def test_escalated_artifact_counts(tower):
    p = ROOT / "eval" / "fire" / "planner_checks.json"
    if not p.exists():
        pytest.skip("run scripts/fire_build_plans.py")
    chk = json.loads(p.read_text(encoding="utf-8"))
    n = chk["tower_config"]["floors"]
    assert chk["escalated_cards_checked"] == chk["escalated_cards_passing"] == n * (n - 1)
    assert 0 < chk["escalated_staging_moved"] < chk["escalated_cards_checked"]
