"""Common-area switching policy: code floors, fail-safe holds, ML only adds light, DR skips egress, proposals."""

import json
from pathlib import Path

import numpy as np
import pytest

from cascade.building.energy import rules as R
from cascade.building.energy.inventory import tower_inventory
from cascade.building.energy.policy import (DEPLOYABLE_MODES, MODES, ProposalLog, check_design, extend_only, floor_level,
                                            hold_on, hold_steps, lighting_levels, propose_setting, safety_counts)

EGRESS = ("corridor", "stairwell", "lobby", "garage")
DESIGN_FC = {"corridor": 10.0, "stairwell": 15.0, "lobby": 20.0, "garage": 5.0, "restroom": 15.0, "amenity": 30.0}


def _signals(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    occ = rng.random(n) < 0.3
    det = occ & (rng.random(n) > 0.2)
    ml = rng.random(n) < 0.4
    dr = rng.random(n) < 0.2
    return occ, det, ml, dr


def test_rule_table_has_life_safety_floors_with_sources():
    ls = {r["key"]: r for r in R.load_rules()["life_safety"]}
    assert ls["EGRESS_FLOOR_FC"]["value"] == 1.0
    assert ls["STAIR_IN_USE_FC"]["value"] == 10.0
    assert ls["EGRESS_MOTION_HOLD_MIN"]["value"] >= 15
    assert ls["SENSOR_FAULT_ACTION"]["value"] == "full_on"
    for r in ls.values():
        assert R.source_url(r["source"]).startswith("https://")
    assert R.zone_rule("stairwell").min_in_use_fc == 10.0
    for k in EGRESS:
        assert R.zone_rule(k).egress
    # the rule table lives in the energy package, never in the drift-hashed rubrics folder
    assert R.RULES_PATH.parent.name == "energy"
    rub = Path(R.__file__).resolve().parents[2] / "rubrics"
    assert not any("common_area" in p.name for p in rub.glob("*.json"))


@pytest.mark.parametrize("kind", list(R.ZONE_KINDS))
@pytest.mark.parametrize("mode", list(MODES))
def test_egress_floor_never_violated(kind, mode):
    rule = R.zone_rule(kind)
    occ, det, ml, dr = _signals()
    sched = (np.arange(len(occ)) % 288) // 12 >= 7
    lv = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode=mode, detect=det, ml_on=ml, schedule_on=sched, dr=dr, n=len(occ))
    sc = safety_counts(rule, lv, occ, DESIGN_FC[kind])
    assert sc["life_safety_violations"] == 0
    if rule.egress:
        assert (lv * DESIGN_FC[kind] >= 1.0 - 1e-9).all()
    assert ((lv >= 0) & (lv <= 1)).all()


def test_stair_design_must_meet_10fc_in_use():
    with pytest.raises(ValueError):
        check_design(R.zone_rule("stairwell"), 8.0)
    check_design(R.zone_rule("stairwell"), 10.0)
    for z in tower_inventory():
        check_design(R.zone_rule(z.kind), z.design_fc)  # every inventory zone passes


@pytest.mark.parametrize("kind", EGRESS)
@pytest.mark.parametrize("step_min", [1, 5, 15])
def test_egress_hold_at_least_15_min(kind, step_min):
    rule = R.zone_rule(kind)
    n = 200
    det = np.zeros(n, bool)
    det[10] = True
    lv = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode="sensor", detect=det, step_min=step_min)
    on_after = np.flatnonzero(lv[10:] >= rule.occupied_level - 1e-9)
    held_steps = on_after.max()  # last step still at occupied level, counted after the detection step
    assert held_steps * step_min >= 15
    assert hold_steps(rule, step_min) * step_min >= 15


def test_restroom_off_within_20_min():
    rule = R.zone_rule("restroom")
    det = np.zeros(100, bool)
    det[5] = True
    lv = lighting_levels(rule, design_fc=15.0, mode="sensor", detect=det, step_min=5)
    off = np.flatnonzero(lv[6:] == 0.0)[0] + 1  # steps after the detection step until fully off
    assert off * 5 <= 20
    assert rule.vacant_level == 0.0 and not rule.egress


def test_sensor_fault_goes_full_on():
    for kind in R.ZONE_KINDS:
        rule = R.zone_rule(kind)
        n = 50
        fault = np.zeros(n, bool)
        fault[20:30] = True
        lv = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode="sensor", detect=np.zeros(n, bool), fault=fault)
        assert (lv[20:30] == 1.0).all()


def test_dr_trim_skips_egress_and_respects_floor():
    n = 100
    det = np.ones(n, bool)
    dr = np.zeros(n, bool)
    dr[40:60] = True
    for kind in R.ZONE_KINDS:
        rule = R.zone_rule(kind)
        a = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode="sensor", detect=det)
        b = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode="sensor", detect=det, dr=dr)
        if rule.egress:
            assert np.array_equal(a, b)
        else:
            assert np.allclose(b[40:60], a[40:60] * (1 - R.load_rules()["demand_response"]["min_reduction_frac"]))
            assert np.array_equal(a[:40], b[:40])


def test_ml_only_adds_light_never_removes():
    for seed in range(5):
        occ, det, ml, _ = _signals(seed=seed)
        for kind in R.ZONE_KINDS:
            rule = R.zone_rule(kind)
            s = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode="sensor", detect=det)
            e = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode="sensor_ml", detect=det, ml_on=ml)
            u = lighting_levels(rule, design_fc=DESIGN_FC[kind], mode="sensor_ml_union", detect=det, ml_on=ml)
            assert (e >= s - 1e-12).all() and (u >= e - 1e-12).all()
    assert "ml_only" not in DEPLOYABLE_MODES


def test_extend_only_cannot_switch_on_from_off():
    s = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], bool)
    m = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], bool)
    out = extend_only(s, m)
    assert out.tolist() == [False, False, False, True, True, True, True, False, False]
    assert not extend_only(np.zeros(5, bool), np.ones(5, bool)).any()


def test_hold_on_matches_loop():
    rng = np.random.default_rng(3)
    sig = rng.random(300) < 0.1
    ref, c = [], -10 ** 9
    for t, x in enumerate(sig):
        c = t if x else c
        ref.append(t - c <= 3)
    assert hold_on(sig, 3).tolist() == ref


def test_floor_level_values():
    assert floor_level(R.zone_rule("garage"), 5.0) == pytest.approx(0.2)
    assert floor_level(R.zone_rule("restroom"), 15.0) == 0.0


def test_proposals_are_bounded_and_need_a_human(tmp_path):
    corr = R.zone_rule("corridor")
    with pytest.raises(ValueError):
        propose_setting(corr, zone="F01-CORR", hold_min=10, design_fc=10.0, reason="shorter hold")
    with pytest.raises(ValueError):
        propose_setting(R.zone_rule("garage"), zone="P1", vacant_level=0.1, design_fc=5.0, reason="too dark")
    with pytest.raises(ValueError):
        propose_setting(corr, zone="exit signs F01", vacant_level=0.5, design_fc=10.0, reason="x")
    with pytest.raises(ValueError):
        propose_setting(R.zone_rule("restroom"), zone="WC", hold_min=30, design_fc=15.0, reason="too long")
    p = propose_setting(corr, zone="F01-CORR", vacant_level=0.5, hold_min=20, design_fc=10.0, reason="ok")
    assert p.status == "pending_approval" and p.citations
    log = ProposalLog(tmp_path / "proposals.jsonl")
    log.add(p)
    with pytest.raises(ValueError):
        log.transition(p.id, "approved", reviewer=" ")
    out = log.transition(p.id, "approved", reviewer="chief engineer")
    assert out["status"] == "approved"
    with pytest.raises(ValueError):
        log.transition(p.id, "rejected", reviewer="someone")
    lines = [json.loads(x) for x in (tmp_path / "proposals.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["event"] for x in lines] == ["add", "transition"]


def test_stair_setback_meets_in_use_floor_only_at_20fc_design():
    """At 15 fc design the 50% setback is 7.5 fc (< 10 fc in use); at 20 fc it is exactly 10 fc."""
    r = R.zone_rule("stairwell")
    occ = np.array([1, 1, 1, 1], bool)
    missed = np.zeros(4, bool)  # the sensor misses the person on the stairs
    for fc, expect in ((15.0, 4), (20.0, 0)):
        lv = lighting_levels(r, design_fc=fc, mode="sensor", detect=missed, n=4)
        assert safety_counts(r, lv, occ, fc)["in_use_floor_violations"] == expect
