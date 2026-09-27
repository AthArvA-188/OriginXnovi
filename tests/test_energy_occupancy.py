"""M1/M2 occupancy models, splits and baselines on a 10-day ROBOD fixture; replay and tower simulation invariants."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.building.energy import data as D
from cascade.building.energy import occupancy as O
from cascade.building.energy import simulate as S
from cascade.building.energy.inventory import tower_inventory

FX = Path(__file__).parent / "fixtures" / "energy"
PIR = {"miss_rate": 0.05, "false_trigger_rate": 0.01}


@pytest.fixture(scope="module")
def rooms():
    r = D.load_robod(FX)
    assert set(r) == {2, 5}
    return r


@pytest.fixture(scope="module")
def m1(rooms):
    return O.m1_room_eval(rooms)


def test_no_leakage_features():
    assert not set(O.M1_FEATURES) & set(O.LEAKY)
    assert "pres_now" not in O.M2_DEPLOY and "m1_now" in O.M2_DEPLOY


def test_day_split_is_blocked_and_forward(rooms):
    d = O.add_features(rooms[5])
    tr, te, cut = O.day_split(d)
    assert tr["day"].max() < te["day"].min()
    assert not set(tr["day"]) & set(te["day"])
    assert te["day"].nunique() == 10 - int(10 * 0.7)


def test_m1_eval_deterministic_with_baselines(rooms, m1):
    res, frames = m1
    res2, _ = O.m1_room_eval(rooms)
    for i in res:
        for k in ("majority", "schedule_48", "schedule_168", "m1_hgb"):
            assert k in res[i]["models"]
            assert res[i]["models"][k]["f1"] == res2[i]["models"][k]["f1"]
        vb = res[i]["vs_best_baseline"]
        assert vb["ci95"][0] <= vb["f1_diff"] <= vb["ci95"][1] or vb["n_days"] < 3
        assert "m1_prob" in frames[i] and frames[i]["m1_prob"].between(0, 1).all()


def test_schedule_is_fit_on_train_only(rooms):
    d = O.add_features(rooms[5])
    tr, te, _ = O.day_split(d)
    tab, prior = O.schedule_fit(tr, 168)
    p1 = O.schedule_prob(tab, prior, te, 168)
    te2 = te.copy()
    te2[D.Y] = 1 - te2[D.Y]
    p2 = O.schedule_prob(tab, prior, te2, 168)
    assert np.array_equal(p1, p2)


def test_metrics_and_bootstrap():
    y = np.array([1, 1, 0, 0, 1, 0])
    m = O.clf_metrics(y, np.array([1, 0, 0, 1, 1, 0]))
    assert m["false_vacancy_rate"] == pytest.approx(1 / 3) and m["false_occupied_rate"] == pytest.approx(1 / 3)
    b = O.day_bootstrap_f1_diff(y, y, y, np.array([0, 0, 1, 1, 2, 2]), n_boot=50)
    assert b["f1_diff"] == 0 and b["ci95"] == [0.0, 0.0] and b["verdict"] == "no clear difference"


def test_pir_rates_and_injection():
    ts = pd.date_range("2024-01-01", periods=40, freq="30s")
    d = pd.DataFrame({"ts": ts, "S6_PIR": 0, "S7_PIR": 0, "Room_Occupancy_Count": 0})
    d.loc[:19, "Room_Occupancy_Count"] = 2
    d.loc[:9, "S6_PIR"] = 1  # first occupied 5-min window triggers, second does not
    r = O.pir_rates(d)
    assert r["n_occupied_windows"] == 2 and r["miss_rate"] == pytest.approx(0.5)
    assert r["n_vacant_windows"] == 2 and r["false_trigger_rate"] == 0.0
    occ = np.array([1, 0, 1, 1, 0], bool)
    assert np.array_equal(O.inject_detection(occ, 0.0, 0.0, np.random.default_rng(0)), occ)


def test_replay_invariants(m1):
    _, frames = m1
    out = S.replay_room(frames[5], PIR, seeds=range(3), what_if_miss=(0.3,))
    rec = out["recorded"]["recorded_kwh"]
    for prof, v in out["profiles"].items():
        P = v["policies"]
        assert P["as_operated"]["kwh"] == pytest.approx(rec) and P["as_operated"]["saved_vs_as_operated_pct"] == pytest.approx(0)
        assert P["ideal_sensor"]["underlit_occupied_share"] == 0
        assert P["sensor_ml"]["kwh"] >= P["sensor"]["kwh"] - 1e-9
        assert P["sensor_ml_union"]["kwh"] >= P["sensor_ml"]["kwh"] - 1e-9
        for p in ("schedule", "ideal_sensor", "sensor", "sensor_ml", "sensor_ml_union", "ml_only"):
            assert P[p]["kwh"] <= rec + 1e-9  # a policy can only remove recorded use
            assert P[p]["egress_floor_violations"] == 0


def test_tower_year_small(m1):
    _, frames = m1
    pools = {i: S.day_pool(tf) for i, tf in frames.items()}
    pools = {r: pools[5] for r in range(1, 6)}  # fixture has two rooms; map every room to the library pool
    zones = [z for z in tower_inventory(floors=2)]
    idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz="America/Los_Angeles")
    wx = pd.DataFrame({"temperature_2m": 20 + 10 * np.sin(np.arange(len(idx)) / 500.0)}, index=idx)
    t = S.tower_year(zones, pools, PIR, wx, year=2025, miss_rates={"measured": 0.05, "what_if_0.3": 0.3})
    assert t["kpi"]["egress_floor_violations_all_deployable"] == 0
    # the stair in-use floor (10 fc) is reported separately and is NOT zero by construction
    k = t["kpi"]
    assert k["stair_in_use_fc"] == 10.0
    assert k["stair_in_use_violation_minutes_recommended"] == t["results"]["measured"]["sensor_ml"]["stair_in_use_violation_minutes"]
    assert k["stair_in_use_violation_minutes_always_on"] == 0
    # alternative stair design: the vacant setback meets the in-use floor, so no stair user is ever below it
    alt = t["stair_alternative"]
    assert alt["design_fc"] == pytest.approx(20.0) and alt["current_design_fc"] == pytest.approx(15.0)
    for p in ("sensor", "sensor_ml"):
        assert alt[p]["stair_in_use_violation_minutes"] == 0
        assert alt[p]["extra_kwh_vs_current_design"] > 0
    # pool sizes are reported per mapped room
    assert t["pools"] and all(v["weekday_days"] >= 1 and (v["weekend_days"] >= 1 or v["weekend_borrowed_from_weekday"])
                              for v in t["pools"].values())
    r = t["results"]["measured"]
    assert r["always_on"]["annual_kwh"] >= r["schedule"]["annual_kwh"]
    assert r["sensor_ml"]["annual_kwh"] >= r["sensor"]["annual_kwh"]
    assert len(t["dr_days"]) == 10 and all("2025-06-15" <= d <= "2025-10-15" for d in t["dr_days"])
    assert t["label"] == "SEMI-SYNTHETIC"
