"""Electrical tests (building_spec sections 8 and 13) on the SYNTHETIC demo tower.

The tower in data/demo/building is synthetic, with labelled injected scenarios (S4 loose lug, S5 overloaded circuit).
These checks confirm the injected signals are detected. They are not an accuracy claim. The demo folder is only
read here; when it is missing, a copy is generated into a temporary folder.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.building import electrical as E
from cascade.building.blueprint import load_building_dir
from cascade.building.graph import subtree
from cascade.building.model import WaterResult
from cascade.building.problems import build_problems, default_now
from cascade.building.rules import t
from cascade.building.synthetic import TowerConfig, generate_tower
from cascade.grade import load_rubric

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo" / "building"
ORDER = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = DEMO if (DEMO / "building.json").exists() else None
    if root is None:  # regenerate the default SYNTHETIC tower outside the repo
        root = tmp_path_factory.mktemp("demo_building")
        generate_tower(root, TowerConfig())
    truth_path = root / "ground_truth.json" if (root / "ground_truth.json").exists() else root / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    data = load_building_dir(root)
    now = default_now(data)
    res = E.analyze_electrical(data, now)
    return data, res, {s["scenario_id"]: s for s in truth["scenarios"]}, now


def _by_kind(res, kind: str) -> dict:
    """Findings of one kind keyed by element id."""
    out = {}
    for f in res.findings:
        parts = f.finding_id.split(":", 3)
        if parts[1] == kind:
            out[parts[2]] = f
    return out


# ------------------------------------------------------------------------------------------------ tree

def test_schedule_tree_matches_building_feeds(demo):
    data, _, _, _ = demo
    b = data.building
    assert data.building.synthetic is True
    assert E.check_tree(b, data.schedule) == []
    sched = {(e.src, e.dst) for e in E.schedule_to_edges(data.schedule)}
    feeds = {(e.src, e.dst) for e in b.edges if e.kind == "feeds" and b.element(e.dst).kind in ("electrical_panel", "circuit")}
    assert sched == feeds and all(e.kind == "feeds" for e in E.schedule_to_edges(data.schedule))
    broken = data.schedule.copy()
    broken.loc[0, "fed_from"] = "SWB-9"
    broken = broken.drop(index=1)
    msgs = E.check_tree(b, broken)
    assert any("SWB-9" in m for m in msgs) and any("panel schedule does not" in m for m in msgs)


# ------------------------------------------------------------------------------------------------ S4 loose lug

def test_s4_loose_lug_is_band_3_and_trending(demo):
    data, res, truth, _ = demo
    s4 = truth["S4"]["root_nodes"][0]
    thermal = _by_kind(res, "thermal")
    f = thermal[s4]
    assert f.native_scale.value == "Band 3" and f.unified.level == "S3" and f.modality == "thermal"
    assert f.measurements.delta_t_k is not None and f.measurements.delta_t_k >= t("DT_BANDS_K")[2]
    others = {k: v.native_scale.value for k, v in thermal.items() if k != s4}
    assert others and set(others.values()) <= {"Band 0", "Band 1"}, others
    slope, sd = E.thermal_trend(data.thermal, s4)
    assert slope > 0 and sd >= 0
    assert E.days_to_next_band(f.measurements.delta_t_k, slope) is None  # already at the top band


def test_days_to_next_band_edges():
    lo, mid, hi = t("DT_BANDS_K")
    assert E.days_to_next_band(2.0, 0.5) == pytest.approx((mid - 2.0) / 0.5)
    assert E.days_to_next_band(0.2, 0.1) == pytest.approx((lo - 0.2) / 0.1)
    assert E.days_to_next_band(hi + 1, 1.0) is None
    assert E.days_to_next_band(2.0, 0.0) is None and E.days_to_next_band(2.0, -0.1) is None
    assert E.days_to_next_band(float("nan"), 0.1) is None
    frame = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01", "2026-01-11"], utc=True), "element_id": ["C-X", "C-X"],
                          "t_element_c": [31.0, 32.0], "t_reference_c": [30.0, 30.0], "load_pct": [60.0, 60.0]})
    assert E.thermal_trend(frame, "C-X") is None  # fewer than 3 readings: no trend, not a flat one


# ------------------------------------------------------------------------------------------------ S5 overload

def test_s5_overload_is_load_continuous_over_80(demo):
    data, res, truth, _ = demo
    s5 = truth["S5"]["root_nodes"][0]
    load = _by_kind(res, "load")
    assert load[s5].native_scale.value == "Load continuous over 80%" and load[s5].unified.level == "S2"
    above = {k: v.unified.level for k, v in load.items() if k != s5 and v.unified.level != "S0"}
    assert above == {}, above
    row = res.load_table.set_index("circuit_id").loc[s5]
    assert t("LOAD_CONT_PCT") <= row["max_cont_pct"] < 100 and row["hours_over_cont"] > 0
    sensor = next(s.sensor_id for s in data.building.sensors if s.element_id == s5)
    onset = pd.Timestamp(truth["S5"]["onset_ts"])
    flagged = res.anomalies[res.anomalies["sensor_id"] == sensor]
    assert len(flagged) and (pd.to_datetime(flagged["ts"], utc=True) >= onset).any()
    runs = E.anomaly_runs(res.anomalies)
    long_runs = runs[runs["hours"] >= t("CONTINUOUS_H")]
    assert set(long_runs["sensor_id"]) == {sensor}  # SYNTHETIC tower: only the injected overload persists
    anom_obs = [o for o in res.observations if o.obs_id.startswith("anom:")]
    assert anom_obs and all(o.level is None and o.element_id == s5 for o in anom_obs)  # indicator only, never a level


# ------------------------------------------------------------------------------------------------ problems

def test_electrical_scenarios_become_problems(demo):
    """S4 and S5 through the problem manager (water left empty). SYNTHETIC, not an accuracy claim."""
    data, res, truth, now = demo
    probs = build_problems(data.building, data, WaterResult(), res, now)
    truth_nodes = set()
    for s in truth.values():
        truth_nodes |= set(s["root_nodes"]) | set(s["affected_nodes"])
    for sid in ("S4", "S5"):
        hits = [p for p in probs if p.root_cause.node_id in truth[sid]["root_nodes"]]
        assert hits, (sid, [p.problem_id for p in probs])
        assert hits[0].domain == "electrical"
        assert ORDER[hits[0].level] >= ORDER[truth[sid]["expected_level_min"]], (sid, hits[0].level)
    s4 = next(p for p in probs if p.root_cause.node_id in truth["S4"]["root_nodes"])
    s5 = next(p for p in probs if p.root_cause.node_id in truth["S5"]["root_nodes"])
    assert s4.escalation is not None and "Band 3" in s4.escalation.what
    assert s5.escalation is not None and "overload" in s5.escalation.what and s5.escalation.hours == 0.0
    for p in probs:
        if p.level in ("S3", "S4"):
            assert set(p.affected_nodes) & truth_nodes, p.problem_id


# ------------------------------------------------------------------------------------------------ U guards

def test_missing_rating_or_current_gives_u(demo):
    data, _, _, _ = demo
    b = data.building
    table = pd.DataFrame([
        {"circuit_id": "C-F01-1", "panel_id": "P-F01", "sensor_id": "C-F01-1-A", "rating_a": np.nan, "peak_a": 30.0,
         "p95_a": 25.0, "max_cont_a": 28.0, "max_cont_pct": np.nan, "max_cont_ts": "2026-01-01T12:00:00Z",
         "hours_over_cont": np.nan, "n_hours": 100},
        {"circuit_id": "C-F01-2", "panel_id": "P-F01", "sensor_id": None, "rating_a": 20.0, "peak_a": np.nan,
         "p95_a": np.nan, "max_cont_a": np.nan, "max_cont_pct": np.nan, "max_cont_ts": None, "hours_over_cont": np.nan,
         "n_hours": 0},
    ], columns=E.LOAD_COLUMNS)
    fs = E.load_findings(b, table)
    assert [f.unified.level for f in fs] == ["U", "U"]
    assert all(f.native_scale.value == "U" and "not_measurable" in f.unified.flags for f in fs)
    assert "rating" in fs[0].justification
    # load_table with no current column: unmeasured (NaN), never 0 A
    lt = E.load_table(b, pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")), data.schedule.head(2))
    assert lt["max_cont_pct"].isna().all() and lt["peak_a"].isna().all()
    assert {f.unified.level for f in E.load_findings(b, lt)} == {"U"}
    # U observations are listed (level U), never folded into S0
    obs = E.electrical_observations(fs, E.baseline_anomalies(pd.DataFrame()), b, now=default_now(data))
    assert [o.level for o in obs] == ["U", "U"]


def test_low_load_or_missing_reference_ir_gives_u(demo):
    data, _, _, _ = demo
    b = data.building
    ts = pd.to_datetime(["2026-03-01T10:00:00Z"] * 4, utc=True)
    frame = pd.DataFrame({"ts": ts, "element_id": ["C-F01-1", "C-F01-2", "C-F01-3", "C-F01-4"],
                          "t_element_c": [45.0, 45.0, 36.0, 45.0], "t_reference_c": [30.0, np.nan, 30.0, 30.0],
                          "t_ambient_c": [22.0] * 4, "load_pct": [t("IR_MIN_LOAD_PCT") - 10, 60.0, 60.0, np.nan],
                          "synthetic": [True] * 4})
    got = {f.finding_id.split(":")[2]: f for f in E.classify_thermal(b, frame)}
    for cid in ("C-F01-1", "C-F01-2", "C-F01-4"):
        assert got[cid].unified.level == "U" and got[cid].native_scale.value == "U", cid
        assert got[cid].measurements.delta_t_k is None
    assert "below" in got["C-F01-1"].justification
    assert got["C-F01-3"].native_scale.value == "Band 2" and got["C-F01-3"].measurements.delta_t_k == pytest.approx(6.0)


def test_ir_overdue_and_never_scanned(demo):
    data, _, _, now = demo
    b = data.building
    old = pd.Timestamp(now) - timedelta(days=400)
    circuits = [n for n in subtree(b, "P-F01") if b.element(n).kind == "circuit"]
    frame = pd.DataFrame({"ts": [old] * len(circuits), "element_id": circuits, "t_element_c": [30.5] * len(circuits),
                          "t_reference_c": [30.0] * len(circuits), "t_ambient_c": [22.0] * len(circuits),
                          "load_pct": [60.0] * len(circuits), "synthetic": [True] * len(circuits)})
    recent = frame.assign(ts=pd.Timestamp(now) - timedelta(days=10), element_id=[c.replace("F01", "F03") for c in circuits])
    got = {f.finding_id.split(":")[2]: f for f in E.ir_overdue(b, pd.concat([frame, recent], ignore_index=True), now)}
    assert got["P-F01"].native_scale.value == "IR overdue" and got["P-F01"].unified.level == "S1"
    assert got["P-F02"].unified.level == "U" and "not healthy" in got["P-F02"].justification  # never scanned
    assert "P-F03" not in got  # scanned 10 days ago: within the interval, no row (the rubric has no S0 row here)
    _, res, _, _ = demo
    assert not [f for f in res.findings if f.finding_id.startswith("elec:ir:")]  # the tower is scanned monthly


# ------------------------------------------------------------------------------------------------ contracts

def test_findings_follow_the_rubric_contract(demo):
    _, res, _, _ = demo
    rub = load_rubric("electrical_equipment")
    criteria = {r["criterion"] for r in rub["rows"]}
    rows = {r["value"]: r for r in rub["rows"]}
    assert res.findings
    for f in res.findings:
        assert f.asset_class == "electrical_equipment" and f.modality in ("thermal", "sensor")
        assert f.native_scale.standard == "ELEC-TP" and f.native_scale.value in rub["allowed_values"]
        assert f.model == "deterministic:cascade.building"
        if f.unified.level == "U":
            assert f.native_scale.value == "U" and "not_measurable" in f.unified.flags
            continue
        assert f.native_scale.criteria_matched and set(f.native_scale.criteria_matched) <= criteria
        assert f.unified.level == rows[f.native_scale.value]["unified"]
        assert f.action.code == rows[f.native_scale.value]["action"]
        assert f.action.basis.startswith("electrical_thermal.json row")
    s0 = [f for f in res.findings if f.unified.level == "S0"]
    assert all(f.native_scale.value in ("Band 0", "Load ok") for f in s0)  # U never lands on S0


def test_baseline_anomalies_step_and_flat():
    idx = pd.date_range("2026-01-05", periods=24 * 7 * 6, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    flat = 5.0 + rng.normal(0, 0.05, len(idx))  # well under the ANOMALY_MAD_FLOOR_A spread floor
    step = flat.copy()
    step[-24:] += 6.0
    cur = pd.DataFrame({"A": flat, "B": step}, index=idx)
    an = E.baseline_anomalies(cur)
    assert list(an.columns) == E.ANOMALY_COLUMNS
    assert set(an["sensor_id"]) == {"B"} and len(an) == 24
    assert (pd.to_datetime(an["ts"], utc=True) >= idx[-24]).all() and (an["score"] > t("ANOMALY_MAD_K")).all()
    runs = E.anomaly_runs(an)
    assert runs["hours"].tolist() == [24]
    assert E.baseline_anomalies(cur.iloc[: 24 * 7]).empty  # shorter than the baseline: nothing to score
    nan_cur = cur.copy()
    nan_cur.loc[:, "B"] = np.nan
    assert E.baseline_anomalies(nan_cur).empty  # no data is unmeasured, not an anomaly


def test_drift_skips_h6_for_electrical_delta_t(demo):
    """The H6 skip line in drift.py is owned by the grading step; this checks it from the electrical side."""
    from cascade.drift import _audit_one
    _, res, _, _ = demo
    f = next(f for f in res.findings if f.measurements.delta_t_k is not None)
    hard, _, _ = _audit_one(f, load_rubric("electrical_equipment"))
    assert not [h for h in hard if h[0] == "H6_impossible_measurement"]
