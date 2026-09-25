"""Drift prevention and validation loops (src/cascade/drift.py): fake runs, no model, no network."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from cascade import drift
from cascade.costlog import usd_for
from cascade.export import bridge_row
from cascade.grade import load_rubric
from cascade.ingest import read_manifest, sha256_of, write_manifest
from cascade.pipeline import RunConfig, load_run, run_cascade
from cascade.review import SCHEMA, ReviewLog
from cascade.schema import Action, Evidence, Finding, GateRecord, ImageRecord, Measurements, NativeScale, Unified, unassessable_finding
from cascade.surge import surge_counts
from test_pipeline import fake_gate, fake_grade, make_records

DEV_MANIFEST = drift.ROOT / "data" / "dev" / "manifest.jsonl"
UI_RUN = drift.ROOT / "runs" / "ui_0925_0856"


def contract_finding(rubric, value, level=None, fid=None, asset_class="steel_coating", **over):
    """A finding that copies a real rubric row verbatim (criterion, unified level, action, flag)."""
    row = next(r for r in rubric["rows"] if r["value"] == value)
    kw = dict(
        finding_id=fid or f"{value.lower()}/full",
        asset_class=asset_class,
        defect_type="corrosion",
        native_scale=NativeScale(standard=rubric["standard"], value=value, criteria_matched=[row["criterion"]]),
        unified=Unified(level=level or row["unified"], uncertainty="+/-1", flags=[row["flag"]] if row.get("flag") else []),
        measurements=Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=0.7),
        action=Action(code=row["action"], sla_days=90, basis="row"),
        justification="Visible rust.",
        evidence=Evidence(image_ids=[(fid or value).split("/")[0]], bbox=[0, 0, 8, 8], tile="full"),
        model="fake-grader",
    )
    kw.update(over)
    return Finding(**kw)


def route_all(img, image_id, *, backend, log, no_damage_min_conf):
    log.record(stage="gate", model="fake-gate", image_id=image_id, input_tokens=10, output_tokens=5, seconds=0.01)
    return GateRecord(image_id=image_id, usable=True, damage_present=True, confidence=0.9, reason="rust", routed=True, model="fake-gate", seconds=0.01, usd=0.0)


def real_sha(recs):
    return [r.model_copy(update={"sha256": sha256_of(Path(r.path))}) for r in recs]


CORR = load_rubric("steel_coating")
BRIDGE = load_rubric("bridge_element")


# ---------------------------------------------------------------------------------------------- M1

def test_fingerprint_is_stable_and_changes_with_env_and_config():
    env = {"GRADER_MODEL": "claude-opus-5"}
    a = drift.fingerprint(RunConfig(gate="none"), env=env, ollama=False)
    b = drift.fingerprint(RunConfig(gate="none"), env=env, ollama=False)
    assert a["id"] == b["id"] and len(a["id"]) == 12
    assert drift.fingerprint_diff(a, b) == []
    c = drift.fingerprint(RunConfig(gate="none"), env={"GRADER_MODEL": "claude-sonnet-5"}, ollama=False)
    assert c["id"] != a["id"] and [d[0] for d in drift.fingerprint_diff(a, c)] == ["model.grader"]
    d = drift.fingerprint(RunConfig(gate="none", gate_min_conf=0.9), env=env, ollama=False)
    assert d["id"] != a["id"] and drift.fingerprint_diff(a, d)[0][0] == "config"
    assert a["components"]["model.effort"] == "high" and a["components"]["crop.max_side"] == 1568


def test_run_cascade_writes_fingerprint_and_summary_carries_it(tmp_path):
    out = tmp_path / "run"
    summary = run_cascade(make_records(tmp_path), out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    fp = json.loads((out / "fingerprint.json").read_text(encoding="utf-8"))
    assert summary["fingerprint"] == fp["id"]


def test_check_models_detects_mixed_graders():
    run = {"calls": [{"stage": "grade", "model": "claude-opus-5"}, {"stage": "grade", "model": "claude-opus-5-5"}, {"stage": "gate", "model": "claude-haiku-4-5", "served_model": "claude-haiku-4-5-20251001"}], "findings": []}
    m = drift.check_models(run, None)
    assert m["mixed"] is True and m["models_per_stage"]["grade"] == ["claude-opus-5", "claude-opus-5-5"]
    assert m["served_mismatch"] == [] and m["matches_fingerprint"] is None  # a dated snapshot of the alias is how the API resolves it
    assert m["served_by_model"] == {"claude-haiku-4-5": ["claude-haiku-4-5-20251001"]} and m["served_multiple"] == {}


def test_served_snapshot_rules_alias_swap_and_pin():
    calls = [{"stage": "gate", "model": "claude-haiku-4-5", "served_model": "claude-haiku-4-5-20251001"}, {"stage": "gate", "model": "claude-haiku-4-5", "served_model": "claude-haiku-4-5-20260301"},
             {"stage": "grade", "model": "claude-opus-5", "served_model": "claude-sonnet-5-20250929"}]
    m = drift.check_models({"calls": calls, "findings": []}, None, pinned_served={"claude-haiku-4-5": "claude-haiku-4-5-20251001"})
    assert [r["model"] for r in m["served_mismatch"]] == ["claude-opus-5"]  # a different family is a mismatch
    assert m["served_multiple"] == {"claude-haiku-4-5": ["claude-haiku-4-5-20251001", "claude-haiku-4-5-20260301"]}  # alias re-pointed mid-run
    assert m["served_changed"]["claude-haiku-4-5"][0] == "claude-haiku-4-5-20251001"  # differs from the baseline pin
    assert drift.first_served(calls) == {"claude-haiku-4-5": "claude-haiku-4-5-20251001", "claude-opus-5": "claude-sonnet-5-20250929"}


def test_fingerprint_ignores_git_head_and_missing_digests_and_splits_model_id(monkeypatch):
    env = {"GRADER_MODEL": "claude-opus-5"}
    monkeypatch.setattr(drift, "_git_head", lambda: "commit_A")
    a = drift.fingerprint(RunConfig(gate="none"), env=env, ollama=False)
    monkeypatch.setattr(drift, "_git_head", lambda: "commit_B")
    b = drift.fingerprint(RunConfig(gate="none"), env=env, ollama=False)
    assert a["id"] == b["id"] and [d[0] for d in drift.fingerprint_diff(a, b)] == ["git.head"]  # kept for diffs, not hashed
    monkeypatch.setattr(drift, "_ollama_digests", lambda tags, env: {t: "sha256:abc" for t in tags})
    c = drift.fingerprint(RunConfig(gate="local"), env=env, ollama=True)
    monkeypatch.setattr(drift, "_ollama_digests", lambda tags, env: {t: None for t in tags})
    d = drift.fingerprint(RunConfig(gate="local"), env=env, ollama=True)
    assert c["id"] != d["id"] and d["id"] == drift.fingerprint(RunConfig(gate="local"), env=env, ollama=False)["id"]  # None digest = not hashed
    # routing config changes the full id but not the model identity; the grader backend changes both
    e = drift.fingerprint(RunConfig(gate="none", force_route_classes=()), env=env, ollama=False)
    assert e["id"] != a["id"] and e["model_id"] == a["model_id"]
    f = drift.fingerprint(RunConfig(gate="none", grader="local"), env=env, ollama=False)
    assert f["model_id"] != a["model_id"] and drift.identity_of(a) == a["model_id"] and drift.identity_of({"id": "x"}) == "x"
    assert {"prompt.grader_user", "grade.max_tokens", "crop.tile_overlap", "crop.upscale_min_side"} <= set(a["components"]) and a["components"]["crop.jpeg_quality"] == 90


# ---------------------------------------------------------------------------------------------- M2

def test_contract_audit_clean_on_row_copies():
    findings = [contract_finding(CORR, v, fid=f"c{i}/full") for i, v in enumerate(CORR["allowed_values"])]
    findings += [contract_finding(BRIDGE, "CS3", fid="b1/full", asset_class="bridge_element"), contract_finding(BRIDGE, "CS4", level="S4", fid="b2/full", asset_class="bridge_element", action=Action(code="escalate", sla_days=0, basis="row"))]
    a = drift.contract_audit(findings, lambda ac: load_rubric(ac), [])
    assert a["n"] == 6 and a["hard_total"] == 0 and a["hard_rate"] == 0.0
    assert all(v == 0 for v in a["hard"].values())


def test_contract_audit_flags_paraphrase_level_action_measurement_defaulted_s0_mixed_u():
    f = [
        contract_finding(CORR, "Poor", fid="h1/full", native_scale=NativeScale(standard="CorrosionCS", value="Bogus", criteria_matched=["x"])),
        contract_finding(CORR, "Poor", fid="h2/full", native_scale=NativeScale(standard="CorrosionCS", value="Poor", criteria_matched=["corrosion that looks deeper than the surface"])),
        contract_finding(CORR, "Poor", fid="h3/full", native_scale=NativeScale(standard="CorrosionCS", value="Poor", criteria_matched=[])),
        contract_finding(CORR, "Poor", fid="h4/full", level="S0"),
        contract_finding(CORR, "Poor", fid="h5/full", action=Action(code="record", sla_days=None, basis="row")),
        contract_finding(CORR, "Poor", fid="h6/full", measurements=Measurements(area_cm2=None, crack_width_mm=0.3, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=0.7)),
        contract_finding(CORR, "Good", fid="h7/full", unified=Unified(level="S0", uncertainty="+/-1", flags=["not_measurable"]), native_scale=NativeScale(standard="CorrosionCS", value="Good", criteria_matched=[])),
        contract_finding(CORR, "Good", fid="w4/full", unified=Unified(level="U", uncertainty="+/-1", flags=["not_measurable"]), native_scale=NativeScale(standard="CorrosionCS", value="Good", criteria_matched=[])),
    ]
    a = drift.contract_audit(f, lambda ac: CORR, [])
    hits = {(v["finding_id"], v["rule"]) for v in a["violations"]}
    for fid, rule in [("h1/full", "H1_value_not_allowed"), ("h2/full", "H2_criteria_not_verbatim"), ("h3/full", "H3_criteria_empty"), ("h4/full", "H4_level_row_mismatch"),
                      ("h5/full", "H5_action_row_mismatch"), ("h6/full", "H6_impossible_measurement"), ("h7/full", "H7_defaulted_s0"), ("w4/full", "W4_mixed_u")]:
        assert (fid, rule) in hits, (fid, rule)
    assert a["findings_breached"] == 7 and a["soft"]["W4_mixed_u"] == 1
    assert not any(r.startswith("H") for fid, r in hits if fid == "w4/full")  # U findings are never a hard breach
    sig = drift.contract_signals(a)
    assert sig[0]["rule"] == "contract_hard" and sig[0]["severity"] == "alarm" and sig[0]["n"] == 8


def test_contract_audit_existing_fake_grade_is_flagged(tmp_path):
    out = tmp_path / "run"
    run_cascade(make_records(tmp_path), out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    run = load_run(out)
    a = drift.contract_audit(run["findings"], lambda ac: load_rubric(ac), run["calls"])
    rules = {v["rule"] for v in a["violations"] if v["finding_id"] == "img_2/full"}
    assert {"H4_level_row_mismatch", "H5_action_row_mismatch"} <= rules  # Severe row is S3/prioritize; fake says S2/schedule
    assert a["u_by_cause"]["gate_unusable"] == 1 and a["grade_calls"] == 2


def test_u_causes_from_justification():
    ev = Evidence(image_ids=["x"])
    mk = lambda reason: unassessable_finding(finding_id="x/full", asset_class="steel_coating", standard="CorrosionCS", evidence=ev, reason=reason)
    assert drift.u_cause(mk("gate marked image unusable: blur")) == "gate_unusable"
    assert drift.u_cause(mk("grader returned no contract (refusal)")) == "refusal"
    assert drift.u_cause(mk("grader returned no contract (parse_error: bad json)")) == "parse_error"
    model_u = contract_finding(CORR, "Good", fid="m/full", unified=Unified(level="U", uncertainty="+/-1", flags=["not_measurable"]))
    assert drift.u_cause(model_u) == "model_u"
    b = drift.u_breakdown([mk("gate marked image unusable: blur"), model_u, contract_finding(CORR, "Fair")], [])
    assert b["n"] == 3 and b["u"] == 2 and b["by_cause"]["model_u"]["count"] == 1


# ---------------------------------------------------------------------------------------------- M3 / M11

def test_p_limits_worked_example():
    assert abs(drift.p_limits(4 / 14, 14)["ucl"] - 0.648) < 0.002
    assert abs(drift.p_limits(4 / 14, 50)["ucl"] - 0.477) < 0.002
    assert drift.p_limits(0.0, 10)["lcl"] == 0.0


def test_min_n_for_shift():
    assert 290 <= drift.min_n_for_shift(0.10, 0.20) <= 305
    assert 68 <= drift.min_n_for_shift(0.79, 0.60, alpha=0.05, one_sided=True) <= 76
    assert abs(drift.z_quantile(0.975) - 1.96) < 0.001


def test_binom_tail_and_alarm_line():
    assert abs(drift.binom_tail(4, 12, 1 / 12) - 0.014) < 0.002 and abs(drift.binom_tail(3, 12, 1 / 12) - 0.072) < 0.002
    assert drift.alarm_line(12, 1 / 12) == 4 and drift.alarm_line(12, 0.0) == 1


# ---------------------------------------------------------------------------------------------- M4

def _two_runs(tmp_path, n):
    tmp_path.mkdir(parents=True, exist_ok=True)
    recs = real_sha(make_records(tmp_path, n=n))
    u = lambda fid, ev: unassessable_finding(finding_id=fid, asset_class="steel_coating", standard="CorrosionCS", evidence=ev, reason="not measurable")

    def grade_a(img, *, finding_id, evidence, **kw):
        return u(finding_id, evidence)

    def grade_b(img, *, finding_id, image_id, evidence, **kw):
        idx = int(image_id.split("_")[-1])
        return contract_finding(CORR, "Good", fid=finding_id, evidence=evidence) if idx < 6 else u(finding_id, evidence)

    dirs = []
    for name, fn in (("a", grade_a), ("b", grade_b)):
        out = tmp_path / name
        write_manifest(recs, out / "manifest.jsonl")
        run_cascade(recs, out, RunConfig(gate="none"), gate_fn=route_all, grade_fn=fn)
        dirs.append(out)
    return dirs


def test_output_shift_refuses_below_20_shared_and_detects_abstention_collapse(tmp_path):
    a, b = _two_runs(tmp_path / "small", 10)
    assert drift.output_shift(a, b)["status"] == "not_comparable"
    a, b = _two_runs(tmp_path / "big", 25)
    s = drift.output_shift(a, b)
    assert s["status"] == "ok" and s["n_shared"] == 25
    c = s["per_class"]["steel_coating"]
    assert abs(c["u_rate_delta"] + 0.24) < 1e-9 and abs(c["s0_share_delta"] - 0.24) < 1e-9 and c["moved_2plus"] == []
    assert s["abstention_collapse"] is True and any(x["rule"] == "abstention_collapse" and x["severity"] == "alarm" for x in s["signals"])


def test_level_shift_counts_two_level_moves_and_u_agreement():
    a = {"i1": "S1", "i2": "U", "i3": "S0", "i4": "S4"}
    b = {"i1": "S3", "i2": "U", "i3": "S1", "i4": "S4"}
    r = drift.level_shift(a, b, {k: "bridge_element" for k in a})
    c = r["per_class"]["bridge_element"]
    assert c["moved_2plus"] == ["i1"] and c["exact"] == 0.5 and c["within_one"] == 0.75


# ---------------------------------------------------------------------------------------------- M5

def test_gate_health_dead_gate_template_collapse_and_forced_not_counted():
    rows = [{"image_id": f"ir_{i}", "usable": True, "damage_present": False, "confidence": 0.95, "reason": "Uniform thermal pattern, no anomaly. [forced: asset class pv_module]", "routed": True, "model": "qwen3-vl:4b-instruct", "seconds": 5.0, "usd": 0.0} for i in range(12)]
    recs = {f"ir_{i}": ImageRecord(image_id=f"ir_{i}", path="x", sha256="0" * 64, width=40, height=24, asset_class="pv_module", source_dataset="ir_solar", labels={"damage_present": True}) for i in range(12)}
    g = drift.gate_health(rows, recs, [], 0.7)
    grp = g["groups"]["ir_solar"]
    assert grp["routed"] == 12 and grp["routed_by_gate"] == 0 and grp["forced"] == 12 and grp["recall_by_gate"] == 0.0
    assert grp["unique_reason_ratio"] < 0.5
    rules = {s["rule"]: s["severity"] for s in g["signals"]}
    assert rules["dead_gate"] == "alarm" and rules["template_collapse"] == "watch"
    assert g["eval"]["gate"]["recall"] == 0.0 and g["min_conf_inert"] is False  # 12 rows < 30


# ---------------------------------------------------------------------------------------------- M7

def test_ops_cold_start_excluded_and_unpriced_cloud_call():
    calls = [{"stage": "grade", "model": "claude-opus-5", "image_id": f"i{i}", "input_tokens": 5000, "output_tokens": 900, "usd": 0.05, "seconds": s} for i, s in enumerate([35.0, 5.0, 5.0, 6.0, 5.0])]
    calls.append({"stage": "gate", "model": "claude-unknown", "image_id": "i0", "input_tokens": 100, "output_tokens": 10, "usd": 0.0, "seconds": 1.0})
    calls.append({"stage": "gate", "model": "fake-gate", "image_id": "i1", "input_tokens": 100, "output_tokens": 10, "usd": 0.0, "seconds": 1.0})
    o = drift.ops_health(calls)
    g = o["by_stage_model"]["grade/claude-opus-5"]
    assert g["median_seconds"] == 5.0 and g["cold_start_excluded"] == "i0" and g["calls"] == 5
    assert [c["model"] for c in o["unpriced"]] == ["claude-unknown"]
    assert o["signals"][0]["rule"] == "unpriced" and o["signals"][0]["severity"] == "alarm"
    o2 = drift.ops_health(calls[:5], baseline={"grade_median_output_tokens": 500, "grade_median_seconds": 2.0})
    assert {s["rule"] for s in o2["signals"]} == {"verbosity", "latency"}


def test_costlog_longest_prefix_prices_dated_id():
    assert usd_for("claude-haiku-4-5-20251001", 1000, 100) > 0
    assert usd_for("claude-opus-5-5", 1000, 0) == 1000 * 4.0 / 1e6  # longest key wins over claude-opus-5
    assert usd_for("claude-unknown", 1000, 100) == 0.0 and usd_for("qwen3-vl:4b-instruct", 1000, 100) == 0.0


# ---------------------------------------------------------------------------------------------- M8

def test_input_shift_video_gap_and_null_gsd_confounder():
    recs = [ImageRecord(image_id=f"f{i}", path="x", sha256=f"{i}" * 64, width=1920, height=1080, asset_class="bridge_element", source_video="a.mp4", frame_time_s=t) for i, t in enumerate([0.0, 0.4, 2.0])]
    s = drift.input_shift(recs, baseline={"null_gsd_share": 0.0})
    assert s["video"]["a.mp4"] == {"frames": 3, "min_gap_s": 0.4}
    assert s["null_gsd_share"] == 1.0 and s["not_comparable_reason"]
    assert {x["rule"] for x in s["signals"]} == {"frame_gap", "input_not_comparable"}
    assert drift.input_shift([], None)["n"] == 0


# ---------------------------------------------------------------------------------------------- M6

def _row(action, prior, new, reviewer="r1", fid=None, blind=0):
    return {"finding_id": fid or f"{prior}-{new}-{reviewer}", "action": action, "prior_level": prior, "new_level": new, "reviewer": reviewer, "blind": blind}


def test_review_health_direction_critical_miss_and_stratification():
    rows = [_row("overridden", "S1", "S2", fid=f"u{i}") for i in range(6)] + [_row("overridden", "S2", "S1", fid=f"d{i}") for i in range(4)]
    rows.append(_row("overridden", "S1", "S4", fid="miss"))
    r = drift.review_health(rows, [])
    assert r["overrides"]["n"] == 11 and r["overrides"]["up"] == 7 and [m["finding_id"] for m in r["critical_misses"]] == ["miss"]
    assert r["big_overrides"][0]["finding_id"] == "miss"
    rules = [s["rule"] for s in r["signals"]]
    assert rules[0] == "critical_miss" and "under_grading" in rules and r["attribution"] == "insufficient_n"
    # one reviewer at 40 percent against another at 90 percent, 20 decisions each -> reviewer calibration, not the model
    rows2 = [_row("accepted" if i < 8 else "overridden", "S1", "S1" if i < 8 else "S2", reviewer="a", fid=f"a{i}") for i in range(20)]
    rows2 += [_row("accepted" if i < 18 else "overridden", "S1", "S1" if i < 18 else "S2", reviewer="b", fid=f"b{i}") for i in range(20)]
    r2 = drift.review_health(rows2, [])
    assert r2["by_reviewer"]["a"]["agreement"] == 0.4 and r2["by_reviewer"]["b"]["agreement"] == 0.9 and r2["attribution"] == "reviewer"
    # blind pairs: 12 blind grades, only 5 within one of the sighted decision -> below the FHWA floor
    rows3 = [_row("accepted", "S1", "S1", fid=f"p{i}") for i in range(12)] + [_row("blind", "S1", "S1" if i < 5 else "S4", fid=f"p{i}", blind=1) for i in range(12)]
    r3 = drift.review_health(rows3, [])
    assert r3["blind"]["n"] == 12 and abs(r3["blind"]["within_one"] - 5 / 12) < 1e-9 and any(s["rule"] == "blind_floor" for s in r3["signals"])
    assert r3["n"] == 12  # blind rows never count as sighted decisions


def test_review_migrate_adds_columns_to_old_db_and_record_blind_does_not_mutate(tmp_path):
    db = tmp_path / "reviews.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO reviews(run_id, finding_id, action, prior_level, new_level, reviewer, reviewed_at) VALUES ('run','old/full','accepted','S1','S1','r1','2026-09-25T00:00:00+00:00')")
    conn.commit()
    conn.close()
    log = ReviewLog(db)
    cols = {row[1] for row in log.conn.execute("PRAGMA table_info(reviews)").fetchall()}
    assert {"blind", "model_level_shown"} <= cols
    f = contract_finding(CORR, "Poor", fid="new/full")
    log.record_blind("run", f, "S4", "r2")
    assert f.unified.level == "S2" and f.review.status == "pending"
    again = log.record_blind("run", f, "S1", "r3")  # second blind grade of the same finding is not stored
    assert again["existing"] is True and again["new_level"] == "S4" and log.blind_ids("run") == {"new/full"}
    rows = log.decisions("run")
    assert len(rows) == 2 and rows[0]["blind"] == 0 and rows[1]["blind"] == 1 and rows[1]["model_level_shown"] == 0 and rows[1]["prior_level"] == "S2"
    assert drift.blind_sample([contract_finding(CORR, "Poor", fid="new/full", review={"status": "accepted"})], exclude_ids={"new/full"}) == []
    # review_health pairs one blind row per finding (last wins) so a repeated finding cannot reach blind_min_n alone
    rows3 = [_row("accepted", "S3", "S3", fid="one")] + [_row("blind", "S3", "S1", fid="one", blind=1) for _ in range(10)]
    assert drift.review_health(rows3, [])["blind"]["n"] == 1
    assert [r["finding_id"] for r in log.timeline("run")] == ["old/full"]
    assert log.agreement("run")["total"] == 1
    assert ReviewLog(db).migrate() == []  # idempotent


# ---------------------------------------------------------------------------------------------- M9

@pytest.mark.skipif(not DEV_MANIFEST.exists(), reason="dev manifest absent")
def test_canary_manifest_is_dev_only_and_disjoint_from_exemplars(tmp_path):
    from cascade.exemplars import select_exemplars

    rows = drift.canary_manifest(DEV_MANIFEST, tmp_path / "canary" / "manifest.jsonl")
    assert len(rows) == 14 and len(read_manifest(tmp_path / "canary" / "manifest.jsonl")) == 14
    controls = [r for r in rows if r.labels.get("control")]
    assert sorted(r.labels["control"] for r in controls) == ["blank", "noise"] and all(r.split == "canary" for r in rows)
    assert (tmp_path / "canary" / "blank_grey.png").exists() and (tmp_path / "canary" / "noise.png").exists()
    dev = read_manifest(DEV_MANIFEST)
    dev_ids = {r.image_id for r in dev}
    for ac in ("bridge_element", "steel_coating", "pv_module", "building_disaster"):
        picks = {r.image_id for r in rows if r.asset_class == ac and not r.labels.get("control")}
        assert len(picks) == 3 and picks <= dev_ids
        assert picks.isdisjoint({r.image_id for r in select_exemplars(dev, ac, 3)})
    fake_eval = [ImageRecord(image_id="e1", path="x", sha256="f" * 64, width=1, height=1, asset_class="steel_coating", split="eval_v1")]
    assert {r.sha256 for r in rows}.isdisjoint({r.sha256 for r in fake_eval})
    # the guard refuses when a canary sha256 leaks into an eval manifest
    m = tmp_path / "eval" / "manifest.jsonl"
    write_manifest(fake_eval + [rows[0].model_copy(update={"image_id": "leak"})], m)
    (m.parent / "FROZEN.sha256").write_text(hashlib.sha256(m.read_bytes()).hexdigest() + "  manifest.jsonl\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="canary"):
        drift.eval_guard(m, "abc", [rows[0].sha256], ledger=tmp_path / "ledger.jsonl")


def _reference(n=12):
    images = {f"d{i}": {"mode_native": "Fair", "mode_level": "S1", "input_tokens": 100, "flips": 0, "control": None} for i in range(n)}
    images["canary_blank"] = {"mode_native": "Good", "mode_level": "S0", "input_tokens": 50, "flips": 0, "control": "blank"}
    images["canary_noise"] = {"mode_native": "U", "mode_level": "U", "input_tokens": 50, "flips": 0, "control": "noise"}
    return {"fingerprint": "x", "n_repeats": 3, "images": images, "f0": 1 / 12, "alarm_k": drift.alarm_line(12, 1 / 12)}


def _observed(flip=0, level="S2", blank="S0", noise="U", tokens=100):
    obs = {f"d{i}": {"native": "Poor" if i < flip else "Fair", "level": level if i < flip else "S1", "input_tokens": tokens, "refused": False} for i in range(12)}
    obs["canary_blank"] = {"native": "Good", "level": blank, "input_tokens": 50, "refused": False}
    obs["canary_noise"] = {"native": "U", "level": noise, "input_tokens": 50, "refused": False}
    return obs


def test_canary_compare_binomial_line_and_controls(tmp_path):
    ref = _reference()
    assert ref["alarm_k"] == 4
    assert drift.canary_verdict(_observed(), ref)["status"] == "pass"
    assert drift.canary_verdict(_observed(flip=3), ref)["status"] == "pass"
    v = drift.canary_verdict(_observed(flip=4), ref)
    assert v["status"] == "alarm" and v["k"] == 4 and v["p_tail"] < 0.02
    v = drift.canary_verdict(_observed(flip=1, level="S3"), ref)
    assert v["status"] == "alarm" and v["big"] == ["d0"]
    assert drift.canary_verdict(_observed(blank="S1"), ref)["status"] == "alarm"
    assert drift.canary_verdict(_observed(noise="S0"), ref)["status"] == "alarm"
    v = drift.canary_verdict(_observed(tokens=101), ref)
    assert v["status"] == "alarm" and len(v["input_token_mismatches"]) == 12
    # file-based path: a fake run whose findings all match the reference passes
    recs = make_records(tmp_path, n=2)
    out = tmp_path / "canary_run"

    def grade_ref(img, *, finding_id, evidence, log, image_id, **kw):
        log.record(stage="grade", model="fake-grader", image_id=image_id, input_tokens=100, output_tokens=5, seconds=0.1)
        return contract_finding(CORR, "Fair", fid=finding_id, evidence=evidence)

    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=route_all, grade_fn=grade_ref)
    small_ref = {"images": {"img_0": {"mode_native": "Fair", "mode_level": "S1", "input_tokens": 100}, "img_1": {"mode_native": "Fair", "mode_level": "S1", "input_tokens": 100}}, "f0": 0.0, "alarm_k": 1}
    assert drift.canary_compare(out, small_ref)["status"] == "pass"
    # canary_reference from repeats of that run: zero flips, f0 = 0, alarm line 1
    ref2 = drift.canary_reference([out, out], tmp_path / "reference.json", controls={})
    assert ref2["f0"] == 0.0 and ref2["alarm_k"] == 1 and ref2["images"]["img_0"]["mode_native"] == "Fair"


# ---------------------------------------------------------------------------------------------- M10

def test_eval_guard_refuses_second_look_and_changed_manifest(tmp_path):
    m = tmp_path / "eval_v1" / "manifest.jsonl"
    write_manifest([ImageRecord(image_id="e1", path="x", sha256="a" * 64, width=1, height=1, asset_class="steel_coating", split="eval_v1")], m)
    sha = hashlib.sha256(m.read_bytes()).hexdigest()
    (m.parent / "FROZEN.sha256").write_text(f"{sha}  manifest.jsonl\n", encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    drift.eval_guard(m, "fp1", ledger=ledger)
    drift.ledger_append("fp1", "run1", sha, "scored", ledger=ledger)
    with pytest.raises(RuntimeError, match="one eval look"):
        drift.eval_guard(m, "fp1", ledger=ledger)
    drift.eval_guard(m, "fp2", ledger=ledger)  # a new fingerprint gets its one look
    with pytest.raises(RuntimeError, match="exemplar"):
        drift.eval_guard(m, "fp2", ["e1"], ledger=ledger)
    m.write_text(m.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FROZEN"):
        drift.eval_guard(m, "fp2", ledger=ledger)


def test_promotion_check_margins_and_insufficient_n():
    base = {"gate": {"recall": 0.80}, "grading": {"steel_coating": {"n_assessed": 40, "within_one_grade": 0.85, "u_rate": 0.10}, "bridge_element": {"n_assessed": 5, "within_one_grade": 1.0, "u_rate": 0.0}}, "ops": {"usd_per_image": 0.08}}
    cand = {"gate": {"recall": 0.78}, "grading": {"steel_coating": {"n_assessed": 40, "within_one_grade": 0.82, "u_rate": 0.15}, "bridge_element": {"n_assessed": 5, "within_one_grade": 0.5, "u_rate": 0.0}}, "ops": {"usd_per_image": 0.09}}
    p = drift.promotion_check(cand, base, {"hard_total": 0})
    assert p["verdict"] == "promote" and p["per_class"]["bridge_element"]["verdict"] == "insufficient_n" and p["per_class"]["bridge_element"]["n_required"] == 20
    cand["grading"]["steel_coating"]["within_one_grade"] = 0.79
    p = drift.promotion_check(cand, base, {"hard_total": 1})
    assert p["verdict"] == "hold" and len(p["reasons"]) == 2


def test_promote_and_rollback_round_trip(tmp_path):
    fp1 = drift.fingerprint(RunConfig(gate="none"), env={"GRADER_MODEL": "claude-opus-5"}, ollama=False)
    fp2 = drift.fingerprint(RunConfig(gate="none"), env={"GRADER_MODEL": "claude-opus-5-5"}, ollama=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "changelog.md").write_text("# Changelog\n", encoding="utf-8")
    drift.promote(fp1, "none", tmp_path / "ref.json", "eval1", "alice", baseline_dir=tmp_path / "_baseline", docs_dir=docs)
    drift.promote(fp2, "none", tmp_path / "ref.json", "eval2", "bob", baseline_dir=tmp_path / "_baseline", docs_dir=docs)
    active = json.loads((tmp_path / "_baseline" / "active.json").read_text(encoding="utf-8"))
    assert active["fingerprint"]["id"] == fp2["id"] and active["changed"] == ["model.grader"] and active["previous"]["fingerprint"]["id"] == fp1["id"]
    assert fp2["id"] in (docs / "changelog.md").read_text(encoding="utf-8")
    r = drift.rollback(tmp_path / "_baseline")
    assert r["restored"] == fp1["id"] and r["env"]["GRADER_MODEL"] == "claude-opus-5"


def test_rollback_repoints_class_cards(tmp_path):
    recs = real_sha(make_records(tmp_path, n=12, size=(300, 200)))
    out = tmp_path / "runs" / "dev_fake"
    write_manifest(recs, out / "manifest.jsonl")
    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    bdir = tmp_path / "_baseline"
    fp1 = drift.fingerprint(RunConfig(gate="none"), env={"GRADER_MODEL": "claude-opus-5"}, ollama=False)
    fp2 = drift.fingerprint(RunConfig(gate="none"), env={"GRADER_MODEL": "claude-opus-5-5"}, ollama=False)
    for fp in (fp1, fp2):  # promote() builds the cards only for runs under ROOT/runs; build them the same way here
        drift.build_baseline(out, drift.identity_of(fp), bdir)
        drift.promote(fp, "none", tmp_path / "ref.json", "eval", "alice", baseline_dir=bdir, docs_dir=tmp_path)
    assert drift.load_baseline("steel_coating", out_dir=bdir)["fingerprint"] == fp2["model_id"]
    r = drift.rollback(bdir)
    assert r["cards_repointed"] == ["steel_coating"] and r["cards_without_restored_id"] == []
    assert drift.load_baseline("steel_coating", out_dir=bdir)["fingerprint"] == fp1["model_id"]
    assert json.loads((bdir / "steel_coating.json").read_text(encoding="utf-8"))["rolled_back_from"] == fp2["model_id"]


# ---------------------------------------------------------------------------------------------- export / surge

def test_export_and_surge_normalise_u():
    f = contract_finding(BRIDGE, "CS1", fid="b/full", asset_class="bridge_element", unified=Unified(level="U", uncertainty="+/-1", flags=["not_measurable"]))
    assert bridge_row(f)["condition_state"] == "U"
    g = Finding.model_validate({**f.model_dump(), "asset_class": "building_disaster", "native_scale": {"standard": "FEMA-PDA", "value": "Affected", "criteria_matched": ["x"]}})
    c = surge_counts([g])
    assert c["by_fema_class"]["Affected"] == 0 and c["by_fema_class"]["U"] == 1 and c["u_count"] == 1


# ---------------------------------------------------------------------------------------------- health entry point

def test_health_alert_budget_and_n_floor(tmp_path):
    out = tmp_path / "run"
    run_cascade(make_records(tmp_path, n=6), out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    with (out / "calls.jsonl").open("a", encoding="utf-8") as fh:  # two refusals and two parse failures on top of the contract breaches
        for note in ("refusal", "refusal", "parse_error: x", "parse_error: y"):
            fh.write(json.dumps({"ts": "", "stage": "grade", "model": "fake-grader", "image_id": "img_9", "input_tokens": 1, "output_tokens": 1, "usd": 0.0, "seconds": 0.1, "note": note}) + "\n")
    h = drift.health(out, baseline_dir=tmp_path / "_baseline")
    assert (out / "health.json").exists() and h["status"] == "alarm" and h["baseline_id"] is None
    assert [a["rule"] for a in h["alarms"]] == ["contract_hard"]  # one alarm per stage surfaced, ranked
    demoted = [w for w in h["watch"] if w.get("demoted_from") == "alarm"]
    assert sorted(w["rule"] for w in demoted) == ["parse_error", "refusal"]
    assert h["stages"]["grade"]["status"] == "alarm" and h["stages"]["grade"]["worst_rule"] == "contract_hard" and h["stages"]["grade"]["n"] == 5
    assert h["u_chart"]["status"] == "insufficient_n" and h["u_chart"]["n_required"] == 10
    assert h["stages"]["review"]["status"] == "insufficient_n" and h["stages"]["review"]["n_required"] == 10
    assert h["fingerprint_id"] == json.loads((out / "fingerprint.json").read_text(encoding="utf-8"))["id"]
    # an acknowledgement covering the alarm rules survives a recompute; one that no longer covers them is dropped
    hd = json.loads((out / "health.json").read_text(encoding="utf-8"))
    hd["acknowledged"] = {"by": "alice", "at": "t", "rules": ["contract_hard", "refusal", "parse_error"]}
    (out / "health.json").write_text(json.dumps(hd), encoding="utf-8")
    assert drift.health(out, baseline_dir=tmp_path / "_baseline")["acknowledged"]["by"] == "alice"
    hd["acknowledged"] = {"by": "bob", "at": "t", "rules": ["refusal"]}
    (out / "health.json").write_text(json.dumps(hd), encoding="utf-8")
    assert drift.health(out, baseline_dir=tmp_path / "_baseline")["acknowledged"] is None


def test_blocking_rules_are_never_demoted_and_gate_miss_is_a_gate_watch():
    sig = lambda rule, stage, sev, **kw: drift._signal(rule, stage, sev, 1, 0, 10, "x", **kw)  # noqa: E731
    alarms, watch, _ = drift._apply_budget([sig("refusal", "grade", "alarm"), sig("abstention_collapse", "grade", "alarm"), sig("moved_2plus", "grade", "alarm"), sig("u_rate", "grade", "alarm")])
    assert [a["rule"] for a in alarms] == ["refusal", "abstention_collapse", "moved_2plus"] and [w["rule"] for w in watch] == ["u_rate"]
    assert set(drift.BLOCKING_RULES) == {"contract_hard", "abstention_collapse", "moved_2plus", "critical_miss"}
    # a force-routed image the gate called clean but the grader put at S3 is a gate_miss watch, not a critical miss
    gate_rows = [{"image_id": "ir_0", "damage_present": False, "reason": "no anomaly [forced: asset class pv_module]"}]
    f = contract_finding(CORR, "Poor", fid="ir_0/full", level="S3")
    r = drift.review_health([], [f], gate_rows=gate_rows)
    assert r["critical_misses"] == [] and [m["finding_id"] for m in r["gate_misses"]] == ["ir_0/full"]
    assert [(s["rule"], s["stage"], s["severity"]) for s in r["signals"]] == [("gate_miss", "gate", "watch")]
    # two S4 misses among the last 100 sighted decisions block the fingerprint
    rows = [_row("overridden", "S1", "S4", fid=f"m{i}") for i in range(2)]
    r2 = drift.review_health(rows, [])
    assert r2["fingerprint_blocked"] is True and "fingerprint blocked" in r2["signals"][0]["action"]


def test_u_chart_below_lcl_watches_only_with_s0_rise():
    base = {"u_p0": 0.30, "level_shares": {"S0": 0.10}}
    fs = [contract_finding(CORR, "Good", fid=f"g{i}/full", unified=Unified(level="U", uncertainty="+/-1", flags=["not_measurable"])) for i in range(1)]
    fs += [contract_finding(CORR, "Fair", fid=f"f{i}/full") for i in range(49)]  # S1: U fell, S0 did not rise
    c = drift.u_chart(fs, [], base)
    assert c["status"] == "in_control" and c.get("below_lcl") is True and c["signals"] == []
    fs2 = fs[:1] + [contract_finding(CORR, "Good", fid=f"s{i}/full") for i in range(49)]  # S0 share 0.98 vs 0.10 on the card
    c2 = drift.u_chart(fs2, [], base)
    assert c2["status"] == "watch" and c2["signals"][0]["rule"] == "u_rate_low"


def test_health_flags_mid_run_fingerprint_change(tmp_path):
    out = tmp_path / "run"
    run_cascade(make_records(tmp_path, n=6), out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    s["fingerprint_at_finish"] = "deadbeef0000"
    (out / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    h = drift.health(out, write=False, baseline_dir=tmp_path / "_baseline")
    assert h["comparable"] is False and any(a["rule"] == "fingerprint_changed_mid_run" and a["stage"] == "config" for a in h["alarms"])


def test_baseline_freeze_refuses_eval_v1_records(tmp_path):
    recs = [r.model_copy(update={"split": "eval_v1"}) for r in make_records(tmp_path, n=4)]
    out = tmp_path / "run"
    write_manifest(recs, out / "manifest.jsonl")
    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    with pytest.raises(RuntimeError, match="eval_v1"):
        drift.baseline_freeze(out, "steel_coating", out_dir=tmp_path / "_baseline")
    assert not (tmp_path / "_baseline").exists()
    labelled = {r.image_id: r.model_copy(update={"labels": {"damage_present": True}}) for r in recs}
    gate_rows = [{"image_id": r.image_id, "usable": True, "damage_present": True, "confidence": 0.9, "reason": "rust", "routed": True} for r in recs]
    assert drift.gate_health(gate_rows, labelled, [], 0.7)["eval"]["gate"] == {}  # eval_v1 labels are never scored here
    assert drift.gate_health(gate_rows, labelled, [], 0.7, allow_eval=True)["eval"]["gate"]["recall"] == 1.0


def test_baseline_freeze_load_and_comparable_health(tmp_path):
    recs = real_sha(make_records(tmp_path, n=12, size=(300, 200)))  # above the 256 px resolution watch
    out = tmp_path / "runs" / "dev_fake"
    write_manifest(recs, out / "manifest.jsonl")

    def grade_clean(img, *, finding_id, evidence, log, image_id, **kw):
        log.record(stage="grade", model="fake-grader", image_id=image_id, input_tokens=100, output_tokens=50, seconds=0.02, note="end_turn")
        idx = int(image_id.split("_")[-1])
        return contract_finding(CORR, CORR["allowed_values"][idx % 4], fid=finding_id, evidence=evidence)

    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=grade_clean)
    bdir = tmp_path / "_baseline"
    card = drift.baseline_freeze(out, "steel_coating", out_dir=bdir)
    assert card["n_findings"] == 11 and abs(card["u_p0"] - 1 / 11) < 1e-9 and card["hard_violations"] == 0 and card["clean_grade_calls"] == 10
    loaded = drift.baseline_load("steel_coating", out_dir=bdir)
    assert loaded["fingerprint"] == json.loads((out / "fingerprint.json").read_text(encoding="utf-8"))["model_id"] and loaded["routed_by_gate_p0"] == {"fake": 11 / 12}  # fake_gate skips img_0 only
    h = drift.health(out, baseline_dir=bdir)
    assert h["comparable"] is True and h["u_chart"]["status"] == "in_control" and h["u_chart"]["p0"] == card["u_p0"]
    assert h["stages"]["grade"]["status"] == "in_control" and h["alarms"] == []
    # fake_gate writes three canned reasons over 12 rows (ratio 0.25): the template-collapse watch is the only signal
    assert [w["rule"] for w in h["watch"]] == ["template_collapse"] and h["stages"]["gate"]["status"] == "watch"
    assert drift.baseline_load("pv_module", out_dir=bdir) is None


def test_mixed_class_run_is_charted_per_class(tmp_path):
    """A two-class run frozen as its own baseline must be in control against itself (it was not: run-wide U vs the first class's p0)."""
    (tmp_path / "pv").mkdir()
    recs = real_sha(make_records(tmp_path, n=12, size=(300, 200)) + make_records(tmp_path / "pv", n=12, size=(300, 200), asset_class="pv_module"))
    recs = [r.model_copy(update={"image_id": f"{r.asset_class[:2]}_{r.image_id}"}) for r in recs]
    out = tmp_path / "runs" / "mixed"
    write_manifest(recs, out / "manifest.jsonl")

    def grade_mixed(img, *, finding_id, evidence, log, image_id, asset_class, **kw):
        log.record(stage="grade", model="fake-grader", image_id=image_id, input_tokens=100, output_tokens=50, seconds=0.02, note="end_turn")
        if asset_class == "pv_module":
            return unassessable_finding(finding_id=finding_id, asset_class="pv_module", standard="IEC-62446-3-CoA", evidence=evidence, reason="irradiance unknown")
        return contract_finding(CORR, "Fair", fid=finding_id, evidence=evidence)

    run_cascade(recs, out, RunConfig(gate="none", force_route_classes=()), gate_fn=route_all, grade_fn=grade_mixed)
    bdir = tmp_path / "_baseline"
    cards = drift.build_baseline(out, drift.identity_of(json.loads((out / "fingerprint.json").read_text(encoding="utf-8"))), bdir)
    assert cards["steel_coating"]["u_p0"] == 0.0 and cards["pv_module"]["u_p0"] == 1.0
    h = drift.health(out, baseline_dir=bdir)
    assert h["comparable"] is True and h["comparable_classes"] == ["pv_module", "steel_coating"]
    uc = h["u_chart"]
    assert uc["n"] == 24 and uc["p0"] is None and uc["status"].startswith("per class")
    assert uc["per_class"]["steel_coating"]["status"] == "in_control" and uc["per_class"]["pv_module"]["status"] == "in_control"
    assert not any(a["rule"] == "u_rate" for a in h["alarms"] + h["watch"])


@pytest.mark.skipif(not (UI_RUN / "findings.json").exists(), reason="stored run absent")
def test_health_on_stored_ui_run():
    """Pins the hand-derived expectation on runs/ui_0925_0856 (hard 0, W4 4, model_u 4, gate recall 9/10)."""
    h = drift.health(UI_RUN, write=False)
    assert all(v == 0 for v in h["contract"]["hard"].values())
    assert h["contract"]["soft"]["W4_mixed_u"] == 4 and h["contract"]["u_by_cause"]["model_u"] == 4
    assert h["contract"]["soft"]["W3_sla_inconsistent"] == 3  # measured: CS3/schedule, CS1/monitor and CS1/record groups mix null and non-null sla_days
    assert h["gate"]["eval"]["gate"]["recall"] == 0.9 and h["alarms"] == []
    assert h["ops"]["by_stage_model"]["grade/claude-opus-5"]["calls"] == 14 and h["ops"]["unpriced"] == []
    assert h["status"] == "candidate" and h["fingerprint_id"] is None
