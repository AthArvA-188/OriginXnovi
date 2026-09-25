"""Workforce impact model: the spec 3.6 worked example (stored run when present, a fake run always), pure-model
identities, per-client sums, reviewer-gap measurement and the labelling rules. Fake backends, no network."""

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.pipeline import RunConfig, run_cascade
from cascade.review import SCHEMA
from cascade.schema import Action, Evidence, Finding, ImageRecord, Measurements, NativeScale, Unified
from cascade.workforce import (
    DEFAULTS,
    DISCLAIMER,
    FOOTER,
    NEGATIVE_SENTENCE,
    estimate,
    estimate_by_client,
    measured_inputs,
    render_markdown,
    review_gap_median,
    rule_of_three_bound,
    to_json,
    tornado,
)
from test_pipeline import fake_gate, fake_grade, make_records

ROOT = Path(__file__).resolve().parents[1]
STORED = ROOT / "runs" / "ui_0925_0856"
GATE_ONLY = ROOT / "runs" / "dev_gate02"

# spec section 8 test 1, tolerance 1e-3
GOLDEN = dict(manual_min=30.0, cascade_min=12.0, saved_pct=60.0, saved_h_1000=34.5, ie_per_1000_week=0.8625, cost_img_human=5.6515, cost_img_cascade=2.3422)


def _finding(image_id: str, tile: str, level: str) -> Finding:
    return Finding(
        finding_id=f"{image_id}/{tile}",
        asset_class="bridge_element",
        defect_type="not_assessable" if level == "U" else "spalling",
        native_scale=NativeScale(standard="MBEI-CS", value="U" if level == "U" else "CS2", criteria_matched=[]),
        unified=Unified(level=level, uncertainty="+/-1", flags=[]),
        measurements=Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=0.5),
        action=Action(code="record", sla_days=None, basis="test"),
        justification="fake",
        evidence=Evidence(image_ids=[image_id], bbox=[0, 0, 10, 10], tile=tile),
        model="fake-grader",
    )


def write_fake_run(out: Path, clients: dict = None, grader: str = "claude") -> Path:
    """Run folder with the spec 3.6 counts: N=10, G=10, R=9, UN=0, GI=9, F=14, U=4, C=$0.8161, gate 33.02 s, grade 221.647 s."""
    out.mkdir(parents=True, exist_ok=True)
    clients = clients or {}
    ids = [f"img_{i}" for i in range(10)]
    gate = [{"image_id": i, "usable": True, "damage_present": i != "img_0", "confidence": 0.9, "reason": "fake", "routed": i != "img_0", "model": "fake-gate", "seconds": 3.3, "usd": 0.002} for i in ids]
    tiles = {"img_1": 2, "img_2": 4, "img_3": 2}  # 2 + 4 + 2 + 6 x 1 = 14 findings on the 9 routed images
    levels = ["S0"] * 4 + ["S1"] * 3 + ["S2"] * 3 + ["U"] * 4
    findings = []
    for i in ids[1:]:
        for t in range(tiles.get(i, 1)):
            findings.append(_finding(i, f"t{t:02d}" if tiles.get(i, 1) > 1 else "full", levels[len(findings)]))
    if grader == "none":
        findings = []
    ts = "2026-09-25T00:00:00+00:00"
    calls = [{"ts": ts, "stage": "gate", "model": "fake-gate", "image_id": i, "input_tokens": 1, "output_tokens": 1, "usd": 0.002 if n else 0.003561, "seconds": 3.3 if n else 3.32, "note": ""} for n, i in enumerate(ids)]
    calls += [{"ts": ts, "stage": "grade", "model": "fake-grader", "image_id": f.evidence.image_ids[0], "input_tokens": 1, "output_tokens": 1, "usd": 0.05 if n else 0.14449, "seconds": 15.8 if n else 16.247, "note": ""} for n, f in enumerate(findings)]
    recs = [ImageRecord(image_id=i, path=f"{i}.jpg", sha256="0" * 64, width=100, height=100, asset_class="bridge_element", source_dataset="dacl10k", split="dev", client_id=clients.get(i)) for i in ids]
    (out / "gate.jsonl").write_text("\n".join(json.dumps(g) for g in gate) + "\n", encoding="utf-8")
    (out / "findings.json").write_text(json.dumps([f.model_dump() for f in findings]), encoding="utf-8")
    (out / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"images": 10, "config": {"gate": "claude", "grader": grader, "tiles": True}}), encoding="utf-8")
    (out / "manifest.jsonl").write_text("\n".join(r.model_dump_json() for r in recs) + "\n", encoding="utf-8")
    return out


def _seed_reviews(out: Path, rows) -> None:
    conn = sqlite3.connect(out / "reviews.sqlite")
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO reviews(run_id, finding_id, action, prior_level, new_level, reviewer, reviewed_at) VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _check_measured(inp) -> None:
    assert (inp.images, inp.gated, inp.routed, inp.unusable, inp.graded_images, inp.findings, inp.u_findings, inp.reviews) == (10, 10, 9, 0, 9, 14, 4, 0)
    assert inp.usd_total == pytest.approx(0.8161, abs=1e-4)
    assert inp.seconds_gate == pytest.approx(33.0, abs=0.1) and inp.seconds_grade == pytest.approx(221.6, abs=0.1)
    assert inp.source_datasets == ["dacl10k"] and inp.complete and inp.gap_n == 0 and inp.gap_median_min is None


def _check_golden(est) -> None:
    for k, v in GOLDEN.items():
        assert getattr(est, k) == pytest.approx(v, abs=1e-3), k
    assert est.audit_n == 1 and est.auto_cleared == 1 and est.audit_1000 == 10
    assert est.routing_fraction == pytest.approx(0.9) and est.fpi == pytest.approx(14 / 9)
    assert est.break_even_m_man == pytest.approx(1.0)
    assert est.manual_h_1000 == pytest.approx(50.0) and est.cascade_h_1000 == pytest.approx(15.5) and est.saved_pct_1000 == pytest.approx(69.0)
    assert est.cost_img_cascade_api == pytest.approx(0.0816, abs=1e-3) and est.cost_img_cascade_1000 == pytest.approx(1.83, abs=0.01)
    assert est.model_s_per_image == pytest.approx(25.5, abs=0.05)
    assert est.m_rev_eff == 1.0 and est.label == "estimate"
    assert est.assumptions["m_aud"].value == est.assumptions["m_man"].value


@pytest.mark.skipif(not STORED.exists(), reason="stored run runs/ui_0925_0856 not present")
def test_worked_example_recomputed_from_stored_run():
    inp = measured_inputs(STORED)
    _check_measured(inp)
    assert inp.grader == "claude" and inp.gate == "claude"
    _check_golden(estimate(inp))


def test_worked_example_recomputed_from_fake_run(tmp_path):
    inp = measured_inputs(write_fake_run(tmp_path / "ui_fake"))
    _check_measured(inp)
    _check_golden(estimate(inp))


@pytest.mark.skipif(not GATE_ONLY.exists(), reason="stored run runs/dev_gate02 not present")
def test_gate_only_stored_run_matches_spec():
    inp = measured_inputs(GATE_ONLY)
    est = estimate(inp)
    assert inp.graded_images == 0 and inp.gated == 50 and inp.routed == 37 and inp.usd_total == 0.0  # old summary has no config key
    assert est.m_rev_eff == DEFAULTS["m_man"].value and est.audit_n == 2
    assert est.manual_min == pytest.approx(150.0) and est.cascade_min == pytest.approx(117.0) and est.saved_pct == pytest.approx(22.0)
    assert any("no graded findings" in w for w in est.warnings)


def test_negative_saving_is_returned_not_clamped(tmp_path):
    inp = measured_inputs(write_fake_run(tmp_path / "ui_fake"))
    est = estimate(inp, {"m_man": DEFAULTS["m_skim"].value})
    assert est.cascade_min == pytest.approx(9.067, abs=1e-3)  # m_aud tracks m_man
    assert est.saved_min == pytest.approx(-8.4, abs=0.01) and est.saved_pct < -1000
    assert NEGATIVE_SENTENCE in est.warnings
    assert estimate(inp, {"m_man": DEFAULTS["m_skim"].value, "m_aud": 3.0}).cascade_min == pytest.approx(12.0)  # explicit override


def test_count_identities_and_audit_rules(tmp_path):
    inp = measured_inputs(write_fake_run(tmp_path / "ui_fake"))
    est = estimate(inp)
    assert est.inputs.routed + est.auto_cleared == est.inputs.gated
    assert estimate(inp, {"a": 0}).audit_n == 0
    all_routed = replace(inp, routed=inp.gated)
    assert estimate(all_routed).auto_cleared == 0 and estimate(all_routed).audit_n == 0
    assert estimate(inp, {"a": 0.01}).audit_n == 1 and any("rounded up" in w for w in estimate(inp, {"a": 0.01}).warnings)
    big = replace(inp, images=100, gated=100, routed=70)  # 0.1 x 30 = 3.0000000000000004 must audit 3, not 4
    assert estimate(big).audit_n == 3
    assert rule_of_three_bound(est.audit_1000) == pytest.approx(0.3) and rule_of_three_bound(0) is None


def test_hourly_rate_moves_dollars_only(tmp_path):
    inp = measured_inputs(write_fake_run(tmp_path / "ui_fake"))
    base, double = estimate(inp), estimate(inp, {"w": 2 * DEFAULTS["w"].value})
    assert double.saved_pct == base.saved_pct and double.ie_per_1000_week == base.ie_per_1000_week and double.saved_h_1000 == base.saved_h_1000
    assert double.cost_img_human == pytest.approx(2 * base.cost_img_human)
    assert double.cost_img_cascade_hum == pytest.approx(2 * base.cost_img_cascade_hum)
    assert double.cost_img_cascade_api == base.cost_img_cascade_api


def test_gate_only_run_falls_back_to_manual_rate(tmp_path):
    inp = measured_inputs(write_fake_run(tmp_path / "gate_only", grader="none"))
    assert inp.graded_images == 0 and inp.grader == "none" and inp.usd_total == pytest.approx(0.021561)
    est = estimate(inp)
    assert est.m_rev_eff == DEFAULTS["m_man"].value and est.cascade_min == pytest.approx(9 * 3.0 + 1 * 3.0)
    assert est.break_even_m_man is None
    assert any("no graded findings" in w for w in est.warnings)
    no_grades = replace(measured_inputs(write_fake_run(tmp_path / "ui_fake")), graded_images=0, findings=0, u_findings=0)
    assert estimate(no_grades).m_rev_eff == DEFAULTS["m_man"].value


def test_adding_u_finding_never_lowers_cascade_min(tmp_path):
    inp = measured_inputs(write_fake_run(tmp_path / "ui_fake"))
    same_image = replace(inp, findings=inp.findings + 1, u_findings=inp.u_findings + 1)
    new_image = replace(inp, images=inp.images + 1, gated=inp.gated + 1, routed=inp.routed + 1, graded_images=inp.graded_images + 1, findings=inp.findings + 1, u_findings=inp.u_findings + 1)
    for ov in ({}, {"m_xf": 0.5}, {"a": 0.0}):
        base = estimate(inp, ov).cascade_min
        assert estimate(same_image, ov).cascade_min >= base
        assert estimate(new_image, ov).cascade_min >= base


def test_client_buckets_sum_to_run_totals(tmp_path):
    clients = {"img_0": "acme", "img_1": "acme", "img_2": "beta", "img_3": "beta", "img_4": "beta"}  # img_5..img_9 unassigned
    out = write_fake_run(tmp_path / "ui_fake", clients=clients)
    records = {r.image_id: r for r in (ImageRecord.model_validate(json.loads(ln)) for ln in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    total = measured_inputs(out, records)
    by_client = estimate_by_client(out, records)
    assert list(by_client) == ["acme", "beta", "unassigned"]
    for fld in ("images", "gated", "routed", "unusable", "graded_images", "findings", "u_findings", "reviews"):
        assert sum(getattr(e.inputs, fld) for e in by_client.values()) == getattr(total, fld), fld
    assert sum(e.inputs.usd_total for e in by_client.values()) == pytest.approx(total.usd_total)
    assert sum(e.inputs.seconds_gate for e in by_client.values()) == pytest.approx(total.seconds_gate)
    assert sum(e.inputs.seconds_grade for e in by_client.values()) == pytest.approx(total.seconds_grade)
    acme = by_client["acme"]
    assert acme.inputs.images == 2 and acme.inputs.routed == 1 and acme.auto_cleared == 1 and acme.inputs.findings == 2 and acme.inputs.usd_total > 0
    assert by_client["unassigned"].inputs.images == 5 and by_client["unassigned"].inputs.images_without_client == 5
    assert total.images_without_client == 5 and any("without a client id" in w for w in estimate(total).warnings)
    assert by_client["beta"].inputs.images_without_client == 0 and by_client["beta"].inputs.client_id == "beta"


def test_client_buckets_on_pipeline_output(tmp_path):
    recs = make_records(tmp_path, n=4)
    for r in recs:
        r.client_id = {"img_0": "acme", "img_1": "acme", "img_2": "beta"}.get(r.image_id)
    out = tmp_path / "run"
    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    records = {r.image_id: r for r in recs}
    total, by_client = measured_inputs(out, records), estimate_by_client(out, records)
    assert list(by_client) == ["acme", "beta", "unassigned"]
    assert sum(e.inputs.routed for e in by_client.values()) == total.routed == 3
    assert by_client["acme"].inputs.u_findings == 1  # img_1 unusable -> U, always costed as human time
    assert total.source_datasets == ["fake"] and any("not a fleet number" in w for w in estimate(total).warnings)


def test_missing_run_files_return_zeros(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    inp = measured_inputs(out)
    assert not inp.complete and inp.images == 0 and inp.gated == 0 and inp.usd_total == 0.0 and inp.gap_n == 0
    est = estimate(inp)
    assert est.saved_pct is None and est.model_s_per_image is None and est.cascade_min == 0.0 and est.break_even_m_man is None
    assert any("incomplete" in w for w in est.warnings)
    assert not (out / "reviews.sqlite").exists()  # reading never creates a review log
    assert FOOTER in render_markdown(est) and to_json(est)["label"] == "estimate"


def test_review_gap_median_groups_by_reviewer_and_drops_breaks(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    assert review_gap_median(out, "run") == (None, 0)

    def t(m, s=0):
        return f"2026-09-25T10:{m:02d}:{s:02d}+00:00"

    rows = [
        ("run", "f1", "accepted", "S1", "S1", "alice", t(0)),
        ("run", "f2", "accepted", "S1", "S1", "bob", t(0, 30)),  # interleaved reviewer must not create a gap for alice
        ("run", "f3", "accepted", "S1", "S1", "alice", t(2)),  # alice 2.0
        ("run", "f4", "overridden", "S1", "S2", "alice", t(6)),  # alice 4.0
        ("run", "f5", "accepted", "S1", "S1", "alice", t(30)),  # 24 min break, dropped
        ("run", "f6", "accepted", "S1", "S1", "alice", t(31)),  # alice 1.0
        ("run", "f7", "accepted", "S1", "S1", "bob", t(3, 30)),  # bob 3.0
        ("other", "f8", "accepted", "S1", "S1", "alice", t(31, 10)),  # other run, ignored
    ]
    _seed_reviews(out, rows)
    med, n = review_gap_median(out, "run", gap_cap_min=10.0)
    assert n == 4 and med == pytest.approx(2.5)
    med, n = review_gap_median(out, "run", gap_cap_min=3.0)
    assert n == 3 and med == pytest.approx(2.0)
    single = tmp_path / "single"
    single.mkdir()
    _seed_reviews(single, rows[:1])
    assert review_gap_median(single, "run") == (None, 0)


def test_measured_m_rev_replaces_assumption_after_ten_gaps(tmp_path):
    out = write_fake_run(tmp_path / "ui_fake")
    inp = measured_inputs(out)
    assert estimate(inp).m_rev_measured is None
    ignored = estimate(inp, {"use_measured_m_rev": 1})
    assert ignored.assumptions["m_rev"].kind == "assumption" and any("not enough decisions" in w for w in ignored.warnings)
    _seed_reviews(out, [("ui_fake", f"f{k}", "accepted", "S1", "S1", "alice", f"2026-09-25T10:{k:02d}:00+00:00") for k in range(11)])  # 10 gaps of 1.0 min
    inp = measured_inputs(out)
    assert inp.gap_n == 10 and inp.gap_median_min == pytest.approx(1.0)
    est = estimate(inp, {"use_measured_m_rev": 1})
    assert est.m_rev_measured == pytest.approx(14 / 9)
    assert est.assumptions["m_rev"].kind == "measured" and est.assumptions["m_rev"].value == pytest.approx(14 / 9)
    assert est.review_min == pytest.approx(14.0) and "n = 10 gaps" in est.assumptions["m_rev"].source
    assert not any("review gaps logged" in w for w in est.warnings)
    assert estimate(inp, {"use_measured_m_rev": 1, "m_rev": 2.0}).assumptions["m_rev"].value == 2.0  # explicit slider wins
    assert "Measured (run ui_fake)" in render_markdown(est)


def test_render_markdown_labels_every_number(tmp_path):
    est = estimate(measured_inputs(write_fake_run(tmp_path / "ui_fake")))
    md = render_markdown(est)
    assert "## Workforce estimate (assumptions listed)" in md and DISCLAIMER in md and FOOTER in md
    assert "Public figure" in md and "*Team assumption*" in md and "*Derived from public figure*" in md
    measured = md.split("### Measured from this run")[1].split("### Assumptions used")[0]
    assert "≈" not in measured and "estimate (" not in measured and measured.count("Measured (run ui_fake)") == 13
    estimated = md.split("### Estimated with the assumptions above")[1].split("Run figure audits")[0]
    rows = [ln for ln in estimated.splitlines() if ln.startswith("| ") and not ln.startswith("| Output")]
    assert len(rows) == 12
    for ln in rows:
        if "≈" in ln:
            assert "*estimate (uses" in ln, ln
        else:
            assert "Measured (run ui_fake)" in ln and "API" in ln, ln
    assert "| Manual review time, this run | ≈ 30.0 min | *estimate (uses 1 assumption: m_man)* |" in md
    assert "uses 3 assumptions: m_rev, a, m_aud" in md  # cascade_min, as in spec 3.6
    assert "+18.0 min (+60.0%)" in md and "+34.5 h (+69.0%)" in md and "≈ +0.86" in md and "≈ $5.65" in md and "≈ $2.34" in md
    assert "Run figure audits 1 of 1 auto-cleared images (rounded up to at least 1); at 1,000 images the same settings audit 10 of 100." in md
    assert NEGATIVE_SENTENCE not in md and "GPU time" in md
    neg = render_markdown(estimate(est.inputs, {"m_man": DEFAULTS["m_skim"].value}))
    assert NEGATIVE_SENTENCE in neg and "-8.4 min (-1253.3%)" in neg


def test_to_json_block_shape(tmp_path):
    est = estimate(measured_inputs(write_fake_run(tmp_path / "ui_fake")))
    j = to_json(est)
    assert j["label"] == "estimate" and set(j) >= {"inputs_measured", "assumptions", "outputs", "warnings"}
    assert j["inputs_measured"]["images"] == 10 and j["outputs"]["saved_pct"] == pytest.approx(60.0)
    assert {a["name"] for a in j["assumptions"]} == set(DEFAULTS)
    for a in j["assumptions"]:
        assert a["kind"] in ("public", "derived_public", "assumption") and a["source"]
    assert json.dumps(j) and j["footer"] == FOOTER and "measured" not in j["outputs"]


def test_tornado_rows(tmp_path):
    inp = measured_inputs(write_fake_run(tmp_path / "ui_fake"))
    rows = tornado(inp)
    assert len(rows) == 15 and [r["lever"] for r in rows].count("m_man") == 3
    by = {(r["lever"], r["value"]): r for r in rows}
    assert by[("m_man", 0.067)]["saved_pct"] < 0 < by[("m_man", 5.0)]["saved_pct"]
    assert by[("w", 94.43)]["saved_pct"] == by[("w", 250.0)]["saved_pct"]
    r_rows = [r for r in rows if r["lever"] == "r"]
    assert r_rows[0]["value"] == pytest.approx(0.9) and not r_rows[0]["hypothetical"] and r_rows[0]["saved_pct"] == pytest.approx(60.0)
    assert r_rows[2]["hypothetical"] and r_rows[2]["saved_pct"] is None
    assert by[("r", 0.3)]["saved_pct_1000"] == pytest.approx(83.0)  # spec 4: s = 1 - r m_rev/m_man - (1 - r) a
    whatif = estimate(inp, {"r": 0.3})
    assert whatif.saved_pct == pytest.approx(60.0) and whatif.routing_fraction == 0.3 and any("hypothetical" in w for w in whatif.warnings)


def test_defaults_carry_source_url_and_kind():
    assert set(DEFAULTS) == {"m_man", "m_skim", "m_man_hi", "m_rev", "m_xf", "m_aud", "a", "w", "H", "gap_cap"}
    for asm in DEFAULTS.values():
        assert asm.kind in ("public", "derived_public", "assumption") and asm.source and asm.unit and asm.note
        assert (asm.url is not None) == (asm.kind in ("public", "derived_public"))
    assert "tdworld.com" in DEFAULTS["m_man"].url and "renewableenergyworld.com" in DEFAULTS["m_skim"].url and "leg.wa.gov" in DEFAULTS["w"].url
    assert DEFAULTS["m_skim"].value == pytest.approx(500 * 60 / 450_000, abs=1e-3)
    inp = measured_inputs(Path("nonexistent"))
    with pytest.raises(ValueError):
        estimate(inp, {"bogus": 1})
    with pytest.raises(ValueError):
        estimate(inp, {"a": 1.5})
