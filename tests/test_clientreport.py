"""Client-wise reports: grouping, files written, HTML contract, U isolation, per-client cost, thumbnail budget,
SVG helpers, today injection and idempotence. Fake backends from test_pipeline; no model, no network."""

import json
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cascade.clientreport import (
    ACTION_ORDER,
    LEVEL_COLOR,
    LEVEL_MEANING,
    ThumbBudget,
    TimelinePoint,
    WorkloadAssumption,
    client_metrics,
    client_of,
    client_slug,
    group_records,
    svg_donut,
    svg_hbar,
    svg_timeline,
    worst_level,
    worst_levels,
    write_client_report,
    write_client_reports,
)
from cascade.pipeline import RunConfig, load_run, run_cascade
from cascade.schema import GateRecord, ImageRecord
from test_pipeline import fake_gate, fake_grade, make_records

ROOT = Path(__file__).resolve().parents[1]
REAL_RUN = ROOT / "runs" / "ui_0925_0856"
CLIENT_BY_INDEX = {0: "acme", 3: "acme", 1: "beta", 4: "beta"}  # 2 and 5 stay None -> source_dataset "fake"


def client_records(tmp_path, n=6):
    recs = make_records(tmp_path, n=n)
    for i, r in enumerate(recs):
        r.client_id = CLIENT_BY_INDEX.get(i)
    return recs


def run_fake(tmp_path, recs, **cfg):
    out = tmp_path / "run"
    run_cascade(recs, out, RunConfig(gate="none", **cfg), gate_fn=fake_gate, grade_fn=fake_grade)
    return out


def forbidden(html_text: str):
    return [tok for tok in ("<script", "<link", "url(http", 'src="http', "href=\"http") if tok in html_text]


# ---------- grouping ----------


def test_client_of_chain_and_slug_collision():
    base = dict(path="x.jpg", sha256="0" * 64, width=8, height=8, asset_class="steel_coating")
    assert client_of(ImageRecord(image_id="a", client_id="Acme", source_dataset="ds", **base)) == "Acme"
    assert client_of(ImageRecord(image_id="b", client_id=None, source_dataset="ds", **base)) == "ds"
    assert client_of(ImageRecord(image_id="c", client_id=None, source_dataset="", **base)) == "unassigned"
    assert client_slug("Acme Bridges") == "acme_bridges"
    a, b = client_slug("Acme Bridges", ["Acme/Bridges"]), client_slug("Acme/Bridges", ["Acme Bridges"])
    assert a != b and a.startswith("acme_bridges-") and re.fullmatch(r"[a-z0-9._-]+", a) and len(client_slug("x" * 100)) <= 64
    groups = group_records({r.image_id: r for r in [ImageRecord(image_id="a", client_id="Acme", source_dataset="ds", **base), ImageRecord(image_id="b", source_dataset="ds", **base)]})
    assert set(groups) == {"Acme", "ds"}


def test_worst_level_rules():
    assert worst_level(["U", "S1", "S3"]) == "S3" and worst_level(["U", "U"]) == "U" and worst_level([]) is None
    assert worst_levels([]) == {}


# ---------- files, index, HTML contract ----------


def test_write_client_reports_groups_and_writes_files(tmp_path):
    recs = client_records(tmp_path)
    out = run_fake(tmp_path, recs)
    paths = write_client_reports(out, {r.image_id: r for r in recs}, today=date(2026, 10, 5))
    assert set(paths) == {"acme", "beta", "fake"}
    index = json.loads((out / "clients" / "index.json").read_text(encoding="utf-8"))
    assert {i["client_id"] for i in index} == {"acme", "beta", "fake"} and all((out / i["html"]).exists() and (out / i["md"]).exists() for i in index)
    findings = load_run(out)["findings"]
    for cid, p in paths.items():
        html_text = p.read_text(encoding="utf-8")
        md_text = (p.parent / "report.md").read_text(encoding="utf-8")
        m = json.loads((p.parent / "report.json").read_text(encoding="utf-8"))
        assert p.name == "report.html" and p.stat().st_size <= 5_000_000 and forbidden(html_text) == []
        assert html_text.startswith("<!doctype html>") and "<svg" in html_text and "data:image/jpeg;base64," in html_text
        assert m["client_id"] == cid and (out / "clients" / m["slug"]).is_dir()
        mine = [f for f in findings if f.evidence.image_ids[0] in {r.image_id for r in recs if client_of(r) == cid}]
        assert m["findings"] == len(mine)
        for f in mine:
            assert f.finding_id in html_text and f.finding_id in md_text
        assert "never counted as S0" in html_text and "AI-assisted, human-certified" in html_text
    acme = json.loads((paths["acme"].parent / "report.json").read_text(encoding="utf-8"))
    assert acme["grouping_basis"] == "client_id" and acme["images"] == 2 and acme["not_routed"] == 1 and acme["levels"]["S4"] == 1
    assert acme["sla"][0]["action"] == "escalate" and acme["sla"][0]["sla_days"] == 0 and acme["sla"][0]["due_in_days"] is not None
    assert "Images the gate cleared without a heavy grade: 1 of 2" in paths["acme"].read_text(encoding="utf-8")
    fake = json.loads((paths["fake"].parent / "report.json").read_text(encoding="utf-8"))
    assert fake["grouping_basis"] == "source_dataset" and "grouped by source_dataset" in paths["fake"].read_text(encoding="utf-8")
    assert all(a["basis"] == "image" for a in fake["assets"]) and "eval" not in fake and "accuracy" not in json.dumps(fake["levels"])


def test_records_none_reads_manifest_or_raises(tmp_path):
    recs = client_records(tmp_path, n=3)
    out = run_fake(tmp_path, recs)
    with pytest.raises(FileNotFoundError, match="manifest.jsonl"):
        write_client_reports(out)
    (out / "manifest.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")
    assert set(write_client_reports(out)) == {"acme", "beta", "fake"}


# ---------- U isolation ----------


def test_all_u_client_has_zero_s0_and_no_undated_u_point(tmp_path):
    recs = make_records(tmp_path, n=2)
    recs[0].client_id, recs[1].client_id = "other", "solo"  # img_1 is the gate-unusable image -> one U finding
    out = run_fake(tmp_path, recs)
    m = client_metrics(out, "solo", {r.image_id: r for r in recs})
    assert m["findings"] == 1 and m["levels"]["U"] == 1 and m["levels"]["S0"] == 0 and sum(m["levels"].values()) == 1
    assert len(m["u_rows"]) == 1 and m["u_rows"][0]["next_step"] == "re-image" and m["assets"][0]["worst_level"] == "U"
    assert m["workload"]["prefilled"] == 0 and m["workload"]["reimage_or_metadata"] == 1
    p = write_client_report(out, "solo", {r.image_id: r for r in recs})
    html_text = p.read_text(encoding="utf-8")
    sla_section = html_text.split('id="sla"')[1].split('id="method"')[0]
    assert not re.search(rf'<circle[^>]*fill="{LEVEL_COLOR["U"]}"', sla_section)  # U row without an SLA is listed, never plotted
    assert "no SLA stated in rubric row" in sla_section and "U rows are never scored and never counted as S0." in html_text


# ---------- cost per client ----------


def priced_gate(img, image_id, *, backend, log, no_damage_min_conf):
    idx = int(image_id.split("_")[-1])
    usd = round(0.001 * (idx + 1), 6)
    log.record(stage="gate", model="fake-gate", image_id=image_id, input_tokens=10, output_tokens=5, seconds=0.5, usd=usd)
    return GateRecord(image_id=image_id, model="fake-gate", seconds=0.5, usd=usd, usable=True, damage_present=True, confidence=0.9, reason="rust", routed=True)


def priced_grade(img, **kw):
    idx = int(kw["image_id"].split("_")[-1])
    f = fake_grade(img, **kw)
    kw["log"].rows[-1]["usd"] = round(0.01 * (idx + 1), 6)
    return f


def test_cost_per_client_equals_sum_of_its_call_rows(tmp_path):
    recs = client_records(tmp_path, n=4)
    out = tmp_path / "run"
    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=priced_gate, grade_fn=priced_grade)
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    totals = {}
    for cid in ("acme", "beta"):
        ids = {r.image_id for r in recs if client_of(r) == cid}
        m = client_metrics(out, cid, {r.image_id: r for r in recs})
        expected = sum(c["usd"] for c in calls if c["image_id"] in ids)
        assert m["usd_total"] == round(expected, 4) and m["cost_source"] == "calls.jsonl" and m["calls"] == len([c for c in calls if c["image_id"] in ids])
        assert m["seconds_gate"] == round(sum(c["seconds"] for c in calls if c["image_id"] in ids and c["stage"] == "gate"), 1)
        totals[cid] = m["usd_total"]
    assert totals["acme"] != totals["beta"] and totals["acme"] > 0


# ---------- thumbnail budget ----------


def test_thumbnail_budget_keeps_file_under_cap(tmp_path):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(80):  # noise so JPEG cannot compress it away; 800x600 keeps the test fast
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)).save(p, quality=85)
        recs.append(ImageRecord(image_id=f"img_{i}", path=str(p), sha256="0" * 64, width=800, height=600, asset_class="steel_coating", source_dataset="fake", split="dev"))
    out = run_fake(tmp_path, recs)
    budget = ThumbBudget(max_bytes=600_000)
    p = write_client_report(out, "fake", {r.image_id: r for r in recs}, budget=budget)
    m = json.loads((p.parent / "report.json").read_text(encoding="utf-8"))
    html_text = p.read_text(encoding="utf-8")
    assert p.stat().st_size <= 600_000 and 0 < m["thumbs"]["embedded"] < 60 and m["thumbs"]["encoding"] == "320px q60"
    assert "thumbnail omitted for file size" in html_text and len(m["thumbs"]["omitted"]) == 60 - m["thumbs"]["embedded"]
    assert len(list((p.parent / "evidence").glob("*.jpg"))) == 60 and m["findings"] == 79
    assert html_text.count("<article") == 79  # cards beyond the cap are still full text cards


# ---------- SVG helpers ----------


def test_svg_helpers_edge_cases():
    one = svg_donut([("a", 3, "#000000"), ("b", 0, "#ffffff")])
    assert one.startswith('<svg xmlns="http://www.w3.org/2000/svg"') and "<circle" in one and "<path" not in one
    assert "none" in svg_donut([("a", 0, "#000000")]) and "<path" in svg_donut([("a", 1, "#000000"), ("b", 2, "#ffffff")])
    bars = svg_hbar([(lvl, 0, LEVEL_COLOR[lvl]) for lvl in ("S0", "S1", "U")])
    assert 'width="-' not in bars and bars.count("<text") == 6 and "<title>" in bars
    tl = svg_timeline([TimelinePoint(x=-2, label="#1", color="#dc2626", y_lane=0), TimelinePoint(x=80, label="#2", color="#f59e0b", y_lane=2)], lanes=list(ACTION_ORDER))
    lane_pos = [tl.index(f">{lane}</text>") for lane in ACTION_ORDER]
    assert lane_pos == sorted(lane_pos) and "today" in tl and tl.count("<circle") == 2 and "<script" not in tl
    assert svg_timeline([], lanes=["x"]).count("<circle") == 0


# ---------- SLA anchoring with today injection ----------


def test_today_injection_and_anchor_basis(tmp_path):
    recs = make_records(tmp_path, n=3)
    recs[2].captured_on = None  # img_2 falls back to the run date; img_0 is not routed, img_1 is U
    out = run_fake(tmp_path, recs)
    run_date = datetime.fromisoformat(json.loads((out / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0])["ts"]).date()
    m = client_metrics(out, "fake", {r.image_id: r for r in recs}, today=run_date + timedelta(days=10))
    row = next(r for r in m["sla"] if r["image_id"] == "img_2")
    assert row["sla_days"] == 90 and row["due_in_days"] == 80 and row["anchor_basis"] == "run date"
    recs[2].captured_on = "2026-09-01"
    m = client_metrics(out, "fake", {r.image_id: r for r in recs}, today=date(2026, 10, 5))
    row = next(r for r in m["sla"] if r["image_id"] == "img_2")
    assert row["due_on"] == "2026-11-30" and row["due_in_days"] == 56 and row["anchor_basis"] == "capture date"
    u = next(r for r in m["sla"] if r["level"] == "U")
    assert u["sla_days"] is None and u["due_in_days"] is None


def test_workload_assumption_only_with_source(tmp_path):
    recs = make_records(tmp_path, n=2)
    out = run_fake(tmp_path, recs)
    plain = write_client_report(out, "fake", {r.image_id: r for r in recs}).read_text(encoding="utf-8")
    assert "[Assumption, source:" not in plain
    est = write_client_report(out, "fake", {r.image_id: r for r in recs}, workload=WorkloadAssumption(2.0, "test fixture")).read_text(encoding="utf-8")
    assert "If first-pass triage takes 2 min per image [Assumption, source: test fixture], the cleared images correspond to 2 min." in est


def test_rerun_is_idempotent_except_generated_at(tmp_path):
    recs = client_records(tmp_path, n=3)
    out = run_fake(tmp_path, recs)
    recs_by = {r.image_id: r for r in recs}
    p = write_client_report(out, "acme", recs_by, today=date(2026, 10, 5))
    first, m1 = p.read_text(encoding="utf-8"), json.loads((p.parent / "report.json").read_text(encoding="utf-8"))
    p = write_client_report(out, "acme", recs_by, today=date(2026, 10, 5))
    second, m2 = p.read_text(encoding="utf-8"), json.loads((p.parent / "report.json").read_text(encoding="utf-8"))
    assert first.replace(m1["generated_at"], "T") == second.replace(m2["generated_at"], "T")


# ---------- the real run, copied so the repo folder is untouched ----------


@pytest.mark.skipif(not (REAL_RUN / "findings.json").exists(), reason="runs/ui_0925_0856 not present")
def test_real_run_ui_0925_0856(tmp_path):
    run_dir = tmp_path / "ui_0925_0856"
    shutil.copytree(REAL_RUN, run_dir)
    paths = write_client_reports(run_dir, today=date(2026, 9, 25))
    assert set(paths) == {"dacl10k"}
    m = json.loads((paths["dacl10k"].parent / "report.json").read_text(encoding="utf-8"))
    assert m["grouping_basis"] == "source_dataset" and m["images"] == 10 and m["gated"] == 10 and m["routed"] == 9 and m["not_routed"] == 1
    assert m["findings"] == 14 and m["levels"] == {"S0": 4, "S1": 3, "S2": 3, "S3": 0, "S4": 0, "U": 4}
    assert m["usd_total"] == 0.8161 and m["models"] == {"gate": ["claude-haiku-4-5"], "grade": ["claude-opus-5"]}
    assert m["reviews"]["pending"] == 14 and len(m["assets"]) == 10 and m["sources"]["licenses"] == {"dacl10k": "CC BY-NC 4.0"}
    assert len(m["u_rows"]) == 4 and {r["action"] for r in m["u_rows"]} == {"monitor", "record"}
    dated = [r for r in m["sla"] if r["action"] == "schedule" and r["sla_days"] == 90]
    assert len(dated) == 2 and all(r["due_on"] == "2026-12-24" and r["anchor_basis"] == "run date" for r in dated)
    html_text = paths["dacl10k"].read_text(encoding="utf-8")
    assert forbidden(html_text) == [] and paths["dacl10k"].stat().st_size < 5_000_000 and "grouped by source_dataset" in html_text
    assert "demo only" in html_text and "AASHTO Manual for Bridge Element Inspection" in html_text
    for lvl, meaning in LEVEL_MEANING.items():
        assert meaning in html_text, lvl
