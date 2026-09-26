"""Facade and interior grading for the building layer (building_spec sections 6, 9 and 13).

Every tower here is SYNTHETIC (drawn facades, labelled synthetic). The fake grader below looks at pixels, not at
labels, and no model API is called. These tests check plumbing and the U rules; they never claim accuracy.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from building_fixtures import small_tower  # noqa: F401
from cascade import drift
from cascade.building import grading as gr
from cascade.building.blueprint import FILES, load_building_dir, place
from cascade.building.model import ElectricalResult, WaterResult
from cascade.building.problems import analyze, build_problems, default_now, image_observations, level_counts
from cascade.building.rules import THRESHOLDS
from cascade.grade import RUBRIC_FOR_CLASS, build_system, load_rubric
from cascade.pipeline import RunConfig
from cascade.schema import LEVEL_ORDER, Action, AssetClass, Evidence, Finding, ImageRecord, Measurements, NativeScale, Unified

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "building"
NEW = ("facade_element", "interior_zone", "electrical_equipment")
RANK = {**LEVEL_ORDER}


# ------------------------------------------------------------------------------------------ helpers

def graded(asset_class: str, value: str, *, level=None, modality="rgb", finding_id="f1", delta=None) -> Finding:
    """A finding as a grader would return it: the row criterion quoted verbatim."""
    rub = load_rubric(asset_class)
    row = next(r for r in rub["rows"] if r["value"] == value)
    return Finding(
        finding_id=finding_id, asset_class=asset_class, defect_type="test",
        native_scale=NativeScale(standard=rub["standard"], value=value, criteria_matched=[row["criterion"]]),
        unified=Unified(level=level or row["unified"], uncertainty="+/-1", flags=[]),
        measurements=Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=delta, percent_area_rusted=None, section_loss_pct=None, confidence=0.8),
        action=Action(code=row["action"], sla_days=None, basis=f"row {value}"), justification="test finding",
        evidence=Evidence(image_ids=["img"]), modality=modality, model="fake-grader")


def brown_stain_px(img) -> int:
    """Pixels in the brown water-stain colour range of the synthetic drawings (a pixel rule, not a label read)."""
    a = np.asarray(img.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return int(((r - b >= 50) & (g - b >= 25) & (r > 140) & (r > g)).sum())


def fake_pixel_grade(img, *, finding_id, image_id, asset_class, backend, rubric, metadata, exemplars, evidence, log):
    """Fake grader for SYNTHETIC drawings: a brown stain gives SWARMP, a clean wall gives Safe. No model call."""
    value = "SWARMP" if brown_stain_px(img) > 1000 else "Safe"
    row = next(r for r in rubric["rows"] if r["value"] == value)
    return Finding(
        finding_id=finding_id, asset_class=asset_class, defect_type="facade distress" if value != "Safe" else "none",
        native_scale=NativeScale(standard=rubric["standard"], value=value, criteria_matched=[row["criterion"]]),
        unified=Unified(level=row["unified"], uncertainty="+/-1", flags=[]),
        measurements=Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=0.7),
        action=Action(code=row["action"], sla_days=None, basis=f"facade_ll11.json row '{value}'"),
        justification=f"fake pixel grader on a synthetic drawing: {brown_stain_px(img)} stain-colour px",
        evidence=evidence, model="fake-grader")


def truth(root: Path) -> dict:
    return {s["scenario_id"]: s for s in json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))["scenarios"]}


@pytest.fixture(scope="module")
def graded_small(small_tower, tmp_path_factory):  # noqa: F811
    """Photos of a COPY of the small synthetic tower graded by the fake grader (the shared fixture stays untouched)."""
    root = tmp_path_factory.mktemp("graded_tower") / "tower"
    shutil.copytree(small_tower.root, root)
    data = load_building_dir(root)
    summary, findings, obs = gr.grade_building_photos(data, cfg=RunConfig(gate="none"), grade_fn=fake_pixel_grade)
    return root, data, summary, findings, obs


# ------------------------------------------------------------------------------------------ rubrics

def test_old_asset_classes_unchanged():
    old = {"bridge_element", "steel_coating", "pv_module", "building_disaster", "underwater_structure", "interior_machinery"}
    assert old <= set(get_args(AssetClass))
    assert set(NEW) <= set(get_args(AssetClass))
    assert set(get_args(AssetClass)) <= set(RUBRIC_FOR_CLASS)


def test_rubrics_load_values_allowed_and_u_rows():
    for ac in NEW:
        rub = load_rubric(ac)
        allowed = rub["allowed_values"]
        assert allowed[0] == "U"
        assert {r["value"] for r in rub["rows"]} <= set(allowed)
        u_rows = [r for r in rub["rows"] if r["value"] == "U"]
        assert u_rows and all(r["unified"] == "U" for r in u_rows)
        assert not any(r["unified"] == "S0" and r["value"] == "U" for r in rub["rows"])
    # the S4-eligible value is last (drift._audit_one allows S4 only there)
    assert load_rubric("facade_element")["allowed_values"][-1] == "Unsafe"
    assert load_rubric("interior_zone")["allowed_values"][-1] == "Active leak"


def test_every_row_is_sourced():
    keys = set(THRESHOLDS)
    for ac in NEW:
        for r in load_rubric(ac)["rows"]:
            text = f"{r['criterion']} {r.get('source', '')}"
            has_url = "https://" in text
            has_key = any(re.search(rf"\b{k}\b", text) for k in keys)
            has_tag = any(tag in text for tag in ("[team-proposed, validate]", "[Assumption]", "[PUBLIC"))
            assert has_url or has_key or has_tag, f"{ac} row {r['value']} has no URL, threshold key or tag"


def test_frozen_row_contract_matches_spec():
    """family, value, min, max, unified and action are coded against by water.py and electrical.py."""
    def sig(ac):
        return [(r["family"], r["value"], r.get("min"), r.get("max"), r["unified"], r["action"]) for r in load_rubric(ac)["rows"]]
    assert sig("facade_element") == [("image", "Safe", None, None, "S0", "record"), ("image", "SWARMP", None, None, "S2", "schedule"),
                                     ("image", "Unsafe", None, None, "S3", "prioritize"), ("not_assessable", "U", None, None, "U", "monitor")]
    iw = sig("interior_zone")
    assert ("rh_hourly", "RH ok", None, 60, "S0", "record") in iw
    assert ("rh_sustained_h", "RH elevated", 72, None, "S2", "schedule") in iw
    assert ("rh_30d_mean", "RH mold-risk", 80, None, "S3", "prioritize") in iw
    assert ("wet_duration_h", "Drying window exceeded", 48, None, "S3", "prioritize") in iw
    assert ("leak_sensor", "Active leak", 1, None, "S3", "prioritize") in iw
    assert ("not_assessable", "U", None, None, "U", "monitor") in iw


def test_build_system_quotes_fisp_safe_verbatim():
    safe = ("A condition of a building wall, any appurtenances thereto or any part thereof not requiring repair or maintenance "
            "to sustain the structural integrity of the exterior of the building and that will not become unsafe during the next five years.")
    system = build_system("facade_element", load_rubric("facade_element"))
    assert safe in system
    assert "NYC-FISP" in system


# ------------------------------------------------------------------------------------------ FISP rules

def test_fisp_carryover():
    assert gr.fisp_carryover("SWARMP", "SWARMP") == "Unsafe"
    assert gr.fisp_carryover("SWARMP", "Unsafe") == "Unsafe"
    assert gr.fisp_carryover("SWARMP", "Safe") == "Safe"  # corrected
    assert gr.fisp_carryover("SWARMP", "U") == "U"  # cannot tell whether corrected; never read as Safe
    assert gr.fisp_carryover(None, "SWARMP") == "SWARMP"
    assert gr.fisp_carryover("Safe", "SWARMP") == "SWARMP"
    f = gr.apply_fisp_carryover(graded("facade_element", "SWARMP"), "SWARMP")
    assert (f.native_scale.value, f.unified.level, f.action.code) == ("Unsafe", "S3", "prioritize")
    hard, _, _ = drift._audit_one(f, load_rubric("facade_element"))
    assert hard == []


def test_fisp_building_status():
    assert gr.fisp_building_status(["Safe", "U"]) == "Not determined (U present)"
    assert gr.fisp_building_status(["Safe", "Safe"]) == "Safe"
    assert gr.fisp_building_status(["Safe", "SWARMP", "U"]) == "SWARMP"
    assert gr.fisp_building_status(["SWARMP", "Unsafe"]) == "Unsafe"
    assert gr.fisp_building_status(["U"]) == "Not determined (U present)"
    assert gr.fisp_building_status([]) == "Not determined (no photos)"


# ------------------------------------------------------------------------------------------ U guards

def test_electrical_s0_without_readings_is_u():
    f = graded("electrical_equipment", "Band 0", modality="thermal")
    out = gr.enforce_u_guards(f, {"image_size": "640x480"})
    assert out.unified.level == "U" and "not_measurable" in out.unified.flags
    assert f.unified.level == "S0"  # input not mutated
    # a band without readings is U too (electrical_thermal.json measurement_note)
    assert gr.enforce_u_guards(graded("electrical_equipment", "Band 2", modality="thermal"), {}).unified.level == "U"


def test_electrical_readings_set_delta_and_band():
    rub = load_rubric("electrical_equipment")
    meta = {"t_element_c": 58.0, "t_reference_c": 38.0, "load_pct": 60}
    out = gr.enforce_u_guards(graded("electrical_equipment", "Band 1", modality="thermal"), meta)
    assert out.native_scale.value == "Band 3" and out.unified.level == "S3" and out.measurements.delta_t_k == 20.0
    assert drift._audit_one(out, rub)[0] == []  # H6 skipped for electrical delta-T, criterion verbatim
    same = gr.enforce_u_guards(graded("electrical_equipment", "Band 0", modality="thermal"), {**meta, "t_element_c": 38.5})
    assert same.native_scale.value == "Band 0" and same.measurements.delta_t_k == 0.5
    low = gr.enforce_u_guards(graded("electrical_equipment", "Band 3", modality="thermal"), {**meta, "load_pct": 20})
    assert low.unified.level == "U" and "IR_MIN_LOAD_PCT" in low.justification


def test_facade_safe_needs_fisp_photo_size():
    small = gr.enforce_u_guards(graded("facade_element", "Safe"), {"image_size": "640x480"})
    assert small.unified.level == "U"
    assert gr.enforce_u_guards(graded("facade_element", "Safe"), {}).unified.level == "U"  # unknown size
    ok = gr.enforce_u_guards(graded("facade_element", "Safe"), {"image_size": "800x600"})
    assert ok.unified.level == "S0" and ok.native_scale.value == "Safe"
    assert gr.enforce_u_guards(graded("facade_element", "Safe"), {"width": 600, "height": 800}).unified.level == "S0"
    defect = gr.enforce_u_guards(graded("facade_element", "SWARMP"), {"image_size": "640x480"})
    assert defect.unified.level == "S2"  # a visible defect on a small photo is still reported


def test_photo_cannot_grade_sensor_rows():
    out = gr.enforce_u_guards(graded("interior_zone", "RH ok"), {"image_size": "1024x768"})
    assert out.unified.level == "U"
    assert gr.enforce_u_guards(graded("interior_zone", "Stain"), {}).unified.level == "S1"
    sensor = graded("interior_zone", "RH ok", modality="sensor")
    assert gr.enforce_u_guards(sensor, {}).unified.level == "S0"  # deterministic sensor findings pass through


def test_h6_still_applies_to_pv():
    f = graded("electrical_equipment", "Band 1", delta=2.0)
    assert not [h for h in drift._audit_one(f, load_rubric("electrical_equipment"))[0] if h[0] == "H6_impossible_measurement"]
    pv_rub = load_rubric("pv_module")
    pv_row = pv_rub["rows"][0]
    pv = f.model_copy(update={"asset_class": "pv_module", "native_scale": NativeScale(standard=pv_rub["standard"], value=pv_row["value"], criteria_matched=[pv_row["criterion"]])})
    assert [h for h in drift._audit_one(pv, pv_rub)[0] if h[0] == "H6_impossible_measurement"]


# ------------------------------------------------------------------------------------------ records and observations

def test_building_image_records(small_tower):  # noqa: F811
    b = small_tower.building
    recs = gr.building_image_records(b, small_tower.images)
    assert recs and all(r.asset_class == "facade_element" for r in recs)
    base = small_tower.images[0]
    room = base.model_copy(update={"image_id": "room", "asset_id": "F04-E", "asset_class": "bridge_element"})
    panel = base.model_copy(update={"image_id": "panel", "asset_id": "P-F04", "asset_class": "interior_zone"})
    panel_default = base.model_copy(update={"image_id": "panel2", "asset_id": "P-F04", "asset_class": "facade_element"})
    out = {r.image_id: r.asset_class for r in gr.building_image_records(b, [room, panel, panel_default])}
    assert out == {"room": "interior_zone", "panel": "interior_zone", "panel2": "electrical_equipment"}
    assert base.asset_class == "facade_element"  # inputs are not mutated
    with pytest.raises(ValueError, match="Valid ids"):
        gr.building_image_records(b, [base.model_copy(update={"asset_id": "NOPE-99"})])


def test_fake_graded_photo_becomes_observation_on_its_zone(small_tower):  # noqa: F811
    b = small_tower.building
    rec = gr.building_image_records(b, small_tower.images)[0]
    f = graded("facade_element", "SWARMP").model_copy(update={"evidence": Evidence(image_ids=[rec.image_id])})
    o = gr.finding_to_observation(f, rec, b)
    assert o.kind == "image_finding" and o.zone_id == rec.asset_id and o.element_id is None
    assert o.level == "S2" and o.synthetic and o.ts.startswith(rec.captured_on)
    floor, xy = place(b, o)
    assert floor == b.zone(rec.asset_id).floor_id and b.zone(rec.asset_id) is not None
    # problems.image_observations uses this converter
    via = image_observations(b, [f], [rec])
    assert [x.obs_id for x in via] == [o.obs_id] and via[0].text == o.text and "(NYC-FISP)" in via[0].text
    # an element photo sits on the element and its zone
    prec = rec.model_copy(update={"image_id": "p", "asset_id": "P-F04", "asset_class": "electrical_equipment", "captured_on": None})
    po = gr.finding_to_observation(graded("electrical_equipment", "Band 0", finding_id="p"), prec, b)
    assert po.element_id == "P-F04" and po.zone_id == b.element("P-F04").zone_id and "Capture date unknown" in po.text


# ------------------------------------------------------------------------------------------ S1 detection (SYNTHETIC)

def _check_s1(root: Path, findings, obs, b) -> None:
    s1 = truth(root)["S1"]
    roots = set(s1["root_nodes"])
    by_zone = {o.zone_id: o for o in obs}
    assert roots & set(by_zone), "the S1 facade drop photo produced no observation"
    for z in roots & set(by_zone):
        o = by_zone[z]
        assert RANK[o.level] >= RANK[s1["expected_level_min"]]
        assert o.finding.native_scale.value == "SWARMP"
    others = [o for z, o in by_zone.items() if z not in roots]
    assert others and all(o.level == "S0" and o.finding.native_scale.value == "Safe" for o in others)  # 800 x 600 drawings
    for f in findings:
        assert drift._audit_one(f, load_rubric(f.asset_class))[0] == []
    assert all(o.synthetic for o in obs)


def test_s1_facade_photo_detected_on_small_tower(graded_small):
    root, data, summary, findings, obs = graded_small
    assert len(findings) == len(data.images)
    _check_s1(root, findings, obs, data.building)
    empty = pd.DataFrame()
    probs = build_problems(data.building, data, WaterResult(), ElectricalResult(load_table=empty, anomalies=empty),
                           default_now(data), extra_obs=obs)
    s1 = truth(root)["S1"]
    facade = [p for p in probs if any(z in p.affected_nodes for z in s1["root_nodes"]) and p.level != "S0"]
    assert facade, "no problem on the S1 facade drop"
    top = facade[0]
    assert top.root_cause.node_id in s1["root_nodes"]
    assert RANK[top.level] >= RANK[s1["expected_level_min"]] and top.priority is not None
    counts = level_counts(probs)
    assert counts["U"] == sum(1 for p in probs if p.level == "U")


def test_u_photo_stays_u_in_problems(small_tower, tmp_path):  # noqa: F811
    """A Safe grade on a photo below FISP size becomes U, and that U never counts as S0."""
    b = small_tower.building
    rec = gr.building_image_records(b, small_tower.images)[-1].model_copy(update={"image_id": "tiny", "width": 640, "height": 480})
    fn = gr.guarded_grade_fn([rec], fake_pixel_grade)
    from PIL import Image
    f = fn(Image.new("RGB", (640, 480), (176, 170, 160)), finding_id="tiny/full", image_id="tiny", asset_class="facade_element",
           backend="none", rubric=load_rubric("facade_element"), metadata={}, exemplars=None, evidence=Evidence(image_ids=["tiny"]), log=None)
    assert f.unified.level == "U"
    o = gr.finding_to_observation(f, rec, b)
    assert o.level == "U"
    empty = pd.DataFrame()
    probs = build_problems(b, small_tower, WaterResult(), ElectricalResult(load_table=empty, anomalies=empty),
                           default_now(small_tower), extra_obs=[o])
    mine = [p for p in probs if o.obs_id in p.observation_ids]
    assert mine and all(p.level == "U" and p.priority is None for p in mine)
    counts = level_counts(mine)
    assert counts["S0"] == 0 and counts["U"] == len(mine)
    assert gr.fisp_building_status([f.native_scale.value]) == "Not determined (U present)"


def test_analyze_picks_up_graded_photos(graded_small):
    root, data, _, _, obs = graded_small
    assert (root / FILES["grading"] / "findings.json").exists()
    _, _, _, problems = analyze(root)
    s1 = truth(root)["S1"]
    s1_obs = {o.obs_id for o in obs if o.zone_id in s1["root_nodes"]}
    hits = [p for p in problems if s1_obs & set(p.observation_ids)]
    assert hits and hits[0].root_cause.node_id in s1["root_nodes"]
    lines = (root / FILES["observations"]).read_text(encoding="utf-8").splitlines()
    assert any(json.loads(x)["obs_id"] in s1_obs for x in lines)


@pytest.mark.skipif(not (DEMO / "ground_truth.json").exists(), reason="demo tower not generated (python -m cascade.building.synthetic)")
def test_s1_facade_photo_detected_on_demo_tower(tmp_path):
    """The 32-floor SYNTHETIC demo tower: graded into tmp_path, the demo folder is only read."""
    data = load_building_dir(DEMO)
    summary, findings, obs = gr.grade_building_photos(data, cfg=RunConfig(gate="none"), grade_fn=fake_pixel_grade, out=tmp_path / "grading")
    assert len(obs) == len(data.images) == 6
    _check_s1(DEMO, findings, obs, data.building)
