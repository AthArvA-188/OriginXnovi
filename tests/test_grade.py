"""Deterministic U guards in grade.grade_image: no model call (the backend is replaced by a fake)."""

import pytest
from PIL import Image

from cascade import grade
from cascade.grade import build_system, grade_image, load_rubric, machinery_reading_missing
from cascade.schema import GraderOutput


def _out(standard, value, level, criterion, action="record"):
    return GraderOutput.model_validate({
        "defect_type": "none",
        "native_scale": {"standard": standard, "value": value, "criteria_matched": [criterion]},
        "unified": {"level": level, "uncertainty": "+/-1", "flags": []},
        "measurements": {"area_cm2": None, "crack_width_mm": None, "delta_t_k": None, "percent_area_rusted": None, "section_loss_pct": None, "confidence": 0.9},
        "action": {"code": action, "sla_days": None, "basis": f"row {value}"},
        "justification": "fake",
    })


def _fake(out, calls):
    def fake(img, asset_class, rubric, metadata, exemplars=None, model=None):
        calls.append(asset_class)
        return out, {"input_tokens": 0, "output_tokens": 0}, "end_turn", {}
    return fake


IMG = Image.new("RGB", (64, 64), (128, 128, 128))


def test_machinery_without_reading_is_u_and_never_calls_the_model(monkeypatch):
    rb = load_rubric("interior_machinery")
    zone_a = next(r for r in rb["rows"] if r["value"] == "Zone A")
    calls = []
    monkeypatch.setattr(grade, "grade_claude", _fake(_out("ISO-20816-3", "Zone A", "S0", zone_a["criterion"]), calls))
    f = grade_image(IMG, finding_id="pump/full", image_id="pump", asset_class="interior_machinery", metadata={"asset_class": "interior_machinery"})
    assert f.unified.level == "U" and "not_measurable" in f.unified.flags and f.measurements.confidence == 0.0
    assert "rms_velocity_mm_s" in f.justification and calls == []


def test_machinery_with_full_reading_goes_to_the_grader(monkeypatch):
    rb = load_rubric("interior_machinery")
    zone_b = next(r for r in rb["rows"] if r["value"] == "Zone B")
    calls = []
    monkeypatch.setattr(grade, "grade_claude", _fake(_out("ISO-20816-3", "Zone B", "S1", zone_b["criterion"], "monitor"), calls))
    meta = {"rms_velocity_mm_s": 2.0, "machine_group": 2, "support_type": "rigid"}
    assert machinery_reading_missing(meta) is None
    assert machinery_reading_missing({"rms_velocity_mm_s": "n/a", "rated_kw": 50, "support_type": "rigid"}) == "rms_velocity_mm_s"
    f = grade_image(IMG, finding_id="pump/full", image_id="pump", asset_class="interior_machinery", metadata=meta)
    assert f.unified.level == "S1" and calls == ["interior_machinery"]


def test_sonar_s0_becomes_u_and_sonar_defect_grade_is_kept(monkeypatch):
    rb = load_rubric("underwater_structure")
    good = next(r for r in rb["rows"] if r["value"] == "Good")
    monkeypatch.setattr(grade, "grade_claude", _fake(_out("NBIS-UW", "Good", "S0", good["criterion"]), []))
    f = grade_image(IMG, finding_id="pile/full", image_id="pile", asset_class="underwater_structure", modality="sonar")
    assert f.unified.level == "U" and f.modality == "sonar" and "not_measurable" in f.unified.flags
    scr = next(r for r in rb["rows"] if r["value"] == "SCR 4-5")
    monkeypatch.setattr(grade, "grade_claude", _fake(_out("NBIS-UW", "SCR 4-5", "S2", scr["criterion"], "schedule"), []))
    f2 = grade_image(IMG, finding_id="pile/full", image_id="pile", asset_class="underwater_structure", metadata={"modality": "sonar"})
    assert f2.unified.level == "S2" and f2.modality == "sonar"
    # an optical ROV frame may still grade Good / S0
    monkeypatch.setattr(grade, "grade_claude", _fake(_out("NBIS-UW", "Good", "S0", good["criterion"]), []))
    assert grade_image(IMG, finding_id="p/full", image_id="p", asset_class="underwater_structure").unified.level == "S0"


def test_underwater_rubric_has_a_u_row_and_prompt_does_not_force_s0():
    rb = load_rubric("underwater_structure")
    assert "U" in rb["allowed_values"]
    u_row = next(r for r in rb["rows"] if r["value"] == "U")
    assert u_row["unified"] == "U" and "never Good" in u_row["criterion"]
    assert "Optical" in next(r for r in rb["rows"] if r["value"] == "Good")["criterion"]
    system = build_system("interior_machinery", load_rubric("interior_machinery"))
    assert "use the lowest native value" not in system and "never S0" in system


@pytest.mark.parametrize("asset_class", ["underwater_structure", "interior_machinery"])
def test_new_rubric_values_are_allowed(asset_class):
    rb = load_rubric(asset_class)
    for r in rb["rows"]:
        assert r["value"] in rb["allowed_values"]
