"""Crack metrology tests: synthetic PIL fixtures with known geometry, no model, no network.

Fixture: a dark meandering polyline of nominal width 6 px on a light textured background plus a
mid-gray shadow ellipse that must be rejected as non-elongated. The drawn path length is computed
from the polyline vertices, so every tolerance below compares against a known number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from cascade import measure as M
from cascade.schema import ImageRecord, Measurements

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = json.loads((ROOT / "src" / "cascade" / "rubrics" / "bridge_mbei.json").read_text(encoding="utf-8"))
MANIFEST = ROOT / "data" / "demo" / "dacl10k" / "manifest.jsonl"

CRACK_PTS = [(20, 60), (70, 90), (120, 70), (170, 120), (220, 100), (270, 150), (300, 130)]
CRACK_WIDTH_PX = 6
BLOB_BOX = (40, 160, 110, 215)  # x0, y0, x1, y1 of the shadow ellipse; the crack never enters it


def drawn_path_length(pts=CRACK_PTS) -> float:
    return float(sum(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)))


def synthetic_crack(seed: int = 0, with_blob: bool = True) -> Image.Image:
    rng = np.random.default_rng(seed)
    bg = (200 + rng.normal(0, 8, (240, 320))).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(bg).convert("RGB")
    d = ImageDraw.Draw(img)
    d.line(CRACK_PTS, fill=(40, 40, 40), width=CRACK_WIDTH_PX)
    if with_blob:
        d.ellipse(BLOB_BOX, fill=(130, 130, 130))
    return img


def rubric_row(defect: str, value: str) -> dict:
    return next(r for r in RUBRIC["rows"] if r["defect"] == defect and r["value"] == value)


# --- scale ---------------------------------------------------------------------------------------


def test_scale_from_points_200px_segment_100mm_is_half_mm_per_px():
    s = M.scale_from_points((10, 20), (210, 20), 100.0)
    assert s.mm_per_px == pytest.approx(0.5)
    assert s.basis == "manual_two_points"
    assert "100.0 mm" in s.detail and "200.0 px" in s.detail
    assert 0.0 < s.confidence < 1.0
    diag = M.scale_from_points((0, 0), (30, 40), 25.0)  # 50 px hypotenuse
    assert diag.mm_per_px == pytest.approx(0.5)


def test_scale_from_points_rejects_degenerate_input():
    with pytest.raises(ValueError):
        M.scale_from_points((5, 5), (5, 5), 10.0)
    with pytest.raises(ValueError):
        M.scale_from_points((0, 0), (10, 0), 0.0)


def test_scale_from_gsd_reads_record_or_returns_none():
    rec = ImageRecord(image_id="a", path="a.jpg", sha256="0" * 64, width=10, height=10, asset_class="bridge_element", gsd_mm_per_px=0.2)
    s = M.scale_from_gsd(rec)
    assert s is not None and s.mm_per_px == 0.2 and s.basis == "gsd_metadata" and "a" in s.detail
    assert M.scale_from_gsd(None) is None
    assert M.scale_from_gsd(rec.model_copy(update={"gsd_mm_per_px": None})) is None


def test_scale_from_reference_uses_given_endpoints_and_checks_bounds():
    img = Image.new("RGB", (300, 200), (220, 220, 220))
    s = M.scale_from_reference(img, [(10, 10), (210, 10)], 100.0, "comparator card 100 mm bar")
    assert s.mm_per_px == pytest.approx(0.5)
    assert s.basis == "scale_object" and "comparator card" in s.detail
    with pytest.raises(ValueError):
        M.scale_from_reference(img, [(10, 10), (400, 10)], 100.0, "off image")
    with pytest.raises(ValueError):
        M.scale_from_reference(img, [(10, 10)], 100.0, "one point")


# --- mask and skeleton ---------------------------------------------------------------------------


def test_skeletonize_bar_is_one_pixel_wide_and_connected():
    bar = np.zeros((40, 100), dtype=bool)
    bar[15:21, 10:90] = True
    sk = M.skeletonize(bar)
    rows = np.unique(np.nonzero(sk)[0])
    assert len(rows) == 1 and 15 <= rows[0] <= 20
    assert 70 <= sk.sum() <= 80  # ends shortened by about half the width
    assert not M.skeletonize(np.zeros((5, 5), dtype=bool)).any()


def test_crack_mask_keeps_line_and_rejects_shadow_blob():
    img = synthetic_crack()
    m = M.crack_mask(img)
    assert m.shape == (240, 320)
    assert m[75, 45]  # on the first segment
    x0, y0, x1, y1 = BLOB_BOX
    assert m[y0 : y1 + 1, x0 : x1 + 1].sum() == 0  # neither the blob nor its rim survives
    assert 0.8 * CRACK_WIDTH_PX * drawn_path_length() < m.sum() < 1.3 * CRACK_WIDTH_PX * drawn_path_length()


def test_crack_mask_bbox_returns_crop_sized_mask():
    img = synthetic_crack()
    m = M.crack_mask(img, bbox=(0, 40, 160, 140))
    assert m.shape == (100, 160)
    assert m.sum() > 100


# --- metrics --------------------------------------------------------------------------------------


def test_synthetic_crack_metrics_in_px_without_scale():
    m = M.crack_mask(synthetic_crack())
    met = M.crack_metrics(m, None)
    L = drawn_path_length()
    assert abs(met["width_px_max"] - CRACK_WIDTH_PX) <= 1.0
    assert abs(met["width_px_p95"] - CRACK_WIDTH_PX) <= 1.0
    assert abs(met["length_px"] - L) <= 0.10 * L
    assert 0.8 * CRACK_WIDTH_PX * L < met["area_px"] < 1.3 * CRACK_WIDTH_PX * L
    assert met["endpoints"] == 2 and met["junctions"] == 0
    assert met["measurable"] is False
    assert any("scale missing" in r for r in met["reasons"])
    assert met["crack_width_mm_p95"] is None and met["area_cm2"] is None and met["scale_basis"] is None


def test_synthetic_crack_with_half_mm_scale_gives_3mm_and_quotes_cs3_row():
    img = synthetic_crack()
    scale = M.Scale(mm_per_px=0.5, basis="gsd_metadata", detail="test", confidence=0.8)
    cm = M.measure_crack(img, scale=scale)
    assert cm.measurable is True
    assert cm.crack_width_mm_p95 == pytest.approx(3.0, abs=0.5)
    assert cm.crack_width_mm_max == pytest.approx(cm.width_px_max * 0.5)
    assert cm.crack_length_mm == pytest.approx(cm.length_px * 0.5)
    assert cm.area_cm2 == pytest.approx(cm.area_px * 0.25 / 100.0)
    assert cm.width_uncertainty_mm == pytest.approx(0.5)
    assert cm.scale_basis == "gsd_metadata" and cm.mm_per_px == 0.5
    hint = cm.mbei_condition_state_hint
    row = rubric_row("cracking (reinforced concrete)", "CS3")
    assert hint["value"] == "CS3"
    assert hint["criterion"] == row["criterion"]  # verbatim rubric text
    assert hint["unified"] == row["unified"] == "S2"
    assert hint["thresholds_in"] == {"CS1_below": 0.012, "CS3_above": 0.05}
    assert hint["band_states"] == ["CS3"]
    assert 0.0 < cm.confidence <= 0.6 * 0.8 + 1e-9
    assert any("p95" in n for n in cm.notes)


def test_mbei_hint_reports_both_states_when_band_straddles_boundary():
    # 0.30 mm = 0.0118 in sits just under the RC CS1/CS2 boundary; +/- 0.05 mm crosses it
    h = M.mbei_hint(0.30, 0.05, "reinforced")
    assert h["value"] == "CS1" and h["band_states"] == ["CS1", "CS2"]
    assert h["criterion"] == rubric_row("cracking (reinforced concrete)", "CS1")["criterion"]
    assert "straddles" in h["note"]
    # prestressed ladder: 0.15 mm = 0.0059 in falls in CS2 (0.004 to 0.009 in)
    p = M.mbei_hint(0.15, 0.01, "prestressed")
    assert p["value"] == "CS2" and p["criterion"] == rubric_row("cracking (prestressed concrete)", "CS2")["criterion"]
    assert p["thresholds_in"] == {"CS1_below": 0.004, "CS3_above": 0.009}


def test_to_measurements_uses_p95_not_max_and_penalises_manual_scale():
    img = synthetic_crack()
    gsd = M.measure_crack(img, scale=M.Scale(0.5, "gsd_metadata", "t", M.SCALE_CONFIDENCE["gsd_metadata"]))
    manual = M.measure_crack(img, scale=M.scale_from_points((0, 0), (200, 0), 100.0))
    d = M.to_measurements(gsd)
    assert d["crack_width_mm"] == pytest.approx(gsd.crack_width_mm_p95)
    assert d["crack_width_mm"] < gsd.crack_width_mm_max
    assert d["area_cm2"] == pytest.approx(gsd.area_cm2)
    assert d["delta_t_k"] is None and d["percent_area_rusted"] is None and d["section_loss_pct"] is None
    assert any("p95" in n and "not max" in n for n in d["notes"])
    parsed = Measurements.model_validate(d)  # extra "notes" key is ignored by the contract
    assert parsed.crack_width_mm == pytest.approx(d["crack_width_mm"])
    dm = M.to_measurements(manual)
    assert dm["confidence"] < d["confidence"]
    assert any("manual" in n for n in dm["notes"])


def test_blank_image_is_not_measurable_with_reason():
    blank = Image.new("RGB", (200, 200), (200, 200, 200))
    cm = M.measure_crack(blank, scale=M.Scale(0.5, "gsd_metadata", "t", 0.8))
    assert cm.measurable is False
    assert cm.width_px_max is None and cm.crack_width_mm_p95 is None and cm.mbei_condition_state_hint is None
    assert cm.confidence == 0.0
    assert any("mask too small" in n for n in cm.notes)
    d = M.to_measurements(cm)
    assert d["crack_width_mm"] is None and d["area_cm2"] is None


def test_measure_crack_uses_record_gsd_and_bbox(tmp_path):
    img = synthetic_crack()
    p = tmp_path / "crack.png"
    img.save(p)
    rec = ImageRecord(image_id="syn", path=str(p), sha256="0" * 64, width=320, height=240, asset_class="bridge_element", gsd_mm_per_px=0.25)
    cm = M.measure_crack(p, record=rec, bbox=(0, 40, 160, 140))
    assert cm.scale_basis == "gsd_metadata" and cm.mm_per_px == 0.25
    assert cm.bbox == [0, 40, 160, 140]
    assert cm.measurable is True
    assert cm.crack_width_mm_p95 == pytest.approx(1.5, abs=0.3)
    explicit = M.measure_crack(p, record=rec, scale=M.Scale(1.0, "scale_object", "given", 0.7))
    assert explicit.scale_basis == "scale_object"  # explicit scale wins over record metadata


def test_cli_prints_json(tmp_path, capsys):
    p = tmp_path / "crack.png"
    synthetic_crack().save(p)
    assert M.main(["--image", str(p), "--points", "0", "0", "200", "0", "--mm", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["measurement"]["scale_basis"] == "manual_two_points"
    assert out["measurement"]["crack_width_mm_p95"] == pytest.approx(3.0, abs=0.5)
    assert out["measurements_contract"]["crack_width_mm"] == pytest.approx(out["measurement"]["crack_width_mm_p95"])
    assert M.main(["--image", str(p), "--gsd", "0.5", "--bbox", "0", "40", "160", "140"]) == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["measurement"]["bbox"] == [0, 40, 160, 140] and out2["measurement"]["scale_basis"] == "gsd_metadata"


@pytest.mark.skipif(not MANIFEST.exists(), reason="dacl10k demo manifest not present")
def test_dacl10k_demo_crack_image_runs_px_only_without_scale():
    rec = None
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "Crack" in row.get("labels", {}).get("classes_present", []):
            rec = ImageRecord(**row)
            break
    if rec is None or not Path(rec.path).exists():
        pytest.skip("no dacl10k demo image with a Crack label on disk")
    assert rec.gsd_mm_per_px is None  # dacl10k ships no scale (research note 10 section 4.2)
    cm = M.measure_crack(rec.path, record=rec)
    assert cm.scale_basis is None and cm.mm_per_px is None
    assert cm.measurable is False
    assert cm.crack_width_mm_p95 is None and cm.crack_length_mm is None and cm.area_cm2 is None
    assert cm.mbei_condition_state_hint is None
    assert cm.width_px_max is not None and cm.width_px_max > 0 and cm.length_px > 0
    assert any("scale missing" in n for n in cm.notes)
    d = M.to_measurements(cm)
    assert d["crack_width_mm"] is None
    Measurements.model_validate(d)
