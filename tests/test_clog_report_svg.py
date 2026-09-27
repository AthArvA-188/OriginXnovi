"""Conclusions are derived from artifact values; the SVG diagram is well-formed."""
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from cascade.building.clog import report, svg

FX = json.loads((Path(__file__).parent / "fixtures" / "clog" / "report_inputs.json").read_text(encoding="utf-8"))


def _c(fx):
    return report.conclusions(fx["riser"], fx["drain"], fx["bell"], fx["sched"])


def test_conclusions_cover_topics_and_labels():
    cs = _c(FX)
    topics = [c["topic"] for c in cs]
    assert "Earliest supply-side signal" in topics and "Where to put sensors" in topics
    assert any(c["label"] == "REAL" for c in cs)
    assert all(c["text"] for c in cs)


def test_wording_follows_the_numbers():
    fx = copy.deepcopy(FX)
    det = fx["riser"]["variants"]["base"]["detectors"]
    det["act4_hgb"]["alarm"]["2"].update(rate=0.99, ci95=[0.97, 1.0])
    det["act4_z"]["alarm"]["2"].update(rate=0.10, ci95=[0.05, 0.15])
    txt = next(c["text"] for c in _c(fx) if c["topic"] == "What machine learning adds")
    assert txt.startswith("Detection: gradient boosting")
    det["act4_z"]["alarm"]["2"].update(rate=0.95, ci95=[0.90, 0.98])
    txt = next(c["text"] for c in _c(fx) if c["topic"] == "What machine learning adds")
    assert txt.startswith("Detection: the plain z-score rule and gradient boosting") and "overlap" in txt
    det["pas_hgb"]["alarm"]["3"]["rate"] = 0.9
    txt = next(c["text"] for c in _c(fx) if c["topic"] == "Earliest supply-side signal")
    assert "did no better" not in txt


def test_svg_parses_and_shows_values():
    s = svg.riser_svg({"L1": 0.5, "STR_M": 0.9}, "flagged")
    root = ET.fromstring(s)
    assert root.tag.endswith("svg")
    assert "L1 50%" in s and "90%" in s and "test valve" in s


def test_baseline_is_named_from_the_training_majority():
    fx = copy.deepcopy(FX)
    txt = next(c["text"] for c in _c(fx) if c["topic"] == "What machine learning adds")
    bl = fx["riser"]["baselines_macro_f1"]
    assert f"always '{bl['majority_class_name']}' (the training majority)" in txt
    assert "always-'clean' baseline" not in txt


def test_overlapping_localisation_cis_are_called_about_the_same():
    fx = copy.deepcopy(FX)
    det = fx["riser"]["variants"]["base"]["detectors"]
    det["act4_z"]["localisation"]["2"].update(accuracy=0.73, ci95=[0.65, 0.80])
    det["act4_hgb"]["localisation"]["2"].update(accuracy=0.70, ci95=[0.62, 0.78])
    txt = next(c["text"] for c in _c(fx) if c["topic"] == "What machine learning adds")
    assert "were about the same (73% vs 70%, overlapping 95% CIs" in txt
    det["act4_hgb"]["localisation"]["2"].update(accuracy=0.30, ci95=[0.20, 0.40])
    txt = next(c["text"] for c in _c(fx) if c["topic"] == "What machine learning adds")
    assert "was right more often (73% vs 30%" in txt
