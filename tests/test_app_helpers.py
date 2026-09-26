"""Pure helpers of app/streamlit_app.py (crack measurement and Sensors tab).

The app is a Streamlit script that renders on import, so the helpers under test are lifted out of its source with
ast and executed in a small namespace holding their imports. Only side-effect-free functions are extracted.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade.measure import SCALE_CONFIDENCE, Scale, scale_from_points  # noqa: E402
from cascade.schema import Measurements  # noqa: E402
from cascade.signals import grade_signal, indicators, ingest_signal_csv, load_seismic_rubric  # noqa: E402

APP = ROOT / "app" / "streamlit_app.py"
FUNCS = {"modality_chip", "crack_width_text", "merged_measurements", "scale_from_args", "synthetic_series", "write_synthetic_pair"}
CONSTS = {"SYNTH"}


@pytest.fixture(scope="module")
def app():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in FUNCS) or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in CONSTS for t in n.targets))]
    assert {n.name for n in keep if isinstance(n, ast.FunctionDef)} == FUNCS
    ns = {"np": np, "pd": pd, "Path": Path, "Scale": Scale, "SCALE_CONFIDENCE": SCALE_CONFIDENCE, "scale_from_points": scale_from_points, "Measurements": Measurements}
    ns.update({"Optional": __import__("typing").Optional, "Tuple": __import__("typing").Tuple})
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(APP), "exec"), ns)
    return ns


def test_modality_chip_hides_rgb(app):
    assert app["modality_chip"]("rgb") == ""
    assert app["modality_chip"](None) == ""
    assert "sonar" in app["modality_chip"]("sonar")


def test_crack_width_text_never_bare_mm(app):
    m = Measurements(area_cm2=None, crack_width_mm=0.42, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=0.5,
                     measurement_basis="gsd_metadata", crack_width_uncertainty_mm=0.05)
    assert app["crack_width_text"](m) == "crack 0.42 ± 0.05 mm (scale: gsd_metadata)"
    model_only = m.model_copy(update={"measurement_basis": None, "crack_width_uncertainty_mm": None})
    assert "no measured scale basis" in app["crack_width_text"](model_only)
    none = m.model_copy(update={"crack_width_mm": None})
    assert "not_measurable" in app["crack_width_text"](none)


def test_merged_measurements_keeps_grader_fields(app):
    old = Measurements(area_cm2=12.0, crack_width_mm=None, delta_t_k=None, percent_area_rusted=3.0, section_loss_pct=None, confidence=0.7)
    tm = {"area_cm2": 4.8, "crack_width_mm": 1.8, "crack_length_mm": 638.0, "crack_width_uncertainty_mm": 0.25, "measurement_basis": "manual_two_points", "confidence": 0.27, "notes": ["x"]}
    new = app["merged_measurements"](old, tm)
    assert (new.crack_width_mm, new.crack_width_uncertainty_mm, new.crack_length_mm, new.measurement_basis) == (1.8, 0.25, 638.0, "manual_two_points")
    assert new.area_cm2 == 12.0  # the grader's area is not overwritten by a crack-mask area
    assert new.confidence == 0.7 and new.percent_area_rusted == 3.0
    filled = app["merged_measurements"](old.model_copy(update={"area_cm2": None}), tm)
    assert filled.area_cm2 == 4.8


def test_scale_from_args(app):
    assert app["scale_from_args"](None) is None
    s = app["scale_from_args"](("gsd", 0.5, "meta"))
    assert (s.basis, s.mm_per_px) == ("gsd_metadata", 0.5)
    p = app["scale_from_args"](("points", 100, 50, 500, 50, 100.0))
    assert p.basis == "manual_two_points" and p.mm_per_px == pytest.approx(0.25)
    with pytest.raises(ValueError):
        app["scale_from_args"](("points", 0, 0, 0, 0, 100.0))
    with pytest.raises(ValueError):
        app["scale_from_args"](("points", 0, 0, 10, 0, 0.0))


def test_synthetic_pair_grades_shift_and_u_without_baseline(app, tmp_path):
    cur, base = app["write_synthetic_pair"](tmp_path)
    assert "synthetic" in cur.name and "synthetic" in base.name
    kw = dict(units="m/s2", mount="structure")
    b = ingest_signal_csv(base, "accelerometer", 200.0, **kw)
    r = ingest_signal_csv(cur, "accelerometer", 200.0, baseline_id=b.signal_id, **kw)
    ind = indicators(r, b)
    expected = (app["SYNTH"]["current_hz"] - app["SYNTH"]["baseline_hz"]) / app["SYNTH"]["baseline_hz"] * 100
    for ch in r.channels:
        assert ind["frequency_shift_pct"][ch] == pytest.approx(expected, abs=ind["frequency_shift_uncertainty_pct"][ch])
    f = grade_signal(r, ind, load_seismic_rubric())
    assert (f.unified.level, f.native_scale.value) == ("S2", "df 5-10%")
    u = grade_signal(r, indicators(r, None), load_seismic_rubric())
    assert u.unified.level == "U"
