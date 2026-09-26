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


# ---------------------------------------------------------------- Building tab helpers (D-015)

from types import SimpleNamespace  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

from building_fixtures import small_tower  # noqa: E402,F401  (session fixture, SYNTHETIC)
from cascade.building import blueprint as bbp, electrical as belec, graph as bgraph, problems as bprob, rules as brules, water as bwater  # noqa: E402
from cascade.building.model import Factor, Observation  # noqa: E402

BFUNCS = {"building_dirs", "building_input_mtimes", "open_level_tiles", "worst_level", "wind_sector", "factor_rows", "hours_text",
          "forecast_frame", "zone_pin_table", "electrical_tree", "basis_labels"}
BCONSTS = {"BUILDING_DEMO_LABEL", "BUILDING_INPUTS", "SIDE_ORDER", "SECTORS"}


@pytest.fixture(scope="module")
def bapp():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in BFUNCS)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in BCONSTS for t in n.targets))]
    assert {n.name for n in keep if isinstance(n, ast.FunctionDef)} == BFUNCS
    import re
    ns = {"np": np, "pd": pd, "Path": Path, "re": re, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "Sequence": Sequence,
          "Tuple": Tuple, "bbp": bbp, "belec": belec, "bgraph": bgraph, "bprob": bprob, "brules": brules, "bwater": bwater}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(APP), "exec"), ns)
    return ns


def test_worst_level_never_reads_u_or_tickets_as_s0(bapp):
    wl = bapp["worst_level"]
    assert wl(["S1", None, "U", "S3"]) == "S3"
    assert wl(["U", None]) == "U"
    assert wl([None, None]) is None
    assert wl([]) is None
    assert wl(["S0", "U"]) == "S0"  # a graded S0 is a grade; U stays out of it


def test_open_level_tiles_counts_u_separately_and_skips_closed(bapp):
    ps = [SimpleNamespace(level=lv, status=st) for lv, st in
          [("S3", "detected"), ("U", "detected"), ("U", "triaged"), ("S2", "resolved"), ("S4", "verified"), ("S1", "work_order")]]
    tiles = bapp["open_level_tiles"](ps)
    assert tiles == {"S4": 0, "S3": 1, "S2": 0, "S1": 1, "S0": 0, "U": 2}


def test_wind_sector(bapp):
    ws = bapp["wind_sector"]
    assert [ws(d) for d in (0, 22, 23, 90, 200, 203, 270, 359, 360, -45)] == ["N", "N", "NE", "E", "S", "SW", "W", "N", "N", "NW"]


def test_factor_rows_print_not_measured(bapp):
    fs = [Factor(name="rain_response", value=None, weight=0.3, contribution=0.0, explanation="no humidity sensor", basis="RISK_WEIGHTS"),
          Factor(name="exposure", value=0.27, weight=0.15, contribution=0.0405, explanation="E side", basis="RISK_WEIGHTS")]
    df = bapp["factor_rows"](fs)
    assert df.loc[0, "value"] == "not measured" and df.loc[0, "contribution"] == "0 (not measured)"
    assert df.loc[1, "value"] == "0.27" and df.loc[1, "contribution"] == "0.041"


def test_hours_text(bapp):
    ht = bapp["hours_text"]
    assert ht(None, None, None) == "does not cross"
    assert ht(None, 9.0, None).startswith("peak stays under")
    assert ht(6.7, 6.3, 7.2) == "6.7 h (band 6.3 to 7.2 h)"
    assert "low band stays under" in ht(6.7, 6.3, None)


def test_building_dirs_and_input_mtimes(bapp, tmp_path):
    demo, store = tmp_path / "demo", tmp_path / "store"
    assert bapp["building_dirs"](demo, store) == {}
    (demo).mkdir()
    (demo / "building.json").write_text("{}", encoding="utf-8")
    (store / "B-2").mkdir(parents=True)
    (store / "B-2" / "building.json").write_text("{}", encoding="utf-8")
    (store / "empty").mkdir()
    dirs = bapp["building_dirs"](demo, store)
    assert list(dirs) == [bapp["BUILDING_DEMO_LABEL"], "B-2 (runs/building)"]
    names = bapp["BUILDING_INPUTS"]
    assert "building.json" in names and "problems.json" not in names  # lifecycle writes never re-run the analysis
    m = dict(bapp["building_input_mtimes"](demo, names))
    assert m["building.json"] > 0 and m["weather.csv"] == 0.0


def test_basis_labels_resolve_registry_and_local_keys(bapp):
    labs = bapp["basis_labels"]("EPA_DRY_WINDOW_H; RESPONSE_MIN_R, EVENT_LEVEL and NOT_A_KEY")
    assert len(labs) == 3
    assert labs[0].startswith("EPA_DRY_WINDOW_H:") and "https://" in labs[0]
    assert "[team-proposed, validate]" in labs[1] and "[team-proposed, validate]" in labs[2]


def test_forecast_frame_on_synthetic_tower(bapp, small_tower):
    """SYNTHETIC small tower: the east-facade scenario room crosses EPA_RH_MAX under an east storm (a check of the
    wiring, not an accuracy claim)."""
    data = small_tower
    w = bwater.analyze_water(data, bprob.default_now(data))
    fcs = bwater.forecast(data.building, data, w.fits, rain_mm=30.0, duration_h=6.0, wind_dir_deg=90.0, wind_speed_ms=10.0)
    df = bapp["forecast_frame"](fcs, w.fits, 6.0)
    assert len(df) == len(fcs) > 0
    east = df[df["driver"] == "wdr_E"]
    assert len(east) and bool(east["crosses"].iloc[0]) and east["hours"].iloc[0] is not None
    assert east["when"].iloc[0].endswith(" h)") and "EPA_RH_MAX" in east["threshold"].iloc[0]
    assert (df["lo_rh"] <= df["peak_rh"]).all() and (df["peak_rh"] <= df["hi_rh"]).all()


def test_zone_pins_and_electrical_tree(bapp, small_tower):
    b = small_tower.building
    circuit = next(e for e in b.elements if e.kind == "circuit")
    room = next(z for z in b.zones if z.kind == "room")
    obs = [Observation(obs_id="o1", kind="thermal_reading", ts="2026-01-01T00:00:00Z", element_id=circuit.element_id, source="thermal", level="S3", synthetic=True),
           Observation(obs_id="o2", kind="ticket", ts="2026-01-01T00:00:00Z", zone_id=room.zone_id, source="ticket:T-1", level=None, synthetic=True)]
    rows = bapp["electrical_tree"](b, obs)
    by = {r["node_id"]: r for r in rows}
    assert rows[0]["kind"] == "switchboard" and rows[0]["depth"] == 0 and rows[0]["worst"] == "S3"
    panel = bgraph.electrical_parent(b)[circuit.element_id]
    assert by[panel]["worst"] == "S3" and by[panel]["own"] is None and by[circuit.element_id]["own"] == "S3"
    others = [r for r in rows if r["kind"] == "electrical_panel" and r["node_id"] != panel]
    assert others and all(r["worst"] is None for r in others)
    assert not any(r["kind"] == "load" for r in rows)
    pins = bapp["zone_pin_table"](b, obs, room.floor_id)
    top = pins.iloc[0]
    assert top["n_obs"] >= 1 and top["worst"] in ("ticket", "S3")
    assert set(pins["zone_id"]) == {z.zone_id for z in b.zones_on(room.floor_id)}
    tick = pins[pins["zone_id"] == room.zone_id].iloc[0]
    assert circuit.zone_id != room.zone_id
    assert tick["worst"] == "ticket" and tick["n_obs"] == 1  # a ticket alone is never shown as a graded level
