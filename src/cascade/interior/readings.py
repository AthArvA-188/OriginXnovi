"""Field-reading schema for interior wall testing, CSV template, and routing of each reading.

One row per reading. Rebound / UPV rows go to the strength estimator (cascade.interior.ndt);
moisture and RH rows are graded by RULES from the existing interior_water rubric and the
WOOD_MC_MAX threshold (no ML). Readings the rubric has no row for become U (never "Dry"),
with the reason stated. Meter readings are only comparable on the same scale with the same
meter, so every row carries method, instrument, scale and a dry reference reading.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = ROOT / "data" / "interior" / "field_readings_template.csv"
RUBRIC_PATH = ROOT / "src" / "cascade" / "rubrics" / "interior_water.json"

Method = Literal["pin_meter", "pinless_meter", "ir_camera", "rh_probe", "rebound", "upv", "borescope",
                 "sounding", "mold_sample", "leak_sensor"]
Scale = Literal["pct_mc_wood", "wme", "gypsum_scale", "relative_0_100", "degC", "pct_rh", "rn", "m_per_s",
                "binary", "none"]
Substrate = Literal["gypsum", "concrete", "cmu", "plaster", "wood", "brick", "other"]


class FieldReading(BaseModel):
    """One interior wall reading. Units live in `scale`; comparisons need the same meter and scale."""

    reading_id: str
    building_id: str
    floor: Optional[str] = None
    zone_id: Optional[str] = None
    wall_id: Optional[str] = None
    face: Optional[str] = None
    substrate: Substrate = "other"
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    height_m: Optional[float] = None
    captured_at: Optional[str] = None
    inspector_id: Optional[str] = None
    method: Method
    instrument_model: Optional[str] = None
    scale: Scale = "none"
    value: Optional[float] = None
    unit: Optional[str] = None
    dry_reference_value: Optional[float] = Field(None, description="same material, same meter, known-dry spot")
    ambient_t_c: Optional[float] = None
    ambient_rh_pct: Optional[float] = None
    surface_t_c: Optional[float] = None
    delta_t_c: Optional[float] = None
    photo_ids: Optional[str] = None
    rn_median: Optional[float] = None
    n_impacts: Optional[int] = None
    orientation: Optional[str] = None
    path_mm: Optional[float] = None
    transit_us: Optional[float] = None
    test_type: Optional[Literal["direct", "semi-direct", "indirect"]] = None
    sample_type: Optional[str] = None
    core_fc_mpa: Optional[float] = Field(None, description="core strength at this location (150x300 cylinder-equivalent MPa), if cored")
    design_fc_mpa: Optional[float] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.method == "rebound" and self.rn_median is None and self.value is None:
            raise ValueError("rebound reading needs rn_median (or value)")
        if self.method == "upv" and self.value is None and not (self.path_mm and self.transit_us):
            raise ValueError("UPV reading needs value in m/s, or path_mm and transit_us")
        return self

    @property
    def rn(self) -> Optional[float]:
        if self.method != "rebound":
            return None
        return self.rn_median if self.rn_median is not None else self.value

    @property
    def vp_ms(self) -> Optional[float]:
        if self.method != "upv":
            return None
        if self.value is not None:
            return self.value
        return self.path_mm / self.transit_us * 1000.0  # mm/us = km/s -> m/s


COLUMNS: List[str] = list(FieldReading.model_fields.keys())

EXAMPLE_ROWS: List[Dict[str, object]] = [
    # EXAMPLE rows so the template is self-explanatory; values are placeholders, not measurements.
    {"reading_id": "EXAMPLE-1", "building_id": "BLDG-EXAMPLE", "floor": "F03", "wall_id": "W-12", "face": "N",
     "substrate": "concrete", "method": "rebound", "instrument_model": "(hammer model)", "scale": "rn",
     "rn_median": None, "n_impacts": 12, "orientation": "horizontal", "design_fc_mpa": None,
     "notes": "EXAMPLE ROW - replace; median of >= 10 impacts per EN 12504-2 / ASTM C805 practice"},
    {"reading_id": "EXAMPLE-2", "building_id": "BLDG-EXAMPLE", "floor": "F03", "wall_id": "W-12", "face": "N",
     "substrate": "concrete", "method": "upv", "instrument_model": "(UPV model)", "scale": "m_per_s",
     "path_mm": None, "transit_us": None, "test_type": "direct", "notes": "EXAMPLE ROW - replace"},
    {"reading_id": "EXAMPLE-3", "building_id": "BLDG-EXAMPLE", "floor": "F03", "wall_id": "W-07", "face": "S",
     "substrate": "gypsum", "method": "pinless_meter", "instrument_model": "(meter model)", "scale": "relative_0_100",
     "value": None, "dry_reference_value": None, "notes": "EXAMPLE ROW - replace; take a dry reference on the same wall type"},
    {"reading_id": "EXAMPLE-4", "building_id": "BLDG-EXAMPLE", "floor": "F03", "zone_id": "F03-N", "substrate": "other",
     "method": "rh_probe", "scale": "pct_rh", "value": None, "ambient_t_c": None, "notes": "EXAMPLE ROW - replace"},
]


def write_template(path: Path = TEMPLATE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in EXAMPLE_ROWS:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})
    return path


def _none_if_blank(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def parse_csv(text: str) -> tuple[List[FieldReading], List[dict]]:
    """Parse a readings CSV. Returns (valid readings, errors[{row, reading_id, error}]).
    Rows whose reading_id starts with EXAMPLE (the template's guidance rows) are skipped."""
    ok, errs = [], []
    for i, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        if str(row.get("reading_id") or "").startswith("EXAMPLE"):
            continue  # template guidance rows carry no values
        clean = {k: _none_if_blank(v) for k, v in row.items() if k in FieldReading.model_fields}
        try:
            ok.append(FieldReading.model_validate(clean))
        except ValidationError as e:
            errs.append({"row": i, "reading_id": row.get("reading_id"), "error": "; ".join(x["msg"] for x in e.errors())})
    return ok, errs


# --- rule grading of moisture rows ------------------------------------------------------------


def _rubric() -> dict:
    return json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))


def _band(rubric: dict, family: str, value: float) -> Optional[dict]:
    for row in rubric.get("rows", []):
        if row.get("family") != family:
            continue
        lo, hi = row.get("min"), row.get("max")
        if lo is not None and value < lo:
            continue
        if hi is not None and value >= hi:
            continue
        return row
    return None


def wood_mc_threshold() -> tuple[float, str]:
    """WOOD_MC_MAX from the shared threshold table (read, not copied)."""
    from cascade.building.rules import THRESHOLDS
    th = THRESHOLDS["WOOD_MC_MAX"]
    return float(th.value), th.url or ""


def grade_moisture(r: FieldReading, rubric: Optional[dict] = None) -> dict:
    """Rule grade for one moisture/RH/leak reading. Returns {level, value_label, basis, action, reason}."""
    rubric = rubric or _rubric()
    src = "src/cascade/rubrics/interior_water.json"
    if r.value is None:
        return {"level": "U", "value_label": "U", "action": "monitor", "basis": src, "reason": "no value recorded"}
    if r.method == "leak_sensor":
        row = _band(rubric, "leak_sensor", r.value)
        if row:
            return {"level": row["unified"], "value_label": row["value"], "action": row["action"], "basis": f"{src} leak_sensor", "reason": row["criterion"]}
        return {"level": "S0", "value_label": "dry sensor", "action": "record", "basis": f"{src} leak_sensor", "reason": "leak sensor reads dry"}
    if r.method == "rh_probe" and r.scale == "pct_rh":
        row = _band(rubric, "rh_hourly", r.value)
        if row:
            return {"level": row["unified"], "value_label": row["value"], "action": row["action"], "basis": f"{src} rh_hourly", "reason": row["criterion"]}
        return {"level": "U", "value_label": "RH >= 60 (spot)", "action": "monitor", "basis": f"{src} rh_hourly / rh_sustained_h",
                "reason": "a single reading at or above 60 %RH; the rubric grades 'RH elevated' only after 72 h sustained, so log it hourly"}
    if r.method in ("pin_meter", "pinless_meter") and r.scale == "pct_mc_wood" and r.substrate == "wood":
        thr, url = wood_mc_threshold()
        if r.value > thr:
            return {"level": "S2", "value_label": f"wood MC > {thr:g} %", "action": "schedule", "basis": f"WOOD_MC_MAX {url}",
                    "reason": f"wood moisture content above {thr:g} %MC (decay risk starts); level S2 is a team mapping [team-proposed, validate]"}
        return {"level": "S0", "value_label": f"wood MC <= {thr:g} %", "action": "record", "basis": f"WOOD_MC_MAX {url}",
                "reason": f"wood moisture content at or below {thr:g} %MC"}
    if r.method in ("pin_meter", "pinless_meter"):
        ref = r.dry_reference_value
        rel = f"; reading is {r.value / ref:.2f} x the dry reference on the same meter" if ref else "; no dry reference recorded"
        return {"level": "U", "value_label": "U", "action": "monitor", "basis": src,
                "reason": f"meter scale '{r.scale}' on {r.substrate} has no rubric row (readings are meter-specific){rel}; confirm by probe or opening"}
    return {"level": "U", "value_label": "U", "action": "monitor", "basis": src, "reason": f"method {r.method} is not rule-graded here"}


def route(readings: List[FieldReading]) -> Dict[str, List[FieldReading]]:
    out: Dict[str, List[FieldReading]] = {"strength": [], "moisture": [], "record_only": []}
    for r in readings:
        if r.method in ("rebound", "upv"):
            out["strength"].append(r)
        elif r.method in ("pin_meter", "pinless_meter", "rh_probe", "leak_sensor"):
            out["moisture"].append(r)
        else:
            out["record_only"].append(r)
    return out


def pair_strength_locations(readings: List[FieldReading]) -> List[dict]:
    """Group rebound + UPV readings at the same (building, wall, face, x, y) into one location."""
    locs: Dict[tuple, dict] = {}
    for r in readings:
        key = (r.building_id, r.wall_id, r.face, r.x_m, r.y_m)
        d = locs.setdefault(key, {"building_id": r.building_id, "wall_id": r.wall_id, "face": r.face, "x_m": r.x_m,
                                  "y_m": r.y_m, "rn": None, "vp": None, "core_fc_mpa": None, "design_fc_mpa": None, "ids": []})
        d["ids"].append(r.reading_id)
        if r.rn is not None:
            d["rn"] = r.rn
        if r.vp_ms is not None:
            d["vp"] = r.vp_ms
        if r.core_fc_mpa is not None:
            d["core_fc_mpa"] = r.core_fc_mpa
        if r.design_fc_mpa is not None:
            d["design_fc_mpa"] = r.design_fc_mpa
    return list(locs.values())
