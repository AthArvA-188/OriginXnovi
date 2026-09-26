"""Facade and interior photo grading for the building layer (building_spec section 9).

Building photos are graded by the existing `grade.grade_image` against facade_ll11.json (NYC FISP), interior_water.json
(EPA moisture) or electrical_thermal.json, unchanged. This module adds what the building needs around that call:
- `building_image_records` checks that each photo names a zone or element and sets its asset class.
- `enforce_u_guards` turns grades a photo cannot support into U (U is never S0).
- `guarded_grade_fn` wraps any grade function for `pipeline.run_cascade` and applies the guards.
- `finding_to_observation` places a graded photo on its zone or element as a building Observation.
- `fisp_carryover` and `fisp_building_status` apply the FISP SWARMP carry-over rule and roll values up.

No model is called here. Grades are a pre-classification for the Qualified Exterior Wall Inspector, never a filing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..grade import grade_image, load_rubric
from ..schema import Finding, ImageRecord, unassessable_finding
from .model import Building, BuildingData, Observation
from .rules import MODEL_TAG, THRESHOLDS, find_band_row, finding_from_row, t

BUILDING_CLASSES: Tuple[str, ...] = ("facade_element", "interior_zone", "electrical_equipment")

# Asset classes a photo of each zone or element kind may carry. The first entry is the default.
# Roof zones, roof drains and risers use the EPA moisture image rows (standing water, stain, active leak): there is
# no roof rubric yet [team-proposed, validate]. Electrical rooms and equipment also accept interior_zone for a photo
# of water on or near the equipment.
ZONE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "facade_drop": ("facade_element",),
    "room": ("interior_zone",),
    "core": ("interior_zone",),
    "shaft": ("interior_zone",),
    "roof": ("interior_zone",),
    "electrical_room": ("interior_zone", "electrical_equipment"),
}
ELEMENT_CLASSES: Dict[str, Tuple[str, ...]] = {
    "facade_panel": ("facade_element",),
    "window": ("facade_element",),
    "roof_drain": ("interior_zone",),
    "riser": ("interior_zone",),
    "sensor": ("interior_zone",),
    "switchboard": ("electrical_equipment", "interior_zone"),
    "electrical_panel": ("electrical_equipment", "interior_zone"),
    "circuit": ("electrical_equipment", "interior_zone"),
    "load": ("electrical_equipment", "interior_zone"),
}

# Rubric families a photo can grade on its own (with the metadata readings for electrical). The other families
# (RH, wet duration, leak sensor, load, days since IR) need a sensor series or a schedule, so a photo that returns
# one of their values is U.
IMAGE_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "facade_element": ("image",),
    "interior_zone": ("image",),
    "electrical_equipment": ("delta_t_similar_k",),
}

# Readings copied from ImageRecord.labels into the grader metadata. Scenario and drawing labels are never copied,
# so a grader never sees ground truth.
READING_KEYS: Tuple[str, ...] = ("t_element_c", "t_reference_c", "t_ambient_c", "load_pct")

FISP_VALUES: Tuple[str, ...] = ("Safe", "SWARMP", "Unsafe")
UNKNOWN_TS = "1970-01-01T00:00:00Z"  # placeholder only: such observations carry ts_unknown=True and never drive a clock


# ----------------------------------------------------------------------------------------- records

def allowed_classes(b: Building, node_id: str) -> Tuple[str, ...]:
    """Asset classes a photo of this zone or element may carry, default first. Empty when the id is unknown."""
    z = b.zone(node_id)
    if z is not None:
        return ZONE_CLASSES.get(z.kind, ("interior_zone",))
    e = b.element(node_id)
    if e is not None:
        return ELEMENT_CLASSES.get(e.kind, ("interior_zone",))
    return ()


def building_image_records(b: Building, records: Sequence[ImageRecord]) -> List[ImageRecord]:
    """Copies of `records` with the building asset class set from the asset_id's zone or element kind.

    A record that already carries a class allowed for its node keeps it (for example electrical_equipment on a
    panel photo, or interior_zone for water on that panel). Any other class is replaced by the node's default.
    Raises ValueError when an asset_id is missing or unknown, and lists valid ids."""
    bad: List[str] = []
    out: List[ImageRecord] = []
    for rec in records:
        allowed = allowed_classes(b, rec.asset_id) if rec.asset_id else ()
        if not allowed:
            bad.append(f"{rec.image_id} (asset_id {rec.asset_id!r})")
            continue
        ac = rec.asset_class if rec.asset_class in allowed else allowed[0]
        out.append(rec.model_copy(update={"asset_class": ac}))
    if bad:
        valid = sorted(z.zone_id for z in b.zones if z.kind != "core") + sorted(e.element_id for e in b.elements if e.kind != "load")
        shown = ", ".join(valid[:40]) + (f" ... ({len(valid)} ids in total)" if len(valid) > 40 else "")
        raise ValueError(f"photos need an asset_id that is a zone or element of {b.building_id}: {'; '.join(bad)}. Valid ids: {shown}")
    return out


# ----------------------------------------------------------------------------------------- U guards

def _num(metadata: dict, key: str) -> Optional[float]:
    v = metadata.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def photo_size(metadata: dict) -> Optional[Tuple[int, int]]:
    """(width, height) from metadata "width"/"height" or the pipeline's "image_size" "WxH". None when unknown."""
    w, h = _num(metadata, "width"), _num(metadata, "height")
    if w is not None and h is not None:
        return int(w), int(h)
    s = str(metadata.get("image_size") or "")
    parts = s.lower().split("x")
    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None


def electrical_reading_missing(metadata: dict) -> Optional[str]:
    """Why an electrical photo cannot be banded (electrical_thermal.json U row), or None when the metadata holds
    t_element_c, t_reference_c and a load at scan of at least IR_MIN_LOAD_PCT."""
    missing = [k for k in ("t_element_c", "t_reference_c", "load_pct") if _num(metadata, k) is None]
    if missing:
        return f"no {', '.join(missing)} in metadata; a thermal image alone gives no delta-T over a similar component"
    load = _num(metadata, "load_pct")
    if load is not None and load < t("IR_MIN_LOAD_PCT"):
        return f"load at scan {load:.0f} % is below IR_MIN_LOAD_PCT ({THRESHOLDS['IR_MIN_LOAD_PCT'].label()})"
    return None


def _as_u(f: Finding, reason: str) -> Finding:
    """A U copy of `f` that keeps its id, evidence, modality, model and cost, and records what the grader said."""
    u = unassessable_finding(finding_id=f.finding_id, asset_class=f.asset_class, standard=f.native_scale.standard,
                             evidence=f.evidence, reason=reason, model=f.model)
    said = f"Grader output before the guard: '{f.native_scale.value}' ({f.unified.level}). {f.justification}".strip()
    return u.model_copy(update={"modality": f.modality, "usd": f.usd, "seconds": f.seconds, "justification": f"{reason}. {said}"})


def _image_values(rubric: dict, asset_class: str) -> set:
    fams = IMAGE_FAMILIES.get(asset_class, ("image",))
    return {r["value"] for r in rubric.get("rows", []) if r.get("family") in fams}


def enforce_u_guards(f: Finding, metadata: dict) -> Finding:
    """Deterministic guards on a graded building photo. Returns a new Finding and never mutates `f`.

    - electrical_equipment: no t_element_c, t_reference_c and load_pct (at least IR_MIN_LOAD_PCT) in metadata gives
      U, whatever the level (electrical_thermal.json U row and measurement_note). With readings, delta_t_k is set
      from them and the band row of that delta wins over a different band from the grader.
    - facade_element graded Safe (or S0) on a photo below FISP_MIN_PHOTO_PX, or of unknown size, gives U.
    - any photo whose value belongs only to a sensor or schedule family (RH ok, Load ok, IR overdue ...) gives U.
    U findings, other asset classes and deterministic sensor findings pass through unchanged."""
    if f.unified.level == "U" or f.asset_class not in BUILDING_CLASSES or f.modality == "sensor" or f.model == MODEL_TAG:
        return f.model_copy(deep=True)
    rubric = load_rubric(f.asset_class)
    value = f.native_scale.value
    if value not in _image_values(rubric, f.asset_class):
        return _as_u(f, f"'{value}' needs a sensor series or schedule data, which a photo cannot give")
    if f.asset_class == "electrical_equipment":
        reason = electrical_reading_missing(metadata)
        if reason is not None:
            return _as_u(f, reason)
        delta = float(_num(metadata, "t_element_c")) - float(_num(metadata, "t_reference_c"))  # type: ignore[arg-type]
        row = find_band_row(rubric, "delta_t_similar_k", delta)
        meas = f.measurements.model_copy(update={"delta_t_k": round(delta, 2)})
        if row is not None and row["value"] != value:
            return finding_from_row(row, rubric, finding_id=f.finding_id, asset_class="electrical_equipment", modality=f.modality,
                                    evidence=f.evidence, measurements=meas.model_dump(), flags=f.unified.flags,
                                    confidence=f.measurements.confidence,
                                    justification=f"Readings give delta-T {delta:.1f} K ({row['value']}); the grader said '{value}'. {f.justification}")
        return f.model_copy(update={"measurements": meas}, deep=True)
    if f.asset_class == "facade_element" and (value == "Safe" or f.unified.level == "S0"):
        size = photo_size(metadata)
        min_w, min_h = t("FISP_MIN_PHOTO_PX")
        if size is None:
            return _as_u(f, f"photo size unknown; Safe needs a photo of at least {min_w} x {min_h} px (FISP_MIN_PHOTO_PX)")
        if max(size) < max(min_w, min_h) or min(size) < min(min_w, min_h):
            return _as_u(f, f"photo {size[0]} x {size[1]} px is below FISP_MIN_PHOTO_PX {min_w} x {min_h} px, so Safe cannot be shown")
    return f.model_copy(deep=True)


def guarded_grade_fn(records: Sequence[ImageRecord], inner: Callable[..., Finding] = grade_image) -> Callable[..., Finding]:
    """A `grade_fn` for pipeline.run_cascade: adds each record's readings (READING_KEYS from labels) to the grader
    metadata, calls `inner`, then applies enforce_u_guards. Tests pass a fake `inner`, so no model is called."""
    extra: Dict[str, Dict[str, Any]] = {}
    for r in records:
        e = {k: r.labels[k] for k in READING_KEYS if k in (r.labels or {})}
        e["image_size"] = f"{r.width}x{r.height}"
        extra[r.image_id] = e

    def grade_fn(img, *, image_id: str, metadata: Optional[dict] = None, **kw: Any) -> Finding:
        meta = dict(metadata or {})
        for k, v in extra.get(image_id, {}).items():
            meta.setdefault(k, v)
        return enforce_u_guards(inner(img, image_id=image_id, metadata=meta, **kw), meta)

    return grade_fn


# ----------------------------------------------------------------------------------------- observations

def _ts(captured_on: Optional[str]) -> str:
    if not captured_on:
        return UNKNOWN_TS
    s = str(captured_on).strip()
    if len(s) == 10:
        return f"{s}T00:00:00Z"
    return s if s.endswith("Z") else s + "Z"


def finding_to_observation(f: Finding, rec: ImageRecord, b: Building) -> Observation:
    """A graded photo as an image_finding Observation on its zone, or on its element and that element's zone.

    ts comes from captured_on. When it is missing, ts is the UNKNOWN_TS placeholder with ts_unknown=True (the text
    says so): problems.build_problems places it at the analysis time and no escalation clock or due date uses it. plan_xy comes from
    rec.labels["plan_xy"] when the uploader clicked a spot; otherwise the plan pin uses the element xy or the
    zone centroid (blueprint.place). The level is the finding's level; U stays U."""
    node = rec.asset_id
    if not node or not allowed_classes(b, node):
        raise ValueError(f"image {rec.image_id}: asset_id {node!r} is not a zone or element of {b.building_id}")
    z = b.zone(node)
    zone_id = node if z is not None else b.element(node).zone_id  # type: ignore[union-attr]
    element_id = None if z is not None else node
    m = f.measurements
    value: Optional[float] = None
    unit: Optional[str] = None
    extra = ""
    if m.crack_width_mm is not None:
        value, unit = m.crack_width_mm, "mm"
        unc = f" +/- {m.crack_width_uncertainty_mm} mm" if m.crack_width_uncertainty_mm is not None else ""
        extra = f" Crack width {m.crack_width_mm} mm{unc} ({m.measurement_basis or 'basis not recorded'})."
    elif m.delta_t_k is not None:
        value, unit = m.delta_t_k, "K"
        extra = f" Delta-T {m.delta_t_k} K."
    when = "" if rec.captured_on else " Capture date unknown."
    xy = rec.labels.get("plan_xy") if rec.labels else None
    plan_xy = (int(xy[0]), int(xy[1])) if isinstance(xy, (list, tuple)) and len(xy) == 2 else None
    return Observation(
        obs_id=f"img:{f.finding_id}", kind="image_finding", ts=_ts(rec.captured_on), zone_id=zone_id, element_id=element_id,
        source="grader", level=f.unified.level, value=value, unit=unit,
        text=f"{f.native_scale.value} ({f.native_scale.standard}): {f.justification}{extra}{when}".strip(),
        finding=f, plan_xy=plan_xy, synthetic=bool((rec.labels or {}).get("synthetic", False)) or b.synthetic,
        ts_unknown=not rec.captured_on,
    )


def grade_building_photos(data: BuildingData, *, cfg=None, grade_fn: Callable[..., Finding] = grade_image,
                          gate_fn: Optional[Callable] = None, progress: Optional[Callable] = None,
                          out: Optional[Path] = None) -> Tuple[dict, List[Finding], List[Observation]]:
    """Grade the building's photos with run_cascade into <root>/grading (or `out`) and return
    (summary, findings, observations). `grade_fn` is wrapped by guarded_grade_fn."""
    from ..pipeline import load_run, run_cascade
    from .blueprint import FILES
    from .problems import image_observations

    recs = building_image_records(data.building, data.images)
    out = Path(out) if out is not None else Path(data.root) / FILES["grading"]
    kw: Dict[str, Any] = {"grade_fn": guarded_grade_fn(recs, grade_fn), "progress": progress}
    if gate_fn is not None:
        kw["gate_fn"] = gate_fn
    summary = run_cascade(recs, out, cfg, **kw)
    findings = list(load_run(out)["findings"])
    return summary, findings, image_observations(data.building, findings, recs)


# ----------------------------------------------------------------------------------------- FISP rules

def fisp_carryover(prior_value: Optional[str], current_value: str) -> str:
    """1 RCNY 103-04 (a): a condition reported SWARMP before and not corrected now must be reported Unsafe.

    The caller matches the location. Prior SWARMP with current SWARMP or Unsafe gives "Unsafe". Current Safe means
    corrected and stays Safe. Current U stays U: the photo cannot show whether the repair was made, so the rule
    cannot be applied either way, and U is never read as Safe."""
    if prior_value == "SWARMP" and current_value in ("SWARMP", "Unsafe"):
        return "Unsafe"
    return current_value


def apply_fisp_carryover(f: Finding, prior_value: Optional[str]) -> Finding:
    """`f` re-graded on the facade_ll11.json Unsafe row when fisp_carryover says so; otherwise an unchanged copy."""
    if f.asset_class != "facade_element" or fisp_carryover(prior_value, f.native_scale.value) != "Unsafe" or f.native_scale.value == "Unsafe":
        return f.model_copy(deep=True)
    rubric = load_rubric("facade_element")
    row = next(r for r in rubric["rows"] if r["value"] == "Unsafe")
    return finding_from_row(row, rubric, finding_id=f.finding_id, asset_class="facade_element", modality=f.modality,
                            evidence=f.evidence, measurements=f.measurements.model_dump(), flags=f.unified.flags,
                            confidence=f.measurements.confidence,
                            justification=f"Reported SWARMP in the previous report and not corrected now, so reported Unsafe "
                                          f"(1 RCNY 103-04 section (a)). {f.justification}")


def fisp_building_status(values: Sequence[str]) -> str:
    """Roll-up of facade values: any Unsafe gives "Unsafe"; else any SWARMP gives "SWARMP"; else any U gives
    "Not determined (U present)"; all Safe gives "Safe". No values gives "Not determined (no photos)".
    U never counts as Safe."""
    vals = list(values)
    if "Unsafe" in vals:
        return "Unsafe"
    if "SWARMP" in vals:
        return "SWARMP"
    if not vals:
        return "Not determined (no photos)"
    if all(v == "Safe" for v in vals):
        return "Safe"
    return "Not determined (U present)"

