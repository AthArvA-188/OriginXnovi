"""Fire de-escalation plan cards for a Building (ask 5, key "fire"). DECISION SUPPORT ONLY.

"De-escalation" here means the smallest response that is still safe: alert and relocate the fire floor and its
neighbours instead of evacuating the whole tower, with written triggers for widening the response. The planner is
deterministic (no ML): given an incident zone it lists floors to alert, the fire department staging floor,
relocation floors, suggested evacuation and attack stairs with the standpipe landing, elevators, the electrical path,
water risers and valves, where firefighting water will migrate (the existing water graph), hazards, occupants,
human decision points and escalation triggers. Every rule reads from fire_rules.json with a URL or the tag
[team-proposed, validate].

Nothing here actuates anything. Every action line has executes=False and names the human who approves it. The
fire alarm system, sprinklers, smoke control, elevator recall and doors run on their own listed controls.

The fire layer (stairs, elevators, fire command center, battery room, occupant design load, assistance list,
impairments) does not exist in the building model; synthetic_fire_layer() derives a SYNTHETIC one from the tower
geometry. The three drawings (flowchart, building section, swimlane) are plain SVG strings, no plotting library.
"""

from __future__ import annotations

import html
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel

from .blueprint import centroid, point_in_polygon
from .graph import electrical_parent, path_to_root, reach, subtree, water_adjacency
from .model import Building
from .rules import Threshold
from .rules import t as rules_t

RULES_PATH = Path(__file__).with_name("fire_rules.json")
LAYER_FILE = "fire_layer.json"
M_PER_FT = 0.3048  # exact definition of the international foot (unit conversion, not a threshold)
SQFT_PER_M2 = 1.0 / (M_PER_FT ** 2)
BANNER = ("DECISION SUPPORT ONLY. Cerebro never controls, silences or overrides the fire alarm system, sprinklers, fire pumps, "
          "smoke control, elevator recall or Phase II, stair locks or HVAC. The Fire Safety Director (FSD) and then the fire "
          "department incident commander (IC) decide.")
SUGGESTION = "suggestion only: the fire department designates stairs on arrival"
STAGING_MOVED_TAG = "[team-proposed, validate]"
Band = Literal["fire", "alert", "staging", "relocate", "smoke", "normal"]
BAND_ORDER: Tuple[str, ...] = ("fire", "alert", "staging", "relocate", "smoke", "normal")


# ------------------------------------------------------------------------------------------ rule table

@lru_cache(maxsize=1)
def load_rules() -> Dict[str, Any]:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def fire_thresholds() -> Dict[str, Threshold]:
    out: Dict[str, Threshold] = {}
    for r in load_rules()["thresholds"]:
        v = r["value"]
        out[r["key"]] = Threshold(key=r["key"], value=tuple(v) if isinstance(v, list) else v, unit=r["unit"], meaning=r["meaning"],
                                  tag=r["tag"], url=r.get("url"), note=r.get("note", ""))
    return out


def ft(key: str) -> Any:
    """Value of a fire-rule threshold. KeyError on an unknown key."""
    return fire_thresholds()[key].value


def basis(*keys: str) -> str:
    """Printable basis for one or more thresholds: URL for PUBLIC rows, the tag otherwise."""
    parts = []
    for k in keys:
        th = fire_thresholds()[k]
        tag = "" if th.tag == "PUBLIC" else f"[{th.tag}] "
        parts.append(f"{k}: {tag}{th.url or ''}".rstrip())
    return "; ".join(parts)


def source(n: int) -> Dict[str, Any]:
    for s in load_rules()["sources"]:
        if s["n"] == n:
            return s
    raise KeyError(n)


def src_urls(ns: Iterable[int]) -> str:
    return "; ".join(source(n)["url"] for n in ns)


# ------------------------------------------------------------------------------------------ fire layer

class Stair(BaseModel):
    stair_id: str  # "A"
    name: str
    zone_side: str  # office zone suffix the enclosure sits in on every floor ("W")
    rect_m: Tuple[float, float, float, float]  # enclosure x0, y0, x1, y1 on the plate, metres
    serves: List[str]
    standpipe: bool = True
    pressurized: bool = True
    roof_access: bool = True
    camera_every: Optional[int] = None


class Elevator(BaseModel):
    elevator_id: str
    kind: Literal["passenger", "fire_service_access"]
    name: str
    serves: List[str]


class Impairment(BaseModel):
    system: str
    floor_ids: List[str]
    since: str
    note: str
    synthetic: bool = True


class Assistance(BaseModel):
    floor_id: str
    count: int
    note: str
    synthetic: bool = True


class EnergyStorage(BaseModel):
    ess_id: str
    zone_id: str
    chemistry: str
    synthetic: bool = True


class FireLayer(BaseModel):
    building_id: str
    synthetic: bool
    stairs: List[Stair]
    elevators: List[Elevator]
    fcc_zone: str
    floor_control_valves: Dict[str, str]  # floor_id -> stair_id holding that floor's sprinkler control valve
    ess: List[EnergyStorage]
    occupants: Dict[str, Dict[str, Optional[int]]]  # floor_id -> {"design_load": int, "surveyed": None}
    assistance: List[Assistance]
    impairments: List[Impairment]
    roof_landing: str
    height_checks: Dict[str, Any]
    notes: List[str]

    def stair(self, stair_id: str) -> Stair:
        return next(s for s in self.stairs if s.stair_id == stair_id)


def occupied_floors(b: Building) -> List[Any]:
    """Floors 1..b.floors in level order (the roof is not an occupied floor)."""
    return sorted([f for f in b.floor_list if 1 <= f.level <= b.floors], key=lambda f: f.level)


def fid_of(b: Building, level: int) -> str:
    for f in b.floor_list:
        if f.level == level:
            return f.floor_id
    raise KeyError(level)


def _core_bbox_m(b: Building, floor_id: str, ppm: float) -> Tuple[float, float, float, float]:
    pts = [p for z in b.zones_on(floor_id) if z.kind in ("shaft", "core", "electrical_room") for p in z.polygon]
    xs = [p[0] / ppm for p in pts]
    ys = [p[1] / ppm for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _zone_side_at(b: Building, floor_id: str, xy_px: Tuple[float, float]) -> str:
    for z in b.zones_on(floor_id):
        if z.kind == "room" and point_in_polygon(xy_px, z.polygon):
            return z.zone_id.split("-")[-1]
    raise ValueError(f"no room zone at {xy_px} on {floor_id}")


def stair_separation_m(a: Stair, b_: Stair) -> float:
    """Straight-line distance between the nearest points of two rectangular enclosures (IBC 403.5.1 wording)."""
    ax0, ay0, ax1, ay1 = a.rect_m
    bx0, by0, bx1, by1 = b_.rect_m
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    return math.hypot(dx, dy)


def synthetic_fire_layer(b: Building) -> FireLayer:
    """A SYNTHETIC fire layer derived from the tower geometry. Choices (all synthetic, none claim code compliance):
    Stair A and Stair B are 3 x 6 m enclosures 1 m outside the west and east edges of the core, in the W and E offices;
    two fire service access elevators serve every floor; a low passenger bank serves the lower half and a high bank
    F01 plus the upper half; the fire command center is F01-CORE; a lithium-ion battery room sits in F01-ELEC; the
    design occupant load is gross plate area / 150 sq ft; three assistance entries and one sprinkler impairment."""
    floors = occupied_floors(b)
    if not floors:
        raise ValueError("building has no occupied floors")
    f1 = floors[0]
    top = floors[-1].level
    ppm = float(f1.px_per_m or 1.0)
    cx0, cy0, cx1, cy1 = _core_bbox_m(b, f1.floor_id, ppm)
    cy = (cy0 + cy1) / 2
    rect_a = (cx0 - 4.0, cy - 3.0, cx0 - 1.0, cy + 3.0)
    rect_b = (cx1 + 1.0, cy - 3.0, cx1 + 4.0, cy + 3.0)
    side_a = _zone_side_at(b, f1.floor_id, ((rect_a[0] + rect_a[2]) / 2 * ppm, cy * ppm))
    side_b = _zone_side_at(b, f1.floor_id, ((rect_b[0] + rect_b[2]) / 2 * ppm, cy * ppm))
    ids = [f.floor_id for f in floors]
    cam = int(ft("STAIR_CAMERA_EVERY"))
    stairs = [Stair(stair_id="A", name="Stair A (west)", zone_side=side_a, rect_m=rect_a, serves=ids, camera_every=cam),
              Stair(stair_id="B", name="Stair B (east)", zone_side=side_b, rect_m=rect_b, serves=ids, camera_every=cam)]
    half = max(1, top // 2)
    elevators = [
        Elevator(elevator_id="FSAE-1", kind="fire_service_access", name="fire service access elevator 1", serves=ids),
        Elevator(elevator_id="FSAE-2", kind="fire_service_access", name="fire service access elevator 2", serves=ids),
        Elevator(elevator_id="LOW", kind="passenger", name="low-rise passenger bank", serves=ids[:half]),
        Elevator(elevator_id="HIGH", kind="passenger", name="high-rise passenger bank", serves=[ids[0]] + ids[half:]),
    ]
    occupants: Dict[str, Dict[str, Optional[int]]] = {}
    for f in floors:
        area = sum(z.area_m2 or 0.0 for z in b.zones_on(f.floor_id))
        occupants[f.floor_id] = {"design_load": int(area * SQFT_PER_M2 // ft("OCC_LOAD_SQFT_PER_PERSON")), "surveyed": None}

    def lvl(frac: float) -> int:
        return int(min(max(round(frac * top), 2), top - 1)) if top >= 3 else 1

    c = lvl(0.6)
    assistance = [Assistance(floor_id=fid_of(b, lv), count=1, note=f"SYNTHETIC entry {i + 1}: self-identified as needing help")
                  for i, lv in enumerate(sorted({max(1, c - 1), min(top, c + 1), lvl(0.85)}))]
    imp_floor = fid_of(b, lvl(0.72))
    impairments = [Impairment(system="sprinkler", floor_ids=[imp_floor], since="2026-09-20",
                              note=f"SYNTHETIC: floor control valve at {imp_floor} closed for tenant fit-out; fire watch assigned")]
    top_elev_m = floors[-1].elevation_m
    trig = ft("IBC_HEIGHT_TRIGGERS_FT")
    top_ft = top_elev_m / M_PER_FT
    height_checks = {"highest_occupied_floor": floors[-1].floor_id, "elevation_m": top_elev_m, "elevation_ft": round(top_ft, 1),
                     "triggered": {k: bool(top_ft > v) for k, v in trig.items()},
                     "basis": basis("IBC_HEIGHT_TRIGGERS_FT")}
    sep = stair_separation_m(stairs[0], stairs[1])
    notes = ["SYNTHETIC fire layer derived from the synthetic tower geometry; not a real building and no code compliance is claimed.",
             f"Stair enclosures are {sep:.1f} m apart at their nearest points (check value {ft('STAIR_MIN_SEP_FT')} ft = "
             f"{ft('STAIR_MIN_SEP_FT') * M_PER_FT:.2f} m; {basis('STAIR_MIN_SEP_FT')}).",
             "No occupant evacuation elevators in this tower; where they exist (IBC 3008) they are usable only until Phase I recall "
             f"({source(13)['url']})."]
    return FireLayer(building_id=b.building_id, synthetic=True, stairs=stairs, elevators=elevators, fcc_zone=f"{f1.floor_id}-CORE",
                     floor_control_valves={f.floor_id: "A" for f in floors},
                     ess=[EnergyStorage(ess_id="ESS-1", zone_id=f"{f1.floor_id}-ELEC", chemistry="lithium-ion (SYNTHETIC)")],
                     occupants=occupants, assistance=assistance, impairments=impairments,
                     roof_landing="roof: emergency helicopter landing facility or approved equivalency (SYNTHETIC; LAFD Requirement No. 10)",
                     height_checks=height_checks, notes=notes)


def save_fire_layer(layer: FireLayer, path: Path) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / LAYER_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(layer.model_dump_json(indent=1), encoding="utf-8")
    return p


def load_fire_layer(path: Path) -> FireLayer:
    p = Path(path)
    if p.is_dir():
        p = p / LAYER_FILE
    return FireLayer.model_validate_json(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ plan card

class PlanLine(BaseModel):
    section: str
    item: str
    detail: str
    basis: str
    executes: bool = False  # always False: the card never actuates anything
    requires_approval_by: Optional[str] = None  # "FSD" or "IC" for proposed actions; None for information lines


class WaterReach(BaseModel):
    node: str
    floor_id: str
    level: int
    kind: str
    score: float
    hops: int
    path: List[str]


class Hazard(BaseModel):
    kind: str
    text: str
    floor_ids: List[str]
    near: bool = False
    basis: str


class Decision(BaseModel):
    id: str
    owner: str
    phase: int
    question: str
    ai_role: str = "evidence only"
    basis: str


class Trigger(BaseModel):
    id: str
    text: str
    status: Literal["fired", "watch", "armed"]
    detail: str = ""
    basis: str


class PlanCard(BaseModel):
    building_id: str
    synthetic: bool
    banner: str
    zone_id: str
    floor_id: str
    level: int
    top: int
    detections: List[str]
    alert_floors: List[int]
    staging_floor: Optional[int]
    relocation_floors: List[int]
    relocation_note: str
    smoke_watch_floors: List[int]
    evac_stair: str
    attack_stair: str
    standpipe_landing: int
    fd_lift_exit_floor: Optional[int]
    passenger_banks: List[str]
    panel: Optional[str]
    circuits: List[str]
    zone_circuits: List[str]
    feeder_path: List[str]
    risers: List[str]
    floor_control_valve: str
    water_watch: List[WaterReach]
    elec_reached: List[str]
    smoke_sensors: Dict[str, List[str]]
    hazards: List[Hazard]
    occupants: Dict[str, Any]
    decisions: List[Decision]
    triggers: List[Trigger]
    automatic_sequence: List[PlanLine]
    lines: List[PlanLine]
    generated_by: str = "cascade.building.fire.plan_for_incident (deterministic rules, no ML)"


def _level_of(b: Building, node: str) -> int:
    f = b.floor(node) or b.floor_of(node)
    if f is None:
        raise KeyError(f"unknown floor, zone or element {node}")
    return f.level


def _fl(b: Building, levels: Sequence[int]) -> str:
    ids = [fid_of(b, x) for x in sorted(levels)]
    if not ids:
        return "none"
    if len(ids) > 2 and sorted(levels) == list(range(min(levels), max(levels) + 1)):
        return f"{ids[0]}-{ids[-1]}"
    return ", ".join(ids)


def _zone_xy_m(b: Building, zone_id: str) -> Tuple[float, float]:
    z = b.zone(zone_id)
    f = b.floor(z.floor_id)
    ppm = float(f.px_per_m or 1.0)
    cx, cy = centroid(z.polygon)
    return cx / ppm, cy / ppm


def choose_stairs(b: Building, layer: FireLayer, zone_id: str) -> Tuple[str, str]:
    """(evacuation stair, attack stair): evacuation = the stair farther from the incident zone centroid, attack = the
    other one (nearest the fire, with the standpipe the fire department uses). Ties go to the lower stair id for
    evacuation. A suggestion only: the fire department designates stairs on arrival."""
    x, y = _zone_xy_m(b, zone_id)

    def dist(s: Stair) -> float:
        x0, y0, x1, y1 = s.rect_m
        return math.hypot((x0 + x1) / 2 - x, (y0 + y1) / 2 - y)

    ranked = sorted(layer.stairs, key=lambda s: (-round(dist(s), 6), s.stair_id))
    return ranked[0].stair_id, ranked[-1].stair_id


def water_watch(b: Building, level: int) -> List[WaterReach]:
    """Where firefighting water applied on the fire floor can migrate: max-product reach on the existing water graph
    from every room and core zone of the fire floor, keeping nodes on lower floors with score >= REACH_MIN, ranked by
    score (then floor, high first)."""
    fid = fid_of(b, level)
    sources = {z.zone_id: 1.0 for z in b.zones_on(fid) if z.kind in ("room", "core")}
    got = reach(water_adjacency(b), sources, max_hops=int(ft("WATER_WATCH_MAX_HOPS")), min_score=float(rules_t("REACH_MIN")))
    out: List[WaterReach] = []
    for node, r in got.items():
        f = b.floor_of(node)
        if f is None or f.level >= level or f.level < 1:
            continue
        z = b.zone(node)
        kind = z.kind if z is not None else (b.element(node).kind if b.element(node) else "node")
        out.append(WaterReach(node=node, floor_id=f.floor_id, level=f.level, kind=kind, score=round(r.score, 4), hops=r.hops, path=list(r.path)))
    out.sort(key=lambda w: (-w.score, -w.level, w.node))
    return out


def plan_for_incident(b: Building, layer: FireLayer, zone_id: str, detections: Sequence[str] = ()) -> PlanCard:
    """The plan card for an incident in `zone_id`. `detections` are further floors or zones with an alarm (escalation
    trigger E1 widens the alert set). Pure and deterministic."""
    z = b.zone(zone_id)
    if z is None:
        raise KeyError(f"unknown zone {zone_id}")
    if z.kind == "roof":
        raise ValueError("roof zones are not occupied floors; pick a zone on floors 1..top")
    floors = occupied_floors(b)
    top = floors[-1].level
    n = _level_of(b, zone_id)
    fire_fid = fid_of(b, n)
    rules = load_rules()
    impaired = {f: imp for imp in layer.impairments for f in imp.floor_ids}

    def clamp(levels: Iterable[int]) -> List[int]:
        return sorted({x for x in levels if 1 <= x <= top})

    # (b) alert floors, E1 and E2
    above, below = int(ft("FIRE_ALERT_ABOVE")), int(ft("FIRE_ALERT_BELOW"))
    base_alert = clamp(range(n - below, n + above + 1))
    alert = set(base_alert)
    det_levels = []
    for d in detections:
        dl = _level_of(b, d)
        det_levels.append(dl)
        alert |= set(clamp(range(dl - below, dl + above + 1)))
    if det_levels:  # floors between two alarming floors are alerted too [team-proposed, validate]
        alert = set(range(min(alert), max(alert) + 1))
    alert_l = sorted(alert)
    e1_new = sorted({d for d in det_levels if d not in base_alert})
    e2 = sorted({d for d in det_levels if d >= n + int(ft("FIRE_SPREAD_ABOVE"))})

    # (c) staging: FIRE_STAGING_BELOW floors below the fire (sourced). When trigger E1 has widened the alert set over
    # that floor, staging moves down to the first floor below it that is not alerted: a team rule, tagged on the card.
    staging_rule = n - int(ft("FIRE_STAGING_BELOW"))
    staging: Optional[int] = None
    for s in range(staging_rule, 0, -1):
        if s not in alert:
            staging = s
            break
    staging_moved = staging != (staging_rule if staging_rule >= 1 else None)

    # (d) relocation
    reloc: List[int] = []
    start = min(n - int(ft("FIRE_RELOCATE_MIN_BELOW")), min(alert_l) - 1)
    skipped_impaired: List[str] = []
    for x in range(start, 1, -1):  # floor 1 is the lobby, fire command center and exit discharge, never a relocation floor
        if len(reloc) >= int(ft("FIRE_RELOCATE_FLOORS")):
            break
        if x in alert or x == staging:
            continue
        if fid_of(b, x) in impaired:
            skipped_impaired.append(fid_of(b, x))
            continue
        reloc.append(x)
    reloc.sort(reverse=True)
    if reloc:
        reloc_note = (f"relocate occupants of {_fl(b, alert_l)} down the evacuation stair to {_fl(b, reloc)}; the approved Emergency "
                      "Plan's predetermined relocation floors override this")
    else:
        reloc_note = "no relocation floor below the fire: evacuate the alert floors to the outside assembly area"
    if skipped_impaired:
        reloc_note += f" (skipped {', '.join(skipped_impaired)}: open system impairment)"

    # (i) smoke watch
    lo, hi = ft("FIRE_SMOKE_WATCH_ABOVE")
    smoke = [x for x in clamp(range(n + int(lo), n + int(hi) + 1)) if x not in alert]
    smoke_sensors: Dict[str, List[str]] = {}
    for x in smoke:
        f = fid_of(b, x)
        smoke_sensors[f] = [s.sensor_id for zz in b.zones_on(f) if zz.kind in ("shaft", "core") for s in b.sensors_in(zz.zone_id)]

    # (e) stairs
    evac, attack = choose_stairs(b, layer, zone_id)
    landing = max(1, n - int(ft("FIRE_EQUIP_BELOW")))

    # (f) elevators
    banks = [e.elevator_id for e in layer.elevators if e.kind == "passenger" and fire_fid in e.serves]
    fsae = [e.elevator_id for e in layer.elevators if e.kind == "fire_service_access"]
    lift_exit: Optional[int] = None
    if n >= int(ft("FIRE_LIFT_MIN_FIRE_FLOOR")) and fsae:
        lift_exit = max(1, n - int(ft("FIRE_LIFT_EXIT_BELOW")))

    # (g) electrical
    panels = [e for zz in b.zones_on(fire_fid) for e in b.elements_in(zz.zone_id) if e.kind == "electrical_panel"]
    panel = panels[0].element_id if panels else None
    circuits = [x for x in subtree(b, panel) if (b.element(x) and b.element(x).kind == "circuit")] if panel else []
    parent = electrical_parent(b)
    zone_loads = [e.element_id for e in b.elements_in(zone_id) if e.kind == "load"]
    zone_circuits = sorted({parent[x] for x in zone_loads if x in parent})
    feeder = path_to_root(b, panel) if panel else []

    # (h) water
    risers = [e.element_id for zz in b.zones_on(fire_fid) for e in b.elements_in(zz.zone_id) if e.kind == "riser"]
    fcv_stair = layer.floor_control_valves.get(fire_fid, "?")
    ww = water_watch(b, n)
    elec_reached = [w.node for w in ww if w.kind == "electrical_room"]

    # (j) hazards: impairments on alert or relocation floors first
    hz: List[Hazard] = []
    relevant = {fid_of(b, x) for x in alert_l + reloc}
    imp_first = [imp for imp in layer.impairments if set(imp.floor_ids) & relevant]
    imp_other = [imp for imp in layer.impairments if not (set(imp.floor_ids) & relevant)]
    for imp in imp_first:
        hz.append(Hazard(kind="impairment", text=f"OPEN IMPAIRMENT ON AN ALERT OR RELOCATION FLOOR: {imp.system} at {', '.join(imp.floor_ids)} "
                         f"since {imp.since}. {imp.note}", floor_ids=imp.floor_ids, near=True, basis=src_urls([32, 33])))
    for e in layer.ess:
        el = _level_of(b, e.zone_id)
        near = abs(el - n) <= int(ft("ESS_NEAR_FLOORS"))
        hz.append(Hazard(kind="ess", text=f"Battery room {e.ess_id} ({e.chemistry}) in {e.zone_id}"
                         + (" is NEAR the fire" if near else "") + ": nobody opens the door; remote gas readings only (trigger E5).",
                         floor_ids=[fid_of(b, el)], near=near, basis=basis("ESS_NEAR_FLOORS")))
    for node in elec_reached:
        hz.append(Hazard(kind="water_near_electrical", text=f"Firefighting water may reach electrical room {node} below the fire; "
                         "isolation or salvage only on IC order by a qualified person.", floor_ids=[b.floor_of(node).floor_id],
                         near=True, basis=basis("WATER_WATCH_MAX_HOPS") + "; REACH_MIN [team-proposed, validate]"))
    for imp in imp_other:
        hz.append(Hazard(kind="impairment", text=f"Open impairment elsewhere: {imp.system} at {', '.join(imp.floor_ids)} since {imp.since}. {imp.note}",
                         floor_ids=imp.floor_ids, near=False, basis=src_urls([32, 33])))
    hz.append(Hazard(kind="roof", text=layer.roof_landing + " Helicopter use is a fire department decision.", floor_ids=[],
                     near=False, basis=source(2)["url"]))

    # (k) occupants
    per_floor = {fid_of(b, x): int(layer.occupants.get(fid_of(b, x), {}).get("design_load") or 0) for x in alert_l}
    total = sum(per_floor.values())
    assist = [a.model_dump() for a in layer.assistance if a.floor_id in per_floor]
    occupants = {"per_alert_floor": per_floor, "total_to_relocate": total, "per_evac_stair": {evac: total},
                 "assistance_on_alert_floors": assist, "assistance_count": sum(a["count"] for a in assist),
                 "basis": basis("OCC_LOAD_SQFT_PER_PERSON") + " (design figure, not a headcount; use the warden count)"}

    # (l) decisions, (m) triggers, (n) automatic sequence
    decisions = [Decision(id=d["id"], owner=d["owner"], phase=d["phase"], question=d["question"], basis=src_urls(d["src"]))
                 for d in rules["decisions"]]
    trig: List[Trigger] = []
    for tr in rules["triggers"]:
        status, detail = "armed", ""
        if tr["id"] == "E1" and e1_new:
            status, detail = "fired", (f"alarm on {_fl(b, e1_new)}: alert set widened to {_fl(b, alert_l)} "
                                       "(floors in between included [team-proposed, validate])")
        elif tr["id"] == "E2" and e2:
            status, detail = "fired", f"alarm on {_fl(b, e2)}, at least {ft('FIRE_SPREAD_ABOVE')} floors above the fire: possible upward spread"
        elif tr["id"] == "E4" and imp_first:
            status, detail = "fired", "impairment on " + ", ".join(f for imp in imp_first for f in imp.floor_ids)
        elif tr["id"] == "E5":
            ess_zones = {e.zone_id for e in layer.ess}
            if zone_id in ess_zones or any(d in ess_zones for d in detections):
                status, detail = "fired", "alarm in the battery room"
        elif tr["id"] == "E6" and assist:
            status, detail = "watch", f"{sum(a['count'] for a in assist)} person(s) needing help on the alert floors"
        trig.append(Trigger(id=tr["id"], text=tr["text"], status=status, detail=detail, basis=src_urls(tr["src"])))
    auto: List[PlanLine] = []
    for a in rules["automatic_sequence"]:
        item = a["item"]
        if a["id"] == "A1":
            item = f"Voice alarm on {_fl(b, base_alert)} (alarm floor, floor above, floor below)"
        auto.append(PlanLine(section="Verify on the fire alarm panel", item=f"[ ] {a['id']} {item}", detail=a["condition"],
                             basis=src_urls(a["src"]), requires_approval_by=None))

    # printable lines
    L: List[PlanLine] = []

    def add(section: str, item: str, detail: str, bas: str, who: Optional[str] = None) -> None:
        L.append(PlanLine(section=section, item=item, detail=detail, basis=bas, requires_approval_by=who))

    add("Incident", f"{zone_id} on {fire_fid}", f"{z.name or zone_id}; floor {n} of {top}", "SYNTHETIC building")
    add("Protect", f"Alert floors: {_fl(b, alert_l)}", "code minimum is the fire floor, the floor above and the floor below"
        + (f"; widened by E1 ({_fl(b, e1_new)})" if e1_new else ""), basis("FIRE_ALERT_ABOVE"), "FSD")
    add("Protect", f"Relocation floors: {_fl(b, reloc) if reloc else 'outside assembly area'}", reloc_note,
        basis("FIRE_RELOCATE_MIN_BELOW", "FIRE_RELOCATE_FLOORS"), "FSD")
    add("Protect", f"Evacuation stair: Stair {evac}", f"the stair farther from the fire zone; {SUGGESTION}", "[team-proposed, validate]", "IC")
    add("Protect", f"Smoke watch: {_fl(b, smoke) if smoke else 'none (top of building)'}",
        "read shaft and core sensors above the alert zone (building sensors, not listed smoke detectors)", basis("FIRE_SMOKE_WATCH_ABOVE"))
    if staging_moved:
        add("Fire department", f"Staging floor: {fid_of(b, staging) if staging else 'exterior / ground lobby'}",
            f"staging at least two floors below the fire; {STAGING_MOVED_TAG}: moved below the widened alert set "
            f"{_fl(b, alert_l)} (never stage on an alerted floor)",
            basis("FIRE_STAGING_BELOW") + f"; {STAGING_MOVED_TAG}: staging below the widened alert set", "IC")
    else:
        add("Fire department", f"Staging floor: {fid_of(b, staging) if staging else 'exterior / ground lobby'}",
            "staging at least two floors below the fire", basis("FIRE_STAGING_BELOW"), "IC")
    add("Fire department", f"Attack stair: Stair {attack}, standpipe hose connection at the {fid_of(b, landing)} landing",
        f"equipment one floor below the fire; {SUGGESTION}", basis("FIRE_EQUIP_BELOW") + "; " + source(11)["url"], "IC")
    if lift_exit is not None:
        add("Fire department", f"Fire service elevators {', '.join(fsae)}: Phase II, exit at {fid_of(b, lift_exit)}",
            "fire department only; crews leave the car at least two floors below the fire", basis("FIRE_LIFT_EXIT_BELOW"), "IC")
    else:
        add("Fire department", "Stairs expected for the fire department",
            f"fire on floor {n}: elevator use is authorized only for fires above the sixth floor (regional guideline)",
            basis("FIRE_LIFT_MIN_FIRE_FLOOR"))
    add("Elevators", f"Passenger banks serving {fire_fid}: {', '.join(banks) or 'none'}",
        "Phase I recall expected ONLY IF an elevator lobby or hoistway detector activates (LAFD matrix, footnote g); verify on the FACP. "
        "Never send staff up in an elevator to check an alarm.", source(1)["url"] + "; " + source(32)["url"])
    add("Electrical", f"Panel {panel or 'none'}; circuits {', '.join(circuits) or 'none'}",
        f"circuits feeding {zone_id}: {', '.join(zone_circuits) or 'none'}; de-energize only by a qualified person on IC order",
        "building model feeds edges", "IC")
    add("Electrical", "Feeder path: " + " -> ".join(feeder) if feeder else "Feeder path: none", "panel to main switchboard", "building model feeds edges")
    add("Water", f"Risers on {fire_fid}: {', '.join(risers) or 'none'}", "domestic riser and storm leader (not fire protection)", "building model")
    add("Water", f"Sprinkler floor control valve for {fire_fid}: in Stair {fcv_stair}",
        "supervised valve per floor; operate only on IC order", source(12)["url"], "IC")
    add("Water", f"Standpipes: Stair {', Stair '.join(s.stair_id for s in layer.stairs if s.standpipe)}", "hose connection at every floor landing",
        source(11)["url"])
    if ww:
        top_ww = ", ".join(f"{w.node} ({w.score:.2f})" for w in ww[:5])
        add("Water", f"Firefighting water may migrate to {len(ww)} node(s) below", f"highest path scores: {top_ww}; plan salvage covers",
            basis("WATER_WATCH_MAX_HOPS") + "; REACH_MIN [team-proposed, validate]")
    for h in hz:
        add("Hazards", h.kind.replace("_", " "), h.text, h.basis)
    add("Occupants", f"Design load on alert floors: {total}", ", ".join(f"{k} {v}" for k, v in per_floor.items()), occupants["basis"])
    add("Occupants", f"People needing help on alert floors: {occupants['assistance_count']}",
        "wardens confirm each one (trigger E6)", source(3)["url"], "FSD")
    for d in decisions:
        add("Human decisions", f"{d.id} ({d.owner})", d.question + " AI role: evidence only.", d.basis, d.owner)

    return PlanCard(building_id=b.building_id, synthetic=bool(b.synthetic or layer.synthetic), banner=BANNER, zone_id=zone_id,
                    floor_id=fire_fid, level=n, top=top, detections=list(detections), alert_floors=alert_l, staging_floor=staging,
                    relocation_floors=reloc, relocation_note=reloc_note, smoke_watch_floors=smoke, evac_stair=evac, attack_stair=attack,
                    standpipe_landing=landing, fd_lift_exit_floor=lift_exit, passenger_banks=banks, panel=panel, circuits=circuits,
                    zone_circuits=zone_circuits, feeder_path=feeder, risers=risers, floor_control_valve=f"Stair {fcv_stair} at {fire_fid}",
                    water_watch=ww, elec_reached=elec_reached, smoke_sensors=smoke_sensors, hazards=hz, occupants=occupants,
                    decisions=decisions, triggers=trig, automatic_sequence=auto, lines=L)


def band_of(card: PlanCard, level: int) -> str:
    if level == card.level:
        return "fire"
    if level in card.alert_floors:
        return "alert"
    if card.staging_floor == level:
        return "staging"
    if level in card.relocation_floors:
        return "relocate"
    if level in card.smoke_watch_floors:
        return "smoke"
    return "normal"


def section_rows(card: PlanCard) -> List[Dict[str, Any]]:
    """One row per floor (top first): floor, level, band, note. section_frame wraps it in a DataFrame."""
    notes = {"fire": "fire floor: voice alarm, relocate", "alert": "alert: voice alarm, relocate",
             "staging": "fire department staging", "relocate": "relocation floor [team-proposed]",
             "smoke": "smoke watch via shafts [team-proposed]", "normal": ""}
    out = []
    for lv in range(card.top, 0, -1):
        band = band_of(card, lv)
        f = f"F{lv:02d}"
        note = notes[band]
        if lv == card.standpipe_landing and lv != card.level:
            note = (note + "; " if note else "") + f"hose connection, Stair {card.attack_stair} landing"
        if card.fd_lift_exit_floor == lv:
            note = (note + "; " if note else "") + "FD leaves the elevator here"
        out.append({"floor": f, "level": lv, "band": band, "note": note})
    return out


def section_frame(card: PlanCard):
    import pandas as pd

    return pd.DataFrame(section_rows(card))


def plan_card_markdown(card: PlanCard) -> str:
    """Printable plan card (Markdown). Labels SYNTHETIC and DECISION SUPPORT ONLY at the top."""
    out = [f"# Fire plan card: {card.zone_id} ({card.floor_id} of {card.top})", "",
           f"**{card.banner}**", "", "**SYNTHETIC building** (generated tower and fire layer)." if card.synthetic else "", ""]
    sec = None
    for ln in card.lines:
        if ln.section != sec:
            sec = ln.section
            out += ["", f"## {sec}", "", "| Item | Detail | Approved by | Basis |", "|---|---|---|---|"]
        out.append(f"| {ln.item} | {ln.detail} | {ln.requires_approval_by or '-'} | {ln.basis} |")
    out += ["", "## Escalation triggers", "", "| Id | Trigger | Status | Detail |", "|---|---|---|---|"]
    for tr in card.triggers:
        out.append(f"| {tr.id} | {tr.text} | {tr.status} | {tr.detail} |")
    out += ["", "## Verify on the fire alarm panel (automatic, listed controls; the AI does not run these)", ""]
    for a in card.automatic_sequence:
        out.append(f"- {a.item}: {a.detail} ({a.basis})")
    out += ["", f"Generated by {card.generated_by}. Every action line has executes=False."]
    return "\n".join(x for x in out if x is not None)


# ------------------------------------------------------------------------------------------ rule-conformance checks

def check_card(b: Building, layer: FireLayer, card: PlanCard) -> List[str]:
    """Rule-conformance violations for a card, with or without extra detections (trigger E1). Empty list = all rules
    hold. With no detections staging must sit exactly FIRE_STAGING_BELOW floors below the fire; with detections it
    must be the first non-alerted floor at or below that, and any move must be tagged [team-proposed, validate]."""
    bad: List[str] = []
    n, top = card.level, card.top
    alert = set(card.alert_floors)
    code_min = {x for x in (n - 1, n, n + 1) if 1 <= x <= top}
    if not code_min <= alert:
        bad.append("alert set misses the code minimum")
    for d in card.detections:
        dl = _level_of(b, d)
        if not {x for x in (dl - 1, dl, dl + 1) if 1 <= x <= top} <= alert:
            bad.append(f"alert set misses the code minimum around the second alarm on {d}")
    rule_stage = n - int(ft("FIRE_STAGING_BELOW"))
    exp_rule = rule_stage if rule_stage >= 1 else None
    exp_stage = next((s for s in range(rule_stage, 0, -1) if s not in alert), None) if card.detections else exp_rule
    if card.staging_floor != exp_stage:
        bad.append(f"staging {card.staging_floor} != {exp_stage}")
    if card.staging_floor is not None and card.staging_floor in alert:
        bad.append("staging on an alerted floor")
    st_lines = [ln for ln in card.lines if ln.item.startswith("Staging floor")]
    if len(st_lines) != 1:
        bad.append("staging line missing")
    elif (card.staging_floor != exp_rule) != (STAGING_MOVED_TAG in st_lines[0].basis and STAGING_MOVED_TAG in st_lines[0].detail):
        bad.append("staging moved below the sourced rule without the [team-proposed, validate] tag (or tagged without a move)")
    for r in card.relocation_floors:
        if r > n - int(ft("FIRE_RELOCATE_MIN_BELOW")):
            bad.append(f"relocation floor {r} too close to the fire")
        if r in card.alert_floors or r == card.staging_floor or r == 1:
            bad.append(f"relocation floor {r} overlaps alert, staging or the lobby")
    if card.evac_stair == card.attack_stair:
        bad.append("evacuation stair equals attack stair")
    if stair_separation_m(layer.stair("A"), layer.stair("B")) < ft("STAIR_MIN_SEP_FT") * M_PER_FT:
        bad.append("stair enclosures closer than the separation check")
    if any(ln.executes for ln in card.lines + card.automatic_sequence):
        bad.append("an action line executes")
    for ln in card.lines:
        if ln.section == "Human decisions" and ln.requires_approval_by not in ("FSD", "IC"):
            bad.append(f"decision line without a human approver: {ln.item}")
        if not ln.basis:
            bad.append(f"line without basis: {ln.item}")
    if card.banner != BANNER or "DECISION SUPPORT ONLY" not in card.banner:
        bad.append("banner missing")
    if card.synthetic != bool(b.synthetic or layer.synthetic):
        bad.append("synthetic flag lost")
    if any(w.level >= n for w in card.water_watch):
        bad.append("water watch lists a floor at or above the fire")
    scores = [w.score for w in card.water_watch]
    if scores != sorted(scores, reverse=True):
        bad.append("water watch not sorted by score")
    ess = [h for h in card.hazards if h.kind == "ess"]
    if len(ess) != len(layer.ess):
        bad.append("battery room hazard missing")
    for h, e in zip(ess, layer.ess):
        if h.near != (abs(_level_of(b, e.zone_id) - n) <= int(ft("ESS_NEAR_FLOORS"))):
            bad.append("battery room 'near' flag wrong")
    if n < int(ft("FIRE_LIFT_MIN_FIRE_FLOOR")) and card.fd_lift_exit_floor is not None:
        bad.append("FD elevator suggested for a low fire")
    if n >= int(ft("FIRE_LIFT_MIN_FIRE_FLOOR")) and card.fd_lift_exit_floor != n - int(ft("FIRE_LIFT_EXIT_BELOW")):
        bad.append("FD elevator exit floor wrong")
    elev = [ln for ln in card.lines if ln.section == "Elevators"]
    if not elev or "lobby or hoistway detector" not in elev[0].detail:
        bad.append("elevator recall not stated as conditional")
    relevant = {f"F{x:02d}" for x in card.alert_floors + card.relocation_floors}
    imp_rel = [imp for imp in layer.impairments if set(imp.floor_ids) & relevant]
    if imp_rel and not (card.hazards and card.hazards[0].kind == "impairment"):
        bad.append("impairment on an alert or relocation floor is not first")
    for d in card.decisions:
        if d.ai_role != "evidence only" or d.owner not in ("FSD", "IC"):
            bad.append(f"decision {d.id} not owned by a human")
    return bad


# ------------------------------------------------------------------------------------------ SVG drawings

COL = {"fire": "#c62828", "alert": "#ef8a17", "staging": "#1f6fb2", "relocate": "#2e9d57", "smoke": "#9c8fd6", "normal": "#eef0f2",
       "ink": "#1d2433", "muted": "#5b6474", "line": "#9aa3b2", "evac": "#2e9d57", "attack": "#d35400", "lift": "#6a3fb5",
       "water": "#1f6fb2", "smokeline": "#7d7d7d", "ai": "#0b7a75", "human": "#8e2b8e", "bg": "#ffffff", "warn": "#b00020"}
FONT = "font-family='Segoe UI, Arial, sans-serif'"


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _t(x: float, y: float, s: Any, size: int = 11, fill: str = COL["ink"], anchor: str = "start", weight: str = "normal", italic: bool = False) -> str:
    st = " font-style='italic'" if italic else ""
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}'{st}>"
            f"{_e(s)}</text>")


def _defs() -> str:
    marks = []
    for name, c in (("ink", COL["ink"]), ("evac", COL["evac"]), ("attack", COL["attack"]), ("lift", COL["lift"]),
                    ("water", COL["water"]), ("smoke", COL["smokeline"]), ("ai", COL["ai"]), ("human", COL["human"])):
        marks.append(f"<marker id='m-{name}' markerWidth='9' markerHeight='9' refX='7' refY='4.5' orient='auto' markerUnits='userSpaceOnUse'>"
                     f"<path d='M0,0 L9,4.5 L0,9 z' fill='{c}'/></marker>")
    return "<defs>" + "".join(marks) + "</defs>"


def _svg(w: int, h: int, body: List[str], title: str) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px;height:auto' "
            f"role='img' aria-label='{_e(title)}' {FONT}>" + f"<title>{_e(title)}</title>" + _defs()
            + f"<rect x='0' y='0' width='{w}' height='{h}' fill='{COL['bg']}' rx='8'/>" + "".join(body) + "</svg>")


def _arrow(x1: float, y1: float, x2: float, y2: float, color: str, mark: str, width: float = 2.5, dash: str = "") -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{width}'{d} "
            f"marker-end='url(#m-{mark})'/>")


def svg_section(card: PlanCard, layer: FireLayer) -> str:
    """Building section (west to east) for the card's fire floor: floor bands, Stair A and Stair B, the core, the
    evacuation route, the attack line to the standpipe landing, the fire department elevator exit, smoke up the
    shafts, firefighting water down, the fire command center, recalled elevators and the battery room on floor 1."""
    top = card.top
    row = max(14, min(22, 620 // (top + 1)))
    y0 = 92
    W = 1000
    H = y0 + (top + 1) * row + 150
    xa0, xa1 = 64, 100  # Stair A
    xw0, xw1 = 100, 250  # west plate
    xc0, xc1 = 250, 330  # core (shaft, lifts)
    xe0, xe1 = 330, 480  # east plate
    xb0, xb1 = 480, 516  # Stair B
    xn = 536  # notes
    stair_x = {"A": ((xa0 + xa1) / 2), "B": ((xb0 + xb1) / 2)}

    def yf(level: int) -> float:  # top edge of a floor row
        return y0 + (top + 1 - level) * row

    rows = section_rows(card)
    body = [_t(16, 26, f"Building section, fire on {card.floor_id} ({card.zone_id}): SYNTHETIC tower, {top} floors", 15, weight="bold"),
            _t(16, 46, "DECISION SUPPORT ONLY: alarms, sprinklers, smoke control and elevators run on their own listed controls. Humans decide.",
               11, COL["warn"], weight="bold"),
            _t(16, 64, "West to east cut through Stair A, the west office, the core (shafts and elevators), the east office and Stair B.", 11, COL["muted"])]
    # roof
    body.append(f"<rect x='{xa0}' y='{yf(top + 1):.1f}' width='{xb1 - xa0}' height='{row}' fill='#dfe3e8' stroke='{COL['line']}'/>")
    body.append(_t(xw0 + 4, yf(top + 1) + row - 4, "ROOF: helicopter landing or equivalency (SYNTHETIC); stairs reach the roof", max(9, row - 6)))
    fs = max(10, min(13, row - 5))
    for r in rows:
        lv, band = r["level"], r["band"]
        y = yf(lv)
        c = COL[band]
        body.append(_t(xa0 - 6, y + row - 4, r["floor"], fs, COL["muted"], anchor="end"))
        body.append(f"<rect x='{xw0}' y='{y:.1f}' width='{xw1 - xw0}' height='{row}' fill='{c}' fill-opacity='{0.85 if band != 'normal' else 1}' stroke='{COL['line']}' stroke-width='0.6'/>")
        body.append(f"<rect x='{xe0}' y='{y:.1f}' width='{xe1 - xe0}' height='{row}' fill='{c}' fill-opacity='{0.85 if band != 'normal' else 1}' stroke='{COL['line']}' stroke-width='0.6'/>")
        body.append(f"<rect x='{xc0}' y='{y:.1f}' width='{xc1 - xc0}' height='{row}' fill='#d9dde3' stroke='{COL['line']}' stroke-width='0.6'/>")
        for sid, (s0, s1) in (("A", (xa0, xa1)), ("B", (xb0, xb1))):
            fill = "#e7f4ec" if sid == card.evac_stair else "#fdeee2"
            body.append(f"<rect x='{s0}' y='{y:.1f}' width='{s1 - s0}' height='{row}' fill='{fill}' stroke='{COL['line']}' stroke-width='0.6'/>")
        cam = layer.stairs[0].camera_every
        if cam and lv % cam == 0:
            for sx in (xa1 - 6, xb1 - 6):
                body.append(f"<circle cx='{sx}' cy='{y + 5:.1f}' r='2.6' fill='{COL['ink']}'/>")
        if r["note"]:
            body.append(_t(xn, y + row - 4, f"{r['floor']}: {r['note']}", fs, COL["ink"] if band != "normal" else COL["muted"]))
    # labels above
    body.append(_t(xa0 - 4, y0 - 8, f"Stair A", 11, weight="bold"))
    body.append(_t(xb0 - 4, y0 - 8, f"Stair B", 11, weight="bold"))
    body.append(_t((xc0 + xc1) / 2, y0 - 8, "core", 11, anchor="middle", weight="bold"))
    body.append(_t((xw0 + xw1) / 2, y0 - 8, "west office", 11, COL["muted"], anchor="middle"))
    body.append(_t((xe0 + xe1) / 2, y0 - 8, "east office", 11, COL["muted"], anchor="middle"))
    # fire marker in the incident zone
    yfire = yf(card.level) + row / 2
    side = card.zone_id.split("-")[-1]
    fx = {"W": (xw0 + xw1) / 2, "E": (xe0 + xe1) / 2, "N": (xw0 + xe1) / 2 - 60, "S": (xw0 + xe1) / 2 + 60}.get(side, (xc0 + xc1) / 2)
    body.append(f"<circle cx='{fx:.1f}' cy='{yfire:.1f}' r='{row / 2 + 2:.1f}' fill='#ffeb3b' stroke='{COL['fire']}' stroke-width='2'/>")
    body.append(_t(fx, yfire + 4, "FIRE", 10, COL["fire"], anchor="middle", weight="bold"))
    # impairments and assistance markers
    for imp in layer.impairments:
        for f in imp.floor_ids:
            lv = int(f[1:])
            body.append(_t(xe1 - 4, yf(lv) + row - 4, "IMPAIRED", max(8, fs - 1), COL["warn"], anchor="end", weight="bold"))
    for a in layer.assistance:
        lv = int(a.floor_id[1:])
        body.append(f"<circle cx='{xw0 + 10}' cy='{yf(lv) + row / 2:.1f}' r='4' fill='#ffffff' stroke='{COL['human']}' stroke-width='2'/>")
    # evacuation arrow: top alert floor down the evacuation stair to the top relocation floor (or floor 1)
    ex = stair_x[card.evac_stair]
    y_from = yf(max(card.alert_floors)) + 3
    y_to = (yf(max(card.relocation_floors)) + row / 2) if card.relocation_floors else (yf(1) + row / 2)
    body.append(_arrow(ex, y_from, ex, y_to, COL["evac"], "evac", 4))
    # attack line: from staging (or floor 1) up the attack stair to the standpipe landing
    ax = stair_x[card.attack_stair]
    y_start = yf(card.staging_floor) + row if card.staging_floor else yf(1) + row
    y_land = yf(card.standpipe_landing) + row / 2
    if y_start - y_land > 4:
        body.append(_arrow(ax, y_start, ax, y_land + 3, COL["attack"], "attack", 4))
    body.append(f"<rect x='{ax - 6}' y='{y_land - 5:.1f}' width='12' height='10' fill='{COL['attack']}'/>")
    # fire department elevator (Phase II) from floor 1 to the exit floor
    lx = xc0 + 58
    if card.fd_lift_exit_floor:
        body.append(_arrow(lx, yf(1) + row - 2, lx, yf(card.fd_lift_exit_floor) + row / 2 + 3, COL["lift"], "lift", 2.5, "6,3"))
    # smoke up the shaft
    sx = xc0 + 16
    top_smoke = max(card.smoke_watch_floors) if card.smoke_watch_floors else max(card.alert_floors)
    if top_smoke > card.level:
        body.append(_arrow(sx, yfire, sx, yf(top_smoke) + 4, COL["smokeline"], "smoke", 3, "2,3"))
    # water down on the plate where the graph says it goes
    if card.water_watch:
        lowest = min(w.level for w in card.water_watch)
        wx = fx + (18 if fx < xc0 else -18)
        body.append(_arrow(wx, yfire + row / 2, wx, yf(lowest) + row - 3, COL["water"], "water", 3))
    # floor 1 labels
    body.append(_t(xa0, yf(1) + row + 16, "F01: fire command center, battery room ESS-1; passenger cars recalled only if a lobby or hoistway "
                   "detector activates", fs, COL["ink"]))
    # legend
    ly = yf(1) + row + 40
    items = [("fire", "fire floor"), ("alert", "alert (voice alarm)"), ("staging", "FD staging"), ("relocate", "relocation"),
             ("smoke", "smoke watch")]
    for i, (k, lab) in enumerate(items):
        body.append(f"<rect x='{16 + i * 150}' y='{ly}' width='14' height='12' fill='{COL[k]}'/>")
        body.append(_t(36 + i * 150, ly + 10, lab, 11))
    lines = [(COL["evac"], f"green: occupants go down Stair {card.evac_stair} (evacuation stair, suggested)"),
             (COL["attack"], f"orange: fire department climbs Stair {card.attack_stair}; square = standpipe hose connection at F{card.standpipe_landing:02d}"),
             (COL["lift"], "purple dashed: fire service elevator in Phase II (fire department only)"),
             (COL["smokeline"], "grey dotted: smoke may rise in shafts (stack effect); blue: firefighting water moving down (water graph)"),
             (COL["ink"], f"dots: stair camera every {layer.stairs[0].camera_every or '-'}th landing (an LAFD Req. 10 option); "
                          "rings: person needing help (SYNTHETIC)")]
    for i, (c, s) in enumerate(lines):
        body.append(_t(16, ly + 32 + i * 16, s, 11, c))
    return _svg(W, H, body, f"Building section for a fire on {card.floor_id}")


def _box(x: float, y: float, w: float, h: float, lines: Sequence[str], fill: str, stroke: str, dash: str = "", bold_first: bool = True,
         size: int = 11, color: str = COL["ink"]) -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    out = [f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='6' fill='{fill}' stroke='{stroke}' stroke-width='1.6'{d}/>"]
    ty = y + 15
    for i, s in enumerate(lines):
        out.append(_t(x + 8, ty + i * 14, s, size, color, weight="bold" if (i == 0 and bold_first) else "normal"))
    return "".join(out)


def _diamond(cx: float, cy: float, w: float, h: float, lines: Sequence[str], stroke: str = COL["human"]) -> str:
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    out = [f"<polygon points='{pts}' fill='#fbeefb' stroke='{stroke}' stroke-width='1.8'/>"]
    y = cy - (len(lines) - 1) * 6.5 + 4
    for i, s in enumerate(lines):
        out.append(_t(cx, y + i * 13, s, 10.5, COL["ink"], anchor="middle", weight="bold" if i == 0 else "normal"))
    return "".join(out)


def svg_flowchart(card: PlanCard) -> str:
    """De-escalation flowchart: phases 0-6 (left), human decision diamonds (middle, 'human decides') and AI support
    boxes marked 'evidence only' (right, dashed). Floor numbers come from the card."""
    fl = lambda xs: (f"F{min(xs):02d}-F{max(xs):02d}" if xs else "none")  # noqa: E731
    reloc = fl(card.relocation_floors) if card.relocation_floors else "outside"
    W, H = 1000, 806
    px, pw = 20, 250  # phase column
    dx = 430  # decision diamond centre
    ax, aw = 700, 285  # AI column
    rows = [
        ("0 PREPARE", ["plan cards, impairments, drills,", "occupant counts kept current"], None,
         ["AI: evidence only", "builds this plan card from the building graph;", "impairment list; drill tracking: not built"]),
        ("1 VERIFY", ["listed alarm, or an AI early alert", "(an AI alert is NOT an alarm)"], ("D1 FSD calls 911", "AI cannot delay it"),
         ["AI: evidence only", "camera near the device: not built; sensor", "label: did not transfer to a new room"]),
        ("2 PROTECT", [f"alert {fl(card.alert_floors)}; relocate to {reloc}", f"evacuation stair {card.evac_stair} (suggested)"],
         ("D2 FSD approves", "reads one message live"),
         ["AI: evidence only", "drafts plan card; lists people needing help", "on alert floors (message template: not built)"]),
        ("3 HAND OVER", ["fire department arrives at the", "fire command center"], ("D3 IC takes", "command"),
         ["AI: evidence only", f"briefing: attack stair {card.attack_stair}, standpipe F{card.standpipe_landing:02d},",
          "elevators, panels, risers, hazards"]),
        ("4 CONTAIN", ["fire department operations", "(FSCP, Phase II: fire department only)"], ("Escalation", "trigger E1-E6?"),
         ["AI: evidence only", "alarm spread; where firefighting water goes;", "live shaft readings, stair cameras: not built"]),
        ("5 RECOVER", ["IC declares knockdown; fire watch", "where systems are impaired"], ("D5 IC authorizes", "resets, re-occupancy"),
         ["AI: evidence only", "salvage list for floors below the fire;", "reset checklist: not built"]),
        ("6 LEARN", ["after-action review with LAFD", ""], None,
         ["AI: evidence only", "timeline: not built; plan card updates", "by rerunning the planner"]),
    ]
    body = [_t(20, 28, f"De-escalation flow for a fire on {card.floor_id}: smallest safe response, widened only by written triggers", 15, weight="bold"),
            _t(20, 48, "Purple diamonds: a human decides (FSD, then the fire department IC). Teal dashed boxes: AI evidence only; "
                   "'not built' = design scope only.",
               11, COL["muted"]),
            _t(20, 64, "The listed fire alarm system runs its own sequence (voice alarm, smoke control, stair unlock) with no AI in the loop.",
               11, COL["warn"], weight="bold")]
    y = 84
    rh = 104
    for i, (ph, ptxt, dec, ai) in enumerate(rows):
        cy = y + i * rh + 28
        body.append(_box(px, y + i * rh, pw, 56, [ph] + ptxt, "#f3f5f8", COL["ink"]))
        body.append(_box(ax, y + i * rh, aw, 56, ai, "#e8f6f5", COL["ai"], dash="6,4", color=COL["ink"]))
        body.append(f"<line x1='{ax}' y1='{cy}' x2='{px + pw}' y2='{cy}' stroke='{COL['ai']}' stroke-width='1' stroke-dasharray='3,3'/>")
        if dec:
            body.append(_arrow(px + pw, cy, dx - 76, cy, COL["ink"], "ink", 1.6))
            body.append(_diamond(dx, cy, 150, 64, dec))
        if i < len(rows) - 1:
            if dec:
                # decision leads down to the next phase
                body.append(f"<path d='M{dx},{cy + 32} L{dx},{cy + rh - 34} L{px + pw / 2 + 4},{cy + rh - 34}' fill='none' stroke='{COL['ink']}' stroke-width='1.6'/>")
                body.append(_arrow(px + pw / 2, cy + rh - 34, px + pw / 2, y + (i + 1) * rh - 1, COL["ink"], "ink", 1.6))
            else:
                body.append(_arrow(px + pw / 2, y + i * rh + 56, px + pw / 2, y + (i + 1) * rh - 1, COL["ink"], "ink", 1.6))
    # escalation loop at phase 4
    c4 = y + 4 * rh + 28
    d4x = dx + 172
    body.append(_diamond(dx, c4 + 0, 150, 64, ("Escalation", "trigger E1-E6?")))
    body.append(_arrow(dx + 75, c4, d4x - 76, c4, COL["human"], "human", 1.6))
    body.append(_diamond(d4x, c4, 150, 58, ("D4 IC widens zone", "or total evacuation")))
    body.append(f"<path d='M{d4x},{c4 - 29} L{d4x},{c4 - 46} L60,{c4 - 46} L60,{c4 - 31}' fill='none' stroke='{COL['human']}' "
                f"stroke-width='1.6' marker-end='url(#m-human)'/>")
    body.append(_t(d4x - 70, c4 - 50, "back to 4 with the wider zone", 10, COL["human"]))
    body.append(_t(dx + 78, c4 - 6, "yes", 10, COL["human"]))
    body.append(_t(dx + 6, c4 + 46, "no / knockdown", 10, COL["muted"]))
    # AI early alert side path at phase 1
    c1 = y + rh + 28
    body.append(_t(px + 4, c1 + 44, "AI early alert only: D0 FSD checks by camera; staff by stair, never by elevator", 10, COL["human"], italic=True))
    body.append(_t(20, H - 14, "Sources: LAFD high-rise sequence policy; LA Ord. 180,648; NFPA high-rise EAP guide; IBC 909.16 (FSCP priority); "
                   "Sacramento Regional high-rise guideline; First Interstate 1988 (LAFD archive).", 10, COL["muted"]))
    return _svg(W, H, body, f"De-escalation flowchart for a fire on {card.floor_id}")


LANES: Tuple[Tuple[str, str], ...] = (("FAS", "Fire alarm system (listed)"), ("AI", "Cerebro AI (read-only)"), ("FSD", "FSD at the FCC"),
                                      ("OCC", "Wardens and occupants"), ("FD", "Fire department IC"))


def swimlane_steps(card: PlanCard) -> List[Tuple[str, str, str, str]]:
    """Ordered (phase, sender, receiver, message) steps; no clock times are claimed."""
    fl = lambda xs: (f"F{min(xs):02d}-F{max(xs):02d}" if xs else "none")  # noqa: E731
    reloc = fl(card.relocation_floors) if card.relocation_floors else "the outside assembly area"
    lift = (f"Phase II cars, exit F{card.fd_lift_exit_floor:02d}" if card.fd_lift_exit_floor else "stairs (low fire)")
    return [
        ("1 Verify", "FAS", "OCC", f"voice alarm on {fl([x for x in card.alert_floors if abs(x - card.level) <= 1])}"),
        ("1 Verify", "FAS", "FAS", "listed sequence: smoke control, stair pressurization, doors; recall only if lobby/hoistway detector"),
        ("1 Verify", "FAS", "AI", "device event (read-only feed)"),
        ("1 Verify", "AI", "FSD", "evidence panel + draft plan card (evidence only)"),
        ("1 Verify", "FSD", "FD", "911 call (D1); never delayed by the AI"),
        ("2 Protect", "FSD", "OCC", f"approved live message (D2): Stair {card.evac_stair} to {reloc}"),
        ("2 Protect", "AI", "FSD", f"{card.occupants.get('assistance_count', 0)} person(s) needing help on alert floors (stair cameras: not built)"),
        ("3 Hand over", "FD", "FSD", "arrives at the fire command center (time varies)"),
        ("3 Hand over", "FSD", "FD", "briefing card (AI drafted, FSD checked)"),
        ("3 Hand over", "FD", "FAS", f"FSCP, {lift}, standpipe Stair {card.attack_stair} F{card.standpipe_landing:02d} (FD only, D3)"),
        ("4 Contain", "AI", "FD", "alarm spread, water migrating below (live shaft readings: not built)"),
        ("4 Contain", "FD", "FSD", "widen zone or total evacuation if a trigger fires (D4)"),
        ("5 Recover", "FD", "FSD", "knockdown; authorizes resets and re-occupancy (D5)"),
        ("6 Learn", "AI", "FSD", "after-action timeline (not built)"),
    ]


def svg_swimlane(card: PlanCard) -> str:
    """Swimlane: five lanes as columns, ordered steps as rows, one arrow per message. No clock times."""
    steps = swimlane_steps(card)
    lane_w = 180
    x0 = 110
    W = x0 + lane_w * len(LANES) + 20
    y0 = 104
    rh = 46
    H = y0 + rh * len(steps) + 60
    cx = {k: x0 + lane_w * i + lane_w / 2 for i, (k, _) in enumerate(LANES)}
    body = [_t(16, 26, f"Who does what, in order (fire on {card.floor_id}); no clock times are claimed", 15, weight="bold"),
            _t(16, 46, "The AI lane only reads and shows evidence. Every arrow into the fire alarm system comes from the fire department.", 11, COL["warn"],
               weight="bold")]
    for i, (k, name) in enumerate(LANES):
        x = x0 + lane_w * i
        fill = "#e8f6f5" if k == "AI" else ("#f7f0f7" if k in ("FSD", "FD") else "#f3f5f8")
        body.append(f"<rect x='{x}' y='{y0 - 36}' width='{lane_w - 6}' height='{rh * len(steps) + 40}' fill='{fill}' rx='6'/>")
        body.append(_t(x + (lane_w - 6) / 2, y0 - 18, name, 13, COL["ai"] if k == "AI" else COL["ink"], anchor="middle", weight="bold"))
        if k == "AI":
            body.append(_t(x + (lane_w - 6) / 2, y0 - 5, "evidence only", 10, COL["ai"], anchor="middle", italic=True))
    last_phase = None
    for j, (ph, s, r, msg) in enumerate(steps):
        y = y0 + j * rh + 22
        if ph != last_phase:
            body.append(f"<line x1='12' y1='{y - 22}' x2='{W - 12}' y2='{y - 22}' stroke='{COL['line']}' stroke-width='0.8' stroke-dasharray='4,3'/>")
            body.append(_t(14, y - 8, ph, 12.5, COL["human"], weight="bold"))
            last_phase = ph
        color = COL["ai"] if s == "AI" else (COL["attack"] if s == "FD" else COL["ink"])
        mark = "ai" if s == "AI" else ("attack" if s == "FD" else "ink")
        if s == r:
            x = x0 + 10
            body.append(f"<path d='M{x},{y - 4} C{x + 50},{y - 18} {x + 50},{y + 14} {x + 6},{y + 4}' fill='none' stroke='{color}' stroke-width='2' marker-end='url(#m-{mark})'/>")
            tx, anchor = x + 52, "start"
        else:
            x1, x2 = cx[s], cx[r]
            body.append(f"<circle cx='{x1}' cy='{y}' r='4' fill='{color}'/>")
            body.append(_arrow(x1, y, x2 + (-8 if x2 > x1 else 8), y, color, mark, 2))
            tx, anchor = ((x1 + x2) / 2, "middle")
        body.append(_t(tx, y - 7, f"{j + 1}. {msg}", 13, COL["ink"], anchor=anchor))
    body.append(_t(16, H - 16, "Sequence after the ask-5 research (LAFD policy, LA Ord. 180,648, NFPA EAP guide, Sacramento guideline). "
                   "SYNTHETIC tower.", 10, COL["muted"]))
    return _svg(W, H, body, f"Swimlane timeline for a fire on {card.floor_id}")
