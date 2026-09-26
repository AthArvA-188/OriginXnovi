"""Data model for smart-building and skyscraper problem management (building_spec section 2).

Pydantic v2 models for the building (floors, zones, elements, sensors, edges), for observations and for
problems, plus plain dataclasses for analysis results that hold DataFrames. No I/O in this module.

Graph edge direction (building_spec section 4):
- drains_to: src to dst in the direction water moves.
- above: src is the upper node, dst the lower node (water moves down through the slab).
- adjacent: same floor, water can move both ways.
- feeds: electrical parent to child (switchboard to panel to circuit to load).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Set, Tuple, TypeVar

from pydantic import BaseModel, PrivateAttr, model_validator

from ..schema import Finding, Flag, ImageRecord, Level

if TYPE_CHECKING:  # pandas is only needed at runtime by the modules that build these containers
    import pandas as pd

T = TypeVar("T")

Orientation = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
# outward facade normal, degrees clockwise from north
AZIMUTH: Dict[str, float] = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}
ZoneKind = Literal["room", "facade_drop", "roof", "shaft", "electrical_room", "core"]
ElementKind = Literal["facade_panel", "window", "roof_drain", "riser", "switchboard", "electrical_panel", "circuit", "load", "sensor"]
SensorType = Literal["humidity", "temperature", "leak", "circuit_current", "surface_moisture"]
EdgeKind = Literal["drains_to", "above", "adjacent", "feeds"]
ObservationKind = Literal["image_finding", "thermal_reading", "sensor_event", "load_event", "ticket"]
ProblemStatus = Literal["detected", "triaged", "work_order", "resolved", "verified"]
ProblemDomain = Literal["water", "electrical", "facade", "mixed"]
TicketCategory = Literal["water", "facade", "electrical", "other"]

WATER_EDGE_KINDS: Tuple[str, ...] = ("drains_to", "above", "adjacent")
ELECTRICAL_KINDS: Tuple[str, ...] = ("switchboard", "electrical_panel", "circuit", "load")


class Floor(BaseModel):
    floor_id: str  # "F01".."F32", "RF" for roof
    level: int  # 1 = ground; roof = floors + 1
    elevation_m: float
    plan_image: Optional[str] = None  # path relative to the building folder
    plan_size_px: Tuple[int, int] = (800, 600)
    px_per_m: Optional[float] = None


class Zone(BaseModel):
    zone_id: str  # "F12-E" room, "F12-FD-E" facade drop, "F12-SHAFT", "F12-ELEC", "F12-CORE", "ROOF-NE"
    floor_id: str
    kind: ZoneKind
    name: str = ""
    polygon: List[Tuple[int, int]]  # plan pixel coordinates, >= 3 points
    orientation: Optional[Orientation] = None  # required when kind == "facade_drop"
    drop_id: Optional[str] = None  # vertical facade drop grouping, e.g. "FD-E"
    area_m2: Optional[float] = None

    @model_validator(mode="after")
    def _check(self) -> "Zone":
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.zone_id}: polygon needs at least 3 points, got {len(self.polygon)}")
        if self.kind == "facade_drop" and self.orientation is None:
            raise ValueError(f"zone {self.zone_id}: a facade_drop needs an orientation (N, NE, E, SE, S, SW, W, NW)")
        return self


class Element(BaseModel):
    element_id: str  # "RD-2", "R1@F12", "P-F12", "C-F12-3", "SWB-1", "FP-F12-E-03"
    kind: ElementKind
    zone_id: str
    name: str = ""
    xy: Optional[Tuple[int, int]] = None  # plan pixels; None means zone centroid
    rating_a: Optional[float] = None  # breaker rating for circuits and panels
    voltage_v: Optional[float] = None
    attrs: Dict[str, Any] = {}


class Sensor(BaseModel):
    sensor_id: str  # column name in sensors.csv.gz
    type: SensorType
    zone_id: str
    unit: str  # "%RH", "degC", "wet", "A", "%MC"
    element_id: Optional[str] = None  # circuit for circuit_current
    synthetic: bool = False


class Edge(BaseModel):
    src: str  # zone_id or element_id
    dst: str
    kind: EdgeKind
    weight: Optional[float] = None  # None = use rules.EDGE_WEIGHT[kind]
    basis: str = "[team-proposed, validate]"


class Building(BaseModel):
    building_id: str
    name: str
    address: str = ""
    height_m: Optional[float] = None
    floors: int
    year_built: Optional[int] = None
    client_id: Optional[str] = None
    fisp_cycle_due: Optional[str] = None  # YYYY-MM-DD end of the current FISP filing window, if known
    synthetic: bool = False
    tz: str = "UTC"  # IANA time zone of the site, for occupancy profiles (hour-of-week baselines follow local DST)
    floor_list: List[Floor]
    zones: List[Zone]
    elements: List[Element]
    sensors: List[Sensor]
    edges: List[Edge]

    _index: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _memo: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _check_refs(self) -> "Building":
        bad = self.validate_refs()
        if bad:
            raise ValueError("building has dangling or duplicate ids: " + "; ".join(bad[:20]) + (" ..." if len(bad) > 20 else ""))
        return self

    # ------------------------------------------------------------------ lookups (pure, cached per instance)
    def _idx(self) -> Dict[str, Dict[str, Any]]:
        if not self._index:
            self._index = {
                "floor": {f.floor_id: f for f in self.floor_list},
                "zone": {z.zone_id: z for z in self.zones},
                "element": {e.element_id: e for e in self.elements},
                "sensor": {s.sensor_id: s for s in self.sensors},
            }
        return self._index

    def memo(self, key: str, factory: Callable[[], T]) -> T:
        """Cache a derived structure (graph adjacency, sensor maps) on this instance. Treat the model as immutable."""
        if key not in self._memo:
            self._memo[key] = factory()
        return self._memo[key]

    def floor(self, floor_id: str) -> Optional[Floor]:
        return self._idx()["floor"].get(floor_id)

    def zone(self, zone_id: str) -> Optional[Zone]:
        return self._idx()["zone"].get(zone_id)

    def element(self, element_id: str) -> Optional[Element]:
        return self._idx()["element"].get(element_id)

    def sensor(self, sensor_id: str) -> Optional[Sensor]:
        return self._idx()["sensor"].get(sensor_id)

    def floor_of(self, node_id: str) -> Optional[Floor]:
        """Floor of a zone, an element (via its zone) or a sensor (via its zone)."""
        z = self.zone(node_id)
        if z is None:
            e = self.element(node_id)
            if e is not None:
                z = self.zone(e.zone_id)
        if z is None:
            s = self.sensor(node_id)
            if s is not None:
                z = self.zone(s.zone_id)
        return self.floor(z.floor_id) if z is not None else None

    def zones_on(self, floor_id: str) -> List[Zone]:
        return [z for z in self.zones if z.floor_id == floor_id]

    def elements_in(self, zone_id: str) -> List[Element]:
        return [e for e in self.elements if e.zone_id == zone_id]

    def sensors_in(self, zone_id: str) -> List[Sensor]:
        by_zone: Dict[str, List[Sensor]] = self.memo("sensors_by_zone", self._sensors_by_zone)
        return list(by_zone.get(zone_id, []))

    def _sensors_by_zone(self) -> Dict[str, List[Sensor]]:
        out: Dict[str, List[Sensor]] = {}
        for s in self.sensors:
            out.setdefault(s.zone_id, []).append(s)
        return out

    def node_ids(self) -> Set[str]:
        """Graph nodes: every zone id and element id (sensors are attached to zones, not graph nodes)."""
        return {z.zone_id for z in self.zones} | {e.element_id for e in self.elements}

    def validate_refs(self) -> List[str]:
        """Dangling and duplicate ids, as readable strings. Empty means the building is consistent."""
        bad: List[str] = []
        for label, ids in (("floor", [f.floor_id for f in self.floor_list]), ("zone", [z.zone_id for z in self.zones]),
                           ("element", [e.element_id for e in self.elements]), ("sensor", [s.sensor_id for s in self.sensors])):
            seen: Set[str] = set()
            for i in ids:
                if i in seen:
                    bad.append(f"duplicate {label} id {i}")
                seen.add(i)
        floors = {f.floor_id for f in self.floor_list}
        zones = {z.zone_id for z in self.zones}
        elements = {e.element_id for e in self.elements}
        overlap = zones & elements
        if overlap:
            bad.append(f"ids used as both zone and element: {sorted(overlap)[:5]}")
        for z in self.zones:
            if z.floor_id not in floors:
                bad.append(f"zone {z.zone_id} -> unknown floor {z.floor_id}")
        for e in self.elements:
            if e.zone_id not in zones:
                bad.append(f"element {e.element_id} -> unknown zone {e.zone_id}")
        for s in self.sensors:
            if s.zone_id not in zones:
                bad.append(f"sensor {s.sensor_id} -> unknown zone {s.zone_id}")
            if s.element_id is not None and s.element_id not in elements:
                bad.append(f"sensor {s.sensor_id} -> unknown element {s.element_id}")
        nodes = zones | elements
        for ed in self.edges:
            for end in (ed.src, ed.dst):
                if end not in nodes:
                    bad.append(f"edge {ed.src} -{ed.kind}-> {ed.dst}: unknown node {end}")
        return bad


class Observation(BaseModel):
    obs_id: str
    kind: ObservationKind
    ts: str  # ISO 8601 UTC, "2026-03-04T05:00:00Z"
    zone_id: Optional[str] = None
    element_id: Optional[str] = None  # at least one of zone_id / element_id
    source: str  # "grader", "sensor:F12-E-RH", "thermal", "ticket:T-0042", "load:C-F20-3"
    level: Optional[Level] = None  # None for tickets (not graded); U stays U
    value: Optional[float] = None
    unit: Optional[str] = None
    text: str = ""
    finding: Optional[Finding] = None  # the graded Finding when there is one
    plan_xy: Optional[Tuple[int, int]] = None
    synthetic: bool = False
    category: Optional[TicketCategory] = None  # tickets only: water, facade, electrical or other (additive to the spec)
    status: Optional[str] = None  # tickets only: the tickets.csv status ("open", "closed" ...); None when not given
    end_ts: Optional[str] = None  # when the condition ended (a leak wet run that dried); None = ongoing or not applicable
    ts_unknown: bool = False  # True when the capture date is unknown: ts is a placeholder and no time clock may use it

    @model_validator(mode="after")
    def _needs_place(self) -> "Observation":
        if self.zone_id is None and self.element_id is None:
            raise ValueError(f"observation {self.obs_id}: needs a zone_id or an element_id")
        return self


class Factor(BaseModel):
    """One named additive term of a score. value None = not measured: contribution 0, and the UI prints
    "not measured", never 0 as healthy."""

    name: str
    value: Optional[float]
    weight: float
    contribution: float
    explanation: str
    basis: str


class RootCause(BaseModel):
    node_id: str
    label: str
    score: float
    factors: List[Factor]
    explanation: str


class StatusChange(BaseModel):
    ts: str
    from_status: Optional[ProblemStatus]
    to_status: ProblemStatus
    actor: str
    note: str = ""


class Escalation(BaseModel):
    hours: Optional[float]
    lo_hours: Optional[float]
    hi_hours: Optional[float]
    what: str  # "drying window exceeded (EPA 24-48 h)", "Band 3 thermal", "SWARMP becomes Unsafe at next cycle"
    basis: str  # threshold key(s) from rules.THRESHOLDS
    as_of: Optional[str] = None  # the analysis time the hours count from (due = as_of + hours)


class Problem(BaseModel):
    problem_id: str  # stable across re-analysis: f"P-{root_node}" (the domain can change, the root keeps the id)
    domain: ProblemDomain
    title: str
    status: ProblemStatus = "detected"
    level: Level  # worst non-U level of its observations; "U" only when none is graded
    priority: Optional[float]  # None when level == "U" (listed, never scored)
    root_cause: RootCause
    alternatives: List[RootCause] = []
    observation_ids: List[str]
    affected_nodes: List[str]
    flags: List[Flag] = []
    escalation: Optional[Escalation] = None
    history: List[StatusChange] = []
    created_ts: str
    updated_ts: str
    synthetic: bool = False
    active: bool = True  # False when the latest analysis did not detect it again (kept as history, not open)


# ------------------------------------------------------------------------ result containers (hold DataFrames)

@dataclass
class ResponseFit:
    """Rain response of one zone to one driver."""

    zone_id: str
    sensor_id: str
    driver: str  # "wdr_E" .. "wdr_NW", "rain", or "none"
    n_events: int
    gain: Optional[float]
    r: Optional[float]
    lag_h: Optional[float]
    resid_sd: Optional[float]
    onset_ts: Optional[str]
    significant: bool


@dataclass
class StormEvent:
    storm_id: str
    start: str
    end: str
    total_mm: float
    peak_mm_h: float
    wind_dir_deg: float
    wind_speed_ms: float
    wdr: Dict[str, float]  # per orientation


@dataclass
class ZoneForecast:
    zone_id: str
    current: float
    peak: float
    lo: float
    hi: float
    hours_to_threshold: Optional[float]
    crosses: bool
    driver: str
    threshold_key: str


@dataclass
class WaterResult:
    storms: List[StormEvent] = field(default_factory=list)
    fits: Dict[str, ResponseFit] = field(default_factory=dict)
    risks: Dict[str, List[Factor]] = field(default_factory=dict)
    observations: List[Observation] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)


@dataclass
class ElectricalResult:
    load_table: "pd.DataFrame"
    anomalies: "pd.DataFrame"
    observations: List[Observation] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)


@dataclass
class BuildingData:
    root: Path
    building: Building
    weather: "pd.DataFrame"  # index ts (UTC DatetimeIndex); rain_mm, wind_speed_ms, wind_dir_deg, synthetic
    sensors: "pd.DataFrame"  # index ts (UTC DatetimeIndex); one float column per sensor_id
    thermal: "pd.DataFrame"  # ts (UTC datetime64), element_id, t_element_c, t_reference_c, t_ambient_c, load_pct, synthetic
    schedule: "pd.DataFrame"  # panel_id, fed_from, circuit_id, breaker_a, poles, voltage_v, load_desc, zone_id
    tickets: "pd.DataFrame"  # ts (UTC datetime64), ticket_id, zone_id, text, category, status, synthetic
    images: List[ImageRecord] = field(default_factory=list)
    synthetic_sources: Dict[str, bool] = field(default_factory=dict)  # per input file: any row labelled synthetic

    @property
    def any_synthetic(self) -> bool:
        """True when the building or any input file is synthetic."""
        return bool(self.building.synthetic or any(self.synthetic_sources.values()))
