"""Code-plausible riser design for the SYNTHETIC 32-floor tower (all values are team assumptions unless tagged).

Fixes the research verifier's blocking issue: pressure-reducing valves (PRVs) feed the LOWER zones and the booster
feeds the top zone directly, so every outlet stays at or below 80 psi (56.2 m) static (UPC 608.2, PUBLIC).

    booster room (z = 0), booster hydraulic grade 134 m  ->  express riser DN100 up the core
      zone L  floors  1-11  PRV station at F1  (strainer + PRV, setpoint 54 m at F1)
      zone M  floors 12-22  PRV station at F12 (strainer + PRV, setpoint 54 m at F12)
      zone H  floors 23-32  fed directly from the express riser (static 53.5 m at F23, 22 m at F32)

Pressure sensors: riser base, each PRV station's strainer inlet and outlet, and the header, middle and top floor of
each zone. Flow meters: one clamp-on meter per zone header. Test valves: one at the top floor of each zone.
Location classes: two riser segments per zone (lower / upper half) plus the two strainers = 8 locations.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

FLOORS = 32
FLOOR_H = 3.5  # m per storey [Assumption]
UNITS_PER_FLOOR = 6  # apartments per floor [Assumption]
BOOSTER_HEAD = 134.0  # m hydraulic grade at the booster discharge [Assumption]
PRV_SET = 54.0  # m pressure at the PRV outlet (header floor) [Assumption]
ROUGHNESS_C = 130.0  # Hazen-Williams C, new pipe (EPANET manual: cast iron 130-140)
ZONE_DIAM = 0.08  # m, zone riser DN80 [Assumption]
EXPRESS_DIAM = 0.10  # m, express riser DN100 [Assumption]
STRAINER_K_CLEAN = 2.0  # minor-loss K of a clean strainer [Assumption]
MAX_STATIC_M = 56.2  # 80 psi (UPC 608.2)
PSI_PER_M = 1.0 / 0.703  # 1 m water = 1.422 psi

ZONES: Dict[str, dict] = {
    "L": {"floors": (1, 11), "fed_by": "PRV", "station_floor": 1, "mid": 6, "name": "Low zone (F1-F11)"},
    "M": {"floors": (12, 22), "fed_by": "PRV", "station_floor": 12, "mid": 17, "name": "Mid zone (F12-F22)"},
    "H": {"floors": (23, 32), "fed_by": "booster", "station_floor": 23, "mid": 28, "name": "High zone (F23-F32)"},
}

# location classes (index = class id used by the localiser)
LOCATIONS: List[str] = ["L1", "L2", "M1", "M2", "H1", "H2", "STR_L", "STR_M"]
LOCATION_NAMES = {
    "L1": "Low zone riser F1-F6", "L2": "Low zone riser F6-F11",
    "M1": "Mid zone riser F12-F17", "M2": "Mid zone riser F17-F22",
    "H1": "High zone riser F23-F28", "H2": "High zone riser F28-F32",
    "STR_L": "Strainer at low-zone PRV (F1)", "STR_M": "Strainer at mid-zone PRV (F12)",
}
LOCATION_ZONE = {"L1": "L", "L2": "L", "M1": "M", "M2": "M", "H1": "H", "H2": "H", "STR_L": "L", "STR_M": "M"}
# sensor pair (upstream, downstream) bracketing each location
SEGMENT_SENSORS: Dict[str, Tuple[str, str]] = {
    "L1": ("F01", "F06"), "L2": ("F06", "F11"),
    "M1": ("F12", "F17"), "M2": ("F17", "F22"),
    "H1": ("F23", "F28"), "H2": ("F28", "F32"),
    "STR_L": ("XL", "SL_OUT"), "STR_M": ("XM", "SM_OUT"),
}
PRESSURE_SENSORS = ["B0", "XL", "SL_OUT", "F01", "F06", "F11", "XM", "SM_OUT", "F12", "F17", "F22", "F23", "F28",
                    "F32"]
FLOW_METERS = {"L": "STR_L", "M": "STR_M", "H": "X3"}  # link whose flow the zone meter reads
TEST_FLOOR = {"L": 11, "M": 22, "H": 32}


def fid(f: int) -> str:
    return f"F{f:02d}"


def z_of_floor(f: int) -> float:
    return f * FLOOR_H


def node_elevation(node: str) -> float:
    if node == "B0":
        return 0.0
    if node in ("XL", "SL_OUT"):
        return z_of_floor(1)
    if node in ("XM", "SM_OUT"):
        return z_of_floor(12)
    if node.startswith("F"):
        return z_of_floor(int(node[1:]))
    raise KeyError(node)


def zone_of_floor(f: int) -> str:
    for z, d in ZONES.items():
        lo, hi = d["floors"]
        if lo <= f <= hi:
            return z
    raise ValueError(f)


def static_pressure_m(f: int) -> float:
    """Static (no-flow) pressure head at floor f's outlets, from the zoning rules above."""
    z = zone_of_floor(f)
    d = ZONES[z]
    if d["fed_by"] == "PRV":
        return PRV_SET - (z_of_floor(f) - z_of_floor(d["station_floor"]))
    return BOOSTER_HEAD - z_of_floor(f)


def static_table() -> List[dict]:
    rows = []
    for f in range(1, FLOORS + 1):
        p = static_pressure_m(f)
        rows.append({"floor": f, "zone": zone_of_floor(f), "static_m": round(p, 2), "static_psi": round(p * PSI_PER_M, 1)})
    return rows


def location_pipes(loc: str) -> List[str]:
    """Pipe names that belong to a location class (a clog is placed on one of them)."""
    if loc == "STR_L":
        return ["STR_L"]
    if loc == "STR_M":
        return ["STR_M"]
    a, b = SEGMENT_SENSORS[loc]
    fa, fb = int(a[1:]), int(b[1:])
    return [f"P{f:02d}" for f in range(fa + 1, fb + 1)]  # pipe Pnn ends at floor nn


def severity_of_k(k: float, bands: Dict[str, list]) -> int:
    """0 clean, 1 mild, 2 moderate, 3 severe from contiguous K bands (lower edge inclusive)."""
    if k <= 0:
        return 0
    if k < bands["moderate"][0]:
        return 1
    if k < bands["severe"][0]:
        return 2
    return 3


SEVERITY_NAMES = ["clean", "mild", "moderate", "severe"]
