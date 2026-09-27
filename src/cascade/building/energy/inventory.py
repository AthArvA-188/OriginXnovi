"""SEMI-SYNTHETIC common-area inventory for the existing SYNTHETIC 32-floor demo tower.

The synthetic tower (cascade.building.synthetic) has no corridor, stair, lobby, garage, restroom or amenity zones and
its Building.tz defaults to UTC. This module defines those zones for the energy module only, in America/Los_Angeles.
Every area, lighting power density and design illuminance below is an [ASSUMPTION] with no public source; a lighting
designer must replace them with the real fixture schedule and a photometric check. Occupancy for each zone kind is
resampled from REAL ROBOD test days of one room; that room-to-zone mapping is also an [ASSUMPTION] (ROBOD has no
corridor, garage or restroom).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from ..synthetic import TowerConfig
from .rules import zone_rule
from .policy import check_design

TZ = "America/Los_Angeles"

# kind: (area_ft2, W_per_ft2, design_fc, ROBOD room used for occupancy)  -- all [ASSUMPTION]
ASSUMED = {
    "corridor": (1500.0, 0.50, 10.0, 4),
    "stairwell": (250.0, 0.60, 15.0, 1),
    "lobby": (4000.0, 1.00, 20.0, 5),
    "garage": (30000.0, 0.15, 5.0, 4),
    "restroom": (200.0, 0.60, 15.0, 3),
    "amenity": (1500.0, 0.80, 30.0, 2),
}
ROBOD_ROOM_NAMES = {1: "Room 1 lecture", 2: "Room 2 lecture", 3: "Room 3 office", 4: "Room 4 office", 5: "Room 5 library"}


@dataclass(frozen=True)
class CommonZone:
    zone_id: str
    kind: str
    floor: str
    area_ft2: float
    w_per_ft2: float
    design_fc: float
    robod_room: int

    @property
    def design_w(self) -> float:
        return self.area_ft2 * self.w_per_ft2


def tower_inventory(floors: int | None = None) -> List[CommonZone]:
    floors = floors or TowerConfig().floors
    zones: List[CommonZone] = []

    def add(zid, kind, floor):
        a, w, fc, room = ASSUMED[kind]
        z = CommonZone(zid, kind, floor, a, w, fc, room)
        check_design(zone_rule(kind), fc)
        zones.append(z)

    for n in range(1, floors + 1):
        f = f"F{n:02d}"
        add(f"{f}-CORR", "corridor", f)
        add(f"{f}-STAIR-A", "stairwell", f)
        add(f"{f}-STAIR-B", "stairwell", f)
        add(f"{f}-WC-1", "restroom", f)
        add(f"{f}-WC-2", "restroom", f)
    add("L1-LOBBY", "lobby", "L1")
    for g in range(1, 5):
        add(f"P{g}-GARAGE", "garage", f"P{g}")
    amen = f"F{max(2, floors // 2):02d}"
    for name in ("GYM", "LOUNGE", "CONF"):
        add(f"{amen}-{name}", "amenity", amen)
    return zones


def inventory_summary(zones: List[CommonZone]) -> Dict:
    by = {}
    for z in zones:
        b = by.setdefault(z.kind, {"count": 0, "area_ft2": 0.0, "design_w": 0.0})
        b["count"] += 1
        b["area_ft2"] += z.area_ft2
        b["design_w"] += z.design_w
    assumptions = [{"kind": k, "area_ft2_each": a, "w_per_ft2": w, "design_fc": fc, "occupancy_from": ROBOD_ROOM_NAMES[r],
                    "tag": "Assumption"} for k, (a, w, fc, r) in ASSUMED.items()]
    return {"label": "SEMI-SYNTHETIC", "tz": TZ, "floors": TowerConfig().floors, "n_zones": len(zones), "by_kind": by,
            "total_design_w": sum(z.design_w for z in zones), "assumptions": assumptions,
            "zones": [asdict(z) for z in zones]}
