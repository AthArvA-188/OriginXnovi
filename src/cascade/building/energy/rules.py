"""Loader for common_area_rules.json: per-zone-type code bounds and life-safety floors (cited, not a compliance tool)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RULES_PATH = Path(__file__).with_name("common_area_rules.json")
ZONE_KINDS = ("corridor", "stairwell", "lobby", "garage", "restroom", "amenity")


@dataclass(frozen=True)
class ZoneRule:
    kind: str
    label: str
    egress: bool
    occupied_level: float
    vacant_level: float
    vacant_bounds: Tuple[float, float]
    hold_min: int
    max_off_delay_min: Optional[int]
    min_in_use_fc: float
    dr_trim_allowed: bool
    tag: str
    basis: List[Dict[str, Any]] = field(default_factory=list)


@lru_cache(maxsize=4)
def load_rules(path: Optional[str] = None) -> Dict[str, Any]:
    return json.loads(Path(path or RULES_PATH).read_text(encoding="utf-8"))


def life_safety(key: str) -> Any:
    for row in load_rules()["life_safety"]:
        if row["key"] == key:
            return row["value"]
    raise KeyError(key)


def zone_rule(kind: str) -> ZoneRule:
    z = load_rules()["zone_types"][kind]
    return ZoneRule(kind=kind, label=z["label"], egress=bool(z["egress"]), occupied_level=float(z["occupied_level"]),
                    vacant_level=float(z["vacant_level"]), vacant_bounds=tuple(z["vacant_level_bounds"]),
                    hold_min=int(z["hold_min"]), max_off_delay_min=z["max_off_delay_min"],
                    min_in_use_fc=float(z["min_in_use_fc"]), dr_trim_allowed=bool(z["dr_trim_allowed"]), tag=z["tag"],
                    basis=list(z["basis"]))


def all_rules() -> Dict[str, ZoneRule]:
    return {k: zone_rule(k) for k in ZONE_KINDS}


def source_url(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    s = load_rules()["sources"].get(key)
    return s["url"] if s else None


def rules_table() -> List[Dict[str, Any]]:
    """Flat rows for display: one per zone type plus the life-safety rows."""
    rows = []
    for row in load_rules()["life_safety"]:
        rows.append({"scope": "life safety", "rule": row["key"], "value": row["value"], "unit": row["unit"],
                     "meaning": row["meaning"], "cite": row["cite"], "tag": row["tag"], "url": source_url(row["source"])})
    for k, r in all_rules().items():
        vac = f"{r.vacant_level:g} of design (allowed {r.vacant_bounds[0]:g}-{r.vacant_bounds[1]:g})"
        rows.append({"scope": r.label, "rule": "vacant level / hold", "value": f"{vac}; hold {r.hold_min} min",
                     "unit": "", "meaning": "; ".join(b["cite"] for b in r.basis), "cite": "egress" if r.egress else "non-egress",
                     "tag": r.tag, "url": source_url(next((b["source"] for b in r.basis if b.get("source")), None))})
    return rows
