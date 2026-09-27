"""Deterministic, code-bounded lighting policy for common areas, plus a pending-approval Proposal log.

level(t) = max(code floor, target(t)), where target is occupied_level while the zone counts as occupied and the
zone's vacant_level otherwise. "Occupied" comes from the motion sensor with a fail-safe hold of at least
EGRESS_MOTION_HOLD_MIN on egress routes. ML may only ADD on-time (extend a hold or switch on early): the sensor+ML
policy is elementwise >= the sensor-only policy. A sensor fault sends the zone to full output. DR trims apply only to
non-egress zones and never go below the floor. ML-only switching is provided solely as an unsafe counterfactual for
evaluation; it is never offered as a deployable mode.

Nothing here actuates equipment. Policies are evaluated in simulation; changes to a real building are Proposals with
status pending_approval that a human approves or rejects.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .rules import ZoneRule, life_safety, load_rules

MODES = ("always_on", "schedule", "sensor", "sensor_ml", "sensor_ml_union", "ml_only")
DEPLOYABLE_MODES = ("always_on", "schedule", "sensor", "sensor_ml", "sensor_ml_union")
EPS = 1e-9


def hold_steps(rule: ZoneRule, step_min: float) -> int:
    """Number of steps the zone stays on after the step with the last detection (>= rule.hold_min minutes)."""
    h = rule.hold_min
    if rule.egress:
        h = max(h, int(life_safety("EGRESS_MOTION_HOLD_MIN")))
    if rule.max_off_delay_min is not None and not rule.egress:
        h = min(h, int(rule.max_off_delay_min))
    return int(math.ceil(h / step_min))


def hold_on(sig: np.ndarray, steps: int) -> np.ndarray:
    """True at t if sig was True at any of t-steps..t (vectorised trailing hold)."""
    sig = np.asarray(sig, dtype=bool)
    n = len(sig)
    t = np.arange(n)
    last = np.where(sig, t, -(10 ** 12))
    last = np.maximum.accumulate(last) if n else last
    return (t - last) <= steps


def extend_only(sensor_on: np.ndarray, ml_on: np.ndarray) -> np.ndarray:
    """ML may keep ON a zone that is already ON (the sensor hold was active when the ML run started or inside it);
    it can never switch a vacant-level zone on by itself and never switch anything off."""
    s = np.asarray(sensor_on, dtype=bool)
    m = np.asarray(ml_on, dtype=bool)
    if not len(s):
        return s
    run_start = m & ~np.r_[False, m[:-1]]
    seed = (s | (run_start & np.r_[False, s[:-1]])) & m
    c = np.cumsum(seed)
    starts = np.flatnonzero(run_start)
    before = np.r_[0, c][starts]  # seed count before each run starts
    run_no = np.cumsum(run_start) - 1
    cum = np.zeros(len(s), dtype=bool)
    cum[m] = (c[m] - before[run_no[m]]) > 0  # any seed so far inside this ML run
    return s | (m & cum)


def floor_level(rule: ZoneRule, design_fc: float) -> float:
    """Minimum fraction of design output while the building is occupied (egress 1 fc floor)."""
    if not rule.egress:
        return 0.0
    return min(1.0, float(life_safety("EGRESS_FLOOR_FC")) / float(design_fc))


def check_design(rule: ZoneRule, design_fc: float) -> None:
    """The occupied level must meet the in-use floor (stairs: 10 fc)."""
    if rule.occupied_level * design_fc + EPS < rule.min_in_use_fc:
        raise ValueError(f"{rule.kind}: design {design_fc} fc x occupied level {rule.occupied_level} is below the in-use "
                         f"floor {rule.min_in_use_fc} fc")


def lighting_levels(rule: ZoneRule, *, design_fc: float, step_min: float = 5.0, mode: str = "sensor",
                    detect: Optional[np.ndarray] = None, ml_on: Optional[np.ndarray] = None,
                    schedule_on: Optional[np.ndarray] = None, fault: Optional[np.ndarray] = None,
                    dr: Optional[np.ndarray] = None, dr_frac: Optional[float] = None, n: Optional[int] = None) -> np.ndarray:
    """Per-step lighting level (fraction of design output) for one zone."""
    if mode not in MODES:
        raise ValueError(mode)
    check_design(rule, design_fc)
    size = n if n is not None else len(next(a for a in (detect, ml_on, schedule_on) if a is not None))
    k = hold_steps(rule, step_min)
    if mode == "always_on":
        on = np.ones(size, dtype=bool)
    elif mode == "schedule":
        on = np.asarray(schedule_on, dtype=bool)
    elif mode == "sensor":
        on = hold_on(detect, k)
    elif mode == "sensor_ml":  # recommended: ML may only lengthen an active hold
        on = extend_only(hold_on(detect, k), ml_on)
    elif mode == "sensor_ml_union":  # ML may also switch on early (costs more energy; reported for transparency)
        on = hold_on(np.asarray(detect, dtype=bool) | np.asarray(ml_on, dtype=bool), k)
    else:  # ml_only: unsafe counterfactual, evaluation only
        on = hold_on(ml_on, k)
    fl = floor_level(rule, design_fc)
    level = np.where(on, rule.occupied_level, rule.vacant_level)
    level = np.maximum(level, fl)
    if dr is not None and rule.dr_trim_allowed:
        frac = float(dr_frac if dr_frac is not None else load_rules()["demand_response"]["min_reduction_frac"])
        level = np.where(np.asarray(dr, dtype=bool), np.maximum(level * (1.0 - frac), fl), level)
    if fault is not None:
        level = np.where(np.asarray(fault, dtype=bool), 1.0, level)  # fail-safe: full ON
    return level


def safety_counts(rule: ZoneRule, level: np.ndarray, occupied: np.ndarray, design_fc: float) -> Dict[str, int]:
    """Counts of steps below each floor. life_safety_violations must always be 0 for deployable modes."""
    occupied = np.asarray(occupied, dtype=bool)
    fc = level * design_fc
    below_egress = int(((fc + EPS) < float(life_safety("EGRESS_FLOOR_FC"))).sum()) if rule.egress else 0
    below_in_use = int((occupied & ((fc + EPS) < rule.min_in_use_fc)).sum()) if rule.min_in_use_fc > 0 else 0
    underlit = int((occupied & (level + EPS < rule.occupied_level)).sum())
    return {"life_safety_violations": below_egress, "in_use_floor_violations": below_in_use, "underlit_occupied_steps": underlit}


def switch_count(level: np.ndarray) -> int:
    level = np.asarray(level)
    return int((np.abs(np.diff(level)) > EPS).sum()) if len(level) > 1 else 0


# --------------------------------------------------------------------------------------------------- proposals
NEVER_WRITE = tuple(load_rules()["never_write"])
STATUSES = ("pending_approval", "approved", "rejected")


@dataclass
class Proposal:
    zone: str
    action: str
    reason: str
    citations: List[str]
    expected_kwh_per_year: Optional[float] = None
    expected_usd_per_year: Optional[float] = None
    data_label: str = "SEMI-SYNTHETIC"
    status: str = "pending_approval"
    id: str = field(default_factory=lambda: "EP-" + uuid.uuid4().hex[:8])
    created_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    reviewer: Optional[str] = None
    reviewed_ts: Optional[str] = None
    note: str = ""


def propose_setting(rule: ZoneRule, *, zone: str, vacant_level: Optional[float] = None, hold_min: Optional[int] = None,
                    design_fc: float, reason: str, **kw) -> Proposal:
    """Build a Proposal after checking it against the code bounds; raises ValueError if it would breach one."""
    cites = [b["cite"] for b in rule.basis]
    if any(w.lower() in zone.lower() for w in NEVER_WRITE):
        raise ValueError(f"{zone}: life-safety circuits are outside Cerebro's control")
    parts = []
    if vacant_level is not None:
        lo, hi = rule.vacant_bounds
        if not (lo - EPS <= vacant_level <= hi + EPS):
            raise ValueError(f"{rule.kind}: vacant level {vacant_level} outside allowed {lo}-{hi}")
        if vacant_level + EPS < floor_level(rule, design_fc):
            raise ValueError(f"{rule.kind}: vacant level {vacant_level} gives < {life_safety('EGRESS_FLOOR_FC')} fc egress floor")
        parts.append(f"vacant level {vacant_level:g}")
    if hold_min is not None:
        if rule.egress and hold_min < int(life_safety("EGRESS_MOTION_HOLD_MIN")):
            raise ValueError(f"{rule.kind}: egress hold {hold_min} min < {life_safety('EGRESS_MOTION_HOLD_MIN')} min (NFPA 101 7.8.1.2.2)")
        if rule.max_off_delay_min is not None and not rule.egress and hold_min > rule.max_off_delay_min:
            raise ValueError(f"{rule.kind}: hold {hold_min} min > {rule.max_off_delay_min} min maximum off delay")
        parts.append(f"hold {hold_min} min")
    action = kw.pop("action", None) or ("set " + ", ".join(parts) if parts else "review")
    return Proposal(zone=zone, action=action, reason=reason, citations=cites, **kw)


class ProposalLog:
    """Append-only JSONL of proposal events. Only pending_approval -> approved | rejected, by a named human."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _events(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def add(self, p: Proposal) -> Proposal:
        if p.status != "pending_approval":
            raise ValueError("new proposals must be pending_approval")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "add", **asdict(p)}) + "\n")
        return p

    def current(self) -> Dict[str, dict]:
        cur: Dict[str, dict] = {}
        for e in self._events():
            if e["event"] == "add":
                cur[e["id"]] = {k: v for k, v in e.items() if k != "event"}
            elif e["event"] == "transition" and e["id"] in cur:
                cur[e["id"]].update(status=e["to"], reviewer=e["reviewer"], reviewed_ts=e["ts"], note=e.get("note", ""))
        return cur

    def transition(self, pid: str, to: str, reviewer: str, note: str = "") -> dict:
        cur = self.current()
        if pid not in cur:
            raise KeyError(pid)
        if to not in ("approved", "rejected"):
            raise ValueError(to)
        if cur[pid]["status"] != "pending_approval":
            raise ValueError(f"{pid} is already {cur[pid]['status']}")
        if not reviewer or not reviewer.strip():
            raise ValueError("a named human reviewer is required")
        ev = {"event": "transition", "id": pid, "to": to, "reviewer": reviewer.strip(), "note": note,
              "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        return self.current()[pid]
