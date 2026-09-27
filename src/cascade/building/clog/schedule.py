"""Risk-based periodic-check scheduler for plumbing clog prevention (transparent rules, no ML).

Order of precedence for each asset:
  1. a regulatory maximum where one exists (backflow: last test + 365 d, LADWP Rule 16-D; grease interceptor:
     before FOG + solids reach 25% of liquid depth, LASAN SSMP 7.5.3; grease traps: daily),
  2. an event trigger (backup in the last 48 h -> camera within 48 h; persistent nightly-test alarm -> 7 days),
  3. otherwise a team-proposed base interval divided by a risk multiplier
     (consequence = floors served, history = 1 + backups in the last 12 months, test result = drain-down grade).
Every row is a PROPOSAL: it names who performs the check and requires human approval. Nothing is actuated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import rules as R

PERFORMER = {
    "backflow": "Certified Backflow Prevention Assembly Tester licensed by LA County Department of Public Health",
    "grease_interceptor": "Licensed grease hauler, logged in the FSE cleaning logbook",
    "grease_trap": "Kitchen staff (daily clean and visual inspection)",
    "prv_strainer": "Building engineer (isolate the PRV station, open and clean the strainer)",
    "stack_base": "Drain contractor (CCTV camera, jetting if needed)",
    "building_drain": "Drain contractor (CCTV camera, jetting if needed)",
}
MIN_MULT, MAX_MULT = 0.5, 4.0
FOG_MARGIN_DAYS = 3  # team-proposed safety margin before the projected 25% date


@dataclass
class Asset:
    asset_id: str
    kind: str
    name: str
    last_check: str  # ISO date of the last test / cleaning / camera inspection
    floors_served: int = 0
    backups_12m: int = 0
    last_backup: Optional[str] = None
    fse_permit: bool = False
    fog_readings: List[Tuple[str, float]] = field(default_factory=list)  # (ISO date, % of liquid depth)
    nightly_alarm_persistent: bool = False
    nightly_alarm_nights: Optional[int] = None  # nights the nightly test alarmed in the reading window
    nightly_nights: Optional[int] = None  # nights in that window
    extra_headloss_m: Optional[float] = None
    drain_down_ratio: Optional[float] = None
    drain_down_censored: bool = False
    provenance: str = "SYNTHETIC"


def _d(s: str) -> date:
    return date.fromisoformat(s)


def fog_projection(readings: List[Tuple[str, float]], limit_pct: float) -> Tuple[Optional[date], Optional[float]]:
    """Least-squares line through (day, %) readings -> date the line reaches limit_pct, and the slope %/day."""
    if len(readings) < 2:
        return None, None
    t0 = _d(readings[0][0])
    x = np.array([(_d(d) - t0).days for d, _ in readings], float)
    y = np.array([p for _, p in readings], float)
    slope, icpt = np.polyfit(x, y, 1)
    if slope <= 0:
        return None, float(slope)
    return t0 + timedelta(days=float(np.ceil((limit_pct - icpt) / slope))), float(slope)


def risk_multiplier(a: Asset) -> Tuple[float, List[str]]:
    parts, why = 1.0, []
    if a.floors_served:
        c = min(max(a.floors_served / 11.0, 0.5), 3.0)  # 11 floors = one pressure zone
        parts *= c
        why.append(f"consequence x{c:.2f} ({a.floors_served} floors drain or feed through it)")
    if a.backups_12m:
        parts *= 1 + a.backups_12m
        why.append(f"history x{1 + a.backups_12m} ({a.backups_12m} backup(s) in 12 months)")
    if a.drain_down_ratio is not None:
        bands = R.value("DRAIN_RATIO_BANDS")
        if a.drain_down_censored or a.drain_down_ratio >= bands["urgent"]:
            parts *= MAX_MULT
            why.append("nightly drain-down test: urgent band")
        elif a.drain_down_ratio >= bands["watch"]:
            parts *= 2.0
            why.append(f"nightly drain-down test ratio {a.drain_down_ratio:.2f} (watch band)")
    return float(min(max(parts, MIN_MULT), MAX_MULT)), why


def plan(a: Asset, as_of: date) -> Dict[str, object]:
    last = _d(a.last_check)
    rule_keys: List[str] = []
    reason = ""
    if a.kind == "backflow":
        due = last + timedelta(days=R.value("BACKFLOW_ANNUAL"))
        rule_keys = ["BACKFLOW_ANNUAL"]
        reason = f"Annual test: last tested {a.last_check}; regulatory maximum 365 days."
    elif a.kind == "grease_trap":
        due = as_of + timedelta(days=R.value("GREASE_TRAP_DAILY"))
        rule_keys = ["GREASE_TRAP_DAILY"]
        reason = "Grease traps are cleaned and visually inspected daily."
    elif a.kind == "grease_interceptor":
        limit = float(R.value("FOG_25PCT"))
        rule_keys = ["FOG_25PCT", "FOG_PROGRAM"]
        if not a.fse_permit:
            due = last + timedelta(days=365)
            reason = "No food service establishment permit on record: 25% rule not triggered; yearly check proposed."
        else:
            latest = a.fog_readings[-1][1] if a.fog_readings else None
            proj, slope = fog_projection(a.fog_readings, limit)
            if latest is not None and latest >= limit:
                due = as_of
                reason = f"Latest FOG + solids reading {latest:.0f}% is at or above the {limit:.0f}% limit: pump out now."
            elif proj is not None:
                due = proj - timedelta(days=FOG_MARGIN_DAYS)
                reason = (f"FOG + solids {latest:.0f}% rising {slope * 7:.1f} points/week; projected to reach "
                          f"{limit:.0f}% on {proj.isoformat()}; pump out {FOG_MARGIN_DAYS} days before.")
            else:
                due = last + timedelta(days=90)
                reason = "Too few probe readings to project; 90-day pump-out proposed until readings exist."
    elif a.kind == "prv_strainer":
        rule_keys = ["UPC_608_2_PRV_80PSI", "STRAINER_BASE_INTERVAL", "PERSISTENCE_2_OF_3"]
        base = last + timedelta(days=R.value("STRAINER_BASE_INTERVAL"))
        k, n = R.value("PERSISTENCE_2_OF_3")
        seen = (f"alarmed on {a.nightly_alarm_nights} of {a.nightly_nights} nights"
                if a.nightly_alarm_nights is not None and a.nightly_nights else None)
        if a.nightly_alarm_persistent:
            due = min(base, as_of + timedelta(days=7))
            xh = f" (extra head loss about {a.extra_headloss_m:.2f} m at 4 L/s)" if a.extra_headloss_m is not None else ""
            reason = (f"Nightly flow test {seen or 'raised a persistent alarm'} on this strainer, meeting the work-order "
                      f"rule of at least {k} alarms in {n} consecutive nights{xh}: inspect within 7 days.")
        else:
            due = base
            reason = (f"No persistent nightly-test alarm" + (f" ({seen})" if seen else "") +
                      f"; base interval {R.value('STRAINER_BASE_INTERVAL')} days from {a.last_check}.")
    elif a.kind in ("stack_base", "building_drain"):
        rule_keys = ["CAMERA_BASE_INTERVAL", "PERFORMANCE_HISTORY"]
        if a.last_backup and (as_of - _d(a.last_backup)).days <= 2:
            due = _d(a.last_backup) + timedelta(days=2)
            rule_keys.append("CCTV_48H_AFTER_OVERFLOW")
            reason = f"Backup on {a.last_backup}: camera inspection within 48 hours."
        else:
            m, why = risk_multiplier(a)
            days = int(round(R.value("CAMERA_BASE_INTERVAL") / m))
            due = last + timedelta(days=days)
            if a.drain_down_ratio is not None:
                rule_keys.append("DRAIN_RATIO_BANDS")
            reason = f"Base {R.value('CAMERA_BASE_INTERVAL')} d / risk x{m:.2f} = {days} d" + \
                     (f" ({'; '.join(why)})" if why else "")
            if a.drain_down_censored or (a.drain_down_ratio or 0) >= R.value("DRAIN_RATIO_BANDS")["urgent"]:
                due = min(due, as_of + timedelta(days=2))
                reason += "; urgent drain-down result: within 2 days."
    else:
        raise ValueError(a.kind)
    days_left = (due - as_of).days
    status = "overdue" if days_left < 0 else ("due within 7 days" if days_left <= 7 else
                                               ("due within 30 days" if days_left <= 30 else "scheduled"))
    return {
        "asset_id": a.asset_id, "asset": a.name, "kind": a.kind, "last_check": a.last_check,
        "next_check": due.isoformat(), "days_from_as_of": days_left, "status": status, "reason": reason,
        "performed_by": PERFORMER[a.kind], "rule_keys": rule_keys,
        "rule_tags": [R.rule(k)["tag"] for k in rule_keys],
        "rule_urls": [R.rule(k)["url"] for k in rule_keys if R.rule(k)["url"]],
        "requires_human_approval": True, "input_provenance": a.provenance,
    }


def schedule(assets: List[Asset], as_of: date) -> List[Dict[str, object]]:
    rows = [plan(a, as_of) for a in assets]
    return sorted(rows, key=lambda r: r["next_check"])
