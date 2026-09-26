"""Problem manager: group observations into problems, rank root causes, estimate escalation, keep a lifecycle
(building_spec section 10).

- cluster: water, water-ticket and facade observations join when they share an upstream water source that is a
  strong explanation for both (SOURCE_SHARE_MIN of each node's best source, joint reach >= REACH_MIN) within
  PROBLEM_WINDOW_H (chained); electrical observations join per element (circuit) in time, and up to the panel only
  when several circuits under it show the same signal in the window. S0 is context only.
- problem_id = P-{root node}: stable when the domain changes (water to mixed); ProblemStore carries status over to a
  re-keyed successor and marks rows the latest run did not detect as inactive (history, not open).
- rank_root_causes: named additive factors (ROOT_WEIGHTS). A factor that could not be measured has value None and
  contributes 0; it is never read as healthy.
- priority: SEVERITY_WEIGHT[level] * (2 with fire_shock_pathway) * URGENCY(hours). U is listed, never scored.
- ProblemStore: problems.json with status history; recurrence reopens a resolved problem; rows are never deleted.
No model API is called here. Analysis never reads truth.json.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..pipeline import load_run, save_findings
from ..prioritize import SEVERITY_WEIGHT
from ..schema import Finding, ImageRecord, Level
from .blueprint import FILES, TS_FORMAT, load_building_dir, ticket_category, tickets_to_observations
from .graph import downstream, electrical_parent, path_to_root, upstream, water_sources, zone_of
from .model import (ELECTRICAL_KINDS, Building, BuildingData, ElectricalResult, Escalation, Factor, Observation, Problem,
                    ProblemStatus, ResponseFit, RootCause, StatusChange, WaterResult)
from .rules import THRESHOLDS, t

ALLOWED: Dict[str, set] = {"detected": {"triaged"}, "triaged": {"work_order", "resolved"}, "work_order": {"resolved"},
                           "resolved": {"verified", "detected"}, "verified": set()}
LEVEL_RANK: Dict[str, int] = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}
VALID_FLAGS = ("fire_shock_pathway", "load_posting_review", "section_loss", "not_measurable")
WET_VALUES = ("Active leak", "Drying window exceeded")
LOAD_VALUES = ("Load continuous over 80%", "Load over 100%")


# ----------------------------------------------------------------------------------------- small helpers

def parse_ts(ts: str) -> datetime:
    d = pd.Timestamp(ts)
    if d.tzinfo is None:
        d = d.tz_localize("UTC")
    return d.tz_convert("UTC").to_pydatetime()


def iso(d: datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).strftime(TS_FORMAT)


def _hours_between(a: str, b: str) -> float:
    return abs((parse_ts(b) - parse_ts(a)).total_seconds()) / 3600.0


def node_of(o: Observation) -> str:
    return o.element_id or o.zone_id or ""


def _native(o: Observation) -> Optional[str]:
    return o.finding.native_scale.value if o.finding is not None else None


def obs_domain(b: Building, o: Observation) -> str:
    """water, facade, electrical or other."""
    if o.kind == "ticket":
        return o.category or ticket_category(o.text)
    if o.kind in ("thermal_reading", "load_event"):
        return "electrical"
    if o.finding is not None and o.finding.asset_class == "electrical_equipment":
        return "electrical"
    if o.element_id:
        e = b.element(o.element_id)
        if e is not None and e.kind in ELECTRICAL_KINDS:
            return "electrical"
    if o.finding is not None and o.finding.asset_class == "facade_element":
        return "facade"
    z = b.zone(o.zone_id) if o.zone_id else None
    if o.kind == "image_finding" and z is not None and z.kind == "facade_drop":
        return "facade"
    return "water"


def _sensor_type(b: Building, o: Observation) -> Optional[str]:
    if o.source.startswith("sensor:"):
        s = b.sensor(o.source.split(":", 1)[1])
        return s.type if s is not None else None
    return None


def _is_wet(b: Building, o: Observation) -> bool:
    """Wet now: a measured wet reading whose condition has not ended. U (unmeasured), an ended wet run (end_ts) and an
    undated photo are never wet for a clock."""
    if o.level == "U" or o.end_ts is not None or o.ts_unknown:
        return False
    return _native(o) in WET_VALUES or (_sensor_type(b, o) == "leak" and o.value is not None and o.value >= 1)


def _worst(levels: Iterable[Optional[str]]) -> Optional[str]:
    graded = [lv for lv in levels if lv in LEVEL_RANK]
    return max(graded, key=lambda lv: LEVEL_RANK[lv]) if graded else None


def _best_source(b: Building, node: str) -> float:
    r = _water_sources_reaching(b, node)
    return max(r.values()) if r else 0.0


def _water_sources_reaching(b: Building, node: str) -> Dict[str, float]:
    """Upstream water sources of a node with reach >= REACH_MIN (the node itself counts when it is a source)."""
    def build() -> Dict[str, float]:
        srcs = set(water_sources(b))
        ups = upstream(b, node)
        return {n: r.score for n, r in ups.items() if n in srcs and r.score >= t("REACH_MIN")}
    return b.memo(f"problems.sources:{node}", build)


def _panel_key(b: Building, o: Observation) -> Optional[str]:
    """The electrical panel (or switchboard) whose subtree holds the observation's node."""
    node = o.element_id
    if node is None and o.zone_id:  # an electrical ticket in a room: the panel feeding a load in that zone
        loads = [e.element_id for e in b.elements_in(o.zone_id) if e.kind == "load"]
        node = loads[0] if loads else None
        if node is None:
            panels = [e.element_id for e in b.elements_in(o.zone_id) if e.kind in ("electrical_panel", "switchboard")]
            node = panels[0] if panels else None
    if node is None:
        return None
    for n in path_to_root(b, node):
        e = b.element(n)
        if e is not None and e.kind == "electrical_panel":
            return n
    chain = path_to_root(b, node)
    return chain[-1] if chain else None


# ----------------------------------------------------------------------------------------- clustering

class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _zone_driver(b: Building, o: Observation, fits: Optional[Dict[str, ResponseFit]]) -> Optional[str]:
    """The significant rain driver of the observation's zone, or None when unmeasured or not significant."""
    if not fits:
        return None
    z = zone_of(b, node_of(o)) if node_of(o) else None
    f = _fit_for(fits, z) if z else None
    return f.driver if f is not None and f.significant else None


def cluster(b: Building, obs: Sequence[Observation], window_h: Optional[float] = None,
            fits: Optional[Dict[str, ResponseFit]] = None) -> List[List[Observation]]:
    """Groups of related observations, each sorted by time, groups ordered by first time. S0 and 'other' tickets
    are context only and never start or join a group.

    Two water observations join through a shared upstream source only when that source is a strong explanation for
    both (its path score is at least SOURCE_SHARE_MIN of each node's best source score), the product of the two path
    scores clears REACH_MIN, and their times are within the window (chained). So one storm that wets many rooms
    does not chain an off-stack room into the roof problem. With `fits`, observations whose zones have different
    significant rain drivers (say wdr_E and rain) never join: they are two mechanisms. A zone with no significant
    driver joins a driven one only through its own best source.

    Electrical observations chain per element (circuit or panel) in time. They join across circuits of one panel
    only when they show the same signal (kind and value, say two Band 3 scans) within the window, a panel-level
    cause. An electrical ticket joins what is under its panel in the window. Undated photos (ts_unknown) never
    chain in time."""
    window = float(t("PROBLEM_WINDOW_H") if window_h is None else window_h)
    active = [o for o in obs if o.level != "S0" and obs_domain(b, o) != "other" and (o.zone_id or o.element_id)]
    drivers = [_zone_driver(b, o, fits) for o in active]
    uf = _UF(len(active))
    by_source: Dict[str, List[Tuple[datetime, int]]] = {}
    by_elem: Dict[str, List[Tuple[datetime, int]]] = {}
    elec: List[Tuple[datetime, int, Optional[str], str]] = []  # (ts, index, panel, element key)
    for i, o in enumerate(active):
        if o.ts_unknown:
            continue
        dom = obs_domain(b, o)
        if dom == "electrical":
            key = o.element_id or _panel_key(b, o)
            if key is not None:
                by_elem.setdefault(key, []).append((parse_ts(o.ts), i))
                elec.append((parse_ts(o.ts), i, _panel_key(b, o), key))
            continue
        node = node_of(o)
        if b.zone(node) is None and b.element(node) is None:
            continue
        for src in _water_sources_reaching(b, node):
            by_source.setdefault(src, []).append((parse_ts(o.ts), i))
    reach_min = float(t("REACH_MIN"))
    share = float(t("SOURCE_SHARE_MIN"))
    for src, items in by_source.items():
        items.sort()
        last: Dict[Optional[str], Tuple[datetime, int]] = {}  # most recent observation per driver on this source
        for ts, i in items:
            d = drivers[i]
            ni = node_of(active[i])
            si = _water_sources_reaching(b, ni)[src]
            strong_i = si >= share * _best_source(b, ni)
            for dk, (tp, ip) in last.items():
                np_ = node_of(active[ip])
                sp = _water_sources_reaching(b, np_)[src]
                strong_p = sp >= share * _best_source(b, np_)
                if d is not None and dk is not None:
                    compatible = d == dk
                elif d is None and dk is None:
                    compatible = True
                else:  # an undriven zone joins a driven one only when this source is its own best explanation
                    compatible = si >= _best_source(b, ni) if d is None else sp >= _best_source(b, np_)
                if (compatible and strong_i and strong_p and si * sp >= reach_min
                        and (ts - tp).total_seconds() / 3600.0 <= window):
                    uf.union(ip, i)
            last[d] = (ts, i)
    for items in by_elem.values():  # one element: chain in time
        items.sort()
        for (ta, ia), (tb, ib) in zip(items, items[1:]):
            if (tb - ta).total_seconds() / 3600.0 <= window:
                uf.union(ia, ib)
    elec.sort(key=lambda x: (x[0], x[1]))
    for a, (ta, ia, pa, ka) in enumerate(elec):  # one panel: same signal on several circuits, or a ticket
        for tb, ib, pb, kb in elec[a + 1:]:
            if (tb - ta).total_seconds() / 3600.0 > window:
                break
            if pa is None or pa != pb or ka == kb:
                continue
            oa, ob = active[ia], active[ib]
            if oa.kind == "ticket" or ob.kind == "ticket" or (oa.kind == ob.kind and _native(oa) == _native(ob) and oa.level == ob.level):
                uf.union(ia, ib)
    groups: Dict[int, List[Observation]] = {}
    for i, o in enumerate(active):
        groups.setdefault(uf.find(i), []).append(o)
    out = [sorted(g, key=lambda o: (o.ts, o.obs_id)) for g in groups.values()]
    out.sort(key=lambda g: (g[0].ts, g[0].obs_id))
    return out


# ----------------------------------------------------------------------------------------- root causes

def _fit_for(fits: Dict[str, ResponseFit], zone_id: str) -> Optional[ResponseFit]:
    if zone_id in fits:
        return fits[zone_id]
    for f in fits.values():
        if f.zone_id == zone_id:
            return f
    return None


def _own_zones(b: Building, cand: str) -> List[str]:
    """Sensored zones that show the candidate's own condition: its zone when it has sensors, else the zones it drains
    into or sits above (one hop)."""
    z = zone_of(b, cand)
    if z is not None and b.sensors_in(z):
        return [z]
    out = []
    for e in b.edges:
        if e.src == cand and e.kind in ("drains_to", "above") and b.zone(e.dst) is not None and b.sensors_in(e.dst):
            out.append(e.dst)
    return out


def _root_label(b: Building, cand: str) -> str:
    z = b.zone(cand)
    if z is not None:
        if z.kind == "facade_drop":
            return f"facade water entry at {cand} ({z.orientation}-facing drop) suspected"
        if z.kind == "roof":
            drains = [e.element_id for e in b.elements_in(cand) if e.kind == "roof_drain"]
            if len(drains) == 1:
                return f"blocked roof drain {drains[0]} suspected (only outlet of {cand})"
            return f"roof leak or blocked drains at {cand} suspected ({', '.join(drains) or 'no drain on record'})"
        return f"water entry at {cand} suspected"
    e = b.element(cand)
    if e is not None:
        if e.kind == "riser":
            return f"{e.attrs.get('system', 'water')} riser leak at {cand} suspected"
        if e.kind == "roof_drain":
            return f"blocked roof drain {cand} suspected"
    return f"source {cand} suspected"


def rank_root_causes(b: Building, group: Sequence[Observation], fits: Dict[str, ResponseFit],
                     all_obs: Sequence[Observation]) -> List[RootCause]:
    """Upstream water sources of the group's nodes, ranked by additive ROOT_WEIGHTS factors (highest first)."""
    w = t("ROOT_WEIGHTS")
    window = float(t("PROBLEM_WINDOW_H"))
    nodes = sorted({node_of(o) for o in group if node_of(o) and (b.zone(node_of(o)) or b.element(node_of(o)))})
    if not nodes:
        return []
    reach_by_node = {n: _water_sources_reaching(b, n) for n in nodes}
    cands = sorted(set().union(*[set(r) for r in reach_by_node.values()]))
    zones = sorted({zone_of(b, n) for n in nodes if zone_of(b, n)})
    zone_fits = [f for f in (_fit_for(fits, z) for z in zones) if f is not None]
    drivers = {f.driver for f in zone_fits if f.significant}
    leak_seen = any(_sensor_type(b, o) == "leak" for o in group)
    measured = bool(zone_fits) or leak_seen
    t0 = min(parse_ts(o.ts) for o in group) - timedelta(hours=window)
    t1 = max(parse_ts(o.ts) for o in group) + timedelta(hours=window)
    in_win = [o for o in all_obs if t0 <= parse_ts(o.ts) <= t1]
    event_zones = {zone_of(b, node_of(o)) for o in in_win if o.level != "S0" and node_of(o)}
    out: List[RootCause] = []
    for c in cands:
        hits = [reach_by_node[n][c] for n in nodes if c in reach_by_node[n]]
        coverage = len(hits) / len(nodes)
        path_strength = float(np.mean(hits)) if hits else 0.0
        z = b.zone(c)
        e = b.element(c)
        sig: Optional[float]
        if not measured:
            sig = None
        elif z is not None and z.kind == "facade_drop":
            sig = 1.0 if f"wdr_{z.orientation}" in drivers else 0.0
        elif (z is not None and z.kind == "roof") or (e is not None and e.kind == "roof_drain"):
            sig = 1.0 if "rain" in drivers else 0.0
        elif e is not None and e.kind == "riser":
            sig = 1.0 if not drivers else 0.0
        else:
            sig = 0.0
        own_levels = [o.level for o in in_win if node_of(o) == c and o.level in LEVEL_RANK]
        own = LEVEL_RANK[_worst(own_levels)] / 4.0 if own_levels else None
        own_z = _own_zones(b, c)
        quiet = (sum(1 for oz in own_z if oz not in event_zones) / len(own_z)) if own_z else None
        vals = {"coverage": coverage, "path_strength": path_strength, "signature": sig, "own_evidence": own, "own_zone_quiet": quiet}
        expl = {
            "coverage": f"reaches {len(hits)} of {len(nodes)} observed nodes (reach >= REACH_MIN {t('REACH_MIN')})",
            "path_strength": "mean best-path score to the observed nodes it reaches",
            "signature": ("not measured: no rain-response fit or leak sensor for these zones" if sig is None else
                          f"significant drivers {sorted(drivers) or ['none']} vs candidate kind"),
            "own_evidence": ("not measured: no graded observation on the candidate itself" if own is None else
                             f"worst level on the candidate {_worst(own_levels)} / 4"),
            "own_zone_quiet": ("not measured: no sensored zone right below the candidate" if quiet is None else
                               f"share of {own_z} with no event in the window (a quiet own zone argues against it)"),
        }
        factors = [Factor(name=k, value=v, weight=float(w[k]), contribution=(float(w[k]) * v if v is not None else 0.0),
                          explanation=expl[k], basis="ROOT_WEIGHTS [team-proposed, validate]") for k, v in vals.items()]
        score = round(sum(f.contribution for f in factors), 4)
        best = max(nodes, key=lambda n: reach_by_node[n].get(c, 0.0))
        out.append(RootCause(node_id=c, label=_root_label(b, c), score=score, factors=factors,
                             explanation=f"{c}: coverage {coverage:.2f}, path {path_strength:.2f}; strongest to {best}"))
    out.sort(key=lambda r: (-r.score, r.node_id))
    return out


def electrical_root(b: Building, group: Sequence[Observation]) -> RootCause:
    """The worst graded element; ties go to their deepest common ancestor in the feeds tree."""
    nodes = [o for o in group if o.element_id and b.element(o.element_id) is not None]
    worst = _worst(o.level for o in nodes)
    if worst is not None:
        tied = sorted({o.element_id for o in nodes if o.level == worst})
    else:
        tied = sorted({o.element_id for o in nodes}) or sorted({_panel_key(b, o) for o in group if _panel_key(b, o)})
    if not tied:
        node = node_of(group[0])
    elif len(tied) == 1:
        node = tied[0]
    else:
        chains = [list(reversed(path_to_root(b, n))) for n in tied]
        common = None
        for level_nodes in zip(*chains):
            if all(x == level_nodes[0] for x in level_nodes):
                common = level_nodes[0]
            else:
                break
        node = common or tied[0]
    kinds = {o.kind for o in group if o.element_id == node}
    values = {_native(o) for o in group if o.element_id == node}
    if "thermal_reading" in kinds:
        label = f"hot connection at {node} suspected (IR delta-T)"
    elif values & set(LOAD_VALUES) or "load_event" in kinds:
        label = f"overloaded circuit {node}"
    else:
        label = f"electrical fault at {node} suspected"
    val = LEVEL_RANK[worst] / 4.0 if worst else None
    f = Factor(name="worst_level", value=val, weight=1.0, contribution=val or 0.0,
               explanation=(f"worst graded level {worst} on {node}" if worst else "not measured: no graded electrical observation"),
               basis="electrical_thermal.json rows")
    return RootCause(node_id=node, label=label, score=round(val or 0.0, 4), factors=[f],
                     explanation=f"{len(tied)} element(s) at the worst level; root = " + ("the element" if len(tied) <= 1 else "their deepest common ancestor"))


# ----------------------------------------------------------------------------------------- escalation

def _rh_trend_clock(b: Building, data: BuildingData, zones: Sequence[str], now: datetime) -> Optional[Escalation]:
    thr = float(t("ASHRAE160_RH_30D"))
    best: Optional[Escalation] = None
    if data.sensors is None or data.sensors.empty:
        return None
    s_all = data.sensors[data.sensors.index <= pd.Timestamp(now)]
    for z in zones:
        for s in b.sensors_in(z):
            if s.type != "humidity" or s.sensor_id not in s_all.columns:
                continue
            ser = s_all[s.sensor_id].dropna()
            if ser.empty:
                continue
            daily = ser.resample("D").mean().dropna()
            recent = daily.tail(int(t("RH_TREND_DAYS")))
            if len(recent) < 3:
                continue
            slope = float(np.polyfit(np.arange(len(recent), dtype=float), recent.to_numpy(float), 1)[0])
            mean30 = float(daily.tail(int(t("ASHRAE160_WINDOW_D"))).mean())
            if slope <= 0 or mean30 >= thr:
                continue
            hours = (thr - mean30) / slope * 24.0
            e = Escalation(hours=round(hours, 1), lo_hours=None, hi_hours=None,
                           what=f"{s.sensor_id} 30-day mean {mean30:.1f} %RH reaches ASHRAE 160 80 %RH at the current trend",
                           basis="ASHRAE160_RH_30D, ASHRAE160_WINDOW_D, RH_TREND_DAYS")
            if best is None or (e.hours or 0) < (best.hours or 0):
                best = e
    return best


def _storm_clock(driver: str, water: WaterResult, now: datetime, label: str) -> Optional[Escalation]:
    starts: List[datetime] = []
    for st in water.storms:
        if driver == "rain":
            ok = True
        else:
            o = driver.split("_", 1)[1]
            vals = st.wdr or {}
            ok = o in vals and vals[o] > 0 and vals[o] >= max(vals.values())
        if ok:
            starts.append(parse_ts(st.start))
    starts = sorted(s for s in starts if s <= now)
    if len(starts) < 2:
        return None
    gaps = np.diff([s.timestamp() / 3600.0 for s in starts])
    q25, med, q75 = (float(np.percentile(gaps, q)) for q in (25, 50, 75))
    since = (now - starts[-1]).total_seconds() / 3600.0
    return Escalation(hours=round(max(0.0, med - since), 1), lo_hours=round(max(0.0, q25 - since), 1), hi_hours=round(max(0.0, q75 - since), 1),
                      what=f"next {label} storm expected (median gap {med / 24:.1f} d between {len(starts)} past matching storms, IQR band)",
                      basis="STORM_MIN_MM, STORM_GAP_H")


def estimate_escalation(p: Problem, group: Sequence[Observation], data: BuildingData, water: WaterResult,
                        now: datetime) -> Optional[Escalation]:
    """The earliest applicable clock: EPA drying window (wetness still on at `now`, from the start of the current
    wet run), next matching storm, ASHRAE 160 RH trend, thermal band, ongoing overload or SWARMP cycle. Every
    clock counts hours from `now` (as_of). A FISP Unsafe pre-classification gives a note with no running clock:
    the fix window runs from a filed QEWI report, not from a model grade. Undated photos drive no clock. None when
    nothing applies (unknown, not 'safe')."""
    b = data.building
    clocks: List[Escalation] = []
    lo_dry, hi_dry = t("EPA_DRY_WINDOW_H")
    wet = [o for o in group if _is_wet(b, o)]
    if wet:
        first = min(parse_ts(o.ts) for o in wet)  # each wet observation is stamped at the start of its current wetness
        elapsed = (now - first).total_seconds() / 3600.0
        what = (f"drying window exceeded (EPA 24-48 h): wet for {elapsed:.0f} h" if elapsed >= hi_dry
                else f"drying window closes (EPA 24-48 h): wet for {elapsed:.0f} h")
        clocks.append(Escalation(hours=round(max(0.0, hi_dry - elapsed), 1), lo_hours=round(max(0.0, lo_dry - elapsed), 1),
                                 hi_hours=round(max(0.0, hi_dry - elapsed), 1), what=what, basis="EPA_DRY_WINDOW_H"))
    root = p.root_cause.node_id
    z = b.zone(root)
    e = b.element(root)
    if z is not None and z.kind == "facade_drop":
        c = _storm_clock(f"wdr_{z.orientation}", water, now, f"{z.orientation}-driven")
        if c:
            clocks.append(c)
    elif (z is not None and z.kind == "roof") or (e is not None and e.kind == "roof_drain"):
        c = _storm_clock("rain", water, now, "rain")
        if c:
            clocks.append(c)
    zones = sorted({zone_of(b, node_of(o)) for o in group if node_of(o) and zone_of(b, node_of(o))})
    c = _rh_trend_clock(b, data, zones, now)
    if c:
        clocks.append(c)
    thermal = data.thermal
    if thermal is not None and len(thermal) and "ts" in thermal.columns:
        thermal = thermal[pd.to_datetime(thermal["ts"], utc=True) <= pd.Timestamp(now)]
    for o in group:
        v = _native(o)
        if o.ts_unknown and v not in ("SWARMP", "Unsafe"):
            continue  # an undated observation drives no time clock
        if v == "Band 3":
            clocks.append(Escalation(hours=0.0, lo_hours=0.0, hi_hours=0.0, what=f"Band 3 thermal reached on {node_of(o)} (top band)", basis="DT_BANDS_K"))
        elif o.kind == "thermal_reading" and o.element_id and v in ("Band 1", "Band 2"):
            from . import electrical as el
            tr = el.thermal_trend(thermal, o.element_id)
            delta = o.finding.measurements.delta_t_k if o.finding and o.finding.measurements.delta_t_k is not None else o.value
            if tr and delta is not None:
                slope, sd = tr
                win = el.next_band_window(delta, sd, slope)
                if win is not None:
                    d, lo, hi = win
                    clocks.append(Escalation(hours=round(d * 24, 1), lo_hours=round(lo * 24, 1), hi_hours=round(hi * 24, 1),
                                             what=f"{o.element_id} reaches the next delta-T band at {slope:.2f} K/day (band: delta-T +/- residual sd)",
                                             basis="DT_BANDS_K"))
        elif v in LOAD_VALUES:
            if o.value is not None and o.value >= float(t("LOAD_CONT_PCT")):
                clocks.append(Escalation(hours=0.0, lo_hours=0.0, hi_hours=0.0,
                                         what=f"overload ongoing on {node_of(o)} ({o.value:.0f} % over {t('CONTINUOUS_H'):g} h within the last {t('LOAD_RECENT_D')} days)",
                                         basis="LOAD_CONT_PCT, CONTINUOUS_H, LOAD_RECENT_D"))
            # else a past overload: no continuous mean at or above LOAD_CONT_PCT in LOAD_RECENT_D (or unknown), so no clock
        elif v == "Unsafe":
            clocks.append(Escalation(hours=None, lo_hours=None, hi_hours=None,
                                     what=(f"pre-classification only: if a QEWI confirms Unsafe, the fix window "
                                           f"({t('FISP_UNSAFE_FIX_DAYS')} days [team-proposed, validate]) runs from the report filing; "
                                           "no filing date is recorded, so no clock runs"),
                                     basis="FISP_UNSAFE_FIX_DAYS"))
        elif v == "SWARMP" and b.fisp_cycle_due:
            due = parse_ts(b.fisp_cycle_due + "T23:59:59Z" if len(b.fisp_cycle_due) == 10 else b.fisp_cycle_due)
            clocks.append(Escalation(hours=round(max(0.0, (due - now).total_seconds() / 3600.0), 1), lo_hours=None, hi_hours=None,
                                     what="SWARMP becomes Unsafe at the next cycle if not corrected (1 RCNY 103-04)", basis="FISP_CYCLE_YEARS"))
    as_of = iso(now)
    timed = [c for c in clocks if c.hours is not None]
    if timed:
        return min(timed, key=lambda c: c.hours).model_copy(update={"as_of": as_of})
    return clocks[0].model_copy(update={"as_of": as_of}) if clocks else None


def urgency(hours: Optional[float]) -> float:
    u = t("URGENCY")
    if hours is None:
        return float(u["min"])
    return float(min(u["max"], max(u["min"], 1.0 + u["numerator_h"] / max(hours, u["floor_h"]))))


def priority(level: Level, flags: Sequence[str], escalation: Optional[Escalation]) -> Optional[float]:
    """SEVERITY_WEIGHT[level] * (2 with fire_shock_pathway) * URGENCY(hours). None for U."""
    if level == "U":
        return None
    mult = 2.0 if "fire_shock_pathway" in flags else 1.0
    return round(SEVERITY_WEIGHT[level] * mult * urgency(escalation.hours if escalation else None), 4)


def sort_problems(problems: Sequence[Problem]) -> List[Problem]:
    """Active before inactive (not re-detected); S4 first, then by priority (highest first); U last and unscored."""
    return sorted(problems, key=lambda p: (0 if p.active else 1, 0 if p.level == "S4" else 1, 1 if p.level == "U" else 0,
                                           -(p.priority or 0.0), p.problem_id))


def level_counts(problems: Sequence[Problem]) -> Dict[str, int]:
    """Counts per level; U is its own key and never added to S0."""
    out = {lv: 0 for lv in ("S0", "S1", "S2", "S3", "S4", "U")}
    for p in problems:
        out[p.level] += 1
    return out


# ----------------------------------------------------------------------------------------- building problems

def _fire_shock(b: Building, root: str, group: Sequence[Observation]) -> bool:
    hops = int(t("WATER_NEAR_PANEL_HOPS"))
    for o in group:
        z = b.zone(zone_of(b, node_of(o)) or "")
        if z is not None and z.kind == "electrical_room":
            return True
    if b.zone(root) is None and b.element(root) is None:
        return False
    for n in downstream(b, root, max_hops=hops):
        z = b.zone(n)
        if z is not None and z.kind == "electrical_room":
            return True
        e = b.element(n)
        if e is not None and e.kind in ("electrical_panel", "switchboard"):
            return True
    return False


def _make_problem(b: Building, group: List[Observation], data: BuildingData, water: WaterResult, all_obs: Sequence[Observation],
                  now: datetime) -> Problem:
    doms = {obs_domain(b, o) for o in group}
    water_group = [o for o in group if obs_domain(b, o) in ("water", "facade")]
    alternatives: List[RootCause] = []
    if water_group:
        ranked = rank_root_causes(b, water_group, water.fits, all_obs)
        if ranked:
            root, alternatives = ranked[0], ranked[1:4]
        else:
            n0 = node_of(water_group[0])
            root = RootCause(node_id=n0, label=f"no upstream source on record for {n0}", score=0.0, factors=[],
                             explanation="the water graph has no source reaching this node")
        domain = "facade" if all(obs_domain(b, o) == "facade" for o in water_group) else "water"
    else:
        root = electrical_root(b, group)
        domain = "electrical"
    flags = sorted({f for o in group if o.finding is not None for f in o.finding.unified.flags if f in VALID_FLAGS})
    if water_group and _fire_shock(b, root.node_id, water_group):
        flags = sorted(set(flags) | {"fire_shock_pathway"})
        domain = "mixed"
    elif water_group and "electrical" in doms:
        domain = "mixed"
    worst = _worst(o.level for o in group)
    level: Level = worst if worst is not None else "U"  # type: ignore[assignment]
    if "fire_shock_pathway" in flags and any(_native(o) == "Active leak" for o in group):
        level = "S4"  # interior_water.json unified_note [team-proposed, validate]
    known_root = b.zone(root.node_id) is not None or b.element(root.node_id) is not None
    ds = downstream(b, root.node_id) if water_group and known_root else {}
    nodes = list(dict.fromkeys(node_of(o) for o in group if node_of(o)))
    if domain == "electrical":
        nodes.sort(key=lambda n: -max((LEVEL_RANK.get(o.level or "", -1) for o in group if node_of(o) == n), default=-1))
    else:
        nodes.sort(key=lambda n: (-(ds[n].score if n in ds else -1.0), n))
    pid = f"P-{root.node_id}"  # stable across re-analysis: the domain can flip (water to mixed), the root keeps the id
    dated = [o.ts for o in group if not o.ts_unknown]
    p = Problem(problem_id=pid, domain=domain, title=f"{root.label}: {len(group)} observations on {len(nodes)} nodes",
                level=level, priority=None, root_cause=root, alternatives=alternatives,
                observation_ids=[o.obs_id for o in group], affected_nodes=nodes, flags=flags,
                created_ts=min(dated) if dated else iso(now), updated_ts=max(dated) if dated else iso(now),
                synthetic=data.any_synthetic or any(o.synthetic for o in group))
    esc = estimate_escalation(p, group, data, water, now)
    return p.model_copy(update={"escalation": esc, "priority": priority(level, flags, esc)})


def build_problems(b: Building, data: BuildingData, water: WaterResult, elec: ElectricalResult, now: datetime,
                   extra_obs: Sequence[Observation] = ()) -> List[Problem]:
    """Problems from water, electrical, ticket and extra (image) observations, as of `now`: anything stamped after
    `now` is left out, and an undated photo (ts_unknown) is placed at `now` without driving any clock. Groups whose
    root gives the same problem_id are merged (for example one facade leak seen after several storms)."""
    uniq = as_of_observations(list(water.observations) + list(elec.observations) + tickets_to_observations(b, data.tickets)
                              + list(extra_obs), now)
    groups = cluster(b, uniq, fits=water.fits)
    problems: Dict[str, Tuple[List[Observation], Problem]] = {}
    for g in groups:
        p = _make_problem(b, g, data, water, uniq, now)
        if p.problem_id in problems:
            merged = sorted(problems[p.problem_id][0] + g, key=lambda o: (o.ts, o.obs_id))
            p2 = _make_problem(b, merged, data, water, uniq, now)
            problems.pop(p.problem_id)
            if p2.problem_id in problems:  # the merged root matches yet another problem: fold it in too
                merged = sorted(problems.pop(p2.problem_id)[0] + merged, key=lambda o: (o.ts, o.obs_id))
                p2 = _make_problem(b, merged, data, water, uniq, now)
            problems[p2.problem_id] = (merged, p2)
        else:
            problems[p.problem_id] = (g, p)
    return sort_problems([p for _, p in problems.values()])


def as_of_observations(obs: Sequence[Observation], now: datetime) -> List[Observation]:
    """Unique observations (first obs_id wins) with ts <= now. Undated ones (ts_unknown) are placed at `now`."""
    seen: set = set()
    out: List[Observation] = []
    stamp = iso(now)
    for o in obs:
        if o.obs_id in seen:
            continue
        seen.add(o.obs_id)
        if o.ts_unknown:
            o = o.model_copy(update={"ts": stamp})
        elif parse_ts(o.ts) > now:
            continue
        out.append(o)
    return out


# ----------------------------------------------------------------------------------------- image observations

def image_observations(b: Building, findings: Sequence[Finding], records: Sequence[ImageRecord]) -> List[Observation]:
    """Graded building photos as observations on their asset_id (a zone or an element). Findings whose image has no
    known asset_id are skipped (they stay in the grading run)."""
    try:
        from . import grading as gr  # the grading owner's converter, when present
        conv = getattr(gr, "finding_to_observation", None)
    except ImportError:
        conv = None
    by_id = {r.image_id: r for r in records}
    out: List[Observation] = []
    for f in findings:
        rec = by_id.get(f.evidence.image_ids[0]) if f.evidence.image_ids else None
        if rec is None or not rec.asset_id:
            continue
        if b.zone(rec.asset_id) is None and b.element(rec.asset_id) is None:
            continue
        if conv is not None:
            try:
                out.append(conv(f, rec, b))
                continue
            except Exception:
                pass
        is_zone = b.zone(rec.asset_id) is not None
        out.append(Observation(obs_id=f"img:{f.finding_id}", kind="image_finding", ts=f"{rec.captured_on or '1970-01-01'}T00:00:00Z",
                               zone_id=rec.asset_id if is_zone else None, element_id=None if is_zone else rec.asset_id,
                               source="grader", level=f.unified.level, text=f.justification, finding=f,
                               synthetic=bool(rec.labels.get("synthetic", False)), ts_unknown=not rec.captured_on))
    return out


# ----------------------------------------------------------------------------------------- store

def _resolved_ts(p: Problem) -> Optional[str]:
    for h in reversed(p.history):
        if h.to_status == "resolved":
            return h.ts
    return None


def _now_iso() -> str:
    return iso(datetime.now(timezone.utc))


class ProblemStore:
    """problems.json (utf-8, indent 1): problems with status and history. Rows are never silently deleted."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> List[Problem]:
        if not self.path.exists():
            return []
        return [Problem.model_validate(r) for r in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self, problems: Sequence[Problem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([p.model_dump(mode="json") for p in problems], indent=1), encoding="utf-8")

    def merge(self, fresh: Sequence[Problem], now: str) -> List[Problem]:
        """Same problem_id keeps status and history. A fresh problem with a new id that shares observations with an
        old row not detected now (its root or domain changed) takes over that row's status and history (note
        "re-keyed from ..."); the old row stays as inactive history. An observation after the resolved time reopens a
        resolved or verified problem to "detected" (note "recurred"). Old rows not detected now are kept with
        active=False (not open). Saves and returns all."""
        old = {p.problem_id: p for p in self.load()}
        out: Dict[str, Problem] = dict(old)
        fresh_ids = {p.problem_id for p in fresh}
        claimed: set = set()
        for pid, q in old.items():
            if pid not in fresh_ids:
                out[pid] = q.model_copy(update={"active": False})
        for p in fresh:
            prev = old.get(p.problem_id)
            rekeyed = False
            if prev is None:
                cands = [q for q in old.values() if q.problem_id not in fresh_ids and q.problem_id not in claimed
                         and set(q.observation_ids) & set(p.observation_ids)]
                if cands:
                    prev = max(cands, key=lambda q: (len(set(q.observation_ids) & set(p.observation_ids)), q.updated_ts))
                    claimed.add(prev.problem_id)
                    rekeyed = True
                    out[prev.problem_id] = prev.model_copy(update={"active": False, "history": list(prev.history) + [
                        StatusChange(ts=now, from_status=prev.status, to_status=prev.status, actor="system",
                                     note=f"superseded by {p.problem_id}")]})
            if prev is None:
                out[p.problem_id] = p.model_copy(update={"status": "detected",
                                                         "history": [StatusChange(ts=now, from_status=None, to_status="detected", actor="system")]})
                continue
            status: ProblemStatus = prev.status
            history = list(prev.history)
            if rekeyed:
                history.append(StatusChange(ts=now, from_status=prev.status, to_status=prev.status, actor="system",
                                            note=f"re-keyed from {prev.problem_id}"))
            rts = _resolved_ts(prev)
            if prev.status in ("resolved", "verified") and rts is not None and parse_ts(p.updated_ts) > parse_ts(rts):
                history.append(StatusChange(ts=now, from_status=prev.status, to_status="detected", actor="system", note="recurred"))
                status = "detected"
            out[p.problem_id] = p.model_copy(update={"status": status, "history": history, "created_ts": min(prev.created_ts, p.created_ts)})
        merged = sort_problems(list(out.values()))
        self.save(merged)
        return merged

    def transition(self, problem_id: str, to: ProblemStatus, actor: str, note: str = "", now: Optional[str] = None) -> Problem:
        """Move one problem along ALLOWED. ValueError on any other move, and on "verified" while an observation is
        newer than the resolved time."""
        problems = self.load()
        idx = next((i for i, p in enumerate(problems) if p.problem_id == problem_id), None)
        if idx is None:
            raise KeyError(f"unknown problem {problem_id!r}")
        p = problems[idx]
        if to not in ALLOWED[p.status]:
            raise ValueError(f"{problem_id}: {p.status} -> {to} is not allowed (allowed: {sorted(ALLOWED[p.status]) or 'none'})")
        if to == "verified":
            rts = _resolved_ts(p)
            if rts is None or parse_ts(p.updated_ts) > parse_ts(rts):
                raise ValueError(f"{problem_id}: cannot verify, an observation ({p.updated_ts}) is newer than the resolution ({rts})")
        ts = now or _now_iso()
        p2 = p.model_copy(update={"status": to, "history": list(p.history) + [StatusChange(ts=ts, from_status=p.status, to_status=to, actor=actor, note=note)]})
        problems[idx] = p2
        self.save(problems)
        return p2


def work_orders_csv(problems: Sequence[Problem]) -> str:
    """CMMS export: id, title, level, priority, status, root, due, basis. U rows carry an empty priority. due is the
    analysis time the clock counts from (Escalation.as_of) plus its hours; empty when there is no running clock, the
    clock has no as_of, or the problem was not detected in the latest run."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "title", "level", "priority", "status", "root", "due", "basis"])
    for p in sort_problems(problems):
        due = ""
        esc = p.escalation
        if p.active and esc is not None and esc.hours is not None and esc.as_of:
            due = iso(parse_ts(esc.as_of) + timedelta(hours=esc.hours))
        basis = p.escalation.basis if p.escalation else "; ".join(sorted({f.basis for f in p.root_cause.factors})) or ""
        w.writerow([p.problem_id, p.title, p.level, "" if p.priority is None else p.priority, p.status, p.root_cause.node_id, due, basis])
    return buf.getvalue()


# ----------------------------------------------------------------------------------------- end to end

def default_now(data: BuildingData) -> datetime:
    """The last timestamp in the sensor data (deterministic for synthetic), else weather, thermal, then the clock."""
    for idx in (data.sensors.index if data.sensors is not None else None, data.weather.index if data.weather is not None else None):
        if idx is not None and len(idx):
            return pd.Timestamp(idx.max()).to_pydatetime()
    if data.thermal is not None and len(data.thermal):
        return pd.Timestamp(data.thermal["ts"].max()).to_pydatetime()
    return datetime.now(timezone.utc)


def analyze(root: Path, now: Optional[datetime] = None) -> Tuple[BuildingData, WaterResult, ElectricalResult, List[Problem]]:
    """load_building_dir -> analyze_water -> analyze_electrical -> tickets + image observations -> build_problems ->
    ProblemStore.merge -> pipeline.save_findings(root / "analysis")."""
    from .electrical import analyze_electrical
    from .water import analyze_water

    root = Path(root)
    data = load_building_dir(root)
    now = now or default_now(data)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    water = analyze_water(data, now)
    elec = analyze_electrical(data, now)
    img_findings: List[Finding] = []
    grading = root / FILES["grading"]
    if (grading / "findings.json").exists():
        img_findings = list(load_run(grading)["findings"])
    img_obs = image_observations(data.building, img_findings, data.images)
    fresh = build_problems(data.building, data, water, elec, now, extra_obs=img_obs)
    merged = ProblemStore(root / FILES["problems"]).merge(fresh, iso(now))
    analysis = root / FILES["analysis"]
    analysis.mkdir(parents=True, exist_ok=True)
    save_findings(list(water.findings) + list(elec.findings), analysis)
    obs = as_of_observations(list(water.observations) + list(elec.observations) + tickets_to_observations(data.building, data.tickets)
                             + img_obs, now)
    (root / FILES["observations"]).write_text("".join(o.model_dump_json() + "\n" for o in obs), encoding="utf-8")
    return data, water, elec, merged
