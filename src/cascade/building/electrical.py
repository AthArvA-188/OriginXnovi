"""Electrical: panel schedule tree, load vs rating, IR thermal bands, per-circuit baselines (building_spec section 8).

Every reading is graded deterministically against rows of rubrics/electrical_thermal.json (standard ELEC-TP):
- delta_t_similar_k: the latest IR reading per element, delta-T = t_element_c - t_reference_c (DT_BANDS_K,
  [team-proposed, validate]). U when the reference or element temperature is missing, or the load at scan time is
  missing or below IR_MIN_LOAD_PCT.
- load_pct_continuous: the highest CONTINUOUS_H-hour mean current as percent of the breaker rating
  (LOAD_CONT_PCT, CONTINUOUS_H, [Assumption]). U when the rating or the current data is missing.
- days_since_ir: per panel, days since the panel or one of its circuits was last scanned (IR_INTERVAL_DAYS).
  Never scanned gives U. Scanned within the interval gives no finding (the rubric has no S0 row for it).

Hour-of-week current baselines give anomaly indicators only. They have no rubric row and never set a level.
U is never S0 here: a missing reading is U with not_measurable, never "Band 0" or "Load ok".
No model API is called. Analysis never reads truth.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..grade import load_rubric
from ..schema import Evidence, Finding
from .graph import electrical_children, electrical_parent
from .model import Building, BuildingData, Edge, ElectricalResult, Observation
from .rules import THRESHOLDS, Threshold, find_band_row, finding_from_row, t, u_finding

ASSET_CLASS = "electrical_equipment"
LOAD_COLUMNS = ["circuit_id", "panel_id", "sensor_id", "rating_a", "peak_a", "p95_a", "max_cont_a", "max_cont_pct",
                "max_cont_ts", "hours_over_cont", "n_hours", "recent_cont_pct", "recent_cont_ts"]
ANOMALY_COLUMNS = ["ts", "sensor_id", "value", "expected", "score"]
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Spec section 8 names a 0.1 A floor on the MAD. It is not in rules.THRESHOLDS yet (a core request), so it is
# registered here with its tag. When core adds the key, the registry value wins (see _th).
LOCAL_THRESHOLDS: Dict[str, Threshold] = {
    "ANOMALY_MAD_FLOOR_A": Threshold(key="ANOMALY_MAD_FLOOR_A", value=0.1, unit="A",
                                     meaning="floor on the hour-of-week MAD so a flat baseline does not flag noise",
                                     tag="team-proposed, validate", note="building_spec section 8; not yet in rules.THRESHOLDS"),
}


def _th(key: str) -> Any:
    """Registry value, falling back to the tagged local entries above."""
    return THRESHOLDS[key].value if key in THRESHOLDS else LOCAL_THRESHOLDS[key].value


def _rubric() -> dict:
    return load_rubric(ASSET_CLASS)


def _iso(ts: Any) -> str:
    d = pd.Timestamp(ts)
    if d.tzinfo is None:
        d = d.tz_localize("UTC")
    return d.tz_convert("UTC").strftime(TS_FMT)


def _utc(d: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _finite(x: Any) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _blank(x: Any) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or (x is pd.NA) or str(x).strip() in ("", "nan", "None", "<NA>")


def _fid(kind: str, element_id: str, ts: Optional[str]) -> str:
    """Finding ids carry the reading time, so electrical_observations can place them in time."""
    return f"elec:{kind}:{element_id}:{ts or 'unknown'}"


def _ts_of(finding_id: str) -> Optional[str]:
    parts = finding_id.split(":", 3)
    return parts[3] if len(parts) == 4 and parts[3] != "unknown" else None


def _element_of(finding_id: str) -> Optional[str]:
    parts = finding_id.split(":", 3)
    return parts[2] if len(parts) >= 3 else None


# ----------------------------------------------------------------------------------------- panel schedule tree

def schedule_to_edges(schedule: pd.DataFrame) -> List[Edge]:
    """feeds edges from the panel schedule: fed_from -> panel and panel -> circuit, each once, in schedule order."""
    out: List[Edge] = []
    seen = set()
    if schedule is None or schedule.empty:
        return out
    for _, r in schedule.iterrows():
        panel, src, circ = r.get("panel_id"), r.get("fed_from"), r.get("circuit_id")
        pairs = []
        if not _blank(src) and not _blank(panel):
            pairs.append((str(src), str(panel)))
        if not _blank(panel) and not _blank(circ):
            pairs.append((str(panel), str(circ)))
        for a, b in pairs:
            if (a, b) not in seen:
                seen.add((a, b))
                out.append(Edge(src=a, dst=b, kind="feeds", weight=None, basis="panel_schedule.csv"))
    return out


def check_tree(b: Building, schedule: pd.DataFrame) -> List[str]:
    """Readable mismatches between the panel schedule and the building's feeds edges. Empty means they agree.

    Circuit -> load edges are not in a panel schedule, so they are not compared."""
    problems: List[str] = []
    sched = {(e.src, e.dst) for e in schedule_to_edges(schedule)}
    kinds = {"switchboard", "electrical_panel", "circuit"}
    bld = set()
    for e in b.edges:
        if e.kind != "feeds":
            continue
        dst = b.element(e.dst)
        if dst is not None and dst.kind in ("electrical_panel", "circuit"):
            bld.add((e.src, e.dst))
    for a, c in sorted(sched):
        for node in (a, c):
            el = b.element(node)
            if el is None:
                problems.append(f"schedule names {node}, which is not an element of the building")
            elif el.kind not in kinds:
                problems.append(f"schedule names {node} as electrical, but the building has it as {el.kind}")
    for a, c in sorted(sched - bld):
        problems.append(f"schedule has {a} feeds {c}; the building does not")
    for a, c in sorted(bld - sched):
        problems.append(f"building has {a} feeds {c}; the panel schedule does not")
    if schedule is not None and not schedule.empty and "breaker_a" in schedule.columns:
        for _, r in schedule.iterrows():
            el = b.element(str(r.get("circuit_id")))
            br = pd.to_numeric(pd.Series([r.get("breaker_a")]), errors="coerce").iloc[0]
            if el is not None and _finite(br) and el.rating_a is not None and abs(float(br) - float(el.rating_a)) > 1e-9:
                problems.append(f"{el.element_id}: schedule breaker {float(br):g} A, building rating {el.rating_a:g} A")
    return list(dict.fromkeys(problems))


# ----------------------------------------------------------------------------------------- load vs rating

def _circuit_sensor(b: Building) -> Dict[str, str]:
    return {s.element_id: s.sensor_id for s in b.sensors if s.type == "circuit_current" and s.element_id}


def _hourly(currents: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(currents[col], errors="coerce")
    if isinstance(s.index, pd.DatetimeIndex) and len(s):
        s = s.resample("h").mean()
    return s


def load_table(b: Building, currents: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit (schedule circuits first, then building circuits missing from it).

    rating_a is the schedule breaker_a, else the building rating_a. max_cont_pct = highest CONTINUOUS_H-hour mean
    current / rating * 100 (NaN when the rating is missing, graded U). hours_over_cont = hours inside a window whose
    mean is at or above LOAD_CONT_PCT. recent_cont_pct is the highest CONTINUOUS_H-hour mean in the last
    LOAD_RECENT_D days before the last reading (is the overload still happening? loads cycle daily and weekly).
    Current data missing gives NaN everywhere (never 0 A)."""
    n = int(t("CONTINUOUS_H"))
    lim = float(t("LOAD_CONT_PCT"))
    sensor_of = _circuit_sensor(b)
    parent = electrical_parent(b)
    rows: List[Dict[str, Any]] = []
    order: List[Tuple[str, Optional[str], Any]] = []
    seen = set()
    if schedule is not None and not schedule.empty:
        for _, r in schedule.iterrows():
            c = r.get("circuit_id")
            if _blank(c) or str(c) in seen:
                continue
            seen.add(str(c))
            order.append((str(c), None if _blank(r.get("panel_id")) else str(r.get("panel_id")), r.get("breaker_a")))
    for e in b.elements:
        if e.kind == "circuit" and e.element_id not in seen:
            seen.add(e.element_id)
            order.append((e.element_id, parent.get(e.element_id), None))
    cols = set(currents.columns) if currents is not None else set()
    for cid, panel, breaker in order:
        el = b.element(cid)
        br = pd.to_numeric(pd.Series([breaker]), errors="coerce").iloc[0]
        rating = float(br) if _finite(br) and float(br) > 0 else (float(el.rating_a) if el is not None and _finite(el.rating_a) and el.rating_a > 0 else np.nan)
        sid = sensor_of.get(cid)
        row: Dict[str, Any] = {"circuit_id": cid, "panel_id": panel or parent.get(cid), "sensor_id": sid, "rating_a": rating,
                               "peak_a": np.nan, "p95_a": np.nan, "max_cont_a": np.nan, "max_cont_pct": np.nan,
                               "max_cont_ts": None, "hours_over_cont": np.nan, "n_hours": 0, "recent_cont_pct": np.nan,
                               "recent_cont_ts": None}
        if sid is not None and sid in cols:
            s = _hourly(currents, sid)
            valid = s.dropna()
            row["n_hours"] = int(len(valid))
            if len(valid):
                row["peak_a"] = float(valid.max())
                row["p95_a"] = float(np.percentile(valid.to_numpy(float), 95))
                roll = s.rolling(n, min_periods=n).mean()
                if roll.notna().any():
                    row["max_cont_a"] = float(roll.max())
                    row["max_cont_ts"] = _iso(roll.idxmax())
                    if np.isfinite(rating):
                        pct = roll / rating * 100.0
                        row["max_cont_pct"] = round(float(pct.max()), 2)
                        valid_pct = pct.dropna()
                        if isinstance(valid_pct.index, pd.DatetimeIndex) and len(valid_pct):
                            recent = valid_pct[valid_pct.index > valid_pct.index[-1] - pd.Timedelta(days=float(t("LOAD_RECENT_D")))]
                            row["recent_cont_pct"] = round(float(recent.max()), 2)
                            row["recent_cont_ts"] = _iso(recent.idxmax())
                        over = (pct >= lim).astype(float)
                        covered = over[::-1].rolling(n, min_periods=1).max()[::-1]
                        row["hours_over_cont"] = int(covered.sum())
        rows.append(row)
    return pd.DataFrame(rows, columns=LOAD_COLUMNS)


def load_findings(b: Building, table: pd.DataFrame) -> List[Finding]:
    """load_pct_continuous rows per circuit; asset_class electrical_equipment, modality sensor. U, never 'Load ok',
    when the rating or the current data is missing."""
    rub = _rubric()
    out: List[Finding] = []
    lim = THRESHOLDS["LOAD_CONT_PCT"].label()
    n = int(t("CONTINUOUS_H"))
    for _, r in table.iterrows():
        cid = str(r["circuit_id"])
        sid = None if _blank(r.get("sensor_id")) else str(r["sensor_id"])
        ts = None if _blank(r.get("max_cont_ts")) else str(r["max_cont_ts"])
        ev = Evidence(signal_id=sid)
        fid = _fid("load", cid, ts)
        if not _finite(r.get("rating_a")):
            out.append(u_finding(rub, finding_id=fid, asset_class=ASSET_CLASS, modality="sensor", evidence=ev,
                                 reason=f"No breaker rating for {cid} in the panel schedule or the building model, so load vs rating cannot be graded."))
            continue
        if not _finite(r.get("max_cont_pct")):
            why = "no current sensor is mapped to it" if sid is None else f"{sid} has no {n} consecutive hours of data"
            out.append(u_finding(rub, finding_id=fid, asset_class=ASSET_CLASS, modality="sensor", evidence=ev,
                                 reason=f"No continuous load for {cid}: {why}. Unmeasured, not 'Load ok'."))
            continue
        pct = float(r["max_cont_pct"])
        row = find_band_row(rub, "load_pct_continuous", pct)
        if row is None:
            out.append(u_finding(rub, finding_id=fid, asset_class=ASSET_CLASS, modality="sensor", evidence=ev,
                                 reason=f"{cid}: load {pct:.1f} % matched no rubric row."))
            continue
        just = (f"Highest {n}-hour mean current on {cid} was {float(r['max_cont_a']):.1f} A, {pct:.1f} % of the "
                f"{float(r['rating_a']):g} A breaker, in the window ending {ts}. {int(r['hours_over_cont'])} h were in windows "
                f"at or above {lim}. Peak {float(r['peak_a']):.1f} A, p95 {float(r['p95_a']):.1f} A.")
        recent = r.get("recent_cont_pct")
        if _finite(recent):
            days = THRESHOLDS["LOAD_RECENT_D"].label()
            state = "" if row["unified"] == "S0" else (
                " (overload ongoing)" if float(recent) >= float(t("LOAD_CONT_PCT")) else " (past overload, not ongoing)")
            just += f" Highest {n}-hour mean in the last {days}: {float(recent):.1f} % at {r.get('recent_cont_ts')}{state}."
        out.append(finding_from_row(row, rub, finding_id=fid, asset_class=ASSET_CLASS, modality="sensor", evidence=ev,
                                    justification=just))
    return out


# ----------------------------------------------------------------------------------------- anomalies (indicator only)

MAD_TO_SD = 1.4826  # consistency constant: 1.4826 * MAD estimates the sd of normally distributed data


def baseline_anomalies(currents: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Per column, over the first ANOMALY_BASELINE_DAYS: the expected current is the hour-of-week median (spec).
    The spread is the MAD around the median of the same hour of day and day type (weekday or weekend), times
    MAD_TO_SD, floored at ANOMALY_MAD_FLOOR_A. Pooling is a deviation from the spec's per-hour-of-week MAD: 28 days
    give only 4 samples per hour of week, and a 4-sample MAD flagged thousands of noise hours on the SYNTHETIC tower.
    Later hours with |x - expected| > ANOMALY_MAD_K * spread are returned as ts, sensor_id, value, expected, score
    (score in spread units). Hours with no baseline are not scored (unmeasured, not normal).
    Indicator only: no rubric row, so it never sets a level by itself.
    Currents are resampled hourly. Hour of week is taken in the building's local time `tz` (Building.tz), so an
    occupancy profile does not shift by one hour at each daylight-saving change."""
    empty = pd.DataFrame({c: pd.Series(dtype=d) for c, d in zip(ANOMALY_COLUMNS, ("object", "object", float, float, float))})
    if currents is None or currents.empty or not isinstance(currents.index, pd.DatetimeIndex):
        return empty
    df = currents.apply(pd.to_numeric, errors="coerce").astype(float).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.resample("h").mean()
    idx = df.index
    local = idx.tz_convert(tz or "UTC")
    how = np.asarray(local.dayofweek * 24 + local.hour)
    hod_type = np.asarray(local.hour + 24 * (local.dayofweek >= 5))  # local hour of day, weekday or weekend
    base_mask = np.asarray(idx < idx.min() + pd.Timedelta(days=float(t("ANOMALY_BASELINE_DAYS"))))
    if base_mask.all() or not base_mask.any():
        return empty
    base = df[base_mask]
    med = base.groupby(how[base_mask]).median()
    ht = hod_type[base_mask]
    pooled = base.groupby(ht).median()  # residuals around a 4-sample median are biased low, so the spread uses this
    mad = (base - pooled.reindex(ht).to_numpy()).abs().groupby(ht).median()
    after = df[~base_mask]
    exp = med.reindex(how[~base_mask]).to_numpy()
    scale = np.maximum(MAD_TO_SD * mad.reindex(hod_type[~base_mask]).to_numpy(), float(_th("ANOMALY_MAD_FLOOR_A")))  # NaN stays NaN
    vals = after.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.abs(vals - exp) / scale
    hit = np.nan_to_num(score, nan=0.0) > float(t("ANOMALY_MAD_K"))
    ri, ci = np.nonzero(hit)
    if not len(ri):
        return empty
    out = pd.DataFrame({"ts": after.index[ri].tz_convert("UTC").strftime(TS_FMT), "sensor_id": np.asarray(after.columns)[ci],
                        "value": vals[ri, ci], "expected": exp[ri, ci], "score": np.round(score[ri, ci], 2)})
    return out.sort_values(["sensor_id", "ts"]).reset_index(drop=True)[ANOMALY_COLUMNS]


def anomaly_runs(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Contiguous hourly runs of anomalies per sensor: sensor_id, start, end, hours, max_score, max_value, expected
    (value and expected at the highest-scoring hour of the run)."""
    cols = ["sensor_id", "start", "end", "hours", "max_score", "max_value", "expected"]
    if anomalies is None or anomalies.empty:
        return pd.DataFrame(columns=cols)
    a = anomalies.copy()
    a["_t"] = pd.to_datetime(a["ts"], utc=True)
    a = a.sort_values(["sensor_id", "_t"]).reset_index(drop=True)
    new = (a["sensor_id"] != a["sensor_id"].shift()) | (a["_t"].diff() != pd.Timedelta(hours=1))
    a["_run"] = new.cumsum()
    g = a.groupby("_run", sort=True)
    top = a.loc[g["score"].idxmax()].set_index("_run")
    out = pd.DataFrame({"sensor_id": g["sensor_id"].first(), "start": g["_t"].min().dt.strftime(TS_FMT),
                        "end": g["_t"].max().dt.strftime(TS_FMT), "hours": g.size(), "max_score": g["score"].max(),
                        "max_value": top["value"], "expected": top["expected"]})
    return out.reset_index(drop=True)[cols]


# ----------------------------------------------------------------------------------------- thermal

def _valid_thermal(thermal: pd.DataFrame) -> pd.DataFrame:
    """Readings with both temperatures and a load at or above IR_MIN_LOAD_PCT, with delta_t_k added."""
    if thermal is None or thermal.empty:
        return pd.DataFrame(columns=["ts", "element_id", "delta_t_k"])
    d = thermal.copy()
    for c in ("t_element_c", "t_reference_c", "load_pct"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["delta_t_k"] = d["t_element_c"] - d["t_reference_c"]
    ok = d["delta_t_k"].notna() & d["load_pct"].notna() & (d["load_pct"] >= float(t("IR_MIN_LOAD_PCT")))
    return d[ok]


def _thermal_invalid_reason(eid: str, ts: str, te: Any, tr: Any, lp: Any) -> Optional[str]:
    """Why one IR reading cannot be banded, or None when it can."""
    if not _finite(tr):
        return f"IR reading of {eid} on {ts} has no reference temperature of a similar component."
    if not _finite(te):
        return f"IR reading of {eid} on {ts} has no element temperature."
    if not _finite(lp):
        return f"IR reading of {eid} on {ts} has no load at scan time, so it cannot show the condition under load."
    if float(lp) < float(t("IR_MIN_LOAD_PCT")):
        delta = float(te) - float(tr)
        return (f"IR reading of {eid} on {ts} was taken at {float(lp):.0f} % load, below "
                f"{THRESHOLDS['IR_MIN_LOAD_PCT'].label()}, so it cannot be banded: its delta-T {delta:.1f} K is a lower bound "
                f"only, since heating rises with load.")
    return None


def classify_thermal(b: Building, thermal: pd.DataFrame) -> List[Finding]:
    """Per element, the latest VALID IR reading graded on delta_t_similar_k. When the newest scan is not valid (a
    temperature missing, or the load at scan missing or below IR_MIN_LOAD_PCT) it adds a U finding next to that
    band, never instead of it, so a later low-load scan cannot hide an earlier Band 3. An element with no valid
    reading gets only the U (never Band 0). fire_shock_pathway is not set here (problems.py does)."""
    rub = _rubric()
    out: List[Finding] = []
    if thermal is None or thermal.empty:
        return out
    d = thermal.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    for c in ("t_element_c", "t_reference_c", "t_ambient_c", "load_pct"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["element_id"]).sort_values(["element_id", "ts"])
    for eid, g in d.groupby("element_id", sort=True):
        ev = Evidence(signal_id=f"thermal:{eid}")
        reasons = [_thermal_invalid_reason(str(eid), _iso(x["ts"]), x["t_element_c"], x["t_reference_c"], x["load_pct"])
                   for _, x in g.iterrows()]
        newest = g.iloc[-1]
        if reasons[-1] is not None:  # the newest scan cannot be banded: list it as U
            out.append(u_finding(rub, finding_id=_fid("thermal", str(eid), _iso(newest["ts"])), asset_class=ASSET_CLASS,
                                 modality="thermal", evidence=ev, reason=reasons[-1]))
        valid = [i for i, why in enumerate(reasons) if why is None]
        if not valid:
            continue
        r = g.iloc[valid[-1]]
        ts = _iso(r["ts"])
        fid = _fid("thermal", str(eid), ts)
        te, tr, lp = r["t_element_c"], r["t_reference_c"], r["load_pct"]
        delta = round(float(te) - float(tr), 2)
        row = find_band_row(rub, "delta_t_similar_k", delta)
        if row is None:  # only when the rubric bands leave a gap; the shipped rubric covers every finite delta
            out.append(u_finding(rub, finding_id=fid, asset_class=ASSET_CLASS, modality="thermal", evidence=ev,
                                 reason=f"{eid}: delta-T {delta:.1f} K matched no rubric row."))
            continue
        just = (f"IR on {ts}: {eid} at {float(te):.1f} degC vs a similar component at {float(tr):.1f} degC under "
                f"{float(lp):.0f} % load gives delta-T {delta:.1f} K (bands {THRESHOLDS['DT_BANDS_K'].label()}).")
        out.append(finding_from_row(row, rub, finding_id=fid, asset_class=ASSET_CLASS, modality="thermal", evidence=ev,
                                    justification=just, measurements={"delta_t_k": delta}))
    return out


def thermal_trend(thermal: pd.DataFrame, element_id: str) -> Optional[Tuple[float, float]]:
    """(slope in K per day, residual sd in K) of delta-T over the valid readings of one element, from >= 3 readings.
    None with fewer (not a flat trend)."""
    d = _valid_thermal(thermal)
    if d.empty:
        return None
    g = d[d["element_id"].astype(str) == str(element_id)].sort_values("ts")
    if len(g) < 3:
        return None
    x = (g["ts"] - g["ts"].iloc[0]).dt.total_seconds().to_numpy(float) / 86400.0
    y = g["delta_t_k"].to_numpy(float)
    if np.ptp(x) == 0:
        return None
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    sd = float(np.sqrt(np.sum(resid ** 2) / max(len(y) - 2, 1)))
    return float(slope), sd


def next_band_window(delta_now: float, sd_k: float, slope_k_day: float) -> Optional[Tuple[float, float, float]]:
    """(days, earliest days, latest days) until delta-T reaches the next DT_BANDS_K edge above the point estimate.
    The edge is fixed from the point estimate; the band shifts delta by +/- sd against that same edge, so
    earliest <= days <= latest. None when days_to_next_band is None."""
    d = days_to_next_band(delta_now, slope_k_day)
    if d is None:
        return None
    edge = min(float(e) for e in t("DT_BANDS_K") if float(e) > float(delta_now))
    sd = abs(float(sd_k)) if _finite(sd_k) else 0.0
    lo = max(0.0, edge - (float(delta_now) + sd)) / float(slope_k_day)
    hi = (edge - max(0.0, float(delta_now) - sd)) / float(slope_k_day)
    return d, round(lo, 2), round(hi, 2)


def days_to_next_band(delta_now: float, slope_k_day: float) -> Optional[float]:
    """Days until delta-T reaches the next DT_BANDS_K edge at the given slope. None at the top band, for a flat or
    falling trend, or for a missing input."""
    if not _finite(delta_now) or not _finite(slope_k_day) or slope_k_day <= 0:
        return None
    edges = [float(e) for e in t("DT_BANDS_K") if float(e) > float(delta_now)]
    if not edges:
        return None
    return round((min(edges) - float(delta_now)) / float(slope_k_day), 2)


def ir_overdue(b: Building, thermal: pd.DataFrame, now: datetime) -> List[Finding]:
    """days_since_ir per electrical panel. A scan of the panel or of a circuit it feeds counts. Never scanned gives
    U. Scanned within IR_INTERVAL_DAYS gives no finding (the rubric has no S0 row for this family)."""
    rub = _rubric()
    kids = electrical_children(b)
    now_ts = _utc(now)
    last: Dict[str, pd.Timestamp] = {}
    if thermal is not None and not thermal.empty:
        d = thermal.dropna(subset=["element_id"]).copy()
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        d = d[d["ts"] <= now_ts]
        last = {str(k): v for k, v in d.groupby("element_id")["ts"].max().items()}
    out: List[Finding] = []
    for e in b.elements:
        if e.kind != "electrical_panel":
            continue
        scans = [last[n] for n in [e.element_id] + kids.get(e.element_id, []) if n in last]
        fid = _fid("ir", e.element_id, _iso(now_ts))
        ev = Evidence(signal_id=f"thermal:{e.element_id}")
        if not scans:
            out.append(u_finding(rub, finding_id=fid, asset_class=ASSET_CLASS, modality="thermal", evidence=ev,
                                 reason=f"No infrared scan of {e.element_id} or its circuits is on record: condition unknown, not healthy."))
            continue
        latest = max(scans)
        days = (now_ts - latest).total_seconds() / 86400.0
        row = find_band_row(rub, "days_since_ir", days)
        if row is None:
            continue
        just = (f"Last infrared scan of {e.element_id} or its circuits was {_iso(latest)}, {days:.0f} days before "
                f"{_iso(now_ts)} (interval {THRESHOLDS['IR_INTERVAL_DAYS'].label()}).")
        out.append(finding_from_row(row, rub, finding_id=fid, asset_class=ASSET_CLASS, modality="thermal", evidence=ev,
                                    justification=just))
    return out


# ----------------------------------------------------------------------------------------- observations

def electrical_observations(findings: Sequence[Finding], anomalies: pd.DataFrame, b: Building,
                            now: Optional[datetime] = None, load: Optional[pd.DataFrame] = None,
                            synthetic: Optional[bool] = None) -> List[Observation]:
    """Observations for the problem manager.

    - One per non-S0 finding (U included, so it is listed): thermal findings as thermal_reading, load findings as
      load_event, IR-schedule findings as sensor_event with source 'ir_schedule'.
    - One load_event per anomaly run of at least CONTINUOUS_H consecutive hours, level None (indicator only).
    `now` places findings whose reading time is unknown (additive to the spec signature). With `load` (the
    load_table), a load observation's value is the highest CONTINUOUS_H-hour mean in % over the last
    LOAD_RECENT_D days (None when unknown), so the
    problem manager can tell an ongoing overload from a past one. `synthetic` (default b.synthetic) labels them."""
    out: List[Observation] = []
    fallback = _iso(now) if now is not None else None
    syn = bool(b.synthetic if synthetic is None else (synthetic or b.synthetic))
    recent_pct: Dict[str, float] = {}
    if load is not None and len(load) and "recent_cont_pct" in load.columns:
        recent_pct = {str(r["circuit_id"]): float(r["recent_cont_pct"]) for _, r in load.iterrows() if _finite(r.get("recent_cont_pct"))}
    for f in findings:
        if f.unified.level == "S0":
            continue
        parts = f.finding_id.split(":", 3)
        kind = parts[1] if len(parts) >= 2 else ""
        eid = _element_of(f.finding_id)
        el = b.element(eid) if eid else None
        ts = _ts_of(f.finding_id) or fallback
        if el is None or ts is None:
            continue
        if kind == "thermal":
            okind, src, val, unit = "thermal_reading", "thermal", f.measurements.delta_t_k, "K"
        elif kind == "load":
            okind, src, val, unit = "load_event", f"load:{eid}", recent_pct.get(str(eid)), "%"
        else:
            okind, src, val, unit = "sensor_event", "ir_schedule", None, None
        out.append(Observation(obs_id=f"obs:{f.finding_id}", kind=okind, ts=ts, zone_id=el.zone_id, element_id=eid,
                               source=src, level=f.unified.level, value=val, unit=unit,
                               text=f"{f.native_scale.value}: {f.justification}", finding=f, synthetic=syn))
    runs = anomaly_runs(anomalies)
    min_h = int(t("CONTINUOUS_H"))
    for _, r in runs.iterrows():
        if int(r["hours"]) < min_h:
            continue
        s = b.sensor(str(r["sensor_id"]))
        if s is None:
            continue
        eid = s.element_id if s.element_id and b.element(s.element_id) is not None else None
        out.append(Observation(
            obs_id=f"anom:{r['sensor_id']}:{r['start']}", kind="load_event", ts=str(r["start"]),
            zone_id=s.zone_id, element_id=eid, source=f"load:{eid or s.sensor_id}", level=None,
            value=round(float(r["max_value"]), 2), unit="A",
            text=(f"Current off its hour-of-week baseline for {int(r['hours'])} h from {r['start']} to {r['end']}: up to "
                  f"{float(r['max_value']):.1f} A vs {float(r['expected']):.1f} A expected ({float(r['max_score']):.1f} MAD; "
                  f"ANOMALY_MAD_K and ANOMALY_BASELINE_DAYS [team-proposed, validate]). Indicator only, not graded."),
            synthetic=syn))
    return out


# ----------------------------------------------------------------------------------------- end to end

def current_frame(data: BuildingData, now: Optional[datetime] = None) -> pd.DataFrame:
    """The circuit_current sensor columns present in the sensor data, up to `now`."""
    ids = [s.sensor_id for s in data.building.sensors if s.type == "circuit_current"]
    sens = data.sensors if data.sensors is not None else pd.DataFrame()
    cols = [c for c in ids if c in sens.columns]
    cur = sens[cols]
    if now is not None and len(cur) and isinstance(cur.index, pd.DatetimeIndex):
        cur = cur[cur.index <= _utc(now)]
    return cur


def analyze_electrical(data: BuildingData, now: datetime) -> ElectricalResult:
    """Load table and findings, thermal and IR-interval findings, anomalies and observations, all up to `now`."""
    b = data.building
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cur = current_frame(data, now)
    table = load_table(b, cur, data.schedule)
    thermal = data.thermal
    if thermal is not None and not thermal.empty:
        thermal = thermal[pd.to_datetime(thermal["ts"], utc=True) <= _utc(now)]
    findings = classify_thermal(b, thermal) + load_findings(b, table) + ir_overdue(b, thermal, now)
    anomalies = baseline_anomalies(cur, tz=b.tz)
    syn = bool(data.synthetic_sources.get("sensors") or data.synthetic_sources.get("thermal"))
    obs = electrical_observations(findings, anomalies, b, now=now, load=table, synthetic=syn)
    return ElectricalResult(load_table=table, anomalies=anomalies, observations=obs, findings=findings)
