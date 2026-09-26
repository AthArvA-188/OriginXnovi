"""Water intrusion tracking and prediction (building_spec section 7).

What it does, in order:
- detect_storms: rain events from the weather series, with a wind-driven-rain (WDR) proxy per facade side.
- storm_responses and fit_response: for each humidity-sensored zone, how much RH rises after each storm, and which
  driver (total rain, or WDR on one facade side) explains the rises best. A zone without a sensor is unmeasured,
  never "no response".
- grade_moisture: deterministic interior_water.json rows per sensor (RH ok, RH elevated, RH mold-risk, Active leak,
  Drying window exceeded). A sensor with no data in the window is graded U, never "RH ok".
- water_observations: storm-response events, leak-sensor wet runs and non-S0 findings as Observations for problems.py.
- zone_risk: additive, explained risk factors (RISK_WEIGHTS). A factor that could not be measured has value None.
- forecast: for a hypothetical storm, per zone, the RH peak with a band and the hours until EPA_RH_MAX is crossed.

Every threshold is read from rules.THRESHOLDS or from LOCAL below (each with a tag). No model API is called here.
Analysis never reads truth.json. Synthetic inputs stay labelled synthetic on every observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..grade import load_rubric
from ..prioritize import SEVERITY_WEIGHT
from ..schema import Evidence, Finding
from .blueprint import TS_FORMAT, ticket_category, tickets_to_observations
from .graph import upstream
from .model import AZIMUTH, Building, BuildingData, Factor, Observation, ResponseFit, StormEvent, WaterResult, ZoneForecast
from .rules import THRESHOLDS, Threshold, find_band_row, finding_from_row, t

RESP_COLUMNS: List[str] = ["storm_id", "start", "pre", "post", "rise", "lag_h", "post_ts"]
GRID_COLUMNS: List[str] = ["floor", "level", "side", "zone_id", "risk", "top_factor", "top_basis", "measured"]
RISK_BASIS = "RISK_WEIGHTS [team-proposed, validate]"
CLOSED_TICKET_STATUSES = ("closed", "resolved", "done", "complete", "completed", "cancelled", "canceled")

# Spec-given scalings that rules.THRESHOLDS does not hold yet (request to core: move them into the registry).
LOCAL: Dict[str, Threshold] = {th.key: th for th in (
    Threshold("EXPOSURE_WINDOW_D", 365, "days", "look-back for the annual wind-driven-rain share of a facade side",
              "team-proposed, validate", note="building_spec section 7: 'annual wdr share'"),
    Threshold("RECENT_WINDOW_D", 30, "days", "look-back for open water tickets and upstream defects in zone risk",
              "team-proposed, validate", note="building_spec section 7: tickets 'in 30 days'"),
    Threshold("TICKET_SATURATION", 3.0, "tickets", "open_tickets factor = min(1, n / this)", "team-proposed, validate",
              note="building_spec section 7"),
    Threshold("RESPONSE_SCALE_X", 3.0, "x RESPONSE_MIN_RISE", "rain_response factor = expected rise / (RESPONSE_MIN_RISE * this)",
              "team-proposed, validate", note="building_spec section 7"),
    Threshold("EVENT_LEVEL", {"single": "S1", "repeated": "S2"}, "",
              "indicator level of one storm-response event: S1 (monitor) alone; S2 (schedule) when the zone's rain response "
              "is significant (repeated and driver-linked). It mirrors interior_water.json 'Stain' (S1) and 'Efflorescence' "
              "(S2, 'a sign of repeated water passage'). No rubric row grades an RH storm response yet",
              "team-proposed, validate"),
)}


def _th(key: str) -> Any:
    """A threshold value from the core registry, else from LOCAL. KeyError on an unknown key."""
    return THRESHOLDS[key].value if key in THRESHOLDS else LOCAL[key].value


def threshold_label(key: str) -> str:
    """Printable label with source or tag, for the UI and justifications."""
    return (THRESHOLDS.get(key) or LOCAL[key]).label()


# ----------------------------------------------------------------------------------------- small helpers

@lru_cache(maxsize=65536)
def _parse(x: str) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _utc(x: Any) -> pd.Timestamp:
    if isinstance(x, str):
        return _parse(x)
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _iso(x: Any) -> str:
    return _utc(x).strftime(TS_FORMAT)


def _hours(a: Any, b: Any) -> float:
    return float((_utc(b) - _utc(a)) / pd.Timedelta(hours=1))


def orientations(b: Building) -> List[str]:
    """Facade orientations present in the building (from facade_drop zones); N, E, S, W when there are none."""
    found = {z.orientation for z in b.zones if z.kind == "facade_drop" and z.orientation}
    return sorted(found, key=lambda o: AZIMUTH[o]) if found else ["N", "E", "S", "W"]


def drivers_for(b: Building) -> List[str]:
    """Candidate rain drivers: total rain plus WDR on each facade orientation in the building."""
    return ["rain"] + [f"wdr_{o}" for o in orientations(b)]


def _series(data: BuildingData, sensor_id: str, now: Optional[pd.Timestamp] = None) -> pd.Series:
    s = data.sensors
    if s is None or sensor_id not in s.columns:
        return pd.Series(dtype=float)
    ser = s[sensor_id].astype(float)
    if now is not None and len(ser):
        ser = ser[ser.index <= now]
    return ser


def _rubric() -> dict:
    return load_rubric("interior_zone")


def _row(rub: dict, value: str) -> dict:
    return next(r for r in rub["rows"] if r["value"] == value)


# ----------------------------------------------------------------------------------------- storms

def wdr_index(weather: pd.DataFrame, orientation: str) -> pd.Series:
    """Hourly wind-driven-rain proxy on a facade (WDR_PROXY): rain_mm * wind_speed_ms * max(0, cos(wind_from - azimuth)).
    wind_dir_deg is the direction the wind comes FROM. Missing inputs stay NaN (never 0)."""
    name = f"wdr_{orientation}"
    if weather is None or len(weather) == 0:
        return pd.Series(dtype=float, name=name)
    ang = np.deg2rad(weather["wind_dir_deg"].astype(float) - AZIMUTH[orientation])
    cos = np.clip(np.cos(ang), 0.0, None)
    return (weather["rain_mm"].astype(float) * weather["wind_speed_ms"].astype(float) * cos).rename(name)


def detect_storms(weather: pd.DataFrame, b: Building) -> List[StormEvent]:
    """Contiguous rain hours, split by at least STORM_GAP_H dry hours; events below STORM_MIN_MM are dropped.
    wdr holds the storm total of the WDR proxy for each facade orientation used in the building."""
    if weather is None or len(weather) == 0 or "rain_mm" not in weather.columns:
        return []
    w = weather.sort_index()
    rain = w["rain_mm"].astype(float).to_numpy()
    wet = np.flatnonzero(np.nan_to_num(rain, nan=0.0) > 0)
    if len(wet) == 0:
        return []
    idx = w.index
    hrs = ((idx - idx[0]) / pd.Timedelta(hours=1)).to_numpy(float)
    gap = float(t("STORM_GAP_H"))
    groups: List[List[int]] = [[int(wet[0])]]
    for p in wet[1:]:
        if hrs[p] - hrs[groups[-1][-1]] - 1.0 >= gap:
            groups.append([int(p)])
        else:
            groups[-1].append(int(p))
    orients = orientations(b)
    wdr = {o: wdr_index(w, o).to_numpy(float) for o in orients}
    speed = w["wind_speed_ms"].astype(float).to_numpy()
    wdir = np.deg2rad(w["wind_dir_deg"].astype(float).to_numpy())
    out: List[StormEvent] = []
    for g in groups:
        sl = slice(g[0], g[-1] + 1)
        r = np.nan_to_num(rain[sl], nan=0.0)
        total = float(r.sum())
        if total < float(t("STORM_MIN_MM")):
            continue
        ok = np.isfinite(wdir[sl]) & np.isfinite(speed[sl]) & (r > 0)
        wts = r[ok]
        if wts.sum() > 0:
            deg = math.degrees(math.atan2(float((wts * np.sin(wdir[sl][ok])).sum()), float((wts * np.cos(wdir[sl][ok])).sum()))) % 360.0
            spd = float((wts * speed[sl][ok]).sum() / wts.sum())
        else:
            deg, spd = float("nan"), float("nan")
        out.append(StormEvent(storm_id=f"ST-{len(out) + 1:03d}", start=_iso(idx[g[0]]), end=_iso(idx[g[-1]]),
                              total_mm=round(total, 1), peak_mm_h=round(float(r.max()), 1), wind_dir_deg=round(deg, 1),
                              wind_speed_ms=round(spd, 2),
                              wdr={o: round(float(np.nansum(wdr[o][sl])), 2) for o in orients}))
    return out


def driver_value(st: StormEvent, driver: str) -> float:
    """The storm's value of a driver: total_mm for "rain", the storm WDR total for "wdr_<o>". NaN when unknown."""
    if driver == "rain":
        return float(st.total_mm)
    if driver.startswith("wdr_"):
        v = (st.wdr or {}).get(driver[4:])
        return float(v) if v is not None else float("nan")
    return float("nan")


# ----------------------------------------------------------------------------------------- rain response

def storm_responses(series: pd.Series, storms: Sequence[StormEvent]) -> pd.DataFrame:
    """Per storm: pre = median over RESPONSE_PRE_H before start; post = max over [start, end + RESPONSE_WINDOW_H],
    cut at the next storm's start so one response is never credited to two storms; rise = post - pre; lag_h = hours
    from the storm peak hour to the post max. The peak hour is taken as the midpoint of start and end, because
    StormEvent carries no peak timestamp. A storm whose pre window overlaps the previous storm's response window has
    a contaminated baseline: it gets NaN (left out of the fits, not 0). Missing data gives NaN (never 0)."""
    rows: List[Dict[str, Any]] = []
    s = series.dropna().sort_index() if series is not None else pd.Series(dtype=float)
    vals = s.to_numpy(float)
    idx = s.index
    ns = pd.DatetimeIndex(idx).as_unit("ns").asi8 if len(s) else np.array([], dtype=np.int64)  # UTC nanoseconds
    hour = 3_600_000_000_000
    pre_ns, win_ns = int(float(t("RESPONSE_PRE_H")) * hour), int(float(t("RESPONSE_WINDOW_H")) * hour)
    ordered = sorted(storms, key=lambda x: _utc(x.start).value)
    for i, st in enumerate(ordered):
        row: Dict[str, Any] = {"storm_id": st.storm_id, "start": st.start, "pre": np.nan, "post": np.nan, "rise": np.nan,
                               "lag_h": np.nan, "post_ts": None}
        start, end = _utc(st.start).value, _utc(st.end).value
        overlap = i > 0 and _utc(ordered[i - 1].end).value + win_ns > start - pre_ns
        if len(s) and not overlap:
            stop = end + win_ns
            if i + 1 < len(ordered):
                stop = min(stop, _utc(ordered[i + 1].start).value - 1)
            a, b0, c = np.searchsorted(ns, start - pre_ns, "left"), np.searchsorted(ns, start, "left"), np.searchsorted(ns, stop, "right")
            pre_v, post_v = vals[a:b0], vals[b0:c]
            if len(pre_v) and len(post_v):
                k = int(np.argmax(post_v))
                pre, post = float(np.median(pre_v)), float(post_v[k])
                mid = start + (end - start) / 2.0
                row.update(pre=round(pre, 2), post=round(post, 2), rise=round(post - pre, 2),
                           lag_h=round(float((ns[b0 + k] - mid) / hour), 1), post_ts=_iso(idx[b0 + k]))
        rows.append(row)
    return pd.DataFrame(rows, columns=RESP_COLUMNS)


def driver_stats(resp: pd.DataFrame, storms: Sequence[StormEvent], drivers: Sequence[str]) -> pd.DataFrame:
    """For each driver: n events used, least-squares gain through the origin (rise ~ gain * driver), Pearson r and
    the residual sd. r is None with fewer than RESPONSE_MIN_EVENTS usable events or no spread."""
    by_id = {st.storm_id: st for st in storms}
    min_n = int(t("RESPONSE_MIN_EVENTS"))
    out = []
    use = resp[resp["rise"].notna()] if resp is not None and len(resp) else pd.DataFrame(columns=RESP_COLUMNS)
    ids = use["storm_id"].tolist()
    y_all = use["rise"].to_numpy(float)
    for d in drivers:
        x = np.array([driver_value(by_id[sid], d) if sid in by_id else np.nan for sid in ids], dtype=float)
        y = y_all
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        n = int(len(x))
        sxx = float((x * x).sum())
        gain = float((x * y).sum() / sxx) if n and sxx > 0 else None
        r: Optional[float] = None
        if n >= min_n and float(np.std(x)) > 0 and float(np.std(y)) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
        sd = float(np.std(y - gain * x, ddof=1)) if gain is not None and n >= 2 else None
        out.append({"driver": d, "n": n, "gain": gain, "r": r, "resid_sd": sd})
    return pd.DataFrame(out, columns=["driver", "n", "gain", "r", "resid_sd"])


def unmeasured_fit(zone_id: str, sensor_id: str = "") -> ResponseFit:
    """The fit of a zone with no humidity data: driver "none", gain None, not significant (unmeasured, not zero)."""
    return ResponseFit(zone_id=zone_id, sensor_id=sensor_id, driver="none", n_events=0, gain=None, r=None, lag_h=None,
                       resid_sd=None, onset_ts=None, significant=False)


def fit_response(zone_id: str, sensor_id: str, resp: pd.DataFrame, storms: Sequence[StormEvent],
                 drivers: Sequence[str]) -> ResponseFit:
    """Pick the driver with the highest Pearson r across storm rises. significant = r >= RESPONSE_MIN_R and at least
    RESPONSE_MIN_EVENTS storms with rise >= RESPONSE_MIN_RISE. onset_ts = start of the first such storm. No data
    gives driver "none" with gain None (unmeasured, not zero)."""
    if resp is None or len(resp) == 0 or resp["rise"].notna().sum() == 0:
        return unmeasured_fit(zone_id, sensor_id)
    min_rise = float(t("RESPONSE_MIN_RISE"))
    r_sorted = resp.sort_values("start")
    events = r_sorted[r_sorted["rise"] >= min_rise]
    n_events = int(len(events))
    onset = str(events["start"].iloc[0]) if n_events else None
    lag = float(np.median(events["lag_h"].to_numpy(float))) if n_events else None
    # change point: a response that starts mid-record (a drain blocks, a joint opens) is fit from its first response
    # on; the storms before it are the zone's quiet baseline, not evidence against the driver
    used = r_sorted[r_sorted["start"] >= onset] if onset is not None else r_sorted
    stats = driver_stats(used, storms, drivers)
    cand = stats[stats["r"].notna()]
    if cand.empty:
        return ResponseFit(zone_id=zone_id, sensor_id=sensor_id, driver="none", n_events=n_events, gain=None, r=None,
                           lag_h=lag, resid_sd=None, onset_ts=onset, significant=False)
    best = cand.loc[cand["r"].astype(float).idxmax()]
    r = float(best["r"])
    significant = r >= float(t("RESPONSE_MIN_R")) and n_events >= int(t("RESPONSE_MIN_EVENTS"))
    return ResponseFit(zone_id=zone_id, sensor_id=sensor_id, driver=str(best["driver"]), n_events=n_events,
                       gain=round(float(best["gain"]), 5) if best["gain"] is not None else None, r=round(r, 3),
                       lag_h=round(lag, 1) if lag is not None else None,
                       resid_sd=round(float(best["resid_sd"]), 3) if best["resid_sd"] is not None else None,
                       onset_ts=onset, significant=bool(significant))


def all_responses(data: BuildingData, storms: Sequence[StormEvent], now: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
    """storm_responses for every humidity sensor, keyed by sensor_id (data after `now` is ignored)."""
    cut = _utc(now) if now is not None else None
    return {s.sensor_id: storm_responses(_series(data, s.sensor_id, cut), storms)
            for s in data.building.sensors if s.type == "humidity"}


def fit_all(data: BuildingData, storms: Sequence[StormEvent],
            responses: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, ResponseFit]:
    """One fit per humidity-sensored zone, keyed by zone_id. With several humidity sensors in a zone, the
    significant fit with the highest r wins."""
    b = data.building
    responses = responses if responses is not None else all_responses(data, storms)
    drivers = drivers_for(b)
    out: Dict[str, ResponseFit] = {}
    for s in b.sensors:
        if s.type != "humidity":
            continue
        resp = responses.get(s.sensor_id)
        if resp is None:
            resp = storm_responses(_series(data, s.sensor_id), storms)
        fit = fit_response(s.zone_id, s.sensor_id, resp, storms, drivers)
        prev = out.get(s.zone_id)
        if prev is None or (fit.significant, fit.r if fit.r is not None else -2.0) > (prev.significant, prev.r if prev.r is not None else -2.0):
            out[s.zone_id] = fit
    return out


# ----------------------------------------------------------------------------------------- moisture grading

@dataclass(frozen=True)
class GradedReading:
    """A graded sensor reading with the time its condition started (for observations and escalation clocks)."""

    finding: Finding
    ts: str
    zone_id: str
    sensor_id: str
    value: Optional[float]
    unit: str
    end_ts: Optional[str] = None  # when the condition ended (dried back); None = ongoing at `now`


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """(first, last) positions of each run of True."""
    out: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _temperature(data: BuildingData, zone_id: str, lo: pd.Timestamp, hi: pd.Timestamp) -> Tuple[Optional[float], Optional[str]]:
    """Mean temperature in the window from the zone's temperature sensor, else one on the same floor."""
    b = data.building
    z = b.zone(zone_id)
    cands = [s for s in b.sensors_in(zone_id) if s.type == "temperature"]
    if not cands and z is not None:
        cands = [s for s in b.sensors if s.type == "temperature" and (b.zone(s.zone_id) is not None and b.zone(s.zone_id).floor_id == z.floor_id)]
    for s in cands:
        ser = _series(data, s.sensor_id)
        ser = ser[(ser.index > lo) & (ser.index <= hi)].dropna()
        if len(ser):
            return float(ser.mean()), s.sensor_id
    return None, None


def grade_moisture(data: BuildingData, now: datetime, storms: Optional[Sequence[StormEvent]] = None,
                   responses: Optional[Dict[str, pd.DataFrame]] = None) -> List[GradedReading]:
    """Deterministic interior_water.json rows per humidity and leak sensor over the ASHRAE160_WINDOW_D window up to now.

    Humidity: rh_hourly (latest reading), rh_sustained_h (current run at or above EPA_RH_MAX), rh_30d_mean (with the
    temperature check), wet_duration_h (latest storm response not back under pre + RESPONSE_MIN_RISE / 2).
    Leak: leak_sensor (latest reading wet) and wet_duration_h (current wet run). No data in the window gives a U row."""
    b = data.building
    rub = _rubric()
    now_ts = _utc(now)
    win_d = int(t("ASHRAE160_WINDOW_D"))
    lo = now_ts - pd.Timedelta(days=win_d)
    rh_max = float(t("EPA_RH_MAX"))
    t_lo, t_hi = t("ASHRAE160_T_RANGE_C")
    storms = list(storms) if storms is not None else detect_storms(data.weather[data.weather.index <= now_ts] if len(data.weather) else data.weather, b)
    out: List[GradedReading] = []

    def add(row: dict, sid: str, zone: str, family: str, value: Optional[float], unit: str, ts: Any, why: str,
            flags: Sequence[str] = (), confidence: float = 1.0, end: Any = None) -> None:
        f = finding_from_row(row, rub, finding_id=f"water:{sid}:{family}", asset_class="interior_zone", modality="sensor",
                             evidence=Evidence(signal_id=sid), justification=why, flags=flags, confidence=confidence)
        out.append(GradedReading(finding=f, ts=_iso(ts), zone_id=zone, sensor_id=sid, value=value, unit=unit,
                                 end_ts=_iso(end) if end is not None else None))

    for s in b.sensors:
        if s.type not in ("humidity", "leak"):
            continue
        ser = _series(data, s.sensor_id, now_ts)
        win = ser[ser.index > lo].dropna()
        if win.empty:
            add(_row(rub, "U"), s.sensor_id, s.zone_id, "not_assessable", None, s.unit, now_ts,
                f"{s.sensor_id}: no data in the {win_d}-day window before {_iso(now_ts)} (ASHRAE160_WINDOW_D). "
                "Unmeasured, not dry.", flags=("not_measurable",), confidence=0.0)
            continue
        last_ts, last = win.index[-1], float(win.iloc[-1])
        if s.type == "leak":
            row = find_band_row(rub, "leak_sensor", last)
            if row is None:
                continue  # dry reading: interior_water.json has no leak-dry row, so nothing is graded (not S0)
            wet = (ser.to_numpy(float) >= 1.0)
            runs = _runs(wet)
            first = ser.index[runs[-1][0]] if runs else last_ts
            add(row, s.sensor_id, s.zone_id, "leak_sensor", last, s.unit, first,
                f"{s.sensor_id} reads wet ({last:g}) at {_iso(last_ts)}; wet since {_iso(first)}.")
            wet_h = _hours(first, now_ts)
            row2 = find_band_row(rub, "wet_duration_h", wet_h)
            if row2 is not None:
                add(row2, s.sensor_id, s.zone_id, "wet_duration_h", round(wet_h, 1), "h", first,  # ts = when wetness began
                    f"{s.sensor_id} wet for {wet_h:.0f} h since {_iso(first)}; "
                    f"EPA drying window {threshold_label('EPA_DRY_WINDOW_H')}.")
            continue
        # humidity
        row = find_band_row(rub, "rh_hourly", last)
        if row is not None:
            add(row, s.sensor_id, s.zone_id, "rh_hourly", last, s.unit, last_ts,
                f"{s.sensor_id} latest {last:.1f} %RH at {_iso(last_ts)}, below {threshold_label('EPA_RH_MAX')}.")
        above = win.to_numpy(float) >= rh_max
        if above[-1]:
            runs = _runs(above)
            first = win.index[runs[-1][0]]
            run_h = _hours(first, last_ts) + 1.0
            row = find_band_row(rub, "rh_sustained_h", run_h)
            if row is not None:
                add(row, s.sensor_id, s.zone_id, "rh_sustained_h", round(run_h, 1), "h",
                    first + pd.Timedelta(hours=float(t("RH_SUSTAINED_H"))),
                    f"{s.sensor_id} at or above {rh_max:g} %RH for {run_h:.0f} h since {_iso(first)} "
                    f"(RH_SUSTAINED_H {threshold_label('RH_SUSTAINED_H')}).")
        mean30 = float(win.mean())
        row = find_band_row(rub, "rh_30d_mean", mean30)
        if row is not None:
            temp, tsid = _temperature(data, s.zone_id, lo, now_ts)
            if temp is None:
                add(_row(rub, "U"), s.sensor_id, s.zone_id, "not_assessable", round(mean30, 1), s.unit, now_ts,
                    f"{s.sensor_id} {win_d}-day mean {mean30:.1f} %RH, but no temperature reading on the floor to check "
                    f"the {t_lo:g}-{t_hi:g} degC condition of ASHRAE160_RH_30D.", flags=("not_measurable",), confidence=0.0)
            elif t_lo <= temp <= t_hi:
                add(row, s.sensor_id, s.zone_id, "rh_30d_mean", round(mean30, 1), s.unit, now_ts,
                    f"{s.sensor_id} {win_d}-day mean {mean30:.1f} %RH with {tsid} mean {temp:.1f} degC "
                    f"({threshold_label('ASHRAE160_RH_30D')}).")
        # the latest storm response in the window that has not dried back
        resp = responses.get(s.sensor_id) if responses is not None else None
        if resp is None:
            resp = storm_responses(ser, storms)
        ev = resp[(resp["rise"] >= float(t("RESPONSE_MIN_RISE"))) & resp["post_ts"].notna()]
        ev = ev[[(_utc(x) > lo) for x in ev["post_ts"]]] if len(ev) else ev
        if len(ev):
            e = ev.iloc[-1]
            st_end = next((_utc(st.end) for st in storms if st.storm_id == e["storm_id"]), _utc(e["post_ts"]))
            level_back = float(e["pre"]) + float(t("RESPONSE_MIN_RISE")) / 2.0
            after = ser[ser.index > _utc(e["post_ts"])].dropna()
            back = after[after < level_back]
            dry_ts = back.index[0] if len(back) else now_ts
            wet_h = _hours(st_end, dry_ts)
            row = find_band_row(rub, "wet_duration_h", wet_h)
            if row is not None:
                dried = len(back) > 0
                add(row, s.sensor_id, s.zone_id, "wet_duration_h", round(wet_h, 1), "h", st_end,  # ts = when wetness began
                    f"{s.sensor_id} stayed above pre-storm {float(e['pre']):.1f} + {float(t('RESPONSE_MIN_RISE')) / 2:g} %RH "
                    f"for {wet_h:.0f} h after storm {e['storm_id']} ended ({threshold_label('EPA_DRY_WINDOW_H')})"
                    + (f"; back under at {_iso(dry_ts)}." if dried else "; still above now."),
                    end=dry_ts if dried else None)
    return out


def moisture_findings(data: BuildingData, now: datetime) -> List[Finding]:
    """interior_water.json findings per sensor (asset_class interior_zone, modality sensor, evidence.signal_id =
    sensor_id). A sensor with no data in the window gives a U finding, never "RH ok"."""
    return [g.finding for g in grade_moisture(data, now)]


# ----------------------------------------------------------------------------------------- observations

def water_observations(data: BuildingData, storms: Sequence[StormEvent], fits: Dict[str, ResponseFit],
                       findings: Sequence[Finding], *, responses: Optional[Dict[str, pd.DataFrame]] = None,
                       graded: Optional[Sequence[GradedReading]] = None, now: Optional[datetime] = None) -> List[Observation]:
    """One sensor_event per (humidity sensor, storm) with rise >= RESPONSE_MIN_RISE, one per leak-sensor wet run, and
    one per non-S0 finding (U included, as U). Storm events carry the EVENT_LEVEL indicator level (no rubric row
    grades them). A wet run still wet at `now` carries the interior_water.json 'Active leak' finding. A run that
    ended is history (end_ts set): 'Drying window exceeded' when it lasted past the EPA drying window, else
    ungraded (level None, 'verify drying'). Series are cut at `now`."""
    b = data.building
    cut = _utc(now) if now is not None else None
    rub = _rubric()
    lv = _th("EVENT_LEVEL")
    min_rise = float(t("RESPONSE_MIN_RISE"))
    responses = responses if responses is not None else all_responses(data, storms)
    out: List[Observation] = []
    by_sensor = {s.sensor_id: s for s in b.sensors}
    storm_by_id = {st.storm_id: st for st in storms}
    for s in b.sensors:
        if s.type != "humidity":
            continue
        resp = responses.get(s.sensor_id)
        if resp is None or len(resp) == 0:
            continue
        fit = fits.get(s.zone_id)
        repeated = fit is not None and fit.significant and fit.sensor_id == s.sensor_id
        level = lv["repeated"] if repeated else lv["single"]
        for e in resp[resp["rise"] >= min_rise].itertuples(index=False):
            st = storm_by_id.get(e.storm_id)
            drv = f"; zone fit driver {fit.driver}, r {fit.r}" if repeated and fit is not None else ""
            out.append(Observation(
                obs_id=f"ev:{s.sensor_id}:{e.storm_id}", kind="sensor_event", ts=str(e.post_ts), zone_id=s.zone_id,
                source=f"sensor:{s.sensor_id}", level=level, value=float(e.rise), unit="%RH rise",
                text=(f"{s.sensor_id} rose {e.rise:.1f} %RH after storm {e.storm_id}"
                      f"{f' ({st.total_mm:g} mm)' if st else ''}, peak at {e.post_ts}{drv}. "
                      f"Indicator level {level} per EVENT_LEVEL [team-proposed, validate]."),
                synthetic=bool(s.synthetic or b.synthetic)))
    leak_row = next(r for r in rub["rows"] if r["family"] == "leak_sensor")
    for s in b.sensors:
        if s.type != "leak":
            continue
        ser = _series(data, s.sensor_id, cut).dropna()
        if ser.empty:
            continue
        syn = bool(s.synthetic or b.synthetic)
        for a, z in _runs(ser.to_numpy(float) >= 1.0):
            start, end = ser.index[a], ser.index[z]
            oid = f"leak:{s.sensor_id}:{_iso(start)}"
            if z == len(ser) - 1:  # still wet at the last reading up to now: an active leak
                f = finding_from_row(leak_row, rub, finding_id=f"water:{s.sensor_id}:leak:{_iso(start)}", asset_class="interior_zone",
                                     modality="sensor", evidence=Evidence(signal_id=s.sensor_id),
                                     justification=f"{s.sensor_id} read wet from {_iso(start)} to now.")
                out.append(Observation(obs_id=oid, kind="sensor_event", ts=_iso(start), zone_id=s.zone_id,
                                       source=f"sensor:{s.sensor_id}", level=f.unified.level, value=1.0, unit="wet",
                                       text=f.justification, finding=f, synthetic=syn))
                continue
            dry_at = ser.index[z + 1]
            wet_h = _hours(start, dry_at)
            row = find_band_row(rub, "wet_duration_h", wet_h)
            why = (f"{s.sensor_id} read wet from {_iso(start)} to {_iso(end)} and was dry at {_iso(dry_at)} ({wet_h:.0f} h wet). "
                   "Wet run ended: history, verify drying.")
            f = None if row is None else finding_from_row(row, rub, finding_id=f"water:{s.sensor_id}:wetrun:{_iso(start)}",
                                                          asset_class="interior_zone", modality="sensor",
                                                          evidence=Evidence(signal_id=s.sensor_id),
                                                          justification=why + f" EPA drying window {threshold_label('EPA_DRY_WINDOW_H')}.")
            out.append(Observation(obs_id=oid, kind="sensor_event", ts=_iso(start), zone_id=s.zone_id,
                                   source=f"sensor:{s.sensor_id}", level=f.unified.level if f is not None else None,
                                   value=round(wet_h, 1), unit="h wet (ended)", text=f.justification if f is not None else why,
                                   finding=f, synthetic=syn, end_ts=_iso(dry_at)))
    meta = {g.finding.finding_id: g for g in (graded or [])}
    last_ts = _iso(data.sensors.index.max()) if data.sensors is not None and len(data.sensors) else None
    for f in findings:
        if f.unified.level == "S0" or f.defect_type == "leak_sensor":
            continue  # S0 is context only; a wet leak reading is already its wet-run observation
        g = meta.get(f.finding_id)
        sid = f.evidence.signal_id or ""
        sensor = by_sensor.get(sid)
        zone = g.zone_id if g else (sensor.zone_id if sensor else None)
        ts = g.ts if g else last_ts
        if zone is None or ts is None:
            continue
        out.append(Observation(obs_id=f"f:{f.finding_id}", kind="sensor_event", ts=ts, zone_id=zone, source=f"sensor:{sid}",
                               level=f.unified.level, value=g.value if g else None, unit=g.unit if g else None,
                               text=f.justification, finding=f, synthetic=bool(sensor.synthetic if sensor else False) or b.synthetic,
                               end_ts=g.end_ts if g else None))
    return out


# ----------------------------------------------------------------------------------------- risk

def zone_side(b: Building, zone_id: str) -> Optional[str]:
    """A facade drop's orientation; for a room, the orientation of the facade drop that drains into it; "roof" for a
    roof zone; None for interior zones."""
    z = b.zone(zone_id)
    if z is None:
        return None
    if z.kind == "facade_drop":
        return z.orientation
    if z.kind == "roof":
        return "roof"
    for e in b.edges:
        if e.dst == zone_id and e.kind == "drains_to":
            src = b.zone(e.src)
            if src is not None and src.kind == "facade_drop":
                return src.orientation
    return None


def rh_trends(data: BuildingData, now: datetime) -> Dict[str, float]:
    """Slope of daily mean RH (%RH per day) over the last RH_TREND_DAYS days, per zone (the steepest of its humidity
    sensors). Zones without at least 3 daily means are left out (unmeasured)."""
    now_ts = _utc(now)
    days = int(t("RH_TREND_DAYS"))
    out: Dict[str, float] = {}
    for s in data.building.sensors:
        if s.type != "humidity":
            continue
        ser = _series(data, s.sensor_id, now_ts)
        ser = ser[ser.index > now_ts - pd.Timedelta(days=days)].dropna()
        if ser.empty:
            continue
        daily = ser.resample("D").mean().dropna()
        if len(daily) < 3:
            continue
        slope = float(np.polyfit(np.arange(len(daily), dtype=float), daily.to_numpy(float), 1)[0])
        out[s.zone_id] = max(slope, out.get(s.zone_id, -math.inf))
    return out


def _factor(name: str, value: Optional[float], explanation: str, basis: str) -> Factor:
    w = float(t("RISK_WEIGHTS")[name])
    v = None if value is None else float(round(min(1.0, max(0.0, value)), 4))
    return Factor(name=name, value=v, weight=w, contribution=round(w * v, 4) if v is not None else 0.0,
                  explanation=explanation, basis=basis)


def zone_risk(b: Building, zone_id: str, fits: Dict[str, ResponseFit], observations: Sequence[Observation],
              storms: Sequence[StormEvent], now: datetime, *, trends: Optional[Dict[str, float]] = None) -> List[Factor]:
    """Additive water-risk factors for one zone (RISK_WEIGHTS), each 0-1 or None when not measured:
    exposure, upstream_defects, rain_response, trend, open_tickets."""
    now_ts = _utc(now)
    recent_d = int(_th("RECENT_WINDOW_D"))
    recent_lo = now_ts - pd.Timedelta(days=recent_d)
    factors: List[Factor] = []

    # exposure: annual WDR share of the zone's facade side
    side = zone_side(b, zone_id)
    exp_lo = now_ts - pd.Timedelta(days=int(_th("EXPOSURE_WINDOW_D")))
    year = [st for st in storms if exp_lo <= _utc(st.start) <= now_ts]
    basis = f"{RISK_BASIS}; WDR_PROXY [team-proposed, validate]; EXPOSURE_WINDOW_D [team-proposed, validate]"
    if side is None:
        factors.append(_factor("exposure", 0.0, "interior zone: no facade side, so no direct wind-driven rain", basis))
    elif not year:
        factors.append(_factor("exposure", None, "not measured: no storms on record in the exposure window", basis))
    elif side == "roof":
        factors.append(_factor("exposure", 1.0, f"roof zone: takes all rainfall ({len(year)} storms in the window)", basis))
    else:
        tot = sum(sum(st.wdr.values()) for st in year)
        mine = sum(st.wdr.get(side, 0.0) for st in year)
        share = mine / tot if tot > 0 else 0.0
        factors.append(_factor("exposure", share, f"{side} side gets {share:.0%} of the wind-driven rain of {len(year)} storms", basis))

    # upstream_defects: strongest path from an upstream node with a graded level >= S1 in the recent window
    worst: Dict[str, str] = {}
    rank = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
    for o in observations:
        node = o.element_id or o.zone_id
        if node is None or o.level not in rank or not (recent_lo <= _utc(o.ts) <= now_ts):
            continue
        if node not in worst or rank[o.level] > rank[worst[node]]:
            worst[node] = o.level
    best, best_node = 0.0, None
    if b.zone(zone_id) is not None:
        for n, r in upstream(b, zone_id).items():
            if n == zone_id or n not in worst:
                continue
            v = r.score * SEVERITY_WEIGHT[worst[n]] / SEVERITY_WEIGHT["S4"]
            if v > best:
                best, best_node = v, n
    factors.append(_factor("upstream_defects", best,
                           (f"{best_node} ({worst[best_node]}) upstream, path score x level weight / {SEVERITY_WEIGHT['S4']:g}" if best_node
                            else f"no graded observation (S1 or worse) upstream in the last {recent_d} days"),
                           f"{RISK_BASIS}; EDGE_WEIGHT [team-proposed, validate]; RECENT_WINDOW_D [team-proposed, validate]"))

    # rain_response: expected rise for a median storm of the fitted driver, scaled by RESPONSE_MIN_RISE * RESPONSE_SCALE_X
    fit = fits.get(zone_id) or next((f for f in fits.values() if f.zone_id == zone_id), None)
    scale = float(t("RESPONSE_MIN_RISE")) * float(_th("RESPONSE_SCALE_X"))
    basis = f"{RISK_BASIS}; RESPONSE_MIN_R, RESPONSE_MIN_EVENTS, RESPONSE_MIN_RISE [team-proposed, validate]"
    if fit is not None and fit.r is None and (fit.n_events > 0 or fit.onset_ts is not None or fit.lag_h is not None):
        factors.append(_factor("rain_response", None,
                               f"not measured: fewer than RESPONSE_MIN_EVENTS ({int(t('RESPONSE_MIN_EVENTS'))}) usable storms or no "
                               f"spread to fit (n={fit.n_events}); the sensor has data (unmeasured, not zero)", basis))
    elif fit is None or fit.r is None:
        factors.append(_factor("rain_response", None, "not measured: no humidity sensor data for this zone (unmeasured, not zero)", basis))
    elif not fit.significant or fit.gain is None:
        factors.append(_factor("rain_response", 0.0, f"measured, no significant response (best {fit.driver}, r {fit.r}, {fit.n_events} events)", basis))
    else:
        xs = [driver_value(st, fit.driver) for st in storms]
        xs = [x for x in xs if np.isfinite(x) and x > 0]
        med = float(np.median(xs)) if xs else 0.0
        exp_rise = fit.gain * med
        factors.append(_factor("rain_response", exp_rise / scale,
                               f"significant {fit.driver} response (r {fit.r}, {fit.n_events} events): about {exp_rise:.1f} %RH "
                               f"for a median storm", basis))

    # trend: RH_TREND_DAYS slope of daily mean RH
    slope = (trends or {}).get(zone_id)
    basis = f"{RISK_BASIS}; RH_TREND_DAYS, RESPONSE_MIN_RISE [team-proposed, validate]"
    if slope is None:
        factors.append(_factor("trend", None, "not measured: no humidity trend for this zone", basis))
    else:
        days = int(t("RH_TREND_DAYS"))
        factors.append(_factor("trend", slope * days / float(t("RESPONSE_MIN_RISE")),
                               f"daily mean RH changes {slope:+.2f} %RH/day over {days} days", basis))

    # open_tickets: open (or status unknown) water tickets in the zone in the recent window; closed ones do not count
    n = sum(1 for o in observations if o.kind == "ticket" and o.zone_id == zone_id and recent_lo <= _utc(o.ts) <= now_ts
            and (o.category or ticket_category(o.text)) == "water" and (o.status or "") not in CLOSED_TICKET_STATUSES)
    factors.append(_factor("open_tickets", n / float(_th("TICKET_SATURATION")),
                           f"{n} open water tickets in the last {recent_d} days (closed tickets not counted; no status = open)",
                           f"{RISK_BASIS}; TICKET_SATURATION, RECENT_WINDOW_D [team-proposed, validate]"))
    return factors


def risk_score(factors: Sequence[Factor]) -> Optional[float]:
    """Sum of contributions; None when every factor value is None (not measured, never 0 as healthy)."""
    vals = [f.contribution for f in factors if f.value is not None]
    return round(sum(vals), 4) if vals else None


def risk_grid(b: Building, risks: Dict[str, List[Factor]]) -> pd.DataFrame:
    """Rooms with a facade side: floor, level, side, zone_id, risk (NaN = not measured), top_factor, top_basis."""
    rows = []
    for z in b.zones:
        if z.kind != "room" or z.zone_id not in risks:
            continue
        side = zone_side(b, z.zone_id)
        if side is None or side == "roof":
            continue
        fs = risks[z.zone_id]
        score = risk_score(fs)
        measured = [f for f in fs if f.value is not None]
        top = max(measured, key=lambda f: f.contribution) if measured else None
        fl = b.floor(z.floor_id)
        rows.append({"floor": z.floor_id, "level": fl.level if fl else None, "side": side, "zone_id": z.zone_id,
                     "risk": float("nan") if score is None else score,
                     "top_factor": f"{top.name}: {top.explanation}" if top and top.contribution > 0 else ("not measured" if top is None else "none above 0"),
                     "top_basis": top.basis if top else "", "measured": score is not None})
    return pd.DataFrame(rows, columns=GRID_COLUMNS)


# ----------------------------------------------------------------------------------------- forecast

def storm_driver(driver: str, *, rain_mm: float, wind_dir_deg: float, wind_speed_ms: float) -> float:
    """A hypothetical storm's value of a driver, with the same formulas as detect_storms."""
    if driver == "rain":
        return float(rain_mm)
    if driver.startswith("wdr_") and driver[4:] in AZIMUTH:
        return float(rain_mm * wind_speed_ms * max(0.0, math.cos(math.radians(wind_dir_deg - AZIMUTH[driver[4:]]))))
    return float("nan")


def _hours_to(current: float, rise: float, thr: float, duration_h: float, lag_h: float) -> Optional[float]:
    if rise <= 0 or current + rise < thr:
        return None
    return round(max(0.0, duration_h / 2.0 + lag_h * (thr - current) / rise), 1)


def forecast_hours_band(fc: ZoneForecast, fit: ResponseFit, duration_h: float) -> Tuple[Optional[float], Optional[float]]:
    """(earliest, latest) hours to the threshold from the peak band: the hi peak crosses soonest, the lo peak latest.
    latest is None when the lo peak stays under the threshold."""
    thr = float(t(fc.threshold_key))
    lag = fit.lag_h or 0.0
    return (_hours_to(fc.current, fc.hi - fc.current, thr, duration_h, lag),
            _hours_to(fc.current, fc.lo - fc.current, thr, duration_h, lag))


def forecast(b: Building, data: BuildingData, fits: Dict[str, ResponseFit], *, rain_mm: float, duration_h: float,
             wind_dir_deg: float, wind_speed_ms: float, threshold_key: str = "EPA_RH_MAX") -> List[ZoneForecast]:
    """For each significant fit: rise = gain * driver of the hypothetical storm; current = median of the last
    RESPONSE_PRE_H hours; peak = current + rise; band = +/- FORECAST_Z * resid_sd. hours_to_threshold =
    duration_h / 2 + lag_h * (thr - current) / rise when the peak reaches the threshold, else None."""
    thr = float(t(threshold_key))
    z = float(t("FORECAST_Z"))
    pre_h = float(t("RESPONSE_PRE_H"))
    out: List[ZoneForecast] = []
    for key in sorted(fits):
        f = fits[key]
        if not f.significant or f.gain is None:
            continue
        ser = _series(data, f.sensor_id).dropna()
        if ser.empty:
            continue
        cur = float(ser[ser.index > ser.index[-1] - pd.Timedelta(hours=pre_h)].median())
        x = storm_driver(f.driver, rain_mm=rain_mm, wind_dir_deg=wind_dir_deg, wind_speed_ms=wind_speed_ms)
        rise = f.gain * x if np.isfinite(x) else 0.0
        peak = cur + rise
        band = z * (f.resid_sd or 0.0)
        hrs = _hours_to(cur, rise, thr, duration_h, f.lag_h or 0.0)
        out.append(ZoneForecast(zone_id=f.zone_id, current=round(cur, 1), peak=round(peak, 1), lo=round(peak - band, 1),
                                hi=round(peak + band, 1), hours_to_threshold=hrs, crosses=peak >= thr, driver=f.driver,
                                threshold_key=threshold_key))
    out.sort(key=lambda fc: (not fc.crosses, fc.hours_to_threshold if fc.hours_to_threshold is not None else math.inf, fc.zone_id))
    return out


# ----------------------------------------------------------------------------------------- end to end

def analyze_water(data: BuildingData, now: datetime) -> WaterResult:
    """Storms -> responses -> fits -> moisture findings -> observations -> zone risks. Data after `now` is ignored."""
    b = data.building
    now_ts = _utc(now)
    weather = data.weather[data.weather.index <= now_ts] if data.weather is not None and len(data.weather) else data.weather
    storms = detect_storms(weather, b)
    responses = all_responses(data, storms, now_ts)
    fits = fit_all(data, storms, responses=responses)
    graded = grade_moisture(data, now_ts, storms=storms, responses=responses)
    findings = [g.finding for g in graded]
    obs = water_observations(data, storms, fits, findings, responses=responses, graded=graded, now=now_ts)
    obs = [o for o in obs if _utc(o.ts) <= now_ts]
    context = obs + tickets_to_observations(b, data.tickets)
    trends = rh_trends(data, now_ts)
    risks = {z.zone_id: zone_risk(b, z.zone_id, fits, context, storms, now_ts, trends=trends) for z in b.zones}
    return WaterResult(storms=storms, fits=fits, risks=risks, observations=obs, findings=findings)
