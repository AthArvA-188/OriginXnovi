"""12-month LA mitigation calendar for common areas.

Rows combine NOAA 1991-2020 monthly normals (LA Downtown USC and LAX, REAL), day length / sunrise / sunset on the
15th from the NOAA GML general solar position equations (computed), LADWP A-2 and SCE seasons, the LADWP DR season,
US federal holidays (pandas) and DST changes (IANA tz database), plus Open-Meteo 2025 REANALYSIS monthly temperature.
Action text is a recommendation built from those rows; every action stays a proposal for a human to approve.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import tariff
from .data import LA_LAT, LA_LON, LA_TZ
from .rules import life_safety, load_rules

USC, LAX = "USW00093134", "USW00023174"
GML_URL = "https://gml.noaa.gov/grad/solcalc/solareqns.PDF"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _span(a: int, b: int) -> str:
    return f"{a:02d}:00-{b:02d}:00"


def _windows() -> Dict[str, object]:
    """Hour windows and thresholds used in the action text, read from the tariff and rule tables (never retyped)."""
    a = tariff.LADWP_A2B["periods_weekday"]
    return {"high_peak": a["high_peak"], "low_peak": a["low_peak"], "sce_on_peak": tariff.SCE_TOU["on_peak_weekday"],
            "flex": tariff.FLEX_ALERT["window"], "dr": tariff.LADWP_DR["window"],
            "hp_txt": ", ".join(_span(x, y) for x, y in a["high_peak"]),
            "hp_start": min(x for x, _ in a["high_peak"]),
            "dr_pct": 100 * float(load_rules()["demand_response"]["min_reduction_frac"]),
            "hold_min": life_safety("EGRESS_MOTION_HOLD_MIN"), "egress_fc": life_safety("EGRESS_FLOOR_FC")}


def solar_day(d: date, lat: float = LA_LAT, lon: float = LA_LON) -> Dict[str, float]:
    """NOAA GML equations: sunrise/sunset in minutes after 00:00 UTC and day length in hours (zenith 90.833 deg)."""
    doy = d.timetuple().tm_yday
    days_in_year = 366 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)) else 365
    g = 2 * math.pi / days_in_year * (doy - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                       - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    la = math.radians(lat)
    cos_ha = math.cos(math.radians(90.833)) / (math.cos(la) * math.cos(decl)) - math.tan(la) * math.tan(decl)
    ha = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
    sunrise = 720 - 4 * (lon + ha) - eqtime
    sunset = 720 - 4 * (lon - ha) - eqtime
    return {"sunrise_utc_min": sunrise, "sunset_utc_min": sunset, "day_length_h": 8 * ha / 60.0}


def _local_hm(d: date, utc_min: float, tz: str = LA_TZ) -> str:
    t = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(minutes=utc_min)
    return t.astimezone(ZoneInfo(tz)).strftime("%H:%M %Z")


def _local_hour(d: date, utc_min: float, tz: str = LA_TZ) -> float:
    t = (datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(minutes=utc_min)).astimezone(ZoneInfo(tz))
    return t.hour + t.minute / 60.0


def dst_changes(year: int, tz: str = LA_TZ) -> List[Dict[str, str]]:
    z = ZoneInfo(tz)
    out = []
    prev = datetime(year, 1, 1, 12, tzinfo=z).utcoffset()
    d = date(year, 1, 2)
    while d.year == year:
        off = datetime(d.year, d.month, d.day, 12, tzinfo=z).utcoffset()
        if off != prev:
            out.append({"date": str(d), "change": "starts (clocks forward)" if off > prev else "ends (clocks back)"})
            prev = off
        d += timedelta(days=1)
    return out


def holidays(year: int) -> List[Dict[str, str]]:
    cal = USFederalHolidayCalendar()
    h = cal.holidays(f"{year}-01-01", f"{year}-12-31", return_name=True)
    return [{"date": str(k.date()), "name": v} for k, v in h.items()]


def _dr_days_in_month(year: int, m: int) -> int:
    a, b = tariff.LADWP_DR["season"]
    lo, hi = pd.Timestamp(f"{year}-{a}"), pd.Timestamp(f"{year}-{b}")
    days = pd.date_range(f"{year}-{m:02d}-01", periods=pd.Timestamp(f"{year}-{m:02d}-01").days_in_month, freq="D")
    return int(((days >= lo) & (days <= hi)).sum())


def build_calendar(year: int, noaa: pd.DataFrame, weather: Optional[pd.DataFrame] = None) -> List[Dict]:
    noaa = noaa.copy()
    noaa["m"] = noaa["DATE"].astype(str).str.strip().astype(int)
    hol = holidays(year)
    dst = dst_changes(year)
    wx_m = None
    if weather is not None:
        t = weather["temperature_2m"]
        wx_m = pd.DataFrame({"mean": t.groupby(t.index.month).mean(), "max": t.groupby(t.index.month).max()})
    rows = []
    for m in range(1, 13):
        usc = noaa[(noaa["STATION"] == USC) & (noaa["m"] == m)].iloc[0]
        lax = noaa[(noaa["STATION"] == LAX) & (noaa["m"] == m)].iloc[0]
        d15 = date(year, m, 15)
        sd = solar_day(d15)
        row = {
            "year": year, "month": m, "name": MONTHS[m - 1],
            "cdd65_usc": float(usc["MLY-CLDD-NORMAL"]), "hdd65_usc": float(usc["MLY-HTDD-NORMAL"]), "tavg_f_usc": float(usc["MLY-TAVG-NORMAL"]),
            "cdd65_lax": float(lax["MLY-CLDD-NORMAL"]), "hdd65_lax": float(lax["MLY-HTDD-NORMAL"]),
            "day_length_h": round(sd["day_length_h"], 2),
            "sunrise_local": _local_hm(d15, sd["sunrise_utc_min"]), "sunset_local": _local_hm(d15, sd["sunset_utc_min"]),
            "sunrise_hour": _local_hour(d15, sd["sunrise_utc_min"]), "sunset_hour": _local_hour(d15, sd["sunset_utc_min"]),
            "ladwp_season": "high" if m in tariff.LADWP_A2B["high_season_months"] else "low",
            "sce_season": "summer" if m in tariff.SCE_TOU["summer_months"] else "winter",
            "ladwp_dr_days": _dr_days_in_month(year, m),
            "holidays": [h for h in hol if int(h["date"][5:7]) == m],
            "dst": [x for x in dst if int(x["date"][5:7]) == m],
        }
        if wx_m is not None and m in wx_m.index:
            row["reanalysis_2025_tmean_c"] = float(wx_m.loc[m, "mean"])
            row["reanalysis_2025_tmax_c"] = float(wx_m.loc[m, "max"])
        row["actions"] = month_actions(row, rows_cdd_rank=None)
        rows.append(row)
    ranks = pd.Series([r["cdd65_usc"] for r in rows]).rank(ascending=False, method="first").astype(int).tolist()
    for r, k in zip(rows, ranks):
        r["cdd_rank_usc"] = k
        r["actions"] = month_actions(r, rows_cdd_rank=k)
    return rows


def month_actions(r: Dict, rows_cdd_rank: Optional[int]) -> List[str]:
    a, w = tariff.LADWP_A2B, _windows()
    acts = []
    for x in r["dst"]:
        acts.append(f"DST {x['change']} on {x['date']}: re-anchor garage/perimeter lighting and any clock-based schedule to local time")
    if r["ladwp_season"] == "high":
        acts.append(f"LADWP high season: cap common-area kW on weekdays {w['hp_txt']} (high-peak demand ${a['demand_usd_per_kw']['high']['high_peak']:.2f}/kW)")
    if r["ladwp_dr_days"]:
        acts.append(f"LADWP DR season ({r['ladwp_dr_days']} days this month): keep the DR playbook ready - trim non-egress lighting >= {w['dr_pct']:.0f}%, never egress")
    if r["sce_season"] == "summer":
        acts.append(f"Watch {_span(*w['sce_on_peak'])} (SCE on-peak; CAISO Flex Alert window {_span(*w['flex'])}): propose switching off non-essential decorative and amenity lighting")
    if rows_cdd_rank is not None and rows_cdd_rank <= 3:
        acts.append(f"Top-3 cooling month at USC (CDD65 {r['cdd65_usc']:.1f}): propose lobby pre-cooling before {w['hp_start']:02d}:00 and occupied-standby setpoints in amenity rooms")
    if r["hdd65_usc"] >= 150:
        acts.append(f"Heating month (HDD65 {r['hdd65_usc']:.1f}): lobby morning warm-up; check door heaters and vestibules")
    if r["month"] == 4:
        acts.append("Commissioning month: audit sensor coverage and hold times on every egress path (sensor coverage, time delay and commissioning drive garage results)")
    if r["month"] == 5:
        acts.append(f"Enrol / confirm LADWP DR participation before the season starts ({tariff.LADWP_DR['season'][0]}); rehearse the playbook")
    if r["day_length_h"] < 10.5:
        acts.append(f"Short days ({r['day_length_h']:.1f} h, sunset {r['sunset_local']}): garage and perimeter lights come on earlier; keep sensor setbacks, not full-off, on egress")
    for h in r["holidays"]:
        acts.append(f"{h['date']} {h['name']}: holiday setback for amenity rooms (egress floors unchanged; priced as a normal weekday because the LADWP TOU page does not list holidays as Base)")
    return acts


def month_hour_modes(cal: List[Dict]) -> List[Dict]:
    """Recommended common-area mode for a typical weekday of each month, by hour (for the heatmap)."""
    w = _windows()
    inside = lambda h, spans: any(x <= h < y for x, y in spans)  # noqa: E731
    out = []
    for r in cal:
        for h in range(24):
            mid = h + 0.5
            dark = mid < r["sunrise_hour"] or mid > r["sunset_hour"]
            if r["ladwp_season"] == "high" and inside(h, w["high_peak"]):
                mode = "Peak cap + DR ready"
            elif r["sce_season"] == "summer" and inside(h, [w["sce_on_peak"], w["flex"]]):
                mode = "Evening grid-stress watch"
            elif inside(h, w["low_peak"]):
                mode = "Price shoulder (LADWP low peak)"
            elif dark:
                mode = f"Dark: sensor setback, egress >= {w['egress_fc']:g} fc"
            else:
                mode = "Daylight off-peak"
            ts = pd.Timestamp(int(r.get("year", 2027)), r["month"], 15, h)
            out.append({"month": r["name"], "month_num": r["month"], "hour": h, "mode": mode,
                        "ladwp_usd_kwh_weekday": float(tariff.energy_rate([ts])[0])})
    return out


def seasonal_plan(cal: List[Dict]) -> List[Dict]:
    """What the building manager does each season (built from the calendar rows and tariff tables)."""
    a, dr, w = tariff.LADWP_A2B, tariff.LADWP_DR, _windows()
    by = {r["month"]: r for r in cal}
    summer_cdd = sum(by[m]["cdd65_usc"] for m in (6, 7, 8, 9))
    year_cdd = sum(r["cdd65_usc"] for r in cal)
    return [
        {"season": "Winter (Dec-Feb)", "months": [12, 1, 2],
         "why": f"Shortest days ({min(by[m]['day_length_h'] for m in (12, 1, 2)):.1f} h); most heating (HDD65 {sum(by[m]['hdd65_usc'] for m in (12, 1, 2)):.0f} at USC)",
         "do": ["Schedule garage and perimeter lighting from sunset, not a fixed clock time",
                "Holiday setbacks in amenity rooms; egress floors never change",
                "Lobby morning warm-up only on working days"]},
        {"season": "Spring (Mar-May)", "months": [3, 4, 5],
         "why": "DST starts in March; mild weather is the window to fix controls before summer",
         "do": ["Re-anchor every clock-based schedule after the DST change",
                f"April: commissioning audit of motion-sensor coverage and hold times on all egress paths (NFPA 101 fail-safe, >= {w['hold_min']} min)",
                f"May: enrol in LADWP DR before {dr['season'][0]} (needs >= {dr['min_curtail_kw']} kW building-wide; common-area lighting is only a contribution)"]},
        {"season": "Summer (Jun-Sep)", "months": [6, 7, 8, 9],
         "why": f"LADWP high season; {summer_cdd:.0f} of {year_cdd:.0f} annual CDD65 at USC fall in Jun-Sep",
         "do": [f"Weekdays {w['hp_txt']}: cap common-area kW (high-peak demand ${a['demand_usd_per_kw']['high']['high_peak']:.2f}/kW vs ${a['demand_usd_per_kw']['low']['high_peak']:.2f}/kW in winter)",
                f"Approve lobby pre-cooling proposals before {w['hp_start']:02d}:00 on hot days",
                f"DR events ({dr['window'][0]}:00-{dr['window'][1]}:00, max {dr['max_events']} a season): trim restroom and amenity lighting >= {w['dr_pct']:.0f}%; never egress",
                f"Flex Alert days ({_span(*w['flex'])}): switch off decorative and non-essential amenity lighting"]},
        {"season": "Autumn (Oct-Nov)", "months": [10, 11],
         "why": f"Heat events can continue; LADWP DR season ends {dr['season'][1]}; DST ends in November",
         "do": ["Keep DR readiness until mid-October",
                "Re-anchor schedules after DST ends; garage lights come on earlier",
                "Review the year: sensor faults found, proposals approved/rejected, kWh vs last year"]},
    ]
