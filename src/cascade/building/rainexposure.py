"""Wind-driven rain (WDR) exposure of building facades from real hourly weather, per ISO 15927-3 method 1.

Physics (ISO 15927-3:2009 method 1, formulas as given in Blocken & Carmeliet 2010, Building and Environment,
eq. 5-9 and Tables 1-2; preprint http://www.urbanphysics.net/2010_BAE_BB_JC_2010_WDRcomp_review__Preprint.pdf,
accessed 2026-09-26):

    hourly airfield index   h(theta) = (2/9) * U10 * Rh^(8/9) * cos(D - theta)     for cos(D - theta) > 0, else 0
    airfield annual index   I_A(theta) = sum_h h(theta) / N_years                    (L/m2 per year)
    wall annual index       I_WA = I_A * C_R(z) * C_T * O * W
    roughness coefficient   C_R(z) = K_R * ln(max(z, z_min) / z0)                    (ISO Table 1)

U10 = wind speed at 10 m (m/s), Rh = hourly rain (mm), D = direction the wind blows FROM, theta = direction the wall
faces (outward normal, degrees clockwise from north). Wind speed enters to the power 1 (not 8/9).

The topography coefficient C_T and the wall factor W are NOT applied (ISO Figure 1 was not read); they stay 1 and every
output says so. Values are relative facade exposure at airfield terrain, not litres on a specific wall.

This module has no network access. It is used by scripts/build_la_rain.py (precompute) and by the website/building
layer through `facade_exposure()`, which reads the precomputed station curves in eval/rain/station_roses.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ROSES_PATH = REPO / "eval" / "rain" / "station_roses.json"

FACADES: Tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
AZ: Dict[str, float] = {f: 45.0 * i for i, f in enumerate(FACADES)}
KT_TO_MS = 0.514444
IN_TO_MM = 25.4
ISO_COEF = 2.0 / 9.0
ISO_R_EXP = 8.0 / 9.0

# Station metadata from the IEM CA_ASOS network GeoJSON (https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson,
# accessed 2026-09-26); GHCN-Daily ids from its 'ncei91' property. elev in metres.
STATIONS: Dict[str, dict] = {
    "LAX": {"name": "Los Angeles Intl", "lat": 33.9382, "lon": -118.3865, "elev": 32.0, "ghcnd": "USW00023174", "region": "Coast & basin"},
    "SMO": {"name": "Santa Monica", "lat": 34.0210, "lon": -118.4471, "elev": 53.0, "ghcnd": "USW00093197", "region": "Coast & basin"},
    "HHR": {"name": "Hawthorne", "lat": 33.9228, "lon": -118.3352, "elev": 19.0, "ghcnd": "USW00003167", "region": "Coast & basin"},
    "CQT": {"name": "Downtown LA / USC", "lat": 34.0235, "lon": -118.2912, "elev": 56.0, "ghcnd": "USW00093134", "region": "Coast & basin"},
    "LGB": {"name": "Long Beach", "lat": 33.8118, "lon": -118.1472, "elev": 12.0, "ghcnd": "USW00023129", "region": "Coast & basin"},
    "BUR": {"name": "Burbank", "lat": 34.2007, "lon": -118.3587, "elev": 236.0, "ghcnd": "USW00023152", "region": "San Fernando Valley"},
    "VNY": {"name": "Van Nuys", "lat": 34.2097, "lon": -118.4892, "elev": 244.0, "ghcnd": "USW00023130", "region": "San Fernando Valley"},
    "FUL": {"name": "Fullerton", "lat": 33.8700, "lon": -117.9800, "elev": 29.0, "ghcnd": "USW00003166", "region": "Orange County"},
    "SNA": {"name": "Santa Ana / John Wayne", "lat": 33.6757, "lon": -117.8682, "elev": 16.0, "ghcnd": "USW00093184", "region": "Orange County"},
    "ONT": {"name": "Ontario", "lat": 34.0560, "lon": -117.6012, "elev": 287.0, "ghcnd": "USW00003102", "region": "Inland Empire"},
    "PMD": {"name": "Palmdale", "lat": 34.6294, "lon": -118.0846, "elev": 774.0, "ghcnd": "USW00023182", "region": "Antelope Valley (desert)"},
    "WJF": {"name": "Lancaster / Fox Field", "lat": 34.7411, "lon": -118.2186, "elev": 715.0, "ghcnd": "USW00003159", "region": "Antelope Valley (desert)"},
}
BASIN_REGIONS = ("Coast & basin", "San Fernando Valley")

# Module-local rules (kept here, not in src/cascade/rubrics or rules.py). tag: PUBLIC (with source) or team-proposed.
RULES: Dict[str, dict] = {
    "CQT_END": {"value": "2024-05-20T15:00:00Z", "tag": "PUBLIC",
                "source": "NWS SCN 24-47 (KCQT moved to Elysian Park 20 May 2024 ~1500 UTC; replacement FHMC1 has no METAR)"},
    "P01I_M_SWITCH": {"value": "2012-01-18T00:00:00Z", "tag": "empirical (this archive), not documented by IEM",
                      "source": "Before this date 'M' in p01i mostly means no precipitation group (treated as 0); from it on, "
                                "'M' means missing (NaN). Share of M falls from 86-97 % (2005-2011) to 0 % after mid-January 2012."},
    "SPIKE_DAY_MM": {"value": 2.5, "tag": "team-proposed, validate",
                     "source": "IEM local-standard-day total above this while GHCN-Daily PRCP = 0 is a spurious spike; the day is zeroed"},
    "QC_RATIO_BAND": {"value": [0.85, 1.15], "tag": "team-proposed, validate",
                      "source": "station water years whose IEM/GHCN-Daily total ratio falls outside this band are excluded"},
    "QC_MIN_GHCND_MM": {"value": 25.0, "tag": "team-proposed, validate",
                        "source": "a water year with less GHCN-Daily rain than this is too dry for a stable ratio; it is kept if no spike remains"},
    "SPELL_GAP_H": {"value": 96, "tag": "PUBLIC", "source": "ISO 15927-3 spell definition via Blocken & Carmeliet 2010 section 3.2.1"},
    "EVENT_PAD_H": {"value": 3, "tag": "team-proposed, validate", "source": "hours added before and after an event window"},
    "WET_HOUR_MM": {"value": 0.1, "tag": "team-proposed, validate",
                    "source": "gridded/forecast hour counted wet at or above this (ERA5 carries many trace values)"},
    "HEAVY_MM_H": {"value": [2.5, 6.0], "tag": "team-proposed, validate", "source": "intensity classes <2.5, 2.5-6, >=6 mm/h"},
}

# ISO 15927-3 Table 1 (via Blocken & Carmeliet 2010 Table 1): category -> (K_R, z0 m, z_min m)
TERRAIN: Dict[str, Tuple[float, float, float]] = {
    "I": (0.17, 0.01, 2.0),     # rough open sea; lake shore; smooth flat country
    "II": (0.19, 0.05, 4.0),    # farm land with hedges, occasional small structures (airfield)
    "III": (0.22, 0.3, 8.0),    # suburban or industrial areas, permanent forest
    "IV": (0.24, 1.0, 16.0),    # urban, >= 15 % covered by buildings taller than 15 m on average
}
TERRAIN_LABEL = {"I": "I open sea / flat open country", "II": "II farmland / airfield",
                 "III": "III suburban or industrial", "IV": "IV dense urban (>=15 % buildings over 15 m)"}
# ISO 15927-3 Table 2 (via Blocken & Carmeliet 2010 Table 2): (lower m, upper m, O)
OBSTRUCTION: Tuple[Tuple[float, float, float], ...] = (
    (4, 8, 0.2), (8, 15, 0.3), (15, 25, 0.4), (25, 40, 0.5), (40, 60, 0.6),
    (60, 80, 0.7), (80, 100, 0.8), (100, 120, 0.9), (120, math.inf, 1.0))

LABELS = {
    "measured": "REAL: measured from ASOS airport observations (IEM), at 10 m on open airfield terrain",
    "factors": "C_T (topography) and W (wall factor) not applied (set to 1); relative exposure, not litres on your wall",
}


def rule(key: str):
    return RULES[key]["value"]


# ----------------------------------------------------------------------------------------- loading and QC

def water_year(idx: pd.DatetimeIndex) -> np.ndarray:
    """US water year: October-September, named by the year it ends in."""
    return np.asarray(idx.year + (idx.month >= 10), dtype=int)


def load_asos_csv(src, station: Optional[str] = None,
                  m_switch: Optional[str] = None, cqt_end: Optional[str] = None) -> pd.DataFrame:
    """IEM asos.py routine METAR (station,valid,drct,sknt,p01i) -> hourly frame indexed by the UTC hour END.

    Columns: r (mm in the hour; NaN = missing), v (m/s), D (deg wind FROM; NaN when missing), p_missing (bool),
    calm (bool: v == 0). Rules: keep observations at minutes 40-59 (routine hourly), ceil to the hour, last wins;
    p01i 'T' -> 0; 'M' -> 0 before P01I_M_SWITCH else NaN (empirical rule); inches x 25.4; knots x 0.514444.
    Station CQT is cut at CQT_END (site moved)."""
    d = pd.read_csv(src, dtype=str, keep_default_na=False)
    d["valid"] = pd.to_datetime(d["valid"], utc=True)
    d = d[d["valid"].dt.minute.between(40, 59)].copy()
    d["ts"] = d["valid"].dt.ceil("h")
    d = d.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    sw = pd.Timestamp(m_switch or rule("P01I_M_SWITCH"))
    p = d["p01i"].str.strip()
    miss = p.eq("M") | p.eq("")
    val = pd.to_numeric(p.replace({"T": "0"}).where(~miss), errors="coerce")
    early = d.index < sw
    r = val.copy()
    r[miss & early] = 0.0
    r[miss & ~early] = np.nan
    out = pd.DataFrame(index=d.index)
    out["r"] = (r * IN_TO_MM).astype(float)
    out["v"] = pd.to_numeric(d["sknt"], errors="coerce") * KT_TO_MS
    D = pd.to_numeric(d["drct"], errors="coerce")
    out["D"] = D.where(out["v"] > 0)
    out["p_missing"] = out["r"].isna() | (miss & early)
    out["calm"] = out["v"].eq(0)
    sid = station or (d["station"].iloc[0] if "station" in d.columns and len(d) else None)
    if sid == "CQT":
        out = out[out.index < pd.Timestamp(cqt_end or rule("CQT_END"))]
    return out


def lst_day(ts_hour_end: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local standard (UTC-8, no DST) calendar day of the hour ending at ts. GHCN-Daily first-order US stations
    use the local-standard-time day. The hour [ts-1h, ts) starts at local time ts-9h."""
    return (ts_hour_end - pd.Timedelta(hours=9)).floor("D").tz_localize(None)


def load_ghcnd_csv(src) -> pd.Series:
    """NCEI Access daily-summaries CSV (STATION,DATE,PRCP in mm) -> Series indexed by naive date."""
    g = pd.read_csv(src)
    s = pd.to_numeric(g["PRCP"], errors="coerce")
    s.index = pd.to_datetime(g["DATE"])
    return s.groupby(level=0).first()


def qc_daily(asos: pd.DataFrame, ghcnd: pd.Series, spike_mm: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-check IEM hourly rain against GHCN-Daily PRCP per local-standard day.

    Returns (cleaned asos copy, daily table). On days where GHCN-D = 0 and the IEM day total exceeds SPIKE_DAY_MM the
    IEM hours are set to 0 (spurious single-hour spikes found in the pre-2012 archive). Daily table columns:
    iem_mm, ghcnd_mm, spike (bool)."""
    spike_mm = float(rule("SPIKE_DAY_MM") if spike_mm is None else spike_mm)
    day = lst_day(asos.index)
    iem = asos["r"].groupby(day).sum(min_count=1)
    tab = pd.DataFrame({"iem_mm": iem})
    tab["ghcnd_mm"] = ghcnd.reindex(tab.index)
    tab["spike"] = (tab["ghcnd_mm"] == 0) & (tab["iem_mm"] > spike_mm)
    clean = asos.copy()
    bad = set(tab.index[tab["spike"]])
    if bad:
        m = np.isin(day, list(bad))
        clean.loc[m, "r"] = np.where(clean.loc[m, "r"].notna(), 0.0, np.nan)
    tab.loc[tab["spike"], "iem_mm"] = 0.0
    return clean, tab


def qc_water_years(daily: pd.DataFrame, band: Optional[Sequence[float]] = None,
                   min_ghcnd_mm: Optional[float] = None) -> pd.DataFrame:
    """Per water year: IEM and GHCN-D totals over days where both exist, their ratio and a pass flag."""
    band = band or rule("QC_RATIO_BAND")
    min_g = float(rule("QC_MIN_GHCND_MM") if min_ghcnd_mm is None else min_ghcnd_mm)
    t = daily.dropna(subset=["iem_mm", "ghcnd_mm"]).copy()
    idx = pd.DatetimeIndex(t.index)
    t["wy"] = idx.year + (idx.month >= 10)
    g = t.groupby("wy").agg(iem_mm=("iem_mm", "sum"), ghcnd_mm=("ghcnd_mm", "sum"), n_days=("iem_mm", "size"),
                            spikes=("spike", "sum"))
    g["ratio"] = g["iem_mm"] / g["ghcnd_mm"].where(g["ghcnd_mm"] > 0)
    in_band = g["ratio"].between(band[0], band[1])
    dry = g["ghcnd_mm"] < min_g
    g["passed"] = in_band | (dry & g["ratio"].isna()) | (dry & (g["iem_mm"] <= min_g))
    g["reason"] = np.where(g["passed"], np.where(in_band, "ratio in band", "dry year (ratio unstable), kept"),
                           "ratio outside band")
    return g.reset_index()


# ----------------------------------------------------------------------------------------- ISO 15927-3 physics

def iso_hourly(v, r, D, azimuths: Iterable[float]) -> np.ndarray:
    """Hourly airfield WDR index (L/m2) for each azimuth: (2/9) * v * r^(8/9) * max(cos(D - az), 0).

    v m/s, r mm/h, D deg wind FROM, azimuths deg (outward wall normal). Missing inputs or calm give 0.
    Returns an array of shape (n_hours, n_azimuths)."""
    v = np.asarray(v, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)
    D = np.asarray(D, dtype=float).reshape(-1)
    az = np.asarray(list(azimuths), dtype=float).reshape(-1)
    base = ISO_COEF * np.clip(np.nan_to_num(v, nan=0.0), 0, None) * np.power(np.clip(np.nan_to_num(r, nan=0.0), 0, None), ISO_R_EXP)
    ok = np.isfinite(D)
    c = np.cos(np.deg2rad(np.where(ok, D, 0.0)[:, None] - az[None, :]))
    return np.where(ok[:, None] & (c > 0), base[:, None] * c, 0.0)


def iso_frame(df: pd.DataFrame, facades: Sequence[str] = FACADES) -> pd.DataFrame:
    """iso_hourly on a frame with r, v, D columns; one column per facade label."""
    h = iso_hourly(df["v"].to_numpy(), df["r"].to_numpy(), df["D"].to_numpy(), [AZ[f] for f in facades])
    return pd.DataFrame(h, index=df.index, columns=list(facades))


def annual_index(hourly: pd.DataFrame, n_years: float) -> pd.Series:
    """Airfield annual index per column (L/m2 per year)."""
    return hourly.sum() / float(n_years)


def spells(hourly_one: pd.Series, gap_h: Optional[int] = None) -> pd.DataFrame:
    """ISO spells for ONE orientation: runs of hours with index > 0 separated by at least gap_h (96) hours in which
    that orientation's index is <= 0 (missing hours count as <= 0). Returns start, end, total (L/m2), hours."""
    gap = int(rule("SPELL_GAP_H") if gap_h is None else gap_h)
    s = hourly_one[hourly_one > 0]
    if len(s) == 0:
        return pd.DataFrame(columns=["start", "end", "total", "hours"])
    t = s.index
    dh = np.diff(np.asarray((t - t[0]) / pd.Timedelta(hours=1), dtype=float))
    new = np.concatenate([[True], (dh - 1) >= gap])
    sid = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    last = np.concatenate([first[1:] - 1, [len(t) - 1]])
    return pd.DataFrame({"start": t[first], "end": t[last],
                         "total": np.bincount(sid, weights=s.to_numpy(dtype=float)), "hours": np.bincount(sid)})


def c_r(z: float, category: str = "IV") -> float:
    """ISO roughness coefficient C_R(z) = K_R ln(max(z, z_min)/z0), Table 1."""
    kr, z0, zmin = TERRAIN[category]
    return kr * math.log(max(float(z), zmin) / z0)


def obstruction_factor(distance_m: Optional[float]) -> float:
    """ISO Table 2 obstruction factor from the distance (m) to the nearest obstruction at least as high as the wall,
    within about 25 deg either side of the wall normal. None or > 120 m gives 1.0; under 4 m is clamped to 0.2."""
    if distance_m is None or not np.isfinite(distance_m):
        return 1.0
    d = float(distance_m)
    if d < 4:
        return 0.2
    for lo, hi, o in OBSTRUCTION:
        if lo <= d < hi:
            return o
    return 1.0


def wall_index(i_a: float, z: float, category: str = "IV", obstruction_m: Optional[float] = None) -> float:
    """I_WA = I_A * C_R(z) * C_T * O * W with C_T = W = 1 (not applied)."""
    return float(i_a) * c_r(z, category) * obstruction_factor(obstruction_m)


# ----------------------------------------------------------------------------------------- events and circular stats

def segment_events(rain: pd.Series, gap_h: float = 6.0, min_mm: float = 5.0, wet_mm: float = 0.0) -> pd.DataFrame:
    """Rain events: wet hours (rain > wet_mm, or >= wet_mm when wet_mm > 0) joined while fewer than gap_h dry hours
    separate them; events with total < min_mm are dropped. The index is the hour END (UTC).
    Returns event (int), start, end (first/last wet hour), total_mm, wet_hours."""
    s = rain.dropna()
    wet = s[(s >= wet_mm) & (s > 0)] if wet_mm > 0 else s[s > 0]
    cols = ["event", "start", "end", "total_mm", "wet_hours"]
    if len(wet) == 0:
        return pd.DataFrame(columns=cols)
    t = wet.index
    dh = np.diff(np.asarray((t - t[0]) / pd.Timedelta(hours=1), dtype=float))
    new = np.concatenate([[True], (dh - 1) >= gap_h])
    eid = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    last = np.concatenate([first[1:] - 1, [len(t) - 1]])
    starts, ends = t[first], t[last]
    cs = np.concatenate([[0.0], np.cumsum(s.to_numpy(dtype=float))])
    i0 = s.index.searchsorted(starts, side="left")
    i1 = s.index.searchsorted(ends, side="right")
    ev = pd.DataFrame({"start": starts, "end": ends, "total_mm": cs[i1] - cs[i0],
                       "wet_hours": np.bincount(eid)})
    ev = ev[ev["total_mm"] >= min_mm].reset_index(drop=True)
    ev.insert(0, "event", np.arange(len(ev)))
    return ev


def weighted_circmean(deg, w=None) -> Tuple[float, float]:
    """Weighted circular mean direction (deg) and mean resultant length Rbar (0 = no preferred direction, 1 = all
    from one direction)."""
    a = np.deg2rad(np.asarray(deg, dtype=float))
    w = np.ones_like(a) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan"), float("nan")
    a, w = a[ok], w[ok]
    C, S = float((w * np.cos(a)).sum()), float((w * np.sin(a)).sum())
    return math.degrees(math.atan2(S, C)) % 360.0, math.hypot(C, S) / float(w.sum())


def rayleigh(deg) -> dict:
    """Rayleigh test of circular uniformity on unweighted angles (Zar 1999 p approximation).
    Returns n, rbar, Z, p."""
    a = np.deg2rad(np.asarray(deg, dtype=float))
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return {"n": n, "rbar": float("nan"), "Z": float("nan"), "p": float("nan")}
    R = math.hypot(float(np.cos(a).sum()), float(np.sin(a).sum()))
    rbar = R / n
    Z = n * rbar ** 2
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return {"n": int(n), "rbar": rbar, "Z": Z, "p": float(min(1.0, max(p, 0.0)))}


def block_sums(D: pd.Series, block_id: np.ndarray) -> pd.DataFrame:
    """Per block: sums of unit vectors (sx = sum sin, sy = sum cos) and count of non-calm hours."""
    a = np.deg2rad(D.to_numpy(dtype=float))
    ok = np.isfinite(a)
    f = pd.DataFrame({"b": np.asarray(block_id)[ok], "sx": np.sin(a[ok]), "sy": np.cos(a[ok])})
    g = f.groupby("b").agg(sx=("sx", "sum"), sy=("sy", "sum"), n=("sx", "size"))
    return g


def block_permutation_test(sx, sy, n, is_rain, n_perm: int = 2000, seed: int = 0) -> dict:
    """Does the wind direction during rain differ from the direction in dry weather?

    Blocks (storms for rain, calendar days for dry weather) are the exchangeable units, so hours inside one storm
    are never treated as independent. Statistic: distance between the mean unit vectors of rain-block hours and
    dry-block hours. The p value counts label permutations of whole blocks with a statistic at least as large."""
    sx, sy, n = (np.asarray(x, dtype=float) for x in (sx, sy, n))
    lab = np.asarray(is_rain, dtype=bool)

    def stat(l):
        r = np.array([sx[l].sum(), sy[l].sum()]) / max(n[l].sum(), 1)
        d = np.array([sx[~l].sum(), sy[~l].sum()]) / max(n[~l].sum(), 1)
        return float(np.hypot(*(r - d)))

    obs = stat(lab)
    rng = np.random.default_rng(seed)
    k = int(lab.sum())
    ge = 0
    idx = np.arange(len(lab))
    for _ in range(int(n_perm)):
        l = np.zeros(len(lab), dtype=bool)
        l[rng.choice(idx, k, replace=False)] = True
        ge += stat(l) >= obs
    return {"stat": obs, "p": (ge + 1) / (n_perm + 1), "n_perm": int(n_perm), "n_rain_blocks": k,
            "n_dry_blocks": int(len(lab) - k)}


def sector_of(deg) -> np.ndarray:
    """8 compass sectors centred on N, NE, ... (index 0..7); NaN stays -1."""
    d = np.asarray(deg, dtype=float)
    out = np.full(d.shape, -1, dtype=int)
    ok = np.isfinite(d)
    out[ok] = (((d[ok] + 22.5) % 360.0) // 45.0).astype(int)
    return out


def sector_shares(deg, w=None) -> Dict[str, float]:
    s = sector_of(deg)
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=float)
    ok = s >= 0
    tot = float(w[ok].sum())
    return {f: (float(w[ok & (s == i)].sum()) / tot if tot > 0 else float("nan")) for i, f in enumerate(FACADES)}


def angdiff(a, b) -> np.ndarray:
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------------------- building-layer API

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def nearest_station(lat: float, lon: float, candidates: Optional[Iterable[str]] = None) -> Tuple[str, float]:
    cands = list(candidates) if candidates is not None else list(STATIONS)
    best = min(cands, key=lambda s: haversine_km(lat, lon, STATIONS[s]["lat"], STATIONS[s]["lon"]))
    return best, haversine_km(lat, lon, STATIONS[best]["lat"], STATIONS[best]["lon"])


def load_roses(path: Optional[Union[str, Path]] = None) -> dict:
    return json.loads(Path(path or ROSES_PATH).read_text(encoding="utf-8"))


def curve_value(curve: Sequence[float], azimuth: float) -> float:
    """Linear interpolation on a 1-degree circular curve (index = azimuth 0..359)."""
    c = np.asarray(curve, dtype=float)
    n = len(c)
    x = (float(azimuth) % 360.0) * n / 360.0
    i0 = int(math.floor(x)) % n
    i1 = (i0 + 1) % n
    f = x - math.floor(x)
    return float(c[i0] * (1 - f) + c[i1] * f)


def exposure_rank(station_entry: dict, azimuth: float) -> dict:
    """Where a wall facing `azimuth` sits in the station's measured exposure: its I_A, the share of the station's
    maximum over all azimuths, the rank of that value among the 8 compass sides (1 = most exposed) and the
    percentile over all 360 azimuths."""
    curve = station_entry["IA_curve_1deg"]
    val = curve_value(curve, azimuth)
    eight = [station_entry["IA_L_m2_yr"][f] for f in FACADES]
    rank = 1 + sum(1 for x in eight if x > val + 1e-12)
    arr = np.asarray(curve, dtype=float)
    return {"azimuth": float(azimuth) % 360.0, "I_A": val, "share_of_max": val / float(arr.max()) if arr.max() > 0 else float("nan"),
            "rank_of_8": int(rank), "percentile": float((arr < val).mean() * 100.0),
            "max_azimuth": int(arr.argmax())}


def facade_exposure(lat: float, lon: float, azimuths_deg: Sequence[float], floor_heights_m: Sequence[float],
                    terrain: str = "IV", obstruction_m: Optional[float] = None, roses: Optional[dict] = None,
                    station: Optional[str] = None) -> dict:
    """Per-facade, per-height annual wall-index estimate (L/m2 per year, relative) from the nearest measured station.

    I_WA(theta, z) = I_A(theta) * C_R(z) * O with C_T = W = 1 (not applied). I_A comes from 20 water years of QC'd
    ASOS observations (eval/rain/station_roses.json). Intended to replace the repo's WDR proxy share in
    cascade.building.water.zone_risk (integration request). Returns a dict with the station, distance and rows."""
    ro = roses or load_roses()
    ent = ro["stations"]
    sid, km = (station, haversine_km(lat, lon, STATIONS[station]["lat"], STATIONS[station]["lon"])) if station \
        else nearest_station(lat, lon, [s for s in STATIONS if s in ent])
    e = ent[sid]
    o = obstruction_factor(obstruction_m)
    rows = []
    for az in azimuths_deg:
        rk = exposure_rank(e, az)
        for z in floor_heights_m:
            cr = c_r(z, terrain)
            rows.append({"azimuth": rk["azimuth"], "height_m": float(z), "I_A": rk["I_A"], "C_R": cr, "O": o,
                         "I_WA": rk["I_A"] * cr * o, "share_of_station_max": rk["share_of_max"],
                         "rank_of_8": rk["rank_of_8"]})
    return {"station": sid, "station_name": STATIONS[sid]["name"], "distance_km": km, "terrain": terrain,
            "obstruction_factor": o, "rows": rows, "labels": [LABELS["measured"], LABELS["factors"]],
            "years_used": e.get("years_used")}


def wdr_iso_series(weather: pd.DataFrame, facade: Union[str, float]) -> pd.Series:
    """ISO 15927-3 hourly airfield index on one facade from a frame in the building weather.csv contract
    (rain_mm, wind_speed_ms, wind_dir_deg = direction the wind comes FROM). `facade` is a compass label (N..NW) or an
    azimuth in degrees. Rows with missing rain or wind stay NaN (never 0), matching water.wdr_index. Drop-in for a
    method='iso' option of cascade.building.water.wdr_index (integration request)."""
    az = AZ[facade] if isinstance(facade, str) else float(facade)
    name = f"wdr_{facade}" if isinstance(facade, str) else f"wdr_{az:g}"
    if weather is None or len(weather) == 0:
        return pd.Series(dtype=float, name=name)
    r = weather["rain_mm"].astype(float)
    v = weather["wind_speed_ms"].astype(float)
    D = weather["wind_dir_deg"].astype(float)
    h = iso_hourly(v.to_numpy(), r.to_numpy(), D.where(v > 0).to_numpy(), [az])[:, 0]
    missing = (r.isna() | v.isna() | (D.isna() & (v > 0))).to_numpy()
    return pd.Series(np.where(missing, np.nan, h), index=weather.index, name=name)


def exposure_shares(lat: float, lon: float, orientations: Sequence[str], roses: Optional[dict] = None) -> Dict[str, float]:
    """Annual WDR share per facade label among the given orientations (sums to 1): the drop-in for the
    'exposure' factor of water.zone_risk, which today uses the WDR proxy on the building's own weather file."""
    res = facade_exposure(lat, lon, [AZ[o] for o in orientations], [10.0], roses=roses)
    vals = {o: r["I_A"] for o, r in zip(orientations, res["rows"])}
    tot = sum(vals.values())
    return {o: (v / tot if tot > 0 else 1.0 / len(vals)) for o, v in vals.items()}
