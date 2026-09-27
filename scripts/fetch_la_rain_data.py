"""Download and cache the REAL data behind the LA rain-exposure feature (key: rain).

Everything lands under data/raw/rain/ (git-ignored). Files already cached are skipped. A manifest with URL, parameters,
access date, licence, size and row count per file is written to data/raw/rain/sources_manifest.json, and a copy the
website may read is written to eval/rain/sources_manifest.json.

Sources (all accessed 2026-09-26):
  IEM ASOS routine hourly METAR   https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py   public domain
  IEM CA_ASOS station metadata    https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson
  Open-Meteo archive (ERA5)       https://archive-api.open-meteo.com/v1/archive               CC BY 4.0
  Open-Meteo Previous Runs (GFS)  https://previous-runs-api.open-meteo.com/v1/forecast        CC BY 4.0
  NCEI Access Data Service        https://www.ncei.noaa.gov/access/services/data/v1           US federal data
      daily-summaries (GHCN-Daily PRCP) and normals-annualseasonal-1991-2020

Pacing: IEM 2 s between requests (1 s per-IP throttle); Open-Meteo 20-240 s between requests (free tier: 600/min,
5,000/h, 10,000/day, fractional call weighting); NCEI 1 s. Open-Meteo HTTP 429 stops that step and is recorded.

Usage:
  python scripts/fetch_la_rain_data.py --steps meta,iem,era5,ghcnd,normals,prev [--seed-from DIR]
  python scripts/fetch_la_rain_data.py --steps grid --grid-pause 240        # ERA5 grid for the map (slow, background)
--seed-from copies files already downloaded on 2026-09-26 by the research pilot (same URLs and parameters) instead of
downloading them again, which saves the shared Open-Meteo quota. Copied files are marked as such in the manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "rain"
EVAL = ROOT / "eval" / "rain"
MANIFEST = RAW / "sources_manifest.json"
UA = {"User-Agent": "PavilionCerebro-hackathon/0.1 (research; contact team@sigmahealthsense.com)"}

STATIONS = ["LAX", "BUR", "LGB", "VNY", "SMO", "HHR", "CQT", "FUL", "SNA", "ONT", "PMD", "WJF"]
# GHCN-Daily ids from the IEM CA_ASOS geojson 'ncei91' property (checked 2026-09-26)
GHCND = {"BUR": "USW00023152", "CQT": "USW00093134", "FUL": "USW00003166", "HHR": "USW00003167",
         "LAX": "USW00023174", "LGB": "USW00023129", "ONT": "USW00003102", "PMD": "USW00023182",
         "SMO": "USW00093197", "SNA": "USW00093184", "VNY": "USW00023130", "WJF": "USW00003159"}

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_GEO = "https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson"
IEM_STS, IEM_ETS = "2005-10-01T00:00Z", "2025-10-01T00:00Z"      # water years 2006-2025
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
NCEI_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"
NCEI_SEARCH = "https://www.ncei.noaa.gov/access/services/search/v1/data"
ERA5_START, ERA5_END = "2015-10-01", "2025-09-30"                 # water years 2016-2025
PREV_START, PREV_END = "2024-01-01", "2025-09-30"                 # Previous Runs archive starts January 2024
ERA5_VARS = "precipitation,wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m"
GRID_VARS = "precipitation,wind_speed_10m,wind_direction_10m"
GRID_LATS = [33.75, 34.0, 34.25, 34.5, 34.75]
GRID_LONS = [-118.75, -118.5, -118.25, -118.0, -117.75]
GRID_SKIP = {(33.75, -118.75)}                                   # open ocean (Santa Monica Bay / San Pedro Channel)

LICENCE = {
    "iem": "public domain (https://mesonet.agron.iastate.edu/disclaimer.php)",
    "open-meteo": "CC BY 4.0, free tier non-commercial (https://open-meteo.com/en/terms); ERA5: Copernicus/ECMWF",
    "ncei": "US federal government data (NOAA NCEI)",
}
TODAY = dt.date.today().isoformat()


def log(*a) -> None:
    print(*a, flush=True)


def load_manifest() -> Dict[str, dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(m: Dict[str, dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    EVAL.mkdir(parents=True, exist_ok=True)
    slim = {k: {kk: v[kk] for kk in ("url", "params", "accessed", "licence", "origin", "rows", "bytes") if kk in v}
            for k, v in m.items()}
    (EVAL / "sources_manifest.json").write_text(json.dumps(slim, indent=1, sort_keys=True), encoding="utf-8")


def _rows(path: Path) -> Optional[int]:
    try:
        if path.suffix == ".csv":
            with open(path, "rb") as f:
                return max(0, sum(1 for _ in f) - 1)
        if path.suffix in (".json", ".geojson"):
            j = read_json(path)
            if isinstance(j, dict) and "hourly" in j:
                return len(j["hourly"].get("time", []))
            if isinstance(j, dict) and "features" in j:
                return len(j["features"])
            if isinstance(j, list):
                return len(j)
    except Exception:  # noqa: BLE001
        return None
    return None


def record(m: Dict[str, dict], path: Path, url: str, params, licence: str, origin: str, accessed: str = TODAY) -> None:
    b = path.read_bytes()
    m[str(path.relative_to(RAW)).replace("\\", "/")] = {
        "url": url, "params": params, "accessed": accessed, "licence": licence, "origin": origin,
        "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "rows": _rows(path)}


def get(url: str, params, timeout: int = 600, tries: int = 3, pause: float = 5.0) -> Optional[requests.Response]:
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            if r.status_code == 200:
                return r
            log("  http", r.status_code, r.text[:200].replace("\n", " "))
            if r.status_code == 429:
                return r
        except Exception as e:  # noqa: BLE001
            log("  error", type(e).__name__, str(e)[:200])
        time.sleep(pause * (k + 1))
    return None


def seed(m: Dict[str, dict], src: Optional[Path], rel_src: str, dst: Path, url: str, params, licence: str) -> bool:
    """Copy a pilot-cached file (same URL/params, downloaded 2026-09-26) when present."""
    if src is None:
        return False
    s = src / rel_src
    if not s.exists() or s.stat().st_size < 1000:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, dst)
    accessed = dt.date.fromtimestamp(s.stat().st_mtime).isoformat()
    record(m, dst, url, params, licence, "copied from research pilot cache (downloaded by the pilot with these "
           "parameters)", accessed)
    log("  seeded", dst.name, "from", s)
    return True


def step_meta(m, src):
    dst = RAW / "ca_asos.geojson"
    if dst.exists():
        return
    r = get(IEM_GEO, None, timeout=120)
    if r is not None and r.status_code == 200:
        dst.write_text(r.text, encoding="utf-8")
        record(m, dst, IEM_GEO, None, LICENCE["iem"], "downloaded by scripts/fetch_la_rain_data.py")
    else:
        seed(m, src, "ca_asos.geojson", dst, IEM_GEO, None, LICENCE["iem"])


def step_iem(m, src):
    for s in STATIONS:
        dst = RAW / "iem" / f"{s}.csv"
        params = [("station", s), ("data", "drct"), ("data", "sknt"), ("data", "p01i"), ("tz", "UTC"),
                  ("format", "onlycomma"), ("latlon", "no"), ("missing", "M"), ("trace", "T"),
                  ("report_type", "3"), ("sts", IEM_STS), ("ets", IEM_ETS)]
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        if seed(m, src, f"iem/{s}.csv", dst, IEM_URL, params, LICENCE["iem"]):
            continue
        log("IEM", s)
        r = get(IEM_URL, params)
        if r is not None and r.status_code == 200:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(r.text, encoding="utf-8")
            record(m, dst, IEM_URL, params, LICENCE["iem"], "downloaded by scripts/fetch_la_rain_data.py")
        time.sleep(2)


def _coords() -> Dict[str, tuple]:
    g = json.loads((RAW / "ca_asos.geojson").read_text(encoding="utf-8"))
    return {f["properties"]["sid"]: (f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])
            for f in g["features"]}


def step_era5(m, src):
    co = _coords()
    for s in STATIONS:
        dst = RAW / "era5" / f"{s}.json"
        lat, lon = co[s]
        params = {"latitude": lat, "longitude": lon, "start_date": ERA5_START, "end_date": ERA5_END,
                  "hourly": ERA5_VARS, "wind_speed_unit": "ms", "timezone": "GMT", "models": "era5"}
        if dst.exists():
            continue
        if seed(m, src, f"era5/{s}.json", dst, OM_ARCHIVE, params, LICENCE["open-meteo"]):
            continue
        log("ERA5", s)
        r = get(OM_ARCHIVE, params)
        if r is None or r.status_code != 200:
            log("  ERA5 stopped (quota or error); rerun later")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(r.text, encoding="utf-8")
        record(m, dst, OM_ARCHIVE, params, LICENCE["open-meteo"], "downloaded by scripts/fetch_la_rain_data.py")
        time.sleep(40)


def step_ghcnd(m, src):
    for s in STATIONS:
        dst = RAW / "ghcnd" / f"{s}.csv"
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        params = {"dataset": "daily-summaries", "dataTypes": "PRCP", "stations": GHCND[s], "startDate": "2005-10-01",
                  "endDate": "2025-09-30", "units": "metric", "format": "csv"}
        log("GHCN-D", s, GHCND[s])
        r = get(NCEI_DATA, params, timeout=300)
        if r is not None and r.status_code == 200 and len(r.text) > 1000:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(r.text, encoding="utf-8")
            record(m, dst, NCEI_DATA, params, LICENCE["ncei"], "downloaded by scripts/fetch_la_rain_data.py")
        else:
            log("  GHCN-D failed for", s)
        time.sleep(1)


def step_normals(m, src):
    dst = RAW / "ncei_normals_la.json"
    if dst.exists():
        return
    sparams = {"dataset": "normals-annualseasonal-1991-2020", "bbox": "34.85,-118.95,33.70,-117.65", "limit": 1000}
    r = get(NCEI_SEARCH, sparams, timeout=120)
    ids: List[str] = []
    if r is not None and r.status_code == 200:
        try:
            j = r.json()
            for res in j.get("results", []):
                for st in res.get("stations", []):
                    ids.append(st["id"])
        except Exception as e:  # noqa: BLE001
            log("  normals search parse error", e)
    ids = sorted(set(ids))
    log("normals stations found:", len(ids))
    if ids:
        dparams = {"dataset": "normals-annualseasonal-1991-2020", "stations": ",".join(ids),
                   "dataTypes": "ANN-PRCP-NORMAL,DJF-PRCP-NORMAL,MAM-PRCP-NORMAL,JJA-PRCP-NORMAL,SON-PRCP-NORMAL",
                   "format": "json", "includeStationName": "true", "includeStationLocation": "1"}
        r2 = get(NCEI_DATA, dparams, timeout=300)
        if r2 is not None and r2.status_code == 200 and r2.text.strip().startswith("["):
            dst.write_text(r2.text, encoding="utf-8")
            record(m, dst, NCEI_DATA, {**dparams, "stations": f"{len(ids)} ids from {NCEI_SEARCH} bbox search"},
                   LICENCE["ncei"], "downloaded by scripts/fetch_la_rain_data.py")
            return
    seed(m, src, "ncei_normals_la.json", dst, NCEI_DATA,
         {"dataset": "normals-annualseasonal-1991-2020", "dataTypes": "ANN-PRCP-NORMAL", "bbox": sparams["bbox"]},
         LICENCE["ncei"])


def step_prev(m, src):
    co = _coords()
    for s in STATIONS:
        dst = RAW / "prev_gfs" / f"{s}.json"
        if dst.exists():
            continue
        lat, lon = co[s]
        params = {"latitude": lat, "longitude": lon, "start_date": PREV_START, "end_date": PREV_END,
                  "hourly": "precipitation_previous_day1,wind_speed_10m_previous_day1,wind_direction_10m_previous_day1",
                  "models": "gfs_seamless", "wind_speed_unit": "ms", "timezone": "GMT"}
        log("Previous Runs GFS day-1", s)
        r = get(OM_PREV, params)
        if r is None or r.status_code != 200:
            log("  Previous Runs stopped (quota or error); rerun later")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(r.text, encoding="utf-8")
        record(m, dst, OM_PREV, params, LICENCE["open-meteo"], "downloaded by scripts/fetch_la_rain_data.py")
        save_manifest(m)
        time.sleep(20)


def read_json(path: Path):
    """JSON from disk; pilot caches were written in the Windows code page (degree sign), so fall back to latin-1."""
    b = path.read_bytes()
    try:
        return json.loads(b.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(b.decode("latin-1"))


def _cell(x: float) -> float:
    return round(round(x * 4) / 4, 2)


def step_grid(m, src, pause: float, max_cells: int):
    """ERA5 10 m wind + precipitation at 0.25 deg cell centres; station files already cover some cells."""
    have = set()
    for p in (RAW / "era5").glob("*.json"):
        j = read_json(p)
        have.add((_cell(j["latitude"]), _cell(j["longitude"])))
    done = 0
    for la in GRID_LATS:
        for lo in GRID_LONS:
            if (la, lo) in GRID_SKIP or (la, lo) in have:
                continue
            dst = RAW / "era5_grid" / f"{la:.2f}_{lo:.2f}.json"
            if dst.exists():
                continue
            if done >= max_cells:
                return
            params = {"latitude": la, "longitude": lo, "start_date": ERA5_START, "end_date": ERA5_END,
                      "hourly": GRID_VARS, "wind_speed_unit": "ms", "timezone": "GMT", "models": "era5"}
            log("ERA5 grid cell", la, lo, dt.datetime.now().strftime("%H:%M:%S"))
            r = get(OM_ARCHIVE, params, tries=2, pause=60)
            if r is None or r.status_code != 200:
                log("  grid stopped (quota or error); rerun later")
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(r.text, encoding="utf-8")
            record(m, dst, OM_ARCHIVE, params, LICENCE["open-meteo"], "downloaded by scripts/fetch_la_rain_data.py")
            save_manifest(m)
            done += 1
            time.sleep(pause)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="meta,iem,era5,ghcnd,normals,prev")
    ap.add_argument("--seed-from", default=None, help="directory with pilot caches (iem/, era5/, ca_asos.geojson)")
    ap.add_argument("--grid-pause", type=float, default=240.0)
    ap.add_argument("--grid-max", type=int, default=30)
    a = ap.parse_args(argv)
    src = Path(a.seed_from) if a.seed_from else None
    RAW.mkdir(parents=True, exist_ok=True)
    m = load_manifest()
    steps = a.steps.split(",")
    fns = {"meta": step_meta, "iem": step_iem, "era5": step_era5, "ghcnd": step_ghcnd, "normals": step_normals,
           "prev": step_prev}
    for s in steps:
        if s == "grid":
            step_grid(m, src, a.grid_pause, a.grid_max)
        elif s in fns:
            log("== step", s)
            fns[s](m, src)
        else:
            log("unknown step", s)
            return 2
        save_manifest(m)
    log("manifest entries:", len(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
