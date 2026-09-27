"""Download, cache and load the public data behind the common-area energy module (ask 3).

Raw files go under data/raw/energy/ (git-ignored). Every download is recorded in data/raw/energy/manifest.json with
URL, byte size, SHA-256 and access time. Loaders never touch the network.

Data labels used across the module:
  REAL        measured public data (ROBOD, UCI 357/864, BDG2, NOAA normals)
  REANALYSIS  Open-Meteo archive (model data for a grid cell, not a station)
  SEMI-SYNTHETIC / INJECTED / SIMULATED are produced later in simulate.py and always labelled.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
RAW = REPO / "data" / "raw" / "energy"
EVAL = REPO / "eval" / "energy"
MODELS = REPO / "models" / "energy"

LA_TZ = "America/Los_Angeles"
LA_LAT, LA_LON = 34.0511, -118.2353  # downtown LA (request point; Open-Meteo snaps to a grid cell)


@dataclass(frozen=True)
class Source:
    key: str
    file: str
    url: str
    licence: str
    label: str
    cite: str


SOURCES: Dict[str, Source] = {s.key: s for s in [
    Source("robod", "robod_SupplementaryData.zip", "https://ndownloader.figshare.com/files/36228765", "CC BY 4.0", "REAL",
           "Tekler et al. 2022, ROBOD, figshare 19234530, doi:10.1007/s12273-022-0925-9"),
    Source("uci357", "uci357_occupancy_detection.zip", "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
           "CC BY 4.0", "REAL", "Candanedo & Feldheim 2016, UCI Occupancy Detection (357)"),
    Source("uci864", "uci864_room_occupancy_estimation.zip",
           "https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip", "CC BY 4.0", "REAL",
           "UCI Room Occupancy Estimation (864)"),
    Source("bdg2_meta", "bdg2_metadata.csv",
           "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv",
           "CC BY-SA (version ambiguous: header 4.0, body 3.0 legal code)", "REAL", "Miller et al., Building Data Genome 2"),
    Source("bdg2_elec", "bdg2_electricity_cleaned.csv",
           "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/meters/cleaned/electricity_cleaned.csv",
           "CC BY-SA (version ambiguous)", "REAL", "Miller et al., Building Data Genome 2"),
    Source("bdg2_weather", "bdg2_weather.csv",
           "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/weather/weather.csv",
           "CC BY-SA (version ambiguous)", "REAL", "Miller et al., Building Data Genome 2"),
    Source("noaa_monthly", "noaa_normals_monthly_1991_2020.csv",
           "https://www.ncei.noaa.gov/access/services/data/v1?dataset=normals-monthly-1991-2020"
           "&stations=USW00093134,USW00023174&dataTypes=MLY-CLDD-NORMAL,MLY-HTDD-NORMAL,MLY-TAVG-NORMAL&includeStationName=true&format=csv",
           "US government data", "REAL", "NOAA NCEI U.S. Climate Normals 1991-2020 (monthly)"),
    Source("noaa_annual", "noaa_normals_annual_1991_2020.csv",
           "https://www.ncei.noaa.gov/access/services/data/v1?dataset=normals-annualseasonal-1991-2020"
           "&stations=USW00093134,USW00023174&dataTypes=ANN-CLDD-NORMAL,ANN-HTDD-NORMAL&includeStationName=true&format=csv",
           "US government data", "REAL", "NOAA NCEI U.S. Climate Normals 1991-2020 (annual)"),
    Source("openmeteo_2025", "openmeteo_la_2025_gmt.json",
           "https://archive-api.open-meteo.com/v1/archive?latitude=34.0511&longitude=-118.2353&start_date=2025-01-01"
           "&end_date=2025-12-31&hourly=temperature_2m,shortwave_radiation&timezone=GMT",
           "CC BY 4.0 (free tier non-commercial)", "REANALYSIS", "Open-Meteo Historical Weather API"),
]}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path() -> Path:
    return RAW / "manifest.json"


def load_manifest() -> Dict[str, dict]:
    p = _manifest_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def normalized_manifest(man: Dict[str, dict]) -> Dict[str, dict]:
    """Copy of the raw-file manifest where 'accessed' is always a YYYY-MM-DD date and any free text moves to
    'provenance' (older entries stored 'copied from research scratch run of <date> (same URL)' in 'accessed')."""
    import re

    out = {}
    for k, v in man.items():
        v = dict(v)
        acc = str(v.get("accessed", ""))
        m = re.search(r"\d{4}-\d{2}-\d{2}", acc)
        if m and acc != m.group(0) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d:]+Z", acc):
            v.setdefault("provenance", "copied from the research download of the same URL (sha256 recorded); not re-downloaded")
        v["accessed"] = m.group(0) if m else acc
        out[k] = v
    return out


# Sources used but not downloaded by fetch_all (access dates from the research note energy_switching.json, refs 24 and 31).
EXTRA_SOURCES: List[Dict[str, str]] = [
    {"key": "noaa_gml_solar", "cite": "NOAA GML, General Solar Position Calculations (equations used for day length and sunset)",
     "url": "https://gml.noaa.gov/grad/solcalc/solareqns.PDF", "licence": "US government work", "label": "COMPUTED",
     "accessed": "2026-09-26"},
    {"key": "chronos_bolt_small", "cite": "Hugging Face model card, amazon/chronos-bolt-small (zero-shot comparator)",
     "url": "https://huggingface.co/amazon/chronos-bolt-small", "licence": "Apache-2.0", "label": "MODEL",
     "accessed": "2026-09-26"},
]


def fetch(key: str, *, force: bool = False, timeout: int = 600) -> Path:
    """Download SOURCES[key] to data/raw/energy/ with GET (the figshare S3 link refuses HEAD) and record it."""
    import requests

    src = SOURCES[key]
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / src.file
    man = load_manifest()
    if out.exists() and not force:
        if key not in man:
            mtime = datetime.fromtimestamp(out.stat().st_mtime, timezone.utc)
            man[key] = {"url": src.url, "file": src.file, "bytes": out.stat().st_size, "sha256": sha256(out),
                        "accessed": mtime.strftime("%Y-%m-%d"), "licence": src.licence,
                        "provenance": "file already on disk (copied from the research download of the same URL); date = file mtime"}
            _manifest_path().write_text(json.dumps(man, indent=1), encoding="utf-8")
        return out
    r = requests.get(src.url, timeout=timeout, stream=True)
    r.raise_for_status()
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.replace(out)
    man[key] = {"url": src.url, "file": src.file, "bytes": out.stat().st_size, "sha256": sha256(out),
                "accessed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "licence": src.licence}
    _manifest_path().write_text(json.dumps(man, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------------------------------------- ROBOD
ROBOD_ROOMS = {1: "Lecture room", 2: "Lecture room", 3: "Office space", 4: "Office space", 5: "Library space"}
Y = "occupant_presence [binary]"
LIGHT = "lighting_energy [kWh]"


def read_robod_csv(f) -> pd.DataFrame:
    """Parse one ROBOD combined_Room CSV. Timestamps are 'YYYY-MM-DD HH:MM +08:00'; we keep the local clock."""
    d = pd.read_csv(f)
    d["ts"] = pd.to_datetime(d["timestamp"].astype(str).str.slice(0, 16))
    d = d.sort_values("ts").reset_index(drop=True)
    return d


def load_robod(path: Optional[Path] = None) -> Dict[int, pd.DataFrame]:
    """All five ROBOD rooms from the zip (or a directory of combined_Room*.csv files)."""
    path = Path(path) if path else RAW / SOURCES["robod"].file
    rooms: Dict[int, pd.DataFrame] = {}
    if path.is_dir():
        for i in ROBOD_ROOMS:
            for f in (path / f"combined_Room{i}.csv", path / f"combined_Room{i}.csv.gz"):
                if f.exists():
                    rooms[i] = read_robod_csv(f)
                    break
        return rooms
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base.startswith("combined_Room") and base.endswith(".csv"):
                i = int(base[len("combined_Room"):-4])
                rooms[i] = read_robod_csv(io.BytesIO(z.read(name)))
    return dict(sorted(rooms.items()))


# ------------------------------------------------------------------------------------------------------------- UCI
def load_uci357(path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    path = Path(path) if path else RAW / SOURCES["uci357"].file
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base in ("datatraining.txt", "datatest.txt", "datatest2.txt"):
                d = pd.read_csv(io.BytesIO(z.read(name)))
                d["date"] = pd.to_datetime(d["date"])
                out[base[:-4]] = d.sort_values("date").reset_index(drop=True)
    return out


def load_uci864(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path else RAW / SOURCES["uci864"].file
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        d = pd.read_csv(io.BytesIO(z.read(name)))
    d["ts"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Time"].astype(str))
    return d.sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------ BDG2
def bdg2_parking_ids(meta: pd.DataFrame) -> List[str]:
    m = meta[(meta["primaryspaceusage"] == "Parking") & (meta["electricity"] == "Yes")]
    return m["building_id"].tolist()


def extract_bdg2_parking(out: Optional[Path] = None) -> Path:
    """Pre-extract the parking electricity columns (explicit usecols list; a callable usecols was far slower)."""
    out = Path(out) if out else RAW / "bdg2_parking_electricity.csv.gz"
    meta = pd.read_csv(RAW / SOURCES["bdg2_meta"].file)
    elec = RAW / SOURCES["bdg2_elec"].file
    hdr = pd.read_csv(elec, nrows=0).columns
    cols = ["timestamp"] + [c for c in bdg2_parking_ids(meta) if c in hdr]
    el = pd.read_csv(elec, usecols=cols)
    el.to_csv(out, index=False, compression="gzip")
    return out


def load_bdg2_parking() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(electricity kWh per hour, wide, local timestamps; metadata indexed by building_id; weather long)."""
    meta = pd.read_csv(RAW / SOURCES["bdg2_meta"].file).set_index("building_id")
    p = RAW / "bdg2_parking_electricity.csv.gz"
    if not p.exists():
        extract_bdg2_parking(p)
    el = pd.read_csv(p)
    el["timestamp"] = pd.to_datetime(el["timestamp"])
    el = el.set_index("timestamp").astype(float)
    wx = pd.read_csv(RAW / SOURCES["bdg2_weather"].file, usecols=["timestamp", "site_id", "airTemperature"])
    wx["timestamp"] = pd.to_datetime(wx["timestamp"])
    wx = wx.drop_duplicates(["site_id", "timestamp"])
    return el, meta, wx


# ------------------------------------------------------------------------------------------------------------ NOAA
def read_noaa_normals(path: Path) -> pd.DataFrame:
    """NOAA NCEI access-API CSV. Values arrive as space-padded strings, so strip before converting."""
    d = pd.read_csv(path, dtype=str)
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()
        if c.startswith(("MLY-", "ANN-")):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ------------------------------------------------------------------------------------------------------ Open-Meteo
def openmeteo_to_local(payload: dict, tz: str = LA_TZ) -> pd.DataFrame:
    """Open-Meteo archive JSON requested with timezone=GMT -> hourly frame indexed in local time (DST-aware).

    Requesting timezone=America/Los_Angeles makes the API apply one fixed offset for the whole year (shifting PST
    hours by 1 h and emitting a nonexistent 02:00 on the spring-forward day), so we always ask for GMT and convert.
    """
    if payload.get("utc_offset_seconds", 0) != 0:
        raise ValueError("expected an Open-Meteo payload requested with timezone=GMT (utc_offset_seconds == 0)")
    h = payload["hourly"]
    idx = pd.to_datetime(pd.Series(h["time"]), utc=True).dt.tz_convert(tz)
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    df.index = pd.DatetimeIndex(idx, name="ts_local")
    return df


def load_openmeteo(path: Optional[Path] = None) -> tuple[pd.DataFrame, dict]:
    path = Path(path) if path else RAW / SOURCES["openmeteo_2025"].file
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = {k: payload.get(k) for k in ("latitude", "longitude", "elevation", "timezone", "utc_offset_seconds")}
    return openmeteo_to_local(payload), meta


def fetch_all(log=print) -> Dict[str, Path]:
    out = {}
    for k in SOURCES:
        t0 = time.time()
        out[k] = fetch(k)
        log(f"  {k}: {out[k].name} ({out[k].stat().st_size:,} B, {time.time() - t0:.1f}s)")
    return out


def as_float(x) -> Optional[float]:
    """JSON-safe float (None for NaN/inf)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
