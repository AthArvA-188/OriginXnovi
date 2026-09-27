"""Fetch (or copy from the research scratch folder) the REAL inputs for the pipe-clog module and write a sha256 manifest.

Inputs (all accessed 2026-09-26):
  * HSB Living Lab household water use, Zenodo 22076411, CC BY 4.0 (REAL demand for the riser simulation)
  * Bellinge urban drainage data set, DTU collection 5029124, item "2 - Sensor data", CC BY 4.0
    - 2_cleaned_data/G71F04R_Level1_System2000p2_proc_v6.csv   (upstream of the throttle pipe)
    - 2_cleaned_data/G71F06R_LevelInlet_System2000p1_proc_v6.csv (downstream of the throttle pipe)
    - 2_Sensordata_v2.pdf (sensor readme that documents the 23-28 July 2020 blockage)
  * Rule documents used by the periodic-check scheduler (LADWP Rule 16-D, LASAN SSMP v3.0, UPC 608.2 page,
    Lansing 2023 CIB W062 stack-capacity paper). Kept only for provenance; the scheduler reads its own rule table.

Raw files go to data/raw/clog/ (git-ignored). The manifest (small) goes to eval/clog/data_manifest.json.

Usage:
  python scripts/fetch_clog_data.py            # copy from the research scratch folder, download what is missing
  python scripts/fetch_clog_data.py --no-net   # copy only
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "clog"
EVAL = ROOT / "eval" / "clog"
SCRATCH = Path(r"C:\Users\HP\AppData\Local\Temp\claude\E--origin-hack\5dd83316-e9df-42fe-9235-8e04db2688c6"
               r"\scratchpad\research")
ACCESSED = "2026-09-26"
BELLINGE_ZIP = "https://ndownloader.figshare.com/files/30592524"

FILES = [
    {
        "name": "hsb.zip",
        "scratch": "ask6_clog/hsb.zip",
        "url": "https://zenodo.org/records/22076411/files/HSB_Living_Lab_Data_Package.zip?download=1",
        "landing": "https://zenodo.org/records/22076411",
        "dataset": "HSB Living Lab fixture-level household water consumption (2019-2023)",
        "licence": "CC BY 4.0",
        "kind": "REAL",
        "use": "Real 10-min apartment water use that drives the synthetic riser simulation",
        "notes": ("Column 'value' is m3 per 10 min (the sum 3,270.8 matches the stated 3,271 m3) although "
                  "DATA_DICTIONARY.txt says litres; timestamps are UTC and are converted to Europe/Stockholm "
                  "before the 03:00 test is placed. Rows without an apartment (shared spaces) are dropped by the "
                  "per-apartment totals; their share of volume is computed in hsb_stats."),
    },
    {
        "name": "G71F04R_Level1_System2000p2_proc_v6.csv",
        "scratch": "ask6_clog/G71F04R_Level1_System2000p2_proc_v6.csv",
        "url": BELLINGE_ZIP,
        "member": "2_cleaned_data/G71F04R_Level1_System2000p2_proc_v6.csv",
        "landing": "https://doi.org/10.11583/DTU.c.5029124",
        "dataset": "Bellinge sensor data, cleaned, G71F04R Level 1 (upstream of the throttle pipe)",
        "licence": "CC BY 4.0",
        "kind": "REAL",
        "use": "Upstream level for the documented July 2020 blockage case study",
        "notes": ("1-min, local time UTC+01 with DST. Readme: scaling changed to 0-2 m instead of 0-2.9 m from "
                  "06-01-2020 to about 19-11-2020 (depth multiplied by 1.45 in the cleaning script), which covers "
                  "the whole analysis window Jan-Oct 2020."),
    },
    {
        "name": "G71F06R_LevelInlet_System2000p1_proc_v6.csv",
        "scratch": "ask6_clog/G71F06R_LevelInlet_System2000p1_proc_v6.csv",
        "url": BELLINGE_ZIP,
        "member": "2_cleaned_data/G71F06R_LevelInlet_System2000p1_proc_v6.csv",
        "landing": "https://doi.org/10.11583/DTU.c.5029124",
        "dataset": "Bellinge sensor data, cleaned, G71F06R Level inlet (downstream of the throttle pipe)",
        "licence": "CC BY 4.0",
        "kind": "REAL",
        "use": "Downstream level for the documented July 2020 blockage case study",
        "notes": "1-min, local time UTC+01 with DST; 2010-08-01 to 2020-10-12; only Jan-Oct 2020 is read.",
    },
    {
        "name": "bellinge_2_Sensordata_v2.pdf",
        "scratch": "ask6_clog/bellinge_sensordata.pdf",
        "url": "https://ndownloader.figshare.com/files/30592500",
        "landing": "https://doi.org/10.11583/DTU.c.5029124",
        "dataset": "Bellinge sensor readme (2_Sensordata_v2.pdf)",
        "licence": "CC BY 4.0",
        "kind": "REAL",
        "use": ("Documents the event: 23-07-2020 - 28-07-2020 'The throttle pipe between sensor G71F04R and "
                "G71F06R was blocked.'"),
        "notes": "",
    },
    {
        "name": "ladwp_rule16d.pdf",
        "scratch": "ask6_clog/ladwp_rule16d.pdf",
        "url": "https://www.ladwp.com/sites/default/files/2025-04/LADWP%20Rule%2016-D%20REVISED%20July_15_2015.pdf",
        "landing": "https://www.ladwp.com/who-we-are/water-system/water-quality/backflow-prevention-requirements",
        "dataset": "LADWP Rule 16-D Protection of Public Water Supply",
        "licence": "public regulation (quoted, not redistributed)",
        "kind": "RULE",
        "use": "Backflow assemblies tested annually by a certified tester (sections 9.1-9.2)",
        "notes": "",
    },
    {
        "name": "lasan_ssmp.pdf",
        "scratch": "ask6_clog/lasan_ssmp.pdf",
        "url": ("https://planning.lacity.gov/eir/Sunset_Wilcox/deir/deir_reference_docs/water-supply/"
                "LASAN%20-%20Sewer%20System%20Management%20Plan%20Hyperion%20Sanitary%20Sewer%20System,"
                "%20January%202019.pdf"),
        "landing": "https://sanitation.lacity.gov/programs/fats-oils-and-grease-fog",
        "dataset": "LA Sanitation Sewer System Management Plan v3.0 (25 Jan 2019)",
        "licence": "public document (quoted, not redistributed)",
        "kind": "RULE",
        "use": "Grease interceptor 25% rule and daily grease-trap cleaning (7.5.3); CCTV about 48 h after overflow",
        "notes": "",
    },
    {
        "name": "upc_608_2.html",
        "scratch": "verify_clog/upc.html",
        "url": "https://up.codes/s/excessive-water-pressure",
        "landing": "https://up.codes/s/excessive-water-pressure",
        "dataset": "Section 608.2 Excessive Water Pressure (UPC-based Hawaii Plumbing Code 2021, UpCodes)",
        "licence": "public code text (quoted, not redistributed)",
        "kind": "RULE",
        "use": "80 psi static limit; regulators of 1-1/2 in (40 mm) or larger do not require a strainer",
        "notes": "California Plumbing Code equivalence not verified.",
    },
    {
        "name": "lansing2023_cib_w062.pdf",
        "scratch": "verify_clog/cib.pdf",
        "url": "https://www.irbnet.de/daten/iconda/CIB_DC39171.pdf",
        "landing": "https://www.irbnet.de/daten/iconda/CIB_DC39171.pdf",
        "dataset": "Lansing 2023, CIB W062 symposium paper on single-stack drainage standards",
        "licence": "conference paper (quoted, not redistributed)",
        "kind": "RULE",
        "use": "DN100 stack maximum 4.0 L/s with square entries, 5.2 L/s with swept entries (EN 12056 / DIN 1986-100)",
        "notes": "",
    },
]


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (re-requests the original URL each time because the
    figshare redirect is a signed S3 URL that expires)."""

    def __init__(self, url: str):
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.url, self.pos = url, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def obtain(spec: dict, allow_net: bool) -> tuple[Path | None, str]:
    dst = RAW / spec["name"]
    if dst.exists() and dst.stat().st_size > 0:
        return dst, "already present"
    src = SCRATCH / spec["scratch"]
    if src.exists():
        shutil.copyfile(src, dst)
        return dst, f"copied from research scratch ({spec['scratch']})"
    if not allow_net:
        return None, "missing (network disabled)"
    if "member" in spec:
        zf = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(spec["url"]), buffer_size=1 << 16))
        with zf.open(spec["member"]) as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, 1 << 20)
        return dst, "downloaded (HTTP Range member read)"
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0 (cerebro research fetch)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as fout:
        shutil.copyfileobj(r, fout, 1 << 20)
    return dst, "downloaded"


def hsb_stats(zip_path: Path) -> dict:
    """Dataset properties computed here (not copied from the research notes): totals, apartment coverage, span."""
    import zipfile

    import pandas as pd

    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(z.open("HSB_Living_Lab_Water_Consumption_Anonymized.csv"), sep=";",
                         usecols=["timestamp", "apartment", "value"])
    total = float(df["value"].sum())
    no_apt = float(df.loc[df["apartment"].isna(), "value"].sum())
    return {"rows": int(len(df)), "total_value_m3": round(total, 1),
            "share_of_volume_without_apartment": round(no_apt / total, 4) if total else None,
            "n_apartments": int(df["apartment"].nunique()),
            "first_timestamp_utc": str(df["timestamp"].min()), "last_timestamp_utc": str(df["timestamp"].max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-net", action="store_true")
    args = ap.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    rows = []
    prev_how = {}
    old = EVAL / "data_manifest.json"
    if old.exists():  # keep how each file was first obtained when a rerun finds it already present
        prev_how = {f["name"]: f.get("how", "") for f in json.loads(old.read_text(encoding="utf-8")).get("files", [])}
    for spec in FILES:
        path, how = obtain(spec, allow_net=not args.no_net)
        if how == "already present" and prev_how.get(spec["name"], "already present") != "already present":
            how = prev_how[spec["name"]]
        row = {k: v for k, v in spec.items() if k != "scratch"}
        row.update(accessed=ACCESSED, how=how, local_path=None, bytes=None, sha256=None)
        if path is not None:
            row.update(local_path=str(path.relative_to(ROOT)).replace("\\", "/"), bytes=path.stat().st_size,
                       sha256=sha256(path))
        rows.append(row)
        print(f"{spec['name']:48s} {how:45s} {row['bytes']}")
    manifest = {
        "generated": str(date.today()),
        "generator": "scripts/fetch_clog_data.py",
        "raw_dir": "data/raw/clog (git-ignored)",
        "not_used": [
            "Bellinge rain gauges (CC BY-NC 4.0)",
            "Sewer-ML, CCTV-Pipe and ISWDS sewer-CCTV data/models (non-commercial terms)",
            "WEUSEDTO fixture data (Zenodo record says GPL-2.0 while its data/LICENSE.txt says CC BY 4.0; not needed "
            "after the verifier descoped per-event drain ML)",
        ],
        "files": rows,
    }
    hsb = RAW / "hsb.zip"
    if hsb.exists():
        manifest["hsb_stats"] = hsb_stats(hsb)
        print("hsb_stats", manifest["hsb_stats"])
    (EVAL / "data_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    missing = [r["name"] for r in rows if r["sha256"] is None]
    if missing:
        print("MISSING:", missing, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
