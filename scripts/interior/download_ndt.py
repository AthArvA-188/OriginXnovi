"""Download the REAL concrete NDT datasets used by the interior-wall strength model.

1. Matthews, Allaix, Wijte, Vullings (2025) "Non-destructive Estimation of Concrete
   Compressive Strength: Databases", Zenodo 10.5281/zenodo.15392443, CC BY 4.0.
   Files: Complete SonReb / RH / UPV Database.csv and Database Guide.pdf (md5 checked).
2. Gebauer et al. "Interrelated Data Set from Nondestructive and Destructive Material
   Testing of Concrete Compressive Strength Specimens", Harvard Dataverse
   doi:10.7910/DVN/AFCITK, CC0 1.0 (per Dataverse metadata). Drilled cores (BAM round
   robin) used as an external test. Originals are semicolon CSV, latin-1.

Writes data/raw/interior/{matthews,bam}/ and data/raw/interior/SOURCES.json.
    python scripts/interior/download_ndt.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "interior"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126 Safari/537.36"}
ZENODO = "https://zenodo.org/api/records/15392443"
DATAVERSE = ("https://dataverse.harvard.edu/api/datasets/:persistentId/"
             "?persistentId=doi:10.7910/DVN/AFCITK")


def md5_of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def get(url: str, browser_ua: bool = True, **kw) -> requests.Response:
    # Observed 2026-09-26: Dataverse returns 403 without a browser User-Agent, while
    # Zenodo returns 403 WITH a spoofed browser UA from python-requests; so it is per host.
    headers = UA if browser_ua else {"User-Agent": "cerebro-ndt-downloader/1.0 (python-requests)"}
    for attempt in range(4):
        r = requests.get(url, headers=headers, timeout=120, **kw)
        if r.status_code == 200:
            return r
        time.sleep(3 * (attempt + 1))
    r.raise_for_status()
    return r


def matthews() -> dict:
    out = RAW / "matthews"
    out.mkdir(parents=True, exist_ok=True)
    rec = get(ZENODO, browser_ua=False).json()
    files = []
    for f in rec["files"]:
        dest = out / f["key"]
        want = f["checksum"].split(":", 1)[1]
        if not (dest.exists() and md5_of(dest) == want):
            dest.write_bytes(get(f["links"]["self"], browser_ua=False).content)
        ok = md5_of(dest) == want
        if not ok:
            raise RuntimeError(f"md5 mismatch {dest}")
        files.append({"file": f["key"], "bytes": dest.stat().st_size, "md5": want, "md5_verified": ok})
        print(f"[matthews] {f['key']} {dest.stat().st_size} B md5 ok", flush=True)
    return {
        "name": rec["metadata"]["title"],
        "creators": [c["name"] for c in rec["metadata"]["creators"]],
        "doi": "10.5281/zenodo.15392443",
        "url": "https://zenodo.org/records/15392443",
        "licence": rec["metadata"]["license"]["id"],
        "published": rec["metadata"].get("publication_date"),
        "files": files,
        "accessed": dt.date.today().isoformat(),
    }


def bam() -> dict:
    out = RAW / "bam"
    out.mkdir(parents=True, exist_ok=True)
    meta = get(DATAVERSE).json()["data"]["latestVersion"]
    files = []
    for f in meta["files"]:
        df = f["dataFile"]
        name = df.get("originalFileName") or df["filename"]
        dest = out / name
        if not (dest.exists() and md5_of(dest) == df["md5"]):
            url = f"https://dataverse.harvard.edu/api/access/datafile/{df['id']}?format=original"
            dest.write_bytes(get(url).content)
        ok = md5_of(dest) == df["md5"]
        files.append({"file": name, "bytes": dest.stat().st_size, "md5": df["md5"], "md5_verified": ok})
        print(f"[bam] {name} {dest.stat().st_size} B md5 {'ok' if ok else 'MISMATCH'}", flush=True)
    title = next(c["value"] for c in meta["metadataBlocks"]["citation"]["fields"] if c["typeName"] == "title")
    return {
        "name": title,
        "doi": "10.7910/DVN/AFCITK",
        "url": "https://doi.org/10.7910/DVN/AFCITK",
        "licence": meta.get("license", {}).get("name"),
        "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10196957/",
        "files": files,
        "accessed": dt.date.today().isoformat(),
    }


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    src = {}
    rc = 0
    for key, fn in (("matthews", matthews), ("bam", bam)):
        try:
            src[key] = fn()
        except Exception as e:
            print(f"[FAIL] {key}: {e}", file=sys.stderr)
            rc = 1
    (RAW / "SOURCES.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
