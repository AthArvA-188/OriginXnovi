"""Download the REAL facade crack datasets into data/raw/facade (git-ignored).

Datasets (licences read at source on 2026-09-26):
- Ozgenel "Concrete Crack Images for Classification" v2, Mendeley 5y9wdsg2zt, CC BY 4.0.
- SDNET2018, Utah State University digital commons, CC BY 4.0 (walls subset W/CW, W/UW only).
- BFDD RGB-IR building facade defect dataset, Mendeley 9ych7czvyg, CC BY 4.0.

Usage (training env or app env, needs only requests):
    python scripts/facade/download_data.py --only ozgenel,bfdd,sdnet

Each step verifies the sha256 when the publisher gives one, extracts, and writes
data/raw/facade/SOURCES.json with URL, licence, size, hash and access date.
SDNET sits behind Cloudflare: one plain request with a browser User-Agent, no
parallel ranges. On 403/429 the script prints a manual-download message.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "facade"
SCRATCH_BFDD = Path(os.environ.get(
    "BFDD_LOCAL_COPY",
    r"C:\Users\HP\AppData\Local\Temp\claude\E--origin-hack\5dd83316-e9df-42fe-9235-8e04db2688c6"
    r"\scratchpad\research\ask2\bfdd\BFDD.tar.gz"))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126 Safari/537.36"}
BSDTAR = r"C:\Windows\System32\tar.exe"  # bsdtar 3.8.8 reads RAR4; Git-Bash tar cannot

DATASETS = {
    "ozgenel": {
        "name": "Concrete Crack Images for Classification (Ozgenel 2019), v2",
        "url": "https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/"
               "8a70d8a5-bce9-4291-bab9-b48cfb3e87c3/file_downloaded",
        "landing": "https://data.mendeley.com/datasets/5y9wdsg2zt/2",
        "licence": "CC BY 4.0",
        "size": 241363336,
        "sha256": "08d7dc505a4f5a0330cee2fa2a1ae4b5b4f98bcffd0549b774caf7959bb1f02f",
        "file": "ozgenel_classification.rar",
        "role": "training only (no source-photo ids in filenames)",
    },
    "sdnet": {
        "name": "SDNET2018 (Maguire, Dorafshan, Thomas 2018)",
        "url": "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?params=/context/all_datasets/"
               "article/1047/type/native/&path_info=",
        "landing": "https://digitalcommons.usu.edu/all_datasets/48/",
        "licence": "CC BY 4.0",
        "size": 528286896,
        "sha256": None,
        "file": "SDNET2018_all.zip",
        "role": "walls subset; grouped train/val/test split by source photo (filename prefix)",
    },
    "bfdd": {
        "name": "BFDD building facade defect dataset, RGB-IR, Dataset_1x (2026-04-08)",
        "url": "https://data.mendeley.com/public-files/datasets/9ych7czvyg/files/"
               "c1c5144b-cb20-4687-b514-d0bbec12209e/file_downloaded",
        "landing": "https://data.mendeley.com/datasets/9ych7czvyg/1",
        "licence": "CC BY 4.0",
        "size": 553316751,
        "sha256": "43d06305bf3c913f59d52c3ffa10caa0e129b668b7b3c9d8f80d619c6e6e8a7a",
        "file": "BFDD_Dataset_1x_20260408.tar.gz",
        "role": "out-of-domain test only (real drone facade photos)",
        "note": "landing page says 788 pairs; archive lists 838 RGB / 838 Label / 839 IR files",
    },
}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_curl(url: str, dest: Path, expected_size: int | None) -> None:
    """One plain curl request (python-requests gets a Cloudflare 403 on the USU host)."""
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"[skip] {dest.name} already complete")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    # Observed 2026-09-26: a plain GET returns 403 from Cloudflare, while one open-ended
    # range request ("bytes=N-") returns 206 with the whole remaining file.
    start = tmp.stat().st_size if tmp.exists() else 0
    if not (expected_size and start == expected_size):
        mode = "ab" if start else "wb"
        cmd = ["curl", "-sS", "-L", "--fail", "-A", UA["User-Agent"], "-r", f"{start}-", url]
        with open(tmp, mode) as f:
            subprocess.run(cmd, check=True, stdout=f)
    if expected_size and tmp.stat().st_size != expected_size:
        raise RuntimeError(f"size {tmp.stat().st_size} != {expected_size}; re-run to resume")
    tmp.replace(dest)
    print(f"[ok] {dest.name}: {dest.stat().st_size} B in {time.time()-t0:.0f} s", flush=True)


def fetch(url: str, dest: Path, expected_size: int | None) -> None:
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"[skip] {dest.name} already complete")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        if r.status_code in (403, 429):
            raise RuntimeError(
                f"HTTP {r.status_code} from {url}. The host is rate limiting. Download it in a browser "
                f"and save it as {dest}, then re-run with the same --only flag.")
        r.raise_for_status()
        n = 0
        last = t0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                n += len(chunk)
                if time.time() - last > 30:
                    last = time.time()
                    rate = n / max(1e-6, last - t0) / 1e6
                    print(f"  {dest.name}: {n/1e6:.0f} MB at {rate:.2f} MB/s", flush=True)
    tmp.replace(dest)
    print(f"[ok] {dest.name}: {dest.stat().st_size} B in {time.time()-t0:.0f} s", flush=True)


def record(key: str, dest: Path, extra: dict) -> None:
    src = RAW / "SOURCES.json"
    data = json.loads(src.read_text(encoding="utf-8")) if src.exists() else {}
    meta = {k: v for k, v in DATASETS[key].items() if k != "file"}
    meta.update({"local_file": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size,
                 "sha256_local": sha256_of(dest), "accessed": dt.date.today().isoformat()})
    meta.update(extra)
    data[key] = meta
    src.write_text(json.dumps(data, indent=2), encoding="utf-8")


def do_ozgenel() -> None:
    d = DATASETS["ozgenel"]
    out = RAW / "ozgenel"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / d["file"]
    fetch(d["url"], dest, d["size"])
    got = sha256_of(dest)
    if got != d["sha256"]:
        raise RuntimeError(f"sha256 mismatch for {dest}: {got}")
    if not (out / "Positive").exists():
        subprocess.run([BSDTAR, "-xf", str(dest), "-C", str(out)], check=True)
    n_pos = len(list((out / "Positive").glob("*.jpg")))
    n_neg = len(list((out / "Negative").glob("*.jpg")))
    print(f"[ozgenel] Positive={n_pos} Negative={n_neg}")
    record("ozgenel", dest, {"sha256_verified": True, "n_positive": n_pos, "n_negative": n_neg})


def do_bfdd() -> None:
    d = DATASETS["bfdd"]
    out = RAW / "bfdd"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / d["file"]
    how = "downloaded"
    if not (dest.exists() and dest.stat().st_size == d["size"]):
        if SCRATCH_BFDD.exists() and SCRATCH_BFDD.stat().st_size == d["size"]:
            shutil.copyfile(SCRATCH_BFDD, dest)
            how = "copied from the research agent's download of the same URL, hash re-verified"
        else:
            fetch(d["url"], dest, d["size"])
    got = sha256_of(dest)
    if got != d["sha256"]:
        raise RuntimeError(f"sha256 mismatch for {dest}: {got}")
    if not (out / "Dataset_1x" / "RGB").exists():
        with tarfile.open(dest, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if "/RGB/" in m.name or "/Label/" in m.name or "/Label_color/" in m.name]
            tf.extractall(out, members=members, filter="data")
    # the archive's top folder name may differ; find RGB
    rgb = next(out.rglob("RGB"))
    n_rgb = len([p for p in rgb.iterdir() if p.is_file()])
    print(f"[bfdd] RGB files={n_rgb} at {rgb}")
    record("bfdd", dest, {"sha256_verified": True, "how": how, "n_rgb_files": n_rgb,
                          "rgb_dir": str(rgb.relative_to(ROOT))})


def do_sdnet() -> None:
    d = DATASETS["sdnet"]
    out = RAW / "sdnet"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / d["file"]
    fetch_curl(d["url"], dest, d["size"])
    walls = out / "W"
    if not (walls / "CW").exists():
        with zipfile.ZipFile(dest) as z:
            inner_name = next(n for n in z.namelist() if n.lower().endswith(".zip"))
            inner = out / "SDNET2018_inner.zip"
            if not inner.exists():
                with z.open(inner_name) as src, open(inner, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        with zipfile.ZipFile(inner) as z2:
            names = [n for n in z2.namelist()
                     if n.startswith(("W/CW/", "W/UW/")) and n.lower().endswith(".jpg")]
            for n in names:
                target = out / n
                target.parent.mkdir(parents=True, exist_ok=True)
                with z2.open(n) as s, open(target, "wb") as t:
                    t.write(s.read())
        inner.unlink(missing_ok=True)
    n_cw = len(list((walls / "CW").glob("*.jpg")))
    n_uw = len(list((walls / "UW").glob("*.jpg")))
    print(f"[sdnet] W/CW={n_cw} W/UW={n_uw}")
    record("sdnet", dest, {"sha256_verified": False, "sha256_note": "publisher gives no hash; size checked",
                           "n_wall_cracked": n_cw, "n_wall_uncracked": n_uw})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ozgenel,bfdd,sdnet")
    a = ap.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    steps = {"ozgenel": do_ozgenel, "bfdd": do_bfdd, "sdnet": do_sdnet}
    rc = 0
    for k in [s.strip() for s in a.only.split(",") if s.strip()]:
        try:
            steps[k]()
        except Exception as e:  # keep going; report clearly
            print(f"[FAIL] {k}: {e}", file=sys.stderr, flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
