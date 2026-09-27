"""Decode every manifest row once into a 224x224 uint8 memmap (data/raw/facade/cache224.u8).

Why: on this shared laptop each small-JPEG open cost ~5.7 ms and decode ~2.8 ms under CPU
contention, so DataLoader workers fed only ~150 img/s and the GPU sat idle. Reading a
pre-decoded row from a memmap is ~0.1 ms. Row i of the memmap = row i of tiles.csv.
BFDD rows are cropped to their window box first (224 px, native resolution, no resize).

    python scripts/facade/cache_tiles.py
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "facade"
S = 224


def load_row(r) -> np.ndarray:
    im = Image.open(ROOT / r.path).convert("RGB")
    if not (isinstance(r.x0, float) and math.isnan(r.x0)):
        im = im.crop((int(r.x0), int(r.y0), int(r.x1), int(r.y1)))
    if im.size != (S, S):
        im = im.resize((S, S), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def main() -> int:
    df = pd.read_csv(RAW / "tiles.csv")
    out = RAW / "cache224.u8"
    mm = np.lib.format.open_memmap(out.with_suffix(".npy"), mode="w+", dtype=np.uint8, shape=(len(df), S, S, 3))
    t0 = time.time()
    rows = list(df.itertuples(index=False))

    def work(i):
        mm[i] = load_row(rows[i])
        return i

    with ThreadPoolExecutor(12) as ex:
        for n, _ in enumerate(ex.map(work, range(len(rows)), chunksize=64), start=1):
            if n % 10000 == 0:
                print(f"  {n}/{len(rows)} {n/(time.time()-t0):.0f} img/s", flush=True)
    mm.flush()
    meta = {"rows": len(df), "shape": [len(df), S, S, 3], "file": str(out.with_suffix('.npy').relative_to(ROOT)),
            "seconds": round(time.time() - t0, 1), "manifest": "data/raw/facade/tiles.csv"}
    (RAW / "cache224.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
