"""Bundle a few REAL facade photos for the website, with their labelled cracks (app env).

Samples are BFDD drone frames (CC BY 4.0, attribution below), picked by a fixed rule so we do not
cherry-pick: frames sorted by labelled crack pixels, then the frames at the 90th, 50th and 10th
percentiles. For each: the photo (copied), the labelled-crack overlay (value 1 in red), and the
per-window scores of the shipped ONNX model with each window's label, written to
eval/facade/samples/samples.json. These frames are part of the BFDD OOD TEST set.

    python scripts/facade/make_samples.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cascade.facade.heatmap import TileClassifier, load_card, score_image  # noqa: E402

RAW = ROOT / "data" / "raw" / "facade"
OUT = ROOT / "eval" / "facade" / "samples"
PCTS = (90, 50, 10)
ATTRIB = ("BFDD: Pixel-Level Aligned RGB-IR Image Dataset for Building Facade Defect Segmentation, "
          "Mendeley Data 9ych7czvyg v1, CC BY 4.0, https://data.mendeley.com/datasets/9ych7czvyg/1 (accessed 2026-09-26)")


def main() -> int:
    df = pd.read_csv(RAW / "tiles.csv")
    b = df[df.dataset == "bfdd"]
    base = RAW / "bfdd" / "Dataset_1x"
    frames = sorted(b.path.unique())
    px = []
    for f in frames:
        lab = np.asarray(Image.open(base / "Label" / (Path(f).stem + ".png")))
        px.append(int((lab == 1).sum()))
    s = pd.DataFrame({"path": frames, "crack_px": px}).sort_values(["crack_px", "path"]).reset_index(drop=True)
    picks = [s.iloc[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] for p in PCTS]
    clf = TileClassifier()
    thr = load_card().get("threshold")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"attribution": ATTRIB, "rule": f"frames sorted by labelled crack pixels; percentiles {PCTS}",
           "label": "REAL drone facade photos from the BFDD out-of-domain TEST set", "model_id": clf.model_id,
           "threshold": thr, "samples": []}
    for pct, row in zip(PCTS, picks):
        src = ROOT / row.path
        stem = Path(row.path).stem
        dst = OUT / f"{stem}.jpg"
        shutil.copyfile(src, dst)
        lab = np.asarray(Image.open(base / "Label" / (stem + ".png")))
        rgb = np.asarray(Image.open(src).convert("RGB")).copy()
        rgb[lab == 1] = (255, 0, 0)
        Image.fromarray(rgb).save(OUT / f"{stem}_labelled_cracks.jpg", quality=88)
        res = score_image(Image.open(src), clf)
        wins = []
        for t in res.tiles:
            x0, y0, x1, y1 = t.box
            cpx = int((lab[y0:y1, x0:x1] == 1).sum())
            wins.append({"box": [x0, y0, x1, y1], "p_crack": round(t.p_crack, 4), "label_crack_px": cpx,
                         "label": "crack" if cpx >= 20 else ("no crack" if cpx == 0 else "ambiguous (1-19 px, dropped in eval)")})
        out["samples"].append({"file": f"eval/facade/samples/{stem}.jpg", "labelled": f"eval/facade/samples/{stem}_labelled_cracks.jpg",
                               "percentile": pct, "labelled_crack_px": int(row.crack_px), "size": list(res.image_size),
                               "windows": wins, "seconds": round(res.seconds, 3)})
        print(stem, pct, row.crack_px, [w["p_crack"] for w in wins])
    (OUT / "samples.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
