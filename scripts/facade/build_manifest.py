"""Build the tile manifest for the facade crack classifier (REAL data only).

Output: data/raw/facade/tiles.csv with columns
    path,label,dataset,group,split,x0,y0,x1,y1,crack_px,other_defect_px
and eval/facade/manifest_summary.json (counts per dataset/split/label, groups, rules used).
Also renders eval/facade/bfdd_label_check.jpg: BFDD label value 1 drawn in red over the RGB
frame, to check by eye that value 1 is "Crack" (the archive has no README).

Rules (team choices, recorded in the summary):
- SDNET2018 walls: group = source photo = filename prefix before '-'. Groups are shuffled with
  seed 0 and split 60/20/20 into train/val/test. Every split must hold cracked and uncracked tiles.
- Ozgenel 40k: all tiles go to train (filenames carry no source-photo id, so it cannot be
  split without leakage).
- BFDD: test only (out-of-domain). Windows of 224 px at native resolution with stride 224 and an
  edge-flush last window (the same geometry as the website heatmap). A window is "crack" if it has
  >= 20 label-1 pixels, "no crack" if 0, and dropped as ambiguous if 1-19. Other defect classes
  (values 2-5) may be present in "no crack" windows (hard negatives). Group = flight: frames sorted
  by timestamp; a new flight starts when the 4-digit DJI counter does not increase or the gap
  exceeds 10 minutes.

    python scripts/facade/build_manifest.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cascade.facade.tiles import WINDOW_PX, window_boxes  # noqa: E402

RAW = ROOT / "data" / "raw" / "facade"
EVAL = ROOT / "eval" / "facade"
SEED = 0
BFDD_CRACK_VALUE = 1
BFDD_MIN_CRACK_PX = 20
BFDD_STRIDE = 224
FLIGHT_GAP_S = 600


def sdnet_rows() -> pd.DataFrame:
    rows = []
    for lab, sub in ((1, "CW"), (0, "UW")):
        for p in sorted((RAW / "sdnet" / "W" / sub).glob("*.jpg")):
            rows.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"), "label": lab,
                         "dataset": "sdnet_walls", "group": "sdnet_" + p.stem.split("-")[0]})
    df = pd.DataFrame(rows)
    groups = np.array(sorted(df.group.unique()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(groups)
    n = len(perm)
    n_test = round(0.2 * n)
    n_val = round(0.2 * n)
    split_of = {g: "test" for g in perm[:n_test]}
    split_of.update({g: "val" for g in perm[n_test:n_test + n_val]})
    split_of.update({g: "train" for g in perm[n_test + n_val:]})
    df["split"] = df.group.map(split_of)
    for s in ("train", "val", "test"):
        labs = set(df[df.split == s].label)
        assert labs == {0, 1}, f"SDNET split {s} lacks a class: {labs}"
    return df


def ozgenel_rows() -> pd.DataFrame:
    rows = []
    for lab, sub in ((1, "Positive"), (0, "Negative")):
        for p in sorted((RAW / "ozgenel" / sub).glob("*.jpg")):
            rows.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"), "label": lab,
                         "dataset": "ozgenel", "group": "ozgenel", "split": "train"})
    return pd.DataFrame(rows)


_DJI = re.compile(r"DJI_(\d{14})_(\d{4})")


def bfdd_flights(names: list[str]) -> dict[str, str]:
    parsed = []
    for n in names:
        m = _DJI.search(n)
        t = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        parsed.append((t, int(m.group(2)), n))
    parsed.sort()
    out, flight, prev = {}, 0, None
    for t, c, n in parsed:
        if prev is not None and (c <= prev[1] or (t - prev[0]).total_seconds() > FLIGHT_GAP_S):
            flight += 1
        out[n] = f"bfdd_{t:%m%d}_f{flight:02d}"
        prev = (t, c)
    return out


def bfdd_rows() -> pd.DataFrame:
    base = RAW / "bfdd" / "Dataset_1x"
    rgbs = sorted((base / "RGB").glob("*.JPG")) + sorted((base / "RGB").glob("*.jpg"))
    rgbs = sorted(set(rgbs))
    flights = bfdd_flights([p.stem for p in rgbs])
    rows = []
    for p in rgbs:
        lab_p = base / "Label" / (p.stem + ".png")
        if not lab_p.exists():
            continue
        lab = np.asarray(Image.open(lab_p))
        h, w = lab.shape[:2]
        for (x0, y0, x1, y1) in window_boxes(w, h, WINDOW_PX, BFDD_STRIDE):
            win = lab[y0:y1, x0:x1]
            cpx = int((win == BFDD_CRACK_VALUE).sum())
            opx = int(((win > 0) & (win != BFDD_CRACK_VALUE)).sum())
            if 0 < cpx < BFDD_MIN_CRACK_PX:
                continue
            rows.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"), "label": int(cpx >= BFDD_MIN_CRACK_PX),
                         "dataset": "bfdd", "group": flights[p.stem], "split": "ood_test",
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1, "crack_px": cpx, "other_defect_px": opx})
    return pd.DataFrame(rows)


def label_check_montage(n: int = 6) -> dict:
    """Value-1 pixels in red over the RGB frame, plus thickness statistics per value."""
    base = RAW / "bfdd" / "Dataset_1x"
    labs = sorted((base / "Label").glob("*.png"))
    pick = labs[:: max(1, len(labs) // n)][:n]
    tiles = []
    for lp in pick:
        rgb_p = next((base / "RGB").glob(lp.stem + ".*"))
        rgb = np.asarray(Image.open(rgb_p).convert("RGB")).copy()
        lab = np.asarray(Image.open(lp))
        rgb[lab == BFDD_CRACK_VALUE] = (255, 0, 0)
        tiles.append(Image.fromarray(rgb).resize((320, 256)))
    W = Image.new("RGB", (320 * 3, 256 * ((len(tiles) + 2) // 3)), "white")
    for i, t in enumerate(tiles):
        W.paste(t, ((i % 3) * 320, (i // 3) * 256))
    EVAL.mkdir(parents=True, exist_ok=True)
    W.save(EVAL / "bfdd_label_check.jpg", quality=85)
    # thickness proxy: max distance-to-background inside each value's pixels (2x = width in px)
    from scipy import ndimage
    thick = {v: [] for v in range(1, 6)}
    presence = Counter()
    for lp in labs[:: max(1, len(labs) // 60)]:
        lab = np.asarray(Image.open(lp))
        for v in range(1, 6):
            m = lab == v
            if m.any():
                presence[v] += 1
                thick[v].append(float(2 * ndimage.distance_transform_edt(m).max()))
    return {"montage": "eval/facade/bfdd_label_check.jpg", "images_sampled_for_stats": sum(1 for _ in labs[:: max(1, len(labs) // 60)]),
            "value_presence": {str(k): presence[k] for k in range(1, 6)},
            "median_max_thickness_px": {str(k): (float(np.median(v)) if v else None) for k, v in thick.items()}}


def main() -> int:
    sd = sdnet_rows()
    oz = ozgenel_rows()
    bf = bfdd_rows()
    df = pd.concat([sd, oz, bf], ignore_index=True)
    out = RAW / "tiles.csv"
    df.to_csv(out, index=False)
    check = label_check_montage()
    summ = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "rules": {
            "sdnet": "group = source photo (filename prefix before '-'); groups shuffled seed 0; 60/20/20 train/val/test",
            "ozgenel": "all tiles train only (no source ids)",
            "bfdd": f"OOD test only; {WINDOW_PX}px windows stride {BFDD_STRIDE} edge-flush at native resolution; "
                    f"crack if >= {BFDD_MIN_CRACK_PX} px of label value {BFDD_CRACK_VALUE}, no-crack if 0, drop 1-{BFDD_MIN_CRACK_PX-1}; "
                    f"group = flight (counter reset or > {FLIGHT_GAP_S}s gap)",
            "bfdd_value_map": "value 1 = Crack is INFERRED (class order on the landing page, thin shapes, red overlay check); not documented in the archive",
        },
        "counts": {},
        "groups": {},
        "bfdd_label_check": check,
    }
    for (ds, sp), g in df.groupby(["dataset", "split"]):
        summ["counts"][f"{ds}/{sp}"] = {"n": int(len(g)), "crack": int(g.label.sum()), "no_crack": int((g.label == 0).sum()),
                                         "groups": int(g.group.nunique())}
    summ["groups"]["sdnet_test"] = sorted(sd[sd.split == "test"].group.unique().tolist())
    summ["groups"]["sdnet_val"] = sorted(sd[sd.split == "val"].group.unique().tolist())
    summ["groups"]["bfdd_flights"] = {k: int(v) for k, v in bf.groupby("group").path.nunique().items()}
    summ["bfdd_frames_with_labels"] = int(bf.path.nunique())
    (EVAL / "manifest_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ["counts"], indent=1))
    print("bfdd flights:", summ["groups"]["bfdd_flights"])
    print("label check:", check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
