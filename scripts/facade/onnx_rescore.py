"""Re-score the evaluation tiles with the SHIPPED ONNX file (app env, CPU) and compare with PyTorch.

Why: eval_tilecls.py computes the headline metrics from PyTorch checkpoint predictions
(data/raw/facade/preds/*.parquet, fp16 autocast in the training env). The website runs
models/facade/tilecls_resnet18_v1.onnx (fp16-stored weights, fp32 compute) in onnxruntime.
This script scores the SAME rows (SDNET validation + test photos, BFDD windows) from the same
224 px tile cache with the shipped ONNX file and reports, per set and at the card threshold:
AUROC, AP, precision/recall/F1, share below threshold, for both prediction sources, plus
agreement (max/mean |p diff|, share of tiles whose flag flips).

    python scripts/facade/onnx_rescore.py            # resumes from a partial file if interrupted
Writes data/raw/facade/preds/onnx_shipped.parquet (git-ignored) and eval/facade/onnx_rescore.json.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade.facade.tiles import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eval_tilecls import at_threshold, pick_threshold  # noqa: E402

RAW = ROOT / "data" / "raw" / "facade"
EVAL = ROOT / "eval" / "facade"
MODELS = ROOT / "models" / "facade"
ONNX = MODELS / "tilecls_resnet18_v1.onnx"
TORCH_PREDS = RAW / "preds" / "resnet18_ozg_sdnet.parquet"
PARTIAL = RAW / "preds" / "onnx_shipped.partial.npy"
OUT_PREDS = RAW / "preds" / "onnx_shipped.parquet"
BATCH = 64


def score_rows(rows: np.ndarray, mm) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    p = np.full(len(rows), np.nan, np.float64)
    if PARTIAL.exists():
        old = np.load(PARTIAL)
        if len(old) == len(rows):
            p = old
    start = int(np.argmax(np.isnan(p))) if np.isnan(p).any() else len(p)
    print(f"resuming at {start} / {len(rows)}", flush=True)
    t0 = time.time()
    for i in range(start, len(rows), BATCH):
        idx = rows[i:i + BATCH]
        x = (np.asarray(mm[idx], np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float32)
        z = s.run(None, {name: x})[0].astype(np.float64)
        e = np.exp(z - z.max(1, keepdims=True))
        p[i:i + len(idx)] = e[:, 1] / e.sum(1)
        if (i // BATCH) % 10 == 0:
            np.save(PARTIAL, p)
            done = i + len(idx)
            print(f"{done} / {len(rows)} tiles, {(time.time() - t0) / max(1, done - start) * 1000:.1f} ms/tile", flush=True)
    np.save(PARTIAL, p)
    return p


def metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    m = at_threshold(y, p, t)
    return {"auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "threshold": t,
            **{k: m[k] for k in ("precision", "recall", "f1", "share_below_threshold", "tp", "fp", "fn", "tn")}}


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(RAW / "tiles.csv")
    mm = np.load(RAW / "cache224.npy", mmap_mode="r")
    torch_p = pd.read_parquet(TORCH_PREDS)
    ev = df[((df.dataset == "sdnet_walls") & df.split.isin(["val", "test"])) | (df.dataset == "bfdd")].copy()
    assert (torch_p.path.to_numpy() == ev.path.to_numpy()).all(), "row order differs from the PyTorch predictions"
    ev["p_torch"] = torch_p.p_crack.to_numpy()
    ev["p_onnx"] = score_rows(ev.index.to_numpy(), mm)
    assert not ev.p_onnx.isna().any()
    ev[["path", "label", "dataset", "group", "split", "x0", "y0", "x1", "y1", "p_onnx"]].to_parquet(OUT_PREDS, index=False)

    card = json.loads((MODELS / "tilecls_v1.card.json").read_text(encoding="utf-8"))
    thr = float(card["threshold"])
    sets = {"sdnet_val": ev[(ev.dataset == "sdnet_walls") & (ev.split == "val")],
            "sdnet_test": ev[(ev.dataset == "sdnet_walls") & (ev.split == "test")],
            "bfdd_ood": ev[ev.dataset == "bfdd"]}
    out_sets = {}
    for name, d in sets.items():
        y = d.label.to_numpy()
        a, b = d.p_torch.to_numpy(), d.p_onnx.to_numpy()
        diff = np.abs(a - b)
        out_sets[name] = {
            "n": int(len(d)), "crack": int(y.sum()), "groups": int(d.group.nunique()),
            "pytorch_checkpoint": metrics(y, a, thr), "shipped_onnx": metrics(y, b, thr),
            "agreement": {"max_abs_p_diff": float(diff.max()), "mean_abs_p_diff": float(diff.mean()),
                          "flag_flips": int(np.sum((a >= thr) != (b >= thr))),
                          "flag_flip_share": float(np.mean((a >= thr) != (b >= thr)))},
        }
    val = sets["sdnet_val"]
    out = {
        "id": "facade_tilecls_v1_onnx_rescore", "created": dt.datetime.now().isoformat(timespec="seconds"),
        "label": "REAL data, same held-out rows as eval/facade/tilecls_v1.json; no synthetic tiles.",
        "provenance": ("eval/facade/tilecls_v1.json scores come from PyTorch checkpoint predictions (training env, fp16 autocast). "
                       "This file re-scores the same rows with the shipped ONNX file in onnxruntime CPU (app env), "
                       "the model the website runs."),
        "onnx_file": str(ONNX.relative_to(ROOT)).replace("\\", "/"), "onnx_bytes": ONNX.stat().st_size,
        "card_threshold": thr,
        "threshold_onnx_val_would_pick": pick_threshold(val.label.to_numpy(), val.p_onnx.to_numpy()),
        "input": "224 px uint8 tile cache (data/raw/facade/cache224.npy), ImageNet normalisation, batch 64, 4 threads",
        "sets": out_sets, "seconds": round(time.time() - t0, 1),
    }
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "onnx_rescore.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    PARTIAL.unlink(missing_ok=True)
    print(json.dumps({k: {"torch": round(v["pytorch_checkpoint"]["auroc"], 4), "onnx": round(v["shipped_onnx"]["auroc"], 4),
                          "rec_onnx": round(v["shipped_onnx"]["recall"], 4), **v["agreement"]} for k, v in out_sets.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
