"""Evaluate the facade tile classifier against baselines on held-out REAL data (app env).

Test sets:
- SDNET2018 walls, test split = 14 held-out SOURCE PHOTOS (in-domain headline).
- BFDD drone facade windows (224 px at native resolution), 10 flights (out-of-domain).
Operating threshold: chosen on the SDNET VALIDATION groups only, recall-first: the highest
threshold whose validation recall is >= RECALL_TARGET. Never tuned on a test set.

Models / baselines on identical tiles:
- constant score (majority / prior): AUROC 0.5 and AP = prevalence by construction
- repo heuristic: score = fraction of pixels flagged by cascade.measure.crack_mask on the tile
- ResNet-18 trained on Ozgenel only (ablation: exposes domain shift)
- ResNet-18 trained on Ozgenel + SDNET train groups (shipped)
Metrics: AUROC, AP, precision/recall/F1 at the validation threshold, confusion matrix,
share of windows below threshold (VLM calls the pre-gate avoids), best recall at precision
>= 0.9 (oracle operating point, reported for context). 95% CIs: 1,000 bootstrap resamples of
whole groups (source photos / flights). Latency: onnxruntime CPU in THIS env.

    python scripts/facade/eval_tilecls.py
Writes eval/facade/tilecls_v1.json, eval/facade/pr_curves.csv and models/facade/tilecls_v1.card.json.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cascade.measure import crack_mask  # noqa: E402

RAW = ROOT / "data" / "raw" / "facade"
EVAL = ROOT / "eval" / "facade"
MODELS = ROOT / "models" / "facade"
RECALL_TARGET = 0.90
N_BOOT = 1000
SEED = 0
TAG_MAIN = "resnet18_ozg_sdnet"
TAG_ABL = "resnet18_ozg_only"


def pick_threshold(y: np.ndarray, p: np.ndarray, target: float = RECALL_TARGET) -> float:
    """Highest threshold with recall >= target on the given (validation) rows."""
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))  # allow k positives below the threshold
    return float(pos[k])


def at_threshold(y, p, t) -> dict:
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and not np.isnan(prec)) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "share_below_threshold": float(np.mean(~pred)), "specificity": tn / (tn + fp) if tn + fp else float("nan")}


def recall_at_precision(y, p, min_prec=0.9) -> float:
    pr, rc, _ = precision_recall_curve(y, p)
    ok = pr >= min_prec
    return float(rc[ok].max()) if ok.any() else 0.0


def boot_ci(df: pd.DataFrame, col: str, t: float) -> dict:
    """Group bootstrap: resample whole groups with replacement."""
    rng = np.random.default_rng(SEED)
    groups = df.group.unique()
    by = {g: d for g, d in df.groupby("group")}
    stats = {"auroc": [], "ap": [], "recall": [], "precision": []}
    for _ in range(N_BOOT):
        pick = rng.choice(groups, len(groups), replace=True)
        d = pd.concat([by[g] for g in pick])
        y, p = d.label.to_numpy(), d[col].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        stats["auroc"].append(roc_auc_score(y, p))
        stats["ap"].append(average_precision_score(y, p))
        m = at_threshold(y, p, t)
        stats["recall"].append(m["recall"])
        if not np.isnan(m["precision"]):
            stats["precision"].append(m["precision"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None for k, v in stats.items()} | \
        {"n_boot_valid": len(stats["auroc"])}


def heuristic_scores(rows: np.ndarray, mm) -> np.ndarray:
    def one(i):
        m = crack_mask(np.asarray(mm[i]))
        return float(m.mean())
    with ThreadPoolExecutor(6) as ex:
        return np.array(list(ex.map(one, rows, chunksize=64)))


def ort_latency(onnx_path: Path, mm, rows) -> dict:
    import onnxruntime as ort
    from cascade.facade.tiles import IMAGENET_MEAN, IMAGENET_STD
    x = (np.asarray(mm[np.sort(rows[:256])], np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    s.run(None, {"input": x[:32]})
    t0 = time.time()
    outs = [s.run(None, {"input": x[i:i + 32]})[0] for i in range(0, len(x), 32)]
    ms = (time.time() - t0) / len(x) * 1000
    z = np.concatenate(outs)
    e = np.exp(z - z.max(1, keepdims=True))
    return {"ms_per_tile": round(ms, 2), "threads": 4, "batch": 32, "n_tiles": int(len(x)),
            "p": (e[:, 1] / e.sum(1)).tolist(), "rows": np.sort(rows[:256]).tolist()}


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(RAW / "tiles.csv")
    mm = np.load(RAW / "cache224.npy", mmap_mode="r")
    main_p = pd.read_parquet(RAW / "preds" / f"{TAG_MAIN}.parquet")
    abl_p = pd.read_parquet(RAW / "preds" / f"{TAG_ABL}.parquet")
    ev = df[((df.dataset == "sdnet_walls") & df.split.isin(["val", "test"])) | (df.dataset == "bfdd")].copy()
    ev["p_main"] = main_p.p_crack.to_numpy()
    ev["p_abl"] = abl_p.p_crack.to_numpy()
    assert (main_p.path.to_numpy() == ev.path.to_numpy()).all()
    print("scoring heuristic on", len(ev), "tiles", flush=True)
    ev["p_heur"] = heuristic_scores(ev.index.to_numpy(), mm)
    ev["p_const"] = 0.0  # constant scorer: AUROC 0.5 and AP = prevalence by construction

    val = ev[(ev.dataset == "sdnet_walls") & (ev.split == "val")]
    sets = {"sdnet_test": ev[(ev.dataset == "sdnet_walls") & (ev.split == "test")], "bfdd_ood": ev[ev.dataset == "bfdd"]}
    models = {"constant_prior": "p_const", "heuristic_crack_mask": "p_heur", "resnet18_ozgenel_only": "p_abl",
              "resnet18_ozgenel_sdnet": "p_main"}
    thresholds = {m: pick_threshold(val.label.to_numpy(), val[c].to_numpy()) for m, c in models.items()}
    thresholds["constant_prior"] = 1.0  # always "no crack" (training majority class)
    res = {"sets": {}, "thresholds_from_val": thresholds}
    curves = []
    for sname, d in sets.items():
        y = d.label.to_numpy()
        res["sets"][sname] = {"n": int(len(d)), "crack": int(y.sum()), "no_crack": int((y == 0).sum()),
                              "prevalence": float(y.mean()), "groups": int(d.group.nunique()), "models": {}}
        for m, c in models.items():
            p = d[c].to_numpy()
            t = thresholds[m]
            r = {"auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
                 "threshold": t, **at_threshold(y, p, t), "recall_at_precision_0_9_oracle": recall_at_precision(y, p)}
            if m != "constant_prior":
                r["ci95_group_bootstrap"] = boot_ci(d, c, t)
            res["sets"][sname]["models"][m] = r
            if m in ("heuristic_crack_mask", "resnet18_ozgenel_only", "resnet18_ozgenel_sdnet"):
                pr, rc, _ = precision_recall_curve(y, p)
                idx = np.linspace(0, len(pr) - 1, min(200, len(pr))).astype(int)
                curves += [{"set": sname, "model": m, "precision": float(pr[i]), "recall": float(rc[i])} for i in idx]
        print(sname, {m: round(v["auroc"], 3) for m, v in res["sets"][sname]["models"].items()}, flush=True)
    val_m = {m: {"auroc": float(roc_auc_score(val.label, val[c])), "ap": float(average_precision_score(val.label, val[c])),
                 **at_threshold(val.label.to_numpy(), val[c].to_numpy(), thresholds[m])} for m, c in models.items()}
    res["sdnet_val"] = {"n": int(len(val)), "groups": int(val.group.nunique()), "models": val_m}

    # per-frame view on BFDD: windows sent to the VLM per frame at the val threshold
    b = sets["bfdd_ood"].copy()
    b["sent"] = b.p_main >= thresholds["resnet18_ozgenel_sdnet"]
    per = b.groupby("path").agg(windows=("sent", "size"), sent=("sent", "sum"), crack_windows=("label", "sum"))
    res["bfdd_frames"] = {"n_frames": int(len(per)), "median_windows_per_frame": float(per.windows.median()),
                          "median_sent_per_frame": float(per.sent.median()),
                          "frames_with_crack_label_and_zero_windows_sent": int(((per.crack_windows > 0) & (per.sent == 0)).sum()),
                          "frames_with_crack_label": int((per.crack_windows > 0).sum())}

    # latency and ORT-vs-torch agreement in the APP env
    onnx_path = MODELS / "tilecls_resnet18_v1.onnx"
    lat = ort_latency(onnx_path, mm, sets["sdnet_test"].index.to_numpy())
    torch_p = ev.loc[lat["rows"], "p_main"].to_numpy()
    res["latency"] = {"ort_cpu_app_env": {k: lat[k] for k in ("ms_per_tile", "threads", "batch", "n_tiles")},
                      "ort_vs_torch_max_abs_p_diff": float(np.max(np.abs(np.array(lat["p"]) - torch_p)))}
    meta_main = json.loads((RAW / "ckpt" / f"{TAG_MAIN}.json").read_text())
    meta_abl = json.loads((RAW / "ckpt" / f"{TAG_ABL}.json").read_text())
    exp = json.loads((RAW / "ckpt" / f"{TAG_MAIN}_export.json").read_text())
    res["latency"]["gpu_torch_fp16_b128_ms_per_tile"] = meta_main["gpu_ms_per_tile_fp16_b128"]
    res["training"] = {"main": meta_main, "ablation": meta_abl}
    res["export"] = exp
    summ = json.loads((EVAL / "manifest_summary.json").read_text())
    out = {"id": "facade_tilecls_v1", "created": dt.datetime.now().isoformat(timespec="seconds"),
           "label": "REAL data: SDNET2018 walls (grouped by source photo), BFDD drone facades (grouped by flight). No synthetic tiles.",
           "recall_target_on_val": RECALL_TARGET, "n_boot": N_BOOT, "manifest": summ, **res,
           "seconds": round(time.time() - t0, 1)}
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "tilecls_v1.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(curves).round(4).to_csv(EVAL / "pr_curves.csv", index=False)

    s_main = res["sets"]["sdnet_test"]["models"]["resnet18_ozgenel_sdnet"]
    b_main = res["sets"]["bfdd_ood"]["models"]["resnet18_ozgenel_sdnet"]
    card = {
        "model_id": "tilecls:resnet18@v1", "file": "models/facade/tilecls_resnet18_v1.onnx",
        "architecture": "timm resnet18.a1_in1k (Apache-2.0 tag), 2-class head, fine-tuned",
        "input": {"name": "input", "shape": ["N", 3, 224, 224], "dtype": "float32", "range": "RGB/255 then ImageNet mean/std",
                  "window_px": 224, "note": "heatmap windows are cut at native resolution"},
        "output": {"name": "logits", "classes": ["no_crack", "crack"]},
        "storage": "fp16 weights with Cast to fp32 (fp32 compute in onnxruntime)",
        "threshold": thresholds["resnet18_ozgenel_sdnet"],
        "threshold_rule": f"highest threshold with recall >= {RECALL_TARGET} on SDNET validation source photos",
        "training_data": [{"name": "Ozgenel Concrete Crack Images for Classification", "licence": "CC BY 4.0",
                           "url": "https://data.mendeley.com/datasets/5y9wdsg2zt/2"},
                          {"name": "SDNET2018 walls (train source photos only)", "licence": "CC BY 4.0",
                           "url": "https://digitalcommons.usu.edu/all_datasets/48/"}],
        "headline": {"sdnet_test_auroc": s_main["auroc"], "sdnet_test_recall": s_main["recall"],
                     "bfdd_ood_auroc": b_main["auroc"], "bfdd_ood_recall": b_main["recall"],
                     "source": "eval/facade/tilecls_v1.json"},
        "limits": ["trained on close-range concrete tiles; drone facade behaviour = BFDD OOD test only",
                   "AI screening for a human inspector, not a QEWI/FISP finding; FISP says drones/photos do not replace close-up inspection",
                   "no GSD rescaling: window size is in pixels, so apparent crack width depends on camera distance"],
    }
    (MODELS / "tilecls_v1.card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps({k: res["sets"][k]["models"]["resnet18_ozgenel_sdnet"] for k in res["sets"]}, indent=1, default=float)[:2000])
    print("latency", res["latency"], "seconds", out["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
