"""Train the facade tile crack classifier on the GPU (training env: E:\\conda_envs\\cerebro_ml).

Binary crack / no-crack at 224 px, timm backbone with ImageNet weights, class-weighted CE,
AdamW + cosine, AMP. Model selection = best average precision on the SDNET VALIDATION groups
(never the test groups). After training, scores every val / test / BFDD row with the selected
checkpoint and writes data/raw/facade/preds/<tag>.parquet for eval_tilecls.py.

    python scripts/facade/train_tilecls.py --tag resnet18_ozg_sdnet --backbone resnet18.a1_in1k --train-sets ozgenel,sdnet_walls
    python scripts/facade/train_tilecls.py --tag resnet18_ozg_only  --backbone resnet18.a1_in1k --train-sets ozgenel

Windows: everything runs under __main__ (DataLoader workers spawn).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "facade"
CKPT = RAW / "ckpt"
PREDS = RAW / "preds"


import timm  # noqa: E402  (training env only)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Tiles(Dataset):
    """Rows of tiles.csv -> uint8 HWC 224x224 (decode + crop + bilinear resize only).

    Augmentation runs batched on the GPU (gpu_augment): per-sample CPU transforms ran at
    ~120 img/s on this shared 16 GB laptop, which made 4 epochs take ~30 min."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(ROOT / r.path).convert("RGB")
        if not (isinstance(r.x0, float) and math.isnan(r.x0)):
            im = im.crop((int(r.x0), int(r.y0), int(r.x1), int(r.y1)))
        if im.size != (224, 224):
            im = im.resize((224, 224), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()), int(r.label)


CACHE = RAW / "cache224.npy"  # built by scripts/facade/cache_tiles.py; row i = tiles.csv row i


class MemmapBatches:
    """Batches of pre-decoded uint8 tiles read from the memmap, prefetched by one thread."""

    def __init__(self, mm, rows, labels, batch, shuffle=False, drop_last=False, seed=0):
        self.mm, self.rows, self.labels = mm, np.asarray(rows), np.asarray(labels)
        self.batch, self.shuffle, self.drop_last = batch, shuffle, drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.rows) // self.batch
        return n if self.drop_last or len(self.rows) % self.batch == 0 else n + 1

    def _gen(self):
        order = self.rng.permutation(len(self.rows)) if self.shuffle else np.arange(len(self.rows))
        for i in range(0, len(order), self.batch):
            b = order[i:i + self.batch]
            if self.drop_last and len(b) < self.batch:
                break
            if not self.shuffle:
                b = np.sort(b)
            r = self.rows[b]
            srt = np.argsort(r)  # sorted reads are faster; keep x/y aligned
            yield torch.from_numpy(np.ascontiguousarray(self.mm[r[srt]])), torch.from_numpy(self.labels[b][srt])

    def __iter__(self):
        import queue
        import threading
        q = queue.Queue(maxsize=4)
        stop = object()

        def run():
            for item in self._gen():
                q.put(item)
            q.put(stop)

        threading.Thread(target=run, daemon=True).start()
        while True:
            item = q.get()
            if item is stop:
                return
            yield item


def to_float(xb, dev):
    x = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return x


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def gpu_augment(x, g):
    """Batched: random resized crop (scale 0.6-1, ratio 3/4-4/3), h/v flips, rot90, brightness/contrast/saturation."""
    B = x.shape[0]
    dev = x.device
    area = torch.empty(B, device=dev).uniform_(0.6, 1.0, generator=g)
    logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3), generator=g)
    r = torch.exp(logr)
    wf = torch.sqrt(area * r).clamp(max=1.0)
    hf = torch.sqrt(area / r).clamp(max=1.0)
    tx = (torch.rand(B, device=dev, generator=g) * 2 - 1) * (1 - wf)
    ty = (torch.rand(B, device=dev, generator=g) * 2 - 1) * (1 - hf)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = wf, tx, hf, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    hflip = torch.rand(B, device=dev, generator=g) < 0.5
    x = torch.where(hflip.view(B, 1, 1, 1), x.flip(-1), x)
    vflip = torch.rand(B, device=dev, generator=g) < 0.5
    x = torch.where(vflip.view(B, 1, 1, 1), x.flip(-2), x)
    k = torch.randint(0, 4, (B,), device=dev, generator=g)
    for kk in (1, 2, 3):
        m = k == kk
        if m.any():
            x[m] = torch.rot90(x[m], kk, dims=(-2, -1))
    bf = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3, generator=g)
    cf = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3, generator=g)
    sf = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2, generator=g)
    x = (x * bf).clamp(0, 1)
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = ((x - gray.mean((2, 3), keepdim=True)) * cf + gray.mean((2, 3), keepdim=True)).clamp(0, 1)
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = ((x - gray) * sf + gray).clamp(0, 1)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    ap.add_argument("--train-sets", default="ozgenel,sdnet_walls")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed)
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.read_csv(RAW / "tiles.csv")
    sets = [s.strip() for s in args.train_sets.split(",")]
    tr = df[(df.split == "train") & df.dataset.isin(sets)]
    va = df[(df.dataset == "sdnet_walls") & (df.split == "val")]
    dev = torch.device("cuda")
    n_pos, n_neg = int(tr.label.sum()), int((tr.label == 0).sum())
    w = torch.tensor([len(tr) / (2 * n_neg), len(tr) / (2 * n_pos)], dtype=torch.float32, device=dev)
    print(f"[{args.tag}] train {len(tr)} (crack {n_pos}, no {n_neg}) from {sets}; val {len(va)}", flush=True)

    mm = np.load(CACHE, mmap_mode="r")
    assert mm.shape[0] == len(df), "cache224.npy is stale: re-run scripts/facade/cache_tiles.py"
    dl_tr = MemmapBatches(mm, tr.index.to_numpy(), tr.label.to_numpy(), args.batch, shuffle=True, drop_last=True, seed=args.seed)
    # validation in-process: two worker pools exhausted the Windows commit limit (error 1455) on 16 GB RAM
    dl_va = MemmapBatches(mm, va.index.to_numpy(), va.label.to_numpy(), 256)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    def score(loader):
        model.eval()
        ps, ys = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for x, y in loader:
                p = torch.softmax(model(normalize(to_float(x, dev))).float(), 1)[:, 1]
                ps.append(p.cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    CKPT.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in dl_tr:
            x = normalize(gpu_augment(to_float(x, dev), gen))
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
            if (n // len(y)) % 100 == 0:
                print(f"  step {n // len(y)}: {n / (time.time() - t0):.0f} img/s", flush=True)
        torch.cuda.synchronize()
        sec = time.time() - t0
        yv, pv = score(dl_va)
        ap_v, auc_v = float(average_precision_score(yv, pv)), float(roc_auc_score(yv, pv))
        log.append({"epoch": ep + 1, "train_loss": tot / n, "val_ap": ap_v, "val_auroc": auc_v, "train_seconds": round(sec, 1)})
        print(f"[{args.tag}] epoch {ep+1}: loss {tot/n:.4f} val AP {ap_v:.4f} AUROC {auc_v:.4f} ({sec:.0f}s)", flush=True)
        if ap_v > best:
            best = ap_v
            torch.save(model.state_dict(), CKPT / f"{args.tag}.pt")

    # score val / test / BFDD with the selected checkpoint
    model.load_state_dict(torch.load(CKPT / f"{args.tag}.pt", map_location=dev))
    ev = df[((df.dataset == "sdnet_walls") & df.split.isin(["val", "test"])) | (df.dataset == "bfdd")].copy()
    del dl_tr
    dl_ev = MemmapBatches(mm, ev.index.to_numpy(), ev.label.to_numpy(), 256)
    _, pe = score(dl_ev)
    ev["p_crack"] = pe
    PREDS.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(PREDS / f"{args.tag}.parquet", index=False)

    # GPU latency, fp16 autocast, batch 128, after warm-up
    model.eval()
    x = torch.randn(128, 3, 224, 224, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
    ms_tile = (time.time() - t0) / (20 * 128) * 1000
    meta = {"tag": args.tag, "backbone": args.backbone, "train_sets": sets, "n_train": len(tr),
            "n_train_crack": n_pos, "n_train_no_crack": n_neg, "n_val": len(va), "epochs": args.epochs,
            "batch": args.batch, "lr": args.lr, "weight_decay": args.wd, "seed": args.seed,
            "class_weights": w.tolist(), "selected_epoch": int(np.argmax([r["val_ap"] for r in log]) + 1),
            "log": log, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "timm": timm.__version__,
            "gpu_ms_per_tile_fp16_b128": round(ms_tile, 3)}
    (CKPT / f"{args.tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("tag", "selected_epoch", "gpu_ms_per_tile_fp16_b128")}), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    raise SystemExit(main())
