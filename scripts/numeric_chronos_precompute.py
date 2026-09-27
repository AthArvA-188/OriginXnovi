"""Step 2 of the numeric backtest: zero-shot Chronos forecasts, run OFFLINE in the cerebro_ml env (torch + chronos).

Kept standalone on purpose: it imports neither cascade nor scikit-learn (a combined sklearn + torch process stalled
in the research pilot). Reads eval/numeric/bdg2_selected.parquet and eval/numeric/backtest_setup.json (REAL BDG2
meters and the origin schedule written by scripts/numeric_backtest.py), loads each model at its PINNED Hugging Face
revision, and writes eval/numeric/forecasts_<tag>.parquet + forecasts_<tag>.provenance.json.

Context = the last 1024 hours before each origin, forward-filled up to 6 h (causal only); longer gaps stay NaN,
which Chronos accepts. The point forecast (q50) and the 80% band (q10, q90) come from the same model call.

Run (Windows):
  set HF_HOME=E:\\hf_cache & set OMP_NUM_THREADS=4
  E:\\conda_envs\\cerebro_ml\\python.exe scripts/numeric_chronos_precompute.py --models chronos-2 chronos-bolt-small
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval" / "numeric"
MODELS = {  # pinned revisions (Hugging Face API, accessed 2026-09-26)
    "chronos-2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", "chronos_2"),
    "chronos-bolt-small": ("amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8", "chronos_bolt_small"),
}
LICENCE = "Apache-2.0"
QUANTILES = [0.1, 0.5, 0.9]
T0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def contexts(sel: pd.DataFrame, meters, origins, context: int):
    ctx, keys = [], []
    for b in meters:
        s = sel[b].asfreq("h").ffill(limit=6)
        for o in origins:
            w = s[s.index < o].to_numpy(dtype="float32")[-context:]
            if len(w) < context:
                raise ValueError(f"{b} @ {o}: only {len(w)} context hours in bdg2_selected.parquet")
            ctx.append(w)
            keys.append((b, o))
    return ctx, keys


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--timing-only", action="store_true",
                    help="do not overwrite the forecasts; write eval/numeric/chronos_timing_<device>.json with the run "
                         "time and the largest q50 difference against the stored forecasts")
    args = ap.parse_args(argv)
    timing = {}

    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    import chronos
    import transformers
    from chronos import BaseChronosPipeline
    from huggingface_hub import try_to_load_from_cache

    log("torch", torch.__version__, "chronos", getattr(chronos, "__version__", "?"), "cuda", torch.cuda.is_available())
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    setup = json.loads((EVAL / "backtest_setup.json").read_text(encoding="utf-8"))
    cfg = setup["config"]
    meters = setup["selection"]["chosen"]
    sel = pd.read_parquet(EVAL / "bdg2_selected.parquet")
    sel.index = pd.to_datetime(sel.index)
    origins = pd.date_range(cfg["test_start"], cfg["last_origin"], freq="D")
    H, C = int(cfg["horizon"]), int(cfg["context"])
    ctx, keys = contexts(sel, meters, origins, C)
    log(f"{len(ctx)} contexts of {C} h; {int(sum(np.isnan(c).sum() for c in ctx))} NaN context values in total")

    for name in args.models:
        model_id, rev, tag = MODELS[name]
        t1 = time.time()
        pipe = BaseChronosPipeline.from_pretrained(model_id, revision=rev, device_map=device, torch_dtype=torch.float32)
        load_s = time.time() - t1
        cfg_path = try_to_load_from_cache(model_id, "config.json", revision=rev)
        snapshot = Path(cfg_path).parent.name if isinstance(cfg_path, str) else None
        log(model_id, "loaded", round(load_s, 1), "s on", device, "snapshot", snapshot)
        t2 = time.time()
        qs = []
        for i in range(0, len(ctx), args.batch):
            q, _ = pipe.predict_quantiles([torch.tensor(c) for c in ctx[i:i + args.batch]], prediction_length=H,
                                          quantile_levels=QUANTILES)
            if isinstance(q, list):  # Chronos-2: one (n_variates, H, Q) tensor per series
                q = torch.stack([x.reshape(-1, H, len(QUANTILES))[0] for x in q])
            qs.append(q.detach().float().cpu().numpy())
        q = np.concatenate(qs)
        pred_s = time.time() - t2
        log(model_id, f"predicted {len(ctx)} x {H} h in {pred_s:.1f} s")
        rows = []
        for (b, o), qq in zip(keys, q):
            ts = pd.date_range(o, periods=H, freq="h")
            rows.append(pd.DataFrame({"meter": b, "origin": o, "ts": ts, "h": np.arange(1, H + 1),
                                      "q10": qq[:, 0], "q50": qq[:, 1], "q90": qq[:, 2]}))
        out = pd.concat(rows, ignore_index=True)
        if args.timing_only:
            ref = pd.read_parquet(EVAL / f"forecasts_{tag}.parquet")
            ref_q50 = ref.sort_values(["meter", "origin", "h"]).q50.to_numpy()
            new_q50 = out.sort_values(["meter", "origin", "h"]).q50.to_numpy()
            timing[model_id] = {"revision": rev, "device": device, "torch_threads": torch.get_num_threads(),
                                "batch": args.batch, "n_forecasts": len(ctx), "horizon": H, "context": C,
                                "load_seconds": round(load_s, 2), "predict_seconds": round(pred_s, 2),
                                "max_abs_q50_diff_vs_stored_kWh": float(np.max(np.abs(new_q50 - ref_q50))),
                                "median_abs_q50_kWh": float(np.median(np.abs(ref_q50)))}
            del pipe
            continue
        out.to_parquet(EVAL / f"forecasts_{tag}.parquet", index=False)
        prov = {"model_id": model_id, "revision": rev, "resolved_snapshot": snapshot, "licence": LICENCE,
                "backend": "chronos-forecasting (zero-shot, univariate)", "context": C, "horizon": H,
                "quantiles": QUANTILES, "device": device, "batch": args.batch,
                "torch_threads": torch.get_num_threads(), "load_seconds": round(load_s, 2),
                "predict_seconds": round(pred_s, 2), "n_forecasts": len(ctx),
                "library_versions": {"torch": torch.__version__, "chronos": getattr(chronos, "__version__", None),
                                     "transformers": transformers.__version__, "python": sys.version.split()[0]},
                "data": "REAL: BDG2 electricity_cleaned.csv (via eval/numeric/bdg2_selected.parquet)",
                "imputation": "context forward-filled up to 6 h; longer gaps left NaN",
                "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}
        (EVAL / f"forecasts_{tag}.provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
    if args.timing_only:
        timing["_note"] = ("measured on a shared laptop with other jobs running; "
                           f"created {dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}")
        (EVAL / f"chronos_timing_{device}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
