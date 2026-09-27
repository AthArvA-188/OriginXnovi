"""Anomaly evaluation on LEAD1.0-small (REAL hourly meter readings with HUMAN anomaly labels), runtime env.

Split by building (60 held-out buildings, seed 0; the other 140 train the supervised model). Scorers:
  - robust_z: residual vs the median of the same hour over the previous 4 weeks / trailing median |residual| (no training)
  - isolation_forest: unsupervised, fitted on the training buildings' features
  - hgb_supervised: HistGradientBoostingClassifier trained on the training buildings' human labels
Metrics on the held-out buildings: ROC-AUC, PR-AUC (average precision; chance = prevalence), precision and recall
at a fixed alert budget (top 1% of scores), a building-level bootstrap 95% interval for PR-AUC, and for robust_z
the operating point at K = ANOMALY_MAD_K on the building layer's 1.4826 x MAD scale (plus, as a labelled
sensitivity row, the same K in raw MADs, which is what an earlier version of RobustZ used).

LEAD states no licence: eval/numeric/anomaly_lead.json holds AGGREGATE numbers only. A labelled timeline for one
held-out building is written to data/raw/numeric/ (git-ignored) for local viewing behind a flag, never published.

Run: C:\\Users\\HP\\miniconda3\\envs\\origin_hack\\python.exe scripts/numeric_anomaly_lead.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from cascade.numeric import datasets as D  # noqa: E402
from cascade.numeric.agent import THRESHOLD_SOURCE, _anomaly_threshold  # noqa: E402
from cascade.numeric.baselines import MAD_TO_SD  # noqa: E402
from cascade.numeric.base import EVAL_DIR, RAW_DIR, git_head, utc_now  # noqa: E402
from cascade.numeric.tabular import ANOMALY_FEATURES, HGBAnomalyClassifier, IsoForestScorer, anomaly_features  # noqa: E402


def log(t0, *a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def at_budget(y: np.ndarray, s: np.ndarray, budget: float = 0.01) -> dict:
    thr = float(np.quantile(s, 1 - budget))
    f = s >= thr
    tp = int((f & (y == 1)).sum())
    return {"budget_share": budget, "n_flagged": int(f.sum()), "precision": tp / max(1, int(f.sum())),
            "recall": tp / max(1, int(y.sum()))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)
    t0 = time.time()
    raw = D.load_lead()
    log(t0, f"LEAD rows {len(raw)}, buildings {raw.building_id.nunique()}, anomalies {int(raw.anomaly.sum())}")
    feats = []
    for b, g in raw.groupby("building_id", sort=True):
        s = g.set_index("timestamp")
        X = anomaly_features(s.meter_reading)
        X["anomaly"] = s.anomaly.reindex(X.index)
        X["building_id"] = b
        feats.append(X.reset_index(names="timestamp"))
    F = pd.concat(feats, ignore_index=True)
    F = F[F.y.notna() & F.anomaly.notna()].copy()
    F["anomaly"] = F.anomaly.astype(int)
    log(t0, f"features: {len(F)} scored rows")
    blds = np.array(sorted(F.building_id.unique()))
    rng = np.random.default_rng(args.seed)
    test_b = sorted(rng.choice(blds, size=args.n_test, replace=False).tolist())
    te = F[F.building_id.isin(test_b)].reset_index(drop=True)
    tr = F[~F.building_id.isin(test_b)].reset_index(drop=True)
    y = te.anomaly.to_numpy()

    scores, fit_s = {}, {}
    scores["robust_z"] = te.robust_z.fillna(0).to_numpy()
    fit_s["robust_z"] = 0.0
    t1 = time.time()
    iso = IsoForestScorer().fit(tr)
    scores["isolation_forest"] = iso.score(te)
    fit_s["isolation_forest"] = round(time.time() - t1, 1)
    t1 = time.time()
    clf = HGBAnomalyClassifier().fit(tr, tr.anomaly)
    scores["hgb_supervised"] = clf.score(te)
    fit_s["hgb_supervised"] = round(time.time() - t1, 1)
    log(t0, "scored", {k: v for k, v in fit_s.items()})

    # building-level bootstrap for PR-AUC
    groups = te.groupby("building_id").indices
    keys = list(groups)
    boots = rng.integers(0, len(keys), size=(args.n_boot, len(keys)))
    out_models = {}
    for name, s in scores.items():
        pr = []
        for row in boots:
            idx = np.concatenate([groups[keys[i]] for i in row])
            if y[idx].sum() > 0:
                pr.append(average_precision_score(y[idx], s[idx]))
        out_models[name] = {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                            "pr_auc_ci95_building_bootstrap": [float(np.percentile(pr, 2.5)),
                                                               float(np.percentile(pr, 97.5))],
                            "at_top_1pct": at_budget(y, s), "fit_seconds": fit_s[name]}
    k = _anomaly_threshold()

    def op_point(f: np.ndarray) -> dict:
        return {"n_flagged": int(f.sum()), "flag_share": float(f.mean()),
                "precision": float((f & (y == 1)).sum() / max(1, f.sum())),
                "recall": float((f & (y == 1)).sum() / max(1, y.sum()))}

    out_models["robust_z"]["at_threshold"] = {
        "threshold": k, "unit": f"{MAD_TO_SD} x MAD (sd-equivalents), same scale as electrical.baseline_anomalies",
        "source": THRESHOLD_SOURCE, **op_point(scores["robust_z"] >= k)}
    out_models["robust_z"]["at_threshold_raw_mads"] = {
        "threshold": k, "unit": "raw MADs (sensitivity only; NOT the building layer's scale)",
        "source": "robust_z x 1.4826 >= K, i.e. K raw median absolute residuals (up to the 1e-6 guard term)",
        **op_point(scores["robust_z"] * MAD_TO_SD >= k)}
    card = D.data_card("lead_small")
    res = {"created_utc": utc_now(), "code_git_sha": git_head(), "sklearn_version": sklearn.__version__,
           "data": {"name": card.name, "label": "REAL (human anomaly labels)", "url": card.source_url,
                    "licence": card.licence, "sha256": card.sha256, "accessed": card.accessed, "citation": card.note,
                    "publication": "aggregate metrics only; the labelled series are not redistributed"},
           "split": {"by": "building", "seed": args.seed, "n_test_buildings": len(test_b),
                     "n_train_buildings": int(len(blds) - len(test_b))},
           "n_rows_test": int(len(te)), "n_anomalies_test": int(y.sum()), "prevalence_test": float(y.mean()),
           "chance_pr_auc": float(y.mean()), "features": ANOMALY_FEATURES, "models": out_models,
           "notes": ["robust_z needs no training and is the baseline",
                     "rows with a missing reading or label are dropped before scoring",
                     "top-1% budget threshold is taken on the held-out scores (an alert budget, not a tuned cut-off)"]}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "anomaly_lead.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # local-only labelled timeline (git-ignored path): the held-out building with the most anomalies
    top = te.groupby("building_id").anomaly.sum().idxmax()
    m = te.building_id == top
    tl = pd.DataFrame({"timestamp": te.timestamp[m], "y": te.y[m], "anomaly_label": te.anomaly[m],
                       **{f"score_{n}": s[m.to_numpy()] for n, s in scores.items()}})
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tl.to_parquet(RAW_DIR / "lead_timeline_local.parquet", index=False)
    log(t0, json.dumps({n: {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in d.items()
                            if kk in ("roc_auc", "pr_auc")} for n, d in out_models.items()}))


if __name__ == "__main__":
    main()
