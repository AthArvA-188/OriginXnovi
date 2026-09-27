"""Fit and evaluate the interior-wall concrete strength estimator (rebound + UPV), REAL data only.

Data: Matthews et al. NDT databases (Zenodo 15392443, CC BY 4.0) and the BAM round-robin
drilled cores (Harvard Dataverse doi:10.7910/DVN/AFCITK, CC0) as an external test.
Run scripts/interior/download_ndt.py first.

Evaluation design (never random rows as the headline):
1. Leave-one-study-out (LOSO) over every study in the SonReb database; pooled and per-study
   macro metrics; the in-situ subset (tests on real structures) is the building-use headline.
2. 5-fold GroupKFold by study (to compare with the research pilot).
3. Leakage contrast: random-row 5-fold KFold vs 5-fold GroupKFold, same model.
4. Per-building calibration: in each held-out study, k in {1,3,5} labelled specimens are drawn
   20 times; the power law is shifted by their mean log residual and scored on the rest.
   These are "k labelled specimens from the same study (mostly lab cubes)", not real cores.
5. Conformal 90% intervals, nested: for each held-out study the quantile is set from inner
   GroupKFold residuals of the OTHER studies only, then coverage is measured on the held-out
   study (k=0 and k=3).
6. Single-instrument fallbacks: RN-only on the rebound DB, Vp-only on the UPV DB, both LOSO.
7. External test on 20 BAM cores: rebound (Original Schmidt 'R' file only) and UPV were
   measured IN LABS ON THE CORES; core strength x the database's own core-to-cylinder factor.

Writes: models/interior/ndt_strength_v1.json (shipped coefficients + quantile tables),
models/interior/sonreb_hgb_v1.joblib (challenger), eval/interior/ndt_strength_v1.json,
eval/interior/loso_predictions.csv, eval/interior/bam_external.csv.

    python scripts/interior/ndt_strength.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cascade.interior.ndt import (COVERAGE, PowerLaw, calibration_shift, conformal_q, load_rebound,  # noqa: E402
                                  load_sonreb, load_upv, metrics)

RAW = ROOT / "data" / "raw" / "interior"
MATT = RAW / "matthews"
BAM = RAW / "bam"
EVAL = ROOT / "eval" / "interior"
MODELS = ROOT / "models" / "interior"
SEED = 0
K_LIST = (1, 3, 5)
N_DRAWS = 20
MIN_ROWS_CALIB = 10  # a study needs >= 10 rows to draw k=5 and still score >= 5 rows


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


def loso_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-study predictions for every row: mean, RN-only, Vp-only, SonReb power law, HGB(rn, vp)."""
    out = df[["study", "in_situ", "fc", "rn", "vp"]].copy()
    for c in ("pred_mean", "pred_rn_only", "pred_vp_only", "pred_sonreb", "pred_hgb", "logpred_sonreb"):
        out[c] = np.nan
    for s in df.study.unique():
        te = (df.study == s).to_numpy()
        tr = df[~te]
        out.loc[te, "pred_mean"] = tr.fc.mean()
        out.loc[te, "pred_rn_only"] = PowerLaw(("rn",)).fit(tr).predict(df[te])
        out.loc[te, "pred_vp_only"] = PowerLaw(("vp",)).fit(tr).predict(df[te])
        pl = PowerLaw(("rn", "vp")).fit(tr)
        out.loc[te, "logpred_sonreb"] = pl.predict_log(df[te])
        out.loc[te, "pred_sonreb"] = np.exp(out.loc[te, "logpred_sonreb"])
        out.loc[te, "pred_hgb"] = hgb().fit(tr[["rn", "vp"]].to_numpy(), tr.fc.to_numpy()).predict(df[te][["rn", "vp"]].to_numpy())
    return out


MODEL_COLS = {"training_mean": "pred_mean", "rn_only_powerlaw": "pred_rn_only", "vp_only_powerlaw": "pred_vp_only",
              "sonreb_powerlaw": "pred_sonreb", "hgb_rn_vp": "pred_hgb"}


def summarize(pred: pd.DataFrame) -> dict:
    res = {}
    for subset, mask in (("all", np.ones(len(pred), bool)), ("in_situ", pred.in_situ.to_numpy())):
        p = pred[mask]
        res[subset] = {"n_rows": int(len(p)), "n_studies": int(p.study.nunique()), "models": {}}
        for name, col in MODEL_COLS.items():
            m = metrics(p.fc.to_numpy(), p[col].to_numpy())
            per = p.groupby("study").apply(lambda g: float(np.mean(np.abs(g[col] - g.fc))), include_groups=False)
            m["macro_mae_by_study"] = float(per.mean())
            m["median_study_mae"] = float(per.median())
            res[subset]["models"][name] = m
        top = p.study.value_counts()
        res[subset]["largest_study"] = {"study": str(top.index[0]), "rows": int(top.iloc[0])}
    return res


def groupkfold_compare(df: pd.DataFrame) -> dict:
    y, g = df.fc.to_numpy(), df.study.to_numpy()
    X = df[["rn", "vp"]]
    out = {}
    for design, cv, groups in (("random_rows_kfold5", KFold(5, shuffle=True, random_state=SEED), None),
                               ("groupkfold5_by_study", GroupKFold(5), g)):
        ph, pp, pm = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            ph[te] = hgb().fit(X.values[tr], y[tr]).predict(X.values[te])
            pp[te] = PowerLaw(("rn", "vp")).fit(df.iloc[tr]).predict(df.iloc[te])
            pm[te] = y[tr].mean()
        out[design] = {"hgb_rn_vp": metrics(y, ph), "sonreb_powerlaw": metrics(y, pp), "training_mean": metrics(y, pm)}
    return out


def calibration_experiment(df: pd.DataFrame, pred: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    res = {}
    for subset in ("all", "in_situ"):
        rows = {k: [] for k in (0,) + K_LIST}
        per_study = {k: [] for k in (0,) + K_LIST}
        studies = [s for s in df.study.unique() if (df.study == s).sum() >= MIN_ROWS_CALIB]
        if subset == "in_situ":
            studies = [s for s in studies if df[df.study == s].in_situ.any()]
        for s in studies:
            m = (pred.study == s).to_numpy()
            if subset == "in_situ":
                m = m & pred.in_situ.to_numpy()
            if m.sum() < MIN_ROWS_CALIB:
                continue
            y = pred.fc.to_numpy()[m]
            lp = pred.logpred_sonreb.to_numpy()[m]
            rows[0].append(np.abs(np.exp(lp) - y))
            per_study[0].append(float(np.mean(np.abs(np.exp(lp) - y))))
            for k in K_LIST:
                errs = []
                for _ in range(N_DRAWS):
                    idx = rng.choice(len(y), k, replace=False)
                    keep = np.ones(len(y), bool)
                    keep[idx] = False
                    sh = calibration_shift(lp[idx], y[idx])
                    errs.append(np.abs(np.exp(lp[keep] + sh) - y[keep]))
                e = np.concatenate(errs)
                rows[k].append(e)
                per_study[k].append(float(np.mean(e)))
        res[subset] = {"n_studies": len(per_study[0]), "draws_per_study": N_DRAWS,
                       "note": "k labelled specimens from the same study (mostly lab cubes in 'all'), not real cores",
                       "by_k": {str(k): {"pooled_mae": float(np.mean(np.concatenate(rows[k]))),
                                         "macro_mae_by_study": float(np.mean(per_study[k])),
                                         "n_scored_errors": int(sum(len(r) for r in rows[k]))} for k in rows}}
    return res


def _inner_log_resid(tr: pd.DataFrame) -> pd.DataFrame:
    """Out-of-study log residuals of the power law inside the training studies (inner GroupKFold)."""
    lp = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(tr, groups=tr.study.to_numpy()):
        lp[b] = PowerLaw(("rn", "vp")).fit(tr.iloc[a]).predict_log(tr.iloc[b])
    return pd.DataFrame({"study": tr.study.to_numpy(), "lp": lp, "ly": np.log(tr.fc.to_numpy())})


def _q_after_calibration(inner: pd.DataFrame, k: int, rng) -> float:
    res = []
    for s, g in inner.groupby("study"):
        if len(g) < k + 1:
            continue
        lp, ly = g.lp.to_numpy(), g.ly.to_numpy()
        for _ in range(5):
            idx = rng.choice(len(g), k, replace=False)
            keep = np.ones(len(g), bool)
            keep[idx] = False
            sh = float(np.mean(ly[idx] - lp[idx]))
            res.append(np.abs(ly[keep] - lp[keep] - sh))
    return conformal_q(np.concatenate(res))


def conformal_nested(df: pd.DataFrame) -> dict:
    """Quantile from other studies only; coverage measured on the held-out study."""
    rng = np.random.default_rng(SEED + 1)
    cov = {0: [], 3: []}
    cov_insitu = {0: [], 3: []}
    per_study = {0: [], 3: []}
    widths = {0: [], 3: []}
    for s in df.study.unique():
        te = (df.study == s).to_numpy()
        tr = df[~te]
        inner = _inner_log_resid(tr)
        q0 = conformal_q(np.abs(inner.ly - inner.lp).to_numpy())
        q3 = _q_after_calibration(inner, 3, rng)
        pl = PowerLaw(("rn", "vp")).fit(tr)
        lp = pl.predict_log(df[te])
        ly = np.log(df[te].fc.to_numpy())
        ins = df[te].in_situ.to_numpy()
        hit0 = np.abs(ly - lp) <= q0
        cov[0].append(hit0)
        cov_insitu[0].append(hit0[ins])
        per_study[0].append(float(hit0.mean()))
        widths[0].append(q0)
        if te.sum() >= 4:
            hits = []
            hits_ins = []
            for _ in range(N_DRAWS):
                idx = rng.choice(te.sum(), 3, replace=False)
                keep = np.ones(te.sum(), bool)
                keep[idx] = False
                sh = float(np.mean(ly[idx] - lp[idx]))
                h = np.abs(ly[keep] - lp[keep] - sh) <= q3
                hits.append(h)
                hits_ins.append(h[ins[keep]])
            hh = np.concatenate(hits)
            cov[3].append(hh)
            cov_insitu[3].append(np.concatenate(hits_ins))
            per_study[3].append(float(hh.mean()))
            widths[3].append(q3)
    out = {"target_coverage": COVERAGE,
           "method": "split conformal on |ln fc - ln pred|; quantile from inner 5-fold GroupKFold residuals of the training studies; coverage on the held-out study (LOSO)"}
    for k in (0, 3):
        allh = np.concatenate(cov[k])
        insh = np.concatenate([c for c in cov_insitu[k] if len(c)])
        out[f"k{k}"] = {"pooled_row_coverage": float(allh.mean()), "n_rows_scored": int(len(allh)),
                        "in_situ_row_coverage": float(insh.mean()) if len(insh) else None, "n_in_situ_rows_scored": int(len(insh)),
                        "macro_study_coverage": float(np.mean(per_study[k])), "n_studies": len(per_study[k]),
                        "median_q_log": float(np.median(widths[k])),
                        "median_interval_factor": float(np.exp(np.median(widths[k])))}
    return out


def fallback_eval(df: pd.DataFrame, feats: tuple) -> dict:
    p = np.zeros(len(df))
    pm = np.zeros(len(df))
    lp = np.zeros(len(df))
    for s in df.study.unique():
        te = (df.study == s).to_numpy()
        law = PowerLaw(feats).fit(df[~te])
        lp[te] = law.predict_log(df[te])
        pm[te] = df[~te].fc.mean()
    p = np.exp(lp)
    y = df.fc.to_numpy()
    ins = df.in_situ.to_numpy()
    return {"n_rows": int(len(df)), "n_studies": int(df.study.nunique()),
            "n_in_situ_rows": int(ins.sum()), "n_in_situ_studies": int(df[ins].study.nunique()),
            "loso_powerlaw": metrics(y, p), "loso_training_mean": metrics(y, pm),
            "loso_powerlaw_in_situ": metrics(y[ins], p[ins]) if ins.sum() else None,
            "loso_training_mean_in_situ": metrics(y[ins], pm[ins]) if ins.sum() else None,
            "q_log_k0": conformal_q(np.abs(np.log(y) - lp)),
            "_lp": lp}


def q_table_by_k(df: pd.DataFrame, lp: np.ndarray, rng) -> dict:
    """Shipped post-calibration quantiles, from LOSO residuals of every study (k draws per study)."""
    out = {}
    ly = np.log(df.fc.to_numpy())
    for k in K_LIST:
        res = []
        for s in df.study.unique():
            m = (df.study == s).to_numpy()
            if m.sum() < k + 1:
                continue
            a, b = lp[m], ly[m]
            for _ in range(N_DRAWS):
                idx = rng.choice(m.sum(), k, replace=False)
                keep = np.ones(m.sum(), bool)
                keep[idx] = False
                res.append(np.abs(b[keep] - a[keep] - np.mean(b[idx] - a[idx])))
        out[str(k)] = conformal_q(np.concatenate(res))
    return out


# --- BAM external test ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    return str(name).strip().replace("/", "_")


def core_to_cyl_factor(path: Path) -> dict:
    """fc,cyl / fc,core from the database's own ~100x200 mm core rows (the DB normalises to 150x300)."""
    d = pd.read_csv(path)
    comp = d["Compression Specimen"].astype(str).str.strip()
    h = pd.to_numeric(d["Height (mm)"], errors="coerce")
    w = pd.to_numeric(d["Width/Diameter (mm)"], errors="coerce")
    fcore = pd.to_numeric(d["fc,core (MPa)"], errors="coerce")
    fcyl = pd.to_numeric(d["fc,cyl (MPa)"], errors="coerce")
    m = (comp == "Core") & w.between(95, 105) & h.between(190, 210) & (fcore > 0) & (fcyl > 0)
    r = (fcyl[m] / fcore[m])
    return {"factor": float(r.median()), "sd": float(r.std()), "n_rows": int(m.sum()),
            "source": "median fc,cyl/fc,core over Matthews RH-database rows with ~100x200 mm cores"}


def load_bam() -> pd.DataFrame:
    enc = "latin-1"
    cs = pd.read_csv(BAM / "core_compressive_strength.csv", sep=";", encoding=enc, skiprows=[1])
    cs = pd.DataFrame({"core": cs["Core Name"].map(_norm), "fc_core": pd.to_numeric(cs["Compression Strength"] if "Compression Strength" in cs else cs["Compressive Strength"], errors="coerce")})
    raw = pd.read_csv(BAM / "core_rn_R_summary.csv", sep=";", encoding=enc, header=None)
    labs = raw.iloc[0, 3:].astype(str).str.strip().tolist()
    data = raw.iloc[4:].reset_index(drop=True)
    rn_rows = []
    for _, r in data.iterrows():
        if pd.isna(r[1]):
            continue
        vals = pd.to_numeric(r.iloc[3:], errors="coerce").to_numpy()
        for lab in sorted(set(labs)):
            sel = np.array([lb == lab for lb in labs])
            v = vals[sel]
            v = v[~np.isnan(v)]
            if len(v):
                rn_rows.append({"core": _norm(r[1]), "rn_lab": lab.replace(" ", ""), "rn": float(np.median(v)), "n_impacts": int(len(v))})
    rn = pd.DataFrame(rn_rows)
    us_rows = []
    for i in range(1, 7):
        p = BAM / f"core_us_lab{i}.csv"
        if not p.exists():
            continue
        raw = pd.read_csv(p, sep=";", encoding=enc, header=None)
        hdr = raw.iloc[2].astype(str).str.strip().tolist()
        cols = [j for j, h in enumerate(hdr) if h == "UPV"]
        for _, r in raw.iloc[4:].iterrows():
            if pd.isna(r[1]):
                continue
            v = pd.to_numeric(r.iloc[cols], errors="coerce").to_numpy()
            v = v[~np.isnan(v)]
            if len(v):
                us_rows.append({"core": _norm(r[1]), "upv_lab": f"Lab{i}", "vp": float(np.mean(v)) * 1000.0, "n_upv": int(len(v))})
    us = pd.DataFrame(us_rows)
    return cs, rn, us


def bam_external(law: PowerLaw, q0: float, qk: dict, factor: float) -> tuple[dict, pd.DataFrame]:
    cs, rn, us = load_bam()
    cs["fc_cyl_equiv"] = cs.fc_core * factor
    cons = cs.merge(rn.groupby("core").rn.median().rename("rn"), on="core").merge(
        us.groupby("core").vp.mean().rename("vp"), on="core")
    cons["lp"] = law.predict_log(cons)
    cons["pred"] = np.exp(cons.lp)
    y = cons.fc_cyl_equiv.to_numpy()
    res = {"n_cores": int(len(cons)), "global": metrics(y, cons.pred.to_numpy()),
           "mean_signed_error_mpa": float(np.mean(cons.pred.to_numpy() - y)),
           "training_mean_baseline_note": "the Matthews training mean is reported for context",
           "interval_coverage_k0": float(np.mean(np.abs(np.log(y) - cons.lp) <= q0))}
    # per lab pair spread
    pair = []
    for (rl, g1) in rn.groupby("rn_lab"):
        for (ul, g2) in us.groupby("upv_lab"):
            d = cs.merge(g1[["core", "rn"]], on="core").merge(g2[["core", "vp"]], on="core")
            if len(d) < 5:
                continue
            m = metrics(d.fc_cyl_equiv.to_numpy(), law.predict(d))
            pair.append({"rn_lab": rl, "upv_lab": ul, "n": m["n"], "mae": m["mae"]})
    res["per_lab_pair"] = pair
    res["per_lab_pair_mae_range"] = [float(min(p["mae"] for p in pair)), float(max(p["mae"] for p in pair))] if pair else None
    # k=3 calibration on BAM: 3 cores calibrate, other 17 scored, 20 draws
    rng = np.random.default_rng(SEED + 2)
    errs, hits = [], []
    lp, ly = cons.lp.to_numpy(), np.log(y)
    for _ in range(N_DRAWS):
        idx = rng.choice(len(cons), 3, replace=False)
        keep = np.ones(len(cons), bool)
        keep[idx] = False
        sh = float(np.mean(ly[idx] - lp[idx]))
        errs.append(np.abs(np.exp(lp[keep] + sh) - y[keep]))
        hits.append(np.abs(ly[keep] - lp[keep] - sh) <= qk["3"])
    res["k3_calibrated"] = {"mae": float(np.mean(np.concatenate(errs))), "draws": N_DRAWS,
                            "n_scored_per_draw": int(len(cons) - 3),
                            "interval_coverage": float(np.mean(np.concatenate(hits)))}
    res["target_range_mpa"] = [float(y.min()), float(y.max())]
    res["rn_range"] = [float(cons.rn.min()), float(cons.rn.max())]
    res["vp_range_ms"] = [float(cons.vp.min()), float(cons.vp.max())]
    return res, cons


def main() -> int:
    t0 = time.time()
    son = load_sonreb(MATT / "Complete SonReb Database.csv")
    reb = load_rebound(MATT / "Complete RH Database.csv")
    upv = load_upv(MATT / "Complete UPV Database.csv")
    print(f"SonReb rows {len(son)} studies {son.study.nunique()} in-situ rows {int(son.in_situ.sum())} "
          f"in-situ studies {son[son.in_situ].study.nunique()}", flush=True)

    pred = loso_predict(son)
    loso = summarize(pred)
    print("LOSO all:", {k: round(v["mae"], 2) for k, v in loso["all"]["models"].items()}, flush=True)
    print("LOSO in-situ:", {k: round(v["mae"], 2) for k, v in loso["in_situ"]["models"].items()}, flush=True)
    gkf = groupkfold_compare(son)
    calib = calibration_experiment(son, pred)
    print("calibration:", {s: {k: round(v["pooled_mae"], 2) for k, v in calib[s]["by_k"].items()} for s in calib}, flush=True)
    conf = conformal_nested(son)
    print("conformal:", {k: conf[k] for k in ("k0", "k3")}, flush=True)
    fb_rn = fallback_eval(reb, ("rn",))
    fb_vp = fallback_eval(upv, ("vp",))

    # shipped models: fit on all rows
    rng = np.random.default_rng(SEED + 3)
    law_son = PowerLaw(("rn", "vp")).fit(son)
    lp_son = pred.logpred_sonreb.to_numpy()
    q0_son = conformal_q(np.abs(np.log(son.fc.to_numpy()) - lp_son))
    qk_son = q_table_by_k(son, lp_son, rng)
    law_rn = PowerLaw(("rn",)).fit(reb)
    law_vp = PowerLaw(("vp",)).fit(upv)
    qk_rn = q_table_by_k(reb, fb_rn["_lp"], rng)
    qk_vp = q_table_by_k(upv, fb_vp["_lp"], rng)
    factor = core_to_cyl_factor(MATT / "Complete RH Database.csv")
    bam, bam_df = bam_external(law_son, q0_son, qk_son, factor["factor"])
    bam["core_to_cylinder_factor"] = factor
    bam["training_mean_mae"] = float(np.mean(np.abs(son.fc.mean() - bam_df.fc_cyl_equiv)))
    print("BAM:", {k: bam[k] for k in ("n_cores", "global", "k3_calibrated", "training_mean_mae")}, flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    spec = {
        "id": "ndt_strength_v1", "created": stamp, "coverage": COVERAGE,
        "target": "fc,cyl (MPa): compressive strength normalised to a 150x300 mm cylinder (Matthews et al. database convention)",
        "data": "REAL: Matthews et al. 2025 NDT databases, Zenodo 10.5281/zenodo.15392443, CC BY 4.0",
        "models": {
            "sonreb": {"law": law_son.as_dict(), "fitted_on": f"SonReb database, {len(son)} rows, {son.study.nunique()} studies",
                       "q_log_k0": q0_son, "q_log_by_k": qk_son, "rn_range": [float(son.rn.min()), float(son.rn.max())],
                       "vp_range_ms": [float(son.vp.min()), float(son.vp.max())]},
            "rn_only": {"law": law_rn.as_dict(), "fitted_on": f"rebound database, {len(reb)} rows, {reb.study.nunique()} studies",
                        "q_log_k0": fb_rn["q_log_k0"], "q_log_by_k": qk_rn, "rn_range": [float(reb.rn.min()), float(reb.rn.max())]},
            "vp_only": {"law": law_vp.as_dict(), "fitted_on": f"UPV database, {len(upv)} rows, {upv.study.nunique()} studies",
                        "q_log_k0": fb_vp["q_log_k0"], "q_log_by_k": qk_vp, "vp_range_ms": [float(upv.vp.min()), float(upv.vp.max())]},
        },
        "interval_note": "q_log_k0 = conformal 90% quantile of |ln fc - ln pred| over leave-one-study-out residuals; q_log_by_k after shifting by k same-study specimens",
        "limits": ["estimate for an engineer, never a structural verdict", "calibrate with cores before any decision",
                   "rebound and UPV depend on moisture, carbonation, surface finish, aggregate and rebar; the database mixes all of these"],
    }
    (MODELS / "ndt_strength_v1.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    joblib.dump(hgb().fit(son[["rn", "vp"]].to_numpy(), son.fc.to_numpy()), MODELS / "sonreb_hgb_v1.joblib")

    for fb in (fb_rn, fb_vp):
        fb.pop("_lp")
    out = {
        "id": "ndt_strength_v1", "created": stamp, "seconds": round(time.time() - t0, 1),
        "label": "REAL data (Matthews et al. NDT databases; BAM drilled cores). No synthetic rows.",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "data": {"sonreb_rows": int(len(son)), "sonreb_studies": int(son.study.nunique()),
                 "sonreb_in_situ_rows": int(son.in_situ.sum()), "sonreb_in_situ_studies": int(son[son.in_situ].study.nunique()),
                 "rebound_rows": int(len(reb)), "rebound_studies": int(reb.study.nunique()),
                 "upv_rows": int(len(upv)), "upv_studies": int(upv.study.nunique()),
                 "cleaning": "vp > 500 m/s, rn > 5, fc > 0; rows missing rn/vp/fc dropped",
                 "fc_range_mpa": [float(son.fc.min()), float(son.fc.max())]},
        "loso": loso,
        "groupkfold_vs_random": gkf,
        "calibration": calib,
        "conformal": conf,
        "fallbacks": {"rn_only_on_rebound_db": fb_rn, "vp_only_on_upv_db": fb_vp},
        "bam_external": bam,
        "shipped": {"model_file": "models/interior/ndt_strength_v1.json", "q_log_k0_sonreb": q0_son, "q_log_by_k_sonreb": qk_son,
                    "coefficients": law_son.as_dict()},
        "published_comparison": {
            "claim": "Matthews et al. (2026) report R2 0.947 for SonReb (TPE-CatBoost)",
            "split": "10-fold k-fold cross-validation, 90/10, no grouping by study stated (Section 4.1.3)",
            "url": "https://publications.tno.nl/publication/34645091/4b25FibC/matthews-2026-advancing.pdf",
            "compare_with": "groupkfold_vs_random.random_rows_kfold5 (same split family), not the leave-study-out numbers"},
    }
    (EVAL / "ndt_strength_v1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pred.drop(columns=["logpred_sonreb"]).round(3).to_csv(EVAL / "loso_predictions.csv", index=False)
    bam_df.round(4).to_csv(EVAL / "bam_external.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
