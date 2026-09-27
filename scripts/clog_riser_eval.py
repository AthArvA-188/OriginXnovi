"""Evaluate riser clog detectors on the SIMULATED scenarios (numpy/pandas/scikit-learn; either env).

Reads data/raw/clog/sims/riser_*.npz + eval/clog/riser_scenarios.csv (from clog_riser_sim.py), adds sensor noise,
builds per-night features, fits every detector on the TRAIN building/weeks only, sets alarm thresholds on
out-of-fold clean training nights (GroupKFold by week, 5% target), and evaluates once on the held-out TEST map (the
same 28 HSB apartments reassigned to units with another seed, same simulated riser) and 2022-23 weeks, plus three
out-of-distribution stress variants and one constant-offset cancellation check. CIs are a two-stage cluster bootstrap
(demand weeks, then scenarios within week), because the scenarios of one week share the same demand.

Writes eval/clog/riser_metrics.json, eval/clog/riser_test_nights.csv and models/clog/riser_act4.joblib.

    python scripts/clog_riser_eval.py [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clog_common as C  # noqa: E402

C.bootstrap()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cascade.building.clog import detect as DT  # noqa: E402
from cascade.building.clog import features as F  # noqa: E402
from cascade.building.clog import geometry as G  # noqa: E402
from cascade.building.clog import rules as R  # noqa: E402

NOISE_SEED = 11
OFFSET_M = 0.3
VARIANTS = {  # name -> sims file, commissioning source (variant, building), noise multiplier, offset drift
    "base": ("base", "base", 1.0, False),
    "ood_noise2": ("base", "base", 2.0, False),
    "ood_offset": ("base", "base", 1.0, True),
    "ood_aged": ("ood_aged", "base", 1.0, False),
    "ood_demand": ("ood_demand", "ood_demand", 1.0, False),
}
VARIANT_LABEL = {
    "base": "Held-out test (different unit map, 2022-23 weeks)",
    "ood_noise2": "Stress: sensor noise x2",
    "ood_offset": f"Check: constant +{OFFSET_M} m offset on one pressure sensor (cancels in the active test)",
    "ood_aged": "Stress: pipes aged since commissioning (C 130 -> 110)",
    "ood_demand": "Stress: LA-like busier tower (demand x2.5)",
}
LOC = list(range(len(G.LOCATIONS)))
COLS = {
    "ACT_A": [f"actA_{j}" for j in LOC], "ACT_B": [f"actB_{j}" for j in LOC],
    "PAS": [f"pas_{j}" for j in LOC] + ["pasq_0", "pasq_1", "pasq_2"], "PAS8": [f"pas_{j}" for j in LOC],
}
DETECTORS = {  # key -> (feature set, kind, label)
    "act4_z": ("ACT_A", "z", "Active test 4.0 L/s, z-score rule (no ML)"),
    "act4_hgb": ("ACT_A", "hgb", "Active test 4.0 L/s, gradient boosting"),
    "act25_z": ("ACT_B", "z", "Active test 2.5 L/s, z-score rule (no ML)"),
    "act25_hgb": ("ACT_B", "hgb", "Active test 2.5 L/s, gradient boosting"),
    "pas_z": ("PAS8", "z", "Passive 24 h, z-score rule (no ML)"),
    "pas_hgb": ("PAS", "hgb", "Passive 24 h, gradient boosting"),
    "bms": ("BMS", "thr", "BMS low-pressure alarm at zone tops (baseline)"),
}
K_BINS = [1, 3, 10, 30, 100, 300.0001]


def noised(p, q, sim_id, code, mult, offset, draw=0):
    """Noise draws depend only on the simulation (and `draw`), not on the stress variant, so variants are paired:
    the same scenario sees the same noise pattern, scaled x2 in the noise stress test."""
    rng = np.random.default_rng([NOISE_SEED, int(sim_id), int(draw)])
    off_rng = np.random.default_rng([NOISE_SEED + 1, int(sim_id)])
    off = (int(off_rng.integers(0, p.shape[1])), OFFSET_M) if offset else None
    return F.add_noise(p, q, rng, p_sd=R.value("PRESSURE_NOISE_M"), rel=R.value("FLOW_NOISE")["rel"],
                       abs_v=R.value("FLOW_NOISE")["abs_velocity_ms"], mult=mult, offset=off)


def rows_for(meta_row, p, q, ref, variant, code, mult, offset):
    pn, qn = noised(p, q, meta_row.sim_id, code, mult, offset)
    b = F.night_rows(pn, qn, ref)
    out = []
    for d in range(b["actA"].shape[0]):
        r = {"variant": variant, "split": meta_row.split, "scenario": int(meta_row.sim_id), "week": meta_row.week,
             "day": d, "sev": int(meta_row.sev), "K": float(meta_row.K), "loc": meta_row.loc if meta_row.sev else "none",
             "loc_id": G.LOCATIONS.index(meta_row.loc) if meta_row.sev else -1, "bms_drop": float(b["bms_drop"][d])}
        for j in LOC:
            r[f"actA_{j}"] = b["actA"][d, j]
            r[f"actB_{j}"] = b["actB"][d, j]
            r[f"rawA_{j}"] = b["actA"][d, j] + ref["rA"][j]
            r[f"rawB_{j}"] = b["actB"][d, j] + ref["rB"][j]
            r[f"pas_{j}"] = b["pas"][d, j]
        for z in range(3):
            r[f"pasq_{z}"] = b["pas_q"][d, z]
            r[f"dqA_{z}"] = b["dqA"][d, z]
        out.append(r)
    return out


def build_tables(meta: pd.DataFrame) -> pd.DataFrame:
    sims = {}
    for v in ("base", "ood_aged", "ood_demand"):
        with np.load(C.SIMS / f"riser_{v}.npz") as z:
            P, Q, ids = z["p"], z["q"], z["sim_id"]
        sims[v] = {int(i): (P[k], Q[k]) for k, i in enumerate(ids)}
    rows = []
    for code, (variant, (simv, commv, mult, offset)) in enumerate(VARIANTS.items()):
        m = meta[meta.variant == simv]
        splits = ("train", "test") if variant == "base" else ("test",)
        for split in splits:
            cm = meta[(meta.variant == commv) & (meta.split == split) & (meta.role == "commission")].iloc[0]
            cp, cq = sims[commv][int(cm.sim_id)]
            # commissioning is measured with the same sensors (same noise level) but before any offset drift
            ref = F.commissioning(*noised(cp, cq, cm.sim_id, code, mult, False))
            for row in m[(m.split == split) & (m.role == "scenario")].itertuples():
                p, q = sims[simv][int(row.sim_id)]
                rows.extend(rows_for(row, p, q, ref, variant, code, mult, offset))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    t0 = time.time()
    meta = pd.read_csv(C.EVAL / "riser_scenarios.csv")
    df = build_tables(meta)
    print(f"night rows {len(df)} in {time.time() - t0:.0f}s")
    tr = df[(df.variant == "base") & (df.split == "train")].reset_index(drop=True)
    ytr, gtr = tr.sev.to_numpy(), tr.week.to_numpy()
    fits = {}
    for key, (fs, kind, _label) in DETECTORS.items():
        if kind == "thr":
            thr = float(np.quantile(tr.loc[tr.sev == 0, "bms_drop"], 0.95))
            fits[key] = {"kind": kind, "thr": thr}
            continue
        X = tr[COLS[fs]].to_numpy(float)
        oof = DT.oof_scores(X, ytr, gtr, kind)
        thr = DT.threshold_at(oof, ytr, R.value("FALSE_ALARM_TARGET"))
        if kind == "hgb":
            fits[key] = {"kind": kind, "thr": thr, "model": DT.hgb().fit(X, ytr),
                         "oof_clean_fa": float((oof[ytr == 0] > thr).mean())}
        else:
            zr = DT.ZRule().fit(X, ytr)
            fits[key] = {"kind": kind, "thr": thr, "model": zr, "cuts": DT.z_cutpoints(oof, ytr, thr),
                         "oof_clean_fa": float((oof[ytr == 0] > thr).mean())}
    locs = {}
    for fs in ("ACT_A", "ACT_B"):
        c = tr[tr.sev > 0]
        locs[fs] = DT.hgb().fit(c[COLS[fs]].to_numpy(float), c.loc_id.to_numpy())
    print(f"fitted in {time.time() - t0:.0f}s")

    def apply(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for key, (fs, kind, _l) in DETECTORS.items():
            f = fits[key]
            if kind == "thr":
                d[f"{key}_flag"] = d["bms_drop"] > f["thr"]
                continue
            X = d[COLS[fs]].to_numpy(float)
            if kind == "hgb":
                m = f["model"]
                score = 1.0 - m.predict_proba(X)[:, list(m.classes_).index(0)]
                d[f"{key}_grade"] = m.predict(X)
            else:
                score = f["model"].maxz(X)
                d[f"{key}_grade"] = DT.z_grade(score, f["thr"], f["cuts"])
                d[f"{key}_loc"] = f["model"].locate(X)
            d[f"{key}_score"] = score
            d[f"{key}_flag"] = score > f["thr"]
        d["act4_hgb_loc"] = locs["ACT_A"].predict(d[COLS["ACT_A"]].to_numpy(float))
        d["act25_hgb_loc"] = locs["ACT_B"].predict(d[COLS["ACT_B"]].to_numpy(float))
        return d

    results = {}
    tests = {}
    for variant in VARIANTS:
        te = apply(df[(df.variant == variant) & (df.split == "test")])
        tests[variant] = te
        vres = {"label": VARIANT_LABEL[variant], "n_scenarios": int(te.scenario.nunique()), "n_nights": int(len(te)),
                "scenarios_by_severity": {str(k): int(v) for k, v in te.groupby("sev").scenario.nunique().items()},
                "detectors": {}}
        for key, (_fs, kind, label) in DETECTORS.items():
            r = {"label": label, "alarm": DT.alarm_rates(te, f"{key}_flag", args.boot)}
            if kind != "thr":
                r["severity"] = DT.severity_metrics(te, f"{key}_grade", args.boot)
            if key.startswith("act"):
                r["localisation"] = DT.localisation(te, f"{key}_loc", args.boot)
            vres["detectors"][key] = r
        results[variant] = vres
        print(variant, {k: {s: round(v["rate"], 3) for s, v in r["alarm"].items()}
                        for k, r in vres["detectors"].items()})
    base = tests["base"]
    kbin = {}
    for key in DETECTORS:
        rows = []
        for lo, hi in zip(K_BINS[:-1], K_BINS[1:]):
            d = base[(base.K >= lo) & (base.K < hi)]
            ps = d.groupby("scenario")[f"{key}_flag"].mean()
            wk = d.groupby("scenario")["week"].first().reindex(ps.index).to_numpy()
            rows.append({"k_lo": lo, "k_hi": round(hi), "rate": float(ps.mean()),
                         "ci95": DT.boot_ci(ps.to_numpy(), args.boot, 0, wk), "n_scenarios": int(len(ps))})
        kbin[key] = rows
    per_loc = {}
    for key in ("act4_z", "act4_hgb", "act25_hgb"):
        rows = []
        for loc in G.LOCATIONS:
            d = base[(base["loc"] == loc) & (base.sev >= 2)]
            ps = d.groupby("scenario")[f"{key}_flag"].mean()
            rows.append({"loc": loc, "name": G.LOCATION_NAMES[loc], "rate": float(ps.mean()), "n_scenarios": int(len(ps))})
        per_loc[key] = rows
    maj = DT.majority_f1(ytr, base.sev.to_numpy())
    # grading baselines at the unit the macro-F1 is scored on (nights); class names from the severity bands
    sev_names = list(R.value("K_BANDS").keys())
    bl = DT.baseline_f1s(ytr, base.sev.to_numpy())
    names = ["clean"] + sev_names
    bl["majority_class_name"] = names[bl["majority_class"]]
    bl["class_names"] = names
    bl["unit"] = "test nights; training class frequencies from training nights"
    # commissioning sensitivity: the test building's reference comes from ONE noisy clean week; redraw it 20 times
    cm = meta[(meta.variant == "base") & (meta.split == "test") & (meta.role == "commission")].iloc[0]
    with np.load(C.SIMS / "riser_base.npz") as z:
        k = int(np.nonzero(z["sim_id"] == cm.sim_id)[0][0])
        cp, cq = z["p"][k], z["q"][k]
    sens = {key: {"0": [], "2": [], "3": []} for key in ("act4_z", "act4_hgb", "act25_z", "act25_hgb")}
    for draw in range(1, 21):
        ref = F.commissioning(*noised(cp, cq, cm.sim_id, 0, 1.0, False, draw=draw))
        d = base.copy()
        for j in LOC:
            d[f"actA_{j}"] = d[f"rawA_{j}"] - ref["rA"][j]
            d[f"actB_{j}"] = d[f"rawB_{j}"] - ref["rB"][j]
        for key in sens:
            fs, kind, _l = DETECTORS[key]
            X = d[COLS[fs]].to_numpy(float)
            f = fits[key]
            if kind == "hgb":
                sc = 1.0 - f["model"].predict_proba(X)[:, list(f["model"].classes_).index(0)]
            else:
                sc = f["model"].maxz(X)
            flag = pd.Series(sc > f["thr"], index=d.index)
            for sv in ("0", "2", "3"):
                m_ = d.sev == int(sv)
                sens[key][sv].append(float(flag[m_].groupby(d.loc[m_, "scenario"]).mean().mean()))
    comm_sens = {key: {sv: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                            "p10": float(np.quantile(v, 0.1)), "p90": float(np.quantile(v, 0.9)), "n_draws": len(v)}
                       for sv, v in dd.items()} for key, dd in sens.items()}
    # strainer demo readings for the scheduler (SIMULATED test scenarios)
    demo = {}
    j_l, j_m = G.LOCATIONS.index("STR_L"), G.LOCATIONS.index("STR_M")
    clean_s = base[base.sev == 0].scenario.min()
    mod_s = base[(base["loc"] == "STR_M") & (base.sev == 2)].scenario.min()
    for name, sid, j in [("STR_L", clean_s, j_l), ("STR_M", mod_s, j_m)]:
        d = base[base.scenario == sid]
        flags = d.sort_values("day")["act4_z_flag"].to_numpy()
        kk, nn = R.value("PERSISTENCE_2_OF_3")
        demo[name] = {"scenario": int(sid), "true_K": float(d.K.iloc[0]), "true_severity": int(d.sev.iloc[0]),
                      "median_extra_headloss_m_at_4lps": float(np.nanmedian(d[f"actA_{j}"]) * 16.0),
                      "nights": int(len(d)), "act4_z_alarm_nights": int(flags.sum()),
                      "act4_z_night_flags": [bool(x) for x in flags],
                      "act4_z_persistent_2of3": bool(DT.persistent(flags, kk, nn)),
                      "act4_z_located_here_nights": int((d["act4_z_loc"] == j).sum())}
    out = {
        "label": "SIMULATED: synthetic riser and clogs (WNTR 1.5.0), REAL HSB Living Lab demand (CC BY 4.0)",
        "design": {
            "unit_of_evaluation": "scenario = one building-week with one clog state (7 nightly tests)",
            "train": {"building": "train map (seed 101): 28 HSB apartments assigned to the tower's units",
                      "weeks": sorted(tr.week.unique().tolist()),
                      "n_scenarios": int(tr.scenario.nunique()), "n_nights": int(len(tr))},
            "test": {"building": "test map (seed 202): the same 28 HSB apartments reassigned to units, same simulated "
                                 "riser geometry", "weeks": sorted(base.week.unique().tolist()),
                     "n_weeks": int(base.week.nunique())},
            "commissioning": "one clean week per building, earlier than all its scenario weeks",
            "threshold": "95th percentile of out-of-fold P(clog) or max-z on clean training nights (GroupKFold by week)",
            "ci": f"95% two-stage cluster bootstrap ({args.boot} resamples): the {base.week.nunique()} test demand "
                  f"weeks with replacement, then scenarios within each drawn week (within true severity for macro-F1)",
            "k_bands": R.value("K_BANDS"), "tests_lps": R.value("ACTIVE_TEST_LPS"),
            "noise": {"pressure_sd_m": R.value("PRESSURE_NOISE_M"), "flow": R.value("FLOW_NOISE")},
            "locations": G.LOCATIONS, "location_names": G.LOCATION_NAMES,
        },
        "thresholds": {k: {kk: vv for kk, vv in f.items() if kk in ("thr", "cuts", "oof_clean_fa")}
                       for k, f in fits.items()},
        "baseline_majority_macro_f1": maj,
        "baselines_macro_f1": bl,
        "variants": results, "commissioning_sensitivity": comm_sens, "rate_by_k": kbin, "moderate_plus_by_location": per_loc, "strainer_demo": demo,
        "seconds": round(time.time() - t0, 1),
    }
    (C.EVAL / "riser_metrics.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    keep = ["scenario", "week", "day", "sev", "K", "loc"] + [c for c in base.columns if c.endswith(("_flag", "_grade"))]
    base[keep].to_csv(C.EVAL / "riser_test_nights.csv", index=False)
    joblib.dump({"severity_hgb": fits["act4_hgb"]["model"], "location_hgb": locs["ACT_A"],
                 "zrule": {"mu": fits["act4_z"]["model"].mu, "sd": fits["act4_z"]["model"].sd,
                           "thr": fits["act4_z"]["thr"], "cuts": fits["act4_z"]["cuts"]},
                 "hgb_thr": fits["act4_hgb"]["thr"], "features": COLS["ACT_A"], "locations": G.LOCATIONS,
                 "note": "SIMULATED training data; advisory only"}, C.MODELS / "riser_act4.joblib", compress=3)
    print("majority macroF1", round(maj, 3), bl["majority_class_name"], "baselines", bl["constant_macro_f1"],
          bl["stratified_random"])
    for k in ("act4_hgb", "act4_z", "act25_hgb", "pas_hgb"):
        s = results["base"]["detectors"][k].get("severity")
        if s:
            print(k, "macroF1", round(s["macro_f1"], 3), [round(x, 3) for x in s["macro_f1_ci95"]])
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
