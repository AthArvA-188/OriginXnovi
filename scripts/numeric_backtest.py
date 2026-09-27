"""Step 1 of the numeric backtest (runtime env, scikit-learn only, no torch).

REAL data: BDG2 office electricity meters (hourly kWh). Draws the test meters (seed 0, pre-test eligibility only,
GIFT-Eval-Pretrain sites excluded), then for every daily origin 2017-11-01 .. 2017-12-24 (24 h ahead):
  - seasonal naive (lag 168) and naive (lag 24) baselines,
  - HGB + split-conformal 80% band from ONE model (train < 2017-10-01, calibrate on October 2017),
  - optionally HGB refit through 2017-10-31 (point only, no band), labelled as such.
Writes eval/numeric/bdg2_selected.parquet, eval/numeric/forecasts_baselines_hgb.parquet,
eval/numeric/backtest_setup.json and models/numeric/hgb_bdg2.joblib. backtest_setup.json also holds
"selection_diagnostics": the pool sizes and per-meter flat-day numbers behind the disclosed flat-day rule.
--diagnostics-only recomputes just those diagnostics, checks that the draw is unchanged, and updates the JSON in place.

Run: C:\\Users\\HP\\miniconda3\\envs\\origin_hack\\python.exe scripts/numeric_backtest.py --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from cascade.numeric import datasets as D  # noqa: E402
from cascade.numeric import metrics as M  # noqa: E402
from cascade.numeric.backtest import BacktestConfig, baseline_frame, hgb_frame  # noqa: E402
from cascade.numeric.base import EVAL_DIR, MODELS_DIR, git_head, utc_now  # noqa: E402


def log(t0: float, *a) -> None:
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--no-refit", action="store_true", help="skip the point-only HGB refit through October")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="only (re)write selection_diagnostics into the existing backtest_setup.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = BacktestConfig()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    meta = D.load_metadata()
    header = pd.read_csv(D.fetch("bdg2_electricity"), nrows=0).columns
    office = [b for b in meta.loc[meta.primaryspaceusage == "Office", "building_id"] if b in header]
    elec = D.load_electricity(office)
    log(t0, f"loaded {len(office)} office meters, {len(elec)} hours")
    chosen, info = D.select_offices(meta, elec, n=args.n, seed=args.seed, test_start=cfg.test_start)
    log(t0, f"eligible {info['n_eligible_before_exclusion']} -> {info['n_eligible_after_exclusion']} after "
            f"excluding GIFT-Eval sites; chosen {len(chosen)}")
    diag = D.selection_diagnostics(meta, elec, n=args.n, seed=args.seed, test_start=cfg.test_start)
    diag["created_utc"] = utc_now()
    at = diag["pool_at_chosen_limit"]
    assert (at["before_exclusion"], at["after_exclusion"]) == (info["n_eligible_before_exclusion"],
                                                                info["n_eligible_after_exclusion"]), (at, info)
    log(t0, "diagnostics: no flat rule", diag["pool_without_flat_rule"], "removed", list(diag["removed_by_flat_rule"]))
    if args.diagnostics_only:
        path = EVAL_DIR / "backtest_setup.json"
        setup = json.loads(path.read_text(encoding="utf-8"))
        if setup["selection"]["chosen"] != chosen:
            raise SystemExit("the draw differs from backtest_setup.json; run the full backtest instead")
        setup["selection_diagnostics"] = diag
        path.write_text(json.dumps(setup, indent=1), encoding="utf-8")
        log(t0, "selection_diagnostics written to", path)
        return

    t_start = pd.Timestamp(cfg.test_start)
    sel = elec[chosen].loc["2017-08-01":"2017-12-31"]
    sel.index.name = "timestamp"
    sel.to_parquet(EVAL_DIR / "bdg2_selected.parquet")
    scales = {b: M.mase_scale(elec[b][elec.index < t_start].to_numpy(), cfg.mase_period) for b in chosen}

    frames, models, fit_s = [], {}, {}
    for b in chosen:
        t1 = time.time()
        bf = baseline_frame(elec[b], b, cfg)
        hf, ms = hgb_frame(elec[b], b, cfg, max_iter=args.max_iter, point_only_refit=not args.no_refit)
        frames.append(bf.merge(hf, on=["meter", "ts"], how="left"))
        models[b] = ms["hgb"]
        fit_s[b] = round(time.time() - t1, 2)
        log(t0, f"{b}: band [{ms['hgb'].lo:+.2f}, {ms['hgb'].hi:+.2f}] kWh, {fit_s[b]} s")
    fr = pd.concat(frames, ignore_index=True)
    fr.to_parquet(EVAL_DIR / "forecasts_baselines_hgb.parquet", index=False)
    joblib.dump({"models": models, "created_utc": utc_now(), "sklearn": sklearn.__version__,
                 "note": "HGB + split-conformal per BDG2 meter; train < 2017-10-01, calibrated on Oct 2017"},
                MODELS_DIR / "hgb_bdg2.joblib", compress=3)

    card = D.data_card("bdg2_electricity")
    setup = {"created_utc": utc_now(), "code_git_sha": git_head(), "config": cfg.to_dict(), "selection": info,
             "selection_diagnostics": diag,
             "mase_scale_kWh": scales, "hgb_params": {"max_iter": args.max_iter, "learning_rate": 0.05,
                                                     "random_state": 0, "lags_h": [24, 48, 72, 168, 336],
                                                     "calendar": ["hour", "dow"]},
             "hgb_fit_seconds_per_meter": fit_s, "hgb_refit_point_only": not args.no_refit,
             "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
             "data": {"name": card.name, "label": card.label, "url": card.source_url, "licence": card.licence,
                      "sha256": card.sha256, "accessed": card.accessed, "citation": card.note},
             "imputation": "causal only: forward fill up to 6 h for lags/contexts; targets are raw and NaN targets "
                           "are not scored"}
    (EVAL_DIR / "backtest_setup.json").write_text(json.dumps(setup, indent=1), encoding="utf-8")
    log(t0, "done", len(fr), "rows")


if __name__ == "__main__":
    main()
