"""Simulate the SYNTHETIC 32-floor riser with REAL HSB demand (run in the cerebro_ml env: needs wntr 1.5.0).

Design (verifier descope): 16 train weeks (2020-2021) and 8 held-out test weeks (2022-23), a train building and a
different test building (different apartment-to-unit maps), each with a clean commissioning week taken from an
EARLIER week. Per week: 6 clean scenarios + 8 locations x 3 clogs with K drawn log-uniform in [1, 300].
Out-of-distribution stress variants reuse the test scenarios: aged pipe (C 130 -> 110 after commissioning) and an
LA-like busier tower (demand x2.5, with its own commissioning week).

Writes noise-free sensor arrays to data/raw/clog/sims/riser_<variant>.npz (git-ignored), the scenario table to
eval/clog/riser_scenarios.csv and an engine cross-check to eval/clog/sim_check.json.

    E:\\conda_envs\\cerebro_ml\\python.exe scripts/clog_riser_sim.py [--jobs 3] [--train-weeks 16] [--test-weeks 8]
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from cascade.building.clog import demand as D  # noqa: E402
from cascade.building.clog import geometry as G  # noqa: E402
from cascade.building.clog import rules as R  # noqa: E402
from cascade.building.clog import sim_riser as S  # noqa: E402
import clog_sim_lib as L  # noqa: E402  (wntr; cerebro_ml only)

SEED = 2026
K_RANGE = (1.0, 300.0)
CLEAN_PER_WEEK = 6
CLOGS_PER_LOC = 3
AGED_C = 110.0
LA_DEMAND_SCALE = 2.5


def _run(task: dict, dem: np.ndarray) -> dict:
    import clog_common

    clog_common.bootstrap()
    import clog_sim_lib as L
    from cascade.building.clog import sim_riser as S

    r = L.simulate(S.add_tests(dem), task["pipe"] or None, task["K"], task["roughness"], tmp_root=C.TMP)
    return {"sim_id": task["sim_id"], "p": r["p"], "q": r["q"], "engine": r["engine"],
            "nonconverged_steps": r["nonconverged_steps"], "seconds": round(r["seconds"], 3)}


def engine_check(apt, week, bmap) -> dict:
    """EPANET (with CONTINUE) vs WNTRSimulator on a few cases: max absolute pressure difference."""
    import wntr

    dem = S.add_tests(D.floor_demands(D.week_matrix(apt, week), bmap))
    rows = []
    for pipe, K in [(None, 0.0), ("P04", 300.0), ("STR_L", 300.0), ("P30", 30.0)]:
        a = L.simulate(dem, pipe, K, tmp_root=C.TMP)
        t0 = time.time()
        b = wntr.sim.WNTRSimulator(L.build_network(dem, pipe, K)).run_sim()
        pb = b.node["pressure"][G.PRESSURE_SENSORS].to_numpy()
        rows.append({"clog_pipe": pipe, "K": K, "epanet_engine": a["engine"], "epanet_s": round(a["seconds"], 3),
                     "wntrsim_s": round(time.time() - t0, 2),
                     "max_abs_diff_m": round(float(np.abs(a["p"] - pb).max()), 5)})
    return {"week": str(week.date()), "cases": rows,
            "note": "Pressure difference between EPANET (unbalanced=CONTINUE 10) and WNTR's own solver, all sensors, "
                    "all 1,008 steps."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--train-weeks", type=int, default=16)
    ap.add_argument("--test-weeks", type=int, default=8)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    apt = D.load_hsb_apartments(C.RAW / "hsb.zip")
    weeks = D.good_weeks(apt)
    train_all, test_all = D.split_weeks(weeks)
    comm = {"train": train_all[0], "test": train_all[-1]}  # commissioning weeks, earlier than every scenario week
    pool = [w for w in train_all if w not in comm.values()]
    rng = np.random.default_rng(SEED)
    wk = {"train": sorted(rng.choice(np.array(pool), args.train_weeks, replace=False)),
          "test": sorted(rng.choice(np.array(test_all), args.test_weeks, replace=False))}
    apts = list(apt.columns)
    bmaps = {"train": D.building_map(apts, 101), "test": D.building_map(apts, 202)}
    bands = R.value("K_BANDS")
    print(f"HSB apartments={len(apts)} good weeks={len(weeks)} (train pool {len(train_all)}, test pool {len(test_all)})"
          f" load {time.time() - t0:.1f}s")

    tasks = []

    def add(variant, split, building, week, role, pipe, K, loc, roughness=G.ROUGHNESS_C, scale=1.0):
        tasks.append({"sim_id": len(tasks), "variant": variant, "split": split, "building": building,
                      "week": str(pd.Timestamp(week).date()), "role": role, "loc": loc, "pipe": pipe or "",
                      "K": float(K), "sev": G.severity_of_k(K, bands), "roughness": roughness, "demand_scale": scale})

    for split in ("train", "test"):
        add("base", split, split, comm[split], "commission", None, 0.0, "")
        for w in wk[split]:
            for _ in range(CLEAN_PER_WEEK):
                add("base", split, split, w, "scenario", None, 0.0, "none")
            for loc in G.LOCATIONS:
                for _ in range(CLOGS_PER_LOC):
                    K = float(np.exp(rng.uniform(np.log(K_RANGE[0]), np.log(K_RANGE[1]))))
                    add("base", split, split, w, "scenario", str(rng.choice(G.location_pipes(loc))), K, loc)
    base_test = [t for t in tasks if t["variant"] == "base" and t["split"] == "test" and t["role"] == "scenario"]
    for t in base_test:  # aged pipe: same scenarios, commissioning stays the new-pipe base reference
        add("ood_aged", "test", "test", t["week"], "scenario", t["pipe"] or None, t["K"], t["loc"], roughness=AGED_C)
    add("ood_demand", "test", "test", comm["test"], "commission", None, 0.0, "", scale=LA_DEMAND_SCALE)
    for t in base_test:
        add("ood_demand", "test", "test", t["week"], "scenario", t["pipe"] or None, t["K"], t["loc"],
            scale=LA_DEMAND_SCALE)
    print(f"tasks={len(tasks)}")

    week_cache = {}

    def dem_for(t):
        key = t["week"]
        if key not in week_cache:
            week_cache[key] = D.week_matrix(apt, pd.Timestamp(key))
        return D.floor_demands(week_cache[key], bmaps[t["building"]], t["demand_scale"])

    t1 = time.time()
    results = Parallel(n_jobs=args.jobs, batch_size=4, verbose=0)(delayed(_run)(t, dem_for(t)) for t in tasks)
    print(f"simulated {len(results)} weeks in {time.time() - t1:.0f}s")
    res = {r["sim_id"]: r for r in results}
    meta = pd.DataFrame(tasks)
    meta["engine"] = [res[i]["engine"] for i in meta.sim_id]
    meta["nonconverged_steps"] = [res[i]["nonconverged_steps"] for i in meta.sim_id]
    meta["seconds"] = [res[i]["seconds"] for i in meta.sim_id]
    meta["synthetic"] = True
    for variant, g in meta.groupby("variant"):
        ids = g.sim_id.to_numpy()
        np.savez_compressed(C.SIMS / f"riser_{variant}.npz", sim_id=ids,
                            p=np.stack([res[i]["p"] for i in ids]), q=np.stack([res[i]["q"] for i in ids]))
    meta.to_csv(C.EVAL / "riser_scenarios.csv", index=False)
    summary = {
        "generated_by": "scripts/clog_riser_sim.py", "seed": SEED, "label": "SIMULATED (synthetic riser, REAL HSB demand)",
        "weeks": {k: [str(pd.Timestamp(w).date()) for w in v] for k, v in wk.items()},
        "commissioning_weeks": {k: str(v.date()) for k, v in comm.items()},
        "hsb_good_weeks": {"train_pool": len(train_all), "test_pool": len(test_all)},
        "n_sims": int(len(meta)), "engine_counts": meta.engine.value_counts().to_dict(),
        "sims_with_nonconverged_steps": int((meta.nonconverged_steps > 0).sum()),
        "median_seconds_per_week": float(meta.seconds.median()), "wall_seconds": round(time.time() - t0, 1),
        "k_range": K_RANGE, "k_bands": bands, "aged_C": AGED_C, "la_demand_scale": LA_DEMAND_SCALE,
    }
    if not args.skip_check:
        summary["engine_check"] = engine_check(apt, wk["train"][0], bmaps["train"])
    (C.EVAL / "sim_check.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "weeks"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
