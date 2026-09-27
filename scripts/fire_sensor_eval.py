"""Fire alarm verifier evaluation on REAL Mendeley multi-sensor data (ask 5 Part B, key "fire").

Stages (run in order; each writes artifacts under eval/fire/):
  prep  - check or download both CSVs (sha256 vs the Mendeley API), write the provenance manifest, segment episodes,
          compute causal features and cache them as parquet under data/raw/fire/ (git-ignored).
  cv    - within-site: leave-one-recording-day-out on the EN54 room (thresholds and model from the other days only),
          for the headline feature set and the pre-registered ablation. Then FREEZE: write frozen_config.json.
  hall  - cross-site, held out: train on all EN54, test once on the unseen Industrial Hall. Refuses to run when the
          current settings differ from frozen_config.json. Every run is appended to hall_runs.log.

Usage (origin_hack env):
  python scripts/fire_sensor_eval.py --stage prep
  python scripts/fire_sensor_eval.py --stage cv
  python scripts/fire_sensor_eval.py --stage hall
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cascade.building import firesense as fs  # noqa: E402

RAW = ROOT / "data" / "raw" / "fire"
OUT = ROOT / "eval" / "fire"
MODELS = ROOT / "models" / "fire"
RESULTS = OUT / "sensor_eval.json"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fs.json_safe(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def read_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def eps_summary(eps: pd.DataFrame) -> dict:
    return {"n": int(len(eps)), "by_kind": {k: int(v) for k, v in eps["kind"].value_counts().items()},
            "by_scenario": {f"{k[0]}:{k[1]}": int(v) for k, v in eps.groupby(["kind", "scenario"]).size().items()}}


def stage_prep() -> None:
    t0 = time.time()
    man = fs.fetch(RAW, download=True)
    for k, d in man["datasets"].items():
        if not d["sha256_ok"]:
            raise SystemExit(f"sha256 mismatch for {k}: {d['sha256_measured']}")
    man["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(RAW / "manifest.json", man)
    write_json(OUT / "dataset_manifest.json", man)
    for site, loader in (("en54", fs.episodes_en54), ("hall", fs.episodes_hall)):
        df = fs.load_site(RAW / fs.DATASETS[site]["file"], site)
        eps = loader(df)
        prep = fs.prepare(df, eps)
        keep = ["t", "ts", "sensor", "site", "scen", "day", "warm", "ep", "kind", "clean_bg"] + list(fs.FEATURE_SETS["full"])
        prep[keep].to_parquet(RAW / f"prepared_{site}.parquet", index=False)
        eps.drop(columns=["start_s", "end_s"]).to_csv(OUT / f"episodes_{site}.csv", index=False)
        log(f"{site}: {len(df)} rows, {df['sensor'].nunique()} sensors, episodes {eps_summary(eps)['by_kind']}, "
            f"clean background rows {int(prep['clean_bg'].sum())}")
    log(f"prep done in {time.time() - t0:.1f} s")


def load_prepared(site: str):
    df = pd.read_parquet(RAW / f"prepared_{site}.parquet")
    df = df.sort_values(["sensor", "t"], kind="stable").reset_index(drop=True)
    eps = pd.read_csv(OUT / f"episodes_{site}.csv", parse_dates=["start", "end"])
    eps["start"] = pd.to_datetime(eps["start"], utc=True, format="mixed")
    eps["end"] = pd.to_datetime(eps["end"], utc=True, format="mixed")
    return df, fs.with_secs(eps)


def stage_cv() -> None:
    t0 = time.time()
    df, eps = load_prepared("en54")
    res = read_results()
    res["frozen_config"] = fs.FROZEN
    res["frozen_hash"] = fs.frozen_hash()
    res["methods"] = fs.METHOD_LABEL
    res["episodes"] = {"en54": eps_summary(eps)}
    cv = {}
    for variant in (fs.FROZEN["headline_features"], fs.FROZEN["ablation_features"]):
        feats = fs.FEATURE_SETS[variant]
        log(f"CV EN54 leave-one-day-out, features={variant} ({len(feats)})")
        run, folds = fs.cv_en54(df, eps, feats, log=log)
        metrics, events = fs.evaluate_run(run)
        cv[variant] = {"folds": folds, "metrics": metrics, "n_features": len(feats)}
        tab = fs.episode_table(run, events)
        tab.to_csv(OUT / f"episodes_cv_en54_{variant}.csv", index=False)
        for m in fs.METHODS:
            r = metrics[m]
            log(f"  {m:10s} fire {r['fire']['alarmed']}/{r['fire']['n']} nuis {r['nuisance']['alarmed']}/{r['nuisance']['n']} "
                f"lat_med {r['latency_min']['median']} bg/24h {r['background']['per_24_sensor_h']} (n={r['background']['alarms']})")
        log(f"  AUROC {metrics['stage2_auroc']}")
    res["cv_en54"] = cv
    res["cv_en54_meta"] = {"design": fs.FROZEN["within_site_cv"], "runtime_s": round(time.time() - t0, 1),
                           "run_at": datetime.now(timezone.utc).isoformat()}
    write_json(RESULTS, res)
    frozen = {"frozen_config": fs.FROZEN, "hash": fs.frozen_hash(), "frozen_at": datetime.now(timezone.utc).isoformat(),
              "note": "All settings fixed before the Industrial Hall was scored. The hall stage refuses to run when these differ."}
    write_json(OUT / "frozen_config.json", frozen)
    log(f"cv done in {time.time() - t0:.1f} s; frozen hash {fs.frozen_hash()}")


def stage_hall(reason: str) -> None:
    import joblib
    import sklearn

    t0 = time.time()
    fz = json.loads((OUT / "frozen_config.json").read_text(encoding="utf-8"))
    if fz["hash"] != fs.frozen_hash():
        raise SystemExit(f"settings changed since the freeze ({fz['hash']} vs {fs.frozen_hash()}); refusing to score the Hall")
    tr, _ = load_prepared("en54")
    te, eps = load_prepared("hall")
    res = read_results()
    res["episodes"] = {**res.get("episodes", {}), "hall": eps_summary(eps)}
    thr_pm = fs.stage1_threshold(tr, fs.FROZEN["stage1_channel"], fs.FROZEN["stage1_quantile"])
    thr_co = fs.stage1_threshold(tr, fs.FROZEN["co_channel"], fs.FROZEN["stage1_quantile"])
    hall = {"thr_pm": thr_pm, "thr_co": thr_co}
    MODELS.mkdir(parents=True, exist_ok=True)
    for variant in (fs.FROZEN["headline_features"], fs.FROZEN["ablation_features"]):
        feats = fs.FEATURE_SETS[variant]
        log(f"Hall cross-site, features={variant}")
        clf, info = fs.train_stage2(tr, feats)
        mpath = MODELS / f"verifier_hgb_{variant}.joblib"
        joblib.dump({"model": clf, "features": list(feats), "frozen_hash": fs.frozen_hash(), "thr_pm": thr_pm, "thr_co": thr_co,
                     "trained_on": fs.DATASETS["en54"]["doi"], "sklearn": sklearn.__version__}, mpath, compress=3)
        d = te.copy()
        d["pfire"] = fs.p_fire(clf, d, feats)
        d["thr_pm"], d["thr_co"] = thr_pm, thr_co
        run = fs.SiteRun(df=d, eps=eps, thr_pm=thr_pm, thr_co=thr_co)
        metrics, events = fs.evaluate_run(run)
        hall[variant] = {"metrics": metrics, "train": info, "n_features": len(feats), "model_file": str(mpath.relative_to(ROOT)).replace("\\", "/"),
                         "model_bytes": mpath.stat().st_size}
        fs.episode_table(run, events).to_csv(OUT / f"episodes_hall_{variant}.csv", index=False)
        for m in fs.METHODS:
            r = metrics[m]
            log(f"  {m:10s} fire {r['fire']['alarmed']}/{r['fire']['n']} nuis {r['nuisance']['alarmed']}/{r['nuisance']['n']} "
                f"other {r['other']['alarmed']}/{r['other']['n']} lat_med {r['latency_min']['median']} "
                f"bg/24h {r['background']['per_24_sensor_h']} (n={r['background']['alarms']}, h={r['background']['sensor_hours']:.1f})")
        log(f"  AUROC {metrics['stage2_auroc']}")
        if variant == fs.FROZEN["headline_features"]:
            rep = fs.replay_frames(run, events)
            rep.to_csv(OUT / "replay_hall.csv", index=False)
            trig = fs.events_for_method(run, "two_stage_all")
            trig["t_utc"] = pd.to_datetime(trig["t"], unit="s", utc=True).astype(str)
            trig[["t_utc", "sensor", "attr", "pmax", "verified"]].to_csv(OUT / "triggers_hall.csv", index=False)
    runs = (OUT / "hall_runs.log").read_text(encoding="utf-8").splitlines() if (OUT / "hall_runs.log").exists() else []
    hall["meta"] = {"design": fs.FROZEN["cross_site"], "runtime_s": round(time.time() - t0, 1), "run_at": datetime.now(timezone.utc).isoformat(),
                    "frozen_hash": fs.frozen_hash(), "frozen_at": fz["frozen_at"], "hall_run_number": len(runs) + 1, "reason": reason,
                    "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    res["hall"] = hall
    write_json(RESULTS, res)
    with (OUT / "hall_runs.log").open("a", encoding="utf-8") as f:
        f.write(f"{hall['meta']['run_at']} run {len(runs) + 1} frozen_hash {fs.frozen_hash()} reason: {reason}\n")
    log(f"hall done in {time.time() - t0:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prep", "cv", "hall", "all"], required=True)
    ap.add_argument("--reason", default="first scoring of the frozen configuration", help="logged with every Hall run")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("prep", "all"):
        stage_prep()
    if a.stage in ("cv", "all"):
        stage_cv()
    if a.stage in ("hall", "all"):
        stage_hall(a.reason)


if __name__ == "__main__":
    main()
