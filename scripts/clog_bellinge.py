"""REAL Bellinge July 2020 blockage case study -> eval/clog/bellinge_case.json + eval/clog/bellinge_jul2020_5min.csv.

Runs in either env (pandas + scikit-learn). Caches a 5-min 2020 extract of the two gauges in data/raw/clog/.

    python scripts/clog_bellinge.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clog_common as C  # noqa: E402

C.bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cascade.building.clog import bellinge as B  # noqa: E402

UP_FILE = "G71F04R_Level1_System2000p2_proc_v6.csv"
DN_FILE = "G71F06R_LevelInlet_System2000p1_proc_v6.csv"
CACHE = C.RAW / "bellinge_2020_5min.csv"
QUANTILES = [0.99, 0.995, 0.999]
RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]
WINDOWS = [3, 7, 14]
HEADLINE = {"q": 0.999, "ratio": 0.7, "win": 7}  # the research pilot's configuration, fixed before this build


def read_depth(fn: str) -> pd.Series:
    parts = []
    for ch in pd.read_csv(C.RAW / fn, usecols=["time", "depth_s"], chunksize=2_000_000):
        ch = ch[ch["time"].str.startswith("2020")]
        if len(ch):
            parts.append(ch)
    df = pd.concat(parts)
    s = pd.Series(df["depth_s"].to_numpy(float), index=pd.to_datetime(df["time"]))
    s = s[(s.index >= "2020-01-01") & (s.index < "2020-10-12")]
    return s.resample(B.STEP).mean()


def load() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    df = pd.DataFrame({"up": read_depth(UP_FILE), "dn": read_depth(DN_FILE)})
    df.to_csv(CACHE)
    return df


def main() -> int:
    t0 = time.time()
    raw = load()
    X = B.make_features(raw["up"], raw["dn"])
    res, res_s, model = B.fit_residuals(X)
    ons = B.onset(X["up"])
    tr = (X.index >= B.TRAIN[0]) & (X.index < B.TRAIN[1])
    te_ok = (X.index >= B.TEST[0]) & (X.index < B.TEST[1]) & ~((X.index >= B.EVENT[0]) & (X.index < B.EVENT[1]))
    mae_oof = float(res[tr].abs().mean())
    mae_test = float(res[te_ok].abs().mean())
    up_tr = X.loc[tr, "up"].dropna()

    # fixed-level baselines
    fixed = []
    for label, thr in [("1.0 m", 1.0)] + [(f"train p{q * 100:g}", float(up_tr.quantile(q))) for q in QUANTILES[1:]] + \
            [("train p99", float(up_tr.quantile(0.99)))]:
        s = B.summarise(X["up"] > thr, ons)
        fixed.append({"detector": "fixed level", "label": label, "threshold_m": thr, **s})
    fixed.sort(key=lambda r: r["threshold_m"])

    # residual-only and paired sweeps (thresholds from time-blocked OOF train residuals)
    res_rows, paired_rows = [], []
    for q in QUANTILES:
        thr = float(res_s[tr].dropna().quantile(q))
        res_rows.append({"detector": "residual only", "q": q, "threshold_m": thr, **B.summarise(res_s > thr, ons)})
        for ratio in RATIOS:
            for win in WINDOWS:
                s = B.summarise(B.paired_mask(res_s, X["dn"], thr, ratio, win), ons)
                paired_rows.append({"detector": "paired", "q": q, "threshold_m": thr, "ratio": ratio, "win_days": win,
                                    **s})
    dn_only = B.summarise(X["dn"].rolling(6, min_periods=4).median() <
                          0.7 * X["dn"].rolling(B.PER_DAY * 7, min_periods=B.PER_DAY).median(), ons)
    head = next(r for r in paired_rows if r["q"] == HEADLINE["q"] and r["ratio"] == HEADLINE["ratio"]
                and r["win_days"] == HEADLINE["win"])
    det = [r for r in paired_rows if r["detected"]]
    daily = X[["up", "dn"]].resample("D").mean()

    def dm(day, col):
        return float(daily.loc[day, col])

    out = {
        "label": "REAL (Bellinge sensor data, CC BY 4.0) - municipal combined sewer, 1 documented event, case study",
        "source": {"dataset": "Pedersen et al. 2021, Dataset for Bellinge", "doi": "https://doi.org/10.11583/DTU.c.5029124",
                   "event_log": "https://ndownloader.figshare.com/files/30592500",
                   "quote": "23-07-2020 - 28-07-2020 ... The throttle pipe between sensor G71F04R and G71F06R was "
                            "blocked. This means that water entered the volume pipe."},
        "gauges": {"upstream": "G71F04R Level 1 (depth_s)", "downstream": "G71F06R Level inlet (depth_s)"},
        "periods": {"train": [str(B.TRAIN[0].date()), "2020-06-30"], "test": [str(B.TEST[0].date()), "2020-10-11"],
                    "event": ["2020-07-23", "2020-07-28"]},
        "n_5min_rows": {"train": int(tr.sum()), "test": int(((X.index >= B.TEST[0]) & (X.index < B.TEST[1])).sum())},
        "test_months": (B.TEST[1] - B.TEST[0]).days / 30.44,
        "onset": {"time": str(ons), "definition": "first 5-min step in the documented window where upstream depth "
                                                  "exceeds its previous-24-h median by more than 0.10 m"},
        "daily_means_m": {"up_2020-07-22": dm("2020-07-22", "up"), "up_2020-07-24": dm("2020-07-24", "up"),
                          "dn_2020-07-22": dm("2020-07-22", "dn"), "dn_2020-07-24": dm("2020-07-24", "dn")},
        "normal_model": {"type": "HistGradientBoostingRegressor(max_iter=300)", "features": B.FEATS,
                         "mae_train_oof_m": mae_oof, "mae_test_excl_event_m": mae_test,
                         "threshold_basis": "quantile of the 30-min median residual on time-blocked (calendar-month) "
                                            "out-of-fold training residuals"},
        "headline": {"paired": head, "config": HEADLINE,
                     "fixed_levels": fixed, "residual_only": next(r for r in res_rows if r["q"] == HEADLINE["q"]),
                     "downstream_drop_only": dn_only},
        "sweep": {"paired": paired_rows, "residual_only": res_rows,
                  "paired_summary": {
                      "n_configs": len(paired_rows), "n_detected": len(det),
                      "first_alarm_range": [min(r["first_alarm"] for r in det), max(r["first_alarm"] for r in det)]
                      if det else None,
                      "other_test_episodes_range": [min(r["other_test_episodes"] for r in paired_rows),
                                                    max(r["other_test_episodes"] for r in paired_rows)],
                      "train_episodes_range": [min(r["train_episodes"] for r in paired_rows),
                                               max(r["train_episodes"] for r in paired_rows)]}},
        "caveats": [
            "One documented event: detection time and false-alarm counts are a case study, not accuracy or recall.",
            "Municipal combined sewer throttle pipe between basins (1.7 km2 catchment), not a building drain.",
            "Readme: G71F04R Level 1/2 scaling changed to 0-2 m instead of 0-2.9 m from 06-01-2020 to about "
            "19-11-2020 (depth x1.45 in the cleaning script); the whole Jan-Oct 2020 window is inside that period.",
            "The event log gives dates only; delays are measured from the visible onset in the data.",
            "Rain is not used (Bellinge rain gauges are CC BY-NC 4.0); some other alarms may be wet weather.",
            "The paired-rule parameters were chosen after the event was known (research pilot); the sweep shows how "
            "sensitive the result is to them.",
            "Fixed-level thresholds are train-period quantiles, so their train episode counts are in-sample.",
        ],
        "seconds": round(time.time() - t0, 1),
    }
    (C.EVAL / "bellinge_case.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    # chart extract 20-31 July 2020
    w = (X.index >= "2020-07-20") & (X.index < "2020-08-01")
    thr_h = head["threshold_m"]
    ext = pd.DataFrame({"up_m": X.loc[w, "up"], "dn_m": X.loc[w, "dn"], "residual_30min_m": res_s[w],
                        "paired_alarm": B.paired_mask(res_s, X["dn"], thr_h, HEADLINE["ratio"], HEADLINE["win"])[w]})
    ext.index.name = "time_local"
    ext.round(4).to_csv(C.EVAL / "bellinge_jul2020_5min.csv")
    print(json.dumps({k: out[k] for k in ("onset", "daily_means_m", "normal_model")}, indent=1, default=str))
    print("PAIRED headline:", {k: head[k] for k in ("first_alarm", "delay_min_from_onset", "other_test_episodes",
                                                    "train_episodes")})
    for f in fixed:
        print("FIXED", f["label"], round(f["threshold_m"], 3), f["first_alarm"], f["other_test_episodes"],
              f["train_episodes"])
    for r in res_rows:
        print("RESID", r["q"], round(r["threshold_m"], 4), r["first_alarm"], r["other_test_episodes"], r["train_episodes"])
    print("SWEEP", out["sweep"]["paired_summary"])
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
