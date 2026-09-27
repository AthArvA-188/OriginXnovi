"""SIMULATED drain-down test (pyswmm, cerebro_ml env) -> eval/clog/drain_test.json + eval/clog/drain_curves.csv.

For each orifice opening (1.0 clean ... 0.08) and night base flow, one SWMM run; then many noisy level-sensor draws
(sd 0.5 cm, 1 cm logging step, both Assumptions). Commissioning value = median clean drain-down on a calibration draw
set; alarm thresholds = 95th percentile of the clean calibration ratios; detection rates are measured on a separate
evaluation draw set. Compares drain-down time with peak depth as the clog signal.

    E:\\conda_envs\\cerebro_ml\\python.exe scripts/clog_drain_sim.py
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

import clog_sim_lib as L  # noqa: E402  (pyswmm; cerebro_ml only)
from cascade.building.clog import drain as DR  # noqa: E402
from cascade.building.clog import features as F  # noqa: E402
from cascade.building.clog import rules as R  # noqa: E402

OPENINGS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08]
BASE_LPS = [0.02, 0.05, 0.1]  # night base flow in the building drain [Assumption]
N_DRAWS = 60


def main() -> int:
    t0 = time.time()
    test = R.value("DRAIN_TEST")
    lvl = R.value("LEVEL_NOISE_M")
    bands = R.value("DRAIN_RATIO_BANDS")
    runs = {}
    for op in OPENINGS:
        for b in BASE_LPS:
            runs[(op, b)] = L.simulate_drain(op, b, test["lps"], test["seconds"], tmp_root=C.TMP)
    sim_s = time.time() - t0

    def draws(seed: int, op: float):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(N_DRAWS):
            b = BASE_LPS[rng.integers(0, len(BASE_LPS))]
            r = runs[(op, b)]
            d = F.quantise(r["cleanout"], rng, lvl["sd"], lvl["step"])
            dd, cens = F.drain_down_seconds(r["t"], d)
            out.append({"dd": dd, "censored": cens, "peak": float(d.max())})
        return pd.DataFrame(out)

    calib = draws(1, 1.0)
    comm_dd = float(calib.dd.median())
    comm_peak = float(calib.peak.median())
    thr_dd = float(np.quantile(calib.dd / comm_dd, 0.95))
    thr_peak = float(np.quantile(calib.peak / comm_peak, 0.95))
    rows = []
    for i, op in enumerate(OPENINGS):
        ev = draws(100 + i, op)
        ratio = ev.dd / comm_dd
        grades = [DR.grade_ratio(r, c, bands) for r, c in zip(ratio, ev.censored)]
        clean = runs[(op, 0.05)]
        dd0, c0 = F.drain_down_seconds(clean["t"], clean["cleanout"])
        rows.append({
            "opening": op, "n_draws": N_DRAWS,
            "noise_free_drain_down_s_at_0.05lps": dd0, "noise_free_censored": c0,
            "noise_free_peak_m_at_0.05lps": float(clean["cleanout"].max()),
            "noise_free_tau_s_at_0.05lps": F.fit_recession_tau(clean["t"], clean["cleanout"]),
            "drain_down_median_s": float(ev.dd.median()), "drain_down_p10_s": float(ev.dd.quantile(0.1)),
            "drain_down_p90_s": float(ev.dd.quantile(0.9)), "censored_share": float(ev.censored.mean()),
            "peak_median_m": float(ev.peak.median()),
            "detect_rate_drain_down": float(((ratio > thr_dd) | ev.censored).mean()),
            "detect_rate_peak": float((ev.peak / comm_peak > thr_peak).mean()),
            "share_watch_or_urgent": float(np.mean([g != "normal" for g in grades])),
            "share_urgent": float(np.mean([g == "urgent" for g in grades])),
        })
    out = {
        "label": "SIMULATED (pyswmm 2.1 building-drain model, synthetic geometry, assumed sensor noise)",
        "model": {"network": "stack base -> 10 m DN100 -> cleanout (level sensor) -> orifice CLOG -> 15 m DN100 -> sewer",
                  "routing_step_s": 1, "duration_s": DR.DURATION_S, "test": test, "base_flows_lps": BASE_LPS,
                  "level_noise": lvl, "drain_down_definition": "seconds from peak until depth < baseline + 2 cm; "
                                                               "censored if never within the 20-min record"},
        "commissioning": {"drain_down_median_s": comm_dd, "peak_median_m": comm_peak, "n_draws": N_DRAWS},
        "thresholds": {"drain_down_ratio_p95_clean": thr_dd, "peak_ratio_p95_clean": thr_peak,
                       "grading_bands": bands},
        "rows": rows, "sim_seconds": round(sim_s, 1), "n_swmm_runs": len(runs),
    }
    (C.EVAL / "drain_test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    curves = []
    for op in (1.0, 0.3, 0.15, 0.08):
        r = runs[(op, 0.05)]
        for tt, d in zip(r["t"][::5], r["cleanout"][::5]):
            curves.append({"opening": op, "t_s": float(tt), "cleanout_depth_m": round(float(d), 5)})
    pd.DataFrame(curves).to_csv(C.EVAL / "drain_curves.csv", index=False)
    print(pd.DataFrame(rows)[["opening", "noise_free_drain_down_s_at_0.05lps", "noise_free_peak_m_at_0.05lps",
                              "drain_down_median_s", "censored_share", "detect_rate_drain_down", "detect_rate_peak",
                              "share_urgent"]].to_string())
    print(f"thresholds dd {thr_dd:.3f} peak {thr_peak:.3f}; comm dd {comm_dd:.0f}s; {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
