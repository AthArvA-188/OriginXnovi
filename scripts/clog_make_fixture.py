"""Write tests/fixtures/clog/report_inputs.json: a small trimmed copy of eval/clog/*.json with only the fields that
cascade.building.clog.report reads, so the conclusion tests run without the full artifacts.

    python scripts/clog_make_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval" / "clog"
OUT = ROOT / "tests" / "fixtures" / "clog" / "report_inputs.json"


def load(name: str) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def slim_detector(d: dict) -> dict:
    out = {"label": d["label"], "alarm": {s: {"rate": a["rate"], "ci95": a["ci95"]} for s, a in d["alarm"].items()}}
    if "severity" in d:
        out["severity"] = {"macro_f1": d["severity"]["macro_f1"], "macro_f1_ci95": d["severity"]["macro_f1_ci95"]}
    if "localisation" in d:
        out["localisation"] = {s: {"accuracy": v["accuracy"], "ci95": v["ci95"]} for s, v in d["localisation"].items()
                               if isinstance(v, dict) and "accuracy" in v}
    return out


def main() -> int:
    riser, drain, bell, sched = (load("riser_metrics.json"), load("drain_test.json"), load("bellinge_case.json"),
                                 load("schedule.json"))
    d = riser["design"]
    fx = {
        "note": "Trimmed copy of eval/clog artifacts made by scripts/clog_make_fixture.py (SIMULATED/REAL labels as in "
                "the source files).",
        "riser": {
            "design": {k: d[k] for k in ("tests_lps", "locations", "noise", "k_bands")},
            "baseline_majority_macro_f1": riser["baseline_majority_macro_f1"],
            "baselines_macro_f1": riser["baselines_macro_f1"],
            "variants": {v: {"label": vr["label"],
                             "detectors": {k: slim_detector(x) for k, x in vr["detectors"].items()}}
                         for v, vr in riser["variants"].items()},
            "commissioning_sensitivity": {"act4_z": riser["commissioning_sensitivity"]["act4_z"]},
            "moderate_plus_by_location": {"act4_z": riser["moderate_plus_by_location"]["act4_z"]},
        },
        "drain": {"model": {"test": drain["model"]["test"]},
                  "rows": [{k: r[k] for k in ("opening", "detect_rate_drain_down", "detect_rate_peak")}
                           for r in drain["rows"]]},
        "bell": {"test_months": bell["test_months"],
                 "headline": {"paired": {k: bell["headline"]["paired"][k] for k in
                                         ("delay_min_from_onset", "other_test_episodes", "train_episodes")},
                              "residual_only": {k: bell["headline"]["residual_only"][k] for k in
                                                ("detected", "delay_min_from_onset", "other_test_episodes",
                                                 "train_episodes")},
                              "fixed_levels": [{k: f[k] for k in ("label", "first_alarm", "delay_min_from_onset",
                                                                  "other_test_episodes", "train_episodes")}
                                               for f in bell["headline"]["fixed_levels"]]},
                 "sweep": {"paired_summary": bell["sweep"]["paired_summary"]}},
        "sched": {"as_of": sched["as_of"], "rows": [{"days_from_as_of": r["days_from_as_of"]} for r in sched["rows"]]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fx, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
