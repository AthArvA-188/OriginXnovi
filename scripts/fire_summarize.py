"""POST-HOC sensitivity summary of the Industrial Hall result (ask 5 Part B). Reads saved artifacts only; nothing is
re-trained or re-scored.

Defined AFTER the single Hall run, because the Hall episodes are packed closely and many first alarms fell inside the
pre-registered 5-minute early window before the labelled start. This summary counts, for the stage-1 PM trigger and
the two-stage label, how many episodes still have a trigger inside [label start, label end] (strict window), and how
many were counted only through the early window. The pre-registered numbers in sensor_eval.json stay the headline.

Inputs:  eval/fire/triggers_hall.csv (every stage-1 trigger with its stage-2 label), eval/fire/episodes_hall.csv
Outputs: eval/fire/sensitivity_hall.json; eval/fire/exact_intervals.json (Clopper-Pearson intervals for every count);
         eval/fire/episode_counts.json
Usage:   python scripts/fire_summarize.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "fire"


def main() -> None:
    trig = pd.read_csv(OUT / "triggers_hall.csv")
    trig["t"] = pd.to_datetime(trig["t_utc"], utc=True, format="mixed")
    eps = pd.read_csv(OUT / "episodes_hall.csv")
    eps["start"] = pd.to_datetime(eps["start"], utc=True, format="mixed")
    eps["end"] = pd.to_datetime(eps["end"], utc=True, format="mixed")
    res = {"post_hoc": True, "defined_after_hall_run": True, "generated_at": datetime.now(timezone.utc).isoformat(),
           "source_files": ["eval/fire/triggers_hall.csv", "eval/fire/episodes_hall.csv"], "methods": {}}
    for method, sel in (("stage1_pm", trig), ("two_stage", trig[trig["verified"]])):
        rows = {}
        for kind in ("fire", "nuisance", "other"):
            e = eps[eps["kind"] == kind]
            strict = early_only = 0
            for ep in e.itertuples():
                mine = sel[sel["attr"] == ep.ep]
                inside = mine[(mine["t"] >= ep.start) & (mine["t"] <= ep.end)]
                if len(inside):
                    strict += 1
                elif len(mine):
                    early_only += 1
            rows[kind] = {"n": int(len(e)), "alarmed_strict_window": strict, "alarmed_only_in_early_window": early_only}
        res["methods"][method] = rows
    gaps = episode_gaps(eps)
    res["episode_gaps_min"] = gaps
    res["first_alarm_before_label_start"] = first_alarm_early(OUT / "episodes_hall_full.csv")
    pre_min = json.loads((OUT / "frozen_config.json").read_text(encoding="utf-8"))["frozen_config"]["pre_start_s"] / 60
    res["note"] = (f"Strict window = trigger time between the labelled start and end. Early window = up to {pre_min:g} minutes "
                   f"before the labelled start (pre-registered). Hall episodes on the same day are {gaps['min']:.1f} to "
                   f"{gaps['max']:.1f} minutes apart (median {gaps['median']:.1f}, {gaps['n']} gaps), so early-window triggers "
                   "may belong to the previous activity.")
    (OUT / "sensitivity_hall.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("methods", "episode_gaps_min", "first_alarm_before_label_start")}, indent=1))
    exact_intervals()
    episode_counts()


def episode_gaps(eps: pd.DataFrame) -> dict:
    """Minutes between the end of one labelled Hall episode and the start of the next on the same day."""
    gaps = []
    for _, g in eps.sort_values("start").groupby(eps["start"].dt.date):
        gaps += [(s - e).total_seconds() / 60 for s, e in zip(g["start"].iloc[1:], g["end"].iloc[:-1])]
    ser = pd.Series(gaps, dtype=float)
    return {"n": int(len(ser)), "min": float(ser.min()), "median": float(ser.median()), "max": float(ser.max())}


def first_alarm_early(path: Path) -> dict:
    """Per method: episodes alarmed, and how many of their first alarms fall before the labelled start (inside the
    pre-registered early window), overall and for fire episodes."""
    e = pd.read_csv(path)
    out = {"source_file": str(path.relative_to(ROOT)).replace("\\", "/")}
    for m in ("stage1_pm", "co_delta", "ml_alone", "two_stage"):
        c = e[f"{m}_first_min"]
        fire = e["kind"] == "fire"
        out[m] = {"alarmed": int(c.notna().sum()), "first_alarm_early": int((c < 0).sum()),
                  "fire_alarmed": int((c.notna() & fire).sum()), "fire_first_alarm_early": int(((c < 0) & fire).sum())}
    return out


def episode_counts() -> None:
    """Episode counts by kind and scenario from the saved episode lists (replaces the by_scenario field of
    sensor_eval.json, whose keys were mangled by a pandas rename bug fixed after the run; metrics are unaffected)."""
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "sites": {}}
    for site in ("en54", "hall"):
        e = pd.read_csv(OUT / f"episodes_{site}.csv")
        out["sites"][site] = {"n": int(len(e)), "by_kind": {k: int(v) for k, v in e["kind"].value_counts().items()},
                              "by_scenario": [{"kind": k, "scenario": s, "n": int(v)} for (k, s), v in e.groupby(["kind", "scenario"]).size().items()]}
    (OUT / "episode_counts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("episode counts", {k: v["by_kind"] for k, v in out["sites"].items()})
    # Repair only the descriptive by_scenario field of sensor_eval.json (metrics untouched, nothing re-scored) and say so.
    r = json.loads((OUT / "sensor_eval.json").read_text(encoding="utf-8"))
    for site, blk in out["sites"].items():
        if site in r.get("episodes", {}):
            if r["episodes"][site]["n"] != blk["n"] or r["episodes"][site]["by_kind"] != blk["by_kind"]:
                raise SystemExit(f"episode list for {site} no longer matches sensor_eval.json; rerun the evaluation")
            r["episodes"][site]["by_scenario"] = {f"{x['kind']}:{x['scenario']}": x["n"] for x in blk["by_scenario"]}
    r["episodes_by_scenario_note"] = ("by_scenario rebuilt by scripts/fire_summarize.py from eval/fire/episodes_<site>.csv "
                                             "after a key-formatting bug in the original write; counts and metrics unchanged")
    (OUT / "sensor_eval.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial interval for k of n (the bootstrap interval collapses to a point when k == n or k == 0)."""
    from scipy.stats import beta

    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def exact_intervals() -> None:
    """Clopper-Pearson 95 % intervals for every episode count in sensor_eval.json (derived, no re-scoring)."""
    r = json.loads((OUT / "sensor_eval.json").read_text(encoding="utf-8"))
    out = {"method": "Clopper-Pearson exact binomial 95 % interval from the saved counts", "generated_at": datetime.now(timezone.utc).isoformat(),
           "source_file": "eval/fire/sensor_eval.json", "blocks": {}}
    blocks = {f"cv_en54/{v}": r["cv_en54"][v]["metrics"] for v in r.get("cv_en54", {})}
    blocks.update({f"hall/{v}": r["hall"][v]["metrics"] for v in ("full", "deltas") if v in r.get("hall", {})})
    for name, mets in blocks.items():
        out["blocks"][name] = {m: {kind: {"k": mets[m][kind]["alarmed"], "n": mets[m][kind]["n"],
                                          "ci95": clopper_pearson(mets[m][kind]["alarmed"], mets[m][kind]["n"])}
                                   for kind in ("fire", "nuisance", "other") if mets[m][kind]["n"]}
                               for m in ("stage1_pm", "co_delta", "ml_alone", "two_stage")}
    (OUT / "exact_intervals.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("exact intervals written for", list(out["blocks"]))


if __name__ == "__main__":
    main()
