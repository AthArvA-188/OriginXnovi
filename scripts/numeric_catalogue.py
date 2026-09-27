"""Build the model catalogue and source list the Numeric AI page shows, from the research files (runtime env).

Inputs (copied into the repo so this reruns offline):
  eval/numeric/research/numeric_hf.json       research note + verifier corrections (2026-09-26)
  eval/numeric/research/hf_api/api_*.json     Hugging Face model API responses (sha, licence tag, params, file sizes)
Outputs:
  eval/numeric/model_catalogue.json   one row per candidate model/tool: HF id, revision, licence, params, weights MB,
                                       fit, status and reason (pilot numbers are stripped; only our own eval JSON
                                       carries measured numbers)
  eval/numeric/sources.json           numbered sources with URLs and access dates (moved URLs updated)

Run: C:\\Users\\HP\\miniconda3\\envs\\origin_hack\\python.exe scripts/numeric_catalogue.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade.numeric import datasets as D  # noqa: E402
from cascade.numeric.base import EVAL_DIR, utc_now  # noqa: E402

RESEARCH = EVAL_DIR / "research" / "numeric_hf.json"
API_DIR = EVAL_DIR / "research" / "hf_api"
MEASURED = {"amazon/chronos-2": "chronos_2", "amazon/chronos-bolt-small": "chronos_bolt_small",
            "sklearn.ensemble.HistGradientBoostingRegressor": "hgb", "sklearn.ensemble.IsolationForest": "isolation_forest"}
# What was actually measured for each measured row; the "fit" text lists POSSIBLE uses, most of them not measured.
MEASURED_SCOPE = {
    "amazon/chronos-2": "BDG2 electricity, day-ahead, univariate only (20 office meters)",
    "amazon/chronos-bolt-small": "BDG2 electricity, day-ahead, univariate only (20 office meters)",
    "sklearn.ensemble.HistGradientBoostingRegressor": "BDG2 electricity, day-ahead only (20 office meters)",
    "sklearn.ensemble.IsolationForest": "LEAD1.0 anomaly labels only (60 held-out buildings)"}
ROLE = {"amazon/chronos-2": "use - primary (precomputed offline)",
        "amazon/chronos-bolt-small": "use - second opinion (precomputed offline)",
        "sklearn.ensemble.HistGradientBoostingRegressor": "use - live fallback in the app (CPU, no new dependency)",
        "sklearn.ensemble.IsolationForest": "use - unsupervised anomaly scorer"}
REASON = {
    "amazon/chronos-2": ("Apache-2.0, not gated, zero-shot (no training on our data), gives 10/50/90 quantiles and "
                         "accepts covariates. Runs in its own env because its transformers needs huggingface-hub<2."),
    "amazon/chronos-bolt-small": ("Apache-2.0, same package, smaller and faster than Chronos-2: a second opinion to "
                                  "check that two pretrained models agree."),
    "sklearn.ensemble.HistGradientBoostingRegressor": ("Already installed, CPU only, loads a stored per-meter model in "
                                                       "the app and gives a split-conformal 80% band."),
    "sklearn.ensemble.IsolationForest": "Unsupervised scorer, compared against the robust-z baseline on LEAD labels.",
}
MOVED = {"https://github.com/NREL/BuildingsBench": "https://github.com/NatLabRockies/BuildingsBench",
         "https://github.com/microsoft/LightGBM/blob/master/LICENSE":
             "https://github.com/lightgbm-org/LightGBM/blob/master/LICENSE"}
NUMERIC_CLAIM = re.compile(r"\d+(\.\d+)?\s*%|WAPE|coverage|PR-AUC|pilot|\d+(\.\d+)?\s*s\b", re.I)


def hf_id(text: str) -> str:
    rid = text.split(" ")[0].split("(")[0].strip().rstrip(";")
    return MOVED.get(rid, rid)


def status_and_reason(verdict: str):
    m = re.match(r"\s*(use as secondary|use as fallback|use|maybe|avoid)\b(?:\s+here|\s+by default)?[\s.:,-]*(.*)",
                 verdict, re.I | re.S)
    status, rest = (m.group(1).lower(), m.group(2)) if m else ("", verdict)
    keep = [s.strip() for s in re.split(r"(?<=[.;])\s+", rest) if s.strip() and not NUMERIC_CLAIM.search(s)]
    return status, " ".join(keep)


def main() -> None:
    research = json.loads(RESEARCH.read_text(encoding="utf-8"))
    api = {}
    for f in glob.glob(str(API_DIR / "api_*.json")):
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        api[a["id"]] = a
    rows = []
    for c in research["research"]["candidates"]:
        if c["kind"] == "dataset":
            continue
        rid = hf_id(c["id_or_url"])
        a = api.get(rid)
        status, reason = status_and_reason(c["verdict"])
        row = {"name": c["name"], "id": rid, "on_hugging_face": a is not None, "fit": c["fit"],
               "runs_on": c["runs_on"] if not NUMERIC_CLAIM.search(c["runs_on"]) else re.sub(
                   r"measured:.*", "see measured results", c["runs_on"], flags=re.I),
               "licence": c["license"], "status": ROLE.get(rid, status), "reason": REASON.get(rid, reason),
               "measured_here": rid in MEASURED, "eval_key": MEASURED.get(rid),
               "measured_scope": MEASURED_SCOPE.get(rid, "not measured")}
        if a is not None:
            sib = {s["rfilename"]: s.get("size") for s in a.get("siblings", []) if s.get("size")}
            weights = [v for k, v in sib.items() if k.endswith((".safetensors", ".ckpt", ".bin", ".pth"))
                       and "training_args" not in k]
            row.update({"revision": a.get("sha"), "hf_licence_tag": (a.get("cardData") or {}).get("license"),
                        "gated": a.get("gated"), "params_millions": round(a["safetensors"]["total"] / 1e6, 2)
                        if a.get("safetensors") else None,
                        "largest_weight_file_MB": round(max(weights) / 1e6, 1) if weights else None,
                        "hf_datasets_listed": (a.get("cardData") or {}).get("datasets"),
                        "api_url": f"https://huggingface.co/api/models/{rid}", "last_modified": a.get("lastModified")})
        rows.append(row)
    datasets = [
        {"name": "Building Data Genome 2 (BDG2)", "label": "REAL", "url": D.BDG2_REPO, "licence": D.BDG2_LICENCE,
         "use": "forecast backtest (office electricity meters)", "citation": D.BDG2_CITATION},
        {"name": "LEAD1.0-small", "label": "REAL (human anomaly labels)", "url": D.LEAD_REPO, "licence": D.LEAD_LICENCE,
         "use": "anomaly evaluation, aggregate metrics only", "citation": D.LEAD_CITATION},
        {"name": "Salesforce/GiftEvalPretrain (folder list only)", "label": "REAL", "url": D.GIFT_EVAL_URL,
         "licence": "Apache-2.0 with a 'research purposes only' note in its README (per research verifier)",
         "use": "only to decide which BDG2 sites to exclude from the test pool", "citation": ""},
    ]
    cat = {"created_utc": utc_now(), "source": "eval/numeric/research/numeric_hf.json + hf_api/*.json (2026-09-26)",
           "note": "params and weight sizes are read from the Hugging Face API (decimal MB). Measured numbers for the "
                   "models marked measured_here are in backtest_bdg2.json and anomaly_lead.json only, for the task "
                   "named in measured_scope. The 'fit' text lists possible uses; they are not measured unless "
                   "measured_scope names them.",
           "models": rows, "datasets": datasets,
           "verifier_corrections": research["verify"]["corrections"]}
    (EVAL_DIR / "model_catalogue.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")

    srcs = []
    for s in research["research"]["sources"]:
        url = MOVED.get(s["url"], s["url"])
        note = s["supports"]
        if s["n"] == 36:
            note = ("Header reads 'Attribution-ShareAlike 4.0 Unported' but the body is the CC BY-SA 3.0 Unported "
                    "legal code; GitHub reports NOASSERTION")
        srcs.append({"n": s["n"], "title": s["title"], "url": url, "accessed": s["accessed"], "supports": note,
                     "moved_from": s["url"] if url != s["url"] else None})
    srcs.append({"n": 51, "title": "Salesforce/GiftEvalPretrain dataset (folder listing)", "url": D.GIFT_EVAL_URL,
                 "accessed": "2026-09-26", "supports": "contains bdg-2_bear, bdg-2_fox, bdg-2_panther, bdg-2_rat, "
                 "bull, cockatoo and hog folders (BDG2 sites excluded from our test pool)", "moved_from": None})
    (EVAL_DIR / "sources.json").write_text(json.dumps({"created_utc": utc_now(), "sources": srcs}, indent=1),
                                           encoding="utf-8")
    for r in rows:
        print(f"{r['id'][:48]:48s} {str(r.get('params_millions')):>8s} {str(r.get('largest_weight_file_MB')):>8s} "
              f"{r['status'][:40]}")


if __name__ == "__main__":
    main()
