"""M9 canary CLI: grade the 14 frozen canary inputs and compare with the reference for this fingerprint.

  python -m cascade.canary --confirm                    # one canary run, exit 1 on alarm
  python -m cascade.canary --build-reference --confirm  # three repeats -> runs/_canary/<model_id>/reference.json

The folder is keyed by the fingerprint's `model_id` (drift.identity_of), which the canary shares with every UI or
CLI run of the same models, prompts, rubrics and grader backend regardless of routing config.

Without --confirm the command prints the cost estimate and exits 2 without calling any model. Runs weekly, on a
fingerprint diff, a served-model mismatch or an Ollama digest change; never daily and never before every batch.
Dev images and two PIL frames only; eval_v1 is never touched (drift.canary_manifest).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import drift
from .exemplars import exemplar_provider
from .ingest import read_manifest, write_manifest
from .pipeline import RunConfig, run_cascade

ROOT = drift.ROOT
ALL_CLASSES = ("bridge_element", "steel_coating", "pv_module", "building_disaster")
# Measured anchor: runs/ui_0925_0856/summary.json, 14 claude-opus-5 grade calls, USD 0.79449 -> $0.0567 per call.
MEASURED_ANCHOR = {"run": "ui_0925_0856", "usd_per_grade_call": 0.0567, "gate_usd_per_call_claude": 0.0022}


def cost_estimate(n_images: int, repeats: int, gate: str, grader: str) -> dict:
    """Extrapolated from the measured anchor above, re-read from the run summary when it is present."""
    anchor = dict(MEASURED_ANCHOR)
    summ = ROOT / "runs" / anchor["run"] / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text(encoding="utf-8")).get("cost_by_stage", {})
        if s.get("grade", {}).get("calls"):
            anchor["usd_per_grade_call"] = round(s["grade"]["usd"] / s["grade"]["calls"], 4)
        if s.get("gate", {}).get("calls"):
            anchor["gate_usd_per_call_claude"] = round(s["gate"]["usd"] / s["gate"]["calls"], 4)
    grade_calls = n_images * repeats if grader != "none" else 0
    gate_calls = n_images * repeats if gate == "claude" else 0
    usd = grade_calls * (anchor["usd_per_grade_call"] if grader == "claude" else 0.0) + gate_calls * anchor["gate_usd_per_call_claude"]
    return {"grade_calls": grade_calls, "gate_calls": gate_calls, "usd_estimate": round(usd, 2), "anchor": anchor,
            "note": "extrapolated from one measured run at tiles=False; re-measure on the first real canary"}


def canary_config(gate: str, grader: str) -> RunConfig:
    return RunConfig(gate=gate, grader=grader, tiles=False, force_route_classes=ALL_CLASSES)


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="python -m cascade.canary", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-reference", action="store_true", help="three repeats to build reference.json for the current fingerprint")
    ap.add_argument("--confirm", action="store_true", help="actually call the models; without it only the estimate is printed")
    ap.add_argument("--gate", choices=["local", "claude", "none"], default="local")
    ap.add_argument("--grader", choices=["claude", "local"], default="claude")
    ap.add_argument("--exemplars", default=str(ROOT / "data/dev/manifest.jsonl"))
    args = ap.parse_args(argv)
    if not drift.CANARY_MANIFEST.exists():
        drift.canary_manifest(ROOT / "data/dev/manifest.jsonl")
    records = read_manifest(drift.CANARY_MANIFEST)
    cfg = canary_config(args.gate, args.grader)
    provider = exemplar_provider(Path(args.exemplars)) if Path(args.exemplars).exists() else None
    fp = drift.fingerprint(cfg, exemplar_ids=getattr(provider, "ids", None))
    repeats = int(drift.T("canary.n_repeats")) if args.build_reference else 1
    est = cost_estimate(len(records), repeats, args.gate, args.grader)
    key = drift.identity_of(fp)  # model identity, shared with every production run of the same models/prompts/rubrics
    print(f"fingerprint {fp['id']} (model identity {key}): {est['grade_calls']} grade calls x ${est['anchor']['usd_per_grade_call']} = about ${est['usd_estimate']} on {fp['components']['model.grader']} ({est['note']})")
    if not args.confirm:
        print("refusing to run without --confirm")
        return 2
    base = drift.CANARY_DIR / key
    if args.build_reference:
        dirs = []
        for i in range(1, repeats + 1):
            out = base / f"ref_{i}"
            write_manifest(records, out / "manifest.jsonl")
            run_cascade(records, out, cfg, exemplars=provider)
            dirs.append(out)
        ref = drift.canary_reference(dirs, base / "reference.json")
        print(json.dumps({k: v for k, v in ref.items() if k != "images"}, indent=1))
        return 0
    ref_path = base / "reference.json"
    if not ref_path.exists():
        print(f"no reference for model identity {key}: run --build-reference first")
        return 2
    out = base / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    write_manifest(records, out / "manifest.jsonl")
    run_cascade(records, out, cfg, exemplars=provider)
    verdict = drift.canary_compare(out, json.loads(ref_path.read_text(encoding="utf-8")))
    doc = drift.health(out, write=False)
    doc["canary"] = verdict
    (out / "health.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    return 1 if verdict["status"] == "alarm" else 0


if __name__ == "__main__":
    raise SystemExit(main())
