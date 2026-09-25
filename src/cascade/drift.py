"""Drift prevention and validation loops: everything here is deterministic and measured from run files.

Mechanisms (docs: scratchpad driftSpec.md, synthesis of the MLOps and SPC designs):
  M1  run fingerprint and model-identity pin            (stage config)
  M2  contract audit of every finding against its rubric (stage grade, zero-baseline c-chart)
  M3  U-rate p-chart with cause split                    (stage grade)
  M4  output shift versus baseline on shared images      (stage grade)
  M5  gate health per dataset                            (stage gate)
  M6  reviewer loop: direction, critical miss, blind QC  (stage review)
  M7  ops: tokens, latency, pricing                      (stage ops)
  M8  input drift on the imagery                         (stage input)
  M9  frozen canary comparison (pure functions; the CLI lives in cascade.canary)
  M10 promotion gate, one-look eval ledger, rollback
  M11 alert budget, n floors, sample-size helper

Every threshold lives in `drift_thresholds.json` beside this file with its basis, and that file is hashed
into the fingerprint so old runs are never silently re-scored. Nothing here rewrites a finding, a level
or a config (D-012); alarms are written to `runs/<run>/health.json` for the report and the UI.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from PIL import Image

from .costlog import PRICES_PER_MTOK
from .crop import MAX_SIDE, tile_boxes, to_base64_jpeg, upscale_small
from .evalmetrics import ROOT, eval_matrix
from .exemplars import select_exemplars
from .gate import GATE_PROMPT
from .grade import EFFORT, GRADE_MAX_TOKENS, OLLAMA_OPTIONS, RUBRIC_DIR, RUBRIC_FOR_CLASS, Exemplar, build_system, build_user_content, load_rubric
from .ingest import read_manifest, sha256_of, write_manifest
from .pipeline import LEVELS, RunConfig, load_run
from .schema import LEVEL_ORDER, Finding, GateOutput, GraderOutput, ImageRecord

THRESHOLDS_PATH = Path(__file__).parent / "drift_thresholds.json"
BASELINE_DIR = ROOT / "runs" / "_baseline"  # underscore: hidden by app.list_runs
CANARY_DIR = ROOT / "runs" / "_canary"
CANARY_MANIFEST = ROOT / "data" / "canary" / "manifest.jsonl"
LEDGER = ROOT / "eval" / "reports" / "ledger.jsonl"
EVAL_MANIFEST = ROOT / "data" / "eval_v1" / "manifest.jsonl"
STAGES = ("config", "gate", "grade", "review", "ops", "input")
HARD_RULES = ("H1_value_not_allowed", "H2_criteria_not_verbatim", "H3_criteria_empty", "H4_level_row_mismatch",
              "H5_action_row_mismatch", "H6_impossible_measurement", "H7_defaulted_s0")
SOFT_RULES = ("W1_flag_missing", "W2_pv_confidence", "W3_sla_inconsistent", "W4_mixed_u")
U_CAUSES = ("gate_unusable", "refusal", "parse_error", "model_u")
# M11 alert budget: lower rank surfaces first; at most one alarm per stage is surfaced, the rest print as watch.
RANK = {"critical_miss": 1, "mixed_models": 2, "served_mismatch": 2, "digest_mismatch": 2, "fingerprint_diff": 2, "fingerprint_changed_mid_run": 2,
        "contract_hard": 3, "refusal": 4, "parse_error": 4, "canary": 5, "abstention_collapse": 6, "moved_2plus": 6,
        "u_rate": 7, "u_rate_low": 7, "dead_gate": 8, "template_collapse": 8, "gate_fallback": 8, "gate_miss": 8, "under_grading": 9, "agreement": 10,
        "blind_floor": 10, "unpriced": 11, "token_mismatch": 11, "verbosity": 11, "latency": 11,
        "resolution": 12, "frame_gap": 12, "duplicates": 12, "input_not_comparable": 12, "digest_unavailable": 12, "env_differs_from_active": 12}
# Rules that grey the queue / bridge exports and block promotion: never demoted by the alert budget.
BLOCKING_RULES = ("contract_hard", "abstention_collapse", "moved_2plus", "critical_miss")
# Fingerprint components kept for diffs but left out of the hashed id: a commit changes no grade by itself.
UNHASHED_COMPONENTS = ("git.head",)
# RunConfig fields that change what the grader sees or which grader runs; the rest (routing, limits) are run config.
IDENTITY_CONFIG_KEYS = ("gate", "grader", "tiles", "rubric_file")

_THRESHOLDS = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))


def T(key: str):
    """Threshold value by dotted key, e.g. T("gate.dead_min_damaged")."""
    group, name = key.split(".")
    return _THRESHOLDS[group][name]["value"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _median(xs: List[float]) -> Optional[float]:
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _pct(xs: List[float], q: float) -> Optional[float]:
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _signal(rule: str, stage: str, severity: str, value, threshold, n: int, action: str, items: Optional[list] = None, n_required: Optional[int] = None) -> dict:
    return {"rank": RANK.get(rule, 99), "stage": stage, "rule": rule, "severity": severity, "value": value, "threshold": threshold,
            "n": n, "n_required": n_required, "action": action, "items": list(items or [])[:50]}


# ----------------------------------------------------------------------------------------------- M1 fingerprint

def _git_head() -> Optional[str]:
    head = ROOT / ".git" / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = ROOT / ".git" / text.split(" ", 1)[1].strip()
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return text


def _ollama_digests(tags: Iterable[str], env: Mapping[str, str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {t: None for t in tags}
    try:
        import requests

        url = env.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        models = requests.get(f"{url}/api/tags", timeout=2).json().get("models", [])
        by_name = {m.get("name"): m.get("digest") for m in models}
        for t in tags:
            out[t] = by_name.get(t)
    except Exception:
        pass
    return out


def fingerprint(cfg: RunConfig, *, exemplar_ids: Optional[Dict[str, List[str]]] = None, env: Mapping[str, str] = os.environ, ollama: bool = True) -> dict:
    """M1: `{"id", "model_id", "components", "built_at"}` over every input that can change a grade.

    Components are sha256 hex of file bytes or prompt text, or plain values where a diff should be readable
    (model ids, effort, config, versions, git head). `id` hashes every component except UNHASHED_COMPONENTS
    and Ollama digests that could not be read (a 2 s /api/tags timeout must not change the id). `model_id`
    hashes the model-identity subset (everything but the run config, of which only IDENTITY_CONFIG_KEYS count):
    baseline cards and the canary reference are keyed by it, so a routing or limit change stays comparable.
    """
    comp: Dict[str, object] = {}
    for p in sorted(RUBRIC_DIR.glob("*.json")):
        comp[f"rubric.{p.name}"] = _sha(p.read_bytes())
    for p in sorted((ROOT / "eval" / "rubrics").glob("*.json")):
        comp[f"truth_map.{p.name}"] = _sha(p.read_bytes())
    for ac in RUBRIC_FOR_CLASS:
        rubric = load_rubric(ac, cfg.rubric_file if ac == "bridge_element" else None)
        comp[f"prompt.grader.{ac}"] = _sha(build_system(ac, rubric).encode("utf-8"))
    comp["prompt.gate"] = _sha(GATE_PROMPT.encode("utf-8"))
    # the rendered user turn (exemplar caption format, target line, metadata phrasing, closing instruction), text parts only
    tiny = Image.new("RGB", (8, 8))
    user_meta = {"asset_class": "x", "gsd_mm_per_px": None, "irradiance_wm2": None, "captured_on": None, "image_size": "8x8"}
    user_text = "\n".join(c["text"] for c in build_user_content(tiny, user_meta, [Exemplar(tiny, "G", "n", "id")]) if c["type"] == "text")
    comp["prompt.grader_user"] = _sha(user_text.encode("utf-8"))
    comp["grade.max_tokens"] = GRADE_MAX_TOKENS
    comp["grade.ollama_options"] = dict(OLLAMA_OPTIONS)
    comp["schema.grader"] = _sha(json.dumps(GraderOutput.model_json_schema(), sort_keys=True).encode("utf-8"))
    comp["schema.gate"] = _sha(json.dumps(GateOutput.model_json_schema(), sort_keys=True).encode("utf-8"))
    comp["model.grader"] = env.get("GRADER_MODEL", "claude-opus-5")
    comp["model.gate_cloud"] = env.get("GATE_CLOUD_MODEL", "claude-haiku-4-5")
    comp["model.gate_local"] = env.get("GATE_LOCAL_MODEL", "qwen3-vl:4b-instruct")
    comp["model.grader_local"] = env.get("GRADER_LOCAL_MODEL", "qwen3-vl:8b-instruct")
    comp["model.effort"] = EFFORT
    local_tags = [comp["model.gate_local"], comp["model.grader_local"]]
    digests = _ollama_digests(local_tags, env) if ollama and (cfg.gate == "local" or cfg.grader == "local") else {t: None for t in local_tags}
    for tag, digest in digests.items():
        comp[f"ollama.{tag}"] = digest
    # JSON round-trip so the in-memory value equals what a re-read fingerprint.json holds (tuples become lists)
    comp["config"] = json.loads(json.dumps({k: v for k, v in asdict(cfg).items() if k != "limit"}))
    for ac in RUBRIC_FOR_CLASS:
        comp[f"exemplar_ids.{ac}"] = sorted((exemplar_ids or {}).get(ac, []))
    comp["crop.max_side"] = MAX_SIDE
    comp["crop.jpeg_quality"] = inspect.signature(to_base64_jpeg).parameters["quality"].default
    comp["crop.tile_overlap"] = inspect.signature(tile_boxes).parameters["overlap"].default
    comp["crop.upscale_min_side"] = inspect.signature(upscale_small).parameters["min_side"].default
    comp["prices"] = _sha(json.dumps(PRICES_PER_MTOK, sort_keys=True).encode("utf-8"))
    comp["thresholds"] = _sha(THRESHOLDS_PATH.read_bytes())
    frozen = ROOT / "data" / "eval_v1" / "FROZEN.sha256"
    comp["eval.frozen_sha"] = frozen.read_text(encoding="utf-8").split()[0] if frozen.exists() else None
    dev = ROOT / "data" / "dev" / "manifest.jsonl"
    comp["dev.manifest_sha"] = _sha(dev.read_bytes()) if dev.exists() else None
    versions = {"python": sys.version.split()[0]}
    for mod, attr in (("anthropic", "__version__"), ("pydantic", "VERSION"), ("PIL", "__version__")):
        try:
            versions[mod] = str(getattr(__import__(mod), attr))
        except Exception:
            versions[mod] = None
    comp["versions"] = versions
    comp["git.head"] = _git_head()
    hashed = {k: v for k, v in comp.items() if k not in UNHASHED_COMPONENTS and not (k.startswith("ollama.") and v is None)}
    fp_id = _sha(json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode("utf-8"))[:12]
    identity = {k: v for k, v in hashed.items() if k != "config"}
    identity["config"] = {k: v for k, v in comp["config"].items() if k in IDENTITY_CONFIG_KEYS}
    model_id = _sha(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))[:12]
    return {"id": fp_id, "model_id": model_id, "components": comp, "built_at": _now()}


def identity_of(fp: Optional[dict]) -> Optional[str]:
    """The key baseline cards and the canary reference are filed under: `model_id`, else `id` for fingerprints
    written before the split, else None."""
    if not fp:
        return None
    return fp.get("model_id") or fp.get("id")


def fingerprint_diff(a: dict, b: dict) -> List[Tuple[str, object, object]]:
    """Components that differ between two fingerprints, sorted by name: (name, a_value, b_value)."""
    ca, cb = a.get("components", {}), b.get("components", {})
    return [(k, ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)]


def served_matches(requested: str, served: Optional[str]) -> bool:
    """A served id matches when it equals the requested id or is a dated / suffixed snapshot of it
    (`claude-haiku-4-5` -> `claude-haiku-4-5-20251001`); the API resolves aliases this way by design."""
    return not served or served == requested or served.startswith(requested + "-")


def check_models(run: dict, fp: Optional[dict], pinned_served: Optional[Mapping[str, str]] = None) -> dict:
    """M1 per run: model ids seen per stage in calls.jsonl, mixing, served-model mismatches, fingerprint match.

    `served_mismatch` holds rows whose served id is not the requested id or a snapshot of it; `served_multiple`
    lists requested ids that resolved to more than one served id inside the run; `served_changed` lists requested
    ids whose served id differs from `pinned_served` (the baseline card's first served id per model)."""
    per_stage: Dict[str, set] = defaultdict(set)
    served_by_model: Dict[str, set] = defaultdict(set)
    for row in run.get("calls", []):
        per_stage[row["stage"]].add(row["model"])
        if row.get("served_model"):
            served_by_model[row["model"]].add(row["served_model"])
    for f in run.get("findings", []):
        if f.model and u_cause(f) != "gate_unusable":  # a gate-unusable U carries the gate's model id, not a grader's
            per_stage["grade"].add(f.model)
    served = [row for row in run.get("calls", []) if not served_matches(row["model"], row.get("served_model"))]
    multiple = {m: sorted(v) for m, v in served_by_model.items() if len(v) > 1}
    changed = {m: (pinned_served[m], sorted(v)) for m, v in served_by_model.items() if pinned_served and pinned_served.get(m) and v != {pinned_served[m]}}
    matches = None
    if fp:
        c = fp.get("components", {})
        expected = {c.get("model.grader"), c.get("model.gate_cloud"), c.get("model.gate_local"), c.get("model.grader_local"), "none"}
        seen = set().union(*per_stage.values()) if per_stage else set()
        matches = seen <= expected
    return {"models_per_stage": {s: sorted(v) for s, v in per_stage.items()}, "mixed": any(len(v) > 1 for v in per_stage.values()),
            "served_mismatch": served, "served_by_model": {m: sorted(v) for m, v in served_by_model.items()}, "served_multiple": multiple,
            "served_changed": changed, "matches_fingerprint": matches}


def first_served(calls: List[dict]) -> Dict[str, str]:
    """requested model id -> first served id logged for it (the value a baseline card pins)."""
    out: Dict[str, str] = {}
    for row in calls:
        if row.get("served_model") and row["model"] not in out:
            out[row["model"]] = row["served_model"]
    return out


# ----------------------------------------------------------------------------------------------- M2 contract audit

_QUOTES = "'\"‘’“”"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", "".join(ch for ch in text if ch not in _QUOTES)).strip().lower()


def u_cause(f: Finding) -> str:
    """Why a finding is U, from the justification prefix the pipeline writes; anything else is the model's own U."""
    j = f.justification or ""
    if j.startswith("gate marked image unusable"):
        return "gate_unusable"
    if j.startswith("grader returned no contract (refusal"):
        return "refusal"
    if j.startswith("grader returned no contract (parse_error"):
        return "parse_error"
    return "model_u"


def _audit_one(f: Finding, rubric: dict) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[dict]]:
    """Hard and soft rule hits for one finding; returns (hard, soft, matched_rows)."""
    hard: List[Tuple[str, str]] = []
    soft: List[Tuple[str, str]] = []
    rows = rubric.get("rows", [])
    allowed = rubric.get("allowed_values", [])
    is_u = f.unified.level == "U"
    value = f.native_scale.value
    quoted = [_norm(c) for c in f.native_scale.criteria_matched if c and c.strip()]
    for q in quoted:
        if not any(q in _norm(r["criterion"]) for r in rows):
            hard.append(("H2_criteria_not_verbatim", f"'{q[:60]}' is not a substring of any rubric criterion"))
    matched = [r for r in rows if r["value"] == value and any(q in _norm(r["criterion"]) for q in quoted)] or [r for r in rows if r["value"] == value]
    if not is_u:
        if value not in allowed:
            hard.append(("H1_value_not_allowed", f"native value '{value}' not in allowed_values"))
        if not quoted:
            hard.append(("H3_criteria_empty", "non-U finding with empty criteria_matched"))
        levels = {r["unified"] for r in matched}
        s4_ok = f.unified.level == "S4" and "S4" in rubric.get("unified_note", "") and allowed and value == allowed[-1]
        if matched and f.unified.level not in levels and not s4_ok:
            hard.append(("H4_level_row_mismatch", f"level {f.unified.level} not in rubric rows for '{value}' ({sorted(levels)})"))
        actions = {r["action"] for r in matched}
        if matched and f.action.code not in actions and not (f.unified.level == "S4" and f.action.code == "escalate"):
            hard.append(("H5_action_row_mismatch", f"action '{f.action.code}' not in rubric rows for '{value}' ({sorted(actions)})"))
        for r in matched:
            if r.get("flag") and r["flag"] not in f.unified.flags:
                soft.append(("W1_flag_missing", f"rubric row flag '{r['flag']}' missing from unified.flags"))
                break
        if f.asset_class == "pv_module" and f.evidence.irradiance_wm2 is None and f.measurements.confidence > 0.6:
            soft.append(("W2_pv_confidence", f"irradiance unknown but confidence {f.measurements.confidence}"))
    m = f.measurements
    if m.crack_width_mm is not None and f.evidence.gsd_mm_per_px is None:
        hard.append(("H6_impossible_measurement", "crack_width_mm without gsd_mm_per_px"))
    if m.delta_t_k is not None and f.evidence.irradiance_wm2 is None:
        hard.append(("H6_impossible_measurement", "delta_t_k without irradiance_wm2"))
    if m.section_loss_pct is not None and f.asset_class == "steel_coating":
        hard.append(("H6_impossible_measurement", "section_loss_pct on steel_coating (needs thickness data)"))
    if f.unified.level == "S0" and "not_measurable" in f.unified.flags and not quoted:
        hard.append(("H7_defaulted_s0", "S0 with not_measurable and no criterion quoted"))
    if is_u and value not in ("U", "Inaccessible"):
        soft.append(("W4_mixed_u", f"level U carries native value '{value}'"))
    return hard, soft, matched


def contract_audit(findings: List[Finding], rubric_for: Callable[[str], dict], calls: List[dict]) -> dict:
    """M2: hard rules H1..H7 (any one is a contract breach) and soft rules W1..W4 per finding; parse/refusal counts
    from grade calls; U causes. Counts are over `findings`; rates are per finding, never per image."""
    hard: Counter = Counter()
    soft: Counter = Counter()
    violations: List[dict] = []
    breached = 0
    sla_groups: Dict[tuple, set] = defaultdict(set)
    causes: Counter = Counter()
    for f in findings:
        h, s, _ = _audit_one(f, rubric_for(f.asset_class))
        if h:
            breached += 1
        for rule, detail in h:
            hard[rule] += 1
            violations.append({"finding_id": f.finding_id, "rule": rule, "detail": detail})
        for rule, detail in s:
            soft[rule] += 1
            violations.append({"finding_id": f.finding_id, "rule": rule, "detail": detail})
        sla_groups[(f.asset_class, f.native_scale.value, f.action.code)].add(f.action.sla_days is None)
        if f.unified.level == "U":
            causes[u_cause(f)] += 1
    w3 = [g for g, kinds in sla_groups.items() if len(kinds) > 1]
    soft["W3_sla_inconsistent"] = len(w3)
    for g in w3:
        violations.append({"finding_id": "*", "rule": "W3_sla_inconsistent", "detail": f"group {g} mixes null and non-null sla_days"})
    grade_calls = [c for c in calls if c.get("stage") == "grade"]
    n = len(findings)
    return {"n": n, "hard": {r: hard.get(r, 0) for r in HARD_RULES}, "soft": {r: soft.get(r, 0) for r in SOFT_RULES},
            "hard_total": sum(hard.values()), "findings_breached": breached, "hard_rate": (breached / n) if n else 0.0,
            "violations": violations, "grade_calls": len(grade_calls),
            "parse_fail": sum(1 for c in grade_calls if str(c.get("note", "")).startswith("parse_error")),
            "refusals": sum(1 for c in grade_calls if c.get("note") == "refusal"),
            "stop_categories": sorted({c["stop_category"] for c in grade_calls if c.get("stop_category")}),
            "u_by_cause": {c: causes.get(c, 0) for c in U_CAUSES}}


def contract_signals(audit: dict, baseline: Optional[dict] = None) -> List[dict]:
    """Turn an audit into M2 alarms/watches. Refusal and parse use the rule of three until the fingerprint has
    `refusal.clean_calls_for_rate` clean baseline calls, then the binomial tail against 3/n_base."""
    out: List[dict] = []
    n = audit["n"]
    hard_thr = max(T("contract.hard_alarm_min"), math.ceil(T("contract.hard_rate") * n)) if n else T("contract.hard_alarm_min")
    if audit["findings_breached"] >= hard_thr and n:
        items = [v["finding_id"] for v in audit["violations"] if v["rule"].startswith("H")]
        out.append(_signal("contract_hard", "grade", "alarm", audit["findings_breached"], hard_thr, n, "review flagged findings; queue and bridge exports greyed until acknowledged", items))
    elif audit["findings_breached"]:
        out.append(_signal("contract_hard", "grade", "watch", audit["findings_breached"], hard_thr, n, "one breach can be transient; look at the finding", [v["finding_id"] for v in audit["violations"] if v["rule"].startswith("H")]))
    n_base = (baseline or {}).get("clean_grade_calls") or 0
    for rule in ("refusal", "parse_error"):
        k = audit["refusals"] if rule == "refusal" else audit["parse_fail"]
        if not k:
            continue
        if n_base >= T("refusal.clean_calls_for_rate"):
            tail = binom_tail(k, audit["grade_calls"], 3 / n_base)
            sev = "alarm" if tail < T("refusal.binomial_tail") else "watch"
            out.append(_signal(rule, "grade", sev, k, f"P(X>={k}|n={audit['grade_calls']},p=3/{n_base})={tail:.3f}", audit["grade_calls"], "pause the Batch tab for this grader backend" if sev == "alarm" else "watch"))
        else:
            sev = "alarm" if k >= T("refusal.alarm_per_run") else "watch"
            out.append(_signal(rule, "grade", sev, k, T("refusal.alarm_per_run"), audit["grade_calls"], "pause the Batch tab for this grader backend" if sev == "alarm" else "single event: rule of three cannot call it"))
    return out


# ----------------------------------------------------------------------------------------------- M3 U-rate p-chart

def u_breakdown(findings: List[Finding], calls: List[dict]) -> dict:
    """U findings over all findings, split by cause."""
    n = len(findings)
    us = [f for f in findings if f.unified.level == "U"]
    causes = Counter(u_cause(f) for f in us)
    return {"n": n, "u": len(us), "rate": (len(us) / n) if n else None,
            "by_cause": {c: {"count": causes.get(c, 0), "rate": (causes.get(c, 0) / n) if n else None} for c in U_CAUSES}}


def p_limits(p0: float, n: int, sigma: float = 3.0) -> dict:
    """Shewhart p-chart limits p0 +/- sigma*sqrt(p0(1-p0)/n), clipped to [0, 1]."""
    if n <= 0:
        return {"ucl": None, "lcl": None, "sigma": None}
    s = math.sqrt(max(0.0, p0 * (1 - p0)) / n)
    return {"ucl": min(1.0, p0 + sigma * s), "lcl": max(0.0, p0 - sigma * s), "sigma": s}


def u_chart(findings: List[Finding], calls: List[dict], baseline: Optional[dict], not_comparable: Optional[str] = None) -> dict:
    """M3 chart data plus signals. Below `u.min_n` findings the chart prints n_required instead of limits.

    Above UCL is the u_rate alarm. Below LCL alone is not a signal (better capture metadata legitimately lowers
    U); it becomes the u_rate_low watch only when the S0 share also rose by `shift.collapse_points` over the
    card's level_shares, the abstention-collapse signature that needs no shared images."""
    b = u_breakdown(findings, calls)
    n = b["n"]
    chart = {**b, "p0": None, "ucl": None, "lcl": None, "status": "candidate: no baseline", "n_required": None, "not_comparable": not_comparable, "signals": []}
    if n < T("u.min_n"):
        chart["status"], chart["n_required"] = "insufficient_n", T("u.min_n")
        return chart
    if baseline is None or baseline.get("u_p0") is None:
        return chart
    if not_comparable:
        chart["status"] = f"not comparable: {not_comparable}"
        return chart
    lim = p_limits(baseline["u_p0"], n, T("u.sigma"))
    chart.update({"p0": baseline["u_p0"], "ucl": lim["ucl"], "lcl": lim["lcl"], "status": "in_control"})
    if b["rate"] > lim["ucl"]:
        chart["status"] = "alarm"
        worst = max(U_CAUSES, key=lambda c: b["by_cause"][c]["count"])
        action = {"model_u": "reviewers look at the U findings first", "refusal": "pause the Batch tab for this grader backend",
                  "parse_error": "pause the Batch tab for this grader backend", "gate_unusable": "capture quality, not the model"}[worst]
        chart["signals"].append(_signal("u_rate", "grade", "alarm", round(b["rate"], 4), round(lim["ucl"], 4), n, f"U above UCL, dominant cause {worst}: {action}"))
    elif lim["lcl"] > 0 and b["rate"] < lim["lcl"]:
        s0_share = sum(1 for f in findings if f.unified.level == "S0") / n
        base_s0 = (baseline.get("level_shares") or {}).get("S0")
        chart["below_lcl"] = True
        if base_s0 is not None and s0_share - base_s0 > T("shift.collapse_points"):
            chart["status"] = "watch"
            chart["signals"].append(_signal("u_rate_low", "grade", "watch", round(b["rate"], 4), round(lim["lcl"], 4), n,
                                            f"U below LCL and S0 share up {s0_share - base_s0:+.2f}: check H6 impossible-measurement counts; possible abstention collapse"))
    return chart


# ----------------------------------------------------------------------------------------------- M4 output shift

def worst_levels(findings: List[Finding]) -> Dict[str, str]:
    """image_id -> worst non-U level across its findings, else U (same rule as the UI)."""
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    out: Dict[str, str] = {}
    for f in findings:
        for iid in f.evidence.image_ids:
            cur = out.get(iid)
            if cur is None or (f.unified.level != "U" and (cur == "U" or order[f.unified.level] > order[cur])):
                out[iid] = f.unified.level
    return out


def _run_bundle(run_dir: Path) -> Tuple[Dict[str, ImageRecord], Dict[str, str], dict]:
    run_dir = Path(run_dir)
    recs = {r.image_id: r for r in read_manifest(run_dir / "manifest.jsonl")} if (run_dir / "manifest.jsonl").exists() else {}
    run = load_run(run_dir)
    return recs, worst_levels(run["findings"]), run["summary"]


def level_shift(levels_a: Dict[str, str], levels_b: Dict[str, str], classes: Dict[str, str]) -> dict:
    """Pure M4 comparison over images present in both maps; `classes` maps image_id -> asset_class."""
    per_class: Dict[str, dict] = {}
    by_ac: Dict[str, list] = defaultdict(list)
    for iid in sorted(set(levels_a) & set(levels_b)):
        by_ac[classes.get(iid, "unknown")].append((iid, levels_a[iid], levels_b[iid]))
    for ac, trip in by_ac.items():
        n = len(trip)
        exact = sum(1 for _, a, b in trip if a == b)
        within = sum(1 for _, a, b in trip if (a == b == "U") or (a != "U" and b != "U" and abs(LEVEL_ORDER[a] - LEVEL_ORDER[b]) <= 1))
        moved = [iid for iid, a, b in trip if a != "U" and b != "U" and abs(LEVEL_ORDER[a] - LEVEL_ORDER[b]) >= 2]
        ha, hb = Counter(a for _, a, _ in trip), Counter(b for _, _, b in trip)
        per_class[ac] = {"n": n, "exact": exact / n, "within_one": within / n, "moved_2plus": moved,
                         "u_rate_delta": (hb["U"] - ha["U"]) / n, "s0_share_delta": (hb["S0"] - ha["S0"]) / n,
                         "severe_mass_delta": (hb["S3"] + hb["S4"] - ha["S3"] - ha["S4"]) / n,
                         "level_hist_l1": sum(abs(ha[l] - hb[l]) for l in LEVELS) / n}
    collapse = any(c["u_rate_delta"] < -T("shift.collapse_points") and c["s0_share_delta"] > T("shift.collapse_points") for c in per_class.values())
    return {"per_class": per_class, "abstention_collapse": collapse}


def output_shift(run_a: Path, run_b: Path, min_shared: Optional[int] = None) -> dict:
    """M4: baseline run A versus candidate run B on images with equal sha256 in both manifests."""
    min_shared = T("shift.min_shared") if min_shared is None else min_shared
    recs_a, lv_a, sum_a = _run_bundle(run_a)
    recs_b, lv_b, sum_b = _run_bundle(run_b)
    shared = [i for i in recs_a if i in recs_b and recs_a[i].sha256 == recs_b[i].sha256 and i in lv_a and i in lv_b]
    tiles_differ = (sum_a.get("config", {}).get("tiles"), sum_b.get("config", {}).get("tiles")) not in ((True, True), (False, False), (None, None))
    out = {"status": "ok", "n_shared": len(shared), "tiles_differ": tiles_differ, "per_class": {}, "abstention_collapse": False, "signals": [], "n_required": None}
    if len(shared) < min_shared:
        out.update({"status": "not_comparable", "n_required": min_shared})
        return out
    res = level_shift({i: lv_a[i] for i in shared}, {i: lv_b[i] for i in shared}, {i: recs_b[i].asset_class for i in shared})
    out.update(res)
    n = len(shared)
    moved = [i for c in res["per_class"].values() for i in c["moved_2plus"]]
    if res["abstention_collapse"]:
        out["signals"].append(_signal("abstention_collapse", "grade", "alarm", True, f"U down > {T('shift.collapse_points')} and S0 up > {T('shift.collapse_points')}", n, "block promotion and disable exports"))
    if len(moved) >= T("shift.moved2_block"):
        out["signals"].append(_signal("moved_2plus", "grade", "alarm", len(moved), T("shift.moved2_block"), n, "review each moved image; promotion blocked", moved))
    elif moved:
        out["signals"].append(_signal("moved_2plus", "grade", "watch", len(moved), T("shift.moved2_block"), n, "review each moved image", moved))
    for ac, c in res["per_class"].items():
        if abs(c["u_rate_delta"]) > T("shift.u_delta_watch"):
            out["signals"].append(_signal("u_rate", "grade", "watch", round(c["u_rate_delta"], 3), T("shift.u_delta_watch"), c["n"], f"U rate moved on {ac}"))
    return out


# ----------------------------------------------------------------------------------------------- M5 gate health

def gate_health(gate_rows: List[dict], records: Dict[str, ImageRecord], calls: List[dict], gate_min_conf: float, baseline: Optional[dict] = None, allow_eval: bool = False) -> dict:
    """M5 per source_dataset (else asset_class, else client_id): routing done by the gate itself versus forced,
    fallback and declined rows, reason diversity, confidence bins, recall on labelled damaged images.

    Recall from labels is skipped (eval = note only) when any labelled record is an eval_v1 row, unless
    `allow_eval` (eval/run_eval.py's one ledgered look); eval_v1 is never scored from a health check."""
    groups: Dict[str, list] = defaultdict(list)
    for g in gate_rows:
        r = records.get(g["image_id"])
        key = (r.source_dataset or r.asset_class or r.client_id or "all") if r else "all"
        groups[key].append(g)
    out_groups: Dict[str, dict] = {}
    signals: List[dict] = []
    fallback_total = 0
    for name, rows in sorted(groups.items()):
        n = len(rows)
        by_gate = [g for g in rows if g["routed"] and "[forced:" not in g["reason"]]
        fallback = sum(1 for g in rows if "routed by default" in g["reason"])
        fallback_total += fallback
        damaged = [g for g in rows if (records.get(g["image_id"]) and (records[g["image_id"]].labels or {}).get("damage_present") is True)]
        damaged_by_gate = sum(1 for g in damaged if g["routed"] and "[forced:" not in g["reason"])
        confs = [g["confidence"] for g in rows]
        bins = {"0-0.5": sum(1 for c in confs if c < 0.5), "0.5-0.7": sum(1 for c in confs if 0.5 <= c < 0.7), "0.7-0.9": sum(1 for c in confs if 0.7 <= c < 0.9), "0.9-1": sum(1 for c in confs if c >= 0.9)}
        ratio = len({g["reason"].strip().lower() for g in rows}) / n
        p0 = ((baseline or {}).get("routed_by_gate_p0") or {}).get(name)
        lim = p_limits(p0, n) if p0 not in (None, 0, 1) else {"ucl": None, "lcl": None}
        out_groups[name] = {"n": n, "routed": sum(1 for g in rows if g["routed"]), "routed_by_gate": len(by_gate), "forced": sum(1 for g in rows if "[forced:" in g["reason"]),
                            "fallback": fallback, "declined": sum(1 for g in rows if g["reason"] == "model declined or returned no output"),
                            "unusable": sum(1 for g in rows if not g["usable"]), "unique_reason_ratio": ratio, "conf_bins": bins,
                            "labelled_damaged": len(damaged), "recall_by_gate": (damaged_by_gate / len(damaged)) if damaged else None,
                            "p0": p0, "ucl": lim["ucl"], "lcl": lim["lcl"]}
        if len(damaged) >= T("gate.dead_min_damaged") and damaged_by_gate == 0:
            signals.append(_signal("dead_gate", "gate", "alarm", 0, f">=1 of {len(damaged)} damaged routed by the gate", len(damaged), f"gate routes nothing on {name}: add the class to force_route_classes for the next run; promotion blocked on a labelled run", [g["image_id"] for g in damaged]))
        if n >= T("gate.template_min_rows") and ratio < T("gate.template_ratio"):
            signals.append(_signal("template_collapse", "gate", "watch", round(ratio, 3), T("gate.template_ratio"), n, f"gate reasons on {name} are templated; verdicts may not be image-specific"))
    if fallback_total >= T("gate.fallback_alarm"):
        signals.append(_signal("gate_fallback", "gate", "watch", fallback_total, T("gate.fallback_alarm"), len(gate_rows), "gate returned no valid JSON more than once: check the serving backend"))
    n_all = len(gate_rows)
    below = sum(1 for g in gate_rows if not g["damage_present"] and g["confidence"] < gate_min_conf)
    share = (below / n_all) if n_all else None
    inert = bool(n_all >= T("gate.inert_min_rows") and below == 0)
    ev = None
    labelled = [r for r in records.values() if "damage_present" in (r.labels or {})]
    if labelled and gate_rows and not allow_eval and any(r.split == "eval_v1" for r in labelled):
        ev = {"gate": {}, "per_dataset": {}, "note": "skipped: run holds eval_v1 records; eval_v1 is scored only by the one-look ledger (eval/run_eval.py)"}
    elif labelled and gate_rows:
        by_gate_rows = [{**g, "routed": bool(g["routed"] and "[forced:" not in g["reason"])} for g in gate_rows]
        m = eval_matrix(records, by_gate_rows, [])
        ev = {"gate": m["gate"], "per_dataset": m["per_dataset"], "note": "recomputed on routed_by_gate (forced routing excluded)"}
    return {"groups": out_groups, "below_min_conf_share": share, "min_conf_inert": inert, "eval": ev, "signals": signals,
            "info": ([f"gate_min_conf={gate_min_conf} has no effect on this backend ({n_all} rows, none below it)"] if inert else [])}


# ----------------------------------------------------------------------------------------------- M7 ops

def _is_local(model: str, local_models: Iterable[str]) -> bool:
    m = model or ""
    return m == "none" or m.startswith("fake") or ":" in m or m in set(local_models) or m in {os.environ.get("GATE_LOCAL_MODEL", ""), os.environ.get("GRADER_LOCAL_MODEL", "")}


def ops_health(calls: List[dict], baseline: Optional[dict] = None, local_models: Iterable[str] = ()) -> dict:
    """M7 per stage/model: median and p90 seconds with the first call per (stage, model) excluded as cold start
    (dev_gate02: 35.0 s first call versus 4.98 s median), median tokens, USD, and unpriced cloud rows."""
    groups: Dict[str, list] = defaultdict(list)
    for c in calls:
        groups[f"{c['stage']}/{c['model']}"].append(c)
    out: Dict[str, dict] = {}
    signals: List[dict] = []
    for key, rows in sorted(groups.items()):
        warm = rows[1:] if len(rows) > 1 else rows
        out[key] = {"calls": len(rows), "median_seconds": _median([r["seconds"] for r in warm]), "p90_seconds": _pct([r["seconds"] for r in warm], 0.9),
                    "median_input_tokens": _median([r["input_tokens"] for r in rows]), "median_output_tokens": _median([r["output_tokens"] for r in rows]),
                    "usd": round(sum(r["usd"] for r in rows), 5), "cold_start_excluded": rows[0]["image_id"] if len(rows) > 1 else None}
    unpriced = [c for c in calls if c["usd"] == 0 and not _is_local(c["model"], local_models)]
    if unpriced:
        signals.append(_signal("unpriced", "ops", "alarm", len(unpriced), 0, len(calls), "every USD figure for this run reads 'unpriced' until PRICES_PER_MTOK covers the model", sorted({c["model"] for c in unpriced})))
    ratios: Dict[str, Optional[float]] = {}
    if baseline:
        grade = [v for k, v in out.items() if k.startswith("grade/")]
        if grade:
            g = grade[0]
            for field_, key, rule, thr in (("median_output_tokens", "grade_median_output_tokens", "verbosity", T("ops.tokens_ratio")), ("median_seconds", "grade_median_seconds", "latency", T("ops.seconds_ratio"))):
                b = baseline.get(key)
                if b and g[field_] is not None:
                    ratios[rule] = g[field_] / b
                    if ratios[rule] > thr:
                        signals.append(_signal(rule, "ops", "watch", round(ratios[rule], 2), thr, g["calls"], f"grade {field_} is {ratios[rule]:.2f}x the baseline"))
    return {"by_stage_model": out, "unpriced": unpriced, "ratios": ratios, "signals": signals}


# ----------------------------------------------------------------------------------------------- M8 input drift

def input_shift(records: List[ImageRecord], baseline: Optional[dict] = None) -> dict:
    """M8 from the run manifest: resolution, missing GSD / irradiance / capture date, duplicates, video frame gaps."""
    n = len(records)
    if not n:
        return {"n": 0, "signals": [], "not_comparable_reason": None}
    longs = sorted(max(r.width, r.height) for r in records)
    small = [r.image_id for r in records if max(r.width, r.height) < 256 and r.asset_class != "pv_module"]
    pv = [r for r in records if r.asset_class == "pv_module"]
    dups: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        dups[r.sha256].append(r.image_id)
    dups_within = {s: ids for s, ids in dups.items() if len(ids) > 1}
    video: Dict[str, dict] = {}
    for src in sorted({r.source_video for r in records if r.source_video}):
        times = sorted(r.frame_time_s for r in records if r.source_video == src and r.frame_time_s is not None)
        gaps = [b - a for a, b in zip(times, times[1:])]
        video[src] = {"frames": sum(1 for r in records if r.source_video == src), "min_gap_s": min(gaps) if gaps else None}
    null_gsd = sum(1 for r in records if r.gsd_mm_per_px is None) / n
    out = {"n": n, "long_side_pcts": {"p10": _pct(longs, 0.1), "p50": _pct(longs, 0.5), "p90": _pct(longs, 0.9)},
           "share_below_256px": len(small) / n, "null_gsd_share": null_gsd,
           "null_irradiance_share": (sum(1 for r in pv if r.irradiance_wm2 is None) / len(pv)) if pv else None,
           "captured_null_share": sum(1 for r in records if not r.captured_on) / n, "dups_within": dups_within, "video": video,
           "asset_mix": dict(Counter(r.asset_class for r in records)), "not_comparable_reason": None, "signals": []}
    b_null = (baseline or {}).get("null_gsd_share")
    if b_null is not None and abs(null_gsd - b_null) > T("input.null_gsd_delta"):
        out["not_comparable_reason"] = f"null GSD share {null_gsd:.2f} vs baseline {b_null:.2f}"
        out["signals"].append(_signal("input_not_comparable", "input", "watch", round(null_gsd - b_null, 3), T("input.null_gsd_delta"), n, "U and shift charts marked not comparable: capture metadata changed, do not fix this in the prompt"))
    if out["share_below_256px"] > T("input.below_256_share"):
        out["signals"].append(_signal("resolution", "input", "watch", round(out["share_below_256px"], 3), T("input.below_256_share"), n, "many small images: grades are resolution-sensitive (R03)", small))
    tight = {s: v for s, v in video.items() if v["min_gap_s"] is not None and v["min_gap_s"] < T("input.min_frame_gap_s")}
    if tight:
        out["signals"].append(_signal("frame_gap", "input", "watch", min(v["min_gap_s"] for v in tight.values()), T("input.min_frame_gap_s"), n, "near-duplicate frames inflate counts: client reports use per-source_video denominators", sorted(tight)))
    if dups_within:
        out["signals"].append(_signal("duplicates", "input", "info", len(dups_within), 0, n, "duplicate images in the run", [ids for ids in dups_within.values()]))
    return out


# ----------------------------------------------------------------------------------------------- M6 reviewer loop

def review_health(rows: List[dict], findings: List[Finding], window: Optional[int] = None, gate_rows: Optional[List[dict]] = None) -> dict:
    """M6 from ReviewLog.decisions() rows: agreement (all and rolling), per-reviewer, override direction, critical
    misses, blind QC pairs and whether a low agreement is attributable to the model or to one reviewer.

    A critical miss is a reviewer override to S4 (review stage, zero tolerance, blocking); `fingerprint_blocked`
    is true once `review.critical_miss_block` of them sit in the last 100 sighted decisions. A force-routed image
    the gate called clean but the grader put at S3/S4 is a gate signal (`gate_miss` watch, gate stage), not a
    reviewer one. Blind pairs are deduplicated per finding (last blind grade wins)."""
    window = T("review.rolling_window") if window is None else window
    sighted = [r for r in rows if r.get("action") != "blind"]
    blind = [r for r in rows if r.get("action") == "blind"]
    n = len(sighted)
    acc = [1 if r["action"] == "accepted" else 0 for r in sighted]
    by_rev: Dict[str, list] = defaultdict(list)
    for r, a in zip(sighted, acc):
        by_rev[r.get("reviewer") or "?"].append(a)
    overrides = [r for r in sighted if r["action"] == "overridden" and r.get("prior_level") in LEVEL_ORDER and r.get("new_level") in LEVEL_ORDER]
    deltas = [LEVEL_ORDER[r["new_level"]] - LEVEL_ORDER[r["prior_level"]] for r in overrides]
    up = sum(1 for d in deltas if d > 0)
    big = [r for r, d in zip(overrides, deltas) if abs(d) >= 2]
    is_miss = lambda r: r["action"] == "overridden" and r.get("new_level") == "S4" and r.get("prior_level") in LEVEL_ORDER and r["prior_level"] != "S4"  # noqa: E731
    misses = [r for r in overrides if is_miss(r)]
    recent_misses = sum(1 for r in sighted[-100:] if is_miss(r))
    blocked = recent_misses >= T("review.critical_miss_block")
    gate_misses: List[dict] = []
    clean_forced: set = set()
    if gate_rows:
        clean_forced = {g["image_id"] for g in gate_rows if g.get("damage_present") is False and "[forced:" in g.get("reason", "")}
        gate_misses = [{"finding_id": f.finding_id, "image_id": f.evidence.image_ids[0], "level": f.unified.level}
                       for f in findings if f.unified.level in ("S3", "S4") and f.evidence.image_ids and f.evidence.image_ids[0] in clean_forced]
    sighted_level = {}
    for r in sighted:
        sighted_level[r["finding_id"]] = r["new_level"] if r["action"] in ("overridden", "marked_u") else r["prior_level"]
    last_blind: Dict[str, dict] = {}
    for r in blind:
        last_blind[r["finding_id"]] = r
    pairs = [(r["new_level"], sighted_level[fid]) for fid, r in last_blind.items() if fid in sighted_level]
    ok = [(a, b) for a, b in pairs if a in LEVEL_ORDER and b in LEVEL_ORDER]
    blind_out = {"n": len(pairs), "within_one": (sum(1 for a, b in ok if abs(LEVEL_ORDER[a] - LEVEL_ORDER[b]) <= 1) / len(pairs)) if pairs else None,
                 "exact": (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else None}
    rev_table = {k: {"n": len(v), "agreement": sum(v) / len(v)} for k, v in by_rev.items()}
    rolling = (sum(acc[-window:]) / len(acc[-window:])) if acc else None
    attribution = "insufficient_n"
    if n >= window:
        attribution = "model"
        for name, v in rev_table.items():
            others = [a for k, vals in by_rev.items() if k != name for a in vals]
            if v["n"] >= T("review.reviewer_min_n") and len(others) >= T("review.reviewer_min_n") and abs(v["agreement"] - sum(others) / len(others)) > T("review.reviewer_spread"):
                attribution = "reviewer"
    signals: List[dict] = []
    if misses:
        thr = T("review.critical_miss_block")
        signals.append(_signal("critical_miss", "review", "alarm", len(misses), 0, n,
                               f"zero tolerance: an S4 the model did not flag; {recent_misses} in the last 100 decisions (block at {thr})" + ("; fingerprint blocked from customer runs" if blocked else ""),
                               [m["finding_id"] for m in misses]))
    if gate_misses:
        signals.append(_signal("gate_miss", "gate", "watch", len(gate_misses), 0, len(clean_forced),
                               "gate called these force-routed images clean but the grader found S3/S4: forced routing caught them; the gate misses on this class",
                               [m["finding_id"] for m in gate_misses]))
    if len(overrides) >= T("review.min_overrides") and up / len(overrides) >= T("review.upward_share"):
        signals.append(_signal("under_grading", "review", "alarm", round(up / len(overrides), 3), T("review.upward_share"), len(overrides), "model under-grading: all S0/S1 findings of the class need review before export"))
    if n >= window and rolling is not None and rolling < T("review.agreement_watch"):
        signals.append(_signal("agreement", "review", "watch", round(rolling, 3), T("review.agreement_watch"), n, "reviewer calibration: schedule the 10-finding double grading" if attribution == "reviewer" else "model disagreement over the last window"))
    if blind_out["n"] >= T("review.blind_min_n") and blind_out["within_one"] is not None and blind_out["within_one"] < T("review.blind_floor"):
        signals.append(_signal("blind_floor", "review", "alarm", round(blind_out["within_one"], 3), T("review.blind_floor"), blind_out["n"], "blind re-grades disagree with sighted decisions beyond the human floor: automation bias check"))
    return {"n": n, "agreement_all": (sum(acc) / n) if n else None, "agreement_rolling": rolling, "by_reviewer": rev_table,
            "overrides": {"n": len(overrides), "up": up, "down": sum(1 for d in deltas if d < 0), "upward_share": (up / len(overrides)) if overrides else None},
            "marked_u": sum(1 for r in sighted if r["action"] == "marked_u"), "big_overrides": big, "critical_misses": misses, "recent_misses": recent_misses,
            "fingerprint_blocked": blocked, "gate_misses": gate_misses, "blind": blind_out, "attribution": attribution, "signals": signals}


def blind_sample(findings: List[Finding], per_class: int = 10, seed: str = "", cap: int = 40, exclude_ids: Optional[set] = None) -> List[Finding]:
    """Loop 3: accepted findings for blind re-grading, stratified by level within each asset class; bounded.
    `exclude_ids` (findings that already hold a blind row) are never offered twice."""
    rng = random.Random(seed)
    out: List[Finding] = []
    by_ac: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    exclude_ids = exclude_ids or set()
    for f in findings:
        if f.review.status == "accepted" and f.finding_id not in exclude_ids:
            by_ac[f.asset_class][f.unified.level].append(f)
    for ac in sorted(by_ac):
        pool = by_ac[ac]
        for lst in pool.values():
            rng.shuffle(lst)
        picked: List[Finding] = []
        while len(picked) < per_class and any(pool.values()):
            for lvl in sorted(pool):
                if pool[lvl] and len(picked) < per_class:
                    picked.append(pool[lvl].pop())
        out += picked
    return out[:cap]


# ----------------------------------------------------------------------------------------------- M11 sample size

def z_quantile(p: float) -> float:
    """Normal quantile, Abramowitz and Stegun 26.2.23 rational approximation (abs error < 4.5e-4), no scipy."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    lower = p < 0.5
    t = math.sqrt(-2.0 * math.log(p if lower else 1 - p))
    z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3)
    return -z if lower else z


def min_n_for_shift(p0: float, p1: float, alpha: float = 0.01, power: float = 0.8, one_sided: bool = False) -> int:
    """Per-arm n to detect a proportion moving from p0 to p1 (two-sample normal approximation)."""
    za = z_quantile(1 - alpha) if one_sided else z_quantile(1 - alpha / 2)
    zb = z_quantile(power)
    pbar = (p0 + p1) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / (p1 - p0) ** 2)


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ----------------------------------------------------------------------------------------------- baseline cards

def _records_for(run_dir: Path, records: Optional[Dict[str, ImageRecord]] = None) -> Dict[str, ImageRecord]:
    if records:
        return records
    mp = Path(run_dir) / "manifest.jsonl"
    return {r.image_id: r for r in read_manifest(mp)} if mp.exists() else {}


def eval_records(records: Iterable[ImageRecord], eval_manifest: Path = EVAL_MANIFEST) -> List[str]:
    """image_ids that belong to eval_v1 by split, or by sha256 present in the frozen eval manifest."""
    frozen = {r.sha256 for r in read_manifest(eval_manifest)} if Path(eval_manifest).exists() else set()
    return [r.image_id for r in records if r.split in ("eval", "eval_v1") or r.sha256 in frozen]


def build_baseline(run_dir: Path, fp_id: str, out_dir: Path = BASELINE_DIR, client_id: Optional[str] = None, records: Optional[Dict[str, ImageRecord]] = None) -> Dict[str, dict]:
    """One frozen card per asset class present in the run (or one client card). Cards are never rolled; a new
    fingerprint adds a card under `cards` and becomes `active`. Refuses (RuntimeError) a run holding any eval_v1
    record by split or by sha256 in the frozen eval manifest: eval_v1 is never used for baselines."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    run = load_run(run_dir)
    recs = _records_for(run_dir, records)
    leaked = eval_records(recs.values())
    if leaked:
        raise RuntimeError(f"{run_dir.name} holds eval_v1 records ({leaked[:5]}): eval_v1 is never frozen into a baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    findings, calls, gate_rows = run["findings"], run["calls"], run["gate"]
    grade_calls = [c for c in calls if c["stage"] == "grade"]
    cfg = run["summary"].get("config", {})
    audit = contract_audit(findings, lambda ac: load_rubric(ac, cfg.get("rubric_file") if ac == "bridge_element" else None), calls)
    gh = gate_health(gate_rows, recs, calls, cfg.get("gate_min_conf", 0.7))
    cards: Dict[str, dict] = {}
    classes = [f"client_{client_id}"] if client_id else sorted({f.asset_class for f in findings} or {r.asset_class for r in recs.values()})
    for ac in classes:
        fs = findings if client_id else [f for f in findings if f.asset_class == ac]
        ub = u_breakdown(fs, calls)
        card = {"built_from": [run_dir.name], "built_on": _now(), "frozen": True,
                "n_images": len({i for f in fs for i in f.evidence.image_ids}), "n_findings": len(fs),
                "u_p0": ub["rate"], "u_by_cause_p0": {c: v["rate"] for c, v in ub["by_cause"].items()},
                "level_shares": {l: (sum(1 for f in fs if f.unified.level == l) / len(fs)) if fs else None for l in LEVELS},
                "native_shares": {k: v / len(fs) for k, v in Counter(f.native_scale.value for f in fs).items()} if fs else {},
                "routed_by_gate_p0": {g: (v["routed_by_gate"] / v["n"]) if v["n"] else None for g, v in gh["groups"].items()},
                "grade_median_seconds": _median([c["seconds"] for c in grade_calls[1:] or grade_calls]),
                "grade_median_input_tokens": _median([c["input_tokens"] for c in grade_calls]),
                "grade_median_output_tokens": _median([c["output_tokens"] for c in grade_calls]),
                "clean_grade_calls": sum(1 for c in grade_calls if c.get("note") in ("end_turn", "", None)),
                "null_gsd_share": (sum(1 for r in recs.values() if r.gsd_mm_per_px is None) / len(recs)) if recs else None,
                "hard_violations": audit["hard_total"], "served_models": first_served(calls), "self_consistency": None, "notes": ""}
        path = out_dir / f"{ac}.json"
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"asset_class": ac, "active": None, "cards": {}}
        doc["cards"][fp_id] = card
        doc["active"] = fp_id
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        cards[ac] = card
    return cards


def load_baseline(asset_class: str, fp_id: Optional[str] = None, client_id: Optional[str] = None, out_dir: Path = BASELINE_DIR) -> Optional[dict]:
    """Active (or `fp_id`) card for a class; a client card wins when present. None when no card exists."""
    for name in ([f"client_{client_id}"] if client_id else []) + [asset_class]:
        path = Path(out_dir) / f"{name}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        card = doc["cards"].get(fp_id or doc.get("active") or "")
        if card:
            return {**card, "fingerprint": fp_id or doc.get("active"), "asset_class": name}
    return None


def baseline_freeze(run_dir: Path, asset_class: str, out_dir: Path = BASELINE_DIR, fp_id: Optional[str] = None) -> dict:
    """Freeze one class's card from a run; keyed by the run's fingerprint `model_id` (see identity_of) unless given."""
    run_dir = Path(run_dir)
    if fp_id is None:
        fpp = run_dir / "fingerprint.json"
        fp_id = identity_of(json.loads(fpp.read_text(encoding="utf-8"))) if fpp.exists() else "reconstructed"
    cards = build_baseline(run_dir, fp_id, out_dir)
    if asset_class not in cards:
        raise KeyError(f"{asset_class} has no findings in {run_dir.name}")
    return cards[asset_class]


baseline_load = load_baseline


# ----------------------------------------------------------------------------------------------- health entry point

def _apply_budget(signals: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    """M11: rank, one alarm surfaced per stage, the rest demoted to watch; BLOCKING_RULES are never demoted."""
    alarms, watch, info = [], [], []
    seen_stage: set = set()
    for s in sorted(signals, key=lambda s: (s["rank"], s["rule"])):
        if s["severity"] == "alarm":
            if s["stage"] in seen_stage and s["rule"] not in BLOCKING_RULES:
                watch.append({**s, "severity": "watch", "demoted_from": "alarm"})
            else:
                seen_stage.add(s["stage"])
                alarms.append(s)
        elif s["severity"] == "watch":
            watch.append(s)
        else:
            info.append(s)
    return alarms, watch, info


def health(run_dir: Path, *, records: Optional[Dict[str, ImageRecord]] = None, review_rows: Optional[List[dict]] = None, write: bool = True, baseline_dir: Path = BASELINE_DIR) -> dict:
    """Loop 1: run M1 to M8 on a run folder with zero model calls and write `health.json`.

    Comparisons against a baseline card are drawn only when the run's fingerprint `model_id` (identity_of) equals
    the card's active id; otherwise the run is a `candidate`. Every asset class in the run is charted against
    its own card (M3 per class; the run-wide chart carries counts only when classes are mixed). Per-stage status
    is in_control / watch / alarm / insufficient_n with the worst rule and n printed for the status strip.
    An acknowledgement already stored in health.json is carried forward while the alarm rule set stays inside it.
    """
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    findings, calls, gate_rows, summary = run["findings"], run["calls"], run["gate"], run["summary"]
    cfg = summary.get("config", {})
    recs = _records_for(run_dir, records)
    fpp = run_dir / "fingerprint.json"
    fp = json.loads(fpp.read_text(encoding="utf-8")) if fpp.exists() else None
    fp_id = fp["id"] if fp else summary.get("fingerprint")
    model_id = identity_of(fp) if fp else summary.get("fingerprint")
    classes = sorted({f.asset_class for f in findings} or {r.asset_class for r in recs.values()})
    baselines = {ac: load_baseline(ac, out_dir=baseline_dir) for ac in classes}
    cards = {ac: b for ac, b in baselines.items() if b}
    baseline = next(iter(cards.values()), None)
    baseline_id = baseline["fingerprint"] if baseline else None
    comparable_classes = sorted(ac for ac, b in cards.items() if model_id and b["fingerprint"] == model_id)
    mid_run = summary.get("fingerprint_at_finish")
    comparable = bool(comparable_classes) and not (mid_run and mid_run != fp_id)
    base = cards[comparable_classes[0]] if comparable else None
    signals: List[dict] = []

    models = check_models(run, fp, pinned_served=(base or {}).get("served_models"))
    if models["mixed"]:
        ids = {s: sorted({c["image_id"] for c in calls if c["stage"] == s and c["model"] == m}) for s, ms in models["models_per_stage"].items() for m in ms if len(ms) > 1}
        signals.append(_signal("mixed_models", "config", "alarm", models["models_per_stage"], "one model per stage", len(calls), "run labelled mixed and excluded from comparisons; re-grade one half", [f"{k}: {v}" for k, v in ids.items()]))
    served_items = [f"{r['model']} -> {r['served_model']}" for r in models["served_mismatch"]] + [f"{m} served as {v}" for m, v in models["served_multiple"].items()] + [f"{m} pinned {p} now {v}" for m, (p, v) in models["served_changed"].items()]
    if served_items:
        signals.append(_signal("served_mismatch", "config", "alarm", len(served_items), 0, len(calls), "served model is not the requested id, or one id resolved to several snapshots, or the snapshot differs from the baseline pin: run the canary before exporting", served_items))
    if mid_run and mid_run != fp_id:
        signals.append(_signal("fingerprint_changed_mid_run", "config", "alarm", mid_run, fp_id, len(calls), "the run was resumed under a different fingerprint (rubric, prompt, exemplar, effort or config edit): halves are not one population; re-grade one half"))
    if baseline and model_id and not comparable_classes:
        signals.append(_signal("fingerprint_diff", "config", "watch", model_id, baseline_id, len(calls), "candidate: fingerprint differs from the active baseline; charts against the baseline are not drawn"))
    if fp and (cfg.get("gate") == "local" or cfg.get("grader") == "local"):
        comp = fp.get("components", {})
        need = ([comp.get("model.gate_local")] if cfg.get("gate") == "local" else []) + ([comp.get("model.grader_local")] if cfg.get("grader") == "local" else [])
        missing = [t for t in need if t and comp.get(f"ollama.{t}") is None]
        if missing:
            signals.append(_signal("digest_unavailable", "config", "watch", missing, "digest per local tag", len(calls), "Ollama /api/tags did not answer at fingerprint time: the local model digest is unverified for this run (id unaffected)", missing))
    active_path = Path(baseline_dir) / "active.json"
    if fp and active_path.exists():
        try:
            active_fp = json.loads(active_path.read_text(encoding="utf-8")).get("fingerprint", {})
            env_diff = [k for k, a, b in fingerprint_diff(active_fp, fp) if k.startswith("model.")]
            if env_diff:
                signals.append(_signal("env_differs_from_active", "config", "watch", env_diff, "active.json model.*", len(calls), "model ids differ from the promoted baseline (after a rollback: restore .env)", env_diff))
        except Exception:
            pass

    def rubric_for(ac: str) -> dict:
        return load_rubric(ac, cfg.get("rubric_file") if ac == "bridge_element" else None)

    contract = contract_audit(findings, rubric_for, calls)
    signals += contract_signals(contract, base)
    inp = input_shift(list(recs.values()), base)
    signals += inp.get("signals", [])
    if len(classes) <= 1:
        uc = u_chart(findings, calls, base, inp.get("not_comparable_reason"))
    else:  # mixed classes: each class against its own card; the run-wide entry carries counts, no limits
        uc = u_chart(findings, calls, None, inp.get("not_comparable_reason"))
        per_class = {}
        for ac in classes:
            c = u_chart([f for f in findings if f.asset_class == ac], calls, cards.get(ac) if (comparable and ac in comparable_classes) else None, inp.get("not_comparable_reason"))
            for s in c["signals"]:
                s["action"] = f"{ac}: {s['action']}"
            per_class[ac] = c
        uc["per_class"] = per_class
        uc["status"] = "per class: " + ", ".join(f"{ac} {c['status']}" for ac, c in per_class.items())
        uc["signals"] = [s for c in per_class.values() for s in c["signals"]]
    signals += uc["signals"]
    shift = None
    if base and base.get("built_from") and not inp.get("not_comparable_reason"):
        base_run = run_dir.parent / base["built_from"][0]
        if base_run.exists() and base_run != run_dir:
            shift = output_shift(base_run, run_dir)
            signals += shift["signals"]
    gate = gate_health(gate_rows, recs, calls, cfg.get("gate_min_conf", 0.7), base)
    signals += gate["signals"]
    ops = ops_health(calls, base)
    signals += ops["signals"]
    if review_rows is None and (run_dir / "reviews.sqlite").exists():
        from .review import ReviewLog

        review_rows = ReviewLog(run_dir / "reviews.sqlite").decisions()
    review = review_health(review_rows or [], findings, gate_rows=gate_rows)
    signals += review["signals"]

    alarms, watch, info = _apply_budget(signals)
    n_by_stage = {"config": len(calls), "gate": len(gate_rows), "grade": len(findings), "review": review["n"], "ops": len(calls), "input": len(recs)}
    stages: Dict[str, dict] = {}
    for st in STAGES:
        hits = [s for s in alarms if s["stage"] == st] or [s for s in watch if s["stage"] == st]
        floor = T(f"floors.{st}")
        if hits:
            status = hits[0]["severity"]
        elif n_by_stage[st] < floor:
            status = "insufficient_n"
        else:
            status = "in_control"
        stages[st] = {"status": status, "worst_rule": hits[0]["rule"] if hits else None, "n": n_by_stage[st], "n_required": floor if status == "insufficient_n" else None}
    overall = "alarm" if alarms else "watch" if watch else ("in_control" if comparable else "candidate")
    prev_ack = None
    hp = run_dir / "health.json"
    if hp.exists():
        try:
            prev_ack = json.loads(hp.read_text(encoding="utf-8")).get("acknowledged")
        except Exception:
            prev_ack = None
    acknowledged = prev_ack if prev_ack and {a["rule"] for a in alarms} <= set(prev_ack.get("rules") or []) else None
    doc = {"run": run_dir.name, "generated_at": _now(), "fingerprint_id": fp_id, "model_id": model_id, "baseline_id": baseline_id, "comparable": comparable,
           "comparable_classes": comparable_classes, "status": overall, "stages": stages, "alarms": alarms, "watch": watch,
           "info": info + [{"rule": "gate_min_conf_inert", "detail": t} for t in gate["info"]],
           "acknowledged": acknowledged, "models": models, "contract": contract, "u_chart": uc, "output_shift": shift, "gate": gate, "ops": ops,
           "input": inp, "review": review, "canary": None, "thresholds_sha8": _sha(THRESHOLDS_PATH.read_bytes())[:8]}
    if write:
        (run_dir / "health.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


# ----------------------------------------------------------------------------------------------- M9 canary

def canary_manifest(dev_manifest: Path, out: Path = CANARY_MANIFEST, exemplar_k: int = 3, seed: int = 0) -> List[ImageRecord]:
    """14 fixed inputs: 3 dev images per class disjoint from the exemplar picks (bridge_element has no native grade,
    so its 3 lowest image_ids), plus a blank grey frame and a uniform-noise frame as synthetic controls."""
    records = read_manifest(Path(dev_manifest))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    picks: List[ImageRecord] = []
    for ac in RUBRIC_FOR_CLASS:
        exemplar_ids = {r.image_id for r in select_exemplars(records, ac, exemplar_k)}
        chosen = select_exemplars(records, ac, 3, exclude_ids=exemplar_ids)
        if len(chosen) < 3:
            pool = sorted((r for r in records if r.asset_class == ac and r.split != "eval_v1" and r.image_id not in exemplar_ids and r not in chosen), key=lambda r: r.image_id)
            chosen += pool[: 3 - len(chosen)]
        picks += [r.model_copy(update={"split": "canary"}) for r in chosen]
    controls = []
    blank = Image.new("RGB", (1024, 768), (128, 128, 128))
    rng = random.Random(seed)
    noise = Image.new("RGB", (1024, 768))
    noise.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(1024 * 768)])
    for name, img in (("blank", blank), ("noise", noise)):
        p = out.parent / ("blank_grey.png" if name == "blank" else "noise.png")
        img.save(p)
        controls.append(ImageRecord(image_id=f"canary_{name}", path=str(p), sha256=sha256_of(p), width=img.width, height=img.height, asset_class="steel_coating",
                                    source_dataset="canary_control", split="canary", labels={"control": name}))
    rows = picks + controls
    write_manifest(rows, out)
    return rows


def _canary_observed(run_dir: Path) -> Dict[str, dict]:
    run = load_run(Path(run_dir))
    tokens = {c["image_id"]: c["input_tokens"] for c in run["calls"] if c["stage"] == "grade"}
    obs: Dict[str, dict] = {}
    for f in run["findings"]:
        iid = f.evidence.image_ids[0] if f.evidence.image_ids else f.finding_id
        if iid in obs and f.evidence.tile != "full":
            continue
        obs[iid] = {"native": f.native_scale.value, "level": f.unified.level, "input_tokens": tokens.get(iid), "refused": u_cause(f) == "refusal"}
    return obs


def _control_ids(run_dir: Optional[Path]) -> Dict[str, str]:
    mp = Path(run_dir) / "manifest.jsonl" if run_dir and (Path(run_dir) / "manifest.jsonl").exists() else CANARY_MANIFEST
    if not mp.exists():
        return {}
    return {r.image_id: r.labels["control"] for r in read_manifest(mp) if r.labels.get("control")}


def canary_reference(run_dirs: List[Path], out: Path, controls: Optional[Dict[str, str]] = None) -> dict:
    """Reference from repeats of the canary set under one fingerprint: modal native value and level per image,
    input tokens, and f0 = repeats whose native value differs from the mode / (dev images x repeats)."""
    controls = controls if controls is not None else _control_ids(run_dirs[0] if run_dirs else None)
    observed = [_canary_observed(d) for d in run_dirs]
    fpp = Path(run_dirs[0]) / "fingerprint.json"
    fp_id = json.loads(fpp.read_text(encoding="utf-8"))["id"] if fpp.exists() else None
    images: Dict[str, dict] = {}
    flips_total = 0
    for iid in sorted(set().union(*[set(o) for o in observed])):
        reps = [o[iid] for o in observed if iid in o]
        mode_native = Counter(r["native"] for r in reps).most_common(1)[0][0]
        mode_level = Counter(r["level"] for r in reps).most_common(1)[0][0]
        flips = sum(1 for r in reps if r["native"] != mode_native)
        images[iid] = {"mode_native": mode_native, "mode_level": mode_level, "input_tokens": reps[0]["input_tokens"], "flips": flips, "control": controls.get(iid)}
        if iid not in controls:
            flips_total += flips
    n_dev = sum(1 for i in images if i not in controls)
    f0 = flips_total / (n_dev * len(run_dirs)) if n_dev and run_dirs else 0.0
    ref = {"fingerprint": fp_id, "n_repeats": len(run_dirs), "images": images, "f0": f0, "alarm_k": alarm_line(n_dev, f0), "built_on": _now()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=1), encoding="utf-8")
    return ref


def alarm_line(n: int, f0: float, tail: Optional[float] = None) -> int:
    """Smallest k with P(X >= k | n, f0) < tail; with f0 = 0 any flip alarms."""
    tail = T("canary.tail") if tail is None else tail
    if f0 <= 0:
        return 1
    for k in range(1, n + 1):
        if binom_tail(k, n, f0) < tail:
            return k
    return n + 1


def canary_verdict(observed: Dict[str, dict], reference: dict) -> dict:
    """Pure M9 comparison of one canary run against the reference (see canary_compare for the file reader)."""
    images = reference["images"]
    controls = {i: v.get("control") for i, v in images.items() if v.get("control")}
    dev = [i for i in images if i not in controls]
    k = sum(1 for i in dev if i in observed and observed[i]["native"] != images[i]["mode_native"])
    big = []
    for i in dev:
        if i not in observed:
            continue
        a, b = images[i]["mode_level"], observed[i]["level"]
        if a in LEVEL_ORDER and b in LEVEL_ORDER and (abs(LEVEL_ORDER[a] - LEVEL_ORDER[b]) >= 2 or (a == "S4") != (b == "S4")):
            big.append(i)
        elif (a == "S4") != (b == "S4") and "U" in (a, b):
            big.append(i)
    alarm_k = reference.get("alarm_k") or alarm_line(len(dev), reference.get("f0", 0.0))
    p_tail = binom_tail(k, len(dev), reference.get("f0", 0.0)) if dev else 1.0
    blank = next((observed[i]["level"] for i, c in controls.items() if c == "blank" and i in observed), None)
    noise = next((observed[i]["level"] for i, c in controls.items() if c == "noise" and i in observed), None)
    mism = [i for i in images if i in observed and images[i].get("input_tokens") is not None and observed[i].get("input_tokens") is not None and observed[i]["input_tokens"] != images[i]["input_tokens"]]
    refusals = [i for i in images if i in observed and observed[i].get("refused") and images[i]["mode_level"] != "U"]
    reasons = []
    if k >= alarm_k:
        reasons.append(f"{k} of {len(dev)} native values differ from the reference mode (alarm line {alarm_k}, f0 {reference.get('f0', 0.0):.3f})")
    if big:
        reasons.append(f"2-level or S4-boundary flips: {big}")
    if blank in LEVEL_ORDER and LEVEL_ORDER[blank] >= 1:
        reasons.append(f"blank frame graded {blank} (hallucination)")
    if noise is not None and noise != "U":
        reasons.append(f"noise frame graded {noise} instead of U")
    if mism:
        reasons.append(f"input tokens differ on the same bytes: {mism}")
    if refusals:
        reasons.append(f"refused images that graded in the reference: {refusals}")
    return {"k": k, "n": len(dev), "alarm_k": alarm_k, "p_tail": p_tail, "big": big, "blank_level": blank, "noise_level": noise,
            "input_token_mismatches": mism, "refusals": refusals, "status": "alarm" if reasons else "pass", "reasons": reasons}


def canary_compare(run_dir: Path, reference: dict) -> dict:
    """M9 on a stored canary run folder."""
    return canary_verdict(_canary_observed(run_dir), reference)


# ----------------------------------------------------------------------------------------------- M10 eval guard, ledger, promotion

def _ledger_rows(ledger: Path) -> List[dict]:
    return [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()] if Path(ledger).exists() else []


def eval_guard(manifest: Path, fp_id: str, extra_ids: Iterable[str] = (), ledger: Path = LEDGER) -> None:
    """Refuse an eval_v1 look when the manifest hash differs from FROZEN.sha256, when the ledger already holds a row
    for this fingerprint (one look per fingerprint), or when any canary / exemplar / override id or sha256 is in
    the eval set. Raises RuntimeError with the reason; returns None when the look is allowed."""
    manifest = Path(manifest)
    frozen = manifest.parent / "FROZEN.sha256"
    if not frozen.exists():
        raise RuntimeError(f"no FROZEN.sha256 beside {manifest}")
    expected = frozen.read_text(encoding="utf-8").split()[0]
    actual = _sha(manifest.read_bytes())
    if actual != expected:
        raise RuntimeError(f"eval manifest sha256 {actual[:12]} != FROZEN {expected[:12]}: the frozen set was edited")
    prior = [r for r in _ledger_rows(ledger) if r.get("fingerprint") == fp_id]
    if prior:
        raise RuntimeError(f"fingerprint {fp_id} already had its one eval look: {prior}")
    extra = set(extra_ids)
    if extra:
        leaked = [r.image_id for r in read_manifest(manifest) if r.image_id in extra or r.sha256 in extra]
        if leaked:
            raise RuntimeError(f"eval set contains canary/exemplar/override images: {leaked[:10]}")


def ledger_append(fp_id: str, run_name: str, manifest_sha: str, verdict: str, ledger: Path = LEDGER) -> dict:
    """Append-only record of every eval_v1 look."""
    row = {"at": _now(), "fingerprint": fp_id, "run": run_name, "manifest_sha": manifest_sha, "verdict": verdict}
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def canary_status(model_id: Optional[str], canary_dir: Path = CANARY_DIR) -> dict:
    """Reference and latest canary verdict filed under CANARY_DIR/<model_id>: `ok` is true only when the newest
    canary run (folders other than ref_*) has canary.status == pass and is newer than reference.json."""
    out = {"reference": None, "reference_built_on": None, "latest_run": None, "latest_status": None, "ok": False}
    if not model_id:
        return out
    base = Path(canary_dir) / model_id
    ref = base / "reference.json"
    if not ref.exists():
        return out
    out["reference"] = str(ref)
    try:
        out["reference_built_on"] = json.loads(ref.read_text(encoding="utf-8")).get("built_on")
    except Exception:
        pass
    runs = sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("ref_") and (p / "health.json").exists())
    if runs:
        latest = runs[-1]
        try:
            verdict = (json.loads((latest / "health.json").read_text(encoding="utf-8")).get("canary") or {})
        except Exception:
            verdict = {}
        out["latest_run"] = latest.name
        out["latest_status"] = verdict.get("status")
        out["ok"] = verdict.get("status") == "pass" and (latest / "health.json").stat().st_mtime >= ref.stat().st_mtime
    return out


def promotion_check(candidate_eval: dict, baseline_eval: dict, contract: dict, canary: Optional[dict] = None) -> dict:
    """M10 rules per class with n_assessed >= promote.min_n: within-one, U rate, gate recall, USD per image, hard = 0,
    and, when `canary` (a canary_status dict) is given, a passing latest canary.
    Classes below the floor return insufficient_n and neither block nor promote."""
    per_class: Dict[str, dict] = {}
    reasons: List[str] = []
    min_n = T("promote.min_n")
    for ac, c in (candidate_eval.get("grading") or {}).items():
        b = (baseline_eval.get("grading") or {}).get(ac, {})
        if (c.get("n_assessed") or 0) < min_n or (b.get("n_assessed") or 0) < min_n:
            per_class[ac] = {"within_one_delta": None, "u_rate_delta": None, "verdict": "insufficient_n", "n": c.get("n_assessed"), "n_required": min_n}
            continue
        w = c["within_one_grade"] - b["within_one_grade"]
        u = (c.get("u_rate") or 0) - (b.get("u_rate") or 0)
        ok = w >= -T("promote.within_one_margin") and u <= T("promote.u_margin")
        per_class[ac] = {"within_one_delta": w, "u_rate_delta": u, "verdict": "pass" if ok else "hold", "n": c["n_assessed"]}
        if not ok:
            reasons.append(f"{ac}: within-one delta {w:+.3f}, U delta {u:+.3f}")
    gc, gb = candidate_eval.get("gate", {}), baseline_eval.get("gate", {})
    recall_delta = (gc.get("recall") - gb.get("recall")) if gc.get("recall") is not None and gb.get("recall") is not None else None
    if recall_delta is not None and recall_delta < -T("promote.recall_margin"):
        reasons.append(f"gate recall delta {recall_delta:+.3f}")
    uc, ub = (candidate_eval.get("ops") or {}).get("usd_per_image"), (baseline_eval.get("ops") or {}).get("usd_per_image")
    usd_ratio = (uc / ub) if uc and ub else None
    if usd_ratio is not None and usd_ratio > T("promote.usd_ratio"):
        reasons.append(f"usd per image {usd_ratio:.2f}x baseline")
    if contract.get("hard_total", 0):
        reasons.append(f"{contract['hard_total']} hard contract violations on the dev run")
    if canary is not None and not canary.get("ok"):
        reasons.append(f"canary: {canary.get('latest_status') or 'no run'} (reference {'present' if canary.get('reference') else 'missing'})")
    verdicts = {v["verdict"] for v in per_class.values()}
    verdict = "hold" if reasons else ("promote" if "pass" in verdicts else "insufficient_n")
    return {"per_class": per_class, "gate_recall_delta": recall_delta, "usd_ratio": usd_ratio, "verdict": verdict, "reasons": reasons}


def promote(fp: dict, dev_run: str, canary_ref: Path, eval_report: str, by: str, baseline_dir: Path = BASELINE_DIR, docs_dir: Path = ROOT / "docs") -> Path:
    """Write runs/_baseline/active.json (keeping `previous`), build the cards, append a changelog line and a decision stub."""
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    active_path = baseline_dir / "active.json"
    previous = json.loads(active_path.read_text(encoding="utf-8")) if active_path.exists() else None
    changed = [k for k, _, _ in fingerprint_diff(previous.get("fingerprint", {}), fp)] if previous else ["initial"]
    run_dir = ROOT / "runs" / dev_run
    if run_dir.exists():
        build_baseline(run_dir, identity_of(fp), baseline_dir)
    doc = {"fingerprint": fp, "dev_run": dev_run, "canary_reference": str(canary_ref), "eval_report": eval_report, "promoted_by": by, "promoted_at": _now(),
           "previous": {k: v for k, v in previous.items() if k != "previous"} if previous else None, "changed": changed}
    active_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    line = f"- Promoted fingerprint `{fp['id']}` (dev run `{dev_run}`, eval `{eval_report}`, by {by}); changed: {', '.join(changed)}.\n"
    for name, text in (("changelog.md", line), ("decisions.md", f"\n## D-0xx. Promotion of fingerprint {fp['id']} ({_now()[:10]})\n\nStub: {line}")):
        p = Path(docs_dir) / name
        if p.exists():
            with p.open("a", encoding="utf-8") as fh:
                fh.write(text)
    return active_path


def rollback(baseline_dir: Path = BASELINE_DIR) -> dict:
    """Restore `previous` from active.json and point every class card whose `cards` hold the restored id back at
    it; returns the .env model ids, the git commit the operator must restore, and the cards re-pointed / left."""
    baseline_dir = Path(baseline_dir)
    active_path = baseline_dir / "active.json"
    doc = json.loads(active_path.read_text(encoding="utf-8"))
    prev = doc.get("previous")
    if not prev:
        raise RuntimeError("nothing to roll back to")
    active_path.write_text(json.dumps({**prev, "rolled_back_from": doc["fingerprint"]["id"], "rolled_back_at": _now()}, indent=1), encoding="utf-8")
    restored_key = identity_of(prev["fingerprint"])
    repointed, left = [], []
    for card_path in sorted(baseline_dir.glob("*.json")):
        if card_path.name == "active.json":
            continue
        card_doc = json.loads(card_path.read_text(encoding="utf-8"))
        if restored_key in card_doc.get("cards", {}):
            card_doc["active"] = restored_key
            card_doc["rolled_back_from"] = identity_of(doc["fingerprint"])
            card_path.write_text(json.dumps(card_doc, indent=1), encoding="utf-8")
            repointed.append(card_path.stem)
        else:
            left.append(card_path.stem)
    c = prev["fingerprint"]["components"]
    env = {"GRADER_MODEL": c.get("model.grader"), "GATE_CLOUD_MODEL": c.get("model.gate_cloud"), "GATE_LOCAL_MODEL": c.get("model.gate_local"), "GRADER_LOCAL_MODEL": c.get("model.grader_local")}
    return {"restored": prev["fingerprint"]["id"], "env": env, "git_head": c.get("git.head"), "cards_repointed": repointed, "cards_without_restored_id": left}
