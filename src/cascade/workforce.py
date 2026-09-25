"""Workforce impact model: image-review desk time the cascade frees, estimated from one run's measured counts.

What is measured (read from the run folder, never moved by a slider): image, gate, routing, finding and
review counts, API dollars and model seconds from `summary.json`, `gate.jsonl`, `findings.json` and
`calls.jsonl`, plus reviewer decision gaps from `reviews.sqlite`. Everything else (minutes per image, hourly
rate, audit rate) is an `Assumption` that carries its source, URL and kind. Every hour, percent or dollar
produced here is an estimate: measured counts multiplied by those assumptions, labelled as such.

`measured_inputs()` is the only reader; `estimate()` is pure so a UI can recompute it on every slider move.
Scope: image review desk time only. Field visits, arm's-length inspections, flights and re-flights are not
modeled and nothing here is a field-accuracy claim (docs/decisions.md D-011, D-012). U findings always carry
human time and are never counted as cleared or as S0.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from .pipeline import load_run
from .review import ReviewLog
from .schema import Finding, ImageRecord

Kind = Literal["public", "derived_public", "assumption", "measured"]

GAP_N_MIN = 10  # review gaps needed before gap_median x fpi may replace the m_rev assumption (spec 3.4)
UNASSIGNED = "unassigned"  # client bucket for images without a client_id; never dropped

DISCLAIMER = "This estimate covers image review desk time only, not the field visit, and is not a field-accuracy claim."
NEGATIVE_SENTENCE = (
    "At these settings the cascade adds review time; it changes what the reviewer gets "
    "(a graded, rubric-cited finding), not how long a skim takes."
)
FOOTER = (
    "Hours and dollars here are estimates: this run's measured counts multiplied by the assumptions above. "
    "Change an assumption and the estimate changes; the measured counts do not. Image review desk time only; "
    "field inspection, flights and re-flights are not modeled. Public datasets, not customer imagery: the routing "
    "fraction on your fleet must be measured on your images."
)
COST_SCOPE = (
    "Cascade cost per image is API list price plus estimated human time; local gate GPU time, storage, support "
    "and software price are excluded."
)

_TDW_URL = "https://www.tdworld.com/smart-utility/article/55242455/modernizing-utility-infrastructure-inspections-where-ai-meets-asset-management"
_TDW = (
    "T&D World 2024-11-13, Chuck Salvo (EchoStor) on a Northeast utility: 'Traditional manual image analysis "
    "required 3-5 minutes per image' (docs/research/08 section 2.1). Sector: utility transmission imagery; no "
    "public per-image figure exists for bridges, solar or wind."
)
_AEP_URL = "https://www.renewableenergyworld.com/power-grid/how-autonomous-drones-and-ai-are-reshaping-utility-inspection-programs/"
_AEP = (
    "AEP Ohio 2025 pilot, Renewable Energy World 2026-01-14: '500 plus hours' by one person over 400,000-500,000 "
    "images; 500 x 60 / 450,000 = 0.067 min/image (docs/research/08 section 2.1, 00 section 3). A triage skim, "
    "not a graded review; '500 plus' is a lower bound so the rate is at least 0.06."
)
_WSDOT_URL = "https://app.leg.wa.gov/ReportsToTheLegislature/Home/GetPDF?fileName=High+Cost+Bridge+Inspection+Cost+Comparison_18ee4d7a-0591-4fef-b707-e9bcd8dc9197.pdf"
_WSDOT = (
    "WSDOT high-cost bridge inspection rate comparison, undated legislative report: inspector $113.03/h "
    "(co-inspector $94.43, report writing $112, UBIT team all-in $542.41) (docs/research/04 section 1.2). "
    "One state's bridge rate, not the customer's."
)
_TEAM = "none; team planning default"


@dataclass(frozen=True)
class Assumption:
    """One slider-bound input with its provenance. `kind` maps to D-011: public = [Sourced], derived_public =
    [Inference], assumption = [Assumption], measured = [Team-measured] (only after `use_measured_m_rev`)."""

    name: str
    value: float
    unit: str
    kind: Kind
    source: str
    url: Optional[str]
    note: str


DEFAULTS: Dict[str, Assumption] = {
    "m_man": Assumption("m_man", 3.0, "min/image", "public", _TDW, _TDW_URL, "Minutes per image, manual review today. Low end of the quoted 3-5 range on purpose."),
    "m_skim": Assumption("m_skim", 0.067, "min/image", "derived_public", _AEP, _AEP_URL, "Skim-rate reference tick for the m_man slider (range 0.060-0.075); not a default and not a graded review."),
    "m_man_hi": Assumption("m_man_hi", 5.0, "min/image", "public", _TDW, _TDW_URL, "Analysis-rate high tick for the m_man slider; upper end of the quoted range."),
    "m_rev": Assumption("m_rev", 1.0, "min/routed image", "assumption", _TEAM, None, "Reviewer sees the evidence crop, the pre-filled native grade and the quoted rubric row, then accepts / overrides / marks U (FR-18). Replaced by gap_median x findings per graded image once gap_n >= 10."),
    "m_xf": Assumption("m_xf", 0.0, "min/finding", "assumption", _TEAM, None, "Minutes per additional finding on the same image; 0 folds multi-tile findings into m_rev."),
    "m_aud": Assumption("m_aud", 3.0, "min/image", "assumption", _TEAM, None, "Minutes per audited auto-cleared image; tracks m_man unless overridden (the auditor gives the image a full manual look)."),
    "a": Assumption("a", 0.10, "fraction", "assumption", _TEAM, None, "Random audit sample rate on auto-cleared images, independent of gate confidence."),
    "w": Assumption("w", 113.03, "USD/hour", "public", _WSDOT, _WSDOT_URL, "Loaded hourly rate; moves dollars only, never hours or percent."),
    "H": Assumption("H", 40.0, "hours/week", "assumption", "convention", None, "Hours per inspector-week; used only for inspector-equivalents of review time freed."),
    "gap_cap": Assumption("gap_cap", 10.0, "min", "assumption", _TEAM, None, "Review gaps longer than this are breaks and are dropped from the gap median."),
}

REFERENCES: Dict[str, Assumption] = {
    # shown beside the sliders, never used in a formula
    "scopito_expert_eur_per_image": Assumption("scopito_expert_eur_per_image", 1.0, "EUR/image", "public", "Scopito price list, expert image analysis EUR 1/image (docs/research/09 section 6).", "https://scopito.com/building-inspection-software/", "Cross-check only: at $113/h it implies about 0.5 min/image, so the m_man slider floor sits well below 3."),
}

_BADGE = {"public": "Public figure", "derived_public": "Derived from public figure", "assumption": "Team assumption"}
_ITALIC_KINDS = ("derived_public", "assumption")


def badge_text(kind: str, run: str = "") -> str:
    """Badge string for one provenance kind (spec 5): measured values name the run they came from."""
    return f"Measured (run {run})" if kind == "measured" else _BADGE[kind]


@dataclass
class WorkforceInputs:
    """Counts measured from one run folder (or one client bucket of it). Nothing here comes from a slider."""

    run: str
    images: int
    gated: int
    routed: int
    unusable: int
    graded_images: int
    findings: int
    u_findings: int
    usd_total: float
    seconds_gate: float
    seconds_grade: float
    reviews: int
    gap_median_min: Optional[float]
    gap_n: int
    grader: str
    gate: str
    source_datasets: List[str]
    client_id: Optional[str]
    complete: bool
    images_without_client: int = 0  # images in the run whose record has no client_id (bucket `unassigned`)


@dataclass
class WorkforceEstimate:
    """Spec 3.1 (this run) and 3.2 (per 1,000 images at this run's routing fraction). Always labelled estimate."""

    inputs: WorkforceInputs
    assumptions: Dict[str, Assumption]
    audit_n: int
    auto_cleared: int
    routing_fraction: Optional[float]
    m_rev_eff: float
    manual_min: float
    review_min: float
    extra_min: float
    audit_min: float
    cascade_min: float
    saved_min: float
    saved_pct: Optional[float]
    break_even_m_man: Optional[float]
    manual_h_1000: float
    cascade_h_1000: float
    saved_h_1000: float
    audit_1000: int
    ie_per_1000_week: float
    cost_img_human: float
    cost_img_cascade_api: float
    cost_img_cascade_hum: float
    cost_img_cascade: float
    model_s_per_image: Optional[float]
    saved_pct_1000: Optional[float] = None
    cost_img_cascade_1000: float = 0.0
    fpi: Optional[float] = None  # findings per graded image, measured
    m_rev_measured: Optional[float] = None  # gap_median x fpi when gap_n >= GAP_N_MIN, else None
    uses: Dict[str, List[str]] = field(default_factory=dict)  # output name -> assumption keys it multiplies
    label: str = "estimate"
    warnings: List[str] = field(default_factory=list)


# --- readers -------------------------------------------------------------------------------------------------


def _image_of(f: Finding) -> str:
    return f.evidence.image_ids[0] if f.evidence.image_ids else ""


def _bucket(image_id: str, records: Dict[str, ImageRecord]) -> str:
    rec = records.get(image_id)
    return rec.client_id if rec is not None and rec.client_id else UNASSIGNED


def _records_or_manifest(out: Path, records: Optional[Dict[str, ImageRecord]]) -> Dict[str, ImageRecord]:
    """Given records win; otherwise the run's own `manifest.jsonl` (written by the app) supplies client ids and datasets."""
    if records is not None:
        return records
    mp = out / "manifest.jsonl"
    if not mp.exists():
        return {}
    recs = [ImageRecord.model_validate(json.loads(line)) for line in mp.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r.image_id: r for r in recs}


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def review_gap_median(out: Path, run_id: str, gap_cap_min: float = 10.0) -> Tuple[Optional[float], int]:
    """Measured reviewer pace from `reviews.sqlite`: (median minutes between consecutive decisions by the same
    reviewer, number of gaps kept). Gaps longer than `gap_cap_min` are breaks and are dropped; negative gaps
    (clock skew) too. Returns (None, 0) when the log is absent or holds fewer than two decisions for `run_id`.
    The app logs decisions under the run folder name, so `run_id` is normally `out.name`."""
    db = Path(out) / "reviews.sqlite"
    if not db.exists():
        return None, 0
    log = ReviewLog(db)
    try:
        rows = log.timeline(run_id)
    finally:
        log.conn.close()
    by_reviewer: Dict[str, List[datetime]] = {}
    for row in rows:
        ts = _parse_ts(row["reviewed_at"])
        if ts is not None:
            by_reviewer.setdefault(row["reviewer"] or "", []).append(ts)
    gaps: List[float] = []
    for stamps in by_reviewer.values():
        for prev, cur in zip(stamps, stamps[1:]):
            minutes = (cur - prev).total_seconds() / 60.0
            if 0.0 <= minutes <= gap_cap_min:
                gaps.append(minutes)
    if not gaps:
        return None, 0
    return statistics.median(gaps), len(gaps)


def _inputs_from(out: Path, run: dict, records: Dict[str, ImageRecord], client_id: Optional[str], gaps: Tuple[Optional[float], int]) -> WorkforceInputs:
    gate_rows, findings, calls, summary = run["gate"], run["findings"], run["calls"], run["summary"]
    run_ids = {g["image_id"] for g in gate_rows}
    if client_id is not None:
        gate_rows = [g for g in gate_rows if _bucket(g["image_id"], records) == client_id]
        findings = [f for f in findings if _bucket(_image_of(f), records) == client_id]
        calls = [c for c in calls if _bucket(c.get("image_id", ""), records) == client_id]
        n_images = len(gate_rows)
        in_bucket = {i for i in run_ids if _bucket(i, records) == client_id}
    else:
        n_images = summary.get("images", len(gate_rows)) if summary else len(gate_rows)
        in_bucket = run_ids
    cfg = summary.get("config", {}) if summary else {}
    complete = all((out / name).exists() for name in ("summary.json", "gate.jsonl", "calls.jsonl")) and (
        (out / "findings.json").exists() or (out / "findings.jsonl").exists()
    )
    without_client = sum(1 for i in run_ids if _bucket(i, records) == UNASSIGNED)
    return WorkforceInputs(
        run=out.name,
        images=n_images,
        gated=len(gate_rows),
        routed=sum(1 for g in gate_rows if g["routed"]),
        unusable=sum(1 for g in gate_rows if not g["usable"]),
        graded_images=len({_image_of(f) for f in findings if f.evidence.image_ids}),
        findings=len(findings),
        u_findings=sum(1 for f in findings if f.unified.level == "U"),
        usd_total=float(sum(c["usd"] for c in calls)),
        seconds_gate=float(sum(c["seconds"] for c in calls if c["stage"] == "gate")),
        seconds_grade=float(sum(c["seconds"] for c in calls if c["stage"] == "grade")),
        reviews=sum(1 for f in findings if f.review.status != "pending"),
        gap_median_min=gaps[0],
        gap_n=gaps[1],
        grader=cfg.get("grader", ""),
        gate=cfg.get("gate", ""),
        source_datasets=sorted({records[i].source_dataset for i in in_bucket if i in records and records[i].source_dataset}),
        client_id=client_id,
        complete=complete,
        images_without_client=without_client if client_id in (None, UNASSIGNED) else 0,
    )


def measured_inputs(out: Path, records: Optional[Dict[str, ImageRecord]] = None, client_id: Optional[str] = None, gap_cap_min: float = DEFAULTS["gap_cap"].value) -> WorkforceInputs:
    """Read one run folder (spec section 1). Missing files count as zero and set `complete=False`; no exception.
    `client_id` restricts every count and the API dollars to that bucket (`unassigned` = images without a
    client id); the run-level image count comes from `summary.json`, a bucket's from its gate rows."""
    out = Path(out)
    return _inputs_from(out, load_run(out), _records_or_manifest(out, records), client_id, review_gap_median(out, out.name, gap_cap_min))


# --- the pure model ------------------------------------------------------------------------------------------


def _ceil(x: float) -> int:
    return int(math.ceil(round(x, 9)))  # 0.1 * 30 = 3.0000000000000004 must not audit a 4th image


def _uniq(*groups: List[str]) -> List[str]:
    seen: List[str] = []
    for g in groups:
        for k in g:
            if k not in seen:
                seen.append(k)
    return seen


def estimate(inputs: WorkforceInputs, overrides: Optional[Dict[str, float]] = None) -> WorkforceEstimate:
    """Spec 3.1 and 3.2 from measured inputs and assumptions. Pure: no I/O, safe to call on every slider move.

    `overrides` replaces DEFAULTS values by key (m_man, m_rev, m_xf, m_aud, a, w, H, gap_cap, m_skim, m_man_hi).
    Two extra keys: `use_measured_m_rev` (truthy) swaps m_rev for gap_median x findings-per-image when
    gap_n >= GAP_N_MIN (an explicit `m_rev` override still wins), and `r` (0..1) is a hypothetical routing
    fraction that moves the per-1,000 figures only; the run's counts are measured and never move.
    Savings may be negative and are never clamped."""
    ov = dict(overrides or {})
    use_measured = bool(ov.pop("use_measured_m_rev", False))
    r_whatif = ov.pop("r", None)
    unknown = sorted(set(ov) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown assumption keys: {unknown}")
    assumptions = {k: (replace(v, value=float(ov[k])) if k in ov else v) for k, v in DEFAULTS.items()}
    m_man = assumptions["m_man"].value
    if "m_aud" not in ov:  # tracks m_man unless the advanced override is set
        assumptions["m_aud"] = replace(assumptions["m_aud"], value=m_man)
    for k, asm in assumptions.items():
        if asm.value < 0 or (k == "a" and asm.value > 1) or (k == "H" and asm.value <= 0):
            raise ValueError(f"assumption {k} out of range: {asm.value}")
    if r_whatif is not None and not 0.0 <= float(r_whatif) <= 1.0:
        raise ValueError(f"routing what-if out of range: {r_whatif}")

    warnings: List[str] = []
    N, G, R, GI, F = inputs.images, inputs.gated, inputs.routed, inputs.graded_images, inputs.findings
    A = G - R
    fpi = (F / GI) if GI > 0 else None
    m_rev_measured = (inputs.gap_median_min * fpi) if (inputs.gap_n >= GAP_N_MIN and inputs.gap_median_min is not None and fpi is not None) else None
    if use_measured and "m_rev" not in ov:
        if m_rev_measured is None:
            warnings.append(f"not enough decisions to measure minutes per routed image ({inputs.gap_n} of {GAP_N_MIN} review gaps needed); m_rev stays a team assumption")
        else:
            assumptions["m_rev"] = Assumption(
                "m_rev", m_rev_measured, "min/routed image", "measured",
                f"reviews.sqlite (run {inputs.run}): median {inputs.gap_median_min:.2f} min between consecutive decisions by the same reviewer x {fpi:.2f} findings per graded image, n = {inputs.gap_n} gaps",
                None, "Indicative: gaps include thinking, multitasking and UI time.",
            )
    m_rev, m_xf, m_aud, a, w, H = (assumptions[k].value for k in ("m_rev", "m_xf", "m_aud", "a", "w", "H"))

    fallback = not (GI > 0 and inputs.grader != "none")
    m_rev_eff = m_man if fallback else m_rev
    if fallback:
        warnings.append("run has no graded findings; routed images costed at manual rate (m_rev_eff = m_man)")

    audit_n = 0 if (a == 0 or A <= 0) else max(1, _ceil(a * A))
    manual_min = N * m_man
    review_min = R * m_rev_eff
    extra_findings = max(0, F - GI)
    extra_min = extra_findings * m_xf
    audit_min = audit_n * m_aud
    cascade_min = review_min + extra_min + audit_min
    saved_min = manual_min - cascade_min
    saved_pct = (100.0 * saved_min / manual_min) if manual_min > 0 else None
    break_even = (R * m_rev_eff / (N - audit_n)) if (N - audit_n) > 0 and not fallback else None

    cost_img_human = (m_man / 60.0) * w
    cost_img_cascade_api = (inputs.usd_total / N) if N else 0.0
    cost_img_cascade_hum = ((cascade_min / 60.0) * w / N) if N else 0.0
    model_s = ((inputs.seconds_gate + inputs.seconds_grade) / N) if N else None

    r_meas = (R / G) if G > 0 else None
    r_used = float(r_whatif) if r_whatif is not None else r_meas
    if r_used is None:
        R_1000 = A_1000 = 0.0
        audit_1000 = 0
        manual_h_1000 = cascade_h_1000 = saved_h_1000 = 0.0
        saved_pct_1000 = None
        warnings.append("no gated images; per-1,000 figures are zero")
    else:
        R_1000 = 1000.0 * r_used
        A_1000 = 1000.0 - R_1000
        audit_1000 = max(1, _ceil(a * A_1000)) if (a > 0 and A_1000 > 0) else 0
        manual_h_1000 = 1000.0 * m_man / 60.0
        extra_1000 = (1000.0 * extra_findings / N * m_xf) if N else 0.0
        cascade_h_1000 = (R_1000 * m_rev_eff + extra_1000 + audit_1000 * m_aud) / 60.0
        saved_h_1000 = manual_h_1000 - cascade_h_1000
        saved_pct_1000 = (100.0 * saved_h_1000 / manual_h_1000) if manual_h_1000 > 0 else None
    ie = saved_h_1000 / H
    cost_1000 = cost_img_cascade_api + cascade_h_1000 * w / 1000.0

    if N == 0:
        warnings.append("run has no images; all estimates are zero")
    if saved_pct is not None and saved_pct < 0:
        warnings.append(NEGATIVE_SENTENCE)
    if audit_n and audit_n > a * A + 1e-9:
        warnings.append(f"audit sample rounded up to {audit_n} of {A} auto-cleared images (a x A = {a * A:.2f}); at 1,000 images the same settings audit {audit_1000} of {A_1000:.0f}")
    if inputs.gap_n < GAP_N_MIN:
        warnings.append(f"minutes per routed image is a team assumption; {inputs.gap_n} of {GAP_N_MIN} review gaps logged")
    if inputs.source_datasets and r_meas is not None:
        warnings.append(f"images come from dataset(s) {', '.join(inputs.source_datasets)}, not customer imagery: routing fraction {r_meas:.2f} is damage-enriched and is not a fleet number")
    if r_whatif is not None:
        warnings.append(f"routing fraction {r_used:.2f} is a hypothetical what-if (measured: {'n/a' if r_meas is None else f'{r_meas:.2f}'}); it moves the per-1,000 figures only")
    if not inputs.complete:
        warnings.append("run folder is incomplete (missing files); counts may be partial")
    if inputs.images_without_client:
        warnings.append(f"{inputs.images_without_client} images without a client id (bucket '{UNASSIGNED}')")

    rev_key = ["m_man"] if fallback else ["m_rev"]
    cascade_keys = _uniq(rev_key, ["m_xf"] if m_xf else [], ["a", "m_aud"])
    uses = {
        "audit_n": ["a"], "audit_1000": ["a"],
        "manual_min": ["m_man"], "review_min": rev_key, "extra_min": ["m_xf"], "audit_min": ["a", "m_aud"],
        "cascade_min": cascade_keys, "saved_min": _uniq(["m_man"], cascade_keys), "saved_pct": _uniq(["m_man"], cascade_keys),
        "break_even_m_man": ["m_rev", "a"],
        "manual_h_1000": ["m_man"], "cascade_h_1000": cascade_keys, "saved_h_1000": _uniq(["m_man"], cascade_keys),
        "saved_pct_1000": _uniq(["m_man"], cascade_keys), "ie_per_1000_week": _uniq(["m_man"], cascade_keys, ["H"]),
        "cost_img_human": ["m_man", "w"], "cost_img_cascade_api": [], "cost_img_cascade_hum": _uniq(cascade_keys, ["w"]),
        "cost_img_cascade": _uniq(cascade_keys, ["w"]), "cost_img_cascade_1000": _uniq(cascade_keys, ["w"]), "model_s_per_image": [],
    }
    return WorkforceEstimate(
        inputs=inputs, assumptions=assumptions, audit_n=audit_n, auto_cleared=A, routing_fraction=r_used, m_rev_eff=m_rev_eff,
        manual_min=manual_min, review_min=review_min, extra_min=extra_min, audit_min=audit_min, cascade_min=cascade_min,
        saved_min=saved_min, saved_pct=saved_pct, break_even_m_man=break_even,
        manual_h_1000=manual_h_1000, cascade_h_1000=cascade_h_1000, saved_h_1000=saved_h_1000, audit_1000=audit_1000, ie_per_1000_week=ie,
        cost_img_human=cost_img_human, cost_img_cascade_api=cost_img_cascade_api, cost_img_cascade_hum=cost_img_cascade_hum,
        cost_img_cascade=cost_img_cascade_api + cost_img_cascade_hum, model_s_per_image=model_s,
        saved_pct_1000=saved_pct_1000, cost_img_cascade_1000=cost_1000, fpi=fpi, m_rev_measured=m_rev_measured, uses=uses, warnings=warnings,
    )


def estimate_by_client(out: Path, records: Dict[str, ImageRecord], overrides: Optional[Dict[str, float]] = None) -> Dict[str, WorkforceEstimate]:
    """One estimate per `client_id` found in the run (spec 3.3), `unassigned` last when any image lacks one.
    Every count and the API dollars are attributed by image_id, so the buckets sum to the run totals."""
    out = Path(out)
    run = load_run(out)
    ids = {g["image_id"] for g in run["gate"]} | {_image_of(f) for f in run["findings"]} | {c.get("image_id", "") for c in run["calls"]}
    buckets = sorted({_bucket(i, records) for i in ids} - {UNASSIGNED})
    if any(_bucket(i, records) == UNASSIGNED for i in ids):
        buckets.append(UNASSIGNED)
    gap_cap = float((overrides or {}).get("gap_cap", DEFAULTS["gap_cap"].value))
    gaps = review_gap_median(out, out.name, gap_cap)
    return {b: estimate(_inputs_from(out, run, records, b, gaps), overrides) for b in buckets}


def rule_of_three_bound(audited: int) -> Optional[float]:
    """95% upper bound on the miss rate among auto-cleared images if a random audit of `audited` finds no miss:
    about 3 / n (Hanley and Lippman-Hand, JAMA 1983, 249(13):1743-5). Valid for the audited population of one
    run only, after the audit is done and logged; not a field-accuracy claim. Cite in docs/ before it goes on a slide."""
    return (3.0 / audited) if audited > 0 else None


def tornado(inputs: WorkforceInputs, overrides: Optional[Dict[str, float]] = None) -> List[dict]:
    """One-at-a-time sensitivity rows for a horizontal bar chart (spec 4): lever, value, label, hypothetical,
    saved_pct (this run; None for hypothetical routing rows because the run's counts are measured),
    saved_pct_1000 and ie_per_1000_week (per 1,000 images at that routing fraction)."""
    base = dict(overrides or {})
    r_meas = (inputs.routed / inputs.gated) if inputs.gated else None
    levers: List[Tuple[str, List[Tuple[Optional[float], str]]]] = [
        ("m_man", [(DEFAULTS["m_skim"].value, "AEP Ohio skim (derived)"), (DEFAULTS["m_man"].value, "T&D World analysis, low"), (DEFAULTS["m_man_hi"].value, "T&D World analysis, high")]),
        ("m_rev", [(0.5, "assumption"), (1.0, "assumption (default)"), (2.0, "assumption")]),
        ("a", [(0.0, "no audit"), (0.1, "assumption (default)"), (0.3, "assumption")]),
        ("w", [(94.43, "WSDOT co-inspector"), (113.03, "WSDOT inspector"), (250.0, "high tick (assumption)")]),
        ("r", [(r_meas, "measured"), (0.5, "hypothetical cleaner fleet"), (0.3, "hypothetical cleaner fleet")]),
    ]
    rows: List[dict] = []
    for lever, values in levers:
        for value, label in values:
            ov = dict(base)
            hypothetical = lever == "r" and label != "measured"
            if lever == "r":
                ov.pop("r", None)
                if hypothetical:
                    ov["r"] = value
            else:
                ov[lever] = value
                if lever == "m_rev":
                    ov.pop("use_measured_m_rev", None)
            est = estimate(inputs, ov)
            rows.append({"lever": lever, "value": value, "label": label, "hypothetical": hypothetical,
                         "saved_pct": None if hypothetical else est.saved_pct, "saved_pct_1000": est.saved_pct_1000, "ie_per_1000_week": est.ie_per_1000_week})
    return rows


# --- rendering -----------------------------------------------------------------------------------------------


def _num(x: Optional[float], nd: int = 1, prefix: str = "", suffix: str = "", signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{prefix}{x:{'+' if signed else ''}.{nd}f}{suffix}"


def _est_label(est: WorkforceEstimate, key: str) -> str:
    keys = [k for k in est.uses.get(key, []) if est.assumptions[k].kind != "measured"]
    if not keys:
        return badge_text("measured", est.inputs.run)
    return f"*estimate (uses {len(keys)} assumption{'s' if len(keys) != 1 else ''}: {', '.join(keys)})*"


def _est_row(est: WorkforceEstimate, title: str, key: str, text: str) -> str:
    approx = "" if _est_label(est, key).startswith("Measured") else "≈ "
    return f"| {title} | {approx}{text} | {_est_label(est, key)} |"


def render_markdown(est: WorkforceEstimate) -> str:
    """Section "## Workforce estimate (assumptions listed)" for report.md (spec 5.8). Measured counts print plain
    with a Measured badge; every number that multiplies an assumption prints as `≈ value` with the italic
    `estimate (uses k assumptions)` suffix; the disclaimer, audit caption and footer are fixed sentences."""
    i = est.inputs
    run = badge_text("measured", i.run)
    r = est.routing_fraction
    meas = [
        ("Images (N)", str(i.images), "summary.json"), ("Gated (G)", str(i.gated), "gate.jsonl"),
        ("Routed to grader (R)", str(i.routed), "gate.jsonl"), ("Auto-cleared (A = G - R)", str(est.auto_cleared), "gate.jsonl"),
        ("Unusable (UN)", str(i.unusable), "gate.jsonl"), ("Graded images (GI)", str(i.graded_images), "findings.json"),
        ("Findings (F)", str(i.findings), "findings.json"), ("U findings (never counted as S0)", str(i.u_findings), "findings.json"),
        ("API dollars, all stages (C)", _num(i.usd_total, 4, "$"), "calls.jsonl"), ("Model seconds per image", _num(est.model_s_per_image, 1, suffix=" s"), "calls.jsonl"),
        ("Reviewer decisions (V)", str(i.reviews), "findings.json"), ("Review gap median / gaps kept", f"{_num(i.gap_median_min, 2, suffix=' min')} / {i.gap_n}", "reviews.sqlite"),
        ("Routing fraction (r = R / G)", _num((i.routed / i.gated) if i.gated else None, 2), "gate.jsonl"),
    ]
    lines: List[str] = [
        "## Workforce estimate (assumptions listed)", "",
        f"*{DISCLAIMER}*", "",
        f"Run `{i.run}`" + (f", client `{i.client_id}`" if i.client_id else "") + ". Two columns: counts measured from this run, then figures estimated with the assumptions listed.", "",
        "### Measured from this run", "",
        "| Quantity | Value | Badge | Source file |", "|---|---|---|---|",
    ]
    lines += [f"| {t} | {v} | {run} | {src} |" for t, v, src in meas]
    lines += ["", "### Assumptions used", "", "| Assumption | Value | Badge | Source |", "|---|---|---|---|"]
    for asm in est.assumptions.values():
        b = badge_text(asm.kind, i.run)
        b = f"*{b}*" if asm.kind in _ITALIC_KINDS else b
        src = asm.source + (f" {asm.url}" if asm.url else "")
        lines.append(f"| {asm.note.split('.')[0]} ({asm.name}) | {asm.value:g} {asm.unit} | {b} | {src} |")
    r_txt = _num(r, 2)
    lines += ["", "### Estimated with the assumptions above", "", "| Output | Value | Label |", "|---|---|---|",
        _est_row(est, "Manual review time, this run", "manual_min", _num(est.manual_min, 1, suffix=" min")),
        _est_row(est, "Cascade review time, this run (routed + extra findings + audit)", "cascade_min", f"{_num(est.cascade_min, 1, suffix=' min')} = {_num(est.review_min, 1)} + {_num(est.extra_min, 1)} + {_num(est.audit_min, 1)}"),
        _est_row(est, "Review time saved, this run", "saved_min", f"{_num(est.saved_min, 1, suffix=' min', signed=True)} ({_num(est.saved_pct, 1, suffix='%', signed=True)})"),
        _est_row(est, "Break-even manual minutes per image (cascade saves time above this)", "break_even_m_man", _num(est.break_even_m_man, 2, suffix=" min/image")),
        _est_row(est, f"Per 1,000 images at this run's routing fraction (r = {r_txt}): manual", "manual_h_1000", _num(est.manual_h_1000, 1, suffix=" h")),
        _est_row(est, f"Per 1,000 images at r = {r_txt}: cascade", "cascade_h_1000", _num(est.cascade_h_1000, 1, suffix=" h")),
        _est_row(est, f"Per 1,000 images at r = {r_txt}: saved", "saved_h_1000", f"{_num(est.saved_h_1000, 1, suffix=' h', signed=True)} ({_num(est.saved_pct_1000, 1, suffix='%', signed=True)})"),
        _est_row(est, "Inspector-equivalents of review time freed per 1,000 images per week", "ie_per_1000_week", _num(est.ie_per_1000_week, 2, signed=True)),
        _est_row(est, "Cost per image, human review", "cost_img_human", _num(est.cost_img_human, 2, "$")),
        _est_row(est, "Cost per image, cascade API (list price)", "cost_img_cascade_api", _num(est.cost_img_cascade_api, 3, "$")),
        _est_row(est, "Cost per image, cascade human time", "cost_img_cascade_hum", _num(est.cost_img_cascade_hum, 2, "$")),
        _est_row(est, "Cost per image, cascade total (API part measured)", "cost_img_cascade", f"{_num(est.cost_img_cascade, 2, '$')} (per 1,000 scale {_num(est.cost_img_cascade_1000, 2, '$')})"),
        "",
        f"Run figure audits {est.audit_n} of {est.auto_cleared} auto-cleared images (rounded up to at least 1); at 1,000 images the same settings audit {est.audit_1000} of {int(round((1 - r) * 1000)) if r is not None else 0}.",
    ]
    if est.saved_pct is not None and est.saved_pct < 0:
        lines += ["", f"**{NEGATIVE_SENTENCE}**"]
    if est.warnings:
        lines += ["", "Warnings:", ""] + [f"- {wtxt}" for wtxt in est.warnings]
    lines += ["", COST_SCOPE, "", FOOTER, ""]
    return "\n".join(lines)


def to_json(est: WorkforceEstimate) -> dict:
    """The `"workforce"` block for report.json (spec 5.8): measured inputs, assumptions with kind and source,
    outputs, `label: "estimate"` and warnings."""
    skip = {"inputs", "assumptions", "uses", "label", "warnings"}
    outputs = {k: v for k, v in asdict(est).items() if k not in skip}
    return {
        "inputs_measured": asdict(est.inputs),
        "assumptions": [asdict(a) for a in est.assumptions.values()],
        "outputs": outputs,
        "uses": est.uses,
        "label": est.label,
        "warnings": list(est.warnings),
        "disclaimer": DISCLAIMER,
        "footer": FOOTER,
        "cost_scope": COST_SCOPE,
    }


__all__ = [
    "Assumption", "DEFAULTS", "REFERENCES", "WorkforceInputs", "WorkforceEstimate", "GAP_N_MIN", "UNASSIGNED",
    "DISCLAIMER", "NEGATIVE_SENTENCE", "FOOTER", "COST_SCOPE", "badge_text", "measured_inputs", "review_gap_median",
    "estimate", "estimate_by_client", "rule_of_three_bound", "tornado", "render_markdown", "to_json",
]
