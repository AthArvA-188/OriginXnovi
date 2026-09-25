"""Per-client inspection reports: one self-contained HTML per client per run, a markdown twin and `report.json`.

Every number is measured from the run's own files (`manifest.jsonl`, `gate.jsonl`, `calls.jsonl`,
`findings.json`, `reviews.sqlite`) or labelled as an assumption with its source. U is its own count and is
never folded into S0. This module never opens `data/eval_v1` and `ClientMetrics` carries no accuracy field;
accuracy lives in `eval/reports/<run>.md` with n and CI, on labelled data only.

Output tree: `runs/<run>/clients/index.json` and `clients/<slug>/{report.html, report.md, report.json, evidence/*.jpg}`.
The HTML has inline CSS, inline SVG charts drawn here, base64 JPEG thumbnails with the bbox drawn, no
`<script>`, no `<link>`, no remote URL, and is kept under `ThumbBudget.max_bytes`.

Usage:
  python -m cascade.clientreport --run runs/ui_0925_0856 [--manifest data/dev/manifest.jsonl] [--today 2026-09-25] [--max-thumbs 60]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .costlog import PRICES_PER_MTOK
from .export import finding_row
from .grade import RUBRIC_FOR_CLASS, load_rubric
from .ingest import read_manifest
from .pipeline import LEVELS, load_run
from .prioritize import SEVERITY_WEIGHT
from .review import ReviewLog
from .schema import Finding, ImageRecord

# ---------- constants (same values as app/streamlit_app.py so the report and the tab share one palette) ----------

LEVEL_COLOR = {"S0": "#7a7a7a", "S1": "#3b82f6", "S2": "#f59e0b", "S3": "#f97316", "S4": "#dc2626", "U": "#8b5cf6"}
LEVEL_MEANING = {
    "S0": "no defect found in the graded region",
    "S1": "minor; monitor at the next routine cycle",
    "S2": "moderate; schedule within the SLA",
    "S3": "severe; prioritize this cycle",
    "S4": "critical; same-day escalation, top of queue",
    "U": "unassessable: refusal, missing metadata or unusable image; never counted as S0",
}
ACTION_ORDER = ("escalate", "prioritize", "schedule", "monitor", "record")
ACTION_COLOR = {"escalate": "#dc2626", "prioritize": "#f97316", "schedule": "#f59e0b", "monitor": "#3b82f6", "record": "#7a7a7a"}
REVIEW_COLOR = {"accepted": "#22c55e", "overridden": "#0ea5e9", "marked_u": "#8b5cf6", "pending": "#9ca3af"}
DEMO_LICENSES = {"dacl10k": "CC BY-NC 4.0", "rescuenet": "CC BY-NC-ND 4.0"}
BANNER = (
    "AI-assisted, human-certified decision support (D-012). Grades are pre-filled proposals with the rubric row quoted; "
    "the inspector of record accepts or overrides each one. This is not a field inspection."
)
DEMO_LINE = "Imagery is a public demo dataset under a non-commercial license; demo only."
U_SENTENCE = "U rows are never scored and never counted as S0."
U_FOOTNOTE = "U is unassessable, listed separately, never counted as S0"
NEXT_STEP_RULES = (
    (r"gsd|scale|width|mm per px", "supply GSD or re-image with a scale reference"),
    (r"unusable|blur|glare|occlu", "re-image"),
    (r"refusal|no contract|parse", "re-run or grade manually"),
    (r"not applicable|substrate|scope", "assign the correct asset class and re-run"),
)
MEASURE_KEYS = ("area_cm2", "crack_width_mm", "delta_t_k", "percent_area_rusted", "section_loss_pct")
SECTIONS = [
    ("cover", "Cover"), ("summary", "Executive summary"), ("levels", "Level distribution"), ("assets", "Assets"),
    ("findings", "Findings"), ("u-list", "U list"), ("sla", "SLA calendar"), ("method", "Methodology"),
]
_ORDER = {lvl: i for i, lvl in enumerate(LEVELS)}


# ---------- grouping ----------


def client_of(rec: ImageRecord) -> str:
    """Client key of a manifest row: `client_id`, else `source_dataset`, else "unassigned"."""
    return rec.client_id or rec.source_dataset or "unassigned"


def _base_slug(client_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", client_id).lower()[:64] or "unassigned"


def client_slug(client_id: str, others: Iterable[str] = ()) -> str:
    """Folder-safe slug ([A-Za-z0-9._-], lowercase, max 64). When a distinct id in `others` slugs to the same
    text, `-` + sha1(client_id)[:6] is appended so two clients never share a folder."""
    s = _base_slug(client_id)
    if any(o != client_id and _base_slug(o) == s for o in others):
        s = f"{s[:57]}-{hashlib.sha1(client_id.encode('utf-8')).hexdigest()[:6]}"
    return s


def finding_slug(finding_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", finding_id.replace("/", "__"))


def group_records(records) -> Dict[str, List[ImageRecord]]:
    """client key -> manifest rows, in manifest order. Accepts an iterable or an image_id mapping."""
    recs = records.values() if isinstance(records, Mapping) else records
    out: Dict[str, List[ImageRecord]] = {}
    for r in recs:
        out.setdefault(client_of(r), []).append(r)
    return out


# ---------- metrics ----------


def worst_level(levels: Iterable[str]) -> Optional[str]:
    """Worst unified level of a group: any graded level beats U, an only-U group shows U, empty gives None."""
    worst: Optional[str] = None
    for lvl in levels:
        if worst is None or (lvl != "U" and (worst == "U" or _ORDER[lvl] > _ORDER[worst])):
            worst = lvl
    return worst


def worst_levels(findings: Iterable[Finding]) -> Dict[str, str]:
    """image_id -> worst level over its findings, the rule the app's gallery badges use."""
    by_img: Dict[str, List[str]] = {}
    for f in findings:
        for iid in f.evidence.image_ids:
            by_img.setdefault(iid, []).append(f.unified.level)
    return {iid: worst_level(lv) or "U" for iid, lv in by_img.items()}


def next_step_for(reason: str) -> str:
    """Keyword class over the model's reason text (labelled as such in the report); "human review" otherwise."""
    text = reason.lower()
    for pattern, step in NEXT_STEP_RULES:
        if re.search(pattern, text):
            return step
    return "human review"


@dataclass(frozen=True)
class WorkloadAssumption:
    """Caller-supplied first-pass triage minutes per image; `source` is printed beside the estimate. No default ships."""

    minutes_per_image: float
    source: str


def _image_of(f: Finding) -> str:
    return f.evidence.image_ids[0] if f.evidence.image_ids else ""


def _parse_date(s: Optional[str]) -> Optional[date]:
    try:
        return datetime.fromisoformat(s).date() if s else None
    except ValueError:
        return None


def _gate_verdict(g: Optional[dict]) -> str:
    if g is None:
        return "not gated"
    if not g["usable"]:
        return "unusable"
    return "routed" if g["routed"] else "cleared"


def client_metrics(run_dir: Path, client_id: str, records: Mapping[str, ImageRecord], *, run: Optional[dict] = None,
                   today: Optional[date] = None, workload: Optional[WorkloadAssumption] = None) -> dict:
    """ClientMetrics for one client, measured from the run folder and the manifest rows it was given.

    Measured: counts, levels (U on its own), cost from `calls.jsonl` at `costlog.PRICES_PER_MTOK` list prices,
    model seconds, review decisions. Estimated: SLA due-in days when anchored on the run date instead of a
    capture date. Assumption: the workload minutes line, present only with a `WorkloadAssumption` and its source.
    """
    run_dir = Path(run_dir)
    run = run if run is not None else load_run(run_dir)
    recs = [r for r in records.values() if client_of(r) == client_id]
    ids = {r.image_id for r in recs}
    known = set(records)
    basis = "client_id" if recs and recs[0].client_id else ("source_dataset" if recs and recs[0].source_dataset else "unassigned")
    fs = [f for f in run["findings"] if _image_of(f) in ids or (client_id == "unassigned" and _image_of(f) not in known)]
    fs.sort(key=lambda f: (f.queue_rank if f.queue_rank is not None else 10**9, f.finding_id))
    gate = [g for g in run["gate"] if g["image_id"] in ids]
    gate_by = {g["image_id"]: g for g in gate}
    calls = [c for c in run["calls"] if c["image_id"] in ids]
    gen = datetime.now(timezone.utc)
    today = today or gen.date()
    cfg = (run.get("summary") or {}).get("config", {})
    rec_by = {r.image_id: r for r in recs}
    levels = {lvl: sum(1 for f in fs if f.unified.level == lvl) for lvl in LEVELS}

    if calls or (run_dir / "calls.jsonl").exists():
        cost_source = "calls.jsonl"
        usd_gate = sum(c["usd"] for c in calls if c["stage"] == "gate")
        usd_grade = sum(c["usd"] for c in calls if c["stage"] == "grade")
        sec_gate = sum(c["seconds"] for c in calls if c["stage"] == "gate")
        sec_grade = sum(c["seconds"] for c in calls if c["stage"] == "grade")
        models = {"gate": sorted({c["model"] for c in calls if c["stage"] == "gate"}), "grade": sorted({c["model"] for c in calls if c["stage"] == "grade"})}
    else:
        cost_source = "gate+findings"
        usd_gate, usd_grade = sum(g["usd"] for g in gate), sum(f.usd for f in fs)
        sec_gate, sec_grade = sum(g["seconds"] for g in gate), sum(f.seconds for f in fs)
        models = {"gate": sorted({g["model"] for g in gate if g["model"]}), "grade": sorted({f.model for f in fs if f.model})}

    status = Counter(f.review.status for f in fs)
    decided = len(fs) - status.get("pending", 0)
    timeline: List[dict] = []
    db = run_dir / "reviews.sqlite"
    if db.exists() and fs:
        fids = {f.finding_id for f in fs}
        log = ReviewLog(db)
        try:
            rows = [r for r in log.timeline() if r["finding_id"] in fids]
        finally:
            log.conn.close()
        acc = 0
        for n, r in enumerate(rows, start=1):
            acc += int(r["action"] == "accepted")
            timeline.append({**r, "n": n, "agreement_rate": acc / n})
    reviews = {"total": decided, "accepted": status.get("accepted", 0), "overridden": status.get("overridden", 0),
               "marked_u": status.get("marked_u", 0), "pending": status.get("pending", 0),
               "agreement_rate": (status.get("accepted", 0) / decided) if decided else None, "decisions": len(timeline), "timeline": timeline}

    first_call = min((c["ts"] for c in run["calls"]), default=None)
    run_date = _parse_date(first_call)

    def asset_key(rec: Optional[ImageRecord]) -> str:
        return (rec.asset_id or rec.image_id) if rec else "unknown image"

    def anchor_for(f: Finding) -> Tuple[date, str]:
        rec = rec_by.get(_image_of(f))
        cap = _parse_date(rec.captured_on) if rec else None
        if cap:
            return cap, "capture date"
        return (run_date, "run date") if run_date else (gen.date(), "report date")

    sla, queue, u_rows = [], [], []
    for f in fs:
        rec = rec_by.get(_image_of(f))
        anchor, anchor_basis = anchor_for(f)
        due = anchor + timedelta(days=f.action.sla_days) if f.action.sla_days is not None else None
        native = f"{f.native_scale.value} on {f.native_scale.standard}"
        sla.append({"rank": f.queue_rank, "finding_id": f.finding_id, "asset": asset_key(rec), "image_id": _image_of(f), "level": f.unified.level,
                    "native": native, "action": f.action.code, "sla_days": f.action.sla_days, "anchor_date": anchor.isoformat(),
                    "anchor_basis": anchor_basis, "due_on": due.isoformat() if due else None, "due_in_days": (due - today).days if due else None})
        row = finding_row(f)
        row.update({"level": f.unified.level, "native": native, "asset": asset_key(rec), "image_size": f"{rec.width}x{rec.height}" if rec else None,
                    "source_video": rec.source_video if rec else None, "frame_time_s": rec.frame_time_s if rec else None,
                    "captured_on": rec.captured_on if rec else None, "criteria_list": list(f.native_scale.criteria_matched),
                    "measurements": {k: getattr(f.measurements, k) for k in MEASURE_KEYS if getattr(f.measurements, k) is not None},
                    "reviewed_at": f.review.reviewed_at, "thumb": None, "image_on_disk": bool(rec and Path(rec.path).exists())})
        queue.append(row)
        if f.unified.level == "U":
            u_rows.append({"rank": f.queue_rank, "image_id": _image_of(f), "finding_id": f.finding_id, "reason": f.justification, "basis": f.action.basis,
                           "action": f.action.code, "sla_days": f.action.sla_days, "next_step": next_step_for(f.justification + " " + f.action.basis)})

    groups: Dict[str, dict] = {}
    for r in recs:
        a = groups.setdefault(asset_key(r), {"asset_id": asset_key(r), "basis": "asset" if r.asset_id else "image", "image_ids": [], "findings": [], "video": None})
        a["image_ids"].append(r.image_id)
        if r.source_video and a["video"] is None:
            a["video"] = f"video {r.source_video} @ {r.frame_time_s} s"
    for f in fs:
        k = asset_key(rec_by.get(_image_of(f)))
        groups.setdefault(k, {"asset_id": k, "basis": "image", "image_ids": [], "findings": [], "video": None})["findings"].append(f)
    assets = []
    for k, a in groups.items():
        afs: List[Finding] = a["findings"]
        wl = worst_level(f.unified.level for f in afs)
        top = next((f for f in afs if f.unified.level == wl), None)
        slas = [f.action.sla_days for f in afs if f.action.sla_days is not None]
        ranks = [f.queue_rank for f in afs if f.queue_rank is not None]
        caption = ""
        if not afs:
            verdicts = [_gate_verdict(gate_by.get(i)) for i in a["image_ids"]]
            caption = "not routed: " + "; ".join(f"{i} {v}" for i, v in zip(a["image_ids"], verdicts))
        assets.append({"asset_id": k, "basis": a["basis"], "images": len(a["image_ids"]), "findings": len(afs), "worst_level": wl or "-",
                       "worst_native": f"{top.native_scale.value} on {top.native_scale.standard}" if top else "",
                       "top_action": min((f.action.code for f in afs), key=ACTION_ORDER.index) if afs else "",
                       "min_sla_days": min(slas) if slas else None, "u_count": sum(1 for f in afs if f.unified.level == "U"),
                       "pending_reviews": sum(1 for f in afs if f.review.status == "pending"), "best_rank": min(ranks) if ranks else None,
                       "video_frames": sum(1 for i in a["image_ids"] if rec_by[i].source_video), "source": a["video"] or "still", "gate_caption": caption})
    assets.sort(key=lambda a: (a["best_rank"] is None, a["best_rank"] or 0, a["asset_id"]))

    classes = sorted({f.asset_class for f in fs} | {r.asset_class for r in recs})
    standards = []
    for ac in classes:
        rf = cfg.get("rubric_file") if ac == "bridge_element" else None
        try:
            rub = load_rubric(ac, rf)
        except (FileNotFoundError, KeyError):
            continue
        standards.append({"asset_class": ac, "standard": rub.get("standard"), "rubric_file": rf or RUBRIC_FOR_CLASS.get(ac), "source_note": rub.get("source_note", "")})
    routed = sum(1 for g in gate if g["routed"])
    datasets = sorted({r.source_dataset for r in recs if r.source_dataset})
    workload_d: dict = {"cleared_by_gate": len(gate) - routed, "gated": len(gate), "prefilled": len(fs) - levels["U"], "awaiting_decision": reviews["pending"],
                        "reimage_or_metadata": levels["U"], "seconds_gate": round(sec_gate, 1), "seconds_grade": round(sec_grade, 1), "assumption": None}
    if workload is not None:
        workload_d["assumption"] = {"minutes_per_image": workload.minutes_per_image, "source": workload.source,
                                    "cleared_minutes": round((len(gate) - routed) * workload.minutes_per_image, 1)}
    used = sorted(set(models["gate"]) | set(models["grade"]))
    return {
        "client_id": client_id, "slug": client_slug(client_id, {client_of(r) for r in records.values()}), "run": run_dir.name,
        "generated_at": gen.isoformat(timespec="seconds"), "today": today.isoformat(), "grouping_basis": basis,
        "run_window": {"first": min(c["ts"] for c in calls), "last": max(c["ts"] for c in calls)} if calls else None,
        "config": cfg, "models": models, "prices_per_mtok": {mdl: PRICES_PER_MTOK.get(mdl) for mdl in used},
        "images": len(recs), "gated": len(gate), "routed": routed, "not_routed": len(gate) - routed, "unusable": sum(1 for g in gate if not g["usable"]),
        "graded_images": len({_image_of(f) for f in fs}), "findings": len(fs), "levels": levels,
        "levels_by_class": {ac: {lvl: sum(1 for f in fs if f.asset_class == ac and f.unified.level == lvl) for lvl in LEVELS} for ac in classes},
        "usd_total": round(usd_gate + usd_grade, 4), "usd_gate": round(usd_gate, 6), "usd_grade": round(usd_grade, 6), "cost_source": cost_source,
        "seconds_gate": round(sec_gate, 1), "seconds_grade": round(sec_grade, 1), "calls": len(calls), "reviews": reviews, "assets": assets,
        "queue": queue, "u_rows": u_rows, "actions": {c: [f.finding_id for f in fs if f.action.code == c] for c in ACTION_ORDER if any(f.action.code == c for f in fs)},
        "action_counts": {c: sum(1 for f in fs if f.action.code == c) for c in ACTION_ORDER}, "sla": sla,
        "sources": {"datasets": datasets, "licenses": {d: DEMO_LICENSES[d] for d in datasets if d in DEMO_LICENSES}},
        "workload": workload_d, "asset_classes": classes, "standards": standards, "severity_weight": SEVERITY_WEIGHT,
        "orphan_findings": sum(1 for f in fs if _image_of(f) not in known),
        "inventory": [{"image_id": r.image_id, "sha256_12": r.sha256[:12], "size": f"{r.width}x{r.height}", "captured_on": r.captured_on, "source_video": r.source_video,
                       "frame_time_s": r.frame_time_s, "asset_class": r.asset_class, "gate": _gate_verdict(gate_by.get(r.image_id))} for r in recs],
        "thumbs": {"embedded": 0, "omitted": [], "encoding": "", "findings": len(fs), "on_disk": 0},
    }


# ---------- evidence thumbnails and the file-size budget ----------


@dataclass
class ThumbBudget:
    max_bytes: int = 5_000_000  # whole HTML file
    max_side: int = 480  # long side of the thumbnail
    quality: int = 72  # JPEG quality
    max_thumbs: int = 60  # findings that get an image, in queue order
    fallback_side: int = 320  # first degradation step
    fallback_quality: int = 60


@lru_cache(maxsize=8)
def _load_rgb(path: str, mtime: float) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except OSError:
        return None


def evidence_thumbnail(f: Finding, rec: ImageRecord, *, max_side: int, quality: int) -> Optional[bytes]:
    """JPEG bytes of the whole frame scaled to `max_side` with the finding's bbox drawn in LEVEL_COLOR after scaling
    (no box when tile == "full": the whole frame is the evidence) and a 1 px neutral border; None if the file is missing."""
    p = Path(rec.path)
    if not p.exists():
        return None
    src = _load_rgb(str(p), p.stat().st_mtime)
    if src is None:
        return None
    w, h = src.size
    img = src
    if max(w, h) < 320:  # thermal 24x40 crops: nearest so pixels stay pixels
        s = 320 / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)
    s2 = min(1.0, max_side / max(img.size))
    img = img.resize((max(1, int(img.width * s2)), max(1, int(img.height * s2))), Image.LANCZOS) if s2 < 1.0 else img.copy()
    scale = img.width / w
    d = ImageDraw.Draw(img)
    if len(f.evidence.bbox) == 4 and f.evidence.tile != "full":
        x0, y0, x1, y1 = (v * scale for v in f.evidence.bbox)
        d.rectangle([x0, y0, max(x0, x1), max(y0, y1)], outline=LEVEL_COLOR.get(f.unified.level, "#ffffff"), width=3)
    d.rectangle([0, 0, img.width - 1, img.height - 1], outline="#9ca3af", width=1)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def fit_thumbs_to_budget(html_without_images_len: int, thumbs: Mapping[str, bytes], budget: ThumbBudget,
                         reencode: Optional[Callable[[str, int, int], Optional[bytes]]] = None) -> Tuple[Dict[str, bytes], dict]:
    """Keep the first `max_thumbs` (queue order); if the estimated page (text + 4/3 x JPEG + 60 B per img) exceeds
    `max_bytes`, re-encode at the fallback size/quality, then drop from the lowest-ranked finding up until it fits.
    Returns (thumbs to embed, {"embedded", "omitted", "encoding"})."""

    def est(t: Mapping[str, bytes]) -> float:
        return html_without_images_len + sum(math.ceil(len(b) * 4 / 3) + 60 for b in t.values())

    chosen = dict(list(thumbs.items())[: budget.max_thumbs])
    encoding = f"{budget.max_side}px q{budget.quality}"
    if chosen and est(chosen) > budget.max_bytes and reencode is not None:
        redone = {fid: reencode(fid, budget.fallback_side, budget.fallback_quality) for fid in chosen}
        chosen = {fid: b for fid, b in redone.items() if b is not None}
        encoding = f"{budget.fallback_side}px q{budget.fallback_quality}"
    omitted: List[str] = []
    while chosen and est(chosen) > budget.max_bytes:
        fid = list(chosen)[-1]
        chosen.pop(fid)
        omitted.append(fid)
    return chosen, {"embedded": len(chosen), "omitted": omitted, "encoding": encoding}


# ---------- SVG helpers (no gradients, no animation, text escaped) ----------


@dataclass
class TimelinePoint:
    x: float
    label: str  # short text drawn beside the point
    color: str  # hex
    y_lane: int = 0  # lane index (e.g. ACTION_ORDER.index(code))
    title: str = ""  # <title> tooltip text


def _e(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _svg_open(width: float, height: float, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" width="{width:g}" height="{height:g}" role="img" '
            f'aria-label="{_e(title)}" style="max-width:100%;height:auto" font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">'
            f"<title>{_e(title)}</title>")


def svg_hbar(rows: Sequence[Tuple[str, float, str]], *, width: int = 640, bar_h: int = 22, gap: int = 8, label_w: int = 110,
             value_fmt: str = "{:g}", title: str = "") -> str:
    """Horizontal bars (label, value, color) scaled to the largest value; the value is printed on every bar so
    colour is never the only carrier. All-zero rows draw empty tracks with "0"."""
    vmax = max((float(v) for _, v, _ in rows), default=0.0) or 1.0
    track = max(10, width - label_w - 56)
    out = [_svg_open(width, gap + len(rows) * (bar_h + gap), title or "bar chart")]
    for i, (label, val, color) in enumerate(rows):
        y = gap + i * (bar_h + gap)
        w = (float(val) / vmax * track) if val and val > 0 else 0.0
        ty = y + bar_h * 0.7
        out.append(f'<rect x="{label_w}" y="{y}" width="{track}" height="{bar_h}" fill="currentColor" opacity="0.08"/>')
        if w > 0:
            out.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{_e(color)}"/>')
        out.append(f'<text x="{label_w - 8}" y="{ty:.1f}" text-anchor="end" font-size="13" fill="currentColor">{_e(label)}</text>')
        txt = _e(value_fmt.format(val))
        if w > 44:
            out.append(f'<text x="{label_w + w - 6:.1f}" y="{ty:.1f}" text-anchor="end" font-size="12" font-weight="600" fill="#ffffff">{txt}</text>')
        else:
            out.append(f'<text x="{label_w + w + 6:.1f}" y="{ty:.1f}" font-size="12" fill="currentColor">{txt}</text>')
    return "".join(out) + "</svg>"


def svg_donut(slices: Sequence[Tuple[str, float, str]], *, size: int = 180, thickness: int = 26, center_text: str = "",
              legend: bool = True, title: str = "") -> str:
    """Donut of (label, value, color). One non-zero slice is a full <circle> (a 360-degree arc collapses);
    zero total draws a grey ring and "none". The legend prints label, value and percent as text."""
    total = sum(max(0.0, float(v)) for _, v, _ in slices)
    cx = cy = size / 2
    r = (size - thickness) / 2
    out = [_svg_open(size + (220 if legend else 0), size, title or "donut chart")]
    nonzero = [(lab, float(v), c) for lab, v, c in slices if v and v > 0]
    if total <= 0:
        out.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="none" stroke="#9ca3af" stroke-width="{thickness}" opacity="0.5"/>')
        out.append(f'<text x="{cx:g}" y="{cy + 5:g}" text-anchor="middle" font-size="14" fill="currentColor">none</text>')
    elif len(nonzero) == 1:
        out.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="none" stroke="{_e(nonzero[0][2])}" stroke-width="{thickness}"><title>{_e(nonzero[0][0])}: {nonzero[0][1]:g}</title></circle>')
    else:
        a = -math.pi / 2
        for label, v, color in nonzero:
            sweep = v / total * 2 * math.pi
            x0, y0 = cx + r * math.cos(a), cy + r * math.sin(a)
            x1, y1 = cx + r * math.cos(a + sweep), cy + r * math.sin(a + sweep)
            out.append(f'<path d="M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 {1 if sweep > math.pi else 0} 1 {x1:.2f} {y1:.2f}" fill="none" '
                       f'stroke="{_e(color)}" stroke-width="{thickness}"><title>{_e(label)}: {v:g}</title></path>')
            a += sweep
    if center_text and total > 0:
        out.append(f'<text x="{cx:g}" y="{cy + 5:g}" text-anchor="middle" font-size="15" font-weight="600" fill="currentColor">{_e(center_text)}</text>')
    if legend:
        for i, (label, v, color) in enumerate(slices):
            y = 18 + i * 20
            pct = (float(v) / total * 100) if total else 0.0
            out.append(f'<rect x="{size + 10}" y="{y - 10}" width="12" height="12" rx="3" fill="{_e(color)}"/>'
                       f'<text x="{size + 28}" y="{y}" font-size="12" fill="currentColor">{_e(label)}: {float(v):g} ({pct:.0f}%)</text>')
    return "".join(out) + "</svg>"


def svg_timeline(points: Sequence[TimelinePoint], *, lanes: Sequence[str], width: int = 760, lane_h: int = 28, x_min: Optional[float] = None,
                 x_max: Optional[float] = None, x_label: str = "days from today", today_x: Optional[float] = 0.0,
                 tick_every: Optional[float] = None, title: str = "") -> str:
    """Lanes at left, ticks auto-chosen from the span, a dashed line at `today_x`, r=6 circles with a 1 px outline
    and labels nudged right; crowded labels in one lane alternate above/below."""
    label_w, pad_r, top, axis_h = 96, 24, 8, 34
    xs = [p.x for p in points]
    x_min = (min([0.0] + xs) - 3 if xs else -3.0) if x_min is None else x_min
    x_max = ((max(xs) + 7) if xs else 7.0) if x_max is None else x_max
    x_max = x_max if x_max > x_min else x_min + 1
    span = x_max - x_min
    tick = tick_every or next((t for t in (1, 7, 14, 30, 90, 180, 365) if span / t <= 12), 365)
    plot_w = width - label_w - pad_r
    y_axis = top + len(lanes) * lane_h

    def sx(x: float) -> float:
        return label_w + (x - x_min) / span * plot_w

    out = [_svg_open(width, y_axis + axis_h, title or "timeline")]
    for i, lane in enumerate(lanes):
        yc = top + i * lane_h + lane_h / 2
        out.append(f'<line x1="{label_w}" y1="{yc:.1f}" x2="{width - pad_r}" y2="{yc:.1f}" stroke="currentColor" opacity="0.15"/>'
                   f'<text x="{label_w - 8}" y="{yc + 4:.1f}" text-anchor="end" font-size="12" fill="currentColor">{_e(lane)}</text>')
    t = math.ceil(x_min / tick) * tick
    while t <= x_max:
        out.append(f'<line x1="{sx(t):.1f}" y1="{top}" x2="{sx(t):.1f}" y2="{y_axis}" stroke="currentColor" opacity="0.12"/>'
                   f'<text x="{sx(t):.1f}" y="{y_axis + 14}" text-anchor="middle" font-size="11" fill="currentColor">{t:g}</text>')
        t += tick
    out.append(f'<text x="{label_w + plot_w / 2:.1f}" y="{y_axis + 28}" text-anchor="middle" font-size="11" fill="currentColor">{_e(x_label)}</text>')
    if today_x is not None and x_min <= today_x <= x_max:
        out.append(f'<line x1="{sx(today_x):.1f}" y1="{top}" x2="{sx(today_x):.1f}" y2="{y_axis}" stroke="currentColor" stroke-dasharray="4 3"/>'
                   f'<text x="{sx(today_x) + 4:.1f}" y="{top + 10}" font-size="11" fill="currentColor">today</text>')
    last_x: Dict[int, float] = {}
    flip: Dict[int, bool] = {}
    for p in sorted(points, key=lambda q: (q.y_lane, q.x)):
        if not 0 <= p.y_lane < len(lanes):
            continue
        yc = top + p.y_lane * lane_h + lane_h / 2
        x = sx(min(max(p.x, x_min), x_max))
        flip[p.y_lane] = (not flip.get(p.y_lane, False)) if (p.y_lane in last_x and x - last_x[p.y_lane] < 60) else False
        last_x[p.y_lane] = x
        ty = yc - 9 if flip[p.y_lane] else yc + 4
        out.append(f'<circle cx="{x:.1f}" cy="{yc:.1f}" r="6" fill="{_e(p.color)}" stroke="#111827" stroke-width="1"><title>{_e(p.title or p.label)}</title></circle>'
                   f'<text x="{x + 9:.1f}" y="{ty:.1f}" font-size="11" fill="currentColor">{_e(p.label)}</text>')
    return "".join(out) + "</svg>"


def svg_legend(items: Sequence[Tuple[str, str]]) -> str:
    """(label, color) chips in one row, used under charts."""
    x, parts = 0, []
    for label, color in items:
        parts.append(f'<rect x="{x}" y="4" width="12" height="12" rx="3" fill="{_e(color)}"/><text x="{x + 16}" y="14" font-size="12" fill="currentColor">{_e(label)}</text>')
        x += 32 + int(len(label) * 6.5)
    return _svg_open(max(x, 20), 20, "legend") + "".join(parts) + "</svg>"


# ---------- HTML rendering ----------

CSS = """
:root{--bg:#ffffff;--fg:#1f2937;--muted:#6b7280;--card:#f9fafb;--line:#e5e7eb;--accent:#2563eb}
@media (prefers-color-scheme: dark){:root{--bg:#0f172a;--fg:#e5e7eb;--muted:#9ca3af;--card:#1e293b;--line:#334155;--accent:#60a5fa}}
body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;max-width:1100px;margin:auto;padding:0 16px;line-height:1.45}
h1{font-size:1.7em;margin:.4em 0 .2em}h2{font-size:1.25em;border-bottom:2px solid var(--accent);padding-bottom:4px;margin-top:0}
nav.toc{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:2;font-size:.9em}
nav.toc a{margin-right:14px;color:var(--accent);text-decoration:none;white-space:nowrap}
section{padding:18px 0;border-bottom:1px solid var(--line)}
.banner{background:var(--card);border-left:4px solid var(--accent);padding:10px 14px;margin:12px 0}
.facts,.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:10px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
.tile .n{font-size:1.6em;font-weight:700;font-variant-numeric:tabular-nums}.tile .v{font-weight:600;word-break:break-word}
.tile.level{border-left:6px solid}
table{border-collapse:collapse;width:100%;font-size:.9em;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--card)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;color:white;padding:1px 8px;border-radius:9px;font-weight:600;letter-spacing:.2px}
.badge-S4{outline:2px solid #dc2626;outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;break-inside:avoid}
.card img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line)}
.card blockquote{margin:6px 0;padding:4px 10px;border-left:3px solid var(--accent);color:var(--muted)}
.card p{margin:6px 0}.muted{color:var(--muted);font-size:.9em}
.charts{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}.charts>div{flex:1 1 320px;min-width:0}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;margin-right:4px}
@media print{nav.toc{display:none}section{break-before:page}.card{break-inside:avoid}*{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
"""


class _Raw(str):
    """A cell already rendered as HTML (badges); everything else is escaped."""


def _fmt(v) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _badge(level: str) -> _Raw:
    return _Raw(f'<span class="badge badge-{_e(level)}" style="background:{LEVEL_COLOR.get(level, "#555")}">{_e(level)}</span>')


def _table(headers: Sequence[str], rows: Sequence[Sequence], numeric: Sequence[int] = ()) -> str:
    def cell(tag: str, i: int, c) -> str:
        cls = ' class="num"' if i in numeric else ""
        return f"<{tag}{cls}>{c if isinstance(c, _Raw) else _e(_fmt(c))}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, c) for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _tiles(items: Sequence[Tuple[str, object, Optional[str]]], cls: str = "n") -> str:
    out = []
    for label, value, color in items:
        style = f' class="tile level" style="border-left-color:{color}"' if color else ' class="tile"'
        out.append(f'<div{style}><div class="muted">{_e(label)}</div><div class="{cls}">{_e(value)}</div></div>')
    return "".join(out)


def _sla_points(m: dict) -> List[TimelinePoint]:
    return [TimelinePoint(x=r["due_in_days"], label=f"#{r['rank']}", color=LEVEL_COLOR.get(r["level"], "#555"), y_lane=ACTION_ORDER.index(r["action"]),
                          title=f"#{r['rank']} {r['level']} {r['finding_id']} due {r['due_on']} ({r['anchor_basis']})") for r in m["sla"] if r["due_in_days"] is not None]


def _workload_lines(m: dict) -> List[str]:
    w = m["workload"]
    lines = [f"Images the gate cleared without a heavy grade: {w['cleared_by_gate']} of {w['gated']} (the reviewer does not need to open these first).",
             f"Findings pre-filled with native grade, quoted criterion, action and justification: {w['prefilled']}.",
             f"Findings awaiting a human decision: {w['awaiting_decision']}.",
             f"Rows needing a re-image or metadata before any grade: {w['reimage_or_metadata']}.",
             f"Model time: gate {w['seconds_gate']} s, grade {w['seconds_grade']} s (wall time of the calls, not human time)."]
    a = w.get("assumption")
    if a:
        lines.append(f"If first-pass triage takes {a['minutes_per_image']:g} min per image [Assumption, source: {a['source']}], "
                     f"the cleared images correspond to {a['cleared_minutes']:g} min.")
    return lines


def _cover_html(m: dict) -> str:
    rw, cfg = m["run_window"], m["config"]
    facts = [("Images", m["images"], None), ("Run window (first to last logged call)", f"{rw['first']} to {rw['last']}" if rw else "no logged calls", None),
             ("Gate model(s)", ", ".join(m["models"]["gate"]) or "none", None), ("Grader model(s)", ", ".join(m["models"]["grade"]) or "none", None),
             ("Tiles", "on" if cfg.get("tiles") else "off", None), ("Gate threshold", cfg.get("gate_min_conf", "n/a"), None),
             ("USD for this client's images", f"{m['usd_total']:.4f} (API list price from {m['cost_source']}; local models $0)", None),
             ("Model seconds", f"gate {m['seconds_gate']} s, grade {m['seconds_grade']} s", None)]
    out = [f'<section id="cover"><h1>Inspection report: {_e(m["client_id"])}</h1>',
           f'<p class="muted">run {_e(m["run"])} &middot; generated {_e(m["generated_at"])} UTC</p>',
           f'<div class="facts">{_tiles(facts, "v")}</div>', f'<div class="banner">{_e(BANNER)}</div>']
    if m["sources"]["licenses"]:
        out.append(f'<div class="banner">{_e(DEMO_LINE)} ' + "; ".join(f"{_e(d)}: {_e(lic)}" for d, lic in m["sources"]["licenses"].items()) + "</div>")
    if m["grouping_basis"] != "client_id":
        out.append(f'<p class="muted">Client id was not supplied; grouped by {_e(m["grouping_basis"])}.</p>')
    if m["orphan_findings"]:
        out.append(f'<p class="muted">{m["orphan_findings"]} finding(s) reference an image id missing from the manifest and were assigned here.</p>')
    return "".join(out) + "</section>"


def _summary_html(m: dict) -> str:
    lv, rv = m["levels"], m["reviews"]
    kpis = [("Images", m["images"], None), ("Routed / gated", f"{m['routed']} / {m['gated']}", None), ("Findings", m["findings"], None)]
    kpis += [(lvl, lv[lvl], LEVEL_COLOR[lvl]) for lvl in ("S4", "S3", "S2", "S1", "S0", "U")]
    kpis += [("Reviews done / pending", f"{rv['total']} / {rv['pending']}", None), ("Assets", len(m["assets"]), None), ("USD", f"{m['usd_total']:.4f}", None)]
    out = [f'<section id="summary"><h2>Executive summary</h2><div class="kpis">{_tiles(kpis)}</div><p class="muted">{_e(U_FOOTNOTE)}.</p>',
           '<div class="panel"><b>Workload: what the cascade took off the reviewer&#39;s desk (measured)</b><ul>' + "".join(f"<li>{_e(x)}</li>" for x in _workload_lines(m)) + "</ul></div>"]
    donut = svg_donut([(k, rv[k], REVIEW_COLOR[k]) for k in ("accepted", "overridden", "marked_u", "pending")], center_text=f"{rv['total']}/{m['findings']}", title="Review status")
    out.append(f'<div class="charts"><div><h3>Review audit</h3>{donut}' + _table(["Status", "Findings"], [(k, rv[k]) for k in ("accepted", "overridden", "marked_u", "pending")], (1,)) + "</div>")
    if rv["total"] > 0 and rv["timeline"]:
        pts = [TimelinePoint(x=r["n"], label=f"{r['agreement_rate']:.0%}", color=REVIEW_COLOR.get(r["action"], "#999"), title=f"#{r['n']} {r['action']} {r['finding_id']} by {r['reviewer']}") for r in rv["timeline"]]
        tl = svg_timeline(pts, lanes=["decisions"], x_min=0, x_max=len(pts) + 1, x_label="decision # (label = running agreement rate, FR-19)", today_x=None, tick_every=1 if len(pts) <= 12 else None, title="Review agreement timeline")
        out.append(f"<div><h3>Agreement over decisions</h3>{tl}" + _table(["#", "Finding", "Action", "Prior", "New", "Reviewer", "Agreement"], [(r["n"], r["finding_id"], r["action"], r["prior_level"], r["new_level"], r["reviewer"], f"{r['agreement_rate']:.2f}") for r in rv["timeline"]], (0, 6)) + "</div>")
    return "".join(out) + "</div></section>"


def _levels_html(m: dict) -> str:
    lv = m["levels"]
    rows = [(lvl, lv[lvl], LEVEL_COLOR[lvl]) for lvl in LEVELS]
    out = [f'<section id="levels"><h2>Level distribution</h2><div class="charts"><div>{svg_hbar(rows, title="Findings per unified level")}</div>',
           "<div>" + _table(["Level", "Count", "Meaning"], [(_badge(lvl), lv[lvl], LEVEL_MEANING[lvl]) for lvl in LEVELS], (1,)) + "</div></div>"]
    if len(m["levels_by_class"]) > 1:
        out.append('<div class="charts">' + "".join(f"<div><h3>{_e(ac)}</h3>{svg_hbar([(lvl, c[lvl], LEVEL_COLOR[lvl]) for lvl in LEVELS], width=420, title=f'Levels for {ac}')}</div>" for ac, c in m["levels_by_class"].items()) + "</div>")
    return "".join(out) + "</section>"


def _assets_html(m: dict) -> str:
    per_image = any(a["basis"] == "image" for a in m["assets"])
    note = '<p class="muted">Asset id not supplied; one row per image.</p>' if per_image else ""
    rows = [(a["asset_id"] + (f" ({a['gate_caption']})" if a["gate_caption"] else ""), a["images"], a["findings"], _badge(a["worst_level"]) if a["worst_level"] != "-" else "-",
             a["worst_native"], a["top_action"], a["min_sla_days"] if a["min_sla_days"] is not None else "none stated", a["u_count"], a["pending_reviews"], a["best_rank"], a["source"]) for a in m["assets"]]
    return (f'<section id="assets"><h2>Assets</h2>{note}'
            + _table(["Asset", "Images", "Findings", "Worst level", "Worst native", "Top action", "SLA days", "U", "Pending review", "Best queue rank", "Source"], rows, (1, 2, 6, 7, 8, 9)) + "</section>")


def _card_html(row: dict, thumb: Optional[bytes], omitted: bool, cap: int) -> str:
    fid, lvl = row["finding_id"], row["level"]
    head = f'<div><b>#{_fmt(row["queue_rank"])}</b> {_badge(lvl)} <b>{_e(row["native"])}</b> &middot; {_e(row["defect_type"])}</div>'
    if thumb is not None:
        img = f'<img src="data:image/jpeg;base64,{base64.standard_b64encode(thumb).decode("ascii")}" alt="evidence {_e(fid)}">'
    elif omitted:
        img = f'<p class="muted">thumbnail omitted for file size; see evidence/{_e(finding_slug(fid))}.jpg</p>'
    elif not row.get("image_on_disk"):
        img = '<p class="muted">source image not on disk</p>'
    else:
        img = f'<p class="muted">no thumbnail (cap of {cap} thumbnails per report)</p>'
    cap_line = f'{row["image_id"]} &middot; {_fmt(row["image_size"])} &middot; tile {_fmt(row["tile"])} &middot; bbox {_fmt(row["bbox"])}'
    if row.get("source_video"):
        cap_line += f' &middot; frame at {_fmt(row["frame_time_s"])} s of {_e(row["source_video"])}'
    crit = "".join(f"<blockquote>{_e(c)}</blockquote>" for c in row["criteria_list"]) or "<blockquote>(none quoted)</blockquote>"
    meas = ", ".join(f"{k} {v:g}" for k, v in row["measurements"].items()) or "none measurable"
    act = f'Action: <b>{_e(row["action_code"])}</b>' + (f' within {row["sla_days"]} days' if row["sla_days"] is not None else "") + f' &middot; {_e(row["action_basis"])}'
    rev = f'Review: {_e(row["review_status"])}'
    if row["review_status"] != "pending":
        rev += f' by {_e(row["reviewer"])} at {_e(row["reviewed_at"])}, prior level {_e(row["prior_level"])}'
    return (f'<article class="card">{head}{img}<p class="muted">{cap_line}</p><p><b>Criteria matched (verbatim from rubric)</b></p>{crit}'
            f'<p>{_e(row["justification"])}</p><p class="muted">Measurements: {_e(meas)}; flags {_e(row["flags"] or "none")}; confidence {_fmt(row["confidence"])}; uncertainty {_e(row["uncertainty"])}</p>'
            f'<p>{act}</p><p class="muted">{rev}</p><p class="muted">{_e(row["model"] or "no model")} &middot; ${row["usd"]:.4f} &middot; {row["seconds"]:.1f} s &middot; {_e(fid)}</p></article>')


def _findings_html(m: dict, thumbs: Mapping[str, bytes], omitted: set) -> str:
    cap = len(thumbs) + len(omitted)
    cards = "".join(_card_html(r, thumbs.get(r["finding_id"]), r["finding_id"] in omitted, cap) for r in m["queue"])
    return f'<section id="findings"><h2>Findings (queue order)</h2><div class="grid">{cards or "<p>No findings for this client.</p>"}</div></section>'


def _u_html(m: dict) -> str:
    rows = [(r["rank"], r["image_id"], r["finding_id"], r["reason"], r["basis"], r["next_step"]) for r in m["u_rows"]]
    body = _table(["Rank", "Image", "Finding", "Reason (verbatim)", "Rubric basis", "Suggested next step (keyword match on the model's reason)"], rows, (0,)) if rows else "<p>No U findings.</p>"
    return f'<section id="u-list"><h2>U list ({len(rows)})</h2>{body}<p><b>{_e(U_SENTENCE)}</b></p></section>'


def _sla_html(m: dict) -> str:
    pts = _sla_points(m)
    bases = sorted({r["anchor_basis"] for r in m["sla"]})
    out = [f'<section id="sla"><h2>SLA calendar and action list</h2><p class="muted">Anchor date: {_e(", ".join(bases) or "n/a")} '
           f'(capture date when the image carries one, else the run date of the first logged call). due_on = anchor + sla_days; due_in_days counted from {_e(m["today"])}. '
           "Rows without an SLA in the rubric row are listed but not charted.</p>",
           svg_timeline(pts, lanes=list(ACTION_ORDER), title="Due-in days per action lane") if pts else "<p>No dated SLA rows.</p>",
           svg_legend([(lvl, LEVEL_COLOR[lvl]) for lvl in LEVELS]),
           '<div class="charts"><div>' + svg_donut([(c, m["action_counts"][c], ACTION_COLOR[c]) for c in ACTION_ORDER], center_text=str(m["findings"]), title="Findings per action code") + "</div>"
           + "<div>" + _table(["Action", "Findings"], [(c, m["action_counts"][c]) for c in ACTION_ORDER], (1,)) + "</div></div>"]
    for code in ACTION_ORDER:
        rows = [r for r in m["sla"] if r["action"] == code]
        if not rows:
            continue
        out.append(f"<h3>{_e(code)} ({len(rows)})</h3>" + _table(
            ["Rank", "Asset", "Image", "Level", "Native", "SLA days", "Due on", "Due in days", "Anchor"],
            [(r["rank"], r["asset"], r["image_id"], _badge(r["level"]), r["native"], r["sla_days"] if r["sla_days"] is not None else "no SLA stated in rubric row",
              r["due_on"], r["due_in_days"], r["anchor_basis"]) for r in rows], (0, 5, 7)))
    return "".join(out) + "</section>"


def _method_html(m: dict) -> str:
    cfg, th = m["config"], m["thumbs"]
    forced = ", ".join(cfg.get("force_route_classes") or []) or "none"
    stds = "".join(f"<li><b>{_e(s['asset_class'])}</b>: {_e(s['standard'])} ({_e(s['rubric_file'])}). {_e(s['source_note'])}</li>" for s in m["standards"]) or "<li>none</li>"
    prices = ", ".join(f"{k} {v[0]}/{v[1]}" if v else f"{k} $0 (local or unknown)" for k, v in m["prices_per_mtok"].items()) or "none"
    est = ["due-in days for rows anchored on the run date rather than a capture date"]
    if m["workload"].get("assumption"):
        est.append(f"the workload minutes line (source: {m['workload']['assumption']['source']})")
    inv = _table(["Image", "sha256[:12]", "Size", "Captured on", "Source", "Asset class", "Gate"],
                 [(r["image_id"], r["sha256_12"], r["size"], r["captured_on"], f"video {r['source_video']} @ {r['frame_time_s']} s" if r["source_video"] else "still", r["asset_class"], r["gate"]) for r in m["inventory"]])
    return (f'<section id="method"><h2>Methodology and honesty notes</h2>'
            f"<p><b>Pipeline</b>: gate &rarr; crop &rarr; grade &rarr; prioritize &rarr; review &rarr; export. Gate backend {_e(cfg.get('gate', '?'))} ({_e(', '.join(m['models']['gate']) or 'no calls')}), "
            f"grader backend {_e(cfg.get('grader', '?'))} ({_e(', '.join(m['models']['grade']) or 'no calls')}), tiles {'on' if cfg.get('tiles') else 'off'}. "
            f"Routing rule: an image goes to the grader when the gate finds it unusable, finds damage, or is below {_e(cfg.get('gate_min_conf', '?'))} confidence that it is clean; forced classes: {_e(forced)}.</p>"
            f"<p><b>Standards used</b>:</p><ul>{stds}</ul>"
            "<p><b>Unified level meanings</b>:</p><ul>" + "".join(f"<li>{_badge(lvl)} {_e(LEVEL_MEANING[lvl])}</li>" for lvl in LEVELS) + "</ul>"
            f"<p><b>Queue formula</b>: score = severity_weight[level] &times; criticality &times; consequence &times; urgency, severity_weight = {_e(json.dumps(m['severity_weight']))}; criticality is 1 unless supplied per asset; any S4 sorts first and is escalated same day; U is listed, never scored.</p>"
            f"<p><b>Measured</b>: counts, levels, cost ({_e(m['cost_source'])} at list prices USD per million tokens in/out: {_e(prices)}), model seconds, review decisions.</p>"
            f"<p><b>Estimated / assumption</b>: {_e('; '.join(est))}.</p>"
            "<p><b>Not claimed</b>: field accuracy, or any accuracy figure in this report; accuracy lives in eval/reports/&lt;run&gt;.md with n and CI, on labelled data only; dataset labels are never shown to the model.</p>"
            f"<p><b>Thumbnails</b>: {th['embedded']} of {th['findings']} findings embedded ({_e(th['encoding'] or 'none')}); {len(th['omitted'])} omitted for file size; {th['on_disk']} source images found on disk.</p>"
            f"<h3>Appendix: image inventory</h3>{inv}</section>")


def render_client_html(m: dict, thumbs: Mapping[str, bytes], omitted: Iterable[str] = ()) -> str:
    """The self-contained page: inline CSS, inline SVG, base64 thumbnails, no script, no link, no remote URL."""
    om = set(omitted)
    nav = '<nav class="toc">' + " ".join(f'<a href="#{sid}">{_e(t)}</a>' for sid, t in SECTIONS) + "</nav>"
    body = _cover_html(m) + _summary_html(m) + _levels_html(m) + _assets_html(m) + _findings_html(m, thumbs, om) + _u_html(m) + _sla_html(m) + _method_html(m)
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_e(f"Inspection report {m['client_id']} {m['run']}")}</title><style>{CSS}</style></head><body>{nav}{body}</body></html>\n')


# ---------- markdown twin ----------


def _md_cell(v) -> str:
    return _fmt(v).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> List[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows]


def render_client_markdown(m: dict, thumb_paths: Mapping[str, Optional[str]]) -> str:
    """Same eight sections as the HTML; charts become tables, cards link `evidence/<slug>.jpg` relatively."""
    rw, lv, rv, cfg = m["run_window"], m["levels"], m["reviews"], m["config"]
    L: List[str] = [f"# Inspection report: {m['client_id']}", "", f"run {m['run']} · generated {m['generated_at']} UTC", ""]
    L += _md_table(["Fact", "Value"], [("Images", m["images"]), ("Run window", f"{rw['first']} to {rw['last']}" if rw else "no logged calls"),
                                       ("Gate model(s)", ", ".join(m["models"]["gate"]) or "none"), ("Grader model(s)", ", ".join(m["models"]["grade"]) or "none"),
                                       ("Tiles", "on" if cfg.get("tiles") else "off"), ("Gate threshold", cfg.get("gate_min_conf", "n/a")),
                                       ("USD for this client's images", f"{m['usd_total']:.4f} (API list price from {m['cost_source']}; local models $0)"),
                                       ("Model seconds", f"gate {m['seconds_gate']} s, grade {m['seconds_grade']} s")])
    L += ["", f"> {BANNER}", ""]
    if m["sources"]["licenses"]:
        L += [f"> {DEMO_LINE} " + "; ".join(f"{d}: {lic}" for d, lic in m["sources"]["licenses"].items()), ""]
    if m["grouping_basis"] != "client_id":
        L += [f"Client id was not supplied; grouped by {m['grouping_basis']}.", ""]
    L += ["## Executive summary", ""]
    L += _md_table(["Images", "Routed / gated", "Findings", "S4", "S3", "S2", "S1", "S0", "U", "Reviews done / pending", "Assets", "USD"],
                   [(m["images"], f"{m['routed']} / {m['gated']}", m["findings"], lv["S4"], lv["S3"], lv["S2"], lv["S1"], lv["S0"], lv["U"], f"{rv['total']} / {rv['pending']}", len(m["assets"]), f"{m['usd_total']:.4f}")])
    L += ["", f"{U_FOOTNOTE}.", "", "**Workload (measured)**", ""] + [f"- {x}" for x in _workload_lines(m)] + [""]
    L += _md_table(["Review status", "Findings"], [(k, rv[k]) for k in ("accepted", "overridden", "marked_u", "pending")]) + [""]
    if rv["timeline"]:
        L += _md_table(["#", "Finding", "Action", "Prior", "New", "Reviewer", "Agreement"], [(r["n"], r["finding_id"], r["action"], r["prior_level"], r["new_level"], r["reviewer"], f"{r['agreement_rate']:.2f}") for r in rv["timeline"]]) + [""]
    L += ["## Level distribution", ""] + _md_table(["Level", "Count", "Meaning"], [(lvl, lv[lvl], LEVEL_MEANING[lvl]) for lvl in LEVELS]) + [""]
    L += ["## Assets", ""] + (["Asset id not supplied; one row per image.", ""] if any(a["basis"] == "image" for a in m["assets"]) else [])
    L += _md_table(["Asset", "Images", "Findings", "Worst level", "Worst native", "Top action", "SLA days", "U", "Pending review", "Best queue rank", "Source"],
                   [(a["asset_id"] + (f" ({a['gate_caption']})" if a["gate_caption"] else ""), a["images"], a["findings"], a["worst_level"], a["worst_native"], a["top_action"],
                     a["min_sla_days"] if a["min_sla_days"] is not None else "none stated", a["u_count"], a["pending_reviews"], a["best_rank"], a["source"]) for a in m["assets"]]) + [""]
    L += ["## Findings (queue order)", ""]
    for r in m["queue"]:
        L += [f"### #{_fmt(r['queue_rank'])} · {r['level']} · {r['native']} · {r['defect_type']}", ""]
        tp = thumb_paths.get(r["finding_id"])
        L += [f"![evidence {r['finding_id']}]({tp})" if tp else ("(source image not on disk)" if not r.get("image_on_disk") else "(no thumbnail written)"), ""]
        cap = f"{r['image_id']} · {_fmt(r['image_size'])} · tile {_fmt(r['tile'])} · bbox {_fmt(r['bbox'])}"
        if r.get("source_video"):
            cap += f" · frame at {_fmt(r['frame_time_s'])} s of {r['source_video']}"
        L += [cap, "", "Criteria matched (verbatim from rubric):", ""] + [f"> {c}" for c in (r["criteria_list"] or ["(none quoted)"])] + ["", r["justification"], ""]
        meas = ", ".join(f"{k} {v:g}" for k, v in r["measurements"].items()) or "none measurable"
        L += [f"Measurements: {meas}; flags {r['flags'] or 'none'}; confidence {_fmt(r['confidence'])}; uncertainty {r['uncertainty']}", ""]
        L += [f"Action: **{r['action_code']}**" + (f" within {r['sla_days']} days" if r["sla_days"] is not None else "") + f" · {r['action_basis']}", ""]
        rev = f"Review: {r['review_status']}" + (f" by {r['reviewer']} at {r['reviewed_at']}, prior level {r['prior_level']}" if r["review_status"] != "pending" else "")
        L += [rev, "", f"{r['model'] or 'no model'} · ${r['usd']:.4f} · {r['seconds']:.1f} s · `{r['finding_id']}`", ""]
    L += [f"## U list ({len(m['u_rows'])})", ""]
    L += _md_table(["Rank", "Image", "Finding", "Reason (verbatim)", "Rubric basis", "Suggested next step (keyword match on the model's reason)"],
                   [(r["rank"], r["image_id"], r["finding_id"], r["reason"], r["basis"], r["next_step"]) for r in m["u_rows"]]) if m["u_rows"] else ["No U findings."]
    L += ["", f"**{U_SENTENCE}**", "", "## SLA calendar and action list", "", f"Due-in days counted from {m['today']}; anchor is the capture date when present, else the run date of the first logged call.", ""]
    L += _md_table(["Action", "Findings"], [(c, m["action_counts"][c]) for c in ACTION_ORDER]) + [""]
    for code in ACTION_ORDER:
        rows = [r for r in m["sla"] if r["action"] == code]
        if rows:
            L += [f"### {code} ({len(rows)})", ""] + _md_table(["Rank", "Asset", "Image", "Level", "Native", "SLA days", "Due on", "Due in days", "Anchor"],
                                                                 [(r["rank"], r["asset"], r["image_id"], r["level"], r["native"], r["sla_days"] if r["sla_days"] is not None else "no SLA stated in rubric row", r["due_on"], r["due_in_days"], r["anchor_basis"]) for r in rows]) + [""]
    th = m["thumbs"]
    L += ["## Methodology and honesty notes", "",
          f"Pipeline: gate -> crop -> grade -> prioritize -> review -> export. Gate backend {cfg.get('gate', '?')} ({', '.join(m['models']['gate']) or 'no calls'}), grader backend {cfg.get('grader', '?')} ({', '.join(m['models']['grade']) or 'no calls'}), tiles {'on' if cfg.get('tiles') else 'off'}, gate threshold {cfg.get('gate_min_conf', '?')}, forced classes {', '.join(cfg.get('force_route_classes') or []) or 'none'}.", "",
          "Standards used:", ""] + [f"- {s['asset_class']}: {s['standard']} ({s['rubric_file']}). {s['source_note']}" for s in m["standards"]] + ["", "Unified level meanings:", ""]
    L += [f"- {lvl}: {LEVEL_MEANING[lvl]}" for lvl in LEVELS]
    L += ["", f"Queue formula: score = severity_weight[level] x criticality x consequence x urgency, severity_weight = {json.dumps(m['severity_weight'])}; criticality is 1 unless supplied per asset.", "",
          f"Measured: counts, levels, cost ({m['cost_source']} at costlog list prices), model seconds, review decisions.",
          "Estimated / assumption: due-in days anchored on the run date" + (f"; workload minutes line (source: {m['workload']['assumption']['source']})" if m["workload"].get("assumption") else "") + ".",
          "Not claimed: field accuracy or any accuracy figure; accuracy lives in eval/reports/<run>.md with n and CI on labelled data only; dataset labels are never shown to the model.",
          f"Thumbnails: {th['embedded']} of {th['findings']} embedded ({th['encoding'] or 'none'}); {len(th['omitted'])} omitted for file size.", "", "### Appendix: image inventory", ""]
    L += _md_table(["Image", "sha256[:12]", "Size", "Captured on", "Source", "Asset class", "Gate"],
                   [(r["image_id"], r["sha256_12"], r["size"], r["captured_on"], f"video {r['source_video']} @ {r['frame_time_s']} s" if r["source_video"] else "still", r["asset_class"], r["gate"]) for r in m["inventory"]])
    return "\n".join(L) + "\n"


# ---------- public API ----------


def write_client_report(run_dir: Path, client_id: str, records: Mapping[str, ImageRecord], *, run: Optional[dict] = None,
                        budget: ThumbBudget = ThumbBudget(), workload: Optional[WorkloadAssumption] = None, today: Optional[date] = None) -> Path:
    """Write `clients/<slug>/{report.html, report.md, report.json, evidence/*.jpg}` for one client; returns report.html.

    The HTML is measured after writing and asserted to be within `budget.max_bytes`."""
    run_dir = Path(run_dir)
    run = run if run is not None else load_run(run_dir)
    m = client_metrics(run_dir, client_id, records, run=run, today=today, workload=workload)
    cdir = run_dir / "clients" / m["slug"]
    (cdir / "evidence").mkdir(parents=True, exist_ok=True)
    by_fid = {f.finding_id: f for f in run["findings"]}

    def encode(fid: str, side: int, quality: int) -> Optional[bytes]:
        rec = records.get(_image_of(by_fid[fid]))
        return evidence_thumbnail(by_fid[fid], rec, max_side=side, quality=quality) if rec else None

    primary: Dict[str, bytes] = {}
    for row in m["queue"][: budget.max_thumbs]:
        b = encode(row["finding_id"], budget.max_side, budget.quality)
        if b is not None:
            primary[row["finding_id"]] = b
            row["thumb"] = f"evidence/{finding_slug(row['finding_id'])}.jpg"
    text_len = len(render_client_html(m, {}, ()).encode("utf-8"))
    chosen, info = fit_thumbs_to_budget(text_len, primary, budget, reencode=encode)
    omitted = [fid for fid in primary if fid not in chosen]
    m["thumbs"].update(info, omitted=omitted, on_disk=len(primary))
    page = render_client_html(m, chosen, omitted)
    while chosen and len(page.encode("utf-8")) > budget.max_bytes:  # the estimate is close, the file is the truth
        last = list(chosen)[-1]
        chosen.pop(last)
        omitted.append(last)
        m["thumbs"].update(embedded=len(chosen), omitted=omitted)
        page = render_client_html(m, chosen, omitted)
    for fid, b in primary.items():
        (cdir / "evidence" / f"{finding_slug(fid)}.jpg").write_bytes(chosen.get(fid, b))
    html_path = cdir / "report.html"
    html_path.write_bytes(page.encode("utf-8"))
    assert html_path.stat().st_size <= budget.max_bytes, f"{html_path} exceeds {budget.max_bytes} bytes"
    (cdir / "report.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    (cdir / "report.md").write_text(render_client_markdown(m, {r["finding_id"]: r["thumb"] for r in m["queue"]}), encoding="utf-8")
    return html_path


def write_client_reports(run_dir: Path, records=None, *, budget: ThumbBudget = ThumbBudget(), workload: Optional[WorkloadAssumption] = None,
                         today: Optional[date] = None) -> Dict[str, Path]:
    """One report per client (client_id, else source_dataset, else "unassigned") plus `clients/index.json`.

    `records` is an image_id mapping or an iterable of ImageRecord; None reads `<run>/manifest.jsonl`."""
    run_dir = Path(run_dir)
    if records is None:
        manifest = run_dir / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} not found; pass the run's records instead")
        records = read_manifest(manifest)
    recs: Dict[str, ImageRecord] = dict(records) if isinstance(records, Mapping) else {r.image_id: r for r in records}
    run = load_run(run_dir)
    clients = group_records(recs)
    if any(_image_of(f) not in recs for f in run["findings"]):
        clients.setdefault("unassigned", [])
    paths: Dict[str, Path] = {}
    index = []
    for cid in clients:
        p = write_client_report(run_dir, cid, recs, run=run, budget=budget, workload=workload, today=today)
        paths[cid] = p
        m = json.loads((p.parent / "report.json").read_text(encoding="utf-8"))
        rel = p.parent.relative_to(run_dir).as_posix()
        index.append({"client_id": cid, "slug": m["slug"], "basis": m["grouping_basis"], "images": m["images"], "findings": m["findings"], "S4": m["levels"]["S4"],
                      "U": m["levels"]["U"], "usd": m["usd_total"], "html": f"{rel}/report.html", "md": f"{rel}/report.md", "json": f"{rel}/report.json"})
    (run_dir / "clients").mkdir(parents=True, exist_ok=True)
    (run_dir / "clients" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cascade.clientreport", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder, e.g. runs/ui_0925_0856")
    ap.add_argument("--manifest", help="manifest to group by; default <run>/manifest.jsonl")
    ap.add_argument("--today", help="YYYY-MM-DD used for due-in days; default today (UTC)")
    ap.add_argument("--max-thumbs", type=int, default=60)
    args = ap.parse_args(argv)
    records = read_manifest(Path(args.manifest)) if args.manifest else None
    paths = write_client_reports(Path(args.run), records, budget=ThumbBudget(max_thumbs=args.max_thumbs), today=date.fromisoformat(args.today) if args.today else None)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
