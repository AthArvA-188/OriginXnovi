"""FR-21 demo UI for the inspection grading cascade.

Top-level tabs: Inspect run (dataset or upload, live counters, image grid, findings with evidence,
review with crack measurement from an in-image scale, queue, surge, export) · Drop & grade (drag-and-drop
inference) · Sensors (seismic and vibration CSV: indicators, rubric grade, save as run) · Batch (several datasets in
one go) · Reports (stored per-run reports, compared visually) · Eval matrix (gate 2x2 and grading
confusion matrices against dataset labels) · Why this approach (sourced comparison with the
incumbents plus the numbers measured on the active run).

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, get_args

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from cascade import drift, workforce  # noqa: E402
from cascade.canary import cost_estimate  # noqa: E402
from cascade.clientreport import ACTION_COLOR, ACTION_ORDER, LEVEL_COLOR, LEVEL_MEANING, _workload_lines, worst_levels, write_client_reports  # noqa: E402
from cascade.evalmetrics import eval_matrix  # noqa: E402
from cascade.exemplars import exemplar_provider  # noqa: E402
from cascade.export import finding_row  # noqa: E402
from cascade.ingest import ingest_folder, read_manifest, write_manifest  # noqa: E402
from cascade.pipeline import LEVELS, Progress, RunConfig, load_run, run_cascade, save_findings  # noqa: E402
from cascade.prioritize import SEVERITY_WEIGHT, consequence_for, urgency_for  # noqa: E402
from cascade.report import REPORT_FIELDS, run_metrics, write_run_report  # noqa: E402
from cascade.review import ReviewLog  # noqa: E402
from cascade.measure import METHOD_FLOOR_MM, SCALE_CONFIDENCE, WIDTH_UNCERTAINTY_PX, Scale, crack_mask, measure_crack, scale_from_points, skeletonize, to_measurements  # noqa: E402
from cascade.schema import AssetClass, Finding, ImageRecord, Measurements, Sensor, SignalRecord  # noqa: E402
from cascade.signals import MOUNTS, append_signal_row, grade_signal, indicators, ingest_signal_csv, load_samples, load_seismic_rubric, write_signal_findings  # noqa: E402
from cascade.surge import surge_counts, write_surge_report  # noqa: E402
from cascade.video import VIDEO_EXTS, VideoError, ffmpeg_available, ingest_video, slug_of  # noqa: E402
from cascade.videodetect import LEVEL_ORDER, NOT_GRADED, NOT_GRADED_COLOR, cue_text, video_timeline, write_video_detections  # noqa: E402

RUNS = ROOT / "runs"
DEV_MANIFEST = ROOT / "data" / "dev" / "manifest.jsonl"
EVAL_MANIFEST = ROOT / "data" / "eval_v1" / "manifest.jsonl"
DEMO_DIR = ROOT / "data" / "demo"
DEMO_VIDEO = DEMO_DIR / "video" / "bridge_walkthrough.mp4"
FF_OK, FF_MSG = ffmpeg_available()
UPLOAD_TYPES = ["jpg", "jpeg", "png"] + (sorted(e.lstrip(".") for e in VIDEO_EXTS) if FF_OK else [])
UPLOAD_LABEL = "JPEG / PNG" + (" / MP4 / MOV / AVI / MKV" if FF_OK else "")
DEDUP_HELP = (
    "A frame whose dHash Hamming distance to the last kept frame is at or below this is dropped as a near-duplicate "
    "(0 = exact repeats only). Measured on data/demo/video/bridge_walkthrough.mp4 (12 s, 1280x960, ffmpeg 8.0.1): "
    "frames 2 s apart sit at distances 28, 29, 34, 35, 33; sampling every 1 s extracts 12 and drops 6 at distance 0."
)
BLOCKING_RULES = drift.BLOCKING_RULES  # contract_hard, abstention_collapse, moved_2plus, critical_miss; never demoted by the alert budget
NEUTRAL_BOX = "#9ca3af"  # bbox colour for blind QC: the level colour would leak the model's grade
STATUS_COLOR = {"in_control": "#22c55e", "watch": "#f59e0b", "alarm": "#dc2626", "insufficient_n": "#9ca3af", "candidate": "#3b82f6"}
DATASET_LABELS = {
    "mixed": "Mixed sample: all four asset classes",
    "corrosion_cs": "Steel coating: corrosion condition state (bridge steel)",
    "dacl10k": "Bridge elements: concrete defects (dacl10k)",
    "ir_solar": "PV thermal modules (InfraredSolarModules)",
    "rescuenet": "Post-disaster UAV (RescueNet), surge mode",
}
ASSET_CLASSES = ["bridge_element", "steel_coating", "pv_module", "building_disaster"]
ASSET_LABEL = {"bridge_element": "Bridge element (concrete)", "steel_coating": "Steel coating (corrosion)", "pv_module": "PV module (thermal)", "building_disaster": "Building (post-disaster)"}
ASSET_ICON = {"bridge_element": "\U0001F309", "steel_coating": "\U0001F529", "pv_module": "\u2600\ufe0f", "building_disaster": "\U0001F3DA\ufe0f"}
AMBER = "#f59e0b"
# Sensors tab (FR-24 UI). Uploads and the synthetic pair live under runs/_* so list_runs never shows them as runs.
SIG_DEMO_DIR = RUNS / "_signals_demo"
SIG_UPLOAD_DIR = RUNS / "_signals_uploads"
SIG_UNITS = ["(unknown)", "m/s2", "g", "mg", "gal", "cm/s2", "m/s", "cm/s", "mm/s"]
# Synthetic demo pair: a free decay plus a small steady tone plus gaussian noise. Every value is a demo design choice
# [Assumption], not a measurement: 3.2 Hz baseline, 2.9 Hz current (a -9.4 % shift), 200 Hz, 60 s, design damping 0.02.
SYNTH = {"baseline_hz": 3.2, "current_hz": 2.9, "rate_hz": 200.0, "duration_s": 60.0, "zeta": 0.02, "decay_amp": 0.2, "tone_amp": 0.005, "noise_sd": 0.002}
SYNTH_LABEL = "SYNTHETIC demo signal"
# PRD section 5A modalities, for the Architecture inputs lane. Status is what this repo has run, not the PRD target:
# no sonar, lidar or real seismic data has been run (sonar and machinery have rubric files only).
MODALITY_PILLS = [
    ("RGB drone and camera", "measured run"), ("Thermal heatmaps", "PV demo"), ("Sonar", "rubric only, not run"),
    ("Seismic and vibration", "synthetic only"), ("Lidar (roadmap)", "no code"), ("Interior machinery images", "rubric only, not run"),
]
CSS = """
<style>
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,.55); } 50% { box-shadow: 0 0 0 7px rgba(220,38,38,0); } }
div[data-testid="stImage"] { animation: fadeUp .45s ease both; }
div[data-testid="stImage"] img { border-radius: 10px; transition: transform .18s ease, box-shadow .18s ease; }
div[data-testid="stImage"] img:hover { transform: scale(1.03); box-shadow: 0 8px 24px rgba(0,0,0,.35); }
div[data-testid="stMetric"] { animation: fadeUp .4s ease both; border-radius: 10px; padding: 6px 10px; background: rgba(127,127,127,.07); }
.badge { display:inline-block; color:white; padding:1px 8px; border-radius:9px; font-weight:600; letter-spacing:.2px; transition: transform .15s ease; }
.badge:hover { transform: translateY(-1px); }
.badge-S4 { animation: pulse 1.6s ease-out infinite; }
.chip { display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:999px; background: rgba(59,130,246,.15); border:1px solid rgba(59,130,246,.35); font-size: .85em; }
button[kind="primary"] { transition: transform .12s ease, filter .12s ease; }
button[kind="primary"]:hover { transform: translateY(-1px); filter: brightness(1.08); }
div[data-testid="stMarkdownContainer"] table { animation: fadeUp .4s ease both; }
</style>
"""
st.set_page_config(page_title="Inspection grading cascade", layout="wide", page_icon="\U0001F50D")
st.markdown(CSS, unsafe_allow_html=True)


# ---------- helpers ----------


def list_manifests() -> dict:
    """label -> manifest path. Demo manifests first, then the dev set."""
    out = {}
    if DEMO_DIR.exists():
        for p in sorted(DEMO_DIR.glob("*/manifest.jsonl"), key=lambda q: (q.parent.name != "mixed", q.parent.name)):
            out[f"demo: {DATASET_LABELS.get(p.parent.name, p.parent.name)}"] = p
    if DEV_MANIFEST.exists():
        out["dev set (all four datasets, 50 images)"] = DEV_MANIFEST
    return out


def records_for(manifest: Path, dataset_filter, limit: int):
    recs = read_manifest(manifest)
    if dataset_filter:
        recs = [r for r in recs if r.source_dataset == dataset_filter]
    return recs[:limit] if limit else recs


def list_runs() -> list:
    if not RUNS.exists():
        return []
    return sorted([p.name for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "gate.jsonl").exists()], reverse=True)


def draw_evidence(f: Finding, rec: ImageRecord, neutral: bool = False) -> Image.Image:
    """Evidence image with the tile bbox outlined in the level colour; `neutral` draws it in NEUTRAL_BOX so a blind
    grader sees where to look but not what the model graded."""
    img = Image.open(rec.path).convert("RGB")
    if f.evidence.bbox and f.evidence.tile != "full":
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = f.evidence.bbox
        d.rectangle([x0, y0, x1, y1], outline=NEUTRAL_BOX if neutral else LEVEL_COLOR.get(f.unified.level, "#ffffff"), width=max(3, img.width // 300))
    if max(img.size) < 320:  # 24x40 thermal crops
        s = 320 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.NEAREST)
    return img


@st.cache_data(show_spinner=False)
def thumbnail(path: str, mtime: float, size: int = 320) -> Optional[Image.Image]:
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p).convert("RGB")
    if max(img.size) < 160:
        s = 160 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.NEAREST)
    img.thumbnail((size, size))
    return img


def badge(level: str, text: str = "") -> str:
    color = LEVEL_COLOR.get(level, "#555")
    return f"<span class='badge badge-{level}' style='background:{color}'>{level}</span> {text}"


def modality_chip(modality: Optional[str]) -> str:
    """A small chip naming a non-RGB modality (thermal, sonar, seismic, lidar); RGB, the default, shows nothing."""
    if not modality or modality == "rgb":
        return ""
    return f"<span class='chip'>{modality}</span> "


def amber(text: str) -> str:
    return f"<span style='color:{AMBER};font-weight:600'>{text}</span>"


def crack_width_text(m: Measurements) -> str:
    """Crack width as the UI must show it: "w +/- u mm (scale: basis)", never a bare millimetre value."""
    if m.crack_width_mm is None:
        return "crack width: none (not_measurable without a scale)"
    if m.crack_width_uncertainty_mm is not None and m.measurement_basis:
        return f"crack {m.crack_width_mm:.2f} ± {m.crack_width_uncertainty_mm:.2f} mm (scale: {m.measurement_basis})"
    return f"crack {m.crack_width_mm} mm (model estimate, no measured scale basis, no uncertainty)"


def gallery(records: List[ImageRecord], captions: Optional[Dict[str, str]] = None, levels: Optional[Dict[str, str]] = None, cols: int = 5, max_n: int = 30, key: str = "g"):
    """Thumbnail grid. `captions` and `levels` are keyed by image_id; a level renders as a colored badge."""
    if not records:
        st.info("No images.")
        return
    shown = records[:max_n]
    for start in range(0, len(shown), cols):
        row = st.columns(cols)
        for col, rec in zip(row, shown[start : start + cols]):
            with col:
                p = Path(rec.path)
                th = thumbnail(str(p), p.stat().st_mtime if p.exists() else 0.0)
                if th is None:
                    st.warning(f"{rec.image_id}: file missing")
                    continue
                st.image(th, width="stretch")
                lvl = (levels or {}).get(rec.image_id)
                cap = (captions or {}).get(rec.image_id, "")
                st.markdown((badge(lvl, "") if lvl else "") + modality_chip(getattr(rec, "modality", "rgb")) + f"<small>`{rec.image_id}`<br>{cap}</small>", unsafe_allow_html=True)
    if len(records) > max_n:
        st.caption(f"Showing {max_n} of {len(records)} images.")


def run_records(out: Path, imgs: Dict[str, ImageRecord]) -> Dict[str, ImageRecord]:
    """Only the records the run gated. `load_records_for_run` also merges the dev and eval manifests as a path
    fallback, which would inflate any per-run count (input stats, client buckets, workforce N)."""
    gp = out / "gate.jsonl"
    if not gp.exists():
        return imgs
    ids = [json.loads(line)["image_id"] for line in gp.read_text(encoding="utf-8").splitlines() if line.strip()]
    sub = {i: imgs[i] for i in ids if i in imgs}
    return sub or imgs


def video_options(key: str) -> dict:
    """Collapsed 'Video options' expander; returns the extraction kwargs for `ingest_video`."""
    with st.expander("Video options", expanded=False):
        if not FF_OK:
            st.caption(f"Video upload off: {FF_MSG}")
        every = st.number_input("Sample every N seconds", 0.5, 30.0, 2.0, 0.5, key=f"{key}_every")
        scene = st.checkbox("Scene-change mode: keep frames where the picture changes", value=False, key=f"{key}_scene")
        thr = st.slider("Scene threshold", 0.1, 0.9, 0.3, 0.05, key=f"{key}_thr", disabled=not scene)
        gap = st.number_input("Scene mode: longest gap without a frame, s", 1.0, 60.0, 10.0, 1.0, key=f"{key}_gap", disabled=not scene)
        max_frames = st.number_input("Max frames per video", 10, 1000, 200, 10, key=f"{key}_max")
        dedup = st.slider("Near-duplicate threshold (dHash distance)", 0, 12, 3, key=f"{key}_dedup", help=DEDUP_HELP)
    return {
        "every_s": float(every),
        "scene_threshold": float(thr) if scene else None,
        "scene_max_gap_s": float(gap) if scene else None,
        "max_frames": int(max_frames),
        "dedup_max_distance": int(dedup),
    }


def ingest_uploads(files, base: Path, asset_class: str, client_id: Optional[str], id_prefix: str, vopts: dict) -> Tuple[List[ImageRecord], List[dict], str]:
    """Write uploads under base/images, base/video/<file> and base/frames/<slug>/ (frames never sit inside the
    folder `ingest_folder` rglobs), ingest stills and videos, return (records, extraction dicts, summary line)."""
    img_dir, vid_dir = base / "images", base / "video"
    img_dir.mkdir(parents=True, exist_ok=True)
    videos: List[Path] = []
    for fobj in files:
        if Path(fobj.name).suffix.lower() in VIDEO_EXTS:
            vid_dir.mkdir(parents=True, exist_ok=True)
            (vid_dir / fobj.name).write_bytes(fobj.getbuffer())
            videos.append(vid_dir / fobj.name)
        else:
            (img_dir / fobj.name).write_bytes(fobj.getbuffer())
    records: List[ImageRecord] = []
    if any(img_dir.iterdir()):
        records = ingest_folder(img_dir, asset_class=asset_class, source_dataset="upload", split="upload", id_prefix=id_prefix, client_id=client_id or None)
    n_img = len(records)
    exts: List[dict] = []
    for vp in videos:
        try:
            recs, ext = ingest_video(vp, base / "frames" / slug_of(vp), asset_class=asset_class, client_id=client_id or None, captured_on_from_mtime=False, source_dataset="upload", split="upload", **vopts)
        except VideoError as e:
            st.error(str(e))
            continue
        records += recs
        exts.append(ext.to_json())
    n_kept, n_dropped = sum(e["n_kept"] for e in exts), sum(e["n_dropped"] for e in exts)
    line = f"{n_img} images ingested; hash, size and EXIF date recorded, missing metadata stays null"
    if videos:
        line = f"{n_img} images + {n_kept} frames from {len(exts)} videos ingested ({n_dropped} near-duplicate frames dropped, dHash ≤ {vopts['dedup_max_distance']})"
    return records, exts, line


def video_strip(ext: dict, records: List[ImageRecord], key: str) -> None:
    """Header measured from frames.json, kept frames with timestamps, dropped frames in an expander."""
    v = ext["video"]
    name = Path(v["path"]).name
    mode = f"every {ext['every_s']:.1f} s" if ext["mode"] == "interval" else f"scene threshold {ext['scene_threshold']}"
    st.markdown(
        f"**{name}** · {v['duration_s']:.1f} s · {v['width']}×{v['height']} @ {v['fps']:.2f} fps · {mode} · extracted {ext['n_extracted']}, kept {ext['n_kept']}, "
        f"dropped {ext['n_dropped']} near-duplicates (dHash ≤ {ext['dedup_max_distance']}) · ffmpeg {ext['seconds_ffmpeg']:.1f} s, dedup {ext['seconds_dedup']:.2f} s"
    )
    if ext.get("truncated"):
        st.warning(f"Max frames ({ext['max_frames']}) reached: only the first {max(f['t_s'] for f in ext['frames']):.0f} s of {v['duration_s']:.1f} s were sampled.")
    recs = [r for r in records if r.source_video and Path(r.source_video).name == name]
    gallery(recs, captions={r.image_id: f"t = {r.frame_time_s:.1f} s" for r in recs}, cols=8, max_n=24, key=key)
    dropped = [f for f in ext["frames"] if not f["kept"]]
    if dropped:
        with st.expander(f"Dropped as near-duplicates ({len(dropped)})"):
            st.table(pd.DataFrame([{"t_s": f["t_s"], "distance to last kept frame": f["distance"]} for f in dropped]))


def stored_extractions(out: Path) -> List[dict]:
    """frames.json copies under runs/<run>/videos/, so the strip survives a restart."""
    if not (out / "videos").exists():
        return []
    sidecars = [p for p in sorted((out / "videos").glob("*.json")) if not p.name.endswith("_timeline.json")]  # <slug>_timeline.json is videodetect output, not frames.json
    return [json.loads(p.read_text(encoding="utf-8")) for p in sidecars]


def stored_timelines(out: Path, imgs: Dict[str, ImageRecord]) -> Dict[str, dict]:
    """slug -> timeline entry. runs/<run>/videos/<slug>_timeline.json (written by write_video_detections) is used
    only while it is at least as new as findings.json; after a review changed a level the timeline is recomputed
    with video_timeline (cheap: reads manifest, gate and findings; no model calls), the stored file being the
    fallback when the run folder cannot be read."""
    vdir = out / "videos"
    fj = out / "findings.json"
    newest = fj.stat().st_mtime if fj.exists() else 0.0
    found: Dict[str, dict] = {}
    stale = False
    for p in sorted(vdir.glob("*_timeline.json")) if vdir.exists() else []:
        try:
            found[p.name[: -len("_timeline.json")]] = entry = json.loads(p.read_text(encoding="utf-8"))
            stale = stale or p.stat().st_mtime < newest or "mp4_offset_s" not in entry  # older entries lack mode/clip_end_s/boxes
        except (OSError, json.JSONDecodeError):
            continue
    if found and not stale:
        return found
    try:
        return video_timeline(out, run_records(out, imgs)) or found
    except Exception as e:
        st.warning(f"Video timeline not {'refreshed' if found else 'computed'}: {type(e).__name__}: {e}")
        return found


def refresh_video_detections(out: Path, imgs: Dict[str, ImageRecord]) -> None:
    """After a review changed a level: rewrite <slug>_timeline.json, the .vtt, the contact sheet and the detections
    mp4 so every artifact carries the reviewed level. No model calls; fails soft; a run without video frames returns
    at once."""
    recs = run_records(out, imgs)
    if not any(r.source_video for r in recs.values()):
        return
    try:
        with st.spinner("Re-rendering the video detections with the reviewed level..."):
            vd = write_video_detections(out, recs)
        for slug_v, item in vd.items():
            if item.get("skipped"):
                st.warning(f"{slug_v}: {item['skipped']}")
    except Exception as e:
        st.warning(f"Video detections not refreshed: {type(e).__name__}: {e}")


def video_sampling_label(entry: dict) -> str:
    """How the kept frames were chosen, from the timeline entry (mode from frames.json), the way video_strip words
    it: interval mode "every 2.0 s", scene mode its threshold and max gap, no frames.json the measured gap."""
    mode, src = entry.get("mode"), entry.get("every_s_source", "frames.json")
    if mode == "scene":
        gap = f", max gap {entry['scene_max_gap_s']:.1f} s" if entry.get("scene_max_gap_s") else ""
        return f"scene mode (threshold {entry.get('scene_threshold')}{gap})"
    if mode == "interval" or src == "frames.json":
        return f"every {entry['every_s']:.1f} s ({src})"
    return f"frame gap {entry['every_s']:.1f} s ({src})"


def video_frame_caption(fr: dict) -> str:
    """"t = 4.0 s - S2 - CS3"; U frames say "unassessable"; a frame the run never graded says "not graded", never a level."""
    lvl = fr.get("level")
    if lvl is None:
        return f"t = {fr['t_s']:.1f} s - {NOT_GRADED}"
    if lvl == "U":
        return f"t = {fr['t_s']:.1f} s - U - unassessable"
    return f"t = {fr['t_s']:.1f} s - {lvl} - {fr.get('native_value')}"


def video_timeline_chart(entry: dict) -> alt.Chart:
    """A tick per graded instant (each kept frame's t_s, coloured by its level, grey for not graded) over one
    translucent rect per segment (x from start_s to end_s, y the level). The ticks are where grades exist; a bar
    only shows how long that frame is held until the next kept sample."""
    frames = entry["frames"]
    rows = []
    for seg in entry["segments"]:
        ids = [f["image_id"] for f in frames if seg["start_s"] <= f["t_s"] < seg["end_s"]]
        rows.append({
            "start_s": seg["start_s"], "end_s": seg["end_s"], "level": seg["level"] or NOT_GRADED, "native": seg.get("native_value") or "",
            "action": seg.get("action") or "", "frames": seg["n_frames"], "image_id": ", ".join(ids), "cue": cue_text(seg),
        })
    ticks = [{"t_s": f["t_s"], "level": f["level"] or NOT_GRADED, "native": f.get("native_value") or "", "image_id": f["image_id"]} for f in frames]
    domain = LEVEL_ORDER + [NOT_GRADED]
    colors = [LEVEL_COLOR[lvl] for lvl in LEVEL_ORDER] + [NOT_GRADED_COLOR]
    x_min = min(min(r["start_s"] for r in rows), min(t["t_s"] for t in ticks)) if rows else 0.0
    x_max = max(r["end_s"] for r in rows) if rows else 1.0
    y = alt.Y("level:O", sort=domain, title=None, scale=alt.Scale(domain=domain))
    color = alt.Color("level:N", scale=alt.Scale(domain=domain, range=colors), legend=None)
    bars = (
        alt.Chart(pd.DataFrame(rows))
        .mark_rect(cornerRadius=3, opacity=0.4)
        .encode(
            x=alt.X("start_s:Q", title="t (s), source clip", scale=alt.Scale(domain=[x_min, x_max], nice=False)),
            x2="end_s:Q", y=y, color=color,
            tooltip=["start_s", "end_s", "level", "native", "action", "frames", "image_id"],
        )
    )
    marks = alt.Chart(pd.DataFrame(ticks)).mark_tick(thickness=3, size=22).encode(x="t_s:Q", y=y, color=color, tooltip=["t_s", "level", "native", "image_id"])
    return (bars + marks).properties(height=30 * len(domain) + 40)


def video_detections_section(out: Path, imgs: Dict[str, ImageRecord], key: str) -> None:
    """FR-4b per video of the run: header, the detections mp4 with WebVTT cues (contact sheet when the mp4 is missing),
    the level timeline, the segments table, the frame gallery and download buttons. Level null is "not graded"; U is U."""
    timelines = stored_timelines(out, imgs)
    if not timelines:
        return
    st.subheader("Video detections")
    st.caption("Frames sampled by ffmpeg were graded like stills. A detections mp4 is built from the N kept sample frames, each held until the next sample, with the level badge, the tile boxes and a subtitle cue per segment; the source frames between samples were not graded. A frame the gate cleared without a heavy grade is 'not graded', never S0; U is unassessable and counted on its own.")
    vdir = out / "videos"
    fj = out / "findings.json"
    for slug, entry in timelines.items():
        s = entry["summary"]
        dur = f"{entry['duration_s']:.1f} s" if entry.get("duration_s") is not None else "duration not probed"
        size = f" · {entry['width']}×{entry['height']} @ {entry['fps']:.1f} fps" if entry.get("width") and entry.get("fps") else ""
        first = f"first detection at {s['first_detection_t_s']:.1f} s" if s.get("first_detection_t_s") is not None else "no S1..S4 detection"
        worst = badge(s["worst_level"], "worst level") if s.get("worst_level") else f"<span class='badge' style='background:{NOT_GRADED_COLOR}'>{NOT_GRADED}</span>"
        offset = float(entry.get("mp4_offset_s") or 0.0)
        offset_note = f" · mp4 time 0 = source t = {offset:.1f} s" if offset > 0 else ""
        st.markdown(
            f"**{Path(entry['video']).name}** · {dur}{size} · {video_sampling_label(entry)} · "
            f"{s['graded']} of {s['frames']} kept frames graded ({s['not_graded']} not graded, U {s['u_frames']}) · {worst} · {first}{offset_note}",
            unsafe_allow_html=True,
        )
        mp4, vtt, tjson, sheet = (vdir / f"{slug}_detections.mp4", vdir / f"{slug}_detections.vtt", vdir / f"{slug}_timeline.json", vdir / f"{slug}_contact.png")
        if fj.exists() and any(p.exists() and p.stat().st_mtime < fj.stat().st_mtime for p in (mp4, vtt, sheet)):
            st.warning(f"{mp4.name}, {vtt.name} and {sheet.name} were rendered before the last change to findings.json (a review); the timeline, table and badges below are current. Regenerate them with `python -m cascade.videodetect --run runs/{out.name}`.")
        v1, v2 = st.columns([3, 2])
        with v1:
            if mp4.exists():
                st.video(str(mp4), subtitles=str(vtt) if vtt.exists() else None)
                st.caption(f"{mp4.name} · {mp4.stat().st_size / 1e6:.2f} MB · {s['frames']} kept frames, each held until the next · subtitles {vtt.name if vtt.exists() else 'none'}")
            elif sheet.exists():
                st.image(str(sheet), caption=f"{sheet.name} (no detections mp4 for this run: {FF_MSG if not FF_OK else 'not written yet'})", width="stretch")
            else:
                st.info("No detections mp4 or contact sheet stored for this video yet." + ("" if FF_OK else f" {FF_MSG}"))
        with v2:
            st.altair_chart(video_timeline_chart(entry), width="stretch")
            st.caption("Grades exist only at the ticks (one per kept frame); a bar shows the hold until the next sample, not that the frames in between were graded.")
            st.table(pd.DataFrame([{"start_s": g["start_s"], "end_s": g["end_s"], "level": g["level"] or NOT_GRADED, "native": g.get("native_value") or "", "action": g.get("action") or "", "frames": g["n_frames"]} for g in entry["segments"]]))
        frame_recs = [imgs[f["image_id"]] for f in entry["frames"] if f["image_id"] in imgs]
        caps = {f["image_id"]: video_frame_caption(f) for f in entry["frames"]}
        lvls = {f["image_id"]: f["level"] for f in entry["frames"] if f.get("level")}  # ungraded frames get no badge
        gallery(frame_recs, captions=caps, levels=lvls, cols=6, max_n=24, key=f"{key}_{slug}_frames")
        dl = st.columns(4)
        for col, fp, label, mime in zip(dl, (mp4, vtt, tjson, sheet), ("detections.mp4", "detections.vtt", "timeline.json", "contact.png"), ("video/mp4", "text/vtt", "application/json", "image/png")):
            if fp.exists():
                col.download_button(f"Download {label}", fp.read_bytes(), file_name=f"{out.name}_{slug}_{label}", mime=mime, key=f"{key}_{slug}_dl_{label.replace('.', '_')}", width="stretch")
            else:
                col.caption(f"{label}: not written")


def write_reports(out: Path, imgs: Dict[str, ImageRecord]) -> Path:
    """report.md/report.json from report.py, then the workforce section appended (laborSpec 5.8), one client
    report per client_id (clientReportSpec 11) and health.json (driftSpec). Each add-on fails soft."""
    rp = write_run_report(out, imgs)
    recs = run_records(out, imgs)
    try:
        est = workforce.estimate(workforce.measured_inputs(out, records=recs))
        with rp.open("a", encoding="utf-8") as fh:
            fh.write("\n\n" + workforce.render_markdown(est))
        rj = out / "report.json"
        doc = json.loads(rj.read_text(encoding="utf-8"))
        doc["workforce"] = workforce.to_json(est)
        rj.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    except Exception as e:
        st.warning(f"Workforce section skipped: {type(e).__name__}: {e}")
    try:
        write_client_reports(out, recs)
    except Exception as e:
        st.warning(f"Client reports skipped: {type(e).__name__}: {e}")
    try:
        drift.health(out, records=recs)
    except Exception as e:
        st.warning(f"health.json skipped: {type(e).__name__}: {e}")
    return rp


def health_for(out: Path, imgs: Dict[str, ImageRecord], refresh: bool = False) -> Optional[dict]:
    """runs/<run>/health.json, recomputed (zero model calls) when missing, stale or asked for."""
    hp = out / "health.json"
    newest = max((p.stat().st_mtime for p in [out / "findings.json", out / "gate.jsonl", out / "calls.jsonl", out / "reviews.sqlite"] if p.exists()), default=0.0)
    if hp.exists() and not refresh and hp.stat().st_mtime >= newest:
        try:
            return json.loads(hp.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        return drift.health(out, records=run_records(out, imgs))
    except Exception as e:
        st.warning(f"Model health not computed: {type(e).__name__}: {e}")
        return None


def status_badge(status: Optional[str], text: str = "") -> str:
    status = status or "n/a"
    return f"<span class='badge' style='background:{STATUS_COLOR.get(status, '#555')}'>{status.replace('_', ' ')}</span> {text}"


def agreement_chart(rows: List[dict]) -> alt.Chart:
    """FR-19 running agreement line; tooltip uses whichever columns the rows carry."""
    tdf = pd.DataFrame(rows)
    tips = [c for c in ["n", "reviewed_at", "finding_id", "action", "prior_level", "new_level", "reviewer", "agreement_rate"] if c in tdf.columns]
    return (
        alt.Chart(tdf)
        .mark_line(point=True, interpolate="monotone")
        .encode(x=alt.X("n:Q", title="decision #"), y=alt.Y("agreement_rate:Q", title="running agreement", scale=alt.Scale(domain=[0, 1])), tooltip=tips)
        .properties(height=200)
    )


def est_note(est: workforce.WorkforceEstimate, key: str) -> str:
    """Mirrors workforce._est_label: a measured input (m_rev after ten review gaps) is not an assumption."""
    uses = [k for k in est.uses.get(key, []) if est.assumptions[k].kind != "measured"]
    return f"estimate (uses {len(uses)} assumption{'s' if len(uses) != 1 else ''}: {', '.join(uses)})" if uses else f"Measured (run {est.inputs.run})"


def workforce_panel(out: Path, imgs: Dict[str, ImageRecord], key: str, client_id: Optional[str] = None) -> None:
    """laborSpec 4 and 5: measured counts on the left, estimates on the right, four sliders, a tornado chart.
    Every figure carries its badge; hours, percent and dollars are always labelled estimate."""
    st.markdown(f"*{workforce.DISCLAIMER}*")
    s1, s2, s3, s4 = st.columns(4)
    m_man = s1.slider("Minutes per image, manual review today", 0.05, 6.0, 3.0, 0.05, key=f"{key}_m_man", help="Public figure: T&D World, 3 to 5 min per image of utility T&D imagery; not a bridge or solar figure. Ticks: 0.067 AEP Ohio skim (derived from a public figure), 3 and 5 analysis.")
    m_rev = s2.slider("Minutes per routed image with the cascade", 0.25, 5.0, 1.0, 0.25, key=f"{key}_m_rev", help="Team assumption: the reviewer sees the evidence crop, the pre-filled native grade and the quoted rubric row, then accepts / overrides / marks U. Replaced by the measured review-gap median x findings per image once 10 gaps are logged.")
    a_pct = s3.slider("Audit sample rate on auto-cleared images, %", 0, 50, 10, 1, key=f"{key}_a", help="Team assumption: random audit of the images the gate cleared without a heavy grade.")
    w = s4.slider("Loaded hourly rate, USD", 40.0, 250.0, 113.03, 1.0, key=f"{key}_w", help="Public figure: WSDOT bridge inspector $113.03/h (one state's undated rate); co-inspector 94.43, report writing 112. Moves dollars only.")
    ov: Dict[str, float] = {"m_man": m_man, "m_rev": m_rev, "a": a_pct / 100.0, "w": w}
    with st.expander("Advanced assumptions (all Team assumption)", expanded=False):
        ov["H"] = float(st.slider("Hours per inspector-week", 20, 60, 40, key=f"{key}_H"))
        same = st.checkbox("Minutes per audited image = manual minutes", value=True, key=f"{key}_same")
        if not same:
            ov["m_aud"] = st.slider("Minutes per audited image", 0.05, 6.0, 3.0, 0.05, key=f"{key}_m_aud")
        ov["m_xf"] = st.slider("Minutes per additional finding on the same image", 0.0, 3.0, 0.0, 0.25, key=f"{key}_m_xf")
        ov["gap_cap"] = float(st.slider("Review break cap, min (longer gaps between decisions are breaks)", 5, 30, 10, key=f"{key}_gap"))
    try:
        inp = workforce.measured_inputs(out, records=run_records(out, imgs), client_id=client_id, gap_cap_min=ov["gap_cap"])
        est = workforce.estimate(inp, ov)
    except Exception as e:
        st.info(f"No workforce estimate for this run: {type(e).__name__}: {e}")
        return
    if est.m_rev_measured is not None:
        s2.caption(f"median {inp.gap_median_min:.2f} min between decisions × {est.fpi:.2f} findings per image = {est.m_rev_measured:.2f} min · Measured (run {inp.run}), n = {inp.gap_n} gaps; indicative only")
        if s2.checkbox("Use the measured value instead of the slider", value=False, key=f"{key}_usem"):
            ov.pop("m_rev", None)
            ov["use_measured_m_rev"] = 1
            est = workforce.estimate(inp, ov)
    else:
        s2.caption(f"Team assumption; {inp.gap_n} of 10 review gaps logged")

    st.markdown("**Estimated with the assumptions above** · per 1,000 images at this run's routing fraction, then this run")
    k = st.columns(6)
    k[0].metric("Manual review h / 1,000 (estimate)", f"≈ {est.manual_h_1000:.1f}", help=est_note(est, "manual_h_1000"))
    k[1].metric("Cascade review h / 1,000 (estimate)", f"≈ {est.cascade_h_1000:.1f}", help=est_note(est, "cascade_h_1000"))
    k[2].metric("Hours freed / 1,000 (estimate)", f"≈ {est.saved_h_1000:+.1f}", help=est_note(est, "saved_h_1000"))
    k[3].metric("Percent saved, this run (estimate)", f"≈ {est.saved_pct:+.1f}%" if est.saved_pct is not None else "n/a", help=est_note(est, "saved_pct"))
    k[4].metric("Inspector-equivalents of review time freed / 1,000 images / week (estimate)", f"≈ {est.ie_per_1000_week:.2f}", help=est_note(est, "ie_per_1000_week"))
    k[5].metric("Cost per image, human vs cascade (estimate)", f"≈ ${est.cost_img_human:.2f} vs ${est.cost_img_cascade:.2f}", help=est_note(est, "cost_img_cascade") + " · " + workforce.COST_SCOPE)
    st.caption(f"*Every figure above is an estimate: this run's measured counts multiplied by the assumptions. Routing fraction r = {est.routing_fraction:.2f} (measured on {', '.join(inp.source_datasets) or 'uploads'}, not a fleet number).*" if est.routing_fraction is not None else "*Every figure above is an estimate.*")
    if est.saved_pct is not None and est.saved_pct < 0:
        st.markdown(f"<span style='color:#dc2626'><b>{est.saved_pct:+.1f}%</b> {workforce.NEGATIVE_SENTENCE}</span>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    badge_m = workforce.badge_text("measured", inp.run)
    gap = f"{inp.gap_median_min:.2f} min / {inp.gap_n}" if inp.gap_median_min is not None else f"n/a / {inp.gap_n}"
    measured_rows = [
        ("Images (N)", inp.images, "summary.json"), ("Gated (G)", inp.gated, "gate.jsonl"), ("Routed to grader (R)", inp.routed, "gate.jsonl"),
        ("Auto-cleared (A = G - R)", est.auto_cleared, "gate.jsonl"), ("Unusable (UN)", inp.unusable, "gate.jsonl"), ("Graded images (GI)", inp.graded_images, "findings.json"),
        ("Findings (F)", inp.findings, "findings.json"), ("U findings (never counted as S0)", inp.u_findings, "findings.json"), ("API dollars, all stages (C)", f"${inp.usd_total:.4f}", "calls.jsonl"),
        ("Model seconds per image", f"{est.model_s_per_image:.1f} s" if est.model_s_per_image is not None else "n/a", "calls.jsonl"), ("Reviewer decisions (V)", inp.reviews, "findings.json"),
        ("Review gap median / gaps kept", gap, "reviews.sqlite"), ("Routing fraction (r = R / G)", f"{est.routing_fraction:.2f}" if est.routing_fraction is not None else "n/a", "gate.jsonl"),
    ]
    with c1:
        st.markdown("**Measured from this run**")
        st.markdown("\n".join(["| Quantity | Value | Badge | File |", "|---|---|---|---|"] + [f"| {q} | {v} | {badge_m} | `{src}` |" for q, v, src in measured_rows]))
    est_rows = [
        ("Manual review minutes, this run", "manual_min", f"{est.manual_min:.1f} min"), ("Cascade review minutes, this run", "cascade_min", f"{est.cascade_min:.1f} min"),
        ("Minutes saved, this run", "saved_min", f"{est.saved_min:+.1f} min"), ("Percent saved, this run", "saved_pct", f"{est.saved_pct:+.1f}%" if est.saved_pct is not None else "n/a"),
        ("Break-even manual minutes per image", "break_even_m_man", f"{est.break_even_m_man:.2f} min" if est.break_even_m_man is not None else "none (no pre-filled grade in this run)"),
        ("Manual hours per 1,000", "manual_h_1000", f"{est.manual_h_1000:.1f} h"), ("Cascade hours per 1,000", "cascade_h_1000", f"{est.cascade_h_1000:.1f} h"), ("Hours freed per 1,000", "saved_h_1000", f"{est.saved_h_1000:+.1f} h"),
        ("Percent saved per 1,000", "saved_pct_1000", f"{est.saved_pct_1000:+.1f}%" if est.saved_pct_1000 is not None else "n/a"), ("Inspector-equivalents freed per 1,000 per week", "ie_per_1000_week", f"{est.ie_per_1000_week:.3f}"),
        ("Cost per image, human", "cost_img_human", f"${est.cost_img_human:.2f}"), ("Cost per image, cascade API (measured)", "cost_img_cascade_api", f"${est.cost_img_cascade_api:.4f}"),
        ("Cost per image, cascade human", "cost_img_cascade_hum", f"${est.cost_img_cascade_hum:.2f}"), ("Cost per image, cascade total", "cost_img_cascade", f"${est.cost_img_cascade:.2f}"),
    ]
    with c2:
        st.markdown("**Estimated with the assumptions above**")
        lines = ["| Quantity | Value | Label |", "|---|---|---|"]
        for q, k_, v in est_rows:
            uses = est.uses.get(k_, [])
            lines.append(f"| {q} | {'≈ ' + v if uses else v} | {'*' + est_note(est, k_) + '*' if uses else est_note(est, k_)} |")
        st.markdown("\n".join(lines))
    a_1000 = 1000 - int(round(1000 * (est.routing_fraction or 0)))
    bound = workforce.rule_of_three_bound(est.audit_1000)
    st.caption(f"Run figure audits {est.audit_n} of {est.auto_cleared} auto-cleared images (rounded up to at least 1); at 1,000 images the same settings audit {est.audit_1000} of {a_1000}."
               + (f" If that audit finds no miss, the 95% upper bound on the miss rate among the auto-cleared images is about {bound:.1%} (rule of three, Hanley and Lippman-Hand 1983); it applies to this run's cleared population only, after the audit is done and logged." if bound else ""))
    st.markdown("**Assumptions used**  " + " · ".join(f"`{k_}` = {asm.value:g} {asm.unit} ({workforce.badge_text(asm.kind, inp.run)})" for k_, asm in est.assumptions.items() if k_ in ("m_man", "m_rev", "m_aud", "a", "w", "H", "m_xf")))
    for wmsg in est.warnings:
        st.caption(f"⚠ {wmsg}")
    st.caption(workforce.COST_SCOPE)
    trows = workforce.tornado(inp, ov)
    tdf = pd.DataFrame(trows)
    tdf["setting"] = tdf.apply(lambda r: f"{r['lever']} = {r['value']:.3g} ({'what-if on a cleaner fleet' if r['hypothetical'] else r['label']})" if r["value"] is not None else f"{r['lever']} (not measurable)", axis=1)
    tdf["kind"] = tdf["hypothetical"].map({True: "what-if on a cleaner fleet", False: "one assumption changed"})
    tornado_chart = (
        alt.Chart(tdf.dropna(subset=["saved_pct_1000"]))
        .mark_bar()
        .encode(
            y=alt.Y("setting:N", sort=None, title=None),
            x=alt.X("saved_pct_1000:Q", title="percent of review time saved per 1,000 images (estimate)"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["one assumption changed", "what-if on a cleaner fleet"], range=["#3b82f6", "#9ca3af"]), title=None),
            tooltip=["lever", "value", "label", "saved_pct_1000", "ie_per_1000_week"],
        )
        .properties(height=360, title="Sensitivity, one lever at a time (estimate)")
    )
    st.altair_chart(tornado_chart, width="stretch")
    st.markdown(f"*{workforce.FOOTER}*")


def multipliers(f: Finding, rec) -> dict:
    return {
        "severity_weight": SEVERITY_WEIGHT.get(f.unified.level),
        "criticality": 1.0,
        "consequence": consequence_for(f),
        "urgency": urgency_for(rec.captured_on if rec else None, date.today()),
    }


def render_counters(box, p: Progress, cfg: RunConfig):
    with box.container():
        c = st.columns(7)
        c[0].metric("Images", p.images)
        c[1].metric("Gated", p.gated, help="stage A ran on this many images")
        c[2].metric("Routed", p.routed, help="sent to the heavy grader: damage, unusable, low-confidence clean, or forced asset class")
        c[3].metric("Graded", p.graded)
        c[4].metric("Findings", p.findings)
        c[5].metric("Cost USD", f"{p.usd:.3f}", help="API list price from the call log; local models cost $0")
        c[6].metric("Model seconds", f"{p.seconds_gate + p.seconds_grade:.0f}", help=f"gate {p.seconds_gate:.0f} s, grade {p.seconds_grade:.0f} s")
        st.caption(
            f"stage **{p.stage}** · image `{p.current_image}` · gate={cfg.gate} grader={cfg.grader} tiles={cfg.tiles} · levels: "
            + "  ".join(f"{k} {v}" for k, v in p.levels.items())
        )


def findings_df(findings) -> pd.DataFrame:
    cols = ["queue_rank", "queue_score", "image_id", "unified_level", "native_value", "standard", "defect_type", "action_code", "sla_days", "confidence", "flags", "review_status", "finding_id"]
    rows = [finding_row(f) for f in findings]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def load_records_for_run(out: Path) -> dict:
    """image_id -> ImageRecord from the run's own manifest, else the dev and eval manifests."""
    recs = []
    for mp in [out / "manifest.jsonl", DEV_MANIFEST, EVAL_MANIFEST]:
        if mp.exists():
            recs += read_manifest(mp)
    return {r.image_id: r for r in recs}


def levels_chart(df: pd.DataFrame, x: str) -> alt.Chart:
    long = df.melt(id_vars=[x], value_vars=list(LEVELS), var_name="level", value_name="count")
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title=None),
            y=alt.Y("count:Q", title="findings"),
            color=alt.Color("level:N", scale=alt.Scale(domain=list(LEVEL_COLOR), range=list(LEVEL_COLOR.values())), sort=list(LEVELS)),
            order=alt.Order("level:N"),
            tooltip=[x, "level", "count"],
        )
        .properties(height=260)
    )


def confusion_chart(labels: List[str], matrix: List[List[int]], title: str) -> alt.Chart:
    rows = [{"truth": t, "predicted": p, "count": matrix[i][j]} for i, t in enumerate(labels) for j, p in enumerate(labels)]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=labels, title="predicted (worst finding per image)"),
        y=alt.Y("truth:N", sort=labels, title="dataset truth"),
    )
    heat = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None), tooltip=["truth", "predicted", "count"])
    text = base.mark_text(fontSize=18, fontWeight="bold").encode(text="count:Q", color=alt.condition("datum.count > 0", alt.value("black"), alt.value("#999")))
    return (heat + text).properties(title=title, height=max(240, 70 * len(labels) + 80), width=max(300, 90 * len(labels) + 120))


def run_with_progress(records: List[ImageRecord], out: Path, cfg: RunConfig, use_exemplars: bool, surge: bool, extractions: Optional[List[dict]] = None) -> Optional[dict]:
    """Run the cascade with live counters in the current container. Returns the summary or None on error.
    `extractions` (frames.json dicts) are copied into runs/<name>/videos/<slug>.json so the run is self-contained."""
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(records, out / "manifest.jsonl")
    for ext in extractions or []:
        src = Path(ext["out_dir"]) / "frames.json"
        if src.exists():
            (out / "videos").mkdir(exist_ok=True)
            shutil.copyfile(src, out / "videos" / f"{slug_of(Path(ext['video']['path']))}.json")
    counters = st.empty()
    bar = st.progress(0.0, text="starting")
    log_box = st.empty()
    lines: List[str] = []
    n_total = max(1, len(records))

    def on_progress(p: Progress):
        render_counters(counters, p, cfg)
        done = min(n_total, p.gated if p.stage == "gate" else max(p.gated, p.graded))
        bar.progress(min(0.99, done / n_total), text=f"{p.stage} - {p.current_image} - {done}/{n_total} images - ${p.usd:.3f}")
        entry = f"{p.current_image} -> {p.stage}"
        if p.current_image and (not lines or lines[-1] != entry):
            lines.append(entry)
            log_box.code("\n".join(lines[-12:]))

    provider = exemplar_provider(DEV_MANIFEST, k=3) if use_exemplars and DEV_MANIFEST.exists() else None
    with st.spinner("Running. A cold local gate can take a minute; each heavy grade is 10 to 40 s."):
        try:
            summary = run_cascade(records, out, cfg, exemplars=provider, progress=on_progress)
            if surge:
                write_surge_report(surge_counts(load_run(out)["findings"]), summary, out)
            write_reports(out, {r.image_id: r for r in records})
            if any(r.source_video for r in records):  # FR-4b: a detections mp4 per clip from its kept sample frames; fails soft, no model calls
                try:
                    vd = write_video_detections(out, {r.image_id: r for r in records})
                    for slug_v, item in vd.items():
                        if item.get("mp4") is not None:
                            st.toast(f"{slug_v}_detections.mp4 written ({item['summary']['graded']} of {item['summary']['frames']} frames graded)", icon="\U0001F3AC")
                        elif item.get("skipped"):
                            st.warning(f"{slug_v}: {item['skipped']}")
                except VideoError as e:
                    st.warning(f"Video detections skipped: {e}")
                except Exception as e:
                    st.warning(f"Video detections skipped: {type(e).__name__}: {e}")
            bar.progress(1.0, text="done")
            st.toast(f"Run {out.name} finished: {summary['findings']} findings", icon="\u2705")
            st.success(f"Done: {summary['findings']} findings from {summary['images']} images, ${summary['usd_total']} total, routed {summary['routed_to_grader']} of {summary['gated']}. Report stored in runs/{out.name}/report.md.")
            return summary
        except Exception as e:  # partial outputs stay on disk; the run is resumable
            st.toast(f"Run {out.name} stopped", icon="\u26a0\ufe0f")
            st.error(f"Run stopped: {type(e).__name__}: {e}. Outputs so far are in runs/{out.name}; press Run again with the same name to resume.")
            return None


def architecture_svg(stats: Optional[dict] = None) -> str:
    """Animated data-flow diagram of the cascade. `stats` (from run_metrics) overlays live counts per stage."""
    st_ = stats or {}

    def n(key, fmt="{}"):
        v = st_.get(key)
        return fmt.format(v) if v is not None and st_ else "-"

    stages = [
        ("ingest", "1 Ingest", "hash, EXIF, GSD,\nmanifest.jsonl", f"{n('images')} images", "#64748b"),
        ("gate", "2 Gate", "small VLM, JSON schema\nusable? damage? conf", f"{n('routed')} routed of {n('gated')}", "#3b82f6"),
        ("crop", "3 Crop", "1568 px tiles, overlap,\ncoords kept as evidence", "tiles on demand", "#0ea5e9"),
        ("grade", "4 Grade", "heavy VLM, rubric rows,\nschema-enforced finding", f"{n('findings')} findings, U {n('U')}", "#f59e0b"),
        ("prio", "5 Prioritize", "severity x criticality x\nconsequence x urgency", f"S4 {n('S4')}  S3 {n('S3')}", "#f97316"),
        ("review", "6 Review", "accept / override / U,\nprior value logged", f"{n('reviews')} decisions", "#8b5cf6"),
        ("export", "7 Export", "queue.csv, findings.json,\nbridge_entry.csv, report.md", n("usd_total", "${}"), "#22c55e"),
    ]
    W, H, bw, bh, gap, y0 = 1400, 560, 168, 96, 30, 190
    x0 = (W - (bw * len(stages) + gap * (len(stages) - 1))) // 2
    out = [f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='width:100%;height:auto;font-family:Inter,system-ui,sans-serif'>",
           "<defs><marker id='ah' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto'><path d='M0,0 L0,6 L9,3 z' fill='#94a3b8'/></marker>",
           "<style>.flow{stroke:#94a3b8;stroke-width:2.5;fill:none;stroke-dasharray:8 6;animation:dash 1.2s linear infinite;marker-end:url(#ah)}"
           ".side{stroke:#64748b;stroke-width:1.8;fill:none;stroke-dasharray:4 5;animation:dash 2s linear infinite;marker-end:url(#ah)}"
           "@keyframes dash{to{stroke-dashoffset:-28}} .box{rx:12;fill:#111827;stroke-width:2.5} .t{fill:#e5e7eb;font-size:15px;font-weight:700}"
           ".s{fill:#9ca3af;font-size:11.5px} .live{fill:#fbbf24;font-size:12.5px;font-weight:600} .lane{fill:#0b1220;stroke:#1f2937;stroke-dasharray:6 4;rx:14}"
           ".lt{fill:#94a3b8;font-size:12px;font-weight:700;letter-spacing:.6px} .pill{rx:9;fill:#1f2937;stroke:#374151} .pt{fill:#cbd5e1;font-size:11.5px}"
           ".pillm{rx:9;fill:#0f2a44;stroke:#3b82f6} .pillr{rx:9;fill:#111827;stroke:#6b7280;stroke-dasharray:4 3} .ps{fill:#fbbf24;font-size:10px}</style></defs>"]
    out.append(f"<rect class='lane' x='20' y='20' width='{W-40}' height='120'/><text class='lt' x='34' y='38'>INPUTS AND KNOWLEDGE (data, not code): six modalities (PRD 5A, D-014), then metadata, rubrics, exemplars</text>")
    out.append(f"<rect class='lane' x='20' y='{y0-30}' width='{W-40}' height='{bh+60}'/><text class='lt' x='34' y='{y0-12}'>ENGINE: one resumable loop, every call logged with tokens, dollars, seconds</text>")
    out.append(f"<rect class='lane' x='20' y='{y0+bh+60}' width='{W-40}' height='{H-(y0+bh+60)-20}'/><text class='lt' x='34' y='{y0+bh+82}'>FEEDBACK AND PROOF</text>")
    # row 1: the six modalities of PRD 5A (lidar dashed: roadmap, no code); status is what this repo has run
    mx = 40
    for text, status in MODALITY_PILLS:
        w = max(8 * len(text), 7 * len(status)) + 22
        cls = "pillr" if "roadmap" in text.lower() else "pillm"
        out.append(f"<rect class='{cls}' x='{mx}' y='46' width='{w}' height='38'/><text class='pt' x='{mx+10}' y='62'>{text}</text><text class='ps' x='{mx+10}' y='77'>{status}</text>")
        mx += w + 14
    # row 2: knowledge the engine reads (the side arrows below start at x = 120, 980 and 1250)
    pills = [("Customer asset metadata: id, class, GSD, date", 40), ("Rubric files: MBEI, NBI, IEC 62446-3, FEMA PDA, seismic, underwater (verbatim rows)", 520), ("Dev-set exemplars (FR-13)", 1180)]
    for text, x in pills:
        w = min(7 * len(text) + 20, W - 40 - x)
        out.append(f"<rect class='pill' x='{x}' y='94' width='{w}' height='26'/><text class='pt' x='{x+10}' y='111'>{text}</text>")
    xs = []
    for i, (key, title, sub, live, color) in enumerate(stages):
        x = x0 + i * (bw + gap)
        xs.append(x)
        out.append(f"<rect class='box' x='{x}' y='{y0}' width='{bw}' height='{bh}' stroke='{color}'/>")
        out.append(f"<text class='t' x='{x+12}' y='{y0+24}'>{title}</text>")
        for j, line in enumerate(sub.split("\n")):
            out.append(f"<text class='s' x='{x+12}' y='{y0+44+j*15}'>{line}</text>")
        out.append(f"<text class='live' x='{x+12}' y='{y0+bh-12}'>{live}</text>")
        if i:
            out.append(f"<path class='flow' d='M{x-gap+2},{y0+bh//2} L{x-6},{y0+bh//2}'/>")
    out.append(f"<path class='side' d='M120,120 L120,{y0-32} L{xs[0]+bw//2},{y0-32} L{xs[0]+bw//2},{y0-4}'/>")
    out.append(f"<path class='side' d='M980,120 L980,{y0-40} L{xs[3]+bw//2},{y0-40} L{xs[3]+bw//2},{y0-4}'/>")
    out.append(f"<path class='side' d='M1250,120 L1250,{y0-48} L{xs[3]+bw//2+30},{y0-48} L{xs[3]+bw//2+30},{y0-4}'/>")
    out.append(f"<path class='side' d='M{xs[1]+bw//2},{y0+bh+4} L{xs[1]+bw//2},{y0+bh+34} L{xs[6]+bw//2},{y0+bh+34} L{xs[6]+bw//2},{y0+bh+4}'/>")
    out.append(f"<text class='s' x='{xs[3]}' y='{y0+bh+30}'>clean and confident: skip the heavy stage (recall-first threshold tuned on dev only)</text>")
    by = y0 + bh + 100
    items = [
        (xs[0], "Cost log", "calls.jsonl: model, tokens,\nUSD, seconds per call"),
        (xs[2] - 10, "Review log (SQLite)", "prior value, reviewer, time;\nagreement over time (FR-19)"),
        (xs[4] - 20, "Frozen eval (D-007)", "eval_v1 manifest, class-to-\nseverity maps written first"),
        (xs[6] - 10, "Reports", "report.md per run, eval report\nwith n and 95% CI"),
    ]
    for x, title, sub in items:
        out.append(f"<rect class='box' x='{x}' y='{by}' width='{bw+40}' height='78' stroke='#334155'/><text class='t' x='{x+12}' y='{by+22}'>{title}</text>")
        for j, line in enumerate(sub.split("\n")):
            out.append(f"<text class='s' x='{x+12}' y='{by+42+j*15}'>{line}</text>")
    out.append(f"<path class='side' d='M{xs[5]+bw//2},{y0+bh+4} L{xs[5]+bw//2},{by+30} L{xs[2]+bw+30},{by+30}'/>")
    out.append(f"<path class='side' d='M{xs[2]+bw//2},{by+78} L{xs[2]+bw//2},{by+100} L{xs[3]+bw//2},{by+100} L{xs[3]+bw//2},{y0+bh+40}'/>")
    out.append(f"<text class='s' x='{xs[2]+bw//2+8}' y='{by+114}'>overrides become dev-set exemplars and eval cases, never touch eval_v1</text>")
    out.append(f"<path class='side' d='M{xs[6]+bw//2},{y0+bh+4} L{xs[6]+bw//2},{by-4}'/>")
    out.append("</svg>")
    return "".join(out)


def startup_svg() -> str:
    """Business-level flow: who pays, what goes in, what comes out, where the moat is."""
    W, H = 1400, 330
    cols = [
        ("Customers", ["Bridge inspection consultants", "County and DOT bridge owners", "PV O&M operators", "Disaster agencies, insurers"], "#22c55e"),
        ("They already have", ["Drone and phone imagery", "Standards they must report to", "Deadlines and fines for late entry", "Too few inspectors"], "#64748b"),
        ("The engine", ["Rubrics as data, not code", "Cheap gate, heavy grade", "Explicit U, verbatim criteria", "Consequence-ranked queue"], "#3b82f6"),
        ("They get", ["Pre-filled condition states", "Work list ranked by risk", "Evidence crop per finding", "Surge counts after an event"], "#f59e0b"),
        ("The moat", ["Frozen eval per asset class", "Review log: agreement over time", "Exemplars from adjudicated dev grades", "New asset class = new rubric file"], "#8b5cf6"),
    ]
    bw, gap, y0, bh = 240, 40, 40, 230
    x0 = (W - (bw * 5 + gap * 4)) // 2
    out = [f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='width:100%;height:auto;font-family:Inter,system-ui,sans-serif'>",
           "<defs><marker id='ah2' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto'><path d='M0,0 L0,6 L9,3 z' fill='#94a3b8'/></marker>"
           "<style>.f2{stroke:#94a3b8;stroke-width:2.5;fill:none;stroke-dasharray:8 6;animation:d2 1.2s linear infinite;marker-end:url(#ah2)}@keyframes d2{to{stroke-dashoffset:-28}}"
           ".b2{rx:12;fill:#111827;stroke-width:2.5}.h2{fill:#e5e7eb;font-size:16px;font-weight:700}.i2{fill:#cbd5e1;font-size:12.5px}</style></defs>"]
    for i, (title, items, color) in enumerate(cols):
        x = x0 + i * (bw + gap)
        out.append(f"<rect class='b2' x='{x}' y='{y0}' width='{bw}' height='{bh}' stroke='{color}'/><text class='h2' x='{x+14}' y='{y0+28}'>{title}</text>")
        for j, it in enumerate(items):
            out.append(f"<text class='i2' x='{x+14}' y='{y0+62+j*34}'>{it}</text>")
        if i:
            out.append(f"<path class='f2' d='M{x-gap+2},{y0+bh//2} L{x-6},{y0+bh//2}'/>")
    out.append(f"<path class='f2' d='M{x0+4*(bw+gap)+bw//2},{y0+bh+4} L{x0+4*(bw+gap)+bw//2},{y0+bh+30} L{x0+2*(bw+gap)+bw//2},{y0+bh+30} L{x0+2*(bw+gap)+bw//2},{y0+bh+6}'/>")
    out.append(f"<text class='i2' x='{x0+2*(bw+gap)+bw//2+10}' y='{y0+bh+48}'>every reviewed run makes the next run cheaper to trust</text>")
    out.append("</svg>")
    return "".join(out)


def gate_caption(g: dict) -> str:
    verdict = "unusable" if not g["usable"] else ("damage" if g["damage_present"] else "clean")
    routing = "routed" if g["routed"] else "not routed"
    forced = " (forced)" if "[forced:" in g.get("reason", "") else ""
    return f"gate: {verdict} {g['confidence']:.2f} · {routing}{forced}"


# ---------- crack measurement (FR-23 UI) ----------


def scale_from_args(args: Optional[tuple]) -> Optional[Scale]:
    """("gsd", mm_per_px, detail) or ("points", x0, y0, x1, y1, known_mm) -> Scale; None -> no scale.
    Raises ValueError for coincident points or a non-positive distance (scale_from_points)."""
    if not args:
        return None
    if args[0] == "gsd":
        return Scale(mm_per_px=float(args[1]), basis="gsd_metadata", detail=str(args[2]), confidence=SCALE_CONFIDENCE["gsd_metadata"])
    _, x0, y0, x1, y1, known_mm = args
    return scale_from_points((int(x0), int(y0)), (int(x1), int(y1)), float(known_mm))


def crack_overlay(img: Image.Image, bbox: List[int], mask: np.ndarray, skel: np.ndarray) -> Image.Image:
    """The bbox crop with the crack mask blended red and the skeleton drawn yellow (thickened on large crops)."""
    x0, y0, x1, y1 = bbox
    arr = np.asarray(img.crop((x0, y0, x1, y1)).convert("RGB")).astype(np.float32)
    red = np.array([220, 38, 38], dtype=np.float32)
    arr[mask] = 0.45 * arr[mask] + 0.55 * red
    grow = max(0, max(arr.shape[:2]) // 600)
    sk = ndimage.binary_dilation(skel, iterations=grow) if grow else skel
    arr[sk] = [250, 204, 21]
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if max(out.size) < 320:
        s = 320 / max(out.size)
        out = out.resize((int(out.width * s), int(out.height * s)), Image.NEAREST)
    return out


@st.cache_data(show_spinner=False, max_entries=24)
def crack_measure_cached(path: str, mtime: float, bbox: Tuple[int, int, int, int], scale_args: Optional[tuple], element: str) -> Tuple[dict, dict, Image.Image]:
    """measure_crack on one image and bbox, plus the overlay. Cached on the file, bbox, scale and element type."""
    img = Image.open(path).convert("RGB")
    cm = measure_crack(img, None, list(bbox), scale_from_args(scale_args), element_type=element)
    mask = crack_mask(img, cm.bbox)
    return cm.as_dict(), to_measurements(cm), crack_overlay(img, cm.bbox, mask, skeletonize(mask))


def merged_measurements(old: Measurements, tm: dict) -> Measurements:
    """Write the crack fields of to_measurements() into a finding's measurements. The width always travels with its
    basis and uncertainty. The grader's other fields and its confidence stay as they are; area_cm2 is filled only when
    the grader left it empty (a spall area is not overwritten by a crack-mask area)."""
    d = old.model_dump()
    for k in ("crack_width_mm", "crack_length_mm", "crack_width_uncertainty_mm", "measurement_basis"):
        d[k] = tm.get(k)
    if tm.get("area_cm2") is not None and d.get("area_cm2") is None:
        d["area_cm2"] = tm["area_cm2"]
    return Measurements.model_validate(d)


def crack_measure_panel(f: Finding, rec: Optional[ImageRecord], findings: List[Finding], out: Path, recs: List[ImageRecord]) -> None:
    """Body of the "Measure crack" expander under the selected finding. Never changes the finding's level."""
    if rec is None or not Path(rec.path).exists():
        st.info("No source image on disk for this finding, so nothing to measure.")
        return
    if rec.split == "eval_v1":
        st.info("This image belongs to the frozen eval set (data/eval_v1): crack measurement is never run on it (FR-23).")
        return
    if f.modality != "rgb":
        st.info(f"Crack measurement reads RGB imagery; this finding's modality is {f.modality}.")
        return
    fid = f.finding_id
    gsd = rec.gsd_mm_per_px or f.evidence.gsd_mm_per_px
    opt_gsd, opt_pts, opt_none = "GSD from image metadata", "Two points on a scale in the image", "No scale (pixels only)"
    options = ([opt_gsd] if gsd else []) + [opt_pts, opt_none]
    m1, m2 = st.columns([3, 1])
    src = m1.radio("Scale source", options, index=options.index(opt_gsd if gsd else opt_none), horizontal=True, key=f"meas_src_{fid}")
    element = m2.radio("MBEI rows", ["reinforced", "prestressed"], horizontal=True, key=f"meas_el_{fid}", help="reinforced: CS1 below 0.012 in, CS3 above 0.05 in; prestressed: CS1 below 0.004 in, CS3 above 0.009 in (bridge_mbei.json)")
    if not gsd:
        st.caption("No GSD for this image (ImageRecord.gsd_mm_per_px and Evidence.gsd_mm_per_px are empty), so the metadata option is off.")
    scale_args: Optional[tuple] = None
    scale_err = None
    if src == opt_gsd:
        scale_args = ("gsd", float(gsd), f"gsd_mm_per_px={gsd} ({rec.image_id})")
    elif src == opt_pts:
        st.caption("Two graduations of a ruler, tape, comparator card or marker of known spacing, in full-image pixels (read them off the image in any viewer). "
                   f"Scale = known mm / pixel distance, with ±1 px on each endpoint. The image is {rec.width}×{rec.height} px; the finding's bbox is {f.evidence.bbox}.")
        p = st.columns(5)
        x0 = p[0].number_input("x0 (px)", 0, max(0, rec.width - 1), 0, 1, key=f"meas_x0_{fid}")
        y0 = p[1].number_input("y0 (px)", 0, max(0, rec.height - 1), 0, 1, key=f"meas_y0_{fid}")
        x1 = p[2].number_input("x1 (px)", 0, max(0, rec.width - 1), 0, 1, key=f"meas_x1_{fid}")
        y1 = p[3].number_input("y1 (px)", 0, max(0, rec.height - 1), 0, 1, key=f"meas_y1_{fid}")
        known = p[4].number_input("Known distance (mm)", 0.0, 10000.0, 0.0, 1.0, key=f"meas_mm_{fid}")
        scale_args = ("points", int(x0), int(y0), int(x1), int(y1), float(known))
        try:
            scale_from_args(scale_args)
        except ValueError as e:
            scale_err = str(e)
            st.caption(f"Scale not set yet: {scale_err}.")
        if not scale_err:
            prev = Image.open(rec.path).convert("RGB")
            d = ImageDraw.Draw(prev)
            lw = max(3, prev.width // 300)
            if f.evidence.bbox:
                d.rectangle(list(f.evidence.bbox), outline=NEUTRAL_BOX, width=lw)
            d.line([(x0, y0), (x1, y1)], fill="#22d3ee", width=lw)
            for cx, cy in ((x0, y0), (x1, y1)):
                d.ellipse([cx - 2 * lw, cy - 2 * lw, cx + 2 * lw, cy + 2 * lw], outline="#22d3ee", width=lw)
            prev.thumbnail((720, 720))
            st.image(prev, caption=f"scale segment (cyan) {known:g} mm over {np.hypot(x1 - x0, y1 - y0):.1f} px; bbox in grey", width="content")
    bbox = tuple(int(v) for v in f.evidence.bbox) if len(f.evidence.bbox) == 4 else (0, 0, rec.width, rec.height)
    req = {"fid": fid, "scale": scale_args, "element": element, "bbox": bbox}
    if st.button("Measure crack", key="meas_go", type="primary", disabled=bool(scale_err)):
        st.session_state["crack_meas"] = req
    if st.session_state.get("crack_meas") != req:
        st.caption("Press Measure crack. The mask is a dark-thin-structure heuristic: shadows, joints, formwork lines and wires can pass as crack, so the reviewer confirms.")
        return
    try:
        with st.spinner("Segmenting, thinning and measuring"):
            cmd, tm, overlay = crack_measure_cached(rec.path, Path(rec.path).stat().st_mtime, bbox, scale_args, element)
    except Exception as e:
        st.error(f"Measurement failed: {type(e).__name__}: {e}")
        return
    scale = scale_from_args(scale_args)
    o1, o2 = st.columns([1.2, 1])
    o1.image(overlay, width="stretch", caption=f"crop {cmd['bbox']} · red: crack mask (heuristic) · yellow: skeleton · {cmd['mask_pixels']} mask px, {cmd['junctions']} junctions, {cmd['endpoints']} endpoints")
    with o2:
        if cmd["measurable"]:
            k, rel = scale.mm_per_px, float(scale.rel_err or 0.0)
            unc_max = float(np.hypot(WIDTH_UNCERTAINTY_PX * k, cmd["width_px_max"] * k * rel))
            le = "≤ " if cmd["width_upper_bound"] else ""
            a1, a2 = st.columns(2)
            a1.metric("Crack width (p95)", f"{le}{cmd['crack_width_mm_p95']:.2f} ± {cmd['width_uncertainty_mm']:.2f} mm", help="95th percentile of 2 x distance transform on the skeleton; ± is 1 px quantisation and the scale error in quadrature. This is the value saved as crack_width_mm.")
            a2.metric("Max width", f"{cmd['crack_width_mm_max']:.2f} ± {unc_max:.2f} mm", help="widest single skeleton pixel; lands on corner fills and joints, so it is shown, not saved")
            a3, a4 = st.columns(2)
            a3.metric("Length", f"{cmd['crack_length_mm']:.0f} mm", help="along the skeleton links")
            a4.metric("Area", f"{cmd['area_cm2']:.2f} cm²", help="mask pixels x scale squared")
        else:
            a1, a2 = st.columns(2)
            a1.metric("Width (p95)", f"{cmd['width_px_p95']:.1f} px" if cmd["width_px_p95"] is not None else "n/a")
            a2.metric("Max width", f"{cmd['width_px_max']:.1f} px" if cmd["width_px_max"] is not None else "n/a")
            a3, a4 = st.columns(2)
            a3.metric("Length", f"{cmd['length_px']:.0f} px" if cmd["length_px"] is not None else "n/a")
            a4.metric("Area", f"{cmd['area_px']} px")
            st.warning("Pixels only: " + "; ".join(r for r in tm["notes"][:3] if r) + ". crack_width_mm stays empty, the finding keeps not_measurable, and U is never counted as S0.")
        st.caption(f"Scale basis: **{scale.basis}** · {scale.mm_per_px:.4f} mm/px · {scale.detail} · relative scale error {float(scale.rel_err or 0):.3f}" if scale else "Scale basis: **none** (pixels only)")
        st.caption(f"measurement confidence {tm['confidence']:.2f} (team-assumption priors: segmentation x scale{', x 0.75 for a manual two-point scale' if scale and scale.basis == 'manual_two_points' else ''})")
    hint = cmd.get("mbei_condition_state_hint")
    if hint:
        states = hint["band_states"]
        lo, hi = hint["band_mm"]
        if len(states) > 1:
            st.markdown(amber(f"Candidate condition states: {', '.join(states)}") + f" · width band {lo:.2f} to {hi:.2f} mm straddles an MBEI boundary, so no single CS is implied.", unsafe_allow_html=True)
        else:
            st.markdown(f"MBEI hint: **{states[0]}** · width band {lo:.2f} to {hi:.2f} mm (widened to at least the {METHOD_FLOOR_MM:g} mm published method error) lies inside one state.")
        st.markdown(f"> {hint['criterion']}")
        st.caption(f"Row {hint['value']} (unified {hint['unified']}) quoted from {hint['source']}; p95 width {hint['width_in']:.4f} in against CS1 below {hint['thresholds_in']['CS1_below']} in and CS3 above {hint['thresholds_in']['CS3_above']} in. A hint for the reviewer: the finding's level is not changed.")
    st.markdown("**Measurement notes**  \n" + "  \n".join(f"- {n}" for n in cmd["notes"]))
    save = st.button("Save to finding", key="meas_save", disabled=not cmd["measurable"], help="writes crack_width_mm (p95), its uncertainty, length and measurement_basis into this finding's measurements; the level is not changed" if cmd["measurable"] else "nothing in mm to save without a scale")
    if save:
        f.measurements = merged_measurements(f.measurements, tm)
        save_findings(findings, out, recs)
        st.toast(f"Saved {crack_width_text(f.measurements)} to {fid}", icon="\U0001F4CF")
        st.rerun()


# ---------- seismic and vibration signals (FR-24 UI) ----------


def synthetic_series(f_hz: float, seed: int) -> pd.DataFrame:
    """Two-channel accelerometer-like series in m/s2: a free decay at f_hz (design damping SYNTH['zeta']) plus a small
    steady tone and gaussian noise. Synthetic by construction; every chart built from it says so."""
    fs, dur = SYNTH["rate_hz"], SYNTH["duration_s"]
    t = np.arange(int(fs * dur)) / fs
    rng = np.random.default_rng(seed)
    w, z = 2 * np.pi * f_hz, SYNTH["zeta"]
    decay = SYNTH["decay_amp"] * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z * z) * t)
    tone = SYNTH["tone_amp"] * np.sin(w * t + 0.3)
    return pd.DataFrame({"time": t, "x": 0.6 * (decay + tone) + rng.normal(0, SYNTH["noise_sd"], t.size), "z": decay + tone + rng.normal(0, SYNTH["noise_sd"], t.size)})


def write_synthetic_pair(d: Path) -> Tuple[Path, Path]:
    d.mkdir(parents=True, exist_ok=True)
    base, cur = d / "baseline_synthetic.csv", d / "current_synthetic.csv"
    synthetic_series(SYNTH["baseline_hz"], 1).to_csv(base, index=False)
    synthetic_series(SYNTH["current_hz"], 2).to_csv(cur, index=False)
    return cur, base


def _sig_params_key(p: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in p.items()))


@st.cache_data(show_spinner=False, max_entries=16)
def analyse_signal_cached(params_key: tuple, mtimes: tuple) -> dict:
    return analyse_signal(dict(params_key))


def analyse_signal(p: dict) -> dict:
    """Ingest the CSV (and optional baseline), compute indicators, grade on rubrics/seismic_shm.json. Params arrive as
    strings from the cache key; 'None' means not given."""
    val = lambda k: None if p.get(k) in (None, "None", "") else p[k]  # noqa: E731
    rate = float(val("rate")) if val("rate") else None
    kw = dict(units=val("units"), asset_class=p["asset_class"], mount=val("mount"))
    base = ingest_signal_csv(Path(val("baseline")), p["sensor"], rate, val("asset_id"), val("client_id"), **kw) if val("baseline") else None
    rec = ingest_signal_csv(Path(p["csv"]), p["sensor"], rate, val("asset_id"), val("client_id"), baseline_id=base.signal_id if base else None, **kw)
    if p.get("synthetic") == "True":
        rec.labels["synthetic"] = True
    if base is not None:
        rec.labels["baseline_record"] = base.model_dump()
    ind = indicators(rec, base)
    return {"record": rec, "baseline": base, "ind": ind, "finding": grade_signal(rec, ind, load_seismic_rubric())}


def _hann_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return np.fft.rfftfreq(x.size, 1.0 / fs), mag / (mag.max() or 1.0)


def signal_charts(rec: SignalRecord, base: Optional[SignalRecord], ind: dict, ch: str, synthetic: bool) -> Tuple[alt.Chart, alt.Chart]:
    """Time series (downsampled to about 2,000 points per series) and the Hann spectrum with both dominant peaks marked."""
    tag = f" · {SYNTH_LABEL}" if synthetic else ""
    series = [("current", rec)] + ([("baseline", base)] if base is not None else [])
    ts_rows, sp_rows = [], []
    fmax = 10.0
    peaks = []
    for name, r in series:
        chans = {c.lower(): i for i, c in enumerate(r.channels)}
        if ch.lower() not in chans:
            continue
        x = load_samples(r)[:, chans[ch.lower()]]
        fs = r.sample_rate_hz
        step = max(1, int(np.ceil(x.size / 2000)))
        t = np.arange(x.size) / fs
        ts_rows += [{"t_s": float(a), "value": float(b), "series": name} for a, b in zip(t[::step], x[::step])]
        fr, mag = _hann_spectrum(x, fs)
        sp_rows.append((name, fr, mag))
        f_pk = (ind.get("tracked_frequency_hz") or {}).get(ch) if name == "current" else None
        if name == "current" and f_pk is None:
            f_pk = (ind.get("dominant_frequency_hz") or {}).get(ch)
        if name == "baseline":
            f_pk = (ind.get("baseline_dominant_frequency_hz") or {}).get(ch)
        if f_pk:
            peaks.append({"f_hz": float(f_pk), "series": name, "label": f"{name} {f_pk:.3f} Hz"})
            fmax = max(fmax, 3 * f_pk)
    fmax = min(fmax, rec.sample_rate_hz / 2)
    sp = []
    for name, fr, mag in sp_rows:
        keep = fr <= fmax
        step = max(1, int(np.ceil(keep.sum() / 1500)))
        sp += [{"f_hz": float(a), "relative amplitude": float(b), "series": name} for a, b in zip(fr[keep][::step], mag[keep][::step])]
    color = alt.Color("series:N", scale=alt.Scale(domain=["current", "baseline"], range=["#3b82f6", "#9ca3af"]), title=None)
    ts = alt.Chart(pd.DataFrame(ts_rows)).mark_line(strokeWidth=1).encode(
        x=alt.X("t_s:Q", title="t (s)"), y=alt.Y("value:Q", title=f"{ch} ({rec.labels.get('units') or 'units unknown'})"), color=color, tooltip=["series", "t_s", "value"]
    ).properties(height=220, title=f"Time series, channel {ch} (downsampled for display){tag}")
    spec = alt.Chart(pd.DataFrame(sp)).mark_line().encode(
        x=alt.X("f_hz:Q", title="frequency (Hz)", scale=alt.Scale(domain=[0, fmax])), y=alt.Y("relative amplitude:Q", title="|FFT| (Hann), normalised"), color=color, tooltip=["series", "f_hz", "relative amplitude"]
    )
    if peaks:
        pdf = pd.DataFrame(peaks)
        rules = alt.Chart(pdf).mark_rule(strokeDash=[5, 4], strokeWidth=2).encode(x="f_hz:Q", color=color, tooltip=["label"])
        text = alt.Chart(pdf).mark_text(align="left", dx=4, dy=-6, fontWeight="bold").encode(x="f_hz:Q", y=alt.value(12), text="label:N", color=color)
        spec = spec + rules + text
    return ts, spec.properties(height=220, title=f"Spectrum, channel {ch}, dominant peaks marked{tag}")


def signal_view(rec: SignalRecord, base: Optional[SignalRecord], ind: dict, key: str) -> None:
    """Metadata strip, indicator metrics per channel, time series and spectrum."""
    synthetic = bool((rec.labels or {}).get("synthetic"))
    if synthetic:
        st.markdown(amber(f"{SYNTH_LABEL}: generated with numpy, not a measurement") + f" · baseline {SYNTH['baseline_hz']} Hz, current {SYNTH['current_hz']} Hz, {SYNTH['rate_hz']:g} Hz, {SYNTH['duration_s']:g} s, design damping {SYNTH['zeta']}, gaussian noise sd {SYNTH['noise_sd']} m/s2 [Assumption: demo design values]", unsafe_allow_html=True)
    lab = rec.labels or {}
    st.caption(f"`{rec.signal_id}` · {rec.sensor} · units {lab.get('units') or 'unknown'} · mount {lab.get('mount') or 'not given'} · {rec.sample_rate_hz:g} Hz ({lab.get('rate_source')}) · {rec.n_samples} samples, {rec.duration_s:.1f} s · channels {', '.join(rec.channels)} · "
               f"baseline {base.signal_id if base is not None else (rec.baseline_id or 'none')} · asset {rec.asset_id or 'n/a'} · client {rec.client_id or 'n/a'}")
    ch = st.radio("Channel", rec.channels, horizontal=True, key=f"{key}_ch") if len(rec.channels) > 1 else rec.channels[0]
    units = ind.get("units") or "units unknown"
    pga, rms, fdom = (ind.get("pga") or {}).get(ch), (ind.get("rms") or {}).get(ch), (ind.get("dominant_frequency_hz") or {}).get(ch)
    pct_g = (ind.get("pga_pct_g") or {}).get(ch)
    pgv = (ind.get("pgv_cm_s") or {}).get(ch)
    zeta, zd = (ind.get("damping_ratio") or {}).get(ch), (ind.get("damping_detail") or {}).get(ch)
    shift = (ind.get("frequency_shift_pct") or {}).get(ch) if ind.get("frequency_shift_pct") is not None else None
    unc = (ind.get("frequency_shift_uncertainty_pct") or {}).get(ch)
    res = ind.get("fft_resolution_hz")
    k = st.columns(5)
    extra = f" ({pct_g:.2f} %g)" if pct_g is not None else (f" ({pgv:.2f} cm/s PGV)" if pgv is not None else "")
    k[0].metric("PGA / peak", f"{pga:.3g} {units}{extra}" if pga is not None else "n/a", help="max |x| after mean removal; %g only for an accelerometer with known units")
    k[1].metric("RMS", f"{rms:.3g} {units}" if rms is not None else "n/a")
    k[2].metric("Dominant frequency", f"{fdom:.3f} ± {res:.3f} Hz" if fdom is not None else "n/a", help=f"Hann-window FFT peak with parabolic refinement; ± is one FFT bin (resolution {res:.4f} Hz = rate / samples used)")
    k[3].metric("Damping ratio", f"{zeta:.3f}" if zeta is not None else "n/a", help=(f"log decrement over {zd['n_peaks']} decaying peaks, R² {zd['r2']:.2f}; indicator only, no threshold row" if zd else "no clean free-decay window found; indicator only"))
    if ind.get("frequency_shift_pct") is None:
        k[4].metric("Frequency shift vs baseline", "n/a", help="no baseline: not computed, the finding is U (never S0)")
    else:
        k[4].metric("Frequency shift vs baseline", f"{shift:+.1f} ± {unc:.1f} %" if shift is not None else "not computed", help="tracked peak within ±20 % of the baseline dominant frequency; ± is one FFT bin over the baseline frequency")
    if synthetic and zeta is not None:
        st.caption(f"The synthetic series was built with damping {SYNTH['zeta']}; the log-decrement estimator reads {zeta:.3f} on it (the steady tone and noise floor flatten the envelope). Shown to make the estimator's bias visible, not as a validation.")
    ts, spec = signal_charts(rec, base, ind, ch, synthetic)
    c1, c2 = st.columns(2)
    c1.altair_chart(ts, width="stretch")
    c2.altair_chart(spec, width="stretch")


def signal_grade_view(finding: Finding) -> None:
    """The graded finding: badge, rubric rows quoted with their source tag (team-proposed rows in amber), U reason."""
    rows = {r["criterion"]: r for r in load_seismic_rubric()["rows"]}
    st.markdown(badge(finding.unified.level, f"**{finding.native_scale.value}** on {finding.native_scale.standard} · action **{finding.action.code}** · confidence {finding.measurements.confidence:.2f} · flags {', '.join(finding.unified.flags) or 'none'}"), unsafe_allow_html=True)
    if finding.unified.level == "U":
        st.warning(f"U, not assessable: {finding.action.basis} U is listed on its own and never counted as S0.")
    st.markdown("**Rubric rows matched (verbatim from rubrics/seismic_shm.json)**")
    for crit in finding.native_scale.criteria_matched or ["(none quoted)"]:
        st.markdown(f"> {crit}")
        src = (rows.get(crit) or {}).get("source")
        if src:
            tag = amber("team-proposed, validate") + " · " if "team-proposed" in src else ""
            st.markdown(f"<small>{tag}source: {src}</small>", unsafe_allow_html=True)
    st.markdown(f"**Justification:** {finding.justification}")


def stored_signal(out: Path, finding_id: str) -> Optional[Tuple[SignalRecord, Optional[SignalRecord], dict]]:
    """(record, baseline record, indicators) for a signal finding from runs/<run>/signals.jsonl (last row wins)."""
    sp = out / "signals.jsonl"
    if not sp.exists():
        return None
    hit = None
    for line in sp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("finding_id") == finding_id:
                hit = row
    if not hit:
        return None
    rec = SignalRecord.model_validate(hit["record"])
    b = (rec.labels or {}).get("baseline_record")
    return rec, SignalRecord.model_validate(b) if b else None, hit["indicators"]


def save_signal_run(p: dict) -> Tuple[Path, Finding]:
    """runs/sig_<ts>/: the CSVs copied under signals/, findings via write_signal_findings, signals.jsonl, an empty
    gate.jsonl (list_runs lists folders with one, so Reports, Client reports and Model health can open the run) and a
    summary.json with zero images. No model call."""
    out = RUNS / f"sig_{time.strftime('%m%d_%H%M%S')}"
    sdir = out / "signals"
    sdir.mkdir(parents=True, exist_ok=True)
    q = dict(p)
    for k in ("csv", "baseline"):
        if q.get(k) not in (None, "None", ""):
            dst = sdir / Path(q[k]).name
            shutil.copyfile(q[k], dst)
            q[k] = str(dst)
    res = analyse_signal({k: str(v) for k, v in q.items()})
    rec, ind, finding = res["record"], res["ind"], res["finding"]
    ranked = write_signal_findings(out, [finding])
    append_signal_row(out, rec, ind, finding)
    (out / "gate.jsonl").touch()
    summary = {
        "images": 0, "gated": 0, "routed_to_grader": 0, "routing_fraction": None, "unusable": 0, "findings": len(ranked),
        "levels": {lvl: sum(1 for f in ranked if f.unified.level == lvl) for lvl in LEVELS}, "signal_findings": sum(1 for f in ranked if f.modality == "seismic"),
        "native_values": {f.native_scale.value: 1 for f in ranked}, "asset_classes": {}, "usd_total": 0.0, "usd_per_image": None, "seconds_total": 0.0,
        "source": "Sensors tab (cascade.signals, deterministic, no model calls)", "synthetic": bool(rec.labels.get("synthetic")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_reports(out, {})
    return out, next(f for f in ranked if f.finding_id == finding.finding_id)


# ---------- sidebar ----------

st.sidebar.title("Inspection grading cascade")
st.sidebar.caption("gate → crop → grade → prioritize → review → export")
mode = st.sidebar.radio("Mode", ["Inspect", "Surge (disaster)"], horizontal=True, help="Surge: building damage only, whole frame, FEMA PDA counts and a surge report")
surge = mode.startswith("Surge")
source = st.sidebar.radio("Images", ["Dataset", "Upload"], horizontal=True)

manifests = list_manifests()
records: List[ImageRecord] = []
sidebar_exts: List[dict] = []
if source == "Dataset":
    if not manifests:
        st.sidebar.error("No manifests found. Run `python scripts/make_demo_manifests.py` first.")
    choice = st.sidebar.selectbox("Dataset", list(manifests) or ["none"])
    dataset_filter = None
    if choice in manifests and manifests[choice] == DEV_MANIFEST:
        pick = st.sidebar.selectbox("Filter dev set", ["all"] + list(DATASET_LABELS), format_func=lambda k: "all" if k == "all" else DATASET_LABELS[k])
        dataset_filter = None if pick == "all" else pick
    limit = st.sidebar.slider("Max images", 1, 50, 12)
    if choice in manifests:
        records = records_for(manifests[choice], dataset_filter, limit)
        if surge:
            records = [r for r in records if r.asset_class == "building_disaster"]
else:
    asset_class = "building_disaster" if surge else st.sidebar.selectbox("Asset class", ASSET_CLASSES)
    sidebar_client = st.sidebar.text_input("Client id (optional; groups the client report)", value="", key="sidebar_client")
    files = st.sidebar.file_uploader(UPLOAD_LABEL, type=UPLOAD_TYPES, accept_multiple_files=True, key="sidebar_upload")
    if not FF_OK:
        st.sidebar.caption(f"Video upload off: {FF_MSG}")
    with st.sidebar:
        sidebar_vopts = video_options("sb")
    if files:
        sig = (tuple((f.name, f.size) for f in files), asset_class, sidebar_client, tuple(sorted(sidebar_vopts.items())))
        cache = st.session_state.get("sidebar_ingest")
        if not cache or cache["sig"] != sig:  # ingest (and extract frames) once per distinct upload, not on every rerun
            up = RUNS / "_uploads" / time.strftime("%Y%m%d_%H%M%S")
            recs_up, exts_up, line_up = ingest_uploads(files, up, asset_class, sidebar_client, "up", sidebar_vopts)
            write_manifest(recs_up, up / "manifest.jsonl")
            cache = {"sig": sig, "records": recs_up, "exts": exts_up, "line": line_up}
            st.session_state["sidebar_ingest"] = cache
        records, sidebar_exts = cache["records"], cache["exts"]
        st.sidebar.success(cache["line"])
if records:
    st.sidebar.caption(f"{len(records)} images selected · " + ", ".join(f"{k} {v}" for k, v in pd.Series([r.asset_class for r in records]).value_counts().items()))

st.sidebar.divider()
gate = st.sidebar.selectbox("Gate (stage A)", ["local", "claude", "none"], help="local = Qwen3-VL-4B via Ollama; claude = Haiku 4.5; none = route everything")
grader = st.sidebar.selectbox("Grader (stage C)", ["claude", "local", "none"], help="claude = " + os.getenv("GRADER_MODEL", "claude-opus-5") + "; local = Qwen3-VL-8B via Ollama; none = gate only")
tiles = st.sidebar.checkbox("Tile large images (FR-8)", value=False, disabled=surge, help="one grader call per 1568 px tile; surge mode always grades the whole frame")
use_exemplars = st.sidebar.checkbox("Few-shot exemplars from dev set (FR-13)", value=False, disabled=not DEV_MANIFEST.exists())
gate_min_conf = st.sidebar.slider("Route clean verdicts below confidence", 0.0, 1.0, 0.7, 0.05)
force_pv = st.sidebar.checkbox("Always grade PV thermal crops (skip gate verdict)", value=True, help="the local gate routed 0 of 12 thermal crops on the dev set; the gate still runs and its verdict is logged")
run_name = st.sidebar.text_input("Run name", value=f"ui_{time.strftime('%m%d_%H%M')}")
reviewer = st.sidebar.text_input("Reviewer name (for the review log)", value=os.getenv("USERNAME", "reviewer"))
run_clicked = st.sidebar.button("Run cascade", type="primary", disabled=not records, width="stretch")

st.sidebar.divider()
open_run = st.sidebar.selectbox("Or open a finished run", ["(none)"] + list_runs())

cfg = RunConfig(gate=gate, grader=grader, tiles=tiles and not surge, gate_min_conf=gate_min_conf, force_route_classes=("pv_module",) if force_pv else ())
active_run = st.session_state.get("active_run")
if open_run != "(none)" and not run_clicked:
    active_run = open_run
    st.session_state["active_run"] = open_run
if run_clicked and records:  # decided before the tabs so the run pickers below follow the run about to finish
    st.session_state["active_run"] = run_name
    active_run = run_name


def sync_run_pickers(name: Optional[str]) -> None:
    """The Reports / Client reports / Eval matrix / Model health pickers are keyed widgets: Streamlit keeps their
    stored value after the first render, so `index=` alone never follows the sidebar. Set them once per change of
    the active run, only to a run that is in their options (a finished run folder)."""
    if name and st.session_state.get("_synced_run") != name and name in list_runs():
        for k in ("report_pick", "client_run", "eval_pick", "health_pick"):
            st.session_state[k] = name
        st.session_state["_synced_run"] = name


sync_run_pickers(active_run)

tab_run, tab_arch, tab_drop, tab_sensors, tab_batch, tab_reports, tab_clients, tab_eval, tab_health, tab_why = st.tabs(
    ["Inspect run", "Architecture", "Drop & grade", "Sensors", "Batch", "Reports", "Client reports", "Eval matrix", "Model health", "Why this approach"]
)

# ---------- Inspect run ----------

with tab_run:
    if run_clicked and records:
        run_with_progress(records, RUNS / run_name, cfg, use_exemplars, surge, extractions=sidebar_exts)
        sync_run_pickers(active_run)  # the run folder exists now; the pickers below are instantiated later in this script

    if not active_run:
        st.title("Inspection grading cascade")
        st.markdown(
            """
Pick a dataset or upload images on the left, choose the gate and grader backends, and press **Run cascade**.
Or drop images straight into **Drop & grade**, run several datasets under **Batch**, and compare stored runs under **Reports**.

**What happens per image**

1. **Gate** (small VLM): usable? damage present? Confidence and a one-line reason. Recall-first routing.
2. **Crop**: large images tiled at 1,568 px with overlap; tile coordinates are kept as evidence.
3. **Grade** (heavy VLM, schema-enforced): native scale value with the rubric criterion quoted verbatim, unified S0 to S4, measurements or `not_measurable`, action and justification. Refusals become U, never S0.
4. **Prioritize**: score = severity × criticality × consequence × urgency; any S4 goes to the top with same-day escalation.
5. **Review**: accept, override or mark U; every action is logged with the prior model value.
6. **Export**: queue CSV, findings JSON, bridge entry CSV, and a stored report.

**Multi-sensor inputs (D-014, PRD section 5A).** The same queue takes six modalities: RGB drone and camera imagery (the measured path),
thermal heatmaps (PV demo), sonar frames and interior-machinery images (rubric files only, no data run yet), seismic and vibration
time series (**Sensors** tab: indicators and rubric rows in plain code, run on a labelled synthetic pair only) and lidar (roadmap, no code).
A crack's width, length and area in mm come from **Findings & review > Measure crack** when the image carries a scale (GSD metadata or
two points on a ruler); without one the width stays in pixels and the finding stays `not_measurable`.

Numbers shown here are measured from each run's call log. Accuracy claims live only in `eval/reports/` and the **Eval matrix** tab;
there are none for seismic, sonar or lidar.
"""
        )
        with st.expander("60-second demo script"):
            st.markdown(
                """
1. Open the **mixed sample** (four asset classes) and press **Run cascade**. Counters tick: gated, routed, graded, cost, seconds.
2. Click one finding: evidence crop, native grade with the rubric criterion quoted verbatim, unified level, confidence, action.
3. Open the **Work queue**: any S4 at the top with same-day escalation; multipliers visible per row.
4. **Override** one grade as a reviewer; the log shows prior value and reviewer; the queue re-ranks; the agreement timeline updates.
5. Same pipeline, different rubric: PV thermal gives IEC classes, RescueNet gives FEMA classes and U counts (**Surge counts**).
6. **Export** the queue CSV and the stored report. End on **Eval matrix**: gate recall and within-one-grade accuracy with n.
"""
            )
        if sidebar_exts:
            st.subheader("Ingested videos")
            for i, ext in enumerate(sidebar_exts):
                video_strip(ext, records, key=f"sb_strip_{i}")
        if records:
            classes = pd.Series([r.asset_class for r in records]).value_counts()
            st.subheader(f"Selected images ({len(records)})")
            st.markdown(" ".join(f"<span class='chip'>{ASSET_ICON.get(k, '')} {ASSET_LABEL.get(k, k)} · {v}</span>" for k, v in classes.items()), unsafe_allow_html=True)
            if len(classes) > 1:
                for ac in ASSET_CLASSES:
                    group = [r for r in records if r.asset_class == ac]
                    if group:
                        st.markdown(f"**{ASSET_ICON.get(ac, '')} {ASSET_LABEL.get(ac, ac)}** · {len(group)}")
                        gallery(group, captions={r.image_id: f"{r.width}x{r.height} · {r.source_dataset}" for r in group}, cols=6, key=f"preview_{ac}")
            else:
                gallery(records, captions={r.image_id: f"{r.asset_class} · {r.width}x{r.height}" for r in records}, key="preview")
    else:
        out = RUNS / active_run
        run = load_run(out)
        findings = run["findings"]
        gate_rows, calls, summary = run["gate"], run["calls"], run["summary"]
        imgs = load_records_for_run(out)
        recs = list(imgs.values())
        review_log = ReviewLog(out / "reviews.sqlite")
        lvl_by_img = worst_levels(findings)
        health_doc = health_for(out, imgs)
        acked = (health_doc or {}).get("acknowledged") or {}
        blocking = [a for a in (health_doc or {}).get("alarms", []) if a["rule"] in BLOCKING_RULES and a["rule"] not in set(acked.get("rules") or [])]

        st.title(f"Run `{active_run}`")
        if health_doc:
            st.markdown("Model health: " + "  ".join(status_badge(s["status"], f"{stage}" + (f" ({s['worst_rule']})" if s["worst_rule"] else "")) for stage, s in health_doc["stages"].items()) + " &nbsp; <small>details under the Model health tab</small>", unsafe_allow_html=True)
        sub_over, sub_find, sub_queue, sub_surge, sub_export = st.tabs(["Overview", "Findings & review", "Work queue", "Surge counts", "Export"])

        with sub_over:
            p = Progress(
                images=summary.get("images", len(gate_rows)),
                gated=len(gate_rows),
                routed=sum(1 for g in gate_rows if g["routed"]),
                unusable=sum(1 for g in gate_rows if not g["usable"]),
                graded=len({f.evidence.image_ids[0] for f in findings if f.evidence.image_ids}),
                findings=len(findings),
                usd=round(sum(c["usd"] for c in calls), 4),
                seconds_gate=round(sum(c["seconds"] for c in calls if c["stage"] == "gate"), 1),
                seconds_grade=round(sum(c["seconds"] for c in calls if c["stage"] == "grade"), 1),
                stage="done",
            )
            for f in findings:
                p.levels[f.unified.level] = p.levels.get(f.unified.level, 0) + 1
            shown_cfg = RunConfig(**summary["config"]) if summary.get("config") else cfg
            render_counters(st.empty(), p, shown_cfg)
            st.markdown("  ".join(badge(lvl, LEVEL_MEANING[lvl]) + " &nbsp;" for lvl in LEVELS), unsafe_allow_html=True)

            st.subheader("Images: gate verdict and worst grade")
            gate_by_id = {g["image_id"]: g for g in gate_rows}
            ordered = [imgs[g["image_id"]] for g in gate_rows if g["image_id"] in imgs]
            gallery(ordered, captions={i: gate_caption(g) for i, g in gate_by_id.items()}, levels=lvl_by_img, key="overview")
            n_sig = sum(1 for f in findings if f.modality == "seismic")
            if n_sig:
                st.caption(f"This run also holds {n_sig} seismic / vibration finding{'s' if n_sig != 1 else ''} from the Sensors tab (no image; open it under Findings & review for the indicators, charts and quoted rubric rows).")
            video_detections_section(out, imgs, key="over_vd")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Gate decisions")
                if gate_rows:
                    st.dataframe(pd.DataFrame(gate_rows)[["image_id", "usable", "damage_present", "confidence", "routed", "seconds", "reason"]], width="stretch", hide_index=True, height=320)
                else:
                    st.info("No gate rows yet.")
            with c2:
                st.subheader("Cost and latency per call")
                if calls:
                    cdf = pd.DataFrame(calls)
                    agg = cdf.groupby(["stage", "model"]).agg(calls=("usd", "size"), usd=("usd", "sum"), median_s=("seconds", "median"), in_tok=("input_tokens", "sum"), out_tok=("output_tokens", "sum")).reset_index()
                    st.dataframe(agg, width="stretch", hide_index=True)
                    st.caption(f"USD per image: {summary.get('usd_per_image')} · API list prices from src/cascade/costlog.py; local models are $0")
                else:
                    st.info("No calls logged.")
            if summary:
                with st.expander("summary.json"):
                    st.json(summary)
            st.subheader("Workforce impact: review desk time with a smaller team")
            st.caption("Counts are measured from this run; hours, percent and dollars are estimates that move with the sliders. No field visit, flight or headcount claim is made.")
            workforce_panel(out, imgs, key="wf_over")

        with sub_find:
            if not findings:
                st.info("No findings. Either nothing was routed or the grader was set to none.")
            else:
                df = findings_df(findings)
                st.dataframe(df, width="stretch", hide_index=True, height=260)
                fid = st.selectbox("Finding", df["finding_id"].tolist(), key="finding_pick")
                f = next(x for x in findings if x.finding_id == fid)
                rec = imgs.get(f.evidence.image_ids[0]) if f.evidence.image_ids else None
                c1, c2 = st.columns([1.1, 1])
                sig = stored_signal(out, f.finding_id) if f.modality == "seismic" else None
                with c1:
                    if sig:
                        signal_view(sig[0], sig[1], sig[2], key="find_sig")
                    elif f.modality == "seismic":
                        st.info("Seismic finding: its series is not stored in this run's signals.jsonl, so no chart can be drawn.")
                    elif rec and Path(rec.path).exists():
                        st.image(draw_evidence(f, rec), caption=f"{rec.image_id} · {rec.width}x{rec.height} · tile {f.evidence.tile} · bbox {f.evidence.bbox}", width="stretch")
                        truth = rec.labels.get("grade_native") or rec.labels.get("source_class")
                        if truth:
                            st.caption(f"Dataset label, for eval only and never shown to the model: **{truth}** ({rec.labels.get('grade_source')})")
                    else:
                        st.warning("Source image not on disk for this run.")
                with c2:
                    st.markdown(f"### {f.native_scale.value} on {f.native_scale.standard} · " + badge(f.unified.level, f.unified.uncertainty) + " " + modality_chip(f.modality), unsafe_allow_html=True)
                    st.caption(LEVEL_MEANING[f.unified.level])
                    sla = f", within {f.action.sla_days} days" if f.action.sla_days is not None else ""
                    st.markdown(f"**Defect:** {f.defect_type}  \n**Action:** {f.action.code}{sla}  \n**Basis:** {f.action.basis}")
                    st.markdown("**Criteria matched (verbatim from rubric):**")
                    for c in f.native_scale.criteria_matched or ["(none quoted)"]:
                        st.markdown(f"> {c}")
                    st.markdown(f"**Justification:** {f.justification}")
                    m = f.measurements
                    length = f", length {m.crack_length_mm:.0f} mm" if m.crack_length_mm is not None else ""
                    st.markdown(
                        f"**Measurements:** area {m.area_cm2} cm², {crack_width_text(m)}{length}, ΔT {m.delta_t_k} K, rust {m.percent_area_rusted} %, section loss {m.section_loss_pct} % · confidence **{m.confidence:.2f}** · flags: {', '.join(f.unified.flags) or 'none'}"
                    )
                    reviewed = f" by {f.review.reviewer} at {f.review.reviewed_at} (prior {f.review.prior_level})" if f.review.reviewer else ""
                    st.caption(f"model {f.model} · ${f.usd:.4f} · {f.seconds:.1f} s · review: {f.review.status}{reviewed}")
                    st.markdown("**Reviewer decision (FR-18)**")
                    b1, b2, b3 = st.columns(3)
                    gradable = [lvl for lvl in LEVELS if lvl != "U"]
                    new_level = b2.selectbox("Override to", gradable, index=gradable.index(f.unified.level) if f.unified.level in gradable else 0, label_visibility="collapsed", key="override_level")
                    action = None
                    if b1.button("Accept", width="stretch", key="btn_accept"):
                        action = ("accepted", None)
                    if b2.button("Override", width="stretch", key="btn_override"):
                        action = ("overridden", new_level)
                    if b3.button("Mark U", width="stretch", key="btn_u"):
                        action = ("marked_u", None)
                    if action:
                        review_log.apply(active_run, f, action[0], reviewer, action[1])
                        save_findings(findings, out, recs)
                        write_reports(out, imgs)
                        refresh_video_detections(out, imgs)
                        st.toast(f"{action[0]} by {reviewer}, prior {f.review.prior_level}", icon="\U0001F4DD")
                        st.rerun()
                if f.modality != "seismic":
                    with st.expander("Measure crack (width, length and area in mm from a scale in the image)", expanded=False):
                        crack_measure_panel(f, rec, findings, out, recs)
                agg = review_log.agreement(active_run)
                rate = f", agreement {agg['agreement_rate']:.0%}" if agg["agreement_rate"] is not None else ""
                st.caption(f"Review log: {agg['total']} decisions, accepted {agg['accepted']}, overridden {agg['overridden']}, marked U {agg['marked_u']}{rate} · persisted in reviews.sqlite")
                tl = review_log.timeline(active_run)
                if tl:
                    st.markdown("**Model vs reviewer agreement over time (FR-19)**")
                    st.altair_chart(agreement_chart(tl), width="stretch")

        with sub_queue:
            st.markdown("`score = severity_weight[S] × criticality × consequence × urgency`. Any **S4** sorts first and is escalated same day. **U** is listed, never scored as S0. Criticality is 1 unless supplied per asset.")
            if findings:
                rows = []
                ranked = sorted(findings, key=lambda x: x.queue_rank or 10**9)
                for f in ranked:
                    rec = imgs.get(f.evidence.image_ids[0]) if f.evidence.image_ids else None
                    rows.append(
                        {
                            "rank": f.queue_rank,
                            "score": f.queue_score,
                            "level": f.unified.level,
                            "native": f.native_scale.value,
                            "image": f.evidence.image_ids[0] if f.evidence.image_ids else "",
                            "asset_class": f.asset_class,
                            "action": f.action.code,
                            "sla_days": f.action.sla_days,
                            **multipliers(f, rec),
                            "flags": ";".join(f.unified.flags),
                            "review": f.review.status,
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=400)
                st.subheader("Top of the queue, visually")
                seen_q, top, caps = set(), [], {}
                for f in ranked:  # one thumbnail per image, captioned by its best-ranked finding
                    iid = f.evidence.image_ids[0] if f.evidence.image_ids else None
                    if iid and iid in imgs and iid not in seen_q:
                        seen_q.add(iid)
                        top.append(imgs[iid])
                        caps[iid] = f"#{f.queue_rank} · {f.native_scale.value} · {f.action.code}"
                    if len(top) >= 10:
                        break
                gallery(top, captions=caps, levels=lvl_by_img, key="queue")
            else:
                st.info("Empty queue.")

        with sub_surge:
            dis = [f for f in findings if f.asset_class == "building_disaster"]
            if not dis:
                st.info("No building_disaster findings in this run. Choose the RescueNet demo set or Surge mode.")
            else:
                counts = surge_counts(dis)
                c1, c2, c3 = st.columns(3)
                c1.metric("Images assessed", counts["n_images"])
                c2.metric("Unassessable (U)", counts["u_count"])
                c3.metric("Mean confidence", f"{counts['mean_confidence']:.2f}" if counts["mean_confidence"] is not None else "n/a")
                cc1, cc2 = st.columns(2)
                cc1.bar_chart(pd.Series(counts["by_fema_class"], name="FEMA PDA class"))
                cc2.bar_chart(pd.Series(counts["by_level"], name="unified level"))
                st.dataframe(pd.DataFrame(counts["ranked"]), width="stretch", hide_index=True)
                rp = out / "surge_report.md"
                if rp.exists():
                    st.download_button("Download surge_report.md", rp.read_bytes(), file_name="surge_report.md", key="dl_surge")

        with sub_export:
            if blocking:
                st.error("Model health alarm blocks the work-list exports until acknowledged under Model health: " + "; ".join(f"{a['rule']} ({a['stage']}, n={a['n']}): {a['action']}" for a in blocking))
            elif acked and any(a["rule"] in BLOCKING_RULES for a in (health_doc or {}).get("alarms", [])):
                st.caption(f"Blocking alarm acknowledged by {acked.get('by')} at {acked.get('at')}: exports released.")
            for name in ["report.md", "report.json", "queue.csv", "findings.json", "bridge_entry.csv", "gate.jsonl", "calls.jsonl", "summary.json", "surge_counts.json", "surge_report.md", "health.json"]:
                fp = out / name
                if fp.exists():
                    st.download_button(f"Download {name}", fp.read_bytes(), file_name=f"{active_run}_{name}", key=f"dl_{name}", disabled=bool(blocking) and name in ("queue.csv", "bridge_entry.csv"))
            for tp in sorted((out / "videos").glob("*_timeline.json")) if (out / "videos").exists() else []:  # FR-4b bundle per video
                vslug = tp.name[: -len("_timeline.json")]
                st.markdown(f"Video `{vslug}`")
                for vname in (f"{vslug}_detections.mp4", f"{vslug}_detections.vtt", f"{vslug}_timeline.json", f"{vslug}_contact.png"):
                    vp = out / "videos" / vname
                    if vp.exists():
                        st.download_button(f"Download videos/{vname}", vp.read_bytes(), file_name=f"{active_run}_{vname}", key=f"dl_video_{vname}")
            if (out / "clients" / "index.json").exists():
                st.caption("Per-client HTML reports are under the Client reports tab.")
            st.caption("queue.csv columns are documented in src/cascade/export.py (QUEUE_COLUMNS). bridge_entry.csv carries element, condition state and quantity columns for SNBI-style entry (FR-17). report.md is the stored run report shown under Reports.")

# ---------- Architecture ----------

with tab_arch:
    st.subheader("How an image becomes a ranked, reviewed finding")
    arch_stats = None
    if active_run and (RUNS / active_run / "gate.jsonl").exists():
        try:
            arch_stats = run_metrics(RUNS / active_run)
        except Exception:
            arch_stats = None
    st.caption(("Yellow numbers are live from run `" + active_run + "`.") if arch_stats else "Open or run a cascade to overlay live counts on each stage.")
    st.markdown(architecture_svg(arch_stats), unsafe_allow_html=True)
    st.caption("Inputs lane: the six modalities of PRD section 5A under D-014 (scope widened to multi-sensor structural health, software plus a sensor kit). "
               "Whiteboard owners, 2026-09-25: Leena aerial RGB (buildings and bridges, drone); Atharva underwater, seismic, thermal and lidar; Jie interior machinery; Runze software; Drew thermal. "
               "The yellow line under each pill is what this repo has actually run: images through the full cascade (measured), seismic series through cascade.signals on synthetic data only (Sensors tab), "
               "sonar and machinery as rubric files with no data run, lidar no code. Seismic series skip the gate and grader: indicators and rubric rows are computed in plain code, then join the same queue and exports.")
    with st.expander("Stage by stage, in words", expanded=False):
        st.markdown(
            """
| Stage | Input | What happens | Output | Where |
|---|---|---|---|---|
| 1 Ingest | folder, upload or manifest | hash, dimensions, EXIF date, GSD and irradiance if given; missing values stay null | `manifest.jsonl` | `cascade.ingest` |
| 2 Gate | one image | small VLM answers a fixed JSON schema: usable, damage_present, confidence, reason; recall-first threshold; forced routing per asset class | `gate.jsonl` | `cascade.gate` |
| 3 Crop | routed image | tiles above 1,568 px with overlap; tile coordinates travel with the finding as evidence | tiles in memory | `cascade.crop` |
| 4 Grade | tile + rubric rows + exemplars | heavy VLM returns the finding contract, schema-enforced: native value, criteria quoted verbatim, unified S0 to S4 or U, measurements or not_measurable, action | `findings.json` | `cascade.grade`, `rubrics/*.json` |
| 5 Prioritize | all findings | score = severity x criticality x consequence x urgency; any S4 first, same-day | `queue.csv` | `cascade.prioritize` |
| 6 Review | one finding | accept, override or mark U; prior value, reviewer and time logged; queue re-ranks | `reviews.sqlite` | `cascade.review` |
| 7 Export | run folder | queue CSV, findings JSON, bridge entry CSV, surge report, run report | `runs/<name>/` | `cascade.export`, `cascade.report` |

Every model call is logged with model, tokens, dollars and seconds (`calls.jsonl`). The loop is resumable: rerun with the same name and finished images are skipped.

A clip takes one extra step at each end (video: frames sampled by ffmpeg and graded like stills; a detections mp4 is then built from the kept sample frames, each held until the next sample, with subtitles; the source frames between samples were not graded): `cascade.video` writes `videos/<slug>.json`, `cascade.videodetect` writes `videos/<slug>_detections.mp4`, `.vtt`, `_timeline.json` and `_contact.png`.

Two side paths added under D-014, both without a model call. **Crack metrology** (`cascade.measure`, Findings & review > Measure crack): a dark-thin-structure mask inside the finding's bbox, a skeleton, widths from the distance transform, converted to mm only when a scale exists (GSD metadata or two points on a ruler of known spacing); the width always carries its ± uncertainty and scale basis, and the MBEI row it falls in is a hint, never an automatic grade. **Seismic and vibration series** (`cascade.signals`, Sensors tab): CSV in, PGA, RMS, dominant frequency, damping and the frequency shift against a baseline out, graded on `rubrics/seismic_shm.json`; no baseline gives U.
"""
        )
    st.subheader("The startup around the engine")
    st.markdown(startup_svg(), unsafe_allow_html=True)
    st.caption("Wedge per D-001: bridge elements graded to AASHTO/NBIS scales, sold first to inspection consultants and county owners. Solar PV thermal second (D-002). Surge mode is a feature of the same engine (D-003). Sources: docs/decisions.md, docs/PRD.md.")

# ---------- Drop & grade ----------

with tab_drop:
    st.subheader("Drop images or video, get graded findings")
    st.caption("Drag files onto the box: camera JPEGs, drone stills, PNGs" + (", or a video (frames are sampled, near-duplicates dropped, each kept frame becomes an image record)" if FF_OK else "") + ". They are hashed and ingested, then run through the cascade with the backends chosen in the sidebar. Each drop becomes its own stored run.")
    d1, d2 = st.columns([2, 1])
    drop_files = d1.file_uploader(f"Drop {UPLOAD_LABEL} here", type=UPLOAD_TYPES, accept_multiple_files=True, key="drop_upload")
    if not FF_OK:
        d1.caption(f"Video upload off: {FF_MSG}")
    with d1:
        drop_vopts = video_options("drop")
    drop_class = d2.selectbox("Asset class", ASSET_CLASSES, index=ASSET_CLASSES.index("building_disaster") if surge else 0, key="drop_class")
    drop_client = d2.text_input("Client id (optional; groups the client report)", value="", key="drop_client")
    d2.markdown(f"Backends: gate `{gate}`, grader `{grader}`" + (", surge posture" if surge else ""))
    go = d2.button("Grade dropped files", type="primary", disabled=not drop_files, key="drop_go", width="stretch")
    demo_go = d2.button("Demo video: bridge_walkthrough.mp4", key="drop_demo", width="stretch", disabled=not (FF_OK and DEMO_VIDEO.exists()),
                        help="12 s 1280x960 clip under data/demo/video, sampled with the Video options above and graded as bridge elements")
    if (go and drop_files) or demo_go:
        name = f"drop_{time.strftime('%m%d_%H%M%S')}"
        out = RUNS / name
        up = out / "uploads"
        up.mkdir(parents=True, exist_ok=True)
        drop_exts: List[dict] = []
        if demo_go:
            drop_recs = []
            try:
                drop_recs, ext = ingest_video(DEMO_VIDEO, up / "frames" / slug_of(DEMO_VIDEO), asset_class="bridge_element", client_id=drop_client or None, captured_on_from_mtime=False, source_dataset="demo_video", split="demo", **drop_vopts)
                drop_exts = [ext.to_json()]
                drop_line = f"0 images + {ext.to_json()['n_kept']} frames from 1 video ingested ({ext.to_json()['n_dropped']} near-duplicate frames dropped, dHash ≤ {drop_vopts['dedup_max_distance']})"
            except VideoError as e:
                st.error(str(e))
            drop_surge = surge
        else:
            drop_recs, drop_exts, drop_line = ingest_uploads(drop_files, up, drop_class, drop_client, "drop", drop_vopts)
            drop_surge = surge or drop_class == "building_disaster"
        if drop_recs:
            st.session_state.setdefault("video_extractions", {})[name] = drop_exts
            if demo_go:
                # ingest only: grading is a separate, visible click so no model is called by surprise
                st.session_state["drop_pending"] = {"name": name, "records": drop_recs, "exts": drop_exts, "surge": drop_surge, "line": drop_line}
            else:
                st.session_state["drop_run"] = name
                st.success(drop_line)
                for i, ext in enumerate(drop_exts):
                    video_strip(ext, drop_recs, key=f"drop_pre_{i}")
                run_with_progress(drop_recs, out, cfg, use_exemplars, drop_surge, extractions=drop_exts)
        else:
            st.warning("Nothing ingested.")
    pend = st.session_state.get("drop_pending")
    if pend and not (RUNS / pend["name"] / "gate.jsonl").exists():
        st.success(pend["line"] + " · nothing graded yet")
        for i, ext in enumerate(pend["exts"]):
            video_strip(ext, pend["records"], key=f"drop_pend_{i}")
        pc1, pc2 = st.columns([1, 2])
        if pc1.button(f"Grade these {len(pend['records'])} frames", type="primary", key="drop_pending_go", width="stretch"):
            st.session_state["drop_run"] = pend["name"]
            st.session_state.pop("drop_pending", None)
            run_with_progress(pend["records"], RUNS / pend["name"], cfg, use_exemplars, pend["surge"], extractions=pend["exts"])
        pc2.caption(f"gate `{gate}`, grader `{grader}` from the sidebar; each routed frame is one grader call")
    drop_run = st.session_state.get("drop_run")
    if drop_run and (RUNS / drop_run / "gate.jsonl").exists():
        out = RUNS / drop_run
        run = load_run(out)
        imgs = load_records_for_run(out)
        by_img: Dict[str, List[Finding]] = {}
        for f in run["findings"]:
            for iid in f.evidence.image_ids:
                by_img.setdefault(iid, []).append(f)
        st.markdown(f"**Run `{drop_run}`** · {len(run['gate'])} images · {len(run['findings'])} findings · ${sum(c['usd'] for c in run['calls']):.3f}")
        for i, ext in enumerate(st.session_state.get("video_extractions", {}).get(drop_run) or stored_extractions(out)):
            video_strip(ext, list(imgs.values()), key=f"drop_post_{i}")
        for g in run["gate"]:
            rec = imgs.get(g["image_id"])
            if not rec:
                continue
            c1, c2 = st.columns([1, 2])
            fs = sorted(by_img.get(rec.image_id, []), key=lambda x: x.queue_rank or 10**9)
            with c1:
                st.image(draw_evidence(fs[0], rec) if fs else thumbnail(rec.path, Path(rec.path).stat().st_mtime), width="stretch")
            with c2:
                st.markdown(f"**{rec.image_id}** · {gate_caption(g)}" + (f" · from {Path(rec.source_video).name} @ {rec.frame_time_s:.1f} s" if rec.source_video and rec.frame_time_s is not None else ""))
                st.caption(f"gate reason: {g['reason']}")
                if not fs:
                    st.write("Not graded (gate said clean, or grader set to none).")
                for f in fs:
                    st.markdown(badge(f.unified.level, f"**{f.native_scale.value}** on {f.native_scale.standard} · {f.defect_type} · action {f.action.code}" + (f" within {f.action.sla_days} d" if f.action.sla_days is not None else "")), unsafe_allow_html=True)
                    st.markdown(f"<small>{f.justification}</small>", unsafe_allow_html=True)
            st.divider()
        video_detections_section(out, imgs, key=f"drop_vd_{drop_run}")
        st.caption("Open this run under Inspect run (sidebar: open a finished run) to review, re-rank and export.")

# ---------- Sensors ----------

with tab_sensors:
    st.subheader("Sensors: seismic and vibration time series")
    st.caption("A CSV time series (a time column plus one column per channel) is ingested, its indicators computed with numpy (PGA, RMS, dominant frequency, damping, frequency shift against a baseline) and graded on rubrics/seismic_shm.json in plain code, no model call. "
               "The frequency-shift and RMS thresholds are team-proposed and unvalidated; no real seismic series has been run through this yet, so no accuracy is claimed. Without a baseline the finding is U, never S0. "
               "Sonar and thermal frames go through the image cascade; lidar is roadmap (PRD section 5A).")
    sc1, sc2 = st.columns([2, 1])
    sig_csv = sc1.file_uploader("Time series CSV", type=["csv"], key="sig_csv")
    sig_base = sc1.file_uploader("Baseline CSV (optional; the same asset and channels in a healthy state)", type=["csv"], key="sig_base")
    sensors = list(get_args(Sensor))
    sig_sensor = sc2.selectbox("Sensor type", sensors, index=sensors.index("accelerometer"), key="sig_sensor")
    sig_units = sc2.selectbox("Units", SIG_UNITS, key="sig_units", help="stored, never inferred: without units no PGA / PGV band is applied")
    sig_mount = sc2.selectbox("Mount", ["(not given)"] + list(MOUNTS), key="sig_mount", help="the ShakeMap PGA / PGV rows apply to free_field or ground_floor records only")
    sig_rate = sc2.number_input("Sample rate, Hz (0 = read from the time column)", 0.0, 100000.0, 0.0, 1.0, key="sig_rate")
    classes_all = list(get_args(AssetClass))
    sig_class = sc2.selectbox("Asset class", classes_all, index=classes_all.index("bridge_element"), key="sig_class")
    sig_asset = sc2.text_input("Asset id", value="", key="sig_asset")
    sig_client = sc2.text_input("Client id", value="", key="sig_client")
    bb1, bb2, bb3 = sc1.columns(3)
    sig_go = bb1.button("Analyse uploaded CSV", type="primary", disabled=sig_csv is None, key="sig_go", width="stretch")
    sig_demo = bb2.button("Use synthetic demo signal", key="sig_demo", width="stretch", help=f"writes a labelled synthetic pair into runs/_signals_demo/: {SYNTH['baseline_hz']} Hz baseline, {SYNTH['current_hz']} Hz current, {SYNTH['rate_hz']:g} Hz, {SYNTH['duration_s']:g} s, with noise")
    sig_nobase = bb3.checkbox("Ignore the baseline (shows the U path)", value=False, key="sig_nobase")
    if sig_go and sig_csv is not None:
        up = SIG_UPLOAD_DIR / time.strftime("%Y%m%d_%H%M%S")
        up.mkdir(parents=True, exist_ok=True)
        cpath = up / Path(sig_csv.name).name
        cpath.write_bytes(sig_csv.getbuffer())
        bpath = None
        if sig_base is not None:
            bpath = up / f"baseline_{Path(sig_base.name).name}"
            bpath.write_bytes(sig_base.getbuffer())
        st.session_state["sig_params"] = {
            "csv": str(cpath), "baseline": str(bpath) if bpath else None, "sensor": sig_sensor, "rate": sig_rate or None,
            "units": None if sig_units == "(unknown)" else sig_units, "mount": None if sig_mount == "(not given)" else sig_mount,
            "asset_class": sig_class, "asset_id": sig_asset or None, "client_id": sig_client or None, "synthetic": False,
        }
    if sig_demo:
        cur, base = write_synthetic_pair(SIG_DEMO_DIR)
        st.session_state["sig_params"] = {
            "csv": str(cur), "baseline": str(base), "sensor": "accelerometer", "rate": SYNTH["rate_hz"], "units": "m/s2", "mount": "structure",
            "asset_class": "bridge_element", "asset_id": sig_asset or "synthetic-demo-bridge", "client_id": sig_client or "synthetic-demo", "synthetic": True,
        }
        st.toast("Synthetic demo pair written to runs/_signals_demo/", icon="\U0001F4C8")
    sp = st.session_state.get("sig_params")
    if not sp:
        st.info("Upload a CSV and press Analyse, or press Use synthetic demo signal.")
    else:
        params = dict(sp)
        if sig_nobase:
            params["baseline"] = None
        files_now = [p_ for p_ in (params["csv"], params.get("baseline")) if p_]
        if not all(Path(p_).exists() for p_ in files_now):
            st.warning("The CSV files of the last analysis are gone from disk; upload again or press the synthetic demo button.")
        else:
            try:
                res = analyse_signal_cached(_sig_params_key(params), tuple(Path(p_).stat().st_mtime for p_ in files_now))
            except Exception as e:
                res = None
                st.error(f"Could not analyse the series: {type(e).__name__}: {e}")
            if res:
                st.divider()
                signal_view(res["record"], res["baseline"], res["ind"], key="sig_view")
                st.markdown("#### Graded finding")
                signal_grade_view(res["finding"])
                if res["ind"]["notes"]:
                    st.markdown("**Indicator notes**  \n" + "  \n".join(f"- {n}" for n in res["ind"]["notes"]))
                sv1, sv2 = st.columns([1, 3])
                if sv1.button("Save as run", key="sig_save", type="primary", width="stretch"):
                    try:
                        sout, sf = save_signal_run(params)
                        st.session_state["sig_saved"] = sout.name
                        st.toast(f"Saved {sf.unified.level} finding to runs/{sout.name}", icon="\U0001F4BE")
                    except Exception as e:
                        st.error(f"Save failed: {type(e).__name__}: {e}")
                    else:
                        st.rerun()  # the sidebar run list and the run pickers above were drawn before the folder existed
                sv2.caption("Writes runs/sig_<time>/ with findings.json, queue.csv, signals.jsonl, a copy of the CSVs, an empty gate.jsonl (no images) and report.md; open it from the sidebar, Reports, Client reports or Model health."
                            + (f" Last saved: `{st.session_state['sig_saved']}`." if st.session_state.get("sig_saved") else ""))

# ---------- Batch ----------

with tab_batch:
    st.subheader("Batch: several datasets in one go")
    st.caption("Each selected dataset becomes its own run (resumable, stored, reported). Runs execute one after another with live counters; compare them under Reports afterwards.")
    b1, b2 = st.columns([2, 1])
    picks = b1.multiselect("Datasets", list(manifests), default=[], key="batch_picks")
    per = b2.slider("Max images per dataset", 1, 50, 10, key="batch_limit")
    prefix = b2.text_input("Run name prefix", value=f"batch_{time.strftime('%m%d_%H%M')}", key="batch_prefix")
    est_imgs = sum(len(records_for(manifests[k], None, per)) for k in picks) if picks else 0
    b2.markdown(f"About **{est_imgs}** images · gate `{gate}`, grader `{grader}`")
    if b1.button("Run batch", type="primary", disabled=not picks, key="batch_go"):
        status = st.empty()
        rows = []
        for k in picks:
            recs_k = records_for(manifests[k], None, per)
            slug = manifests[k].parent.name if manifests[k] != DEV_MANIFEST else "dev"
            name = f"{prefix}_{slug}"
            rows.append({"run": name, "dataset": k, "images": len(recs_k), "status": "running"})
            status.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            with st.expander(f"{name} ({len(recs_k)} images)", expanded=True):
                summ = run_with_progress(recs_k, RUNS / name, cfg, use_exemplars, surge)
            rows[-1]["status"] = "done" if summ else "stopped"
            if summ:
                rows[-1].update(findings=summ["findings"], usd=summ["usd_total"], routed=summ["routed_to_grader"])
            status.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.session_state["batch_runs"] = [r["run"] for r in rows]
        st.success("Batch finished. See Reports for the comparison.")
    if st.session_state.get("batch_runs"):
        st.markdown("Last batch: " + ", ".join(f"`{r}`" for r in st.session_state["batch_runs"]))

# ---------- Reports ----------

with tab_reports:
    st.subheader("Stored run reports")
    all_runs = list_runs()
    if not all_runs:
        st.info("No runs yet.")
    else:
        metrics = []
        for r in all_runs:
            try:
                metrics.append(run_metrics(RUNS / r))
            except Exception as e:  # a half-written run must not break the page
                metrics.append({"run": r, "error": f"{type(e).__name__}: {e}"})
        mdf = pd.DataFrame(metrics)
        show_cols = [c for c in ["run", "mode", "gate", "grader", "images", "routed", "findings", *LEVELS, "usd_total", "usd_per_image", "seconds_gate", "seconds_grade", "reviews"] if c in mdf.columns]
        st.table(mdf[show_cols].set_index("run"))
        default = st.session_state.get("batch_runs") or ([active_run] if active_run else all_runs[:3])
        compare = st.multiselect("Compare runs", all_runs, default=[r for r in default if r in all_runs], key="cmp_runs")
        if compare and all(lvl in mdf.columns for lvl in LEVELS):
            sub = mdf[mdf["run"].isin(compare)].fillna(0)
            c1, c2 = st.columns(2)
            c1.altair_chart(levels_chart(sub[["run"] + list(LEVELS)], "run"), width="stretch")
            cost = sub[["run", "usd_per_image", "seconds_gate", "seconds_grade", "images"]].copy()
            cost["seconds_per_image"] = (cost["seconds_gate"] + cost["seconds_grade"]) / cost["images"].replace(0, pd.NA)
            cost = cost.astype({"usd_per_image": float, "seconds_per_image": float})
            with c2:
                k1, k2 = st.columns(2)
                for col, (metric, title) in zip((k1, k2), (("usd_per_image", "USD per image"), ("seconds_per_image", "model seconds per image"))):
                    col.altair_chart(
                        alt.Chart(cost).mark_bar(color="#3b82f6").encode(x=alt.X("run:N", title=None), y=alt.Y(f"{metric}:Q", title=title), tooltip=["run", metric]).properties(height=240),
                        width="stretch",
                    )
        st.divider()
        pick = st.selectbox("Open a report", all_runs, index=all_runs.index(active_run) if active_run in all_runs else 0, key="report_pick")
        rp = RUNS / pick / "report.md"
        cA, cB = st.columns([1, 3])
        if cA.button("Generate / refresh report.md", key="report_gen", help="report.md with the workforce estimate section, report.json, per-client reports and health.json; no model calls"):
            write_reports(RUNS / pick, load_records_for_run(RUNS / pick))
            st.rerun()
        if rp.exists():
            cB.download_button("Download report.md", rp.read_bytes(), file_name=f"{pick}_report.md", key="report_dl")
            st.markdown(rp.read_text(encoding="utf-8"))
            rr = load_run(RUNS / pick)
            ranked = sorted(rr["findings"], key=lambda x: x.queue_rank or 10**9)[:10]
            imgs_r = load_records_for_run(RUNS / pick)
            top = [imgs_r[f.evidence.image_ids[0]] for f in ranked if f.evidence.image_ids and f.evidence.image_ids[0] in imgs_r]
            if top:
                st.markdown("**Top findings, visually**")
                gallery(top, captions={f.evidence.image_ids[0]: f"#{f.queue_rank} · {f.native_scale.value} · {f.action.code}" for f in ranked if f.evidence.image_ids}, levels=worst_levels(rr["findings"]), key="report_gallery")
        else:
            st.info("No report stored for this run yet. Press Generate.")

# ---------- Client reports ----------

with tab_clients:
    st.subheader("Client reports: one page per client")
    st.caption("Grouped by client_id, else the source dataset, else 'unassigned'. Each report is a single HTML file (embedded evidence thumbnails and SVG charts, no scripts, under 5 MB) with a markdown twin and report.json. U rows are listed separately and never counted as S0.")
    all_runs = list_runs()
    if not all_runs:
        st.info("No runs yet.")
    else:
        cpick = st.selectbox("Run", all_runs, index=all_runs.index(active_run) if active_run in all_runs else 0, key="client_run")
        crun = RUNS / cpick
        g1, g2 = st.columns([1, 3])
        if g1.button("Generate / refresh client reports", key="client_gen", type="primary"):
            with st.spinner("rendering client reports (no model calls)"):
                try:
                    write_client_reports(crun, run_records(crun, load_records_for_run(crun)))
                    st.toast("client reports written", icon="\U0001F4C4")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
            st.rerun()
        idx = crun / "clients" / "index.json"
        if not idx.exists():
            st.info("No client reports yet. Press Generate.")
        else:
            crows = json.loads(idx.read_text(encoding="utf-8"))
            labels = {r["client_id"]: f"{r['client_id']} · {r['images']} images · {r['findings']} findings" for r in crows}
            g2.caption(" · ".join(f"{r['client_id']} ({r['basis']}): S4 {r['S4']}, U {r['U']}, ${r['usd']:.3f}" for r in crows))
            cid = st.selectbox("Client", list(labels), format_func=lambda k_: labels[k_], key="client_pick")
            crow = next(r for r in crows if r["client_id"] == cid)
            m = json.loads((crun / crow["json"]).read_text(encoding="utf-8"))
            st.markdown(f"### {m['client_id']} · run `{m['run']}` · grouped by {m['grouping_basis']} · generated {m['generated_at'][:19]}")
            k = st.columns(8)
            k[0].metric("Images", m["images"])
            k[1].metric("Routed / gated", f"{m['routed']} / {m['gated']}")
            k[2].metric("Findings", m["findings"])
            k[3].metric("S4", m["levels"]["S4"], help=LEVEL_MEANING["S4"])
            k[4].metric("S3", m["levels"]["S3"], help=LEVEL_MEANING["S3"])
            k[5].metric("U", m["levels"]["U"], help="never counted as S0: " + LEVEL_MEANING["U"])
            k[6].metric("Reviews pending", m["reviews"]["pending"])
            k[7].metric("USD", f"{m['usd_total']:.3f}", help=f"{m['cost_source']}")
            ch1, ch2, ch3 = st.columns(3)
            ldf = pd.DataFrame([{"level": lvl, "count": m["levels"].get(lvl, 0)} for lvl in LEVELS])
            lbase = alt.Chart(ldf).encode(y=alt.Y("level:N", sort=list(LEVELS), title=None), x=alt.X("count:Q", title="findings"))
            lbar = lbase.mark_bar().encode(color=alt.Color("level:N", scale=alt.Scale(domain=list(LEVEL_COLOR), range=list(LEVEL_COLOR.values())), legend=None), tooltip=["level", "count"])
            ltext = lbase.mark_text(align="left", dx=4).encode(text="count:Q")
            ch1.altair_chart((lbar + ltext).properties(height=220, title="Findings by unified level (U listed, never S0)"), width="stretch")
            adf = pd.DataFrame([{"action": a, "count": m["action_counts"].get(a, 0)} for a in ACTION_ORDER if m["action_counts"].get(a, 0)])
            if not adf.empty:
                donut = alt.Chart(adf).mark_arc(innerRadius=50).encode(theta=alt.Theta("count:Q"), color=alt.Color("action:N", sort=list(ACTION_ORDER), scale=alt.Scale(domain=list(ACTION_ORDER), range=[ACTION_COLOR[a] for a in ACTION_ORDER])), tooltip=["action", "count"]).properties(height=220, title="Actions")
                ch2.altair_chart(donut, width="stretch")
            else:
                ch2.info("No actions yet.")
            sdf = pd.DataFrame([r for r in m["sla"] if r.get("due_in_days") is not None])
            if not sdf.empty:
                pts = alt.Chart(sdf).mark_circle(size=90).encode(x=alt.X("due_in_days:Q", title="due in days (negative = overdue)"), y=alt.Y("action:N", sort=list(ACTION_ORDER), title=None),
                                                                  color=alt.Color("level:N", scale=alt.Scale(domain=list(LEVEL_COLOR), range=list(LEVEL_COLOR.values()))), tooltip=["finding_id", "rank", "level", "action", "due_in_days"])
                rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#dc2626").encode(x="x:Q")
                ch3.altair_chart((pts + rule).properties(height=220, title="SLA calendar"), width="stretch")
            else:
                ch3.info("No dated SLA rows (rows without a due date are not chartable).")
            if m["reviews"].get("timeline"):
                st.markdown("**Model vs reviewer agreement over time for this client (FR-19)**")
                st.altair_chart(agreement_chart(m["reviews"]["timeline"]), width="stretch")
            st.markdown("**Assets**")
            with st.container():
                if m["assets"]:
                    st.dataframe(pd.DataFrame(m["assets"]), width="stretch", hide_index=True)
                else:
                    st.info("No assets.")
            st.markdown("**Workload, measured from this run**")
            st.markdown("\n".join(f"- {line}" for line in _workload_lines(m)))
            rr_c = load_run(crun)
            imgs_c = load_records_for_run(crun)
            seen_c, ctop, ccaps = set(), [], {}
            for q in m["queue"]:  # one thumbnail per image: the first (best-ranked) finding captions it
                if q["image_id"] in imgs_c and q["image_id"] not in seen_c:
                    seen_c.add(q["image_id"])
                    ctop.append(imgs_c[q["image_id"]])
                    ccaps[q["image_id"]] = f"#{q['queue_rank']} · {q['native']} · {q['action_code']}"
                if len(ctop) >= 10:
                    break
            if ctop:
                st.markdown("**Top of this client's queue**")
                gallery(ctop, captions=ccaps, levels=worst_levels(rr_c["findings"]), key="client_gallery")
            html_path, md_path = crun / crow["html"], crun / crow["md"]
            dl1, dl2, dl3 = st.columns(3)
            if html_path.exists():
                dl1.download_button("Download report.html", html_path.read_bytes(), file_name=f"{cpick}_{m['slug']}_report.html", mime="text/html", key="client_dl_html")
            if md_path.exists():
                dl2.download_button("Download report.md", md_path.read_bytes(), file_name=f"{cpick}_{m['slug']}_report.md", key="client_dl_md")
            dl3.caption(f"thumbnails embedded {m['thumbs'].get('embedded')} of {m['thumbs'].get('findings')} ({m['thumbs'].get('encoding')}), html {html_path.stat().st_size / 1e6:.2f} MB" if html_path.exists() else "")
            if html_path.exists():
                with st.expander("Preview report.html", expanded=False):
                    components.html(html_path.read_text(encoding="utf-8"), height=900, scrolling=True)  # inline HTML; st.iframe takes a URL/path, not markup

# ---------- Eval matrix ----------

with tab_eval:
    st.subheader("Eval matrix: gate and grading against dataset labels")
    st.caption("Truth comes from the datasets' own labels and the class-to-severity maps frozen before any model output (D-007). Images without a label are listed, not guessed. Confidence intervals are in eval/reports/; this tab shows counts.")
    all_runs = list_runs()
    if not all_runs:
        st.info("No runs yet.")
    else:
        pick = st.selectbox("Run", all_runs, index=all_runs.index(active_run) if active_run in all_runs else 0, key="eval_pick")
        rr = load_run(RUNS / pick)
        imgs_e = load_records_for_run(RUNS / pick)
        em = eval_matrix(imgs_e, rr["gate"], [f.model_dump() for f in rr["findings"]])
        g = em["gate"]
        st.markdown("#### Gate (stage A): damage present vs routed")
        if g["n"] == 0:
            st.info("No image in this run carries a damage_present label (uploads never do).")
        else:
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("Recall (headline)", f"{100 * g['recall']:.1f}%" if g["recall"] is not None else "n/a", help="damaged images that were routed; recall-first by design")
            c1.metric("Precision", f"{100 * g['precision']:.1f}%" if g["precision"] is not None else "n/a")
            c2.altair_chart(confusion_chart(["clean", "damaged"], [[g["tn"], g["fp"]], [g["fn"], g["tp"]]], f"gate, n={g['n']}"), width="content")
            if em["per_dataset"]:
                pdf = pd.DataFrame([{"dataset": ds, **d} for ds, d in em["per_dataset"].items()])
                c3.dataframe(pdf[["dataset", "n", "tp", "fp", "fn", "tn", "recall", "precision"]], width="stretch", hide_index=True)
            if g["misses"]:
                st.markdown("**Missed damaged images** (routed = no):")
                gallery([imgs_e[i] for i in g["misses"] if i in imgs_e], captions={i: "missed by gate" for i in g["misses"]}, key="eval_misses", max_n=10)
        sc1, sc2 = st.columns([1, 3])
        if sc1.button("Score with bootstrap CIs", key="eval_score", help="runs eval/run_eval.py on this run's manifest; writes eval/reports/<run>.md and .json, no model calls"):
            mp = RUNS / pick / "manifest.jsonl"
            if not mp.exists():
                mp = DEV_MANIFEST
            with st.spinner("bootstrapping 1000 resamples"):
                proc = subprocess.run([sys.executable, str(ROOT / "eval" / "run_eval.py"), "--manifest", str(mp), "--run", str(RUNS / pick), "--name", pick], capture_output=True, text=True, cwd=str(ROOT))
            if proc.returncode == 0:
                st.toast(f"eval/reports/{pick}.md written", icon="\U0001F4CA")
                st.rerun()
            else:
                st.error(proc.stderr[-1500:] or proc.stdout[-1500:])
        sc2.caption("The scored report carries n and 95% bootstrap CIs per metric; the matrices below are raw counts.")
        st.markdown("#### Grading (stage C): worst finding per image vs truth")
        if not em["grading"]:
            st.info("No routed image with a grade label in this run. Use a demo or dev dataset with the grader on.")
        for ac, gr in em["grading"].items():
            st.markdown(f"**{ac}** · truth: {gr['truth_source']}")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.altair_chart(confusion_chart(gr["labels"], gr["matrix"], f"{ac}: n assessed {gr['n_assessed']} of {gr['n_routed_with_truth']}"), width="content")
            with c2:
                if gr["n_assessed"]:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Exact match", f"{100 * gr['exact_match']:.0f}%")
                    m2.metric("Within one grade", f"{100 * gr['within_one_grade']:.0f}%", help="the PRD's headline grading metric; FHWA found 68% of human ratings within one point")
                    m3.metric("Weighted kappa", f"{gr['qwk']:.2f}")
                    m1.metric("MAE (grades)", f"{gr['mae']:.2f}")
                    m2.metric("Over-graded", gr["over_graded"])
                    m3.metric("Under-graded", gr["under_graded"], help="under-grading is the costly direction for safety")
                st.metric("U rate", f"{100 * gr['u_rate']:.0f}%" if gr["u_rate"] is not None else "n/a", help="routed images with truth that produced no gradable finding")
                if gr["pairs"]:
                    with st.expander("Per-image truth vs predicted"):
                        st.dataframe(pd.DataFrame(gr["pairs"]), width="stretch", hide_index=True)
                if gr["u_images"]:
                    st.caption("U images: " + ", ".join(gr["u_images"][:20]))
        st.divider()
        st.markdown("#### Scored eval reports (eval/reports)")
        reports = sorted((ROOT / "eval" / "reports").glob("*.md"), key=lambda q: q.stat().st_mtime, reverse=True)
        if not reports:
            st.info("No scored report yet. Press Score with bootstrap CIs above, or run eval/run_eval.py.")
        else:
            rp = st.selectbox("Report", reports, format_func=lambda q: q.name, index=next((i for i, q in enumerate(reports) if q.stem == pick), 0), key="eval_report_pick")
            st.markdown(rp.read_text(encoding="utf-8"))

# ---------- Model health ----------

with tab_health:
    st.subheader("Model health: drift detection and validation loops")
    st.caption("Everything here is computed from the run folder with zero model calls (src/cascade/drift.py, thresholds in drift_thresholds.json with a basis string each). Baseline comparisons are drawn per asset class, only when the run's model identity (fingerprint model_id: models, prompts, rubrics, grader backend) equals the class card's active id; otherwise the run is a candidate. Nothing is computed from eval_v1 except the one-look ledger: baseline freezes refuse eval_v1 records and gate recall from eval_v1 labels is skipped.")
    all_runs = list_runs()
    if not all_runs:
        st.info("No runs yet.")
    else:
        h1, h2 = st.columns([3, 1])
        hpick = h1.selectbox("Run", all_runs, index=all_runs.index(active_run) if active_run in all_runs else 0, key="health_pick")
        hrun = RUNS / hpick
        imgs_h = load_records_for_run(hrun)
        hd = health_for(hrun, imgs_h, refresh=h2.button("Recompute health.json", key="health_recompute"))
        if not hd:
            st.info("health.json could not be computed for this run.")
        else:
            fpp = hrun / "fingerprint.json"
            fp_doc = json.loads(fpp.read_text(encoding="utf-8")) if fpp.exists() else None
            # 1. status strip
            st.markdown("#### Status")
            st.markdown(status_badge(hd["status"], f"run `{hpick}` · fingerprint `{hd['fingerprint_id'] or 'none'}` · model identity `{hd.get('model_id') or 'none'}` · baseline `{hd['baseline_id'] or 'candidate: no baseline'}`" + (f" · comparable classes {', '.join(hd['comparable_classes'])}" if hd.get("comparable_classes") else "") + f" · thresholds {hd['thresholds_sha8']} · generated {hd['generated_at'][:19]}"), unsafe_allow_html=True)
            sc = st.columns(len(hd["stages"]))
            for col, (stage, s) in zip(sc, hd["stages"].items()):
                detail = f"n = {s['n']}" + (f", need {s['n_required']}" if s["n_required"] else "") + (f"<br><small>{s['worst_rule']}</small>" if s["worst_rule"] else "")
                col.markdown(f"**{stage}**<br>{status_badge(s['status'])}<br><small>{detail}</small>", unsafe_allow_html=True)
            mps = hd["models"]["models_per_stage"]
            st.caption(("Models per stage " + ("(reconstructed from calls.jsonl; this run predates fingerprint.json): " if not fp_doc else "(requested): ") + "; ".join(f"{s}: {', '.join(ms)}" for s, ms in mps.items()))
                       + (f" · served-model mismatches: {len(hd['models']['served_mismatch'])}" if hd["models"]["served_mismatch"] else " · served model ids match the requested ids where recorded"))
            for kind, items in (("alarm", hd["alarms"]), ("watch", hd["watch"])):
                if items:
                    st.markdown(f"**{kind.capitalize()}s** (alert budget: one per stage, ranked)")
                    st.table(pd.DataFrame([{k_: (", ".join(map(str, a[k_])) if isinstance(a[k_], list) else str(a[k_])) for k_ in ("rank", "stage", "rule", "severity", "value", "threshold", "n", "action")} | ({"demoted_from": a["demoted_from"]} if a.get("demoted_from") else {}) for a in items]))
            if hd["info"]:
                st.caption("Info: " + "; ".join(f"{i.get('rule')}: {i.get('detail', i.get('action', ''))}" for i in hd["info"]))
            # 2. fingerprint diff
            st.markdown("#### Fingerprint")
            if not fp_doc:
                st.info("This run predates fingerprint.json; no component list to diff. New runs write one before the first model call.")
            else:
                others = [r for r in all_runs if r != hpick and (RUNS / r / "fingerprint.json").exists()]
                other = st.selectbox("Compare with run", ["(none)"] + others, key="health_fp_other")
                if other != "(none)":
                    fp_b = json.loads((RUNS / other / "fingerprint.json").read_text(encoding="utf-8"))
                    diff = {name: (a, b) for name, a, b in drift.fingerprint_diff(fp_doc, fp_b)}
                    rows_fp = [{"component": k_, hpick: str(v)[:60], other: str(fp_b.get("components", {}).get(k_))[:60], "same": "no" if k_ in diff else "yes"} for k_, v in fp_doc.get("components", {}).items()]
                    st.caption(f"Label: {'identical fingerprint, runs comparable' if not diff else f'{len(diff)} components differ, runs are not comparable'}")
                else:
                    rows_fp = [{"component": k_, "value": str(v)[:80]} for k_, v in fp_doc.get("components", {}).items()]
                st.table(pd.DataFrame(rows_fp))
            # 3. contract audit
            c = hd["contract"]
            st.markdown(f"#### Contract audit · n = {c['n']} findings, {c['grade_calls']} grade calls")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.table(pd.DataFrame([{"rule": r, "count": v, "kind": "hard"} for r, v in c["hard"].items()] + [{"rule": r, "count": v, "kind": "soft"} for r, v in c["soft"].items()]))
                st.caption(f"hard total {c['hard_total']} on {c['findings_breached']} findings (rate {c['hard_rate']:.3f}) · parse failures {c['parse_fail']} · refusals {c['refusals']} · stop categories {', '.join(map(str, c['stop_categories'])) or 'none recorded'}")
            with cc2:
                udf = pd.DataFrame([{"cause": k_, "count": v} for k_, v in c["u_by_cause"].items()])
                st.altair_chart(alt.Chart(udf).mark_bar(color="#8b5cf6").encode(x=alt.X("cause:N", title=None), y=alt.Y("count:Q", title="U findings"), tooltip=["cause", "count"]).properties(height=200, title="U by cause"), width="stretch")
            if c["violations"]:
                with st.container():
                    st.dataframe(pd.DataFrame(c["violations"]), width="stretch", hide_index=True, height=200)
                v1, v2 = st.columns([3, 1])
                vfid = v1.selectbox("Flagged finding", sorted({v["finding_id"] for v in c["violations"]}), key="health_violation")
                if v2.button("Open in Findings & review", key="health_open_finding"):
                    st.session_state["active_run"] = hpick
                    st.session_state["finding_pick"] = vfid
                    st.rerun()
            ack1, ack2 = st.columns([1, 3])
            if ack1.button("Acknowledge alerts", key="health_ack", disabled=not (hd["alarms"] or hd["watch"])):
                hd["acknowledged"] = {"by": reviewer, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rules": [a["rule"] for a in hd["alarms"] + hd["watch"]]}
                (hrun / "health.json").write_text(json.dumps(hd, indent=1, default=str), encoding="utf-8")
                st.rerun()
            ack2.caption(f"acknowledged: {hd['acknowledged']} (carried across recomputes while the alarm rules stay inside it; releases queue.csv and bridge_entry.csv)" if hd.get("acknowledged") else "no acknowledgement recorded")
            # 4. U-rate chart across runs with the same model identity (the selected run is always drawn)
            uc = hd["u_chart"]
            st.markdown(f"#### U-rate p-chart · status: {uc['status']}")
            if uc.get("per_class"):
                st.table(pd.DataFrame([{"asset_class": k_, "n": v["n"], "U": v["u"], "rate": v["rate"], "p0": v["p0"], "UCL": v["ucl"], "LCL": v["lcl"], "status": v["status"]} for k_, v in uc["per_class"].items()]))
            same_fp = []
            for r in all_runs:
                fpr = RUNS / r / "fingerprint.json"
                fid = drift.identity_of(json.loads(fpr.read_text(encoding="utf-8"))) if fpr.exists() else None
                if (fid is not None and fid == hd.get("model_id")) or r == hpick:
                    rr_h = load_run(RUNS / r)
                    b = drift.u_breakdown(rr_h["findings"], rr_h["calls"])
                    same_fp += [{"run": r, "cause": cause, "rate": d["rate"], "count": d["count"], "n": b["n"], "enough_n": b["n"] >= (uc.get("n_required") or 10)} for cause, d in b["by_cause"].items()]
            if same_fp:
                udf2 = pd.DataFrame(same_fp)
                ubar = alt.Chart(udf2).mark_bar().encode(x=alt.X("run:N", title=None), y=alt.Y("rate:Q", title="U rate (stacked by cause)"), color=alt.Color("cause:N"), opacity=alt.condition("datum.enough_n", alt.value(1.0), alt.value(0.35)), tooltip=["run", "cause", "count", "n", "rate"])
                layers = [ubar]
                if uc["status"] in ("in_control", "alarm"):
                    for name_, val, col_ in (("p0", uc["p0"], "#22c55e"), ("UCL", uc["ucl"], "#dc2626"), ("LCL", uc["lcl"], "#3b82f6")):
                        if val is not None:
                            layers.append(alt.Chart(pd.DataFrame({"y": [val], "line": [name_]})).mark_rule(color=col_, strokeDash=[6, 4]).encode(y="y:Q", tooltip=["line", "y"]))
                st.altair_chart(alt.layer(*layers).properties(height=240, title=(f"runs with model identity {hd['model_id']}" if hd.get("model_id") else "this run only (no fingerprint to group by)") + " (faded bars: n below the floor)"), width="stretch")
            st.caption(f"this run: U {uc['u']} of {uc['n']}" + (f" = {uc['rate']:.3f}" if uc.get("rate") is not None else "") + (f" · needs n ≥ {uc['n_required']} for limits" if uc["status"] == "insufficient_n" else "") + (f" · p0 {uc['p0']:.3f}, UCL {uc['ucl']:.3f}, LCL {uc['lcl']:.3f}" if uc.get("ucl") is not None else " · limits need a baseline card with this fingerprint")
                       + (f" · not comparable: {uc['not_comparable']}" if uc.get("not_comparable") else ""))
            # 5. output shift
            st.markdown("#### Output shift versus baseline on identical images")
            sh = hd.get("output_shift")
            if not sh:
                st.info("candidate: no baseline with this fingerprint, so no shift is drawn. Freeze a baseline below once a run is reviewed.")
            else:
                st.caption(f"status {sh['status']} · n shared {sh['n_shared']} (needs {sh.get('n_required')}) · tiles differ: {sh['tiles_differ']} · abstention collapse: {sh['abstention_collapse']}")
                if sh.get("per_class"):
                    st.table(pd.DataFrame([{"asset_class": k_, **{kk: str(vv) for kk, vv in v.items()}} for k_, v in sh["per_class"].items()]))
            # 6. gate health
            g = hd["gate"]
            st.markdown("#### Gate health")
            gdf = pd.DataFrame([{"group": k_, **{kk: vv for kk, vv in v.items() if kk != "conf_bins"}} for k_, v in g["groups"].items()])
            if not gdf.empty:
                st.table(gdf.set_index("group").T.astype(str).replace({"None": "n/a"}))
                bins = pd.DataFrame([{"group": k_, "confidence": bk, "count": bv} for k_, v in g["groups"].items() for bk, bv in v["conf_bins"].items()])
                gc1, gc2 = st.columns(2)
                gc1.altair_chart(alt.Chart(bins).mark_bar().encode(x=alt.X("confidence:N", title="gate confidence bin", sort=None), y=alt.Y("count:Q"), color="group:N", tooltip=["group", "confidence", "count"]).properties(height=200, title="gate confidence"), width="stretch")
                ge = (g.get("eval") or {}).get("gate") or {}  # eval is None for every upload / video / Drop & grade run
                gc2.markdown(f"share of clean verdicts below the routing threshold: {g['below_min_conf_share']:.2f}" + (" · the threshold had no effect on this run" if g["min_conf_inert"] else "")
                             + (f"  \nrecall on labelled damaged images, gate only (forced routing excluded): {ge['recall']:.2f} (tp {ge['tp']}, fn {ge['fn']}, n {ge['n']})" if ge.get("recall") is not None else f"  \n{(g.get('eval') or {}).get('note') or 'no damage labels in this run (uploads never carry them)'}")
                             + (f"  \nmissed: {', '.join(ge['misses'])}" if ge.get("misses") else ""))
            else:
                st.info("No gate rows.")
            # 7. reviewer loop
            rv = hd["review"]
            st.markdown(f"#### Reviewer loop · n = {rv['n']} decisions · attribution: {rv['attribution']}")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"agreement all {rv['agreement_all']:.2f} · rolling {rv['agreement_rolling']:.2f}" if rv["agreement_all"] is not None else "agreement: insufficient n (needs 10 decisions)")
                st.markdown(f"overrides {rv['overrides']['n']} (up {rv['overrides']['up']}, down {rv['overrides']['down']}) · marked U {rv['marked_u']} · big overrides {len(rv['big_overrides'])} · critical misses {len(rv['critical_misses'])} (zero tolerance) · blind grades {rv['blind']['n']}")
                if rv["critical_misses"]:
                    st.error("Critical misses: " + "; ".join(map(str, rv["critical_misses"])))
                if rv["by_reviewer"]:
                    st.table(pd.DataFrame([{"reviewer": k_, **v} for k_, v in rv["by_reviewer"].items()]))
            with r2:
                if (hrun / "reviews.sqlite").exists():
                    tl_h = ReviewLog(hrun / "reviews.sqlite").timeline(hpick)
                    if tl_h:
                        st.altair_chart(agreement_chart(tl_h), width="stretch")
            with st.expander("Blind QC: grade a sampled finding without seeing the model's grade", expanded=False):
                rr_h = load_run(hrun)
                done_blind = ReviewLog(hrun / "reviews.sqlite").blind_ids(hpick) if (hrun / "reviews.sqlite").exists() else set()
                sample = drift.blind_sample(rr_h["findings"], per_class=10, seed=hd["fingerprint_id"] or hpick, exclude_ids=done_blind)
                if not sample:
                    st.info("No findings left to sample." + (f" {len(done_blind)} already blind-graded." if done_blind else ""))
                else:
                    bfid = st.selectbox("Sampled finding", [f.finding_id for f in sample], key="blind_pick")
                    bf = next(f for f in sample if f.finding_id == bfid)
                    brec = imgs_h.get(bf.evidence.image_ids[0]) if bf.evidence.image_ids else None
                    if brec and Path(brec.path).exists():
                        st.image(draw_evidence(bf, brec, neutral=True), width="stretch", caption=f"{brec.image_id} · {bf.asset_class} · tile {bf.evidence.tile} (box in neutral grey: the level colour would leak the grade)")
                    blvl = st.radio("Your unified level", [lvl for lvl in LEVELS if lvl != "U"], horizontal=True, key="blind_level")
                    if st.button("Record blind grade", key="blind_record"):
                        ReviewLog(hrun / "reviews.sqlite").record_blind(hpick, bf, blvl, reviewer)
                        health_for(hrun, imgs_h, refresh=True)
                        st.rerun()
            # 8. ops
            op = hd["ops"]
            st.markdown("#### Ops: latency, tokens, price")
            if op["by_stage_model"]:
                st.table(pd.DataFrame([{"stage/model": k_, **v} for k_, v in op["by_stage_model"].items()]).set_index("stage/model"))
            st.caption((f"unpriced models ({len(op['unpriced'])} calls): {', '.join(sorted({c['model'] for c in op['unpriced']}))}" if op["unpriced"] else "every call priced from costlog list prices (local models $0)") + (f" · ratios vs baseline: {op['ratios']}" if op.get("ratios") else ""))
            # 9. input
            inp_h = hd["input"]
            st.markdown(f"#### Input: imagery statistics · n = {inp_h['n']} records")
            share = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"  # noqa: E731
            pcts = inp_h.get("long_side_pcts") or {}
            st.table(pd.DataFrame([{"long side p10/p50/p90 px": f"{pcts.get('p10')} / {pcts.get('p50')} / {pcts.get('p90')}", "share below 256 px": share(inp_h.get("share_below_256px")),
                                     "null GSD share": share(inp_h.get("null_gsd_share")), "null irradiance share": share(inp_h.get("null_irradiance_share")), "null capture date share": share(inp_h.get("captured_null_share")),
                                     "duplicates within run": str(inp_h.get("dups_within") or "none"), "asset mix": ", ".join(f"{k_} {v}" for k_, v in (inp_h.get("asset_mix") or {}).items())}]).T.rename(columns={0: "value"}))
            if inp_h.get("video"):
                st.table(pd.DataFrame([{"source_video": Path(k_).name, **v} for k_, v in inp_h["video"].items()]))
            # 10. canary and promotion
            st.markdown("#### Canary and promotion gate")
            fp_id_h = hd["fingerprint_id"]
            model_id_h = hd.get("model_id")
            cst = drift.canary_status(model_id_h)
            ref_path = drift.CANARY_DIR / (model_id_h or "none") / "reference.json"
            eff_grader = grader if grader != "none" else "claude"  # the canary CLI has no gate-only mode: it grades with claude
            ce = cost_estimate(14, 1, gate, eff_grader)
            st.caption((f"canary reference for this model identity: {cst['reference']} (built {cst['reference_built_on'] or '?'}) · latest canary: {cst['latest_run'] or 'none'} {cst['latest_status'] or ''}" if cst["reference"] else "no canary reference for this model identity yet (python -m cascade.canary --build-reference --confirm builds one, three repeats)")
                       + f" · Run canary: 14 images, about ${ce['usd_estimate']:.2f} with gate {gate} and grader {eff_grader} ({ce['note']})")
            cn1, cn2 = st.columns([1, 3])
            confirm_canary = cn2.checkbox(f"I understand the canary calls the models (grader {eff_grader}) and spends the amount above", key="canary_confirm")
            if cn1.button("Run canary", key="canary_run", disabled=not confirm_canary):
                with st.spinner("running the frozen canary"):
                    proc = subprocess.run([sys.executable, "-m", "cascade.canary", "--confirm", "--gate", gate, "--grader", eff_grader], capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
                (st.error if proc.returncode == 1 else st.success if proc.returncode == 0 else st.warning)(f"exit {proc.returncode} (1 = alarm)\n\n```\n{(proc.stdout + proc.stderr)[-2000:]}\n```")
            ledger_rows = [json.loads(line) for line in drift.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] if drift.LEDGER.exists() else []
            looks = [r for r in ledger_rows if r.get("fingerprint") == fp_id_h or r.get("fp_id") == fp_id_h]
            cand_eval = ROOT / "eval" / "reports" / f"{hpick}.json"
            base_card = None
            classes_h = sorted({f.asset_class for f in load_run(hrun)["findings"]} or {r.asset_class for r in run_records(hrun, imgs_h).values()})
            base_card = next((card for card in (drift.load_baseline(ac) for ac in classes_h) if card), None)
            base_eval = (ROOT / "eval" / "reports" / f"{base_card['built_from'][0]}.json") if base_card and base_card.get("built_from") else None
            checks = [
                ("fingerprint.json present", bool(fp_doc)),
                ("hard contract violations = 0", c["hard_total"] == 0),
                (f"no blocking alarm ({', '.join(BLOCKING_RULES)})", not any(a["rule"] in BLOCKING_RULES for a in hd["alarms"])),
                ("no dead gate on a labelled run", not any(a["rule"] == "dead_gate" for a in hd["alarms"] + hd["watch"])),
                (f"latest canary for this model identity passed and is newer than its reference ({cst['latest_run'] or 'no canary run'}: {cst['latest_status'] or 'n/a'})", cst["ok"]),
                (f"one-look eval report for this run (eval/reports/{hpick}.json)", cand_eval.exists()),
                ("baseline eval report to compare against", bool(base_eval and base_eval.exists())),
                (f"eval ledger looks for this fingerprint: {len(looks)} (one allowed)", len(looks) <= 1),
            ]
            st.markdown("\n".join(f"- {'✅' if ok else '⬜'} {label}" for label, ok in checks))
            pc = None
            if cand_eval.exists() and base_eval and base_eval.exists():
                pc = drift.promotion_check(json.loads(cand_eval.read_text(encoding="utf-8")), json.loads(base_eval.read_text(encoding="utf-8")), c, canary=cst)
                st.caption(f"promotion_check verdict: {pc['verdict']} · gate recall delta {pc['gate_recall_delta']} · usd ratio {pc['usd_ratio']} · " + ("; ".join(pc["reasons"]) or "no holds"))
                st.table(pd.DataFrame([{"asset_class": k_, **v} for k_, v in pc["per_class"].items()]))
            can_promote = all(ok for _, ok in checks) and pc is not None and pc["verdict"] == "promote"
            p1, p2, p3 = st.columns(3)
            if p1.button("Promote this fingerprint", key="health_promote", disabled=not can_promote, help="greyed until every check above passes and promotion_check says promote"):
                try:
                    path = drift.promote(fp_doc, hpick, ref_path, str(cand_eval), reviewer)
                    st.success(f"promoted: {path}")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
            if p2.button("Rollback to the previous baseline", key="health_rollback", disabled=not (drift.BASELINE_DIR / "active.json").exists()):
                try:
                    st.json(drift.rollback())
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
            fz_class = p3.selectbox("Freeze this run as the baseline card for", classes_h or ["(no findings)"], key="health_freeze_class")
            if p3.button("Freeze baseline", key="health_freeze", disabled=not classes_h, help="writes runs/_baseline/<asset_class>.json keyed by the model identity; later runs with the same identity are charted against it; refuses a run holding eval_v1 records"):
                try:
                    card = drift.baseline_freeze(hrun, fz_class)
                    st.success(f"baseline card written: built from {card.get('built_from')}, fingerprint {card.get('fingerprint', hd['fingerprint_id'])}")
                    health_for(hrun, imgs_h, refresh=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
            st.caption(f"Changelog line that promote() records (preview): promote fingerprint {fp_id_h or 'none'} from dev run {hpick}, eval report {cand_eval.name}, by {reviewer}. Sample-size helper: to detect a U-rate move from 0.10 to 0.25 at alpha 0.01, power 0.8 you need n = {drift.min_n_for_shift(0.10, 0.25)} findings per run.")

# ---------- Why this approach ----------

with tab_why:
    st.subheader("How this differs from the incumbents")
    st.caption("Incumbent facts are from the team's web research of 2026-09-24 (docs/research/R01, R02, R06, R08, R09). No head-to-head benchmark has been run; the right-hand column is measured from the run selected in the sidebar, or blank.")
    live = run_metrics(RUNS / active_run, load_records_for_run(RUNS / active_run)) if active_run and (RUNS / active_run / "gate.jsonl").exists() else None

    def measured(fn, default="not measured yet"):
        try:
            return fn() if live else default
        except Exception:
            return default

    rows = [
        ("Asset coverage", "Single-vertical: SkySpecs, Clobotics, Cornis on blades; Raptor Maps, Sitemark, Above on solar; Buzz, Sharper Shape, Hepta on grid (R01, R02)",
         "One engine, four asset classes: bridge steel coating, bridge concrete elements, PV thermal modules, post-disaster buildings; rubrics are data, not code",
         measured(lambda: ", ".join(live["asset_classes"]) or "none graded")),
        ("Grading vocabulary", "\"Severity ratings\" with no published definitions; none ships MBEI condition states with clauses (R01, R02)",
         "Native scale first (MBEI CS1 to CS4, IEC TS 62446-3 CoA, FEMA PDA), the rubric row quoted verbatim in every finding, unified S0 to S4 overlay",
         measured(lambda: f"{live['findings']} findings, each with quoted criteria")),
        ("Rationale per finding", "No per-finding rationale published by any incumbent (R01, R02, R09)",
         "Justification, measurements or not_measurable, confidence, +/-1 uncertainty, evidence crop with bbox",
         measured(lambda: f"{live['graded_images']} images with evidence crops")),
        ("Unassessable state", "Not exposed; a refusal or bad image silently becomes 'no finding'",
         "Explicit U: refusals, missing GSD and unusable frames are U, never S0, and are listed in the queue",
         measured(lambda: f"U = {live['U']} of {live['findings']}")),
        ("Hallucination control", "Zero-shot VLM grading: 65% hallucination in the 2026 hybrid study; frontier VLMs weak at small defects (R02, R06)",
         "Cascade: cheap gate, crop, then schema-enforced grade constrained to rubric values; hybrid pipelines reached 4% in the same study",
         measured(lambda: f"routed {live['routed']} of {live['gated']} ({100 * live['routed'] / max(1, live['gated']):.0f}%)")),
        ("Prioritization", "Findings framed in AEP or dollars inside one vertical; no cross-asset consequence-ranked queue found (R01, R02)",
         "Deterministic queue: severity × criticality × consequence × urgency; any S4 same-day; multipliers shown per row",
         measured(lambda: f"S4 = {live['S4']}, S3 = {live['S3']}")),
        ("Human sign-off", "Not published",
         "Mandatory accept / override / mark U with the prior model value logged; agreement rate measured per run",
         measured(lambda: f"{live['reviews']} reviewer decisions logged")),
        ("Review labor", "Throughput claims only (Buzz 25,000+ images/hr); no public per-image review time for bridges or solar; T&D World 3 to 5 min/image is utility imagery (R08)",
         "Gate auto-clears clean images, the reviewer opens one pre-filled, rubric-cited finding per routed image, a random audit covers the cleared set; hours and dollars are estimates under Inspect run > Overview > Workforce impact, never measured",
         measured(lambda: f"routed {live['routed']} of {live['gated']}, auto-cleared {live['gated'] - live['routed']}, U = {live['U']}")),
        ("Video and drone footage", "Frame review is manual or vendor-specific; no published dedup rule",
         "ffmpeg sampling at a fixed interval or on scene change, dHash near-duplicate drop, every kept frame carries source video and timestamp into the finding id",
         measured(lambda: f"{sum(1 for r in load_records_for_run(RUNS / active_run).values() if r.source_video)} video frames in this run" if (RUNS / active_run / 'videos').exists() else "no video in this run")),
        ("Unit cost", "Quote-only pricing except Scopito (EUR 160 / 80 per turbine); $100 to $300 analytics add-on per turbine (R01, R04, R09)",
         "About $10 per 1,000 frames heavy-only, about $3 with a local gate at 30% pass-through (R06 estimate, Sonnet 5 Sept 2026 prices)",
         measured(lambda: f"${live['usd_per_image']:.4f} per image = ${1000 * live['usd_per_image']:.2f} per 1,000" if live["usd_per_image"] is not None else "no priced calls")),
        ("Compliance output", "RFPs require state-system entry by deadline, CS3/CS4 photo documentation, a per-structure list; one county fines $100 per day late (R08)",
         "bridge_entry.csv (element, condition state, quantity) and queue.csv per run; evidence crops per finding",
         measured(lambda: "bridge_entry.csv written" if (RUNS / active_run / "bridge_entry.csv").exists() else "no bridge findings in this run")),
        ("Disaster surge", "Insurers deliver building-level classes in 24 to 48 h (ICEYE, Nearmap, Vexcel); nobody grades engineered infrastructure post-event (R03)",
         "Surge posture on the same engine: whole-frame FEMA PDA counts and a report; infrastructure surge is roadmap, not demo",
         measured(lambda: "surge run" if live["mode"] == "surge" else "inspect run")),
        ("Sensor modalities (D-014)", "Structural-monitoring platforms (Move Solutions, Worldsensing, Resensys, Bentley iTwin IoT) are quote-only and single-sensor, none publishes a price or a grading rule; imaging-sonar vendors publish specs, not prices (R10 sections 1.3 and 6)",
         "One queue for six modalities (PRD 5A): RGB and thermal images through the cascade; seismic and vibration CSVs graded on rubric rows in plain code, U without a baseline; sonar and machinery rubric files only, not run; lidar roadmap. No accuracy is claimed for seismic, sonar or lidar",
         measured(lambda: ", ".join(f"{k} {v}" for k, v in pd.Series([f.modality for f in load_run(RUNS / active_run)["findings"]]).value_counts().items()) or "no findings")),
        ("Crack width in mm", "Scale-referenced methods report 0.22 mm precision (planar markers) and 0.16 mm MAE (laser calibration) in the literature (R10 section 4.2, [43] [44])",
         "Measured only when a scale exists (GSD metadata or two points on a ruler), shown as width ± uncertainty with its basis; a band that straddles an MBEI boundary lists both states; no scale keeps not_measurable. The mask is a heuristic with no measured error of its own",
         measured(lambda: f"{sum(1 for f in load_run(RUNS / active_run)['findings'] if f.measurements.measurement_basis)} findings with a measured basis")),
        ("Consistency baseline", "Human inspectors: 68% of ratings within one point across 49 inspectors (FHWA); 30% matched expected (Indiana 2026) (R08)",
         "Within-one-grade and weighted kappa reported per asset class with n and CI, against frozen label maps",
         measured(lambda: "; ".join(f"{ac}: within-1 {100 * gr['within_one_grade']:.0f}% (n={gr['n_assessed']})" for ac, gr in live.get("eval", {}).get("grading", {}).items() if gr["n_assessed"]) or "no labelled grades in this run")),
    ]
    head = ["Capability", "Incumbents (as researched)", "This cascade", f"Measured on `{active_run}`" if active_run else "Measured (no run open)"]
    table = ["| " + " | ".join(head) + " |", "|---|---|---|---|"]
    table += ["| **" + r[0] + "** | " + " | ".join(str(x).replace("|", "/") for x in r[1:]) + " |" for r in rows]
    st.markdown(chr(10).join(table))
    st.markdown(
        """
**What is deliberately not claimed.** No field accuracy: all numbers come from public datasets. No trained model: the business is rubrics,
evaluation and workflow (R07). No comparison benchmark against any named vendor exists yet. The R06 cost figures are estimates from
list prices; the measured column is the only number from this codebase. No accuracy is claimed for seismic, sonar or lidar: the seismic
module has run on a labelled synthetic pair only, its frequency thresholds are team-proposed (validate), sonar has a rubric file and no
data run, lidar has no code. Crack widths in mm carry a heuristic mask with no measured error of its own.
"""
    )
