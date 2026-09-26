"""FR-4b: video-based detection output. A finished run's frames become a per-video timeline, an annotated
detections mp4, WebVTT cues and a contact sheet.

Everything here is read from what the run already wrote (`manifest.jsonl`, `gate.jsonl`, `findings.json`,
`videos/<slug>.json`, the kept frame JPEGs). Nothing is graded here: a frame the cascade never graded keeps
`level: null` and is drawn as "not graded", and U is its own count, never folded into S0.

The mp4 is a slideshow of the kept sample frames, not the source clip with overlays: the annotated frames go into
`videos/<slug>_annotated/frame_%05d.jpg`, a concat-demuxer list gives each one its hold (the next frame's
`t_s` minus its own; the last frame holds until the clip end recorded in frames.json, `min(video.duration_s,
end_s)`, else the sampling step), `fps=<fps_out>` makes the stream constant-rate and `-t` pins the length to the
sum of the holds; ffmpeg runs through `video._run` exactly as extraction does. Source frames between samples were
never graded and never appear. The mp4 starts at the first kept frame, so its clock is source time minus
`mp4_offset_s` (= frames[0].t_s, non-zero for clips extracted with --start); the WebVTT cues are written on the
mp4 clock, the on-frame "t = 12.0 s" stamp and the timeline stay in source time.
tests/test_videodetect.py checks the result with ffprobe.

Measured on this laptop (ffmpeg 8.0.1 gyan build, Pillow 12.3, 2026-09-25, tests/test_videodetect.py):
  - data/demo/video/bridge_walkthrough.mp4 (six frames every 2 s, fake backends): the detections mp4 probes as
    12.0 s, 1280x960 @ 10 fps, h264, 1,355,108 bytes; the concat list with the last file repeated plus `-t`
    gives the exact source length (without the repeat the last frame's duration is dropped by the demuxer).
  - JPEG q=92 shifts the badge colour by up to 6 per channel (226,38,37 for #dc2626); PNG keeps it exact.

Usage:
  python -m cascade.videodetect --run runs/<run> [--slug bridge_walkthrough]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from . import video as _video
from .clientreport import LEVEL_COLOR, worst_level
from .ingest import read_manifest
from .pipeline import load_run
from .schema import Finding, ImageRecord
from .video import VideoError, _run, _stderr_tail, ffmpeg_available, slug_of

LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "U"]
DETECTION_LEVELS = ("S1", "S2", "S3", "S4")
NOT_GRADED = "not graded"
NOT_GRADED_COLOR = "#374151"
TIME_COLOR = "#111827"
_U_REASON_CHARS = 60


def level_ordinal(level: Optional[str]) -> int:
    """S0..S4 -> 0..4 and U -> 5 (U sorts after S4 for display only; it is never S0); None or unknown -> -1."""
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else -1


# ----------------------------------------------------------------------------- timeline


def _sidecar(run_dir: Path, slug: str) -> dict:
    """`videos/<slug>.json` (the frames.json copy the app stores) or {} when absent or unreadable."""
    p = run_dir / "videos" / f"{slug}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _top_finding(fs: Sequence[Finding]) -> Optional[Finding]:
    """The finding that sets a frame's level: worst level (any graded level beats U), then best queue rank, then id."""
    wl = worst_level(f.unified.level for f in fs)
    cands = [f for f in fs if f.unified.level == wl]
    return min(cands, key=lambda f: (f.queue_rank is None, f.queue_rank or 0, f.finding_id)) if cands else None


def _frame_entry(rec: ImageRecord, gate_row: Optional[dict], fs: Sequence[Finding]) -> dict:
    """One timeline frame. `level` etc. come from the top finding; `bbox`/`tile` are the top finding's box (kept for
    older readers) and `boxes` lists every tiled finding's box with its own level, so a frame with several graded
    tiles keeps all of its localisations."""
    top = _top_finding(fs)
    boxed = top is not None and top.evidence.tile != "full" and len(top.evidence.bbox) == 4
    ordered = sorted(fs, key=lambda f: (f.queue_rank is None, f.queue_rank or 0, f.finding_id))
    boxes = [
        {"tile": f.evidence.tile, "bbox": [int(v) for v in f.evidence.bbox], "level": f.unified.level, "finding_id": f.finding_id}
        for f in ordered
        if f.evidence.tile != "full" and len(f.evidence.bbox) == 4
    ]
    return {
        "t_s": float(rec.frame_time_s),
        "image_id": rec.image_id,
        "file": rec.path,
        "gate": {k: gate_row[k] for k in ("usable", "damage_present", "confidence", "routed")} if gate_row else None,
        "level": top.unified.level if top else None,
        "native_value": top.native_scale.value if top else None,
        "standard": top.native_scale.standard if top else None,
        "action": top.action.code if top else None,
        "n_findings": len(fs),
        "bbox": [int(v) for v in top.evidence.bbox] if boxed else None,
        "tile": top.evidence.tile if top else None,
        "boxes": boxes,
        "finding_ids": [f.finding_id for f in ordered],
        "queue_rank": top.queue_rank if top else None,
        "u_reason": top.justification if top and top.unified.level == "U" else None,
    }


def _step_s(sidecar: dict, times: Sequence[float]) -> tuple[float, str]:
    """Nominal spacing of the kept frames and where it comes from. Interval mode: `every_s` from frames.json.
    Scene mode: `scene_max_gap_s` when set, else the measured median gap (frames.json still carries `every_s`
    there, but it is the unused interval default, never how the frames were chosen). No frames.json: the median
    gap; a single frame with nothing else: 1 s. The last frame's hold prefers the clip end (`_clip_end`)."""
    mode = sidecar.get("mode")
    if mode == "scene":
        if sidecar.get("scene_max_gap_s"):
            return float(sidecar["scene_max_gap_s"]), "scene mode: max gap (frames.json)"
    elif sidecar.get("every_s"):
        return float(sidecar["every_s"]), "frames.json"
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    if gaps:
        return round(gaps[len(gaps) // 2], 3), "scene mode: median frame gap" if mode == "scene" else "median frame gap"
    return 1.0, "default: single frame and no frames.json"


def _clip_end(sidecar: dict) -> Optional[float]:
    """End of the sampled span in source seconds, from frames.json: `end_s` capped at `video.duration_s`, else the
    duration; None without a probed duration."""
    vdur = (sidecar.get("video") or {}).get("duration_s")
    if not vdur:
        return None
    end = sidecar.get("end_s")
    return round(min(float(vdur), float(end)) if end else float(vdur), 3)


def _seg_key(d: dict) -> tuple:
    return (d["level"], d["native_value"], d["standard"], d["action"])


def _segments(frames: Sequence[dict]) -> List[dict]:
    """Consecutive frames with the same (level, native_value, standard, action) merge, so a cue never labels a frame
    with a native value it did not receive; `end_s` is the next frame's t_s (the last one t_s + hold)."""
    segs: List[dict] = []
    for fr in frames:
        end = round(fr["t_s"] + fr["hold_s"], 3)
        if segs and _seg_key(segs[-1]) == _seg_key(fr):
            segs[-1]["end_s"] = end
            segs[-1]["n_frames"] += 1
        else:
            segs.append({"start_s": fr["t_s"], "end_s": end, "level": fr["level"], "native_value": fr["native_value"],
                         "standard": fr["standard"], "action": fr["action"], "n_frames": 1})
    return segs


def _summary(frames: Sequence[dict]) -> dict:
    levels = {lvl: sum(1 for f in frames if f["level"] == lvl) for lvl in LEVEL_ORDER}
    graded = [f["level"] for f in frames if f["level"] is not None]
    return {
        "frames": len(frames),
        "graded": len(graded),
        "not_graded": len(frames) - len(graded),
        "levels": levels,
        "worst_level": worst_level(graded),
        "first_detection_t_s": next((f["t_s"] for f in frames if f["level"] in DETECTION_LEVELS), None),
        "u_frames": levels["U"],
    }


def video_timeline(run_dir: Path, records: Optional[Dict[str, ImageRecord]] = None) -> Dict[str, dict]:
    """slug -> timeline entry for every distinct `source_video` in the run's records.

    `records` defaults to `<run>/manifest.jsonl`; frames without `frame_time_s` cannot be placed and are left out.
    Each entry: video, slug, mode ("interval" | "scene" | null), every_s (+ every_s_source), scene_threshold,
    scene_max_gap_s, clip_end_s (+ last_hold_source), mp4_offset_s (first frame's t_s: the detections mp4's time 0),
    fps, duration_s, width, height, frames (sorted by t_s, each with gate, level or null, native_value, standard,
    action, n_findings, bbox or null, boxes, finding_ids, hold_s), segments and summary. Every hold is the gap to
    the next kept frame; the last frame holds until the clip end from frames.json, else one sampling step. Pure
    apart from reading the run folder; it never assigns a level the run did not produce.
    """
    run_dir = Path(run_dir)
    if records is None:
        mp = run_dir / "manifest.jsonl"
        records = {r.image_id: r for r in read_manifest(mp)} if mp.exists() else {}
    by_video: Dict[str, List[ImageRecord]] = {}
    for r in records.values():
        if r.source_video and r.frame_time_s is not None:
            by_video.setdefault(r.source_video, []).append(r)
    if not by_video:
        return {}
    run = load_run(run_dir)
    gate_by = {g["image_id"]: g for g in run["gate"]}
    f_by: Dict[str, List[Finding]] = {}
    for f in run["findings"]:
        for iid in f.evidence.image_ids:
            f_by.setdefault(iid, []).append(f)
    out: Dict[str, dict] = {}
    for source, recs in by_video.items():
        base = slug_of(Path(source))
        slug, n = base, 1
        while slug in out:  # two clips whose stems collapse to the same slug never share an entry
            n += 1
            slug = f"{base}_{n}"
        side = _sidecar(run_dir, slug)
        vinfo = side.get("video") or {}
        frames = sorted((_frame_entry(r, gate_by.get(r.image_id), f_by.get(r.image_id, [])) for r in recs), key=lambda f: f["t_s"])
        every_s, source_of_step = _step_s(side, [f["t_s"] for f in frames])
        clip_end = _clip_end(side)
        if clip_end is not None and clip_end > frames[-1]["t_s"]:
            last_hold, last_hold_source = clip_end - frames[-1]["t_s"], "clip end from frames.json"
        else:
            last_hold, last_hold_source = every_s, source_of_step
        for i, fr in enumerate(frames):
            fr["hold_s"] = round((frames[i + 1]["t_s"] - fr["t_s"]) if i + 1 < len(frames) else last_hold, 3)
        out[slug] = {
            "video": vinfo.get("path") or source, "slug": slug, "mode": side.get("mode"), "every_s": every_s, "every_s_source": source_of_step,
            "scene_threshold": side.get("scene_threshold"), "scene_max_gap_s": side.get("scene_max_gap_s"),
            "clip_end_s": clip_end, "last_hold_source": last_hold_source, "mp4_offset_s": frames[0]["t_s"],
            "fps": vinfo.get("fps"), "duration_s": vinfo.get("duration_s"), "width": vinfo.get("width"), "height": vinfo.get("height"),
            "frames": frames, "segments": _segments(frames), "summary": _summary(frames),
        }
    return out


def _entry(run_dir: Path, slug: str, timeline: Optional[dict]) -> dict:
    """Accept the whole mapping, one entry, or None (recomputed from the run)."""
    if timeline is not None and "frames" in timeline:
        return timeline
    entry = (timeline if timeline is not None else video_timeline(run_dir)).get(slug)
    if entry is None:
        raise VideoError(f"{slug}: no video frames in {run_dir} (manifest rows need source_video and frame_time_s)")
    return entry


# ----------------------------------------------------------------------------- drawing (PIL only)


def _font_for(img: Image.Image, font) -> ImageFont.ImageFont:
    if font is not None:
        return font
    try:
        return ImageFont.load_default(size=max(14, img.height // 32))
    except (TypeError, OSError):  # Pillow without FreeType: bitmap font, fixed size
        return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, font, pad: int) -> int:
    """Filled box with white text at `xy`; returns the y just below it."""
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    x, y = xy
    draw.rectangle([x, y, x + (r - l) + 2 * pad, y + (b - t) + 2 * pad], fill=fill)
    draw.text((x + pad - l, y + pad - t), text, fill="#ffffff", font=font)
    return y + (b - t) + 2 * pad


def badge_text(frame: dict) -> str:
    """What the badge says: level, native value and action for graded frames (plus "+N tiles" when the frame has
    more than one finding, the badge showing the worst); "not graded" for level null."""
    lvl = frame.get("level")
    if lvl is None:
        return NOT_GRADED
    n = frame.get("n_findings") or 0
    more = f"  +{n - 1} tiles" if n > 1 else ""
    if lvl == "U":
        return f"U  unassessable  {frame.get('action') or ''}".rstrip() + more
    return f"{lvl}  {frame.get('native_value')} on {frame.get('standard')}  {frame.get('action')}{more}"


def _boxes_of(frame: dict) -> List[dict]:
    """Boxes to draw: `boxes` (one per tiled finding) when the entry carries them, else the top finding's `bbox`."""
    if frame.get("boxes") is not None:
        return list(frame["boxes"])
    if frame.get("bbox"):
        return [{"bbox": frame["bbox"], "level": frame.get("level"), "tile": frame.get("tile")}]
    return []


def annotate_frame(img: Image.Image, frame: dict, *, font=None) -> Image.Image:
    """Copy of `img` with one level-coloured box per tiled finding (only when tile != "full" and bbox has 4 values;
    an S0 tile reported no defect and gets no box that would read as a detection), a filled badge top-left,
    "U: <short reason>" under it for U frames, and "t = 12.0 s" bottom-left. A frame with level null gets a grey
    "not graded" badge and no level text or box anywhere."""
    out = img.convert("RGB")
    if out is img:
        out = img.copy()
    d = ImageDraw.Draw(out)
    fnt = _font_for(out, font)
    lvl = frame.get("level")
    color = LEVEL_COLOR.get(lvl, NOT_GRADED_COLOR) if lvl is not None else NOT_GRADED_COLOR
    if lvl is not None:
        for bx in _boxes_of(frame):
            bb = bx.get("bbox")
            if bx.get("level") in (None, "S0") or not bb or len(bb) != 4 or bx.get("tile") == "full":
                continue
            x0, y0, x1, y1 = bb
            d.rectangle([x0, y0, max(x0, x1), max(y0, y1)], outline=LEVEL_COLOR.get(bx["level"], NOT_GRADED_COLOR), width=max(3, out.width // 300))
    m = max(6, out.width // 160)
    pad = max(3, m // 2)
    y = _label(d, (m, m), badge_text(frame), color, fnt, pad)
    if lvl == "U" and frame.get("u_reason"):
        reason = " ".join(str(frame["u_reason"]).split())
        short = reason if len(reason) <= _U_REASON_CHARS else reason[: _U_REASON_CHARS - 3] + "..."
        _label(d, (m, y + pad), f"U: {short}", TIME_COLOR, fnt, pad)
    stamp = f"t = {float(frame['t_s']):.1f} s"
    l, t, r, b = d.textbbox((0, 0), stamp, font=fnt)
    _label(d, (m, out.height - m - (b - t) - 2 * pad), stamp, TIME_COLOR, fnt, pad)
    return out


def _annotated(entry: dict) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    font = None
    for fr in entry["frames"]:
        with Image.open(fr["file"]) as im:
            im.load()
            font = _font_for(im, font)
            imgs.append(annotate_frame(im, fr, font=font))
    return imgs


# ----------------------------------------------------------------------------- outputs


def write_detection_video(run_dir: Path, slug: str, *, timeline: Optional[dict] = None, out: Optional[Path] = None, fps_out: int = 10) -> Path:
    """Annotate every kept frame, hold each for its `hold_s`, encode libx264/yuv420p/+faststart to
    `videos/<slug>_detections.mp4` (or `out`). Raises VideoError when ffmpeg is missing or fails."""
    run_dir = Path(run_dir)
    ok, msg = ffmpeg_available()
    if not ok:
        raise VideoError(f"cannot write {slug}_detections.mp4: {msg}")
    if not 1 <= int(fps_out) <= 60:
        raise VideoError(f"fps_out must be between 1 and 60, got {fps_out}")
    entry = _entry(run_dir, slug, timeline)
    if not entry["frames"]:
        raise VideoError(f"{slug}: no frames to render")
    ann_dir = run_dir / "videos" / f"{slug}_annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for stale in ann_dir.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)
    lines = ["ffconcat version 1.0"]
    total = 0.0
    name = ""
    for i, (fr, im) in enumerate(zip(entry["frames"], _annotated(entry)), start=1):
        name = f"frame_{i:05d}.jpg"
        im.save(ann_dir / name, format="JPEG", quality=92)
        hold = max(float(fr["hold_s"]), 1.0 / fps_out)
        lines += [f"file '{name}'", f"duration {hold:.3f}"]
        total += hold
    lines.append(f"file '{name}'")  # concat demuxer quirk: the last file is listed again so its duration is honoured
    list_path = ann_dir / "frames.ffconcat"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = Path(out) if out is not None else run_dir / "videos" / f"{slug}_detections.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_video.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
           "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={int(fps_out)}", "-t", f"{total:.3f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    cp = _run(cmd, timeout_s=120.0 + 2.0 * total)
    if cp.returncode != 0:
        raise VideoError(f"{slug}: ffmpeg failed ({_stderr_tail(cp)})")
    return out


def _vtt_ts(s: float) -> str:
    ms = int(round(float(s) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    sec, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def cue_text(seg: dict) -> str:
    """"S3 - CS3 on MBEI-CS - prioritize", "U - unassessable", or "not graded"."""
    lvl = seg.get("level")
    if lvl is None:
        return NOT_GRADED
    if lvl == "U":
        return "U - unassessable"
    return f"{lvl} - {seg.get('native_value')} on {seg.get('standard')} - {seg.get('action')}"


def write_subtitles(timeline_entry: dict, out: Path, *, offset_s: Optional[float] = None) -> Path:
    """WebVTT with one cue per segment on the detections mp4's clock. The mp4 starts at the first kept frame, so
    `offset_s` (default: the entry's `mp4_offset_s`, else its first frame's t_s, else 0) is subtracted from the
    segments' source times; a clip extracted with --start 30 gets its first cue at 00:00:00.000."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if offset_s is None:
        offset_s = timeline_entry.get("mp4_offset_s")
        if offset_s is None:
            frames = timeline_entry.get("frames") or []
            offset_s = frames[0]["t_s"] if frames else 0.0
    t0 = float(offset_s)
    parts = ["WEBVTT", ""]
    for i, seg in enumerate(timeline_entry["segments"], start=1):
        parts += [str(i), f"{_vtt_ts(max(0.0, seg['start_s'] - t0))} --> {_vtt_ts(max(0.0, seg['end_s'] - t0))}", cue_text(seg), ""]
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def write_contact_sheet(run_dir: Path, slug: str, timeline_entry: Optional[dict] = None, out: Optional[Path] = None, cols: int = 6, thumb: int = 320) -> Path:
    """PNG grid of the annotated frames (badge and timestamp are drawn at full size, then scaled)."""
    run_dir = Path(run_dir)
    entry = _entry(run_dir, slug, timeline_entry)
    imgs = _annotated(entry)
    if not imgs:
        raise VideoError(f"{slug}: no frames for a contact sheet")
    cols = max(1, min(int(cols), len(imgs)))
    tiles = []
    for im in imgs:
        im.thumbnail((thumb, thumb), Image.LANCZOS)
        tiles.append(im)
    cw, ch = max(t.width for t in tiles), max(t.height for t in tiles)
    gap = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw + (cols + 1) * gap, rows * ch + (rows + 1) * gap), "#1f2937")
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (gap + c * (cw + gap), gap + r * (ch + gap)))
    out = Path(out) if out is not None else run_dir / "videos" / f"{slug}_contact.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG", optimize=True)
    return out


def write_video_detections(run_dir: Path, records: Optional[Dict[str, ImageRecord]] = None) -> Dict[str, dict]:
    """slug -> {timeline, mp4, vtt, sheet, summary[, skipped]} for every video in the run; {} when it has no video frames.
    The mp4 is skipped (reason under "skipped", "mp4" None) when ffmpeg is unavailable; the other files are still written."""
    run_dir = Path(run_dir)
    tl = video_timeline(run_dir, records)
    if not tl:
        return {}
    vdir = run_dir / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    ok, msg = ffmpeg_available()
    result: Dict[str, dict] = {}
    for slug, entry in tl.items():
        tpath = vdir / f"{slug}_timeline.json"
        tpath.write_text(json.dumps(entry, indent=1), encoding="utf-8")
        item: dict = {"timeline": tpath, "mp4": None, "vtt": write_subtitles(entry, vdir / f"{slug}_detections.vtt"),
                      "sheet": write_contact_sheet(run_dir, slug, entry), "summary": entry["summary"]}
        if ok:
            item["mp4"] = write_detection_video(run_dir, slug, timeline=entry)
        else:
            item["skipped"] = f"mp4 skipped: {msg}"
        result[slug] = item
    return result


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m cascade.videodetect", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder (runs/<run>) with manifest.jsonl, gate.jsonl and findings.json")
    ap.add_argument("--slug", default=None, help="only this video slug (default: every video in the run)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run)
    try:
        records = None
        if args.slug is not None:
            mp = run_dir / "manifest.jsonl"
            rows = read_manifest(mp) if mp.exists() else []
            records = {r.image_id: r for r in rows if r.source_video and slug_of(Path(r.source_video)) == args.slug}
            if not records:
                have = sorted({slug_of(Path(r.source_video)) for r in rows if r.source_video})
                raise VideoError(f"{args.slug}: not a video slug of {run_dir} (have: {', '.join(have) or 'none'})")
        result = write_video_detections(run_dir, records)
    except VideoError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
