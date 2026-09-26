"""FR-4b video detections: timeline from a run, annotated frames, detections mp4, WebVTT, contact sheet.

The run is built once per module from data/demo/video/bridge_walkthrough.mp4 (every 2 s, no dedup, so all six
stills are kept) through `run_cascade` with the fake backends of test_pipeline. The fakes pick their verdict from
an integer id suffix, so the wrappers map each frame's `_t<ms>` suffix to `img_<ms // 2000>`: frame 0 is clean and
not routed (never graded), frame 1 is unusable (U), frame 3 grades S4, the rest S2. ffmpeg-dependent tests skip
when ffmpeg/ffprobe are not on PATH; the drawing, subtitle and ordering tests run everywhere.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageColor

from cascade import video, videodetect
from cascade.costlog import CallLog
from cascade.ingest import read_manifest, write_manifest
from cascade.pipeline import RunConfig, run_cascade
from cascade.schema import Evidence
from cascade.video import VideoError, ingest_video, probe
from cascade.videodetect import LEVEL_ORDER, annotate_frame, level_ordinal, video_timeline, write_contact_sheet, write_detection_video, write_subtitles, write_video_detections

from test_pipeline import fake_gate, fake_grade, make_records

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
DEMO = Path(__file__).resolve().parents[1] / "data" / "demo" / "video" / "bridge_walkthrough.mp4"
SLUG = "bridge_walkthrough"
EXPECTED_LEVELS = [None, "U", "S2", "S4", "S2", "S2"]


def _idx(image_id: str) -> int:
    return int(image_id.rsplit("_t", 1)[1]) // 2000


def _gate_by_frame(img, image_id, **kw):
    return fake_gate(img, f"img_{_idx(image_id)}", **kw).model_copy(update={"image_id": image_id})


def _grade_by_frame(img, **kw):
    kw["image_id"] = f"img_{_idx(kw['image_id'])}"  # finding_id and evidence keep the vid_ ids
    return fake_grade(img, **kw)


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory) -> Path:
    """runs/<run> laid out as the app does it: manifest.jsonl, videos/<slug>.json, gate.jsonl, findings.json."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    if not DEMO.exists():
        pytest.skip("demo clip missing")
    d = tmp_path_factory.mktemp("vd")
    records, ext = ingest_video(DEMO, d / "frames" / SLUG, asset_class="bridge_element", every_s=2, dedup_max_distance=None)
    out = d / "run"
    out.mkdir()
    write_manifest(records, out / "manifest.jsonl")
    (out / "videos").mkdir()
    shutil.copyfile(Path(ext.out_dir) / "frames.json", out / "videos" / f"{SLUG}.json")
    summary = run_cascade(records, out, RunConfig(gate="none"), gate_fn=_gate_by_frame, grade_fn=_grade_by_frame)
    assert summary["images"] == 6 and summary["findings"] == 5
    return out


# ----------------------------------------------------------------------------- ordering


def test_level_order_and_ordinal():
    assert LEVEL_ORDER == ["S0", "S1", "S2", "S3", "S4", "U"]
    assert [level_ordinal(lvl) for lvl in LEVEL_ORDER] == [0, 1, 2, 3, 4, 5]
    assert level_ordinal("U") > level_ordinal("S4")
    assert level_ordinal("U") != level_ordinal("S0")
    assert level_ordinal(None) == -1 and level_ordinal("bogus") == -1


# ----------------------------------------------------------------------------- timeline


@needs_ffmpeg
def test_timeline_frames_segments_summary(demo_run):
    tl = video_timeline(demo_run)
    assert list(tl) == [SLUG]
    e = tl[SLUG]
    assert Path(e["video"]).name == DEMO.name and e["slug"] == SLUG
    assert e["every_s"] == 2.0 and e["every_s_source"] == "frames.json" and e["mode"] == "interval"
    assert e["clip_end_s"] == pytest.approx(12.0, abs=0.05) and e["last_hold_source"] == "clip end from frames.json" and e["mp4_offset_s"] == 0.0
    assert e["fps"] == pytest.approx(10.0) and e["duration_s"] == pytest.approx(12.0, abs=0.05)
    assert (e["width"], e["height"]) == (1280, 960)

    frames = e["frames"]
    assert len(frames) == 6
    assert [f["t_s"] for f in frames] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert [f["level"] for f in frames] == EXPECTED_LEVELS
    assert [f["hold_s"] for f in frames] == [2.0] * 6
    assert all(Path(f["file"]).exists() for f in frames)
    assert [f["image_id"] for f in frames] == [f"vid_{SLUG}_t{ms:08d}" for ms in range(0, 12000, 2000)]

    f0 = frames[0]  # gated clean, never graded: no level, no native value, no finding, nothing invented
    assert f0["gate"] == {"usable": True, "damage_present": False, "confidence": 0.95, "routed": False}
    assert f0["level"] is None and f0["native_value"] is None and f0["action"] is None and f0["n_findings"] == 0
    assert f0["finding_ids"] == [] and f0["bbox"] is None and f0["u_reason"] is None
    f1 = frames[1]
    assert f1["gate"]["usable"] is False and f1["level"] == "U" and f1["native_value"] == "U" and "unusable" in f1["u_reason"]
    f3 = frames[3]
    assert f3["level"] == "S4" and f3["native_value"] == "CS4" and f3["standard"] == "MBEI-CS" and f3["action"] == "escalate"
    assert f3["tile"] == "full" and f3["bbox"] is None and f3["boxes"] == []  # whole frame graded: no box
    assert f3["finding_ids"] == [f"vid_{SLUG}_t00006000/full"] and f3["queue_rank"] == 1

    segs = e["segments"]
    assert [(s["start_s"], s["end_s"], s["level"]) for s in segs] == [(0.0, 2.0, None), (2.0, 4.0, "U"), (4.0, 6.0, "S2"), (6.0, 8.0, "S4"), (8.0, 12.0, "S2")]
    assert segs[-1]["n_frames"] == 2  # two S2 frames merged; end_s = last t_s + every_s

    s = e["summary"]
    assert s["frames"] == 6 and s["graded"] == 5 and s["not_graded"] == 1
    assert s["levels"] == {"S0": 0, "S1": 0, "S2": 3, "S3": 0, "S4": 1, "U": 1}
    assert s["levels"]["S0"] == 0 and s["u_frames"] == 1  # U is never counted as S0
    assert s["worst_level"] == "S4" and s["first_detection_t_s"] == 4.0

    assert video_timeline(demo_run, {}) == {}
    json.dumps(tl)


def _synthetic_video_run(tmp_path, times, sidecar=None):
    """A run whose records are frames of tmp_path/clip.mp4 at `times` (fake backends), optionally with videos/clip.json."""
    recs = make_records(tmp_path, n=len(times))
    recs = [r.model_copy(update={"source_video": str(tmp_path / "clip.mp4"), "frame_time_s": t}) for r, t in zip(recs, times)]
    out = tmp_path / "run"
    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    write_manifest(recs, out / "manifest.jsonl")
    if sidecar is not None:
        (out / "videos").mkdir(exist_ok=True)
        (out / "videos" / "clip.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return out


def test_timeline_without_sidecar_uses_measured_gap(tmp_path):
    """No videos/<slug>.json: every_s comes from the median gap between frames and fps/duration stay null."""
    out = _synthetic_video_run(tmp_path, [0.0, 3.0, 6.0])
    e = video_timeline(out)["clip"]
    assert e["every_s"] == 3.0 and e["every_s_source"] == "median frame gap"
    assert e["mode"] is None and e["clip_end_s"] is None and e["last_hold_source"] == "median frame gap"
    assert e["fps"] is None and e["duration_s"] is None
    assert [f["level"] for f in e["frames"]] == [None, "U", "S2"]
    last = e["segments"][-1]
    assert (last["start_s"], last["end_s"], last["level"], last["action"], last["n_frames"]) == (6.0, 9.0, "S2", "schedule", 1)
    assert e["summary"]["levels"] == {"S0": 0, "S1": 0, "S2": 1, "S3": 0, "S4": 0, "U": 1}


@needs_ffmpeg
def test_last_frame_holds_until_the_clip_end(demo_run, tmp_path):
    """Trailing frames dropped by dedup (here: the 8 s and 10 s manifest rows removed): the last kept frame holds
    until frames.json's clip end, so the timeline, the last segment and the mp4 still cover the 12 s clip."""
    run = tmp_path / "run"
    shutil.copytree(demo_run, run)
    rows = [r for r in read_manifest(run / "manifest.jsonl") if r.frame_time_s not in (8.0, 10.0)]
    assert len(rows) == 4
    write_manifest(rows, run / "manifest.jsonl")
    e = video_timeline(run)[SLUG]
    assert [f["t_s"] for f in e["frames"]] == [0.0, 2.0, 4.0, 6.0]
    assert [f["hold_s"] for f in e["frames"]] == [2.0, 2.0, 2.0, 6.0]
    assert e["last_hold_source"] == "clip end from frames.json" and e["clip_end_s"] == pytest.approx(12.0, abs=0.05)
    last = e["segments"][-1]
    assert (last["start_s"], last["end_s"], last["level"]) == (6.0, 12.0, "S4")
    mp4 = write_detection_video(run, SLUG, timeline=e)
    assert probe(mp4).duration_s == pytest.approx(12.0, abs=0.3)
    vtt = write_subtitles(e, run / "videos" / "d.vtt").read_text(encoding="utf-8")
    assert "00:00:06.000 --> 00:00:12.000\nS4 - CS4 on MBEI-CS - escalate" in vtt

    # end_s in frames.json caps the clip end; a clip end before the last frame falls back to the sampling step
    side = json.loads((run / "videos" / f"{SLUG}.json").read_text(encoding="utf-8"))
    side["end_s"] = 9.0
    (run / "videos" / f"{SLUG}.json").write_text(json.dumps(side), encoding="utf-8")
    e2 = video_timeline(run)[SLUG]
    assert e2["clip_end_s"] == 9.0 and e2["frames"][-1]["hold_s"] == 3.0
    side["end_s"] = 5.0
    (run / "videos" / f"{SLUG}.json").write_text(json.dumps(side), encoding="utf-8")
    e3 = video_timeline(run)[SLUG]
    assert e3["frames"][-1]["hold_s"] == 2.0 and e3["last_hold_source"] == "frames.json"


def test_scene_mode_ignores_every_s_and_holds_to_the_clip_end(tmp_path):
    """Scene-mode frames.json still carries the unused interval default every_s=2.0; the step is the scene max gap
    (else the measured median gap) and the last frame holds to video.duration_s, never t_last + 2.0."""
    video_info = {"path": str(tmp_path / "clip.mp4"), "duration_s": 30.0, "fps": 25.0, "width": 640, "height": 480}
    side = {"mode": "scene", "every_s": 2.0, "scene_threshold": 0.3, "scene_max_gap_s": 10.0, "start_s": 0.0, "end_s": None, "video": video_info}
    out = _synthetic_video_run(tmp_path, [0.0, 7.3, 15.1, 25.1], side)
    e = video_timeline(out)["clip"]
    assert e["mode"] == "scene" and e["scene_threshold"] == 0.3 and e["scene_max_gap_s"] == 10.0
    assert e["every_s"] == 10.0 and e["every_s_source"] == "scene mode: max gap (frames.json)"
    assert [f["hold_s"] for f in e["frames"]] == [7.3, 7.8, 10.0, 4.9]
    assert e["last_hold_source"] == "clip end from frames.json" and e["segments"][-1]["end_s"] == 30.0
    assert sum(f["hold_s"] for f in e["frames"]) == pytest.approx(30.0)

    side["scene_max_gap_s"] = None
    (out / "videos" / "clip.json").write_text(json.dumps(side), encoding="utf-8")
    e = video_timeline(out)["clip"]
    assert e["every_s"] == 7.8 and e["every_s_source"] == "scene mode: median frame gap"
    assert [f["hold_s"] for f in e["frames"]] == [7.3, 7.8, 10.0, 4.9]


def test_segments_merge_only_on_identical_level_native_and_action():
    """Two S2 frames with different native values (CS3 cracking, CS2 exposed rebar both map to S2) stay separate
    segments, so the cue and the table never label the second frame with the first frame's native value."""
    base = {"standard": "MBEI-CS", "hold_s": 2.0}
    frames = [
        {**base, "t_s": 8.0, "level": "S2", "native_value": "CS3", "action": "schedule"},
        {**base, "t_s": 10.0, "level": "S2", "native_value": "CS2", "action": "schedule"},
        {**base, "t_s": 12.0, "level": "S2", "native_value": "CS2", "action": "schedule"},
        {**base, "t_s": 14.0, "level": "S2", "native_value": "CS2", "action": "monitor"},
    ]
    segs = videodetect._segments(frames)
    assert [(s["start_s"], s["end_s"], s["level"], s["native_value"], s["action"], s["n_frames"]) for s in segs] == [
        (8.0, 10.0, "S2", "CS3", "schedule", 1),
        (10.0, 14.0, "S2", "CS2", "schedule", 2),
        (14.0, 16.0, "S2", "CS2", "monitor", 1),
    ]
    assert [videodetect.cue_text(s) for s in segs][:2] == ["S2 - CS3 on MBEI-CS - schedule", "S2 - CS2 on MBEI-CS - schedule"]


# ----------------------------------------------------------------------------- drawing


def _rgb(hex_color: str) -> tuple:
    return ImageColor.getrgb(hex_color)


def test_annotate_frame_box_only_for_tiles():
    base = Image.new("RGB", (640, 480), (90, 120, 90))
    common = {"t_s": 12.0, "native_value": "CS3", "standard": "MBEI-CS", "action": "prioritize", "u_reason": None}
    full = annotate_frame(base, {**common, "level": "S3", "bbox": [200, 200, 500, 400], "tile": "full"})
    assert full.size == base.size and full is not base
    assert full.getpixel((201, 300)) == (90, 120, 90)  # whole frame graded: no box drawn
    assert base.getpixel((5, 5)) == (90, 120, 90)  # input untouched

    tiled = annotate_frame(base, {**common, "level": "S3", "bbox": [200, 200, 500, 400], "tile": "t01"})
    assert tiled.getpixel((201, 300)) == _rgb(videodetect.LEVEL_COLOR["S3"])
    assert tiled.getpixel((499, 300)) == _rgb(videodetect.LEVEL_COLOR["S3"])
    assert tiled.getpixel((350, 300)) == (90, 120, 90)  # interior untouched

    empty_box = annotate_frame(base, {**common, "level": "S3", "bbox": [], "tile": "t01"})
    assert empty_box.getpixel((201, 300)) == (90, 120, 90)

    ungraded = annotate_frame(base, {**common, "level": None, "native_value": None, "standard": None, "action": None, "bbox": [200, 200, 500, 400], "tile": "t01"})
    assert ungraded.getpixel((201, 300)) == (90, 120, 90)  # never a box or a level on an ungraded frame
    assert videodetect.badge_text({"level": None}) == "not graded"
    assert videodetect.badge_text({"level": "U", "action": "monitor"}) == "U  unassessable  monitor"
    assert videodetect.badge_text({**common, "level": "S3"}) == "S3  CS3 on MBEI-CS  prioritize"
    # the badge (top-left) and the timestamp (bottom-left) are filled boxes in the level colour / dark ink
    assert tiled.getpixel((8, 8)) == _rgb(videodetect.LEVEL_COLOR["S3"])
    assert ungraded.getpixel((8, 8)) == _rgb(videodetect.NOT_GRADED_COLOR)
    assert tiled.getpixel((8, 470)) == _rgb(videodetect.TIME_COLOR)

    u = annotate_frame(base, {**common, "level": "U", "native_value": "U", "action": "monitor", "bbox": None, "tile": "full", "u_reason": "gate marked image unusable: blur " * 5})
    assert u.getpixel((8, 8)) == _rgb(videodetect.LEVEL_COLOR["U"])


def _tiled_finding(finding_id: str, image_id: str, tile: str, bbox: list, level: str):
    rubric = {"standard": "MBEI-CS", "allowed_values": ["CS1", "CS2", "CS3", "CS4"], "rows": [{"criterion": "row 1"}]}
    f = fake_grade(None, finding_id=finding_id, image_id="img_2", asset_class="bridge_element", backend=None, rubric=rubric, metadata=None, exemplars=None,
                   evidence=Evidence(image_ids=[image_id], bbox=bbox, tile=tile), log=CallLog())
    return f.model_copy(update={"unified": f.unified.model_copy(update={"level": level})})


def test_every_tiled_finding_gets_its_own_box_and_s0_tiles_none(tmp_path):
    """A frame graded as three tiles keeps three boxes in the timeline entry, each drawn in its own level colour;
    the badge says the worst level and how many more tiles; S0 tiles (no defect found) get no box."""
    rec = make_records(tmp_path, n=1, size=(3000, 600))[0].model_copy(update={"source_video": str(tmp_path / "clip.mp4"), "frame_time_s": 1.0})
    iid = rec.image_id
    fs = [
        _tiled_finding(f"{iid}/t01", iid, "t01", [0, 0, 1000, 600], "S3"),
        _tiled_finding(f"{iid}/t02", iid, "t02", [1000, 0, 2000, 600], "S2"),
        _tiled_finding(f"{iid}/t03", iid, "t03", [2000, 0, 3000, 600], "S2"),
    ]
    fr = videodetect._frame_entry(rec, None, fs)
    assert fr["level"] == "S3" and fr["n_findings"] == 3 and fr["bbox"] == [0, 0, 1000, 600] and fr["tile"] == "t01"
    assert [(b["tile"], b["level"], b["bbox"]) for b in fr["boxes"]] == [("t01", "S3", [0, 0, 1000, 600]), ("t02", "S2", [1000, 0, 2000, 600]), ("t03", "S2", [2000, 0, 3000, 600])]
    assert videodetect.badge_text(fr) == "S3  CS4 on MBEI-CS  schedule  +2 tiles"
    base = Image.new("RGB", (3000, 600), (90, 120, 90))
    img = annotate_frame(base, fr)
    assert img.getpixel((5, 300)) == _rgb(videodetect.LEVEL_COLOR["S3"])
    assert img.getpixel((1005, 300)) == _rgb(videodetect.LEVEL_COLOR["S2"]) and img.getpixel((1995, 300)) == _rgb(videodetect.LEVEL_COLOR["S2"])
    assert img.getpixel((2005, 300)) == _rgb(videodetect.LEVEL_COLOR["S2"]) and img.getpixel((2995, 300)) == _rgb(videodetect.LEVEL_COLOR["S2"])
    assert img.getpixel((1500, 300)) == (90, 120, 90)

    clean = [_tiled_finding(f"{iid}/t01", iid, "t01", [0, 0, 1000, 600], "S0"), _tiled_finding(f"{iid}/t02", iid, "t02", [1000, 0, 2000, 600], "S0")]
    fr0 = videodetect._frame_entry(rec, None, clean)
    assert fr0["level"] == "S0" and len(fr0["boxes"]) == 2
    img0 = annotate_frame(base, fr0)
    assert img0.getpixel((5, 300)) == (90, 120, 90) and img0.getpixel((1005, 300)) == (90, 120, 90)  # no box reads as a detection
    assert img0.getpixel((20, 20)) == _rgb(videodetect.LEVEL_COLOR["S0"])  # the badge (margin 3000 // 160 = 18 px) still says S0
    assert videodetect.badge_text({"level": "S3", "native_value": "CS3", "standard": "MBEI-CS", "action": "prioritize", "n_findings": 1}) == "S3  CS3 on MBEI-CS  prioritize"


# ----------------------------------------------------------------------------- outputs


@needs_ffmpeg
def test_write_detection_video_matches_source_timeline(demo_run):
    mp4 = write_detection_video(demo_run, SLUG)
    assert mp4 == demo_run / "videos" / f"{SLUG}_detections.mp4" and mp4.stat().st_size > 0
    info = probe(mp4)
    print(f"\n[measured] {mp4.name}: duration {info.duration_s} s, {info.width}x{info.height} @ {info.fps} fps, codec {info.codec}, {info.size_bytes} bytes")
    assert info.duration_s == pytest.approx(12.0, abs=0.3)
    assert (info.width, info.height) == (1280, 960)
    assert info.codec == "h264" and info.fps == pytest.approx(10.0)
    ann = sorted((demo_run / "videos" / f"{SLUG}_annotated").glob("frame_*.jpg"))
    assert [p.name for p in ann] == [f"frame_{i:05d}.jpg" for i in range(1, 7)]
    with Image.open(ann[3]) as im:  # JPEG re-quantises the badge colour slightly; PNG-exact checks live in test_annotate_frame_box_only_for_tiles
        assert im.size == (1280, 960)
        assert all(abs(a - b) <= 12 for a, b in zip(im.getpixel((12, 12)), _rgb(videodetect.LEVEL_COLOR["S4"])))
    listing = (demo_run / "videos" / f"{SLUG}_annotated" / "frames.ffconcat").read_text(encoding="utf-8")
    assert listing.count("duration 2.000") == 6 and listing.rstrip().endswith("file 'frame_00006.jpg'")

    slow = write_detection_video(demo_run, SLUG, out=demo_run / "videos" / "slow.mp4", fps_out=2)
    assert probe(slow).duration_s == pytest.approx(12.0, abs=0.3)
    with pytest.raises(VideoError, match="fps_out"):
        write_detection_video(demo_run, SLUG, fps_out=0)
    with pytest.raises(VideoError, match="no video frames"):
        write_detection_video(demo_run, "not_a_slug")


@needs_ffmpeg
def test_write_subtitles_one_cue_per_segment(demo_run, tmp_path):
    e = video_timeline(demo_run)[SLUG]
    vtt = write_subtitles(e, tmp_path / "d.vtt")
    text = vtt.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT\n\n")
    cues = [c for c in text.split("\n\n") if "-->" in c]
    assert len(cues) == 5
    assert cues[0].splitlines() == ["1", "00:00:00.000 --> 00:00:02.000", "not graded"]
    assert cues[1].splitlines() == ["2", "00:00:02.000 --> 00:00:04.000", "U - unassessable"]
    assert cues[2].splitlines() == ["3", "00:00:04.000 --> 00:00:06.000", "S2 - CS4 on MBEI-CS - schedule"]
    assert cues[3].splitlines() == ["4", "00:00:06.000 --> 00:00:08.000", "S4 - CS4 on MBEI-CS - escalate"]
    assert cues[4].splitlines() == ["5", "00:00:08.000 --> 00:00:12.000", "S2 - CS4 on MBEI-CS - schedule"]
    assert videodetect.cue_text({"level": "S3", "native_value": "CS3", "standard": "MBEI-CS", "action": "prioritize"}) == "S3 - CS3 on MBEI-CS - prioritize"
    assert videodetect._vtt_ts(3725.5) == "01:02:05.500"


def test_subtitles_run_on_the_mp4_clock_for_offset_frames(tmp_path):
    """Frames from 30 s on (a --start 30 extraction): the mp4 starts at the first kept frame, so the first cue is at
    00:00:00.000; `offset_s=0` keeps source time for anyone who wants it."""
    out = _synthetic_video_run(tmp_path, [30.0, 32.0, 34.0])
    e = video_timeline(out)["clip"]
    assert e["mp4_offset_s"] == 30.0 and [f["t_s"] for f in e["frames"]] == [30.0, 32.0, 34.0]
    cues = [c for c in write_subtitles(e, tmp_path / "o.vtt").read_text(encoding="utf-8").split("\n\n") if "-->" in c]
    assert cues[0].splitlines() == ["1", "00:00:00.000 --> 00:00:02.000", "not graded"]
    assert cues[-1].splitlines()[1] == "00:00:04.000 --> 00:00:06.000"
    source_time = [c for c in write_subtitles(e, tmp_path / "s.vtt", offset_s=0).read_text(encoding="utf-8").split("\n\n") if "-->" in c]
    assert source_time[0].splitlines()[1] == "00:00:30.000 --> 00:00:32.000"
    legacy = {k: v for k, v in e.items() if k != "mp4_offset_s"}  # a stored entry written before mp4_offset_s existed
    assert write_subtitles(legacy, tmp_path / "l.vtt").read_text(encoding="utf-8").count("00:00:00.000 --> 00:00:02.000") == 1


@needs_ffmpeg
def test_start_offset_extraction_cues_match_the_mp4(tmp_path):
    """ingest_video(start_s=4.0) on the demo clip: frames at 4, 6, 8, 10 s; the mp4 is 8 s long, the cues run
    00:00:00 -> 00:00:08 and the S4 cue for the 10 s frame is the last one shown, not one past the end."""
    if not DEMO.exists():
        pytest.skip("demo clip missing")
    records, ext = ingest_video(DEMO, tmp_path / "frames" / SLUG, asset_class="bridge_element", every_s=2, start_s=4.0, dedup_max_distance=None)
    assert [r.frame_time_s for r in records] == [4.0, 6.0, 8.0, 10.0]
    out = tmp_path / "run"
    out.mkdir()
    write_manifest(records, out / "manifest.jsonl")
    (out / "videos").mkdir()
    shutil.copyfile(Path(ext.out_dir) / "frames.json", out / "videos" / f"{SLUG}.json")
    run_cascade(records, out, RunConfig(gate="none"), gate_fn=_gate_by_frame, grade_fn=_grade_by_frame)
    res = write_video_detections(out)[SLUG]
    e = json.loads(res["timeline"].read_text(encoding="utf-8"))
    assert e["mp4_offset_s"] == 4.0 and [f["level"] for f in e["frames"]] == [None, "U", "S2", "S4"]
    assert [f["hold_s"] for f in e["frames"]] == [2.0] * 4 and e["segments"][-1]["end_s"] == 12.0  # source time
    assert probe(res["mp4"]).duration_s == pytest.approx(8.0, abs=0.3)
    cues = [c for c in res["vtt"].read_text(encoding="utf-8").split("\n\n") if "-->" in c]
    assert cues[0].splitlines() == ["1", "00:00:00.000 --> 00:00:02.000", "not graded"]
    assert cues[-1].splitlines() == ["4", "00:00:06.000 --> 00:00:08.000", "S4 - CS4 on MBEI-CS - escalate"]
    with Image.open(sorted((out / "videos" / f"{SLUG}_annotated").glob("frame_*.jpg"))[0]) as im:
        assert im.size == (1280, 960)  # the on-frame stamp stays source time ("t = 4.0 s"); only the cues shift


@needs_ffmpeg
def test_contact_sheet_and_bundle(demo_run, capsys):
    e = video_timeline(demo_run)[SLUG]
    sheet = write_contact_sheet(demo_run, SLUG, e, cols=3, thumb=200)
    with Image.open(sheet) as im:
        assert im.format == "PNG"
        assert im.width == 3 * 200 + 4 * 4 and im.height == 2 * 150 + 3 * 4  # 1280x960 frames -> 200x150 tiles, 3 columns, 2 rows

    res = write_video_detections(demo_run)
    assert list(res) == [SLUG]
    item = res[SLUG]
    assert "skipped" not in item
    for key in ("timeline", "mp4", "vtt", "sheet"):
        assert isinstance(item[key], Path) and item[key].exists(), key
    assert item["timeline"] == demo_run / "videos" / f"{SLUG}_timeline.json"
    assert item["mp4"].name == f"{SLUG}_detections.mp4" and item["vtt"].name == f"{SLUG}_detections.vtt" and item["sheet"].name == f"{SLUG}_contact.png"
    assert item["summary"] == e["summary"]
    stored = json.loads(item["timeline"].read_text(encoding="utf-8"))
    assert stored["summary"]["levels"]["U"] == 1 and stored["summary"]["levels"]["S0"] == 0

    assert videodetect.main(["--run", str(demo_run), "--slug", SLUG]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert list(printed) == [SLUG] and printed[SLUG]["summary"]["worst_level"] == "S4"
    assert printed[SLUG]["mp4"].endswith(f"{SLUG}_detections.mp4")
    assert videodetect.main(["--run", str(demo_run), "--slug", "nope"]) == 2
    assert "not a video slug" in capsys.readouterr().err


@needs_ffmpeg
def test_missing_ffmpeg_skips_mp4_but_writes_the_rest(demo_run, monkeypatch):
    monkeypatch.setattr(video, "FFMPEG", "ffmpeg_not_here")
    monkeypatch.setattr(video, "FFPROBE", "ffprobe_not_here")
    with pytest.raises(VideoError, match="PATH"):
        write_detection_video(demo_run, SLUG)
    res = write_video_detections(demo_run)
    item = res[SLUG]
    assert item["mp4"] is None and "ffmpeg" in item["skipped"]
    assert item["vtt"].exists() and item["sheet"].exists() and item["timeline"].exists()


def test_run_without_video_returns_empty(tmp_path, capsys):
    recs = make_records(tmp_path)
    out = tmp_path / "run"
    run_cascade(recs, out, RunConfig(gate="none"), gate_fn=fake_gate, grade_fn=fake_grade)
    write_manifest(recs, out / "manifest.jsonl")
    assert video_timeline(out) == {}
    assert write_video_detections(out) == {}
    assert not (out / "videos").exists()
    assert video_timeline(tmp_path / "nowhere") == {}
    assert videodetect.main(["--run", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {}
