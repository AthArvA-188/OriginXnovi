"""FR-4 video ingestion: ffprobe/ffmpeg subprocess extraction, dHash dedup, provenance on records.

ffmpeg-dependent tests skip when ffmpeg/ffprobe are not on PATH; the pure-PIL hashing tests and the
missing-binary test run everywhere. Synthetic clips are built once per module with lavfi sources
(each build takes about 0.1 s). The demo-clip test prints the measured dHash distances so UI help
text can quote numbers from a real run rather than guesses.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from cascade import video
from cascade.ingest import ingest_folder, read_manifest, write_manifest
from cascade.pipeline import RunConfig, load_run, run_cascade
from cascade.video import Extraction, Fingerprint, VideoError, dhash, extract_frames, ffmpeg_available, fingerprint, hamming, ingest_video, is_near_duplicate, probe

from test_pipeline import fake_gate, fake_grade

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
DEMO = Path(__file__).resolve().parents[1] / "data" / "demo" / "video" / "bridge_walkthrough.mp4"
SIZE = "320x240"


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict:
    """Synthetic clips: 6 s at 25 fps unless stated. Built with lavfi so no fixture binary is committed."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path_factory.mktemp("clips")
    test6, static6, cut6, ntsc, rot, mkv = (d / n for n in ("test6.mp4", "static6.mp4", "cut6.mp4", "ntsc.mp4", "rot.mp4", "test6.mkv"))
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=duration=6:size={SIZE}:rate=25", "-pix_fmt", "yuv420p", str(test6))
    _ffmpeg("-f", "lavfi", "-i", f"color=c=blue:size={SIZE}:duration=6:rate=25", "-pix_fmt", "yuv420p", str(static6))
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=c=red:size={SIZE}:duration=3:rate=25",
        "-f", "lavfi", "-i", f"color=c=blue:size={SIZE}:duration=3:rate=25",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", str(cut6),
    )
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=duration=6:size={SIZE}:rate=30000/1001", "-pix_fmt", "yuv420p", str(ntsc))
    _ffmpeg("-display_rotation", "90", "-i", str(test6), "-c", "copy", str(rot))
    _ffmpeg("-i", str(test6), "-c", "copy", str(mkv))
    return {"test6": test6, "static6": static6, "cut6": cut6, "ntsc": ntsc, "rot": rot, "mkv": mkv}


def _times(ext: Extraction) -> list[float]:
    return [f.t_s for f in ext.frames]


# ----------------------------------------------------------------------------- probe


@needs_ffmpeg
def test_probe_fields(clips):
    info = probe(clips["test6"])
    assert info.duration_s == pytest.approx(6.0, abs=0.05)
    assert info.fps == 25.0
    assert (info.width, info.height) == (320, 240)
    assert info.rotation == 0
    assert info.nb_frames == 150 and info.nb_frames_estimated is False
    assert len(info.sha256) == 64 and all(c in "0123456789abcdef" for c in info.sha256)
    assert info.codec == "h264" and info.size_bytes > 0

    mkv = probe(clips["mkv"])
    assert mkv.nb_frames_estimated is True and mkv.nb_frames == 150
    assert mkv.duration_s == pytest.approx(6.0, abs=0.05)

    rot = probe(clips["rot"])
    assert rot.rotation == 90 and (rot.width, rot.height) == (240, 320)


@needs_ffmpeg
def test_probe_rejects_non_video(tmp_path):
    text = tmp_path / "notes.mp4"
    text.write_text("not a video", encoding="utf-8")
    with pytest.raises(VideoError, match="not a readable video"):
        probe(text)
    still = tmp_path / "still.mp4"
    Image.new("RGB", (64, 48), (200, 30, 30)).save(still, format="JPEG")
    with pytest.raises(VideoError, match="no duration"):
        probe(still)


# ----------------------------------------------------------------------------- extraction


@needs_ffmpeg
def test_extract_interval_count_and_timestamps(clips, tmp_path):
    ext = extract_frames(clips["test6"], tmp_path / "frames", every_s=2, dedup_max_distance=None)
    assert ext.mode == "interval" and ext.truncated is False
    assert _times(ext) == [0.0, 2.0, 4.0]
    assert [Path(f.path).name for f in ext.frames] == ["test6_t00000000.jpg", "test6_t00002000.jpg", "test6_t00004000.jpg"]
    for f in ext.frames:
        with Image.open(f.path) as im:
            assert im.format == "JPEG" and im.size == (320, 240) and im.mode == "RGB"
    assert (tmp_path / "frames" / "frames.json").exists()
    assert json.loads((tmp_path / "frames" / "frames.json").read_text(encoding="utf-8"))["n_extracted"] == 3
    assert ext.ffmpeg_version and "ffmpeg" in ext.ffmpeg_version
    assert ext.command[0] == video.FFMPEG and "-fps_mode" in ext.command


@needs_ffmpeg
def test_extract_start_end_window(clips, tmp_path):
    ext = extract_frames(clips["test6"], tmp_path / "frames", every_s=1, start_s=2.0, end_s=4.0, dedup_max_distance=None)
    assert _times(ext) == [2.0, 3.0]
    assert ext.start_s == 2.0 and ext.end_s == 4.0


@needs_ffmpeg
def test_extract_ntsc_names_are_ms(clips, tmp_path):
    ext = extract_frames(clips["ntsc"], tmp_path / "frames", every_s=2, dedup_max_distance=None)
    assert [f.ms for f in ext.frames] == [0, 2000, 4000]


@needs_ffmpeg
def test_max_frames_caps_and_flags_truncated(clips, tmp_path):
    ext = extract_frames(clips["test6"], tmp_path / "frames", every_s=1, max_frames=2, dedup_max_distance=None)
    assert len(ext.frames) == 2 and ext.truncated is True
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 2


@needs_ffmpeg
def test_dedup_static_keeps_first(clips, tmp_path):
    out = tmp_path / "frames"
    ext = extract_frames(clips["static6"], out, every_s=1)
    assert (len(ext.frames), len(ext.kept), len(ext.dropped)) == (6, 1, 5)
    assert ext.kept[0].t_s == 0.0 and ext.kept[0].distance is None
    assert all(f.distance == 0 for f in ext.dropped)
    assert all(not Path(f.path).exists() for f in ext.dropped)
    assert [p.name for p in out.glob("*.jpg")] == ["static6_t00000000.jpg"]
    j = ext.to_json()
    assert (j["n_extracted"], j["n_kept"], j["n_dropped"]) == (6, 1, 5)
    json.dumps(j)  # must be serialisable as-is

    out2 = tmp_path / "frames2"
    ext2 = extract_frames(clips["static6"], out2, every_s=1, keep_dropped=True)
    dropped_dir = tmp_path / "frames2_dropped"
    assert dropped_dir.is_dir() and len(list(dropped_dir.glob("*.jpg"))) == 5
    assert all(Path(f.path).parent == dropped_dir for f in ext2.dropped)
    assert len(ingest_folder(out2, asset_class="steel_coating")) == 1


@needs_ffmpeg
def test_dedup_flat_colour_cut_is_not_a_duplicate(clips, tmp_path):
    ext = extract_frames(clips["cut6"], tmp_path / "frames", every_s=1)
    assert [f.t_s for f in ext.kept] == [0.0, 3.0]
    assert len(ext.dropped) == 4
    # both flat colours hash to 0; only the mean-luma guard keeps the cut
    assert ext.kept[1].distance == 0


@needs_ffmpeg
def test_scene_mode(clips, tmp_path):
    ext = extract_frames(clips["cut6"], tmp_path / "scene", scene_threshold=0.3, dedup_max_distance=None)
    assert ext.mode == "scene" and _times(ext) == [0.0, 3.0]
    ext2 = extract_frames(clips["test6"], tmp_path / "gap", scene_threshold=0.3, scene_max_gap_s=2, dedup_max_distance=None)
    assert _times(ext2) == [0.0, 2.0, 4.0]


@needs_ffmpeg
def test_rerun_into_same_folder_does_not_accumulate(clips, tmp_path):
    out = tmp_path / "frames"
    extract_frames(clips["test6"], out, every_s=1, dedup_max_distance=None)
    ext = extract_frames(clips["test6"], out, every_s=2, dedup_max_distance=None)
    assert len(ext.frames) == 3 and len(list(out.glob("*.jpg"))) == 3


@needs_ffmpeg
def test_bad_arguments_are_video_errors(clips, tmp_path):
    for kw in (dict(every_s=0.01), dict(max_frames=0), dict(scene_threshold=1.5), dict(start_s=5, end_s=2), dict(dedup_max_distance=-1)):
        with pytest.raises(VideoError):
            extract_frames(clips["test6"], tmp_path / "x", **kw)


# ----------------------------------------------------------------------------- demo clip


@needs_ffmpeg
@pytest.mark.skipif(not DEMO.exists(), reason="demo clip missing")
def test_demo_bridge_walkthrough_dedup(tmp_path):
    """data/demo/video/bridge_walkthrough.mp4: 12 s, six 2 s stills built from dataset images, 1280x960 @ 10 fps."""
    info = probe(DEMO)
    assert info.duration_s == pytest.approx(12.0, abs=0.05) and (info.width, info.height) == (1280, 960)

    ext2 = extract_frames(DEMO, tmp_path / "every2", every_s=2)
    d2 = [f.distance for f in ext2.frames[1:]]
    print(f"\n[measured] bridge_walkthrough every 2 s: extracted {len(ext2.frames)}, kept {len(ext2.kept)}, dropped {len(ext2.dropped)}; "
          f"dHash distance to previous kept frame: {d2}; ffmpeg {ext2.seconds_ffmpeg} s, dedup {ext2.seconds_dedup} s")
    assert len(ext2.frames) == 6
    assert 5 <= len(ext2.kept) <= 6  # six distinct stills, one per 2 s slot
    assert all(d is not None and d > ext2.dedup_max_distance for d in d2)

    ext1 = extract_frames(DEMO, tmp_path / "every1", every_s=1)
    d1 = [(f.t_s, f.kept, f.distance) for f in ext1.frames]
    print(f"[measured] bridge_walkthrough every 1 s: extracted {len(ext1.frames)}, kept {len(ext1.kept)}, dropped {len(ext1.dropped)}; (t_s, kept, distance) = {d1}")
    assert len(ext1.frames) == 12
    assert 5 <= len(ext1.kept) <= 7  # the second sample of each still is a near-duplicate
    assert all(f.distance is not None and f.distance <= ext1.dedup_max_distance for f in ext1.dropped)


# ----------------------------------------------------------------------------- ingest + cascade


@needs_ffmpeg
def test_ingest_video_records(clips, tmp_path):
    src = tmp_path / "test6.mp4"
    shutil.copy(clips["test6"], src)
    stamp = datetime(2026, 3, 14, 12, 0, 0).timestamp()
    os.utime(src, (stamp, stamp))
    out = tmp_path / "frames"
    records, ext = ingest_video(src, out, asset_class="pv_module", client_id="acme", asset_id="A-1", dedup_max_distance=None, captured_on_from_mtime=True)
    assert len(records) == 3 and len(ext.kept) == 3
    assert {r.frame_time_s for r in records} == {0.0, 2.0, 4.0}
    assert {r.image_id for r in records} == {f"vid_test6_t{ms:08d}" for ms in (0, 2000, 4000)}
    for r in records:
        assert r.source_video == str(src.resolve())
        assert r.client_id == "acme" and r.asset_id == "A-1" and r.asset_class == "pv_module"
        assert r.captured_on == "2026-03-14"
        assert (r.width, r.height) == (320, 240)
        assert r.gsd_mm_per_px is None and r.irradiance_wm2 is None
        assert r.source_dataset == "video" and r.split == "upload"
        assert len(r.sha256) == 64
    assert len({r.sha256 for r in records}) == 3
    assert (out / "frames.json").exists() and (out / "sidecar.json").exists()
    assert json.loads((out / "frames.json").read_text(encoding="utf-8"))["sidecar"] == str(out / "sidecar.json")

    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    assert [r.model_dump() for r in read_manifest(manifest)] == [r.model_dump() for r in records]

    recs_no_mtime, _ = ingest_video(src, tmp_path / "f2", asset_class="pv_module", dedup_max_distance=None)  # default: mtime is never a capture date
    assert all(r.captured_on is None for r in recs_no_mtime)
    recs_explicit, _ = ingest_video(src, tmp_path / "f3", asset_class="pv_module", dedup_max_distance=None, captured_on="2026-01-02")
    assert all(r.captured_on == "2026-01-02" for r in recs_explicit)


def _route_all_gate(img, image_id, **kw):
    # fake_gate derives its verdict from an integer id suffix; "img_2" = damaged + routed
    return fake_gate(img, "img_2", **kw).model_copy(update={"image_id": image_id})


def _grade(img, **kw):
    kw["image_id"] = "img_2"  # fake_grade picks the level from the id suffix; finding_id and evidence keep the vid_ ids
    return fake_grade(img, **kw)


@needs_ffmpeg
def test_records_run_through_cascade(clips, tmp_path):
    records, _ = ingest_video(clips["test6"], tmp_path / "frames", asset_class="steel_coating", dedup_max_distance=None)
    out = tmp_path / "run"
    summary = run_cascade(records, out, RunConfig(gate="none"), gate_fn=_route_all_gate, grade_fn=_grade)
    assert summary["images"] == 3 and summary["findings"] == 3
    ids = {f.evidence.image_ids[0] for f in load_run(out)["findings"]}
    assert ids == {f"vid_test6_t{ms:08d}" for ms in (0, 2000, 4000)}
    finding_ids = [f["finding_id"] for f in json.loads((out / "findings.json").read_text(encoding="utf-8"))]
    assert len(finding_ids) == 3 and all(fid.startswith("vid_test6_t") and fid.endswith("/full") for fid in finding_ids)


# ----------------------------------------------------------------------------- pure PIL (no ffmpeg)


def _textured(size=(200, 150)) -> Image.Image:
    """Deterministic gradient with a dark block, so dHash has structure to compare (a flat image hashes to 0)."""
    w, h = size
    im = Image.new("L", size)
    im.putdata([(x * 255 // w + y * 255 // h) // 2 for y in range(h) for x in range(w)])
    im.paste(20, (w // 3, h // 3, w // 3 + w // 5, h // 3 + h // 5))
    return im


def test_dhash_pure_pil(tmp_path):
    base = _textured()
    assert hamming(dhash(base), dhash(base.copy())) == 0
    w, h = base.size
    shifted = base.crop((2, 0, w, h)).resize((w, h))  # 1 % horizontal shift
    assert hamming(dhash(base), dhash(shifted)) <= 1
    assert hamming(dhash(base), dhash(base.rotate(180))) >= 16

    red, blue = tmp_path / "red.jpg", tmp_path / "blue.jpg"
    Image.new("RGB", (64, 48), (255, 0, 0)).save(red, quality=90)
    Image.new("RGB", (64, 48), (0, 0, 255)).save(blue, quality=90)
    fr, fb = fingerprint(red), fingerprint(blue)
    assert hamming(fr.dhash, fb.dhash) == 0
    assert fr.flat and fb.flat
    assert is_near_duplicate(fr, fb, 3) is False  # luma guard: 76 vs 29
    assert is_near_duplicate(fr, fingerprint(red), 0) is True

    textured_path = tmp_path / "tex.png"
    base.save(textured_path)
    ft = fingerprint(textured_path)
    assert ft.flat is False and ft.dhash == dhash(base)
    assert is_near_duplicate(Fingerprint(0b1011, 100, True), Fingerprint(0b1011, 120, True), 3) is False
    assert is_near_duplicate(Fingerprint(0b1011, 100, True), Fingerprint(0b1010, 104, True), 3) is True


def test_missing_ffmpeg_is_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "FFMPEG", "ffmpeg_not_here")
    monkeypatch.setattr(video, "FFPROBE", "ffprobe_not_here")
    ok, msg = ffmpeg_available()
    assert ok is False and "ffmpeg" in msg and "PATH" in msg
    assert video.ffmpeg_version() is None
    clip = tmp_path / "x.mp4"
    clip.write_bytes(b"\x00" * 16)
    with pytest.raises(VideoError, match="PATH"):
        probe(clip)
    with pytest.raises(VideoError, match="ffmpeg"):
        extract_frames(clip, tmp_path / "frames")
    with pytest.raises(VideoError):
        ingest_video(clip, tmp_path / "frames", asset_class="pv_module")


# ----------------------------------------------------------------------------- CLI


@needs_ffmpeg
def test_cli(clips, tmp_path, capsys):
    rc = video.main(["--video", str(clips["test6"]), "--out", str(tmp_path / "frames"), "--every", "2", "--asset-class", "steel_coating", "--no-dedup"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kept"] == 3 and report["extracted"] == 3 and report["dropped"] == 0
    assert report["mode"] == "interval" and report["width"] == 320
    manifest = tmp_path / "manifest.jsonl"
    assert report["manifest"] == str(manifest)
    rows = read_manifest(manifest)
    assert len(rows) == 3 and all(r.asset_class == "steel_coating" and r.source_video for r in rows)

    with pytest.raises(SystemExit) as e:
        video.main(["--video", str(clips["test6"]), "--out", str(tmp_path / "f2")])
    assert e.value.code == 2

    text = tmp_path / "bad.mp4"
    text.write_text("nope", encoding="utf-8")
    rc = video.main(["--video", str(text), "--out", str(tmp_path / "f3"), "--asset-class", "pv_module"])
    assert rc == 2
    assert "not a readable video" in capsys.readouterr().err
