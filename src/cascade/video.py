"""FR-4: video in, frame records out. ffmpeg/ffprobe subprocesses only; PIL for hashing.

Frames are written as JPEG q=2 named <slug>_t<ms:08d>.jpg so lexical order is time order
(ingest_folder sorts by path). Near-duplicates (dHash) are removed before ingest so the gate
never sees a hovering drone's 50 identical frames. Every decision is recorded in frames.json.

Measured on this laptop (ffmpeg 8.0.1 gyan build, Pillow 12.3, 2026-09-25):
  - `-enc_time_base 1/1000 -frame_pts 1` names output files in milliseconds (0, 2000, 4000)
    in both interval and scene mode; without `-fps_mode passthrough|vfr` image2 duplicates
    frames to 1000 fps (6,000 files from a 6 s clip), so the fps_mode flag is mandatory.
  - dHash Hamming distance, measured by tests/test_video.py on this build: static colour clip 0;
    data/demo/video/bridge_walkthrough.mp4 (six 2 s stills from dataset images) repeated still 0,
    consecutive distinct stills 28-35; testsrc frames 2 s apart 4-6. From the spec's table
    (team-measured on the same laptop): JPEG re-save 0; <=1 % shift 0; 2 % shift 3; 180 deg
    rotation 33. Flat red vs flat blue both hash to 0 (no pixel is strictly less than its
    neighbour), hence the mean-luma guard on flat frames.
  - Extracted JPEGs carry no EXIF, so captured_on comes from the caller or the container
    creation_time; it is never invented. The file mtime is used only on request (--mtime-date /
    captured_on_from_mtime=True): a copied or downloaded clip's mtime is the copy date, and the
    client report would anchor SLA due dates on it as the "capture date".

Example:
  python -m cascade.video --video data/demo/video/bridge_walkthrough.mp4 --out runs/v1/frames \
      --every 2 --asset-class bridge_element
  python -m cascade.run --manifest runs/v1/manifest.jsonl --out runs/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, get_args

from dotenv import load_dotenv
from PIL import Image

from .ingest import ingest_folder, sha256_of, write_manifest
from .schema import AssetClass, ImageRecord

ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
FFMPEG = os.getenv("FFMPEG", "ffmpeg")  # module attributes so tests can monkeypatch
FFPROBE = os.getenv("FFPROBE", "ffprobe")

_MISSING_MSG = (
    "ffmpeg/ffprobe not found on PATH. Install: winget install Gyan.FFmpeg (Windows), "
    "apt install ffmpeg, brew install ffmpeg, or set FFMPEG/FFPROBE. "
    "Video upload is disabled until then; images still work."
)
_HASH_SIZE = 8
_FLAT_MAX_BITS = 4  # popcount(dhash) <= 4 means the frame is essentially uniform
_FLAT_LUMA_TOL = 8  # mean-luma difference allowed between two flat frames to still call them duplicates
_PROBE_TIMEOUT_S = 60.0


class VideoError(RuntimeError):
    """Complete, user-facing message; safe to show in the UI and print from the CLI."""


# ----------------------------------------------------------------------------- binaries


def ffmpeg_available() -> tuple[bool, str]:
    """(True, path-to-ffmpeg) or (False, install hint). Uses shutil.which on both binaries; never raises."""
    ffmpeg_path = shutil.which(FFMPEG)
    ffprobe_path = shutil.which(FFPROBE)
    if ffmpeg_path and ffprobe_path:
        return True, ffmpeg_path
    return False, _MISSING_MSG


def ffmpeg_version() -> Optional[str]:
    """First line of `ffmpeg -version`, or None when the binary is absent. Stored in frames.json."""
    try:
        cp = _run([FFMPEG, "-version"], timeout_s=_PROBE_TIMEOUT_S)
    except VideoError:
        return None
    first = (cp.stdout or "").strip().splitlines()
    return first[0].strip() if first else None


def _run(cmd: List[str], *, timeout_s: Optional[float]) -> subprocess.CompletedProcess:
    """Run ffmpeg/ffprobe as an argument list. Missing binary -> VideoError; timeout -> VideoError.

    stderr is decoded with errors="replace" because ffmpeg can emit non-UTF-8 bytes. On Windows
    CREATE_NO_WINDOW keeps a console from flashing under Streamlit.
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            creationflags=flags,
        )
    except FileNotFoundError as e:
        raise VideoError(_MISSING_MSG) from e
    except subprocess.TimeoutExpired as e:
        raise VideoError(f"{Path(cmd[0]).name} exceeded {timeout_s:.0f} s; use --start/--end or a larger --every") from e


def _stderr_tail(cp: subprocess.CompletedProcess) -> str:
    lines = [ln.strip() for ln in (cp.stderr or "").splitlines() if ln.strip()]
    return lines[-1] if lines else f"exit code {cp.returncode}"


# ----------------------------------------------------------------------------- probe


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe reports for the first video stream; width/height as displayed (rotation applied)."""

    path: str
    sha256: str
    size_bytes: int
    codec: str
    duration_s: float
    fps: float
    time_base: str
    width: int
    height: int
    rotation: int
    nb_frames: Optional[int]
    nb_frames_estimated: bool
    creation_time: Optional[str]


def _parse_rate(text: Optional[str]) -> Optional[Fraction]:
    if not text:
        return None
    num, _, den = text.partition("/")
    try:
        n, d = int(num), int(den or "1")
    except ValueError:
        return None
    if d == 0 or n <= 0:
        return None
    return Fraction(n, d)


def _rotation_of(stream: dict) -> int:
    for sd in stream.get("side_data_list") or []:
        if sd.get("side_data_type") == "Display Matrix" and sd.get("rotation") is not None:
            return int(round(float(sd["rotation"])))
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return int(tag)
        except ValueError:
            return 0
    return 0


def probe(path: Path) -> VideoInfo:
    """ffprobe the first video stream. Raises VideoError for unreadable input or a still image renamed .mp4."""
    path = Path(path)
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-select_streams", "v:0", str(path)]
    cp = _run(cmd, timeout_s=_PROBE_TIMEOUT_S)
    try:
        data = json.loads(cp.stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    streams = data.get("streams") or []
    if cp.returncode != 0 or not streams:
        raise VideoError(f"{path.name}: not a readable video ({_stderr_tail(cp)})")
    stream, fmt = streams[0], data.get("format") or {}
    duration_text = stream.get("duration") or fmt.get("duration")
    if not duration_text:
        raise VideoError(f"{path.name}: no duration: not a video stream (a still image renamed .mp4 probes as {stream.get('codec_name', 'unknown')})")
    duration_s = float(duration_text)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if fps is None:
        raise VideoError(f"{path.name}: no frame rate reported (avg_frame_rate={stream.get('avg_frame_rate')!r})")
    rotation = _rotation_of(stream)
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    nb_text = stream.get("nb_frames")
    if nb_text:
        nb_frames, estimated = int(nb_text), False
    else:
        nb_frames, estimated = int(round(duration_s * float(fps))), True
    creation = (fmt.get("tags") or {}).get("creation_time") or (stream.get("tags") or {}).get("creation_time")
    return VideoInfo(
        path=str(path),
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        codec=str(stream.get("codec_name", "")),
        duration_s=duration_s,
        fps=float(fps),
        time_base=str(stream.get("time_base", "")),
        width=width,
        height=height,
        rotation=rotation,
        nb_frames=nb_frames,
        nb_frames_estimated=estimated,
        creation_time=creation,
    )


# ----------------------------------------------------------------------------- hashing


def _dhash_bytes(px: bytes, size: int) -> int:
    """Row-major 64-bit hash over a (size+1) x size grayscale buffer: bit = px[r][c] < px[r][c+1]."""
    stride = size + 1
    h = 0
    for r in range(size):
        row = px[r * stride : (r + 1) * stride]
        for c in range(size):
            h = (h << 1) | (1 if row[c] < row[c + 1] else 0)
    return h


def dhash(img: Image.Image, size: int = _HASH_SIZE) -> int:
    """Difference hash: L channel, BILINEAR resize to (size+1) x size, compare horizontal neighbours."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    return _dhash_bytes(g.tobytes(), size)


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class Fingerprint:
    """dHash plus the mean luma of the 9x8 thumbnail; `flat` marks near-uniform frames whose hash is uninformative."""

    dhash: int
    mean_luma: int
    flat: bool


def fingerprint(path: Path) -> Fingerprint:
    """Hash one frame. `Image.draft` makes JPEG decode at 1/8 scale (5 ms vs 41 ms at 4K, identical hash)."""
    with Image.open(path) as im:
        im.draft("L", ((_HASH_SIZE + 1) * 8, _HASH_SIZE * 8))
        g = im.convert("L").resize((_HASH_SIZE + 1, _HASH_SIZE), Image.BILINEAR)
        px = g.tobytes()
    h = _dhash_bytes(px, _HASH_SIZE)
    return Fingerprint(dhash=h, mean_luma=sum(px) // len(px), flat=h.bit_count() <= _FLAT_MAX_BITS)


def is_near_duplicate(a: Fingerprint, b: Fingerprint, max_distance: int) -> bool:
    """Hamming <= max_distance, and for flat frames also mean luma within 8 (flat red and flat blue both hash to 0)."""
    if hamming(a.dhash, b.dhash) > max_distance:
        return False
    if a.flat or b.flat:
        return abs(a.mean_luma - b.mean_luma) <= _FLAT_LUMA_TOL
    return True


# ----------------------------------------------------------------------------- extraction


@dataclass
class Frame:
    """One extracted frame. `distance` is the Hamming distance to the last KEPT frame (None for the first)."""

    path: str
    ms: int
    t_s: float
    kept: bool = True
    distance: Optional[int] = None


@dataclass
class Extraction:
    """Everything measured during one extraction; `to_json()` is what frames.json and the UI strip show."""

    video: VideoInfo
    out_dir: str
    mode: str  # "interval" | "scene"
    every_s: float
    scene_threshold: Optional[float]
    scene_max_gap_s: Optional[float]
    max_frames: int
    dedup_max_distance: Optional[int]
    start_s: float
    end_s: Optional[float]
    frames: List[Frame] = field(default_factory=list)
    truncated: bool = False  # max_frames reached before the end of the clip
    seconds_ffmpeg: float = 0.0
    seconds_dedup: float = 0.0
    command: List[str] = field(default_factory=list)
    ffmpeg_version: Optional[str] = None

    @property
    def kept(self) -> List[Frame]:
        return [f for f in self.frames if f.kept]

    @property
    def dropped(self) -> List[Frame]:
        return [f for f in self.frames if not f.kept]

    def to_json(self) -> dict:
        d = asdict(self)
        d.update(n_extracted=len(self.frames), n_kept=len(self.kept), n_dropped=len(self.dropped))
        return d


def slug_of(video: Path) -> str:
    """Filesystem- and id-safe stem: [^A-Za-z0-9_-] -> '_', at most 40 chars."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", Path(video).stem)[:40] or "video"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _remove_raw(out_dir: Path) -> None:
    for p in out_dir.glob("_raw_*.jpg"):
        p.unlink(missing_ok=True)


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    every_s: float = 2.0,
    max_frames: int = 200,
    scene_threshold: Optional[float] = None,
    scene_max_gap_s: Optional[float] = None,
    dedup_max_distance: Optional[int] = 3,
    keep_dropped: bool = False,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    timeout_s: Optional[float] = None,
) -> Extraction:
    """Extract frames to `out_dir/<slug>_t<ms>.jpg`, then drop near-duplicates against the last kept frame.

    Interval mode samples one frame per `every_s` (ffmpeg `fps=` filter; the frame is the first source
    frame at or after each slot, so `t_s` is the nominal slot time, within one frame period of the true
    time). Scene mode (`scene_threshold` set) keeps frame 0 plus every ffmpeg scene change, with an
    optional forced frame every `scene_max_gap_s`. `dedup_max_distance=None` disables the dHash pass,
    `0` drops exact-hash repeats only. Dropped frames are deleted, or moved to the sibling folder
    `<out_dir>_dropped/` when `keep_dropped` (a sibling, never a subfolder, because ingest_folder rglobs).
    Stale frames of the same slug in `out_dir` are removed first so a rerun never double-counts.
    """
    video, out_dir = Path(video), Path(out_dir)
    if not 0.1 <= every_s <= 3600:
        raise VideoError(f"every_s must be between 0.1 and 3600 s, got {every_s}")
    if not 1 <= max_frames <= 5000:
        raise VideoError(f"max_frames must be between 1 and 5000, got {max_frames}")
    if scene_threshold is not None and not 0 < scene_threshold < 1:
        raise VideoError(f"scene_threshold must be strictly between 0 and 1, got {scene_threshold}")
    if scene_max_gap_s is not None and scene_max_gap_s <= 0:
        raise VideoError(f"scene_max_gap_s must be positive, got {scene_max_gap_s}")
    if dedup_max_distance is not None and dedup_max_distance < 0:
        raise VideoError(f"dedup_max_distance must be >= 0 or None, got {dedup_max_distance}")
    if start_s < 0:
        raise VideoError(f"start_s must be >= 0, got {start_s}")
    if end_s is not None and end_s <= start_s:
        raise VideoError(f"end_s ({end_s}) must be greater than start_s ({start_s})")

    info = probe(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_of(video)
    _remove_raw(out_dir)
    for stale in out_dir.glob(f"{slug}_t????????.jpg"):
        stale.unlink(missing_ok=True)

    mode = "scene" if scene_threshold is not None else "interval"
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(video)]
    if end_s is not None:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    if mode == "interval":
        rate = Fraction(1 / every_s).limit_denominator(1000)
        cmd += ["-vf", f"fps={rate}", "-fps_mode", "passthrough"]
    else:
        expr = f"eq(n,0)+gt(scene,{scene_threshold})"
        if scene_max_gap_s is not None:
            expr += f"+gte(t-prev_selected_t,{scene_max_gap_s})"
        cmd += ["-vf", f"select='{expr}'", "-fps_mode", "vfr"]
    cmd += ["-enc_time_base", "1/1000", "-frame_pts", "1", "-frames:v", str(max_frames), "-q:v", "2", "-pix_fmt", "yuvj420p", "-f", "image2", str(out_dir / "_raw_%08d.jpg")]

    if timeout_s is None:
        timeout_s = 120.0 + 2.0 * info.duration_s
    t0 = time.perf_counter()
    try:
        cp = _run(cmd, timeout_s=timeout_s)
    except VideoError:
        _remove_raw(out_dir)
        raise
    seconds_ffmpeg = time.perf_counter() - t0
    if cp.returncode != 0:
        _remove_raw(out_dir)
        raise VideoError(f"{video.name}: ffmpeg failed ({_stderr_tail(cp)})")

    frames: List[Frame] = []
    for raw in sorted(out_dir.glob("_raw_*.jpg")):
        ms = int(raw.stem[len("_raw_") :])
        target = out_dir / f"{slug}_t{ms:08d}.jpg"
        raw.replace(target)
        frames.append(Frame(path=str(target), ms=ms, t_s=round(start_s + ms / 1000.0, 3)))

    clip_end = min(end_s, info.duration_s) if end_s is not None else info.duration_s
    truncated = len(frames) >= max_frames and (mode == "scene" or frames[-1].t_s + every_s < clip_end)

    t1 = time.perf_counter()
    if dedup_max_distance is not None:
        dropped_dir = out_dir.parent / f"{out_dir.name}_dropped"
        last: Optional[Fingerprint] = None
        for fr in frames:
            fp = fingerprint(Path(fr.path))
            if last is not None and is_near_duplicate(fp, last, dedup_max_distance):
                fr.kept, fr.distance = False, hamming(fp.dhash, last.dhash)
                src = Path(fr.path)
                if keep_dropped:
                    dropped_dir.mkdir(parents=True, exist_ok=True)
                    dest = dropped_dir / src.name
                    src.replace(dest)
                    fr.path = str(dest)
                else:
                    src.unlink()
            else:
                fr.kept, fr.distance = True, (None if last is None else hamming(fp.dhash, last.dhash))
                last = fp
    seconds_dedup = time.perf_counter() - t1

    ext = Extraction(
        video=info,
        out_dir=str(out_dir),
        mode=mode,
        every_s=every_s,
        scene_threshold=scene_threshold,
        scene_max_gap_s=scene_max_gap_s,
        max_frames=max_frames,
        dedup_max_distance=dedup_max_distance,
        start_s=start_s,
        end_s=end_s,
        frames=frames,
        truncated=truncated,
        seconds_ffmpeg=round(seconds_ffmpeg, 3),
        seconds_dedup=round(seconds_dedup, 3),
        command=cmd,
        ffmpeg_version=ffmpeg_version(),
    )
    _write_json(out_dir / "frames.json", ext.to_json())
    return ext


# ----------------------------------------------------------------------------- ingest


def _date_of(iso: Optional[str]) -> Optional[str]:
    """YYYY-MM-DD from an ISO-ish timestamp such as '2026-03-14T10:22:31.000000Z'; None when malformed."""
    if not iso or len(iso) < 10:
        return None
    head = iso[:10]
    try:
        date.fromisoformat(head)
    except ValueError:
        return None
    return head


def ingest_video(
    video: Path,
    out_dir: Path,
    *,
    asset_class: AssetClass,
    client_id: Optional[str] = None,
    asset_id: Optional[str] = None,
    every_s: float = 2.0,
    max_frames: int = 200,
    scene_threshold: Optional[float] = None,
    scene_max_gap_s: Optional[float] = None,
    dedup_max_distance: Optional[int] = 3,
    captured_on: Optional[str] = None,
    captured_on_from_mtime: bool = False,
    gsd_mm_per_px: Optional[float] = None,
    irradiance_wm2: Optional[float] = None,
    source_dataset: str = "video",
    split: str = "upload",
    id_prefix: Optional[str] = None,
    keep_dropped: bool = False,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
) -> tuple[List[ImageRecord], Extraction]:
    """Extract, dedup, write `sidecar.json` + `frames.json`, then `ingest_folder` the kept frames.

    `captured_on` precedence: explicit argument, container creation_time, file mtime (only when
    `captured_on_from_mtime`, off by default: an upload's or a copied clip's mtime is not a capture
    date, and clientreport anchors SLA due dates on captured_on), else None. Records carry `source_video` (absolute path) and `frame_time_s`; image ids read
    `vid_<slug>_t<ms:08d>` so a finding id says how far into which clip it was seen.
    """
    video, out_dir = Path(video), Path(out_dir)
    ext = extract_frames(
        video,
        out_dir,
        every_s=every_s,
        max_frames=max_frames,
        scene_threshold=scene_threshold,
        scene_max_gap_s=scene_max_gap_s,
        dedup_max_distance=dedup_max_distance,
        keep_dropped=keep_dropped,
        start_s=start_s,
        end_s=end_s,
    )
    captured = captured_on or _date_of(ext.video.creation_time)
    if captured is None and captured_on_from_mtime:
        captured = datetime.fromtimestamp(video.stat().st_mtime).date().isoformat()
    slug = slug_of(video)
    source = str(video.resolve())
    sidecar: dict = {}
    for fr in ext.kept:
        sidecar[Path(fr.path).name] = {
            "image_id": f"vid_{slug}_t{fr.ms:08d}",
            "source_video": source,
            "frame_time_s": fr.t_s,
            "captured_on": captured,
            "client_id": client_id,
            "asset_id": asset_id,
            "gsd_mm_per_px": gsd_mm_per_px,
            "irradiance_wm2": irradiance_wm2,
        }
    sidecar_path = out_dir / "sidecar.json"
    _write_json(sidecar_path, sidecar)
    _write_json(out_dir / "frames.json", {**ext.to_json(), "sidecar": str(sidecar_path)})
    records = ingest_folder(
        out_dir,
        asset_class=asset_class,
        sidecar=sidecar_path,
        source_dataset=source_dataset,
        split=split,
        id_prefix=id_prefix or f"vid_{slug}",
        client_id=client_id,
        asset_id=asset_id,
    )
    return records, ext


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m cascade.video", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="mp4/mov/avi/mkv file")
    ap.add_argument("--out", required=True, help="folder for the extracted frames (frames.json and sidecar.json are written beside them)")
    ap.add_argument("--asset-class", required=True, choices=list(get_args(AssetClass)))
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames (interval mode)")
    ap.add_argument("--scene", type=float, default=None, metavar="THR", help="scene mode: keep frame 0 and every scene change above THR (0-1; ffmpeg docs suggest 0.3-0.5)")
    ap.add_argument("--scene-max-gap", type=float, default=None, metavar="S", help="scene mode: also force a frame at least every S seconds")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--dedup", type=int, default=3, metavar="BITS", help="drop a frame whose dHash is within BITS of the last kept frame (0 = exact repeats only)")
    ap.add_argument("--no-dedup", action="store_true", help="keep every extracted frame")
    ap.add_argument("--keep-dropped", action="store_true", help="move near-duplicates to <out>_dropped/ instead of deleting them")
    ap.add_argument("--client-id", default=None)
    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip to start")
    ap.add_argument("--end", type=float, default=None, help="seconds into the clip to stop")
    ap.add_argument("--captured-on", default=None, help="YYYY-MM-DD; overrides the container creation_time and the file mtime")
    ap.add_argument("--mtime-date", action="store_true", help="fall back to the file mtime for captured_on (off by default: a copied clip's mtime is the copy date, not the capture date)")
    ap.add_argument("--manifest", default=None, help="manifest to write (default: <out>/../manifest.jsonl)")
    return ap


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    manifest = Path(args.manifest) if args.manifest else out.parent / "manifest.jsonl"
    try:
        records, ext = ingest_video(
            Path(args.video),
            out,
            asset_class=args.asset_class,
            client_id=args.client_id,
            asset_id=args.asset_id,
            every_s=args.every,
            max_frames=args.max_frames,
            scene_threshold=args.scene,
            scene_max_gap_s=args.scene_max_gap,
            dedup_max_distance=None if args.no_dedup else args.dedup,
            captured_on=args.captured_on,
            captured_on_from_mtime=args.mtime_date,
            keep_dropped=args.keep_dropped,
            start_s=args.start,
            end_s=args.end,
        )
    except VideoError as e:
        print(str(e), file=sys.stderr)
        return 2
    write_manifest(records, manifest)
    info = ext.video
    print(
        json.dumps(
            {
                "video": info.path,
                "sha256": info.sha256,
                "duration_s": info.duration_s,
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "rotation": info.rotation,
                "mode": ext.mode,
                "extracted": len(ext.frames),
                "kept": len(ext.kept),
                "dropped": len(ext.dropped),
                "truncated": ext.truncated,
                "seconds_ffmpeg": ext.seconds_ffmpeg,
                "seconds_dedup": ext.seconds_dedup,
                "frames_dir": str(out),
                "manifest": str(manifest),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
