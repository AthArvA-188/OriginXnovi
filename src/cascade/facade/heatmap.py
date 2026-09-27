"""Facade crack heatmap: ONNX tile classifier -> probability grid -> overlay -> top-K tiles.

AI screening for a human inspector, not a QEWI/FISP finding. The classifier was trained on
close-range concrete tiles (Ozgenel, SDNET2018 walls); its drone-facade behaviour is only what
the BFDD out-of-domain test in eval/facade/tilecls_v1.json measures.

Runtime: onnxruntime (CPU) + numpy + PIL, no torch and no matplotlib.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .tiles import DEFAULT_STRIDE_PX, WINDOW_PX, cap_long_side, crop_window, grid_shape, to_input, window_boxes

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "facade"
DEFAULT_MODEL = MODEL_DIR / "tilecls_resnet18_v1.onnx"
DEFAULT_CARD = MODEL_DIR / "tilecls_v1.card.json"
MAX_SIDE_PX = 2400  # long-side cap before tiling (latency); shown on screen when applied
NMS_IOU = 0.3


def load_card(path: Path = DEFAULT_CARD) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


class TileClassifier:
    """onnxruntime session for the exported classifier. Input float32 [N,3,224,224] -> p(crack) [N]."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, threads: int = 4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.path = Path(model_path)
        self.session = ort.InferenceSession(str(self.path), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        card = load_card(self.path.with_name("tilecls_v1.card.json")) if self.path == DEFAULT_MODEL else {}
        self.model_id = card.get("model_id", self.path.stem)
        self.threshold = card.get("threshold")

    def predict(self, x: np.ndarray, batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(x), batch):
            logits = self.session.run(None, {self.input_name: x[i:i + batch].astype(np.float32)})[0]
            z = logits - logits.max(1, keepdims=True)
            e = np.exp(z)
            out.append(e[:, 1] / e.sum(1))
        return np.concatenate(out) if out else np.zeros(0, np.float32)


@dataclass
class TileScore:
    box: Tuple[int, int, int, int]  # original image pixels
    p_crack: float
    row: int
    col: int
    rank: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeatmapResult:
    image_size: Tuple[int, int]
    work_size: Tuple[int, int]
    scale_factor: float  # work = original * factor (<= 1)
    window_px: int
    stride_px: int
    grid: np.ndarray  # (rows, cols) p(crack)
    tiles: List[TileScore]
    model_id: str
    seconds: float
    notes: List[str] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.tiles)


def image_hash(img: Image.Image) -> str:
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()[:16]


def score_image(img: Image.Image, clf: TileClassifier, *, window: int = WINDOW_PX, stride: int = DEFAULT_STRIDE_PX,
                max_side: int = MAX_SIDE_PX) -> HeatmapResult:
    """Tile at native resolution (after an optional long-side cap) and score every window."""
    t0 = time.time()
    src = img.convert("RGB")
    work, f = cap_long_side(src, max_side)
    boxes = window_boxes(work.width, work.height, window, stride)
    x = np.stack([to_input(crop_window(work, b, window), window) for b in boxes]) if boxes else np.zeros((0, 3, window, window), np.float32)
    p = clf.predict(x)
    rows, cols = grid_shape(work.width, work.height, window, stride)
    grid = p.reshape(rows, cols) if len(p) == rows * cols else np.zeros((rows, cols))
    tiles = []
    for i, (b, pi) in enumerate(zip(boxes, p)):
        ob = tuple(int(round(v / f)) for v in b)
        ob = (min(ob[0], src.width), min(ob[1], src.height), min(ob[2], src.width), min(ob[3], src.height))
        tiles.append(TileScore(box=ob, p_crack=float(pi), row=i // cols, col=i % cols))
    notes = []
    if f < 1.0:
        notes.append(f"photo downscaled by {f:.3f} (long side capped at {max_side} px) before tiling; windows are {window} px on the downscaled photo")
    if min(src.size) < window:
        notes.append("photo is smaller than one window; the window was padded")
    return HeatmapResult(image_size=src.size, work_size=work.size, scale_factor=f, window_px=window, stride_px=stride,
                         grid=grid, tiles=tiles, model_id=clf.model_id, seconds=time.time() - t0, notes=notes)


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(tiles: List[TileScore], iou_thr: float = NMS_IOU) -> List[TileScore]:
    kept: List[TileScore] = []
    for t in sorted(tiles, key=lambda t: -t.p_crack):
        if all(iou(t.box, k.box) <= iou_thr for k in kept):
            kept.append(t)
    return kept


def review_candidates(res: HeatmapResult, iou_thr: Optional[float] = None) -> List[TileScore]:
    """Windows eligible for the review list, highest p(crack) first.

    Side-by-side placement (stride >= window): every window is listed. The edge-flush windows
    overlap a neighbour (e.g. rows at y=224 and y=288 on a 512 px frame) but also cover pixels no
    other window covers, so suppressing them could hide a cracked edge strip from the reviewer.
    Half-overlapping placement: NMS at NMS_IOU so the list does not repeat one crack several times.
    Pass iou_thr explicitly to override (>= 1.0 disables suppression)."""
    if iou_thr is None:
        iou_thr = 1.0 if res.stride_px >= res.window_px else NMS_IOU
    if iou_thr >= 1.0:
        return sorted(res.tiles, key=lambda t: -t.p_crack)
    return nms(res.tiles, iou_thr)


def top_k(res: HeatmapResult, k: int = 8, iou_thr: Optional[float] = None, min_p: float = 0.0) -> List[TileScore]:
    out = [t for t in review_candidates(res, iou_thr) if t.p_crack >= min_p][:k]
    for i, t in enumerate(out, start=1):
        t.rank = i
    return out


# --- colour and overlay ------------------------------------------------------------------------

# Sequential ramp (light yellow -> orange -> dark red); low values are transparent in the overlay.
_RAMP = np.array([[255, 245, 200], [253, 200, 90], [245, 130, 40], [215, 60, 30], [150, 15, 25]], dtype=np.float32)


def colormap(p: np.ndarray) -> np.ndarray:
    """p in [0,1] -> uint8 RGB via linear interpolation over _RAMP."""
    p = np.clip(np.asarray(p, dtype=np.float32), 0, 1) * (len(_RAMP) - 1)
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, len(_RAMP) - 1)
    w = (p - i0)[..., None]
    return (_RAMP[i0] * (1 - w) + _RAMP[i1] * w).astype(np.uint8)


def prob_map(res: HeatmapResult) -> np.ndarray:
    """Per-pixel p(crack) at ORIGINAL size: mean over the windows covering each pixel."""
    W, H = res.image_size
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for t in res.tiles:
        x0, y0, x1, y1 = t.box
        acc[y0:y1, x0:x1] += t.p_crack
        cnt[y0:y1, x0:x1] += 1
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def overlay(img: Image.Image, res: HeatmapResult, *, alpha: float = 0.55, threshold: Optional[float] = None,
            highlight: Sequence[TileScore] = (), max_side: int = 1600) -> Image.Image:
    """Heat colours blended in proportion to p(crack) (p=0 leaves the photo untouched); windows at or
    above `threshold` get a thin outline; `highlight` tiles get a numbered thick outline."""
    base = img.convert("RGB")
    pm = prob_map(res)
    f = min(1.0, max_side / max(base.size))
    if f < 1.0:
        size = (max(1, round(base.width * f)), max(1, round(base.height * f)))
        base = base.resize(size, Image.LANCZOS)
        pm = np.asarray(Image.fromarray(pm).resize(size, Image.BILINEAR))
    rgb = np.asarray(base, dtype=np.float32)
    heat = colormap(pm).astype(np.float32)
    a = (alpha * pm)[..., None]
    out = Image.fromarray((rgb * (1 - a) + heat * a).clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    s = lambda b: tuple(int(round(v * f)) for v in b)  # noqa: E731
    if threshold is not None:
        for t in res.tiles:
            if t.p_crack >= threshold:
                d.rectangle(s(t.box), outline=(215, 60, 30), width=1)
    for t in highlight:
        bb = s(t.box)
        d.rectangle(bb, outline=(20, 90, 200), width=3)
        d.rectangle((bb[0], bb[1], bb[0] + 22, bb[1] + 18), fill=(20, 90, 200))
        d.text((bb[0] + 5, bb[1] + 3), str(t.rank), fill=(255, 255, 255))
    return out


def legend_strip(width: int = 240, height: int = 14) -> Image.Image:
    ramp = colormap(np.linspace(0, 1, width))[None, :, :].repeat(height, 0)
    return Image.fromarray(ramp)
