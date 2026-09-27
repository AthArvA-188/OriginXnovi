"""Window geometry and pixel preprocessing shared by training, evaluation and the heatmap.

Numpy + PIL only, so it imports in both the training env (no pydantic) and the app env.
The same functions cut the BFDD out-of-domain test windows and the website heatmap windows,
so the OOD number describes the geometry users see (verifier recommendation).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

WINDOW_PX = 224  # model input size; heatmap windows are cut at native resolution with this size
DEFAULT_STRIDE_PX = 224  # non-overlapping by default (latency; verifier recommendation)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

Box = Tuple[int, int, int, int]


def window_starts(length: int, win: int = WINDOW_PX, stride: int = DEFAULT_STRIDE_PX) -> List[int]:
    """Start offsets covering [0, length): regular steps, plus one window flush with the far edge.

    A side shorter than the window gets a single start at 0 (the crop is padded later)."""
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")
    if length <= win:
        return [0]
    s = list(range(0, length - win + 1, stride))
    if s[-1] != length - win:
        s.append(length - win)
    return s


def window_boxes(width: int, height: int, win: int = WINDOW_PX, stride: int = DEFAULT_STRIDE_PX) -> List[Box]:
    """Row-major (y then x) window boxes (x0, y0, x1, y1), clipped to the image."""
    xs = window_starts(width, win, stride)
    ys = window_starts(height, win, stride)
    return [(x, y, min(x + win, width), min(y + win, height)) for y in ys for x in xs]


def grid_shape(width: int, height: int, win: int = WINDOW_PX, stride: int = DEFAULT_STRIDE_PX) -> Tuple[int, int]:
    return len(window_starts(height, win, stride)), len(window_starts(width, win, stride))


def crop_window(img: Image.Image, box: Box, win: int = WINDOW_PX) -> Image.Image:
    """Crop a box; pad with edge-mean grey when the image is smaller than the window."""
    c = img.crop(box)
    if c.size != (win, win):
        canvas = Image.new("RGB", (win, win), tuple(int(v) for v in np.asarray(c).reshape(-1, 3).mean(0)))
        canvas.paste(c, (0, 0))
        c = canvas
    return c


def to_input(tile: Image.Image, size: int = WINDOW_PX) -> np.ndarray:
    """PIL RGB -> float32 CHW normalised with ImageNet statistics (bilinear resize to size x size)."""
    t = tile.convert("RGB")
    if t.size != (size, size):
        t = t.resize((size, size), Image.BILINEAR)
    a = np.asarray(t, dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def cap_long_side(img: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    """Downscale so the long side is <= max_side. Returns (image, factor) where factor<=1."""
    m = max(img.size)
    if m <= max_side:
        return img, 1.0
    f = max_side / m
    return img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.LANCZOS), f
