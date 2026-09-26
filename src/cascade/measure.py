"""Crack metrology from imagery: pixel-scale resolution, dark-thin-structure mask, skeleton, widths.

User request (verbatim): "also can we have the crack measurements with the data if there is any
scale present in the image? to get the dimentions of the crack as a number?"

What this module does
- Resolves a pixel scale (mm/px) from three sources, in the order the pipeline prefers them
  (docs/research/10_multisensor_scope.md section 4.3): image metadata (`ImageRecord.gsd_mm_per_px`),
  a scale object whose endpoints were located by a model or detector, or two manually clicked points.
- Segments a dark, thin, elongated structure inside a bbox (grayscale -> gaussian background
  subtraction -> percentile threshold -> small-component removal -> elongation filter).
- Thins the mask with Zhang-Suen and reads length along the skeleton and width as
  2 x euclidean distance transform on skeleton pixels (max / p95 / median).
- Converts to mm / cm2 only when a scale is present; otherwise reports pixels and leaves the
  millimetre fields None so the grader's `not_measurable` path stays in force.
- Places the p95 width against the MBEI crack-width rows of src/cascade/rubrics/bridge_mbei.json,
  quoting the matched row verbatim and listing every state the +/- band touches.

Honest limits (read before trusting a number)
- The mask is a heuristic, not a learned segmenter. Anything dark and thin is a "crack" to it:
  shadows cast by edges, construction joints, formwork lines, rebar impressions, cables, wires,
  tar bleed and graffiti strokes all pass. Blobby shadows and stains are rejected by the
  elongation filter, thin ones are not.
- Widths are read perpendicular to the skeleton through the distance transform, which quantises
  to whole pixels; every width carries a +/- 1 px band (WIDTH_UNCERTAINTY_PX). Oblique views make
  widths read high (research note section 4.3); no view-angle correction is applied.
- Published fiducial-marker methods report 0.16-0.22 mm error (research note section 4.2), the
  same order as the 0.30 mm RC CS1/CS2 boundary, so the MBEI hint is a hint for the reviewer,
  never a grade. The grader and the reviewer decide the condition state.
- No ruler / marker detection by vision is implemented here. `scale_from_reference` takes
  endpoints that a Claude call in the grader (or a detector) supplies later; tests use given points.

All thresholds below are labelled: a rubric source, an arithmetic consequence, or a team assumption.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

from .schema import ImageRecord

ScaleBasis = Literal["gsd_metadata", "scale_object", "manual_two_points"]
ElementType = Literal["reinforced", "prestressed"]
ImageLike = Union[Image.Image, np.ndarray, str, Path]

RUBRIC_PATH = Path(__file__).parent / "rubrics" / "bridge_mbei.json"

# --- constants; every one is a team assumption unless a source is named ------------------------

# Segmentation (team assumptions, tuned on the synthetic fixtures in tests/test_measure.py only)
BACKGROUND_SIGMA_PX = 8.0  # gaussian sigma for the background estimate; must exceed the crack width
DARK_PERCENTILE = 97.0  # percentile of local darkness used as the noise floor
MIN_CONTRAST = 0.08  # gray units in [0, 1] (about 20/255): below this nothing is "dark"
MIN_COMPONENT_PX = 30  # connected components smaller than this are noise
MIN_AXIS_RATIO = 3.0  # sqrt(lambda_major / lambda_minor) from second moments; straight-ish cracks
MIN_THINNESS = 4.0  # area / (2 * max_dt)^2, about length / width; meandering cracks pass this one
MIN_MASK_PX = 30  # below this the measurement is not attempted

# Confidence (team assumptions). Segmentation is heuristic so the ceiling is deliberately low.
SEGMENTATION_CONFIDENCE = 0.6
SCALE_CONFIDENCE = {"gsd_metadata": 0.8, "scale_object": 0.7, "manual_two_points": 0.6}
MANUAL_SCALE_PENALTY = 0.75  # applied in to_measurements when the basis is manual_two_points

# Width band: 2 x distance transform quantises to whole pixels (arithmetic consequence of the method)
WIDTH_UNCERTAINTY_PX = 1.0

# MBEI crack-width boundaries in inches, copied from src/cascade/rubrics/bridge_mbei.json rows:
#   reinforced:  CS1 "Width less than 0.012 in", CS2 "0.012 to 0.05 in", CS3 "greater than 0.05 in"
#   prestressed: CS1 "Width less than 0.004 in", CS2 "0.004 to 0.009 in", CS3 "greater than 0.009 in"
MBEI_WIDTH_BOUNDS_IN = {"reinforced": (0.012, 0.05), "prestressed": (0.004, 0.009)}
MBEI_DEFECT_NAME = {"reinforced": "cracking (reinforced concrete)", "prestressed": "cracking (prestressed concrete)"}
MM_PER_IN = 25.4


# --- scale ---------------------------------------------------------------------------------------


@dataclass
class Scale:
    """Pixel scale of an image or crop.

    mm_per_px: millimetres per image pixel.
    basis: where it came from; gsd_metadata is the pipeline's normal source (ImageRecord.gsd_mm_per_px).
    detail: human-readable provenance (field name, marker label, the two points).
    confidence: team-assumption prior per basis (SCALE_CONFIDENCE), not a measured accuracy.
    """

    mm_per_px: float
    basis: ScaleBasis
    detail: str
    confidence: float


def scale_from_gsd(record: Optional[ImageRecord]) -> Optional[Scale]:
    """Scale from the record's ground sample distance, or None when the record has none."""
    if record is None or record.gsd_mm_per_px is None or record.gsd_mm_per_px <= 0:
        return None
    return Scale(
        mm_per_px=float(record.gsd_mm_per_px),
        basis="gsd_metadata",
        detail=f"ImageRecord.gsd_mm_per_px={record.gsd_mm_per_px} ({record.image_id})",
        confidence=SCALE_CONFIDENCE["gsd_metadata"],
    )


def _segment_px(p0: Sequence[float], p1: Sequence[float]) -> float:
    return float(np.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))


def scale_from_points(p0: Tuple[int, int], p1: Tuple[int, int], known_mm: float) -> Scale:
    """Manual calibration: two clicked pixel points (x, y) on a ruler, tape or gauge and the known
    distance between them in mm. Raises ValueError when the points coincide or known_mm <= 0."""
    if known_mm <= 0:
        raise ValueError("known_mm must be positive")
    d = _segment_px(p0, p1)
    if d < 1.0:
        raise ValueError("the two points must be at least 1 px apart")
    return Scale(
        mm_per_px=known_mm / d,
        basis="manual_two_points",
        detail=f"{known_mm} mm between {tuple(int(v) for v in p0)} and {tuple(int(v) for v in p1)} = {d:.1f} px",
        confidence=SCALE_CONFIDENCE["manual_two_points"],
    )


def scale_from_reference(img: ImageLike, endpoints_px: List[Tuple[int, int]], known_mm: float, label: str) -> Scale:
    """Scale from a reference object of known size whose endpoints were located by a model or detector.

    Same arithmetic as scale_from_points; the basis is "scale_object" and the label names the object
    (for example "comparator card 50 mm bar", "ArUco 40 mm", "tape 0-100 mm"). Ruler / marker
    detection by vision is NOT implemented here: a Claude call in the grader can return the two
    endpoints later (the image is accepted so that call has a home); tests pass given points.
    Endpoints must lie inside the image."""
    if len(endpoints_px) != 2:
        raise ValueError("endpoints_px must hold exactly two (x, y) points")
    if known_mm <= 0:
        raise ValueError("known_mm must be positive")
    w, h = _image_size(img)
    for x, y in endpoints_px:
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"endpoint {(x, y)} is outside the {w}x{h} image")
    d = _segment_px(endpoints_px[0], endpoints_px[1])
    if d < 1.0:
        raise ValueError("the two endpoints must be at least 1 px apart")
    return Scale(
        mm_per_px=known_mm / d,
        basis="scale_object",
        detail=f"{label}: {known_mm} mm over {d:.1f} px between {tuple(endpoints_px[0])} and {tuple(endpoints_px[1])}",
        confidence=SCALE_CONFIDENCE["scale_object"],
    )


# --- image helpers -------------------------------------------------------------------------------


def _to_gray(img: ImageLike) -> np.ndarray:
    """Float32 grayscale in [0, 1] from a PIL image, a path, or a 2-D / 3-D array."""
    if isinstance(img, (str, Path)):
        img = Image.open(Path(img))
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).convert("L"))
    integer_input = np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool
    arr = arr.astype(np.float32)
    if integer_input or arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _image_size(img: ImageLike) -> Tuple[int, int]:
    if isinstance(img, (str, Path)):
        with Image.open(Path(img)) as im:
            return im.size
    if isinstance(img, Image.Image):
        return img.size
    arr = np.asarray(img)
    return int(arr.shape[1]), int(arr.shape[0])


def _clip_bbox(bbox: Optional[Sequence[int]], w: int, h: int) -> Tuple[int, int, int, int]:
    if not bbox:
        return 0, 0, w, h
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, x1 = sorted((max(0, x0), min(w, x1)))
    y0, y1 = sorted((max(0, y0), min(h, y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"bbox {tuple(bbox)} is empty after clipping to {w}x{h}")
    return x0, y0, x1, y1


# --- mask ----------------------------------------------------------------------------------------


def _local_darkness(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Background minus gray, positive where a pixel is darker than its surroundings.

    Two passes: the first background estimate is pulled down by the crack itself, so pixels the first
    pass calls dark are replaced by that background and the blur is repeated. This removes most of the
    self-dip for structures narrower than sigma."""
    bg = ndimage.gaussian_filter(gray, sigma)
    diff = bg - gray
    first = diff > max(MIN_CONTRAST, 0.5 * float(np.percentile(diff, 99.9)))
    if first.any():
        filled = np.where(first, bg, gray)
        bg = ndimage.gaussian_filter(filled, sigma)
        diff = bg - gray
    return diff


def _axis_ratio(ys: np.ndarray, xs: np.ndarray) -> float:
    """sqrt(major / minor eigenvalue) of the pixel-coordinate covariance (second central moments)."""
    if len(ys) < 3:
        return 1.0
    cov = np.cov(np.vstack([xs.astype(np.float64), ys.astype(np.float64)]))
    ev = np.linalg.eigvalsh(cov)
    lo, hi = max(float(ev[0]), 1e-6), max(float(ev[1]), 1e-6)
    return float(np.sqrt(hi / lo))


def crack_mask(
    img: ImageLike,
    bbox: Optional[Sequence[int]] = None,
    dark_percentile: Optional[float] = None,
    *,
    background_sigma: float = BACKGROUND_SIGMA_PX,
    min_component_px: int = MIN_COMPONENT_PX,
    min_axis_ratio: float = MIN_AXIS_RATIO,
    min_thinness: float = MIN_THINNESS,
) -> np.ndarray:
    """Boolean mask of dark, thin, elongated structure inside bbox (returned at the bbox's size).

    Steps: grayscale -> local darkness = gaussian background - gray (two-pass, see _local_darkness)
    -> threshold at max(MIN_CONTRAST, percentile) capped at half the robust peak (a half-maximum
    width definition, so antialiased edge pixels split evenly) -> drop components under
    min_component_px -> keep components that are elongated by either test: second-moment axis ratio
    >= min_axis_ratio (straight cracks) or thinness = area / (2 * max_dt)^2 >= min_thinness
    (meandering cracks, roughly length >= 4 x width; a disc scores 0.79).

    Limits: shadows along edges, joints, formwork lines, rebar impressions, wires and dark stripes are
    kept; a uniform or featureless crop returns an all-False mask. Nothing here knows what a crack is."""
    gray = _to_gray(img)
    h, w = gray.shape
    x0, y0, x1, y1 = _clip_bbox(bbox, w, h)
    gray = gray[y0:y1, x0:x1]
    pct = DARK_PERCENTILE if dark_percentile is None else float(dark_percentile)

    diff = _local_darkness(gray, background_sigma)
    peak = float(np.percentile(diff, 99.9))
    if peak < MIN_CONTRAST:
        return np.zeros(gray.shape, dtype=bool)
    thr = max(MIN_CONTRAST, float(np.percentile(diff, pct)))
    thr = min(thr, 0.5 * peak)
    raw = diff > thr

    labels, n = ndimage.label(raw, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros(gray.shape, dtype=bool)
    sizes = ndimage.sum(raw, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None or float(sizes[i - 1]) < min_component_px:
            continue
        comp = labels[sl] == i
        if _is_elongated(comp, min_axis_ratio, min_thinness):
            keep[i] = True
    return keep[labels]


def _is_elongated(comp: np.ndarray, min_axis_ratio: float, min_thinness: float) -> bool:
    """Elongation of one connected component, judged on its hole-filled shape.

    Background subtraction flattens the interior of any dark region wider than the blur (a shadow,
    stain or patch) and leaves its outline as a thin closed ring, which would pass a thinness test;
    filling holes first turns the ring back into the disc it outlines, and the disc fails. The cost is
    that closed crack loops (map-cracking cells) are rejected too: pattern cracking is graded from the
    rubric's pattern rows, not from a width. Open arcs of a broken rim still pass; a documented limit."""
    filled = ndimage.binary_fill_holes(np.pad(comp, 1))
    area = float(filled.sum())
    dt = ndimage.distance_transform_edt(filled)
    thinness = area / max(1.0, (2.0 * float(dt.max())) ** 2)
    if thinness >= min_thinness:
        return True
    ys, xs = np.nonzero(filled)
    return _axis_ratio(ys, xs) >= min_axis_ratio


# --- skeleton ------------------------------------------------------------------------------------


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning (1984), two vectorised sub-iterations per pass until nothing changes.

    Neighbour order P2..P9 clockwise from north; A = 0->1 transitions around the ring, B = neighbour
    count. Sub-iteration 1 removes when P2*P4*P6 == 0 and P4*P6*P8 == 0; sub-iteration 2 when
    P2*P4*P8 == 0 and P2*P6*P8 == 0. Output is an 8-connected, 1-px-wide skeleton; ends are shortened
    by about half the width, and rasterised diagonals can leave short spurs."""
    img = np.asarray(mask, dtype=bool).astype(np.uint8)
    if img.ndim != 2 or not img.any():
        return np.zeros(np.asarray(mask).shape, dtype=bool)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[:-2, 1:-1]
            P3 = p[:-2, 2:]
            P4 = p[1:-1, 2:]
            P5 = p[2:, 2:]
            P6 = p[2:, 1:-1]
            P7 = p[2:, :-2]
            P8 = p[1:-1, :-2]
            P9 = p[:-2, :-2]
            ring = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(int_arr.astype(np.int16) for int_arr in ring[:8])
            A = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c1, c2 = (P2 * P4 * P6) == 0, (P4 * P6 * P8) == 0
            else:
                c1, c2 = (P2 * P4 * P8) == 0, (P2 * P6 * P8) == 0
            remove = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return _remove_redundant(img)


_RING_OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))  # P2..P9 clockwise from north


def _redundant_lut() -> np.ndarray:
    """For each of the 256 neighbourhood patterns (bit i = ring position i set): True when the set
    neighbours number at least two and stay one 8-connected component with the centre removed.
    Such a centre pixel is redundant for connectivity. Built once with ndimage.label."""
    lut = np.zeros(256, dtype=bool)
    eight = np.ones((3, 3), dtype=int)
    for code in range(256):
        win = np.zeros((3, 3), dtype=np.uint8)
        for i, (dy, dx) in enumerate(_RING_OFFSETS):
            if code >> i & 1:
                win[1 + dy, 1 + dx] = 1
        if win.sum() >= 2 and ndimage.label(win, structure=eight)[1] == 1:
            lut[code] = True
    return lut


_REDUNDANT = _redundant_lut()


def _remove_redundant(img: np.ndarray) -> np.ndarray:
    """Sequential clean-up after Zhang-Suen, which is known to leave 2-px-thick staircases on
    diagonals. A skeleton pixel is deleted when its set neighbours remain one 8-connected component
    without it (lookup table _REDUNDANT): endpoints, line interiors (two neighbours on opposite sides)
    and junctions whose arms only meet through the centre are kept. Sequential, not vectorised, so two
    adjacent deletions cannot cut the line; skeletons are small so the Python loop is cheap."""
    p = np.pad(np.asarray(img, dtype=np.uint8), 1)
    changed = True
    while changed:
        changed = False
        ys, xs = np.nonzero(p)
        for y, x in zip(ys, xs):
            code = 0
            for i, (dy, dx) in enumerate(_RING_OFFSETS):
                if p[y + dy, x + dx]:
                    code |= 1 << i
            if _REDUNDANT[code]:
                p[y, x] = 0
                changed = True
    return p[1:-1, 1:-1].astype(bool)


def _skeleton_widths_px(mask: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """Width at each skeleton pixel = 2 x euclidean distance transform, with the boundary placed at
    the half pixel: the mask is upsampled 2x (nearest), the transform is taken there, and each
    skeleton pixel reads the larger of its two-by-two sub-pixels (the one nearer the medial axis).
    On the plain mask the estimator reads a 6 px band at 31 degrees as 7.2 px because the nearest
    background pixel centre sits half a pixel past the edge; the upsampled read halves that bias
    (measured in tests/test_measure.py). Still whole-pixel quantised: see WIDTH_UNCERTAINTY_PX."""
    up = np.repeat(np.repeat(mask.astype(np.uint8), 2, axis=0), 2, axis=1)
    dt_up = ndimage.distance_transform_edt(up)
    h, w = mask.shape
    block_max = dt_up.reshape(h, 2, w, 2).max(axis=(1, 3))
    return 2.0 * block_max[skel] / 2.0


def _skeleton_length_px(skel: np.ndarray) -> float:
    """Sum of link lengths in the 8-connected skeleton graph: 1 per orthogonal link, sqrt(2) per
    diagonal link. Diagonal links that close a triangle with two orthogonal links are skipped so a
    staircase is not double counted."""
    s = skel.astype(bool)
    horiz = s[:, :-1] & s[:, 1:]
    vert = s[:-1, :] & s[1:, :]
    d1 = s[:-1, :-1] & s[1:, 1:]  # down-right
    d2 = s[:-1, 1:] & s[1:, :-1]  # down-left
    # a down-right diagonal (r,c)-(r+1,c+1) is redundant if (r,c+1) or (r+1,c) is also set
    d1 = d1 & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = d2 & ~s[:-1, :-1] & ~s[1:, 1:]
    return float(horiz.sum() + vert.sum() + np.sqrt(2.0) * (d1.sum() + d2.sum()))


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    p = np.pad(skel.astype(np.int16), 1)
    total = sum(
        p[1 + dy : p.shape[0] - 1 + dy, 1 + dx : p.shape[1] - 1 + dx]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    return np.where(skel, total, 0)


# --- metrics -------------------------------------------------------------------------------------


def crack_metrics(mask: np.ndarray, scale: Optional[Scale] = None) -> dict:
    """Length, width statistics, area and topology of a mask, in px and (with a scale) in mm / cm2.

    width = 2 x euclidean distance transform sampled on skeleton pixels (perpendicular half-width
    doubled); reported as max, p95 and median. length_px follows the skeleton links. junctions are
    8-connected clusters of skeleton pixels with three or more neighbours; endpoints have one.
    "measurable" is False with a reason when the mask is too small, thinning leaves nothing, or
    (for the mm fields only) the scale is missing; px fields are still filled when they exist."""
    m = np.asarray(mask, dtype=bool)
    reasons: List[str] = []
    out: dict = {
        "mask_pixels": int(m.sum()),
        "area_px": int(m.sum()),
        "length_px": None,
        "width_px_max": None,
        "width_px_p95": None,
        "width_px_median": None,
        "skeleton_pixels": 0,
        "junctions": 0,
        "endpoints": 0,
        "crack_length_mm": None,
        "crack_width_mm_max": None,
        "crack_width_mm_p95": None,
        "crack_width_mm_median": None,
        "area_cm2": None,
        "width_uncertainty_mm": None,
        "mm_per_px": scale.mm_per_px if scale else None,
        "scale_basis": scale.basis if scale else None,
        "measurable": False,
        "reasons": reasons,
    }
    if out["mask_pixels"] < MIN_MASK_PX:
        reasons.append(f"mask too small ({out['mask_pixels']} px < {MIN_MASK_PX})")
        if scale is None:
            reasons.append("scale missing")
        return out
    skel = skeletonize(m)
    n_skel = int(skel.sum())
    out["skeleton_pixels"] = n_skel
    if n_skel == 0:
        reasons.append("no skeleton after thinning")
        if scale is None:
            reasons.append("scale missing")
        return out
    widths = _skeleton_widths_px(m, skel)
    nc = _neighbour_count(skel)
    junction_px = nc >= 3
    _, n_junctions = ndimage.label(junction_px, structure=np.ones((3, 3), dtype=int))
    out.update(
        length_px=_skeleton_length_px(skel),
        width_px_max=float(widths.max()),
        width_px_p95=float(np.percentile(widths, 95)),
        width_px_median=float(np.median(widths)),
        junctions=int(n_junctions),
        endpoints=int((nc == 1).sum()),
    )
    if scale is None:
        reasons.append("scale missing: widths in px only")
        out["measurable"] = False
        return out
    k = scale.mm_per_px
    out.update(
        crack_length_mm=out["length_px"] * k,
        crack_width_mm_max=out["width_px_max"] * k,
        crack_width_mm_p95=out["width_px_p95"] * k,
        crack_width_mm_median=out["width_px_median"] * k,
        area_cm2=out["area_px"] * k * k / 100.0,
        width_uncertainty_mm=WIDTH_UNCERTAINTY_PX * k,
        measurable=True,
    )
    return out


# --- MBEI hint -----------------------------------------------------------------------------------


def _rubric_rows(element_type: ElementType) -> dict:
    rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    name = MBEI_DEFECT_NAME[element_type]
    return {row["value"]: row for row in rubric["rows"] if row["defect"] == name}


def _state_for_width_in(width_in: float, bounds: Tuple[float, float]) -> str:
    lo, hi = bounds
    if width_in < lo:
        return "CS1"
    if width_in > hi:
        return "CS3"
    return "CS2"


def mbei_hint(width_mm: float, uncertainty_mm: float, element_type: ElementType = "reinforced") -> dict:
    """Which MBEI crack-width row the p95 width falls in, with the row text quoted verbatim from
    bridge_mbei.json, the boundary used (inches, as the rubric states it) and every state the
    width +/- uncertainty band touches. A hint for the reviewer; the grade is not decided here."""
    bounds = MBEI_WIDTH_BOUNDS_IN[element_type]
    width_in = width_mm / MM_PER_IN
    u_in = uncertainty_mm / MM_PER_IN
    state = _state_for_width_in(width_in, bounds)
    band = sorted({_state_for_width_in(width_in - u_in, bounds), state, _state_for_width_in(width_in + u_in, bounds)})
    rows = _rubric_rows(element_type)
    row = rows.get(state)
    return {
        "element_type": element_type,
        "value": state,
        "criterion": row["criterion"] if row else None,
        "unified": row["unified"] if row else None,
        "width_in": round(width_in, 4),
        "width_mm": round(width_mm, 3),
        "band_states": band,
        "thresholds_in": {"CS1_below": bounds[0], "CS3_above": bounds[1]},
        "source": f"{RUBRIC_PATH.name} rows for '{MBEI_DEFECT_NAME[element_type]}'",
        "note": ("width band straddles a boundary; both states are candidates" if len(band) > 1 else "width band lies inside one state") + "; hint only, the grader and reviewer decide",
    }


# --- measurement record --------------------------------------------------------------------------


@dataclass
class CrackMeasurement:
    """One crack measurement. Millimetre fields are None whenever no scale was available."""

    measurable: bool
    crack_width_mm_max: Optional[float]
    crack_width_mm_p95: Optional[float]
    crack_width_mm_median: Optional[float]
    crack_length_mm: Optional[float]
    area_cm2: Optional[float]
    width_uncertainty_mm: Optional[float]
    width_px_max: Optional[float]
    width_px_p95: Optional[float]
    width_px_median: Optional[float]
    length_px: Optional[float]
    area_px: int
    mask_pixels: int
    junctions: int
    endpoints: int
    scale_basis: Optional[ScaleBasis]
    mm_per_px: Optional[float]
    confidence: float
    mbei_condition_state_hint: Optional[dict]
    bbox: Optional[List[int]]
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _load_image(img: ImageLike) -> Image.Image:
    if isinstance(img, (str, Path)):
        return Image.open(Path(img)).convert("RGB")
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(np.asarray(img))


def measure_crack(
    img: ImageLike,
    record: Optional[ImageRecord] = None,
    bbox: Optional[Sequence[int]] = None,
    scale: Optional[Scale] = None,
    *,
    element_type: ElementType = "reinforced",
    dark_percentile: Optional[float] = None,
) -> CrackMeasurement:
    """Mask -> skeleton -> metrics -> MBEI hint for one image or crop.

    Scale precedence: an explicit `scale`, else the record's gsd_mm_per_px, else none (px only).
    bbox is (x0, y0, x1, y1) in image pixels, normally Evidence.bbox. element_type picks the
    reinforced or prestressed MBEI rows (the prestressed ladder is three times tighter)."""
    image = _load_image(img)
    x0, y0, x1, y1 = _clip_bbox(bbox, image.width, image.height)
    used_scale = scale or scale_from_gsd(record)
    notes: List[str] = []
    mask = crack_mask(image, (x0, y0, x1, y1), dark_percentile)
    m = crack_metrics(mask, used_scale)
    notes.extend(m["reasons"])
    if used_scale is not None:
        notes.append(f"scale: {used_scale.basis} ({used_scale.detail})")
        notes.append(f"width band is +/- {WIDTH_UNCERTAINTY_PX:g} px = +/- {WIDTH_UNCERTAINTY_PX * used_scale.mm_per_px:.3f} mm (pixel quantisation only; published marker methods report 0.16-0.22 mm error, research note 10 section 4.2)")
    else:
        notes.append("no scale: crack_width_mm stays None and the grader keeps not_measurable (bridge_mbei.json unified_note)")
    notes.append("mask is a dark-thin-structure heuristic: shadows, joints, rebar lines and wires can be counted as crack; reviewer must confirm")
    hint = None
    if m["measurable"]:
        hint = mbei_hint(m["crack_width_mm_p95"], m["width_uncertainty_mm"], element_type)
        notes.append(f"MBEI hint uses the p95 width {m['crack_width_mm_p95']:.3f} mm = {hint['width_in']:.4f} in against {hint['thresholds_in']} ({hint['source']})")
    has_px = m["width_px_max"] is not None
    confidence = 0.0
    if has_px:
        confidence = SEGMENTATION_CONFIDENCE * (used_scale.confidence if used_scale else 1.0)
        notes.append(f"confidence = segmentation prior {SEGMENTATION_CONFIDENCE} x scale prior {used_scale.confidence if used_scale else 1.0} (team assumptions)")
    return CrackMeasurement(
        measurable=bool(m["measurable"]),
        crack_width_mm_max=m["crack_width_mm_max"],
        crack_width_mm_p95=m["crack_width_mm_p95"],
        crack_width_mm_median=m["crack_width_mm_median"],
        crack_length_mm=m["crack_length_mm"],
        area_cm2=m["area_cm2"],
        width_uncertainty_mm=m["width_uncertainty_mm"],
        width_px_max=m["width_px_max"],
        width_px_p95=m["width_px_p95"],
        width_px_median=m["width_px_median"],
        length_px=m["length_px"],
        area_px=int(m["area_px"]),
        mask_pixels=int(m["mask_pixels"]),
        junctions=int(m["junctions"]),
        endpoints=int(m["endpoints"]),
        scale_basis=used_scale.basis if used_scale else None,
        mm_per_px=used_scale.mm_per_px if used_scale else None,
        confidence=round(confidence, 3),
        mbei_condition_state_hint=hint,
        bbox=[x0, y0, x1, y1],
        notes=notes,
    )


def to_measurements(cm: CrackMeasurement) -> dict:
    """schema.Measurements-compatible dict (Measurements.model_validate ignores the extra "notes" key).

    crack_width_mm is the p95 width, never the max: the max sits on the widest single skeleton pixel,
    which the distance transform pins to corner fills, spalled edges and joints; p95 tracks the crack
    body while still reporting the wide end. Confidence is lowered by MANUAL_SCALE_PENALTY when the
    scale came from two clicked points. Without a scale every millimetre field is None."""
    notes = list(cm.notes)
    notes.append("crack_width_mm = p95 skeleton width, not max (max is one pixel and lands on corner fills and joints)")
    confidence = cm.confidence
    if cm.scale_basis == "manual_two_points":
        confidence = round(confidence * MANUAL_SCALE_PENALTY, 3)
        notes.append(f"confidence x {MANUAL_SCALE_PENALTY}: scale is a manual two-point calibration (team assumption)")
    return {
        "area_cm2": cm.area_cm2 if cm.measurable else None,
        "crack_width_mm": cm.crack_width_mm_p95 if cm.measurable else None,
        "delta_t_k": None,
        "percent_area_rusted": None,
        "section_loss_pct": None,
        "confidence": float(min(1.0, max(0.0, confidence))),
        "notes": notes,
    }


# --- CLI -----------------------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o).__name__}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cascade.measure", description="Crack width / length / area from one image, in px and (with a scale) in mm.")
    p.add_argument("--image", required=True, help="image path")
    p.add_argument("--bbox", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"), help="crop box in image pixels")
    p.add_argument("--gsd", type=float, help="ground sample distance in mm/px (metadata scale)")
    p.add_argument("--points", nargs=4, type=int, metavar=("X", "Y", "X", "Y"), help="two pixel points on a ruler or gauge (manual scale)")
    p.add_argument("--mm", type=float, help="known distance between --points in mm")
    p.add_argument("--element", choices=["reinforced", "prestressed"], default="reinforced", help="MBEI crack row family")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    scale: Optional[Scale] = None
    if args.gsd is not None:
        scale = Scale(mm_per_px=float(args.gsd), basis="gsd_metadata", detail=f"--gsd {args.gsd}", confidence=SCALE_CONFIDENCE["gsd_metadata"])
    elif args.points is not None:
        if args.mm is None:
            raise SystemExit("--points needs --mm")
        x0, y0, x1, y1 = args.points
        scale = scale_from_points((x0, y0), (x1, y1), args.mm)
    cm = measure_crack(Path(args.image), bbox=args.bbox, scale=scale, element_type=args.element)
    out = {"measurement": cm.as_dict(), "measurements_contract": to_measurements(cm)}
    print(json.dumps(out, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
