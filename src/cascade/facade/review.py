"""Glue from heatmap tiles to the EXISTING VLM grader (facade_ll11 rubric) and cascade.measure.

Nothing here files, approves or actuates anything: each tile becomes a review note with
status "pending human review". The grader is only called when ANTHROPIC_API_KEY is set; without
it the caller gets a clear "grader needs ANTHROPIC_API_KEY" result and no grade is invented.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from .heatmap import TileScore

ROOT = Path(__file__).resolve().parents[3]
DISCLAIMER = "AI screening, not a QEWI/FISP finding. A qualified inspector decides; photos and drones do not replace close-up inspection (1 RCNY 103-04)."
ASSET_CLASS = "facade_element"


def load_env() -> None:
    """Load the repo .env (same file the console app loads) without overriding set variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass


def grader_status() -> Tuple[bool, str]:
    load_env()
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True, f"grader ready (model {os.environ.get('GRADER_MODEL', 'claude-opus-5')}, rubric facade_ll11.json)"
    return False, "grader needs ANTHROPIC_API_KEY (set it in .env); tiles were not sent and no grade was produced"


def pad_box(box: Sequence[int], size: Tuple[int, int], frac: float = 0.25) -> Tuple[int, int, int, int]:
    """Grow a window by `frac` on each side (context for the grader), clipped to the image."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    W, H = size
    return (max(0, int(x0 - frac * w)), max(0, int(y0 - frac * h)), min(W, int(x1 + frac * w)), min(H, int(y1 + frac * h)))


def grade_tiles(img: Image.Image, tiles: Sequence[TileScore], *, image_id: str, backend: str = "claude",
                gsd_mm_per_px: Optional[float] = None, log=None) -> List[dict]:
    """Send each tile crop to cascade.grade.grade_image(asset_class='facade_element').

    Returns one dict per tile: {rank, box, p_crack, finding (dict) | None, error | None}.
    Raises nothing for a missing key: returns entries with error set instead."""
    ok, why = grader_status()
    if backend == "claude" and not ok:
        return [{"rank": t.rank, "box": list(t.box), "p_crack": t.p_crack, "finding": None, "error": why} for t in tiles]
    from cascade.grade import grade_image
    from cascade.schema import Evidence

    out = []
    src = img.convert("RGB")
    for t in tiles:
        crop_box = pad_box(t.box, src.size)
        crop = src.crop(crop_box)
        ev = Evidence(image_ids=[image_id], bbox=list(crop_box), tile=f"window r{t.row}c{t.col}", gsd_mm_per_px=gsd_mm_per_px)
        try:
            f = grade_image(crop, finding_id=f"{image_id}-tile{t.rank:02d}", image_id=image_id, asset_class=ASSET_CLASS,
                            backend=backend, evidence=ev, log=log,
                            metadata={"source": "facade heatmap pre-gate", "p_crack_tile_classifier": round(t.p_crack, 4)})
            out.append({"rank": t.rank, "box": list(t.box), "p_crack": t.p_crack, "finding": f.model_dump(), "error": None})
        except Exception as e:  # network / API errors surface on screen, never as a fake grade
            out.append({"rank": t.rank, "box": list(t.box), "p_crack": t.p_crack, "finding": None, "error": f"{type(e).__name__}: {e}"})
    return out


def measure_tile(img: Image.Image, tile: TileScore, *, mm_per_px: Optional[float] = None,
                 ref_px: Optional[float] = None, ref_mm: Optional[float] = None) -> dict:
    """Crack width via cascade.measure inside the tile box. A scale must exist: either a known
    mm/px (e.g. from GSD metadata) or a reference length (ref_px pixels = ref_mm mm) on the photo.
    Without a scale, widths stay in pixels and mm fields are None (no invented millimetres)."""
    from cascade.measure import Scale, measure_crack, scale_from_points, to_measurements

    scale = None
    if mm_per_px and mm_per_px > 0:
        scale = Scale(mm_per_px=float(mm_per_px), basis="gsd_metadata", detail=f"user-entered {mm_per_px} mm/px", confidence=0.8)
    elif ref_px and ref_mm and ref_px >= 1 and ref_mm > 0:
        scale = scale_from_points((0, 0), (float(ref_px), 0), float(ref_mm))
    cm = measure_crack(img.convert("RGB"), bbox=list(tile.box), scale=scale)
    meas = to_measurements(cm)
    return {"measurable_mm": cm.measurable, "width_px_p95": cm.width_px_p95, "length_px": cm.length_px,
            "crack_width_mm": meas["crack_width_mm"], "crack_width_uncertainty_mm": meas["crack_width_uncertainty_mm"],
            "crack_length_mm": meas["crack_length_mm"], "basis": meas["measurement_basis"],
            "upper_bound": cm.width_upper_bound, "notes": cm.notes}


def review_note(image_id: str, tile: TileScore, *, graded: Optional[dict] = None, measured: Optional[dict] = None,
                model_id: str = "", threshold: Optional[float] = None) -> dict:
    """The record a human reviewer sees and signs off. Status is always pending until a person acts."""
    f = (graded or {}).get("finding") or {}
    return {
        "image_id": image_id, "tile_rank": tile.rank, "box_px": list(tile.box), "p_crack": round(tile.p_crack, 4),
        "screen_model": model_id, "threshold": threshold,
        "above_threshold": (tile.p_crack >= threshold) if threshold is not None else None,
        "fisp_class": (f.get("native_scale") or {}).get("value"), "unified_level": (f.get("unified") or {}).get("level"),
        "grader_error": (graded or {}).get("error"),
        "crack_width_mm": (measured or {}).get("crack_width_mm"),
        "crack_width_uncertainty_mm": (measured or {}).get("crack_width_uncertainty_mm"),
        "status": "pending human review", "created": dt.datetime.now().isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
    }
