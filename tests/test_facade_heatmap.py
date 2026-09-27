"""Facade tile geometry, heatmap, NMS/top-K, overlay, grader glue and crack measurement glue.

Images drawn inside these tests are SYNTHETIC-FOR-TEST (a grey wall with a dark line); they
check plumbing, not accuracy. Accuracy lives in eval/facade/tilecls_v1.json (REAL data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from cascade.facade import heatmap as hm
from cascade.facade import review
from cascade.facade.tiles import (WINDOW_PX, cap_long_side, crop_window, grid_shape, to_input, window_boxes,
                                  window_starts)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "facade"


def wall(w=600, h=460, line=True) -> Image.Image:
    im = Image.new("RGB", (w, h), (170, 168, 160))
    if line:
        ImageDraw.Draw(im).line([(40, 60), (300, 260)], fill=(40, 40, 40), width=4)
    return im


class FakeClf:
    """Scores a window by its darkness so tests are deterministic without the ONNX file."""

    model_id = "fake:dark@test"
    threshold = 0.5

    def predict(self, x):
        return np.clip(1.0 - (x.mean(axis=(1, 2, 3)) + 2.0) / 4.0, 0, 1) + np.linspace(0, 1e-6, len(x))


def test_window_starts_cover_edges():
    assert window_starts(224) == [0]
    assert window_starts(100) == [0]
    assert window_starts(600, 224, 224) == [0, 224, 376]
    assert window_starts(448, 224, 224) == [0, 224]
    s = window_starts(1000, 224, 112)
    assert s[0] == 0 and s[-1] == 1000 - 224 and all(b - a <= 112 for a, b in zip(s, s[1:]))
    with pytest.raises(ValueError):
        window_starts(10, 0, 5)


def test_window_boxes_in_bounds_and_grid_shape():
    boxes = window_boxes(640, 512)
    r, c = grid_shape(640, 512)
    assert len(boxes) == r * c == 9
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 512
        assert x1 - x0 == WINDOW_PX and y1 - y0 == WINDOW_PX


def test_small_image_padding_and_input_tensor():
    im = wall(150, 120)
    c = crop_window(im, window_boxes(150, 120)[0])
    assert c.size == (WINDOW_PX, WINDOW_PX)
    x = to_input(c)
    assert x.shape == (3, WINDOW_PX, WINDOW_PX) and x.dtype == np.float32


def test_cap_long_side():
    im, f = cap_long_side(wall(3000, 1500), 1500)
    assert im.size == (1500, 750) and f == pytest.approx(0.5)
    im2, f2 = cap_long_side(wall(), 5000)
    assert f2 == 1.0 and im2.size == (600, 460)


def test_score_image_grid_topk_nms_and_overlay_are_deterministic():
    im = wall()
    a = hm.score_image(im, FakeClf())
    b = hm.score_image(im, FakeClf())
    assert a.grid.shape == grid_shape(600, 460)
    np.testing.assert_allclose(a.grid, b.grid)
    for t in a.tiles:
        assert 0 <= t.box[0] < t.box[2] <= 600 and 0 <= t.box[1] < t.box[3] <= 460
        assert 0.0 <= t.p_crack <= 1.0
    top = hm.top_k(a, k=3)
    assert [t.rank for t in top] == [1, 2, 3]
    assert [t.p_crack for t in top] == sorted([t.p_crack for t in top], reverse=True)
    half = hm.score_image(im, FakeClf(), stride=WINDOW_PX // 2)
    top_half = hm.top_k(half, k=3)
    for i in range(len(top_half)):
        for j in range(i + 1, len(top_half)):
            assert hm.iou(top_half[i].box, top_half[j].box) <= hm.NMS_IOU
    ov = hm.overlay(im, a, threshold=0.5, highlight=top)
    assert ov.size == im.size
    assert hm.prob_map(a).shape == (460, 600)


def test_score_image_rescales_boxes_to_original():
    im = wall(3000, 1000)
    r = hm.score_image(im, FakeClf(), max_side=1500)
    assert r.scale_factor == pytest.approx(0.5)
    assert max(t.box[2] for t in r.tiles) == 3000 and max(t.box[3] for t in r.tiles) == 1000
    assert any("downscaled" in n for n in r.notes)


def test_nms_suppresses_overlaps():
    T = hm.TileScore
    tiles = [T((0, 0, 100, 100), 0.9, 0, 0), T((10, 10, 110, 110), 0.8, 0, 1), T((300, 300, 400, 400), 0.7, 1, 1)]
    kept = hm.nms(tiles, 0.3)
    assert [t.p_crack for t in kept] == [0.9, 0.7]


def test_side_by_side_review_list_keeps_edge_flush_windows():
    # 640x512 (BFDD frame size): 3x3 windows; the bottom row (y=288) overlaps the middle row (y=224).
    # Review-list regression: NMS used to cap the list at 6 and could hide a cracked edge-row square.
    im = wall(640, 512)
    r = hm.score_image(im, FakeClf())
    assert r.n_windows == 9
    assert len(hm.review_candidates(r)) == 9
    top = hm.top_k(r, k=12)
    assert len(top) == 9 and [t.rank for t in top] == list(range(1, 10))
    assert [t.p_crack for t in top] == sorted((t.p_crack for t in r.tiles), reverse=True)
    # explicit override keeps the old suppression available
    assert len(hm.top_k(r, k=12, iou_thr=hm.NMS_IOU)) < 9


def test_colormap_endpoints():
    c = hm.colormap(np.array([0.0, 1.0]))
    assert c.dtype == np.uint8 and tuple(c[0]) == (255, 245, 200) and tuple(c[1]) == (150, 15, 25)


def test_grader_without_key_is_skipped_never_faked(monkeypatch):
    monkeypatch.setattr(review, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, msg = review.grader_status()
    assert not ok and "ANTHROPIC_API_KEY" in msg
    tile = hm.TileScore((0, 0, 224, 224), 0.9, 0, 0, rank=1)
    out = review.grade_tiles(wall(), [tile], image_id="test")
    assert out[0]["finding"] is None and "ANTHROPIC_API_KEY" in out[0]["error"]
    note = review.review_note("test", tile, graded=out[0], model_id="m", threshold=0.5)
    assert note["status"] == "pending human review" and note["fisp_class"] is None
    assert "not a QEWI/FISP finding" in note["disclaimer"]


def test_measure_tile_needs_a_scale_for_mm():
    im = wall()
    tile = hm.TileScore((0, 0, 360, 300), 0.9, 0, 0, rank=1)
    no_scale = review.measure_tile(im, tile)
    assert no_scale["crack_width_mm"] is None
    scaled = review.measure_tile(im, tile, ref_px=100, ref_mm=50)
    if scaled["measurable_mm"]:
        assert scaled["crack_width_mm"] > 0 and scaled["crack_width_uncertainty_mm"] is not None


def test_pad_box_clips():
    assert review.pad_box((0, 0, 100, 100), (120, 120), 0.25) == (0, 0, 120, 120)


def test_shipped_onnx_runs_if_present():
    if not hm.DEFAULT_MODEL.exists():
        pytest.skip("models/facade/tilecls_resnet18_v1.onnx not built yet")
    clf = hm.TileClassifier()
    im = Image.open(FIX / "sdnet_wall_cracked.jpg") if (FIX / "sdnet_wall_cracked.jpg").exists() else wall()
    r1 = hm.score_image(im, clf)
    r2 = hm.score_image(im, clf)
    np.testing.assert_allclose(r1.grid, r2.grid, rtol=1e-6)
    assert np.all((r1.grid >= 0) & (r1.grid <= 1))
    card = hm.load_card()
    assert card.get("model_id") == clf.model_id and 0 < card["threshold"] < 1
