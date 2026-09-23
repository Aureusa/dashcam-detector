"""Tests for app.render (PLAN §8 acceptance cases)."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import RenderConfig
from app.render import color_for_class, draw_detections, draw_hud, render_frame, resize_for_output
from fakes import FakeDetection


def det(x1, y1, x2, y2, score=0.9, cid=2, label="car"):
    return FakeDetection(x1, y1, x2, y2, score, cid, label)


@pytest.fixture
def img():
    return np.zeros((240, 320, 3), np.uint8)


@pytest.fixture
def cfg():
    return RenderConfig()


def test_color_deterministic_and_bgr():
    for cid in range(0, 100):
        c = color_for_class(cid)
        assert c == color_for_class(cid)
        assert len(c) == 3 and all(0 <= v <= 255 for v in c)
    assert color_for_class(0) != color_for_class(2)


def test_empty_list(img, cfg):
    out = draw_detections(img, [], cfg)
    assert out is img and out.shape == (240, 320, 3)
    assert not out.any()


def test_draws_in_place(img, cfg):
    out = draw_detections(img, [det(50, 60, 150, 160)], cfg)
    assert out is img
    assert img.any()


@pytest.mark.parametrize("d", [
    det(-50, -50, 100, 100),          # partially outside top-left
    det(250, 200, 999, 999),          # partially outside bottom-right
    det(-500, -500, -10, -10),        # fully outside
    det(0, 0, 319, 239),              # full frame / at edges
    det(0, 0, 5, 5),                  # at corner, tiny
    det(319, 239, 319, 239),          # degenerate at bottom-right corner
    det(100, 100, 100.4, 100.4),      # very small
    det(200, 150, 100, 50),           # inverted coords
    det(10, 10, 60, 60, label="a very very long label name that exceeds the frame width " * 3),
    det(float("nan"), 0, float("inf"), 50),  # garbage coords
    det(10, 0, 50, 40),               # at top edge: label must go inside the box
])
def test_edge_cases_no_exceptions(img, cfg, d):
    out = draw_detections(img, [d], cfg)
    assert out.shape == (240, 320, 3) and out.dtype == np.uint8


def test_label_inside_box_when_at_top(cfg):
    img = np.zeros((240, 320, 3), np.uint8)
    draw_detections(img, [det(100, 0, 200, 100)], cfg)
    assert img[5:15, 100:150].any()  # tag drawn inside the box area near the top


def test_many_detections(img, cfg):
    dets = [det(i, i, i + 20, i + 20, cid=i % 80, label=f"c{i}") for i in range(0, 200, 3)]
    out = draw_detections(img, dets, cfg)
    assert out.shape == (240, 320, 3)


def test_hud(img, cfg):
    stats = {"capture_fps": 30.0, "inference_fps": 29.5, "inference_ms": 12.3, "e2e_ms": 40.0,
             "num_detections": 3, "device": "cuda:0", "model": "rtdetr_v2_r18vd"}
    out = draw_hud(img, stats, cfg)
    assert out is img and img[:20, :100].any()
    draw_hud(np.zeros((10, 10, 3), np.uint8), stats, cfg)  # tiny image
    draw_hud(np.zeros((100, 100, 3), np.uint8), {}, cfg)   # missing keys


def test_hud_darkens_background(cfg):
    img = np.full((240, 320, 3), 200, np.uint8)
    draw_hud(img, {"capture_fps": 1}, cfg)
    assert img[2, 2].max() < 200          # darkened corner
    assert (img[200:, 300:] == 200).all()  # rest untouched


def test_output_width_resize(cfg):
    img = np.zeros((720, 1280, 3), np.uint8)
    out = resize_for_output(img, 640)
    assert out.shape == (360, 640, 3)
    assert resize_for_output(img, 0) is img
    cfg.output_width = 960
    out = render_frame(np.zeros((720, 1280, 3), np.uint8), [det(10, 10, 100, 100)], cfg, {"capture_fps": 1})
    assert out.shape == (540, 960, 3)
