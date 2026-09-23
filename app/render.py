"""Drawing helpers: bounding boxes, labels and the stats HUD (OpenCV, in place)."""

from __future__ import annotations

import colorsys
import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from app.config import RenderConfig
    from app.detector import Detection

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GOLDEN = 0.618033988749895

# Distinct, high-contrast colors for the most common driving classes (COCO ids), BGR.
_FIXED_COLORS: dict[int, tuple[int, int, int]] = {
    0: (60, 76, 231),     # person  - red
    1: (219, 152, 52),    # bicycle - blue
    2: (113, 204, 46),    # car     - green
    3: (182, 89, 155),    # motorcycle - purple
    5: (15, 196, 241),    # bus     - yellow
    7: (34, 126, 230),    # truck   - orange
    9: (156, 188, 26),    # traffic light - teal
    11: (43, 57, 192),    # stop sign - dark red
}


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR color for a class id."""
    cid = int(class_id)
    if cid in _FIXED_COLORS:
        return _FIXED_COLORS[cid]
    h = (cid * _GOLDEN) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever contrasts better with the BGR background."""
    b, g, r = bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def draw_detections(img: np.ndarray, dets: Iterable["Detection"], cfg: "RenderConfig") -> np.ndarray:
    """Draw boxes and ``"label score"`` tags on ``img`` in place and return it.

    Works with any objects exposing ``x1, y1, x2, y2, score, class_id, label``.
    Coordinates are clamped; degenerate/outside boxes are skipped or drawn minimally.
    """
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    thickness = max(1, int(cfg.line_thickness))
    font_scale = max(0.1, float(cfg.font_scale))
    text_thick = max(1, thickness // 2)

    for d in dets:
        x1 = int(round(min(max(_finite(d.x1), 0.0), w - 1)))
        y1 = int(round(min(max(_finite(d.y1), 0.0), h - 1)))
        x2 = int(round(min(max(_finite(d.x2), 0.0), w - 1)))
        y2 = int(round(min(max(_finite(d.y2), 0.0), h - 1)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        color = color_for_class(int(getattr(d, "class_id", 0)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        text = f"{d.label} {_finite(d.score):.2f}"
        (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, text_thick)
        pad = 3
        box_h = th + baseline + pad
        # Above the box if there is room, otherwise inside it (top edge).
        if y1 - box_h >= 0:
            ty1 = y1 - box_h
        else:
            ty1 = min(y1, max(0, h - box_h))
        ty2 = min(h - 1, ty1 + box_h)
        tx1 = min(x1, max(0, w - tw - 2 * pad))
        tx2 = min(w - 1, tx1 + tw + 2 * pad)
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, cv2.FILLED)
        cv2.putText(img, text, (tx1 + pad, ty2 - baseline - 1), _FONT, font_scale,
                    _text_color(color), text_thick, cv2.LINE_AA)
    return img


def format_hud_lines(stats: Mapping[str, Any]) -> list[str]:
    """Build the HUD text lines from a stats dict (missing keys are shown as '-')."""

    def num(key: str, fmt: str) -> str:
        v = stats.get(key)
        if v is None:
            return "-"
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    return [
        f"cap {num('capture_fps', '.1f')} fps | inf {num('inference_fps', '.1f')} fps",
        f"inf {num('inference_ms', '.1f')} ms | e2e {num('e2e_ms', '.0f')} ms",
        f"dets {stats.get('num_detections', '-')} | {stats.get('device', '-')}",
        f"{stats.get('model', '-')}",
    ]


def draw_hud(img: np.ndarray, stats: Mapping[str, Any], cfg: "RenderConfig") -> np.ndarray:
    """Draw a semi-transparent stats box in the top-left corner (in place)."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    lines = format_hud_lines(stats)
    scale = max(0.1, float(cfg.font_scale))
    thick = 1
    sizes = [cv2.getTextSize(t, _FONT, scale, thick) for t in lines]
    line_h = max(s[0][1] + s[1] for s in sizes) + 6
    box_w = min(w, max(s[0][0] for s in sizes) + 16)
    box_h = min(h, line_h * len(lines) + 10)
    x0, y0 = 0, 0
    roi = img[y0:y0 + box_h, x0:x0 + box_w]
    if roi.size:
        # Darken the ROI in place (alpha ~0.55) without allocating a full-frame overlay.
        cv2.convertScaleAbs(roi, dst=roi, alpha=0.45, beta=0)
    for i, t in enumerate(lines):
        y = y0 + 6 + line_h * (i + 1) - 6
        if y >= h:
            break
        cv2.putText(img, t, (x0 + 8, y), _FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def resize_for_output(img: np.ndarray, output_width: int) -> np.ndarray:
    """Resize (keeping aspect ratio) to ``output_width`` if > 0 and different; else return ``img``."""
    if not output_width or output_width <= 0 or img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if w == output_width:
        return img
    new_h = max(1, int(round(h * output_width / w)))
    interp = cv2.INTER_AREA if output_width < w else cv2.INTER_LINEAR
    return cv2.resize(img, (int(output_width), new_h), interpolation=interp)


def render_frame(img: np.ndarray, dets: Iterable["Detection"], cfg: "RenderConfig",
                 hud_stats: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """Draw detections (+ HUD if enabled and stats given), then apply ``output_width``."""
    draw_detections(img, dets, cfg)
    if cfg.draw_hud and hud_stats is not None:
        draw_hud(img, hud_stats, cfg)
    return resize_for_output(img, cfg.output_width)
