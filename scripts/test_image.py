#!/usr/bin/env python3
"""Run the RT-DETR detector on one image, print detections + timing, and save an annotated copy.

Usage (from the project root):
    .venv/bin/python scripts/test_image.py --image samples/coco/000000039769.jpg \
        --out samples/cats_det.jpg [--reference] [--checkpoint C] [--threshold T] [--all-classes]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import ConfigError, RenderConfig, load_config, setup_logging  # noqa: E402
from app.detector import Detection, RTDetrDetector  # noqa: E402


def _fallback_draw(img: np.ndarray, dets: list[Detection], cfg: RenderConfig) -> np.ndarray:
    """Minimal drawer used only if app.render is not available."""
    for d in dets:
        color = tuple(int(c) for c in np.random.default_rng(d.class_id).integers(64, 256, 3))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, color, cfg.line_thickness)
        cv2.putText(img, f"{d.label} {d.score:.2f}", (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, color, 1, cv2.LINE_AA)
    return img


def get_drawer():
    """Return app.render.draw_detections if importable, else the inline fallback."""
    try:
        from app.render import draw_detections  # type: ignore[attr-defined]

        return draw_detections, "app.render"
    except Exception as exc:  # noqa: BLE001 - any import problem -> fallback
        logging.getLogger(__name__).warning("app.render unavailable (%s); using inline fallback drawer", exc)
        return _fallback_draw, "fallback"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--out", required=True, help="annotated output image path")
    ap.add_argument("--reference", action="store_true", help="use the HF image processor (CPU) for preprocessing")
    ap.add_argument("--checkpoint", default=None, help="override model checkpoint")
    ap.add_argument("--threshold", type=float, default=None, help="override score threshold")
    ap.add_argument("--all-classes", action="store_true", help="disable the class filter from config")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="config file (default: project config.yaml)")
    ap.add_argument("--runs", type=int, default=5, help="timed runs after one warm-up (default 5)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    try:
        cfg = load_config(args.config if Path(args.config).is_file() else None)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    mcfg = cfg.model
    if args.checkpoint:
        mcfg.checkpoint = args.checkpoint
    if args.threshold is not None:
        mcfg.score_threshold = args.threshold
    if args.device:
        mcfg.device = args.device
    if args.all_classes:
        mcfg.classes = []

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"cannot read image: {args.image}", file=sys.stderr)
        return 2

    det = RTDetrDetector(mcfg, use_reference_preprocess=args.reference)
    det.warmup(img.shape[0], img.shape[1])  # cudnn autotune + per-thread CUDA init
    times = []
    dets: list[Detection] = []
    for _ in range(max(1, args.runs)):
        dets = det.detect(img)
        times.append(det.last_timings)

    h, w = img.shape[:2]
    print(f"\nimage {args.image} ({w}x{h}) | model {det.model_name} on {det.device} | "
          f"preprocess={'reference(HF)' if args.reference else 'fast(GPU)'} | threshold {det.threshold:.2f} | "
          f"filter {det.class_filter or 'ALL'}")
    print(f"{len(dets)} detections:")
    for d in dets:
        print(f"  {d.label:<15s} {d.score:.3f}  box=({d.x1:.1f}, {d.y1:.1f}, {d.x2:.1f}, {d.y2:.1f})")
    keys = ("preprocess", "forward", "postprocess", "total")
    mean = {k: sum(t[k] for t in times) / len(times) for k in keys}
    print("timing (mean of %d runs, ms): " % len(times) + "  ".join(f"{k}={mean[k]:.2f}" for k in keys))

    draw, which = get_drawer()
    out = draw(img.copy(), dets, cfg.render)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.out, out):
        print(f"failed to write {args.out}", file=sys.stderr)
        return 1
    print(f"saved annotated image ({which} drawer) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
