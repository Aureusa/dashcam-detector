#!/usr/bin/env python3
"""Per-stage latency benchmark for the detection pipeline (detect + render + JPEG encode).

Frames are decoded up-front (decode is not timed), then each frame goes through
preprocess -> forward -> postprocess (RTDetrDetector.last_timings, CUDA-synchronized),
render (app.render.draw_detections [+ draw_hud]) and cv2.imencode JPEG.

Examples (from the project root):
    .venv/bin/python scripts/benchmark.py --source samples/clip.mp4 --frames 300
    .venv/bin/python scripts/benchmark.py --checkpoints r18vd,r34vd,r50vd,r101vd --resize 1280x720,800x600
    .venv/bin/python scripts/benchmark.py --checkpoint PekingU/rtdetr_v2_r50vd --no-fp16
If --source is missing/unreadable, COCO images from samples/coco (resized to 1280x720) are used.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.config import RenderConfig, load_config, setup_logging  # noqa: E402
from app.detector import Detection, RTDetrDetector  # noqa: E402

log = logging.getLogger("benchmark")
STAGES = ("preprocess", "forward", "postprocess", "render", "encode", "total")


def _fallback_draw(img: np.ndarray, dets: list[Detection], cfg: RenderConfig) -> np.ndarray:
    for d in dets:
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), cfg.line_thickness)
        cv2.putText(img, f"{d.label} {d.score:.2f}", (int(d.x1), max(int(d.y1) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def get_renderers():
    """Return (draw_detections, draw_hud_or_None, name); falls back to an inline drawer."""
    try:
        from app.render import draw_detections  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        log.warning("app.render unavailable (%s); using inline fallback drawer", exc)
        return _fallback_draw, None, "fallback"
    try:
        from app.render import draw_hud  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        draw_hud = None
    return draw_detections, draw_hud, "app.render"


def expand_checkpoint(name: str) -> str:
    """Allow short names like 'r18vd' or 'r50' for PekingU/rtdetr_v2_* checkpoints."""
    name = name.strip()
    if "/" in name or Path(name).exists():
        return name
    if not name.endswith("vd"):
        name += "vd"
    return f"PekingU/rtdetr_v2_{name}"


def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def load_frames(source: str | None, n: int) -> tuple[list[np.ndarray], str]:
    """Decode up to n frames from a video (looping if shorter), else COCO images at 1280x720."""
    frames: list[np.ndarray] = []
    if source and Path(source).is_file():
        cap = cv2.VideoCapture(source)
        while cap.isOpened() and len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if frames:
            while len(frames) < n:
                frames.extend(frames[: n - len(frames)])
            return frames, f"{source} ({frames[0].shape[1]}x{frames[0].shape[0]})"
        log.warning("could not decode %s; falling back to COCO images", source)
    elif source:
        log.warning("source %s not found; falling back to COCO images", source)
    imgs = [cv2.imread(str(p)) for p in sorted((PROJECT_ROOT / "samples" / "coco").glob("*.jpg"))]
    imgs = [cv2.resize(i, (1280, 720), interpolation=cv2.INTER_LINEAR) for i in imgs if i is not None]
    if not imgs:
        raise SystemExit("no frames: provide --source or put images in samples/coco")
    frames = [imgs[i % len(imgs)] for i in range(n)]
    return frames, f"samples/coco x{len(imgs)} resized to 1280x720"


def stats(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(np.percentile(a, 95))


def run_one(checkpoint: str, frames: list[np.ndarray], size: tuple[int, int] | None, args, cfg) -> dict:
    """Benchmark one checkpoint at one frame size; returns a summary dict."""
    mcfg = dataclasses.replace(cfg.model, checkpoint=checkpoint, fp16=not args.no_fp16,
                               compile=args.compile, warmup_iters=args.warmup)
    if args.all_classes:
        mcfg.classes = []
    if args.threshold is not None:
        mcfg.score_threshold = args.threshold
    rcfg = dataclasses.replace(cfg.render, output_width=args.output_width if args.output_width is not None
                               else cfg.render.output_width)
    quality = args.jpeg_quality if args.jpeg_quality is not None else cfg.server.jpeg_quality
    if size is not None:
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) if (f.shape[1], f.shape[0]) != size else f
                  for f in frames]
    h, w = frames[0].shape[:2]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    det = RTDetrDetector(mcfg, use_reference_preprocess=args.reference)
    t0 = time.perf_counter()
    det.warmup(h, w)
    for f in frames[: min(10, len(frames))]:  # also warm up on real content
        det.detect(f)
    warm_s = time.perf_counter() - t0
    draw, hud, _ = get_renderers()
    use_hud = hud is not None and rcfg.draw_hud
    fake_stats = {"capture_fps": 30.0, "inference_fps": 30.0, "inference_ms": 10.0, "e2e_latency_ms": 40.0,
                  "num_detections": 0, "device": str(det.device), "model": det.model_name}

    rec: dict[str, list[float]] = {k: [] for k in STAGES}
    ndets: list[int] = []
    for f in frames:
        img = f.copy()  # pipeline owns its frame; copy is outside the timed region
        t_start = time.perf_counter()
        dets = det.detect(img)
        t_det = time.perf_counter()
        img = draw(img, dets, rcfg)
        if use_hud:
            img = hud(img, {**fake_stats, "num_detections": len(dets)}, rcfg)
        if rcfg.output_width and rcfg.output_width != img.shape[1]:
            oh = int(round(img.shape[0] * rcfg.output_width / img.shape[1]))
            img = cv2.resize(img, (rcfg.output_width, oh), interpolation=cv2.INTER_AREA)
        t_render = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        t_end = time.perf_counter()
        assert ok
        lt = det.last_timings
        rec["preprocess"].append(lt["preprocess"])
        rec["forward"].append(lt["forward"])
        rec["postprocess"].append(lt["postprocess"])
        rec["render"].append((t_render - t_det) * 1e3)
        rec["encode"].append((t_end - t_render) * 1e3)
        rec["total"].append((t_end - t_start) * 1e3)
        ndets.append(len(dets))

    mem = {}
    if det.device.type == "cuda":
        mem = {"alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
               "reserved_mb": torch.cuda.max_memory_reserved() / 2**20}
    summary = {"checkpoint": checkpoint, "size": f"{w}x{h}", "fp16": mcfg.fp16 and det.device.type == "cuda",
               "compile": mcfg.compile, "frames": len(frames), "warmup_s": warm_s,
               "dets_mean": float(np.mean(ndets)), **{k: stats(v) for k, v in rec.items()}, **mem}
    del det
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def print_table(s: dict) -> None:
    print(f"\n== {s['checkpoint']} | frames {s['size']} | fp16={s['fp16']} compile={s['compile']} | "
          f"{s['frames']} frames, {s['dets_mean']:.1f} dets/frame, warmup {s['warmup_s']:.1f}s")
    print(f"   {'stage':<12}{'mean ms':>9}{'p95 ms':>9}")
    for k in STAGES:
        m, p = s[k]
        print(f"   {k:<12}{m:>9.2f}{p:>9.2f}")
    m, p = s["total"]
    line = f"   FPS (1000/mean total) {1000 / m:.1f}   FPS@p95 {1000 / p:.1f}"
    if "alloc_mb" in s:
        line += f"   GPU peak alloc {s['alloc_mb']:.0f} MB / reserved {s['reserved_mb']:.0f} MB"
    print(line)


def print_summary(rows: list[dict]) -> None:
    print("\n== summary (ms: mean/p95) ==")
    hdr = f"{'checkpoint':<26}{'frames':>10}{'forward':>14}{'detect':>14}{'render+enc':>12}{'total':>14}{'FPS':>7}{'GPU MB':>8}"
    print(hdr)
    for s in rows:
        det_m = s["preprocess"][0] + s["forward"][0] + s["postprocess"][0]
        re_m = s["render"][0] + s["encode"][0]
        print(f"{s['checkpoint'].split('/')[-1]:<26}{s['size']:>10}"
              f"{s['forward'][0]:>8.2f}/{s['forward'][1]:<5.2f}{det_m:>14.2f}{re_m:>12.2f}"
              f"{s['total'][0]:>8.2f}/{s['total'][1]:<5.2f}{1000 / s['total'][0]:>7.1f}{s.get('reserved_mb', 0):>8.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(PROJECT_ROOT / "samples" / "clip.mp4"), help="video file (default samples/clip.mp4)")
    ap.add_argument("--frames", type=int, default=300)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", default=None, help="one checkpoint (default: config.yaml)")
    g.add_argument("--checkpoints", default=None, help="comma list, e.g. r18vd,r34vd,r50vd,r101vd or full ids")
    ap.add_argument("--resize", default=None, help="frame size(s) WxH, comma list, e.g. 1280x720,800x600,640x480")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--compile", action="store_true", help="enable torch.compile (experimental)")
    ap.add_argument("--reference", action="store_true", help="use HF processor preprocessing")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--all-classes", action="store_true")
    ap.add_argument("--output-width", type=int, default=None)
    ap.add_argument("--jpeg-quality", type=int, default=None)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config if Path(args.config).is_file() else None)
    if args.checkpoints:
        ckpts = [expand_checkpoint(c) for c in args.checkpoints.split(",") if c.strip()]
    else:
        ckpts = [expand_checkpoint(args.checkpoint) if args.checkpoint else cfg.model.checkpoint]
    sizes = [parse_size(s) for s in args.resize.split(",")] if args.resize else [None]

    frames, desc = load_frames(args.source, args.frames)
    _, _, renderer = get_renderers()
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"source: {desc} | {len(frames)} frames | device: {dev} | torch {torch.__version__} | renderer: {renderer}")

    rows = []
    for ck in ckpts:
        for size in sizes:
            s = run_one(ck, frames, size, args, cfg)
            print_table(s)
            rows.append(s)
    if len(rows) > 1:
        print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
