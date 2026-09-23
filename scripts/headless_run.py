#!/usr/bin/env python3
"""Run capture + InferenceWorker without Flask and report FPS / latency (PLAN §9.5).

Examples::

    .venv/bin/python scripts/headless_run.py --source samples/clip.mp4 --seconds 300
    .venv/bin/python scripts/headless_run.py --source samples/clip.mp4 --seconds 30 --fake-detector --fake-delay 0.2
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

from app.buffers import LatestValue  # noqa: E402
from app.capture import CameraStream, CaptureError  # noqa: E402
from app.config import ConfigError, add_cli_args, apply_overrides, load_config, setup_logging  # noqa: E402
from app.pipeline import InferenceWorker  # noqa: E402
from app.stats import PipelineMeters  # noqa: E402


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(ap)
    ap.add_argument("--seconds", type=float, default=300.0, help="run duration")
    ap.add_argument("--interval", type=float, default=5.0, help="stats line period (s)")
    ap.add_argument("--fake-detector", action="store_true", help="use tests/fakes.FakeDetector (no torch)")
    ap.add_argument("--fake-delay", type=float, default=0.0, help="FakeDetector per-frame delay (s)")
    args = ap.parse_args()

    config_path = args.config if os.path.exists(args.config) else None
    try:
        cfg = apply_overrides(load_config(config_path), args)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.logging.level)
    log = logging.getLogger("headless")

    if args.fake_detector:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))
        from fakes import FakeDetector  # type: ignore[import-not-found]

        detector: Any = FakeDetector(delay_s=args.fake_delay, threshold=cfg.model.score_threshold)
    else:
        from app.detector import RTDetrDetector

        detector = RTDetrDetector(cfg.model)

    stop = threading.Event()
    frames: LatestValue = LatestValue()
    results: LatestValue = LatestValue()
    meters = PipelineMeters()

    def on_signal(signum: int, _frame: Any) -> None:
        log.info("signal %d: stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    camera = CameraStream(cfg.camera, frames, stop, meters.capture_fps)
    try:
        neg = camera.open()
    except CaptureError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    camera.start()
    version, first = frames.wait_newer(0, timeout=cfg.camera.first_frame_timeout_s)
    if first is None:
        print("ERROR: no first frame", file=sys.stderr)
        stop.set()
        camera.join(2.0)
        return 1
    h, w = first.image.shape[:2]
    log.info("source %s negotiated %s", cfg.camera.source, neg)

    # Warmup runs inside the inference thread (CUDA/cuDNN handles are per thread).
    worker = InferenceWorker(detector, frames, results, stop, cfg.render, cfg.server, meters,
                             {"device": str(detector.device), "model": detector.model_name},
                             stats_interval_s=args.interval, warmup_shape=(h, w))

    # Collect every published result for whole-run statistics.
    e2e_all: list[float] = []
    inf_all: list[float] = []
    ndet_all: list[int] = []

    def collector() -> None:
        last = 0
        while not stop.is_set():
            v, res = results.wait_newer(last, timeout=0.5)
            if res is None or v <= last:
                continue
            last = v
            e2e_all.append((res.publish_ts - res.capture_ts) * 1000.0)
            inf_all.append(res.inference_ms)
            ndet_all.append(len(res.detections))

    col = threading.Thread(target=collector, name="collector", daemon=True)
    t0 = time.monotonic()
    worker.start()
    while not worker.ready.wait(0.25):
        if stop.is_set():
            break
    log.info("warmup done in %.1f s; measuring for %.0f s", time.monotonic() - t0, args.seconds)
    col.start()
    start = time.monotonic()
    cap_count0 = meters.capture_fps.count
    lines: list[dict] = []
    try:
        next_print = start + args.interval
        while not stop.is_set():
            now = time.monotonic()
            if now - start >= args.seconds:
                break
            if now >= next_print:
                next_print += args.interval
                row = {
                    "t": now - start,
                    "cap": meters.capture_fps.rate(),
                    "inf": meters.inference_fps.rate(),
                    "inf_mean": meters.inference_ms.mean(), "inf_p95": meters.inference_ms.p95(),
                    "e2e_mean": meters.e2e_ms.mean(), "e2e_p95": meters.e2e_ms.p95(),
                }
                lines.append(row)
                print(f"[{row['t']:6.1f}s] capture {row['cap']:5.1f} fps | inference {row['inf']:5.1f} fps | "
                      f"infer {row['inf_mean']:6.1f}/{row['inf_p95']:6.1f} ms | "
                      f"e2e {row['e2e_mean']:6.1f}/{row['e2e_p95']:6.1f} ms (mean/p95)", flush=True)
            stop.wait(min(0.2, max(0.0, next_print - time.monotonic())))
    finally:
        elapsed = time.monotonic() - start
        cap_frames = meters.capture_fps.count - cap_count0
        stop.set()
        frames.close()
        results.close()
        worker.join(3.0)
        camera.join(3.0)
        col.join(1.0)

    n = len(e2e_all)
    print("=" * 78)
    print(f"summary: {elapsed:.1f} s, detector={detector.model_name} on {detector.device}, "
          f"source={cfg.camera.source}")
    print(f"  capture   : {cap_frames} frames, {cap_frames / elapsed:.2f} fps avg")
    print(f"  inference : {n} frames, {n / elapsed:.2f} fps avg, errors {worker.errors}")
    print(f"  infer ms  : mean {mean(inf_all):.1f}  p50 {pct(inf_all, 50):.1f}  p95 {pct(inf_all, 95):.1f}  "
          f"max {max(inf_all, default=0):.1f}")
    print(f"  e2e ms    : mean {mean(e2e_all):.1f}  p50 {pct(e2e_all, 50):.1f}  p95 {pct(e2e_all, 95):.1f}  "
          f"max {max(e2e_all, default=0):.1f}")
    print(f"  detections: mean {mean([float(x) for x in ndet_all]):.2f} per frame")
    if len(lines) >= 2:
        first_q = lines[: max(1, len(lines) // 4)]
        last_q = lines[-max(1, len(lines) // 4):]
        a = mean([r["e2e_mean"] for r in first_q])
        b = mean([r["e2e_mean"] for r in last_q])
        print(f"  e2e drift : first-quarter mean {a:.1f} ms -> last-quarter mean {b:.1f} ms "
              f"({'bounded' if b < a * 1.5 + 20 else 'GROWING'})")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
