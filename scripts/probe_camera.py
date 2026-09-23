#!/usr/bin/env python3
"""Probe a video source: print negotiated properties, measure delivered FPS, save frames, record a clip.

Examples::

    .venv/bin/python scripts/probe_camera.py --source /dev/video0 --seconds 10 --save-frames 5
    .venv/bin/python scripts/probe_camera.py --source /dev/video0 --seconds 30 --record samples/live_clip.mp4
    .venv/bin/python scripts/probe_camera.py --source samples/clip.mp4 --seconds 10 --save-frames 5
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from typing import Any, Optional  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.capture import CaptureError, _as_bgr, classify_source, open_capture  # noqa: E402
from app.config import CameraConfig  # noqa: E402


class Recorder:
    """Writes BGR frames to a video file (ffmpeg/libx264 if available, else OpenCV mp4v)."""

    def __init__(self, path: str, width: int, height: int, fps: float, wallclock: bool = False) -> None:
        """``wallclock=True`` (live sources): timestamp frames on arrival, so the clip keeps the real
        delivered frame rate even if the camera does not deliver its advertised FPS."""
        self.path = path
        self.proc: Optional[subprocess.Popen] = None
        self.writer: Any = None
        self.frames = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            timing = (["-use_wallclock_as_timestamps", "1"] if wallclock else ["-r", f"{fps:.3f}"])
            cmd = [ffmpeg, "-loglevel", "error", "-y", *timing, "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{width}x{height}", "-i", "-",
                   "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   *(["-fps_mode", "passthrough"] if wallclock else []),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.kind = "ffmpeg/libx264"
        else:
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {path}")
            self.kind = "opencv/mp4v"
        self.size = (width, height)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=60)
        if self.writer is not None:
            self.writer.release()


def build_cfg(args: argparse.Namespace) -> CameraConfig:
    source: Any = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    base = CameraConfig()
    return dataclasses.replace(
        base,
        source=source,
        backend=args.backend or base.backend,
        width=args.width or base.width,
        height=args.height or base.height,
        fps=args.fps or base.fps,
        fourcc=base.fourcc if args.fourcc is None else args.fourcc,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/dev/video0", help="device index, /dev/videoN, file or URL")
    ap.add_argument("--seconds", type=float, default=10.0, help="measurement duration")
    ap.add_argument("--save-frames", type=int, default=0, metavar="N",
                    help="save N frames spread over the run to <out-dir>/frame_XX.jpg")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "samples"))
    ap.add_argument("--record", metavar="PATH", help="record the whole run to this video file (mp4)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--fourcc", help='4-char FOURCC (e.g. MJPG, YUYV); "" = do not set')
    ap.add_argument("--backend", choices=["auto", "v4l2", "ffmpeg", "gstreamer"])
    ap.add_argument("--no-pace", action="store_true",
                    help="for files: read as fast as possible instead of at native FPS")
    ap.add_argument("--first-frame-timeout", type=float, default=10.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = build_cfg(args)
    kind = classify_source(cfg.source)
    try:
        cap, neg = open_capture(cfg)
    except CaptureError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("negotiated properties:")
    for k, v in neg.items():
        print(f"  {k:12s} {v}")
    if kind == "device":
        print(f"  requested    {cfg.width}x{cfg.height} @ {cfg.fps} fps, fourcc={cfg.fourcc or '(unchanged)'}")

    target_fps = float(neg.get("fps") or cfg.fps)
    pace = kind == "file" and not args.no_pace and target_fps > 0
    period = 1.0 / target_fps if pace else 0.0
    save_n = max(0, args.save_frames)
    save_every = args.seconds / save_n if save_n else 0.0
    saved: list[str] = []
    recorder: Optional[Recorder] = None
    stamps: list[float] = []
    shape = None
    exit_code = 0

    try:
        start = time.monotonic()
        next_t = start
        next_save = start
        first_deadline = start + args.first_frame_timeout
        failures = 0
        while True:
            now = time.monotonic()
            if stamps and now - stamps[0] >= args.seconds:
                break
            if not stamps and now > first_deadline:
                print(f"ERROR: no frame received within {args.first_frame_timeout:.0f} s", file=sys.stderr)
                return 3
            if pace:
                if next_t > now:
                    time.sleep(next_t - now)
                next_t = max(next_t + period, time.monotonic() - period)
            ok, frame = cap.read()
            ts = time.monotonic()
            if not ok or frame is None:
                if kind == "file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                failures += 1
                if failures > 100:
                    print("ERROR: too many consecutive read failures", file=sys.stderr)
                    exit_code = 4
                    break
                time.sleep(0.005)
                continue
            failures = 0
            frame = _as_bgr(frame)
            stamps.append(ts)
            shape = frame.shape
            if args.record:
                if recorder is None:
                    recorder = Recorder(args.record, frame.shape[1], frame.shape[0], target_fps or 30.0,
                                        wallclock=kind != "file")
                recorder.write(frame)
            if save_n and len(saved) < save_n and ts >= next_save:
                os.makedirs(args.out_dir, exist_ok=True)
                path = os.path.join(args.out_dir, f"frame_{len(saved):02d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append(path)
                next_save = ts + save_every
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        cap.release()
        if recorder is not None:
            recorder.close()

    n = len(stamps)
    if n < 2:
        print("ERROR: fewer than 2 frames received", file=sys.stderr)
        return exit_code or 3
    elapsed = stamps[-1] - stamps[0]
    fps = (n - 1) / elapsed if elapsed > 0 else 0.0
    dts = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    print(f"frames: {n} in {elapsed:.2f} s, frame shape {shape}")
    print(f"measured FPS: {fps:.2f} (target {target_fps:.2f}, {100.0 * fps / target_fps:.1f}% of target)"
          if target_fps else f"measured FPS: {fps:.2f}")
    print(f"frame interval ms: mean {statistics.mean(dts):.1f}  median {statistics.median(dts):.1f}  "
          f"max {max(dts):.1f}  stdev {statistics.pstdev(dts):.1f}")
    if target_fps:
        within = abs(fps - target_fps) <= 0.1 * target_fps
        print("FPS within 10% of target: " + ("YES" if within else "NO"))
    for p in saved:
        print(f"saved {p}")
    if recorder is not None:
        print(f"recorded {recorder.frames} frames to {args.record} ({recorder.kind})")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
