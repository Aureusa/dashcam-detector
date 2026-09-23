"""Entrypoint: config -> detector -> capture -> warmup -> inference worker -> Flask -> shutdown.

Usage::

    python -m app.main --config config.yaml [--source ...] [--checkpoint ...] [--device ...]
                       [--host ...] [--port ...] [--threshold ...] [--log-level ...]
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time
from types import FrameType
from typing import Any, Optional

from app.buffers import LatestValue
from app.capture import CameraStream, CaptureError, check_device_access, classify_source, device_path
from app.config import ConfigError, add_cli_args, apply_overrides, load_config, setup_logging, validate_config
from app.pipeline import InferenceWorker
from app.server import AppState, create_app
from app.stats import PipelineMeters

log = logging.getLogger("app.main")

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_CONFIG = 2


def _err(msg: str) -> None:
    """Print a user-facing error to stderr (used before logging is configured)."""
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.stderr.flush()


def preflight_source(cam_cfg: Any) -> None:
    """Cheap checks that fail fast (before loading the model) for obviously bad sources."""
    kind = classify_source(cam_cfg.source)
    if kind == "device":
        path = device_path(cam_cfg.source)
        if path:
            check_device_access(path)
    elif kind == "file" and not os.path.isfile(str(cam_cfg.source)):
        raise CaptureError(f"Video file not found: {cam_cfg.source!r} (cwd: {os.getcwd()}).\n"
                           "Pass an existing file, a device (/dev/video0) or a URL via --source.")


def lan_ips() -> list[str]:
    """Best-effort list of non-loopback IPv4 addresses of this host."""
    ips: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET, no packets are sent for UDP connect
            ips.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(str(info[4][0]))
    except OSError:
        pass
    out: list[str] = []
    for ip in ips:
        if not ip.startswith("127.") and ip not in out:
            out.append(ip)
    return out


def log_urls(host: str, port: int) -> None:
    """Log where to open the UI (+ exposure warning for 0.0.0.0)."""
    if host in ("0.0.0.0", "::", ""):
        urls = [f"http://127.0.0.1:{port}"] + [f"http://{ip}:{port}" for ip in lan_ips()]
        log.info("open the UI at: %s", "  ".join(urls))
        log.warning("server bound to %s: the camera feed is exposed to the whole network "
                    "(no authentication). Use --host 127.0.0.1 for local-only access.", host)
    else:
        log.info("open the UI at: http://%s:%d", host, port)


def build_detector(cfg: Any) -> Any:
    """Import and construct the real RT-DETR detector (lazy: torch is only imported here)."""
    from app.detector import RTDetrDetector  # noqa: WPS433 - intentional lazy import

    return RTDetrDetector(cfg.model)


def _release_cuda() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            log.info("CUDA memory released")
    except Exception as exc:  # pragma: no cover
        log.warning("could not release CUDA memory: %s", exc)


def wait_first_frame(frames: LatestValue, camera: CameraStream, stop: threading.Event,
                     timeout_s: float) -> Optional[Any]:
    """Wait for the first captured frame; returns the FramePacket or None on timeout/stop."""
    deadline = time.monotonic() + timeout_s
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        version, pkt = frames.wait_newer(0, timeout=min(0.25, remaining))
        if pkt is not None and version > 0:
            return pkt
    return None


def first_frame_hints(camera: CameraStream, source: Any, timeout_s: float) -> str:
    """User-facing message when no frame arrived in time."""
    lines = [f"No frame received from {source!r} within {timeout_s:.0f} s."]
    if camera.last_error:
        lines.append(f"Last error: {camera.last_error}")
    lines += [
        "Troubleshooting:",
        "  * Is the camera plugged in and in webcam/UVC mode?  v4l2-ctl --list-devices",
        "  * Supported formats:  v4l2-ctl -d /dev/video0 --list-formats-ext  (try camera.fourcc MJPG "
        "or \"\", a listed resolution/fps)",
        "  * Some cameras expose 2 nodes; the 2nd is metadata-only: try /dev/video0 vs /dev/video2",
        "  * Close other apps using the camera:  fuser -v /dev/video*",
        "  * Test outside the app:  .venv/bin/python scripts/probe_camera.py --source <src> --seconds 5",
        "  * Develop with a clip instead:  --source samples/clip.mp4",
    ]
    return "\n".join(lines)


def run(argv: Optional[list[str]] = None) -> int:
    """Run the app; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="python -m app.main",
                                     description="Real-time dashcam object detection (RT-DETR + Flask).")
    add_cli_args(parser)
    args = parser.parse_args(argv)

    # 1. config
    try:
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args)
        validate_config(cfg)
    except (ConfigError, OSError, ValueError) as exc:
        _err(f"invalid configuration ({getattr(args, 'config', None)}): {exc}")
        return EXIT_CONFIG
    setup_logging(cfg.logging.level)
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        # The UI polls several endpoints per second; keep werkzeug's per-request lines out of the log.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
    log.info("starting: source=%r checkpoint=%s device=%s", cfg.camera.source, cfg.model.checkpoint,
             cfg.model.device)

    stop = threading.Event()
    frames: LatestValue = LatestValue()
    results: LatestValue = LatestValue()
    meters = PipelineMeters()
    server_holder: dict[str, Any] = {}

    def on_signal(signum: int, _frame: Optional[FrameType]) -> None:
        name = signal.Signals(signum).name
        if stop.is_set():
            log.warning("%s received again: forcing exit", name)
            os._exit(130)
        log.info("%s received: shutting down", name)
        stop.set()
        frames.close()
        results.close()
        srv = server_holder.get("srv")
        if srv is not None:
            # shutdown() blocks until serve_forever() returns, so call it from another thread.
            threading.Thread(target=srv.shutdown, name="server-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    camera: Optional[CameraStream] = None
    worker: Optional[InferenceWorker] = None
    exit_code = EXIT_OK
    try:
        # fail fast on obviously bad sources before the (slow) model load
        try:
            preflight_source(cfg.camera)
        except CaptureError as exc:
            log.error("cannot open video source:\n%s", exc)
            return EXIT_RUNTIME

        # 2. detector
        try:
            detector = build_detector(cfg)
        except Exception as exc:
            log.error("failed to load detector %r on device %r: %s\n"
                      "Hints: check internet access for the first download (Hugging Face cache: "
                      "~/.cache/huggingface), the checkpoint name, and --device cpu if CUDA is broken.",
                      cfg.model.checkpoint, cfg.model.device, exc)
            log.debug("detector load traceback", exc_info=True)
            return EXIT_RUNTIME
        log.info("detector ready: model=%s device=%s threshold=%.2f classes=%s",
                 getattr(detector, "model_name", "?"), getattr(detector, "device", "?"),
                 detector.threshold, detector.class_filter or "all")
        if stop.is_set():
            return EXIT_OK

        # 3. capture + first frame
        camera = CameraStream(cfg.camera, frames, stop, meters.capture_fps)
        try:
            neg = camera.open()
        except CaptureError as exc:
            log.error("cannot open video source:\n%s", exc)
            return EXIT_RUNTIME
        camera.start()
        timeout_s = float(getattr(cfg.camera, "first_frame_timeout_s", 10.0))
        first = wait_first_frame(frames, camera, stop, timeout_s)
        if first is None:
            if stop.is_set():
                return EXIT_OK
            log.error(first_frame_hints(camera, cfg.camera.source, timeout_s))
            return EXIT_RUNTIME
        h, w = first.image.shape[:2]
        log.info("first frame received: %dx%d (negotiated %s)", w, h, neg)

        # 4+5. inference worker; warmup runs inside the inference thread (CUDA handles are per thread)
        hud_info = {"device": str(getattr(detector, "device", "?")),
                    "model": getattr(detector, "model_name", "?")}
        worker = InferenceWorker(detector, frames, results, stop, cfg.render, cfg.server, meters, hud_info,
                                 warmup_shape=(h, w))
        t0 = time.monotonic()
        worker.start()
        log.info("warming up detector (%d iters at %dx%d)...", getattr(cfg.model, "warmup_iters", 0), w, h)
        while not worker.ready.wait(0.25):
            if stop.is_set():
                return EXIT_OK
        if stop.is_set():
            return EXIT_OK
        log.info("pipeline ready (warmup %.1f s)", time.monotonic() - t0)

        # 6. web server in the main thread
        state = AppState(config=cfg, frames=frames, results=results, meters=meters, detector=detector,
                         stop=stop, camera=camera, worker=worker)
        app = create_app(state)
        from werkzeug.serving import make_server

        try:
            srv = make_server(cfg.server.host, int(cfg.server.port), app, threaded=True)
        except (OSError, SystemExit) as exc:
            log.error("cannot start web server on %s:%s: %s (port in use? try --port)",
                      cfg.server.host, cfg.server.port, exc)
            return EXIT_RUNTIME
        srv.daemon_threads = True  # never block exit on open /video_feed connections
        server_holder["srv"] = srv
        if stop.is_set():  # signal arrived while starting
            return EXIT_OK
        log_urls(cfg.server.host, int(cfg.server.port))
        srv.serve_forever(poll_interval=0.2)
        log.info("web server stopped")
    except KeyboardInterrupt:  # pragma: no cover - handler normally catches SIGINT
        log.info("interrupted")
    except Exception:
        log.exception("fatal error")
        exit_code = EXIT_RUNTIME
    finally:
        stop.set()
        frames.close()
        results.close()
        if worker is not None:
            worker.join(timeout=2.0)
        if camera is not None:
            camera.join(timeout=2.0)
            camera.release()
        srv = server_holder.get("srv")
        if srv is not None:
            try:
                srv.server_close()
            except Exception:  # pragma: no cover
                pass
        _release_cuda()
        log.info("shutdown complete")
        logging.shutdown()
    return exit_code


def main() -> None:
    """Console entrypoint."""
    sys.exit(run())


if __name__ == "__main__":
    main()
