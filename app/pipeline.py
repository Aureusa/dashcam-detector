"""Inference stage: newest frame -> detect -> draw -> JPEG encode (once) -> publish."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import cv2

from app.buffers import LatestValue
from app.capture import FramePacket
from app.render import render_frame
from app.stats import PipelineMeters

if TYPE_CHECKING:  # pragma: no cover
    from app.config import RenderConfig, ServerConfig
    from app.detector import Detection

log = logging.getLogger(__name__)


@dataclass
class ResultPacket:
    """One processed frame, ready to be served."""

    frame_id: int
    capture_ts: float
    publish_ts: float
    inference_ms: float
    detections: list["Detection"]
    jpeg: bytes
    width: int
    height: int


@dataclass
class _ErrorLimiter:
    """Rate-limits logging of repeated identical exceptions."""

    interval_s: float = 10.0
    last_key: Optional[str] = None
    last_logged: float = 0.0
    suppressed: int = 0
    total: int = 0

    def report(self, exc: BaseException) -> None:
        self.total += 1
        key = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        if key != self.last_key or now - self.last_logged >= self.interval_s:
            if self.suppressed:
                log.error("previous error repeated %d more times", self.suppressed)
            log.exception("inference iteration failed: %s", key)
            self.last_key = key
            self.last_logged = now
            self.suppressed = 0
        else:
            self.suppressed += 1


@dataclass
class _SlowWarn:
    """Tracks whether inference is sustainedly slower than capture."""

    ratio: float = 0.8
    intervals_needed: int = 3
    count: int = 0
    warned: bool = False


class InferenceWorker:
    """Inference thread consuming the newest :class:`FramePacket` and publishing :class:`ResultPacket`.

    Args:
        detector: Object implementing ``detect(frame_bgr) -> list[Detection]`` (RTDetrDetector or fake).
        frames: Input buffer written by the capture thread.
        results: Output buffer read by the web server.
        stop: Shared shutdown event.
        render_cfg: Rendering options.
        server_cfg: Server options (``jpeg_quality`` is used).
        meters: Shared meters (the capture meter is read for the HUD / stats line).
        hud_info: Static HUD values, e.g. ``{"device": "cuda:0", "model": "rtdetr_v2_r18vd"}``.
        stats_interval_s: Period of the one-line stats log.
        warmup_shape: ``(height, width)``: if given, ``detector.warmup(h, w)`` is run *inside the
            inference thread* before the first frame (CUDA/cuDNN handles are per thread, so a
            main-thread warmup does not cover the first ~0.5 s detect() call here).
            :attr:`ready` is set once warmup is done (immediately if no warmup).
    """

    def __init__(self, detector: Any, frames: LatestValue[FramePacket], results: LatestValue[ResultPacket],
                 stop: threading.Event, render_cfg: "RenderConfig", server_cfg: "ServerConfig",
                 meters: PipelineMeters, hud_info: Optional[dict] = None,
                 stats_interval_s: float = 5.0,
                 warmup_shape: Optional[tuple[int, int]] = None) -> None:
        self.detector = detector
        self.frames = frames
        self.results = results
        self.stop = stop
        self.render_cfg = render_cfg
        self.jpeg_quality = int(getattr(server_cfg, "jpeg_quality", 80))
        self.meters = meters
        self.hud_info = dict(hud_info or {})
        self.stats_interval_s = stats_interval_s
        self._thread: Optional[threading.Thread] = None
        self._errors = _ErrorLimiter()
        self._slow = _SlowWarn()
        self._processed = 0
        self._last_publish_ts: Optional[float] = None
        self._skipped_frames = 0
        self.warmup_shape = warmup_shape
        self.ready = threading.Event()
        self.warmup_s: Optional[float] = None

    # -- public API -------------------------------------------------------

    def start(self) -> None:
        """Spawn the daemon inference thread."""
        if self._thread is not None:
            raise RuntimeError("InferenceWorker already started")
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the inference thread to exit."""
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("inference thread did not exit within %.1fs", timeout or 0.0)

    @property
    def is_alive(self) -> bool:
        """Whether the inference thread is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def processed(self) -> int:
        """Number of frames published."""
        return self._processed

    @property
    def errors(self) -> int:
        """Number of failed iterations."""
        return self._errors.total

    @property
    def last_publish_ts(self) -> Optional[float]:
        """Monotonic time of the last published result."""
        return self._last_publish_ts

    def hud_stats(self, num_detections: int) -> dict:
        """Current stats snapshot used for the HUD overlay."""
        m = self.meters
        return {
            "capture_fps": m.capture_fps.rate(),
            "inference_fps": m.inference_fps.rate(),
            "inference_ms": m.inference_ms.mean(),
            "e2e_ms": m.e2e_ms.mean(),
            "num_detections": num_detections,
            "device": self.hud_info.get("device", "-"),
            "model": self.hud_info.get("model", "-"),
        }

    # -- internals --------------------------------------------------------

    def process(self, pkt: FramePacket) -> ResultPacket:
        """Run one full iteration on ``pkt`` and publish the result (also used by tests)."""
        t0 = time.perf_counter()
        dets = self.detector.detect(pkt.image)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        hud = self.hud_stats(len(dets)) if self.render_cfg.draw_hud else None
        img = render_frame(pkt.image, dets, self.render_cfg, hud)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        publish_ts = time.monotonic()
        res = ResultPacket(
            frame_id=pkt.frame_id,
            capture_ts=pkt.capture_ts,
            publish_ts=publish_ts,
            inference_ms=inference_ms,
            detections=list(dets),
            jpeg=buf.tobytes(),
            width=int(img.shape[1]),
            height=int(img.shape[0]),
        )
        self.results.put(res)
        self.meters.inference_fps.tick()
        self.meters.inference_ms.add(inference_ms)
        self.meters.e2e_ms.add((publish_ts - pkt.capture_ts) * 1000.0)
        self._processed += 1
        self._last_publish_ts = publish_ts
        return res

    def _log_stats(self) -> None:
        m = self.meters
        cap = m.capture_fps.rate()
        inf = m.inference_fps.rate()
        log.info(
            "stats: capture %.1f fps | inference %.1f fps | infer %.1f/%.1f ms (mean/p95) | "
            "e2e %.1f/%.1f ms | processed %d | dropped %d | errors %d",
            cap, inf, m.inference_ms.mean(), m.inference_ms.p95(), m.e2e_ms.mean(), m.e2e_ms.p95(),
            self._processed, self._skipped_frames, self._errors.total,
        )
        slow = self._slow
        if cap > 1.0 and inf < slow.ratio * cap:
            slow.count += 1
            if slow.count >= slow.intervals_needed and not slow.warned:
                slow.warned = True
                log.warning(
                    "inference (%.1f fps) has been well below capture (%.1f fps) for ~%ds: frames are "
                    "being dropped. Consider a smaller checkpoint, fp16, output_width, or a GPU.",
                    inf, cap, int(slow.count * self.stats_interval_s))
        else:
            if slow.warned:
                log.info("inference has caught up with capture (%.1f / %.1f fps)", inf, cap)
            slow.count = 0
            slow.warned = False

    def _run(self) -> None:
        log.info("inference worker started (jpeg_quality=%d)", self.jpeg_quality)
        last_version = 0
        try:
            if self.warmup_shape is not None and not self.stop.is_set():
                h, w = self.warmup_shape
                t0 = time.monotonic()
                try:
                    self.detector.warmup(int(h), int(w))
                    self.warmup_s = time.monotonic() - t0
                    log.info("warmup (%dx%d) done in inference thread in %.2f s", w, h, self.warmup_s)
                except Exception as exc:  # a failed warmup must not prevent serving
                    self._errors.report(exc)
            self.ready.set()
            # Drop frames captured during warmup: start from the newest one.
            last_version = max(0, self.frames.version - 1)
            next_stats = time.monotonic() + self.stats_interval_s
            while not self.stop.is_set():
                now = time.monotonic()
                if now >= next_stats:
                    next_stats = now + self.stats_interval_s
                    self._log_stats()
                version, pkt = self.frames.wait_newer(last_version, timeout=1.0)
                if pkt is None or version <= last_version:
                    continue
                if self.stop.is_set():
                    break
                if last_version and version > last_version + 1:
                    self._skipped_frames += version - last_version - 1
                last_version = version
                try:
                    self.process(pkt)
                except Exception as exc:  # keep the worker alive (bad frame, CUDA OOM, ...)
                    self._errors.report(exc)
                    self.stop.wait(0.01)
        finally:
            self.ready.set()
            log.info("inference worker stopped (processed %d frames, %d errors)",
                     self._processed, self._errors.total)
