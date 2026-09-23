"""Capture stage: reads frames from a camera / file / URL in a dedicated thread.

The capture thread drains the source as fast as it delivers frames and
publishes each one into a single-slot :class:`~app.buffers.LatestValue`,
so downstream stages always see the newest frame (stale frames are dropped).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import cv2
import numpy as np

from app.buffers import LatestValue
from app.stats import RateMeter

if TYPE_CHECKING:  # pragma: no cover
    from app.config import CameraConfig

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_DEV_RE = re.compile(r"^/dev/video(\d+)$")


class CaptureError(RuntimeError):
    """Raised when a source cannot be opened. The message is user-facing (includes hints)."""


@dataclass
class FramePacket:
    """One captured frame."""

    frame_id: int          # monotonically increasing
    capture_ts: float      # time.monotonic() right after read()
    image: np.ndarray      # HxWx3 uint8, BGR (OpenCV native)


# ---------------------------------------------------------------------------
# Source classification / opening helpers
# ---------------------------------------------------------------------------

def classify_source(source: Union[str, int]) -> str:
    """Return the source kind: ``"device"``, ``"url"``, ``"rtsp"``, ``"gstreamer"`` or ``"file"``."""
    if isinstance(source, int):
        return "device"
    s = str(source).strip()
    if s.isdigit() or _DEV_RE.match(s) or s.startswith("/dev/"):
        return "device"
    if s.lower().startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if _URL_RE.match(s):
        return "url"
    if "!" in s and ("appsink" in s or "src" in s):
        return "gstreamer"
    return "file"


def device_path(source: Union[str, int]) -> Optional[str]:
    """Map a device index / ``/dev/videoN`` source to its device node path (None for other kinds)."""
    if isinstance(source, int):
        return f"/dev/video{source}"
    s = str(source).strip()
    if s.isdigit():
        return f"/dev/video{int(s)}"
    if s.startswith("/dev/"):
        return s
    return None


def _device_hints(path: str) -> str:
    return (
        "Troubleshooting:\n"
        "  * List video devices:        v4l2-ctl --list-devices\n"
        f"  * Check formats of the node:  v4l2-ctl -d {path} --list-formats-ext\n"
        "    (some cameras expose two nodes; the second is often a metadata node with no formats)\n"
        "  * Make sure no other app (Cheese, VLC, ffplay, a browser tab, another instance) has the camera open:\n"
        f"      fuser -v {path}\n"
        "  * Is the dashcam in webcam/UVC mode (not USB mass-storage)? Check lsusb / the dashcam menu.\n"
        "  * Or use a recorded clip for testing:  --source samples/clip.mp4"
    )


def check_device_access(path: str) -> None:
    """Raise :class:`CaptureError` with a friendly message if ``path`` is missing or not accessible."""
    if not os.path.exists(path):
        raise CaptureError(f"Camera device {path} does not exist (camera unplugged or wrong index?).\n"
                           + _device_hints(path))
    if not os.access(path, os.R_OK | os.W_OK):
        raise CaptureError(
            f"Permission denied opening {path}: user '{_username()}' cannot read/write the device.\n"
            "Fix: add your user to the 'video' group and log out/in (or reboot):\n"
            "      sudo usermod -aG video $USER\n"
            "  then verify with:  groups | grep video   and   ls -l " + path
        )


def _username() -> str:
    try:
        import getpass

        return getpass.getuser()
    except Exception:  # pragma: no cover - extremely unusual environments
        return "?"


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars if chars.isprintable() else f"0x{code:08x}"


def _backend_flag(cfg_backend: str, kind: str) -> int:
    b = (cfg_backend or "auto").lower()
    if b == "v4l2":
        return cv2.CAP_V4L2
    if b == "ffmpeg":
        return cv2.CAP_FFMPEG
    if b == "gstreamer":
        return cv2.CAP_GSTREAMER
    # auto
    if kind == "device":
        return cv2.CAP_V4L2
    if kind == "gstreamer":
        return cv2.CAP_GSTREAMER
    return cv2.CAP_FFMPEG


def open_capture(cfg: "CameraConfig") -> tuple[Any, dict]:
    """Open the configured source and apply capture properties.

    Returns:
        ``(cap, negotiated)`` where ``cap`` is an opened ``cv2.VideoCapture`` and
        ``negotiated`` is a dict with the actual width/height/fps/fourcc/backend/source.

    Raises:
        CaptureError: with a user-facing message and troubleshooting hints.
    """
    source = cfg.source
    kind = classify_source(source)
    backend = _backend_flag(cfg.backend, kind)

    target: Union[str, int]
    if kind == "device":
        path = device_path(source)
        assert path is not None
        check_device_access(path)
        m = _DEV_RE.match(path)
        # Opening by index is the most portable form for V4L2.
        target = int(m.group(1)) if m and backend == cv2.CAP_V4L2 else path
    elif kind == "file":
        target = str(source)
        if not os.path.isfile(target):
            raise CaptureError(
                f"Video file not found: {target!r} (cwd: {os.getcwd()}).\n"
                "Pass an existing file, a device (/dev/video0) or a URL (rtsp://...) via --source."
            )
    else:
        target = str(source)
        if kind == "rtsp":
            # TCP avoids UDP packet-loss artifacts (smeared / grey frames).
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    log.info("opening source %r (kind=%s, backend=%s)", source, kind, _backend_name(backend))
    try:
        cap = cv2.VideoCapture(target, backend)
    except Exception as exc:  # cv2 can raise on bad backends
        raise CaptureError(f"OpenCV failed to open {source!r}: {exc}") from exc

    if not cap.isOpened():
        cap.release()
        msg = f"Could not open source {source!r} with backend {_backend_name(backend)}."
        if kind == "device":
            path = device_path(source) or str(source)
            msg += " The device exists and is accessible, so it is probably busy or not a capture node.\n"
            msg += _device_hints(path)
        elif kind in ("rtsp", "url"):
            msg += ("\nCheck the URL/credentials and network (try: ffplay " + str(source) + ").")
        else:
            msg += "\nThe file may be corrupt or use an unsupported codec (check with ffprobe)."
        raise CaptureError(msg)

    if kind == "device":
        # Property order matters on many UVC drivers: FOURCC first, then size, FPS, buffer size.
        if cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    if kind != "file":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (0.0 < fps < 1000.0):
        fps = 0.0
    negotiated = {
        "source": str(source),
        "kind": kind,
        "backend": _safe_backend_name(cap, backend),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": round(fps, 3),
        "fourcc": _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0),
    }
    if kind == "file":
        negotiated["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log.info("negotiated: %s", negotiated)
    if kind == "device":
        req = (cfg.width, cfg.height)
        got = (negotiated["width"], negotiated["height"])
        if req != got:
            log.warning("requested %dx%d but driver negotiated %dx%d", *req, *got)
        if fps and abs(fps - cfg.fps) > 0.5:
            log.warning("requested %s fps but driver negotiated %.2f fps", cfg.fps, fps)
        if cfg.fourcc and negotiated["fourcc"] and negotiated["fourcc"] != cfg.fourcc:
            log.warning("requested FOURCC %s but driver negotiated %s", cfg.fourcc, negotiated["fourcc"])
    return cap, negotiated


def _backend_name(flag: int) -> str:
    return {cv2.CAP_V4L2: "V4L2", cv2.CAP_FFMPEG: "FFMPEG", cv2.CAP_GSTREAMER: "GSTREAMER"}.get(flag, str(flag))


def _safe_backend_name(cap: Any, flag: int) -> str:
    try:
        return str(cap.getBackendName())
    except Exception:
        return _backend_name(flag)


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure an HxWx3 uint8 BGR frame."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# ---------------------------------------------------------------------------
# CameraStream
# ---------------------------------------------------------------------------

class CameraStream:
    """Capture thread publishing :class:`FramePacket` objects into a :class:`LatestValue`.

    Usage::

        cam = CameraStream(cfg, frames, stop, capture_meter)
        cam.open()      # optional: raises CaptureError synchronously with a friendly message
        cam.start()
        ...
        stop.set(); cam.join(2.0)
    """

    def __init__(self, cfg: "CameraConfig", out: LatestValue[FramePacket],
                 stop: threading.Event, stats: RateMeter) -> None:
        self.cfg = cfg
        self.out = out
        self.stop = stop
        self.stats = stats
        self._cap: Any = None
        self._negotiated: dict = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._last_frame_ts: Optional[float] = None
        self._connected = False
        self._last_error: Optional[str] = None
        self._reconnects = 0
        self._kind = classify_source(cfg.source)

    # -- public API -------------------------------------------------------

    def open(self) -> dict:
        """Open the source synchronously (idempotent). Raises :class:`CaptureError`."""
        with self._lock:
            if self._cap is None:
                cap, neg = open_capture(self.cfg)
                self._cap = cap
                self._negotiated = neg
                self._last_error = None
            return dict(self._negotiated)

    def start(self) -> None:
        """Spawn the daemon capture thread."""
        if self._thread is not None:
            raise RuntimeError("CameraStream already started")
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the capture thread to exit."""
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("capture thread did not exit within %.1fs", timeout or 0.0)

    def release(self) -> None:
        """Release the capture handle if the thread is not running (safe to call multiple times)."""
        if self._thread is not None and self._thread.is_alive():
            return  # the thread releases in its finally block
        self._release()

    @property
    def negotiated(self) -> dict:
        """Actual width/height/fps/fourcc/backend/source after open (empty before)."""
        with self._lock:
            return dict(self._negotiated)

    @property
    def last_frame_ts(self) -> Optional[float]:
        """``time.monotonic()`` of the last successfully read frame."""
        return self._last_frame_ts

    @property
    def connected(self) -> bool:
        """True while the source is open and delivering frames."""
        return self._connected

    @property
    def last_error(self) -> Optional[str]:
        """Most recent open/read error message (None if healthy)."""
        return self._last_error

    @property
    def reconnects(self) -> int:
        """Number of reconnects performed."""
        return self._reconnects

    @property
    def is_alive(self) -> bool:
        """Whether the capture thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        """Snapshot for /healthz and /api/stats."""
        age = None if self._last_frame_ts is None else round(time.monotonic() - self._last_frame_ts, 3)
        return {
            "connected": self._connected,
            "thread_alive": self.is_alive,
            "last_frame_age_s": age,
            "frames": self._frame_id,
            "reconnects": self._reconnects,
            "last_error": self._last_error,
        }

    # -- internals --------------------------------------------------------

    def _release(self) -> None:
        with self._lock:
            cap, self._cap = self._cap, None
        self._connected = False
        if cap is not None:
            try:
                cap.release()
                log.info("capture released")
            except Exception:  # pragma: no cover
                log.exception("error releasing capture")

    def _try_open(self) -> bool:
        try:
            self.open()
            return True
        except CaptureError as exc:
            self._last_error = str(exc).splitlines()[0]
            log.error("open failed: %s", self._last_error)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            self._last_error = repr(exc)
            log.exception("unexpected error opening source")
            return False

    def _reconnect(self) -> None:
        self._release()
        self._reconnects += 1
        log.warning("reconnecting to %r in %.1fs (reconnect #%d)",
                    self.cfg.source, self.cfg.reconnect_delay_s, self._reconnects)
        self.stop.wait(self.cfg.reconnect_delay_s)

    def _run(self) -> None:
        cfg = self.cfg
        is_file = self._kind == "file"
        failures = 0
        next_t: Optional[float] = None
        period = 0.0
        paced_cap: Any = None
        log.info("capture thread started")
        try:
            while not self.stop.is_set():
                if self._cap is None:
                    if not self._try_open():
                        self.stop.wait(cfg.reconnect_delay_s)
                        continue
                    failures = 0
                if self._cap is not paced_cap:
                    # new handle (first open or reopen): (re)compute file pacing
                    paced_cap = self._cap
                    next_t = None
                    fps = self._negotiated.get("fps") or 0.0
                    period = 0.0
                    if is_file and cfg.pace_file:
                        if fps <= 0:
                            log.warning("file reports no FPS; assuming %s for pacing", cfg.fps)
                            fps = float(cfg.fps)
                        period = 1.0 / fps

                cap = self._cap
                # Pace files at their native FPS so they behave like a live camera.
                if period > 0:
                    now = time.monotonic()
                    if next_t is None or now - next_t > period:
                        next_t = now  # (re)start schedule if we fell behind
                    elif next_t > now:
                        if self.stop.wait(next_t - now):
                            break
                    next_t += period

                try:
                    ok, frame = cap.read()
                except Exception as exc:  # pragma: no cover - cv2 rarely raises here
                    ok, frame = False, None
                    self._last_error = f"read raised {exc!r}"
                ts = time.monotonic()

                if ok and frame is not None and frame.size > 0:
                    failures = 0
                    self._frame_id += 1
                    self._last_frame_ts = ts
                    if not self._connected:
                        self._connected = True
                        self._last_error = None
                        log.info("receiving frames from %r (%sx%s)", cfg.source,
                                 frame.shape[1], frame.shape[0])
                    self.out.put(FramePacket(self._frame_id, ts, _as_bgr(frame)))
                    self.stats.tick()
                    continue

                # --- read failure ---
                if is_file:
                    if cfg.loop_file:
                        if cap.set(cv2.CAP_PROP_POS_FRAMES, 0) and failures == 0:
                            failures += 1  # one retry after seek; a second failure reopens
                            log.debug("end of file, looping")
                            continue
                        log.info("end of file, reopening to loop")
                        self._release()
                        failures = 0
                        self.stop.wait(0.05)
                        continue
                    log.info("end of file reached (loop_file=false); capture idle")
                    self._connected = False
                    self.stop.wait()
                    break

                failures += 1
                self._connected = False if failures >= 3 else self._connected
                if failures == 1 or failures % 10 == 0:
                    log.warning("frame read failed (%d consecutive)", failures)
                if failures >= cfg.max_consecutive_failures:
                    self._last_error = f"{failures} consecutive read failures"
                    self._reconnect()
                    failures = 0
                else:
                    self.stop.wait(0.01)
        except Exception:  # pragma: no cover - last-resort guard
            log.exception("capture thread crashed")
        finally:
            self._release()
            log.info("capture thread stopped")
