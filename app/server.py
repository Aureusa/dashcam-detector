"""Flask app factory: MJPEG stream, JSON APIs and the browser UI."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from app.buffers import LatestValue
from app.stats import PipelineMeters

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Classes selected by the UI's "driving preset" button (filtered to what the checkpoint knows).
DRIVING_PRESET = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]

HEALTH_MAX_AGE_S = 5.0

#: COCO vs. VOC-style spellings used by some checkpoints' id2label (e.g. PekingU: "motorbike", "sofa").
LABEL_ALIASES: tuple[tuple[str, str], ...] = (
    ("motorcycle", "motorbike"),
    ("airplane", "aeroplane"),
    ("couch", "sofa"),
    ("potted plant", "pottedplant"),
    ("dining table", "diningtable"),
    ("tv", "tvmonitor"),
)


class ClientCounter:
    """Thread-safe counter of connected /video_feed clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0
        self._total = 0

    def inc(self) -> int:
        with self._lock:
            self._n += 1
            self._total += 1
            return self._n

    def dec(self) -> int:
        with self._lock:
            self._n = max(0, self._n - 1)
            return self._n

    @property
    def value(self) -> int:
        with self._lock:
            return self._n

    @property
    def total(self) -> int:
        with self._lock:
            return self._total


@dataclass
class AppState:
    """Everything the web layer needs (no module-level globals).

    Attributes:
        config: The :class:`app.config.AppConfig`.
        frames: Capture -> inference buffer (``FramePacket``).
        results: Inference -> server buffer (``ResultPacket``).
        meters: Shared meters.
        detector: Detector handle (threshold / class filter / labels / device / model_name).
        stop: Shared shutdown event.
        camera: ``CameraStream`` (or anything with ``negotiated`` / ``status()``), optional.
        worker: ``InferenceWorker``, optional.
        start_time: ``time.monotonic()`` at startup.
    """

    config: Any
    frames: LatestValue
    results: LatestValue
    meters: PipelineMeters
    detector: Any
    stop: threading.Event
    camera: Any = None
    worker: Any = None
    start_time: float = field(default_factory=time.monotonic)
    clients: ClientCounter = field(default_factory=ClientCounter)
    settings_lock: threading.Lock = field(default_factory=threading.Lock)


class SettingsError(ValueError):
    """Invalid /api/settings payload."""


def _label_list(detector: Any) -> list[str]:
    labels = detector.labels or {}
    return [labels[k] for k in sorted(labels)]


def label_lookup(detector: Any) -> dict[str, str]:
    """Lower-case name (incl. COCO/VOC aliases) -> canonical checkpoint label."""
    canon = {name.strip().lower(): name for name in _label_list(detector)}
    for a, b in LABEL_ALIASES:
        if a in canon and b not in canon:
            canon[b] = canon[a]
        elif b in canon and a not in canon:
            canon[a] = canon[b]
    return canon


def validate_settings(payload: Any, detector: Any) -> dict:
    """Validate a settings payload fully (nothing is applied here).

    Returns:
        Normalized dict with optional ``score_threshold`` (float) and ``classes`` (canonical names).

    Raises:
        SettingsError: with a user-facing message.
    """
    if not isinstance(payload, dict):
        raise SettingsError("request body must be a JSON object")
    allowed = {"score_threshold", "classes"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SettingsError(f"unknown field(s): {', '.join(unknown)}; allowed: score_threshold, classes")
    if not payload:
        raise SettingsError("nothing to update: provide score_threshold and/or classes")
    out: dict = {}
    if "score_threshold" in payload:
        t = payload["score_threshold"]
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise SettingsError("score_threshold must be a number")
        t = float(t)
        if not math.isfinite(t) or not 0.0 <= t <= 1.0:
            raise SettingsError("score_threshold must be within [0, 1]")
        out["score_threshold"] = t
    if "classes" in payload:
        classes = payload["classes"]
        if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
            raise SettingsError("classes must be a list of strings ([] = all classes)")
        canon = label_lookup(detector)
        bad = [c for c in classes if c.strip().lower() not in canon]
        if bad:
            raise SettingsError(f"unknown class name(s): {', '.join(repr(b) for b in bad)}")
        seen: list[str] = []
        for c in classes:
            name = canon[c.strip().lower()]
            if name not in seen:
                seen.append(name)
        out["classes"] = seen
    return out


def _age(ts: Optional[float], now: float) -> Optional[float]:
    return None if ts is None else round(now - ts, 3)


def create_app(state: AppState) -> Flask:
    """Build the Flask app bound to ``state``."""
    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, "templates"),
        static_folder=os.path.join(PROJECT_ROOT, "static"),
    )
    app.config["JSON_SORT_KEYS"] = False
    app.json.sort_keys = False  # type: ignore[attr-defined]
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    scfg = state.config.server

    def current_settings() -> dict:
        return {"score_threshold": float(state.detector.threshold),
                "classes": list(state.detector.class_filter)}

    def capture_info() -> dict:
        neg = dict(getattr(state.camera, "negotiated", {}) or {}) if state.camera is not None else {}
        if not neg.get("width"):
            _, fpkt = state.frames.get()
            if fpkt is not None:
                neg["height"], neg["width"] = fpkt.image.shape[:2]
        return neg

    # ------------------------------------------------------------------ UI

    @app.get("/")
    def index() -> str:
        return render_template("index.html", model=getattr(state.detector, "model_name", "?"))

    # ------------------------------------------------------------ streaming

    @app.get("/video_feed")
    def video_feed() -> Response:
        client = request.remote_addr or "?"
        min_dt = 1.0 / max(1, int(scfg.max_stream_fps))

        def mjpeg_stream() -> Iterator[bytes]:
            n = state.clients.inc()
            log.info("stream client connected: %s (%d active)", client, n)
            sent = 0
            try:
                last = 0  # version 0 = empty buffer; waiting on -1 would busy-spin until the first result
                next_t = 0.0
                while not state.stop.is_set():
                    version, pkt = state.results.wait_newer(last, timeout=1.0)
                    if pkt is None or version == last:
                        if state.results.closed:
                            break
                        continue
                    last = version
                    now = time.monotonic()
                    if now < next_t:
                        continue
                    # Schedule-based limiter: averages to max_stream_fps, but tolerates up to half a
                    # period of jitter so a 30 fps pipeline is not halved by a 30 fps cap.
                    next_t = max(next_t + min_dt, now + 0.5 * min_dt)
                    sent += 1
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(pkt.jpeg)).encode() + b"\r\n\r\n" + pkt.jpeg + b"\r\n")
            except GeneratorExit:
                log.debug("stream generator closed for %s", client)
                raise
            finally:
                left = state.clients.dec()
                log.info("stream client disconnected: %s after %d frames (%d active)", client, sent, left)

        resp = Response(stream_with_context(mjpeg_stream()),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # ------------------------------------------------------------------ API

    @app.get("/api/stats")
    def api_stats() -> Response:
        m = state.meters
        now = time.monotonic()
        _, res = state.results.get()
        cap = capture_info()
        return jsonify({
            "capture_fps": round(m.capture_fps.rate(), 2),
            "inference_fps": round(m.inference_fps.rate(), 2),
            "inference_ms": m.inference_ms.snapshot(),
            "e2e_latency_ms": m.e2e_ms.snapshot(),
            "frame_id": res.frame_id if res is not None else None,
            "num_detections": len(res.detections) if res is not None else 0,
            "device": str(getattr(state.detector, "device", "?")),
            "model": getattr(state.detector, "model_name", "?"),
            "capture": {
                "width": cap.get("width"), "height": cap.get("height"), "fps": cap.get("fps"),
                "fourcc": cap.get("fourcc"), "backend": cap.get("backend"), "source": cap.get("source"),
            },
            "capture_resolution": (f"{cap['width']}x{cap['height']}"
                                   if cap.get("width") and cap.get("height") else None),
            "stream_resolution": f"{res.width}x{res.height}" if res is not None else None,
            "uptime_s": round(now - state.start_time, 1),
            "threshold": float(state.detector.threshold),
            "classes": list(state.detector.class_filter),
            "stream_clients": state.clients.value,
            "timings_ms": dict(getattr(state.detector, "last_timings", {}) or {}),
        })

    @app.get("/api/detections")
    def api_detections() -> Response:
        _, res = state.results.get()
        if res is None:
            return jsonify({"frame_id": None, "width": None, "height": None, "detections": []})
        return jsonify({
            "frame_id": res.frame_id,
            "width": res.width,
            "height": res.height,
            "age_ms": round((time.monotonic() - res.capture_ts) * 1000.0, 1),
            "detections": [d.to_dict() for d in res.detections],
        })

    @app.get("/api/labels")
    def api_labels() -> Response:
        labels = _label_list(state.detector)
        lookup = label_lookup(state.detector)
        preset: list[str] = []
        for c in DRIVING_PRESET:
            if c in lookup and lookup[c] not in preset:
                preset.append(lookup[c])
        return jsonify({
            "labels": labels,
            "id2label": {str(k): v for k, v in sorted((state.detector.labels or {}).items())},
            "driving_preset": preset,
        })

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings() -> tuple[Response, int] | Response:
        if request.method == "GET":
            return jsonify(current_settings())
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({"error": "request body must be valid JSON (Content-Type: application/json)"}), 400
        with state.settings_lock:
            try:
                upd = validate_settings(payload, state.detector)
            except SettingsError as exc:
                return jsonify({"error": str(exc)}), 400
            old = current_settings()
            try:
                if "classes" in upd:
                    state.detector.set_class_filter(upd["classes"])
                if "score_threshold" in upd:
                    state.detector.set_threshold(upd["score_threshold"])
            except ValueError as exc:  # should not happen after validation; roll back
                state.detector.set_class_filter(old["classes"])
                state.detector.set_threshold(old["score_threshold"])
                return jsonify({"error": str(exc)}), 400
            new = current_settings()
        log.info("settings updated: %s", new)
        return jsonify(new)

    @app.get("/healthz")
    def healthz() -> tuple[Response, int]:
        now = time.monotonic()
        cam_ts = getattr(state.camera, "last_frame_ts", None) if state.camera is not None else None
        if cam_ts is None:
            _, fpkt = state.frames.get()
            cam_ts = fpkt.capture_ts if fpkt is not None else None
        _, res = state.results.get()
        inf_ts = res.publish_ts if res is not None else None
        cap_age, inf_age = _age(cam_ts, now), _age(inf_ts, now)
        cap_ok = cap_age is not None and cap_age <= HEALTH_MAX_AGE_S
        inf_ok = inf_age is not None and inf_age <= HEALTH_MAX_AGE_S
        ok = cap_ok and inf_ok and not state.stop.is_set()
        body: dict = {
            "status": "ok" if ok else "unhealthy",
            "capture": {"ok": cap_ok, "last_frame_age_s": cap_age},
            "inference": {"ok": inf_ok, "last_result_age_s": inf_age},
            "stopping": state.stop.is_set(),
            "uptime_s": round(now - state.start_time, 1),
        }
        if state.camera is not None and hasattr(state.camera, "status"):
            body["capture"].update({k: v for k, v in state.camera.status().items()
                                    if k in ("connected", "reconnects", "last_error", "thread_alive")})
        if state.worker is not None:
            body["inference"].update({"thread_alive": bool(getattr(state.worker, "is_alive", False)),
                                      "errors": getattr(state.worker, "errors", 0)})
        return jsonify(body), (200 if ok else 503)

    @app.errorhandler(404)
    def not_found(_e: Exception) -> tuple[Response, int]:
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(405)
    def not_allowed(_e: Exception) -> tuple[Response, int]:
        return jsonify({"error": "method not allowed"}), 405

    return app
