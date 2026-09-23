"""Tests for the Flask app using the test client, a fake detector and fake pipeline state."""

from __future__ import annotations

import json
import logging
import threading
import time

import pytest

from app.buffers import LatestValue
from app.capture import FramePacket
from app.config import AppConfig
from app.pipeline import ResultPacket
from app.server import AppState, create_app
from app.stats import PipelineMeters
from fakes import FakeDetection, FakeDetector, make_frame

JPEG = b"\xff\xd8\xff\xe0fakejpegdata\xff\xd9"


def publish(state: AppState, frame_id: int = 1, age_s: float = 0.0) -> None:
    now = time.monotonic() - age_s
    state.frames.put(FramePacket(frame_id, now, make_frame(720, 1280)))
    dets = [FakeDetection(10, 20, 110, 220, 0.9, 2, "car"), FakeDetection(0, 0, 5, 5, 0.6, 0, "person")]
    state.results.put(ResultPacket(frame_id, now, now + 0.01, 12.5, dets, JPEG, 1280, 720))
    state.meters.inference_ms.add(12.5)
    state.meters.e2e_ms.add(30.0)


@pytest.fixture
def state():
    s = AppState(config=AppConfig(), frames=LatestValue(), results=LatestValue(), meters=PipelineMeters(),
                 detector=FakeDetector(), stop=threading.Event())
    yield s
    s.stop.set()
    s.results.close()


@pytest.fixture
def client(state):
    app = create_app(state)
    app.testing = True
    return app.test_client()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'id="stream"' in body and "/video_feed" in body
    assert "app.js" in body and "style.css" in body
    assert "http://" not in body and "https://" not in body  # no CDNs
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_stats_before_any_frame(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    s = r.get_json()
    assert s["frame_id"] is None and s["num_detections"] == 0 and s["inference_fps"] == 0


def test_stats(client, state):
    publish(state, frame_id=5)
    s = client.get("/api/stats").get_json()
    for key in ("capture_fps", "inference_fps", "inference_ms", "e2e_latency_ms", "frame_id", "num_detections",
                "device", "model", "capture_resolution", "uptime_s", "threshold", "stream_clients"):
        assert key in s, key
    assert s["frame_id"] == 5 and s["num_detections"] == 2
    assert s["inference_ms"] == {"mean": 12.5, "p95": 12.5}
    assert s["e2e_latency_ms"]["mean"] == 30.0
    assert s["capture_resolution"] == "1280x720"
    assert s["model"] == "fake-detector" and s["device"] == "cpu"
    assert s["stream_clients"] == 0 and s["threshold"] == 0.5


def test_detections(client, state):
    assert client.get("/api/detections").get_json()["detections"] == []
    publish(state, frame_id=9)
    d = client.get("/api/detections").get_json()
    assert d["frame_id"] == 9
    assert d["detections"][0] == {"x1": 10, "y1": 20, "x2": 110, "y2": 220, "score": 0.9, "class_id": 2,
                                  "label": "car"}


def test_labels(client):
    d = client.get("/api/labels").get_json()
    assert len(d["labels"]) == 80 and d["labels"][0] == "person"
    assert d["driving_preset"] == ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light",
                                   "stop sign"]


def test_settings_get_and_valid_post(client, state):
    assert client.get("/api/settings").get_json() == {"score_threshold": 0.5, "classes": []}
    r = client.post("/api/settings", json={"score_threshold": 0.35, "classes": ["CAR", "traffic light", "car"]})
    assert r.status_code == 200, r.get_json()
    assert r.get_json() == {"score_threshold": 0.35, "classes": ["car", "traffic light"]}
    assert client.get("/api/settings").get_json() == r.get_json()
    assert state.detector.threshold == 0.35
    r = client.post("/api/settings", json={"score_threshold": 1})  # int accepted
    assert r.status_code == 200 and r.get_json()["score_threshold"] == 1.0
    r = client.post("/api/settings", json={"classes": []})
    assert r.status_code == 200 and r.get_json()["classes"] == []


@pytest.mark.parametrize("body,ctype", [
    ("not json", "application/json"),
    ("", "application/json"),
    ("null", "application/json"),
    ("[0.5]", "application/json"),
    ('{"score_threshold": NaN}', "application/json"),
    ('{"score_threshold": Infinity}', "application/json"),
    (json.dumps({"score_threshold": True}), "application/json"),
    (json.dumps({"score_threshold": "0.5"}), "application/json"),
    (json.dumps({"score_threshold": 1.5}), "application/json"),
    (json.dumps({"score_threshold": -0.1}), "application/json"),
    (json.dumps({"score_threshold": None}), "application/json"),
    (json.dumps({"classes": "car"}), "application/json"),
    (json.dumps({"classes": ["car", 3]}), "application/json"),
    (json.dumps({"classes": ["car", "spaceship"]}), "application/json"),
    (json.dumps({"threshold": 0.5}), "application/json"),
    (json.dumps({}), "application/json"),
])
def test_settings_invalid(client, state, body, ctype):
    before = client.get("/api/settings").get_json()
    r = client.post("/api/settings", data=body, content_type=ctype)
    assert r.status_code == 400
    assert "error" in r.get_json()
    assert client.get("/api/settings").get_json() == before


def test_settings_aliases(client, state):
    # FakeDetector uses "sofa" (VOC spelling) like the PekingU checkpoints; "couch" must resolve to it.
    r = client.post("/api/settings", json={"classes": ["couch", "car"]})
    assert r.status_code == 200 and r.get_json()["classes"] == ["sofa", "car"]


def test_settings_atomic(client, state):
    r = client.post("/api/settings", json={"score_threshold": 0.2, "classes": ["car", "unicorn"]})
    assert r.status_code == 400 and "unicorn" in r.get_json()["error"]
    assert state.detector.threshold == 0.5 and state.detector.class_filter == []


def test_healthz(client, state):
    r = client.get("/healthz")
    assert r.status_code == 503 and r.get_json()["status"] == "unhealthy"
    publish(state)
    r = client.get("/healthz")
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["capture"]["ok"] and r.get_json()["inference"]["ok"]
    publish(state, frame_id=2, age_s=10)
    assert client.get("/healthz").status_code == 503


def test_unknown_route_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.get_json() == {"error": "not found"}


def test_video_feed_stream_and_disconnect(client, state, caplog):
    caplog.set_level(logging.INFO, logger="app.server")
    publish(state, frame_id=1)
    r = client.get("/video_feed", buffered=False)
    assert r.status_code == 200
    assert r.mimetype == "multipart/x-mixed-replace"
    assert "boundary=frame" in r.headers["Content-Type"]
    assert "no-store" in r.headers["Cache-Control"]
    it = iter(r.response)
    chunk = next(it)
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(JPEG))
    assert b"\xff\xd8" in chunk and chunk.endswith(b"\r\n")
    assert state.clients.value == 1
    assert client.get("/api/stats").get_json()["stream_clients"] == 1
    # next frame arrives when a new result is published
    threading.Timer(0.1, publish, args=(state, 2)).start()
    chunk2 = next(it)
    assert b"\xff\xd8" in chunk2
    r.close()  # client disconnect -> generator closed
    assert state.clients.value == 0
    assert any("stream client disconnected" in rec.getMessage() for rec in caplog.records)


def test_video_feed_ends_on_stop(client, state):
    publish(state)
    r = client.get("/video_feed", buffered=False)
    it = iter(r.response)
    next(it)
    threading.Timer(0.1, state.stop.set).start()
    t0 = time.monotonic()
    with pytest.raises(StopIteration):
        next(it)
    assert time.monotonic() - t0 < 1.5
    r.close()
    assert state.clients.value == 0


def test_video_feed_rate_limit(client, state):
    state.config.server.max_stream_fps = 5
    app = create_app(state)
    c = app.test_client()
    stop_pub = threading.Event()

    def pub():
        i = 0
        while not stop_pub.is_set():
            i += 1
            publish(state, i)
            time.sleep(0.01)

    t = threading.Thread(target=pub, daemon=True)
    t.start()
    r = c.get("/video_feed", buffered=False)
    it = iter(r.response)
    t0 = time.monotonic()
    n = 0
    while time.monotonic() - t0 < 1.0:
        next(it)
        n += 1
    stop_pub.set()
    r.close()
    assert n <= 7  # ~5 fps despite ~100 results/s
