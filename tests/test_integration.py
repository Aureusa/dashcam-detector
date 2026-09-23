"""End-to-end test (PLAN §13.1): run ``python -m app.main`` on samples/clip.mp4 with the REAL detector.

Skipped when the clip, torch/transformers or the model weights are unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = os.path.join(PROJECT_ROOT, "samples", "clip.mp4")

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


def _skip_reason() -> str | None:
    if not os.path.exists(CLIP):
        return "samples/clip.mp4 missing"
    for mod in ("torch", "transformers"):
        if importlib.util.find_spec(mod) is None:
            return f"{mod} not installed"
    if not os.path.exists(os.path.join(PROJECT_ROOT, "app", "detector.py")):
        return "app/detector.py missing"
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def post_json(url: str, data: bytes, timeout: float = 5.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


@pytest.fixture(scope="module")
def app_proc(tmp_path_factory):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    port = free_port()
    log_path = tmp_path_factory.mktemp("integration") / "app.log"
    logf = open(log_path, "w")
    cmd = [sys.executable, "-m", "app.main", "--source", CLIP, "--host", "127.0.0.1", "--port", str(port)]
    if os.path.exists(os.path.join(PROJECT_ROOT, "config.yaml")):
        cmd[3:3] = ["--config", "config.yaml"]
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        yield proc, base, log_path
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)
        logf.close()


def _log(log_path) -> str:
    return open(log_path).read()


def test_full_app(app_proc):
    proc, base, log_path = app_proc

    # 1. inference_fps > 0 within 60 s (model download/warmup may take a while on first run)
    deadline = time.monotonic() + 60
    stats = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"app exited early with {proc.returncode}:\n{_log(log_path)[-3000:]}")
        try:
            code, stats = get_json(base + "/api/stats", timeout=2)
            if code == 200 and stats.get("inference_fps", 0) > 0:
                break
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    else:
        pytest.fail(f"inference_fps never > 0; last stats={stats}\n{_log(log_path)[-3000:]}")
    assert stats["frame_id"] is not None

    # 2. MJPEG stream: boundary + JPEG SOI
    with urllib.request.urlopen(base + "/video_feed", timeout=5) as r:
        assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
        head = r.read(4096)
    assert head.startswith(b"--frame")
    assert b"\xff\xd8" in head

    # 3. settings validation
    code, body = post_json(base + "/api/settings", b'{"score_threshold": 2}')
    assert code == 400 and "error" in body
    code, body = post_json(base + "/api/settings", b"garbage")
    assert code == 400
    code, body = post_json(base + "/api/settings", b'{"classes": ["car", "flying saucer"]}')
    assert code == 400
    code, body = post_json(base + "/api/settings", b'{"score_threshold": 0.4, "classes": ["car", "person"]}')
    assert code == 200, body
    code, got = get_json(base + "/api/settings")
    assert code == 200 and got["score_threshold"] == pytest.approx(0.4)
    assert sorted(c.lower() for c in got["classes"]) == ["car", "person"]

    # 4. other endpoints
    code, labels = get_json(base + "/api/labels")
    assert code == 200 and "car" in [l.lower() for l in labels["labels"]]
    code, dets = get_json(base + "/api/detections")
    assert code == 200 and "detections" in dets
    time.sleep(0.5)
    code, dets = get_json(base + "/api/detections")
    assert all(d["label"].lower() in ("car", "person") and d["score"] >= 0.4 for d in dets["detections"])
    code, health = get_json(base + "/healthz")
    assert code == 200, health

    # 5. clean shutdown on SIGINT
    t0 = time.monotonic()
    proc.send_signal(signal.SIGINT)
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pytest.fail("app did not exit within 5 s of SIGINT")
    elapsed = time.monotonic() - t0
    log = _log(log_path)
    assert rc == 0, log[-3000:]
    assert "shutdown complete" in log
    assert "Traceback" not in log, log[-3000:]
    print(f"shutdown took {elapsed:.2f} s")
