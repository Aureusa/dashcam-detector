"""Tests for app.capture using file sources (no camera needed)."""

from __future__ import annotations

import dataclasses
import os
import threading
import time

import pytest

from app import capture as capmod
from app.buffers import LatestValue
from app.capture import CameraStream, CaptureError, FramePacket, classify_source, device_path, open_capture
from app.config import CameraConfig
from app.stats import RateMeter

CLIP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "clip.mp4")


def cam_cfg(source, **kw) -> CameraConfig:
    return dataclasses.replace(CameraConfig(), source=source, **kw)


@pytest.mark.parametrize("src,kind", [
    (0, "device"), ("2", "device"), ("/dev/video0", "device"),
    ("rtsp://1.2.3.4/stream", "rtsp"), ("http://cam.local/mjpeg", "url"),
    ("samples/clip.mp4", "file"), ("/tmp/x.avi", "file"),
    ("v4l2src ! videoconvert ! appsink", "gstreamer"),
])
def test_classify_source(src, kind):
    assert classify_source(src) == kind


def test_device_path():
    assert device_path(3) == "/dev/video3"
    assert device_path("/dev/video1") == "/dev/video1"
    assert device_path("clip.mp4") is None


def test_open_file_negotiated(tiny_video):
    cap, neg = open_capture(cam_cfg(tiny_video))
    try:
        assert (neg["width"], neg["height"]) == (160, 120)
        assert neg["fps"] == pytest.approx(20.0)
        assert neg["kind"] == "file" and neg["source"] == tiny_video
        assert neg["frame_count"] == 20
        ok, frame = cap.read()
        assert ok and frame.shape == (120, 160, 3)
    finally:
        cap.release()


def test_missing_file_friendly_error(tmp_path):
    with pytest.raises(CaptureError, match="Video file not found"):
        open_capture(cam_cfg(str(tmp_path / "nope.mp4")))


def test_missing_device_friendly_error():
    with pytest.raises(CaptureError) as ei:
        open_capture(cam_cfg("/dev/video97"))
    msg = str(ei.value)
    assert "does not exist" in msg and "v4l2-ctl --list-devices" in msg


def test_permission_denied_friendly_error(monkeypatch):
    real_exists = os.path.exists
    monkeypatch.setattr(capmod.os.path, "exists", lambda p: True if p == "/dev/video42" else real_exists(p))
    monkeypatch.setattr(capmod.os, "access", lambda p, mode: False)
    with pytest.raises(CaptureError) as ei:
        open_capture(cam_cfg(42))
    msg = str(ei.value)
    assert "Permission denied" in msg and "usermod -aG video" in msg


def test_rtsp_sets_tcp_transport(monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)

    class DummyCap:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(capmod.cv2, "VideoCapture", DummyCap)
    with pytest.raises(CaptureError, match="Could not open"):
        open_capture(cam_cfg("rtsp://127.0.0.1:1/none"))
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def _run_stream(cfg, seconds):
    out: LatestValue[FramePacket] = LatestValue()
    stop = threading.Event()
    meter = RateMeter()
    cam = CameraStream(cfg, out, stop, meter)
    cam.open()
    cam.start()
    ids = []
    last = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        v, pkt = out.wait_newer(last, timeout=0.2)
        if v > last and pkt is not None:
            ids.append(pkt.frame_id)
            last = v
    return cam, stop, meter, ids, out


def test_stream_paced_and_looping(tiny_video):
    cam, stop, meter, ids, out = _run_stream(cam_cfg(tiny_video, loop_file=True, pace_file=True), 1.6)
    try:
        assert cam.connected
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        # 20 fps file for 1.6 s → ~32 frames: more than the 20 in the file → looping works
        assert 24 <= meter.count <= 40, meter.count
        assert 15 <= meter.rate() <= 25
        _, pkt = out.get()
        assert pkt.image.shape == (120, 160, 3) and pkt.capture_ts <= time.monotonic()
        st = cam.status()
        assert st["connected"] and st["thread_alive"] and st["last_frame_age_s"] < 1.0
        assert cam.negotiated["width"] == 160
    finally:
        t0 = time.monotonic()
        stop.set()
        cam.join(2.0)
        assert not cam.is_alive
        assert time.monotonic() - t0 < 1.0
    assert not cam.connected  # released


def test_stream_unpaced_runs_fast(tiny_video):
    cam, stop, meter, ids, _ = _run_stream(cam_cfg(tiny_video, pace_file=False), 0.5)
    stop.set()
    cam.join(2.0)
    assert meter.count > 60  # well beyond native 20 fps


def test_stream_no_loop_stops_at_eof(tiny_video):
    cam, stop, meter, ids, _ = _run_stream(cam_cfg(tiny_video, loop_file=False, pace_file=False), 0.5)
    try:
        assert meter.count == 20
        assert cam.is_alive  # idles until stop
    finally:
        stop.set()
        cam.join(2.0)
    assert not cam.is_alive


def test_stream_opens_in_thread_and_retries(tmp_path):
    """Starting without open(): a missing file is retried (no crash) until stop."""
    cfg = cam_cfg(str(tmp_path / "later.avi"), reconnect_delay_s=0.1)
    out: LatestValue[FramePacket] = LatestValue()
    stop = threading.Event()
    cam = CameraStream(cfg, out, stop, RateMeter())
    cam.start()
    time.sleep(0.3)
    assert cam.is_alive and not cam.connected
    assert "not found" in (cam.last_error or "")
    stop.set()
    cam.join(2.0)
    assert not cam.is_alive


@pytest.mark.skipif(not os.path.exists(CLIP), reason="samples/clip.mp4 not present")
def test_sample_clip_properties():
    cap, neg = open_capture(cam_cfg(CLIP))
    cap.release()
    assert (neg["width"], neg["height"]) == (1280, 720)
    assert neg["fps"] == pytest.approx(30.0, abs=0.1)
