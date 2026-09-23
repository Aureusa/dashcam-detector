"""Tests for the InferenceWorker with a fake detector (no torch)."""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from app.buffers import LatestValue
from app.capture import FramePacket
from app.config import RenderConfig, ServerConfig
from app.pipeline import InferenceWorker, ResultPacket
from app.stats import PipelineMeters
from fakes import FakeDetector, make_frame


def make_worker(detector, **kw):
    frames: LatestValue[FramePacket] = LatestValue()
    results: LatestValue[ResultPacket] = LatestValue()
    stop = threading.Event()
    meters = PipelineMeters()
    w = InferenceWorker(detector, frames, results, stop, kw.pop("render_cfg", RenderConfig()),
                        ServerConfig(), meters, {"device": "cpu", "model": "fake"}, **kw)
    return w, frames, results, stop, meters


def producer(frames, stop, fps=30.0, meter=None, shape=(360, 640)):
    """Simulated camera: puts a fresh frame every 1/fps s."""
    i = 0
    period = 1.0 / fps
    nxt = time.monotonic()
    while not stop.is_set():
        i += 1
        frames.put(FramePacket(i, time.monotonic(), make_frame(*shape)))
        if meter is not None:
            meter.tick()
        nxt += period
        stop.wait(max(0.0, nxt - time.monotonic()))


def test_process_single_frame():
    det = FakeDetector()
    w, frames, results, stop, meters = make_worker(det)
    pkt = FramePacket(7, time.monotonic(), make_frame())
    res = w.process(pkt)
    assert results.get() == (1, res)
    assert res.frame_id == 7 and res.width == 640 and res.height == 360
    assert res.jpeg[:2] == b"\xff\xd8" and res.jpeg[-2:] == b"\xff\xd9"
    assert [d.label for d in res.detections] == ["car", "person", "traffic light", "cat"]
    assert res.publish_ts >= res.capture_ts and res.inference_ms >= 0
    img = cv2.imdecode(np.frombuffer(res.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (360, 640, 3)
    assert meters.inference_fps.count == 1 and len(meters.e2e_ms) == 1


def test_output_width_applied():
    rc = RenderConfig()
    rc.output_width = 320
    w, *_ = make_worker(FakeDetector(), render_cfg=rc)
    res = w.process(FramePacket(1, time.monotonic(), make_frame()))
    assert (res.width, res.height) == (320, 180)


def test_threshold_and_filter_take_effect_next_frame():
    det = FakeDetector()
    w, *_ = make_worker(det)
    det.set_threshold(0.7)
    det.set_class_filter(["car", "Person"])
    res = w.process(FramePacket(1, time.monotonic(), make_frame()))
    assert sorted(d.label for d in res.detections) == ["car", "person"]


def test_worker_thread_runs_and_stops_quickly():
    w, frames, results, stop, meters = make_worker(FakeDetector())
    w.start()
    p = threading.Thread(target=producer, args=(frames, stop, 30.0, meters.capture_fps), daemon=True)
    p.start()
    time.sleep(1.0)
    v, res = results.get()
    assert v > 10 and res is not None
    assert meters.inference_fps.rate() > 15
    t0 = time.monotonic()
    stop.set()
    frames.close()
    w.join(2.0)
    assert not w.is_alive and time.monotonic() - t0 < 1.5


def test_worker_exits_on_stop_while_idle():
    w, frames, results, stop, meters = make_worker(FakeDetector())
    w.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    stop.set()
    w.join(3.0)
    assert not w.is_alive and time.monotonic() - t0 <= 1.2  # wait_newer timeout is 1 s


def test_slow_detector_latency_bounded():
    """PLAN §9.5: with a 0.2 s detector, frames are dropped (not queued) and e2e stays ~200-300 ms."""
    det = FakeDetector(delay_s=0.2)
    w, frames, results, stop, meters = make_worker(det)
    p = threading.Thread(target=producer, args=(frames, stop, 30.0, meters.capture_fps), daemon=True)
    e2e: list[float] = []

    def collect():
        last = 0
        while not stop.is_set():
            v, r = results.wait_newer(last, timeout=0.3)
            if r is not None and v > last:
                last = v
                e2e.append((r.publish_ts - r.capture_ts) * 1000)

    c = threading.Thread(target=collect, daemon=True)
    p.start()
    w.start()
    c.start()
    time.sleep(3.0)
    stop.set()
    frames.close()
    results.close()
    w.join(2.0)
    assert 10 <= len(e2e) <= 16          # ~5 fps
    assert max(e2e) < 320, e2e            # never more than ~one frame period + detect time
    first, last = np.mean(e2e[:4]), np.mean(e2e[-4:])
    assert last < first + 40              # not growing
    assert meters.capture_fps.count > 5 * len(e2e)  # most frames dropped


def test_exceptions_do_not_kill_worker(caplog):
    det = FakeDetector(fail_every=2)
    w, frames, results, stop, meters = make_worker(det)
    caplog.set_level(logging.ERROR, logger="app.pipeline")
    w.start()
    p = threading.Thread(target=producer, args=(frames, stop, 50.0), daemon=True)
    p.start()
    time.sleep(0.8)
    stop.set()
    w.join(2.0)
    assert w.errors >= 5 and w.processed >= 5
    fails = [r for r in caplog.records if "inference iteration failed" in r.getMessage()]
    assert 1 <= len(fails) <= 2  # identical errors are rate-limited


def test_stats_log_line_and_slow_warning(caplog):
    det = FakeDetector(delay_s=0.1)
    w, frames, results, stop, meters = make_worker(det, stats_interval_s=0.3)
    w._slow.intervals_needed = 2
    caplog.set_level(logging.INFO, logger="app.pipeline")
    p = threading.Thread(target=producer, args=(frames, stop, 30.0, meters.capture_fps), daemon=True)
    p.start()
    w.start()
    time.sleep(3.0)
    stop.set()
    w.join(2.0)
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("stats: capture") for m in msgs)
    assert any("well below capture" in m for m in msgs)


def test_hud_drawn_when_enabled():
    rc_on, rc_off = RenderConfig(), RenderConfig()
    rc_on.draw_hud, rc_off.draw_hud = True, False
    frame = np.full((360, 640, 3), 200, np.uint8)
    a, *_ = make_worker(FakeDetector(threshold=1.0), render_cfg=rc_on)
    b, *_ = make_worker(FakeDetector(threshold=1.0), render_cfg=rc_off)
    ja = a.process(FramePacket(1, time.monotonic(), frame.copy())).jpeg
    jb = b.process(FramePacket(1, time.monotonic(), frame.copy())).jpeg
    ia = cv2.imdecode(np.frombuffer(ja, np.uint8), cv2.IMREAD_COLOR)
    ib = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
    assert ia[5, 5].mean() < 150 and ib[5, 5].mean() > 180


def test_warmup_runs_in_inference_thread():
    calls = []

    class D(FakeDetector):
        def warmup(self, height, width):
            calls.append((threading.current_thread().name, height, width))
            time.sleep(0.2)

    w, frames, results, stop, meters = make_worker(D(), warmup_shape=(360, 640))
    frames.put(FramePacket(1, time.monotonic(), make_frame()))  # stale frame captured during warmup
    w.start()
    assert not w.ready.is_set()
    assert w.ready.wait(2.0)
    assert calls == [("inference", 360, 640)]
    assert w.warmup_s is not None and w.warmup_s >= 0.2
    frames.put(FramePacket(2, time.monotonic(), make_frame()))
    v, res = results.wait_newer(0, timeout=2.0)
    assert res is not None and res.frame_id in (1, 2)
    stop.set()
    frames.close()
    w.join(2.0)
