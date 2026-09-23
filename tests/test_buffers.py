"""Tests for LatestValue (single-slot buffer) and the rolling meters."""

from __future__ import annotations

import threading
import time

import pytest

from app.buffers import LatestValue
from app.stats import LatencyMeter, RateMeter


def test_empty_get():
    b: LatestValue[int] = LatestValue()
    assert b.get() == (0, None)
    assert b.version == 0


def test_put_overwrites_and_versions_increase():
    b: LatestValue[str] = LatestValue()
    assert b.put("a") == 1
    assert b.put("b") == 2
    assert b.put("c") == 3
    assert b.get() == (3, "c")  # only the newest value is kept


def test_wait_newer_returns_immediately_if_already_newer():
    b: LatestValue[int] = LatestValue()
    b.put(10)
    t0 = time.monotonic()
    assert b.wait_newer(0, timeout=5.0) == (1, 10)
    assert time.monotonic() - t0 < 0.1


def test_wait_newer_timeout():
    b: LatestValue[int] = LatestValue()
    b.put(1)
    t0 = time.monotonic()
    v, val = b.wait_newer(1, timeout=0.2)
    dt = time.monotonic() - t0
    assert (v, val) == (1, 1)  # unchanged: caller detects timeout via v <= last
    assert 0.15 <= dt < 1.0


def test_wait_newer_wakes_on_put():
    b: LatestValue[int] = LatestValue()
    threading.Timer(0.1, b.put, args=(42,)).start()
    t0 = time.monotonic()
    v, val = b.wait_newer(0, timeout=5.0)
    assert (v, val) == (1, 42)
    assert time.monotonic() - t0 < 1.0


def test_multiple_concurrent_readers_each_see_value():
    b: LatestValue[int] = LatestValue()
    got: list[tuple[int, int]] = []
    lock = threading.Lock()
    ready = threading.Barrier(6)

    def reader() -> None:
        ready.wait()
        v, val = b.wait_newer(0, timeout=5.0)
        with lock:
            got.append((v, val))

    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    ready.wait()
    time.sleep(0.05)
    b.put(7)
    for t in threads:
        t.join(2.0)
    assert got == [(1, 7)] * 5


def test_readers_track_versions_independently():
    b: LatestValue[int] = LatestValue()
    for i in range(3):
        b.put(i)
    # slow reader at version 1 and fast reader at version 3
    assert b.wait_newer(1, timeout=0.01) == (3, 2)
    assert b.wait_newer(3, timeout=0.01) == (3, 2)


def test_close_releases_waiters():
    b: LatestValue[int] = LatestValue()
    threading.Timer(0.1, b.close).start()
    t0 = time.monotonic()
    assert b.wait_newer(0, timeout=5.0) == (0, None)
    assert time.monotonic() - t0 < 1.0
    assert b.closed
    b.put(1)  # still usable
    assert b.get() == (1, 1)


def test_concurrent_producer_consumer_never_goes_backwards():
    b: LatestValue[int] = LatestValue()
    stop = threading.Event()

    def producer() -> None:
        i = 0
        while not stop.is_set():
            i += 1
            b.put(i)

    t = threading.Thread(target=producer)
    t.start()
    last_v, seen = 0, []
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        v, val = b.wait_newer(last_v, timeout=0.1)
        if v > last_v:
            seen.append(val)
            last_v = v
    stop.set()
    t.join()
    assert seen == sorted(seen) and len(seen) > 1


# --- stats ---------------------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_rate_meter_steady_rate():
    clk = FakeClock()
    m = RateMeter(window_s=2.0, clock=clk)
    assert m.rate() == 0.0
    for _ in range(100):
        clk.t += 1 / 30
        m.tick()
    assert m.rate() == pytest.approx(30.0, rel=0.05)
    assert m.count == 100


def test_rate_meter_decays_when_ticks_stop():
    clk = FakeClock()
    m = RateMeter(window_s=2.0, clock=clk)
    for _ in range(60):
        clk.t += 1 / 30
        m.tick()
    clk.t += 1.0
    assert m.rate() < 20
    clk.t += 5.0
    assert m.rate() == 0.0


def test_latency_meter_mean_p95_window():
    m = LatencyMeter(window=100)
    assert m.mean() == 0.0 and m.p95() == 0.0
    for v in range(1, 101):
        m.add(float(v))
    assert m.mean() == pytest.approx(50.5)
    assert m.p95() == 95.0
    for _ in range(100):
        m.add(1.0)  # old samples fall out of the window
    assert m.mean() == 1.0 and m.p95() == 1.0
    m.add(float("nan"))  # ignored
    assert len(m) == 100


def test_meters_thread_safe():
    r, l = RateMeter(), LatencyMeter(window=1000)

    def work() -> None:
        for i in range(1000):
            r.tick()
            l.add(i)

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert r.count == 4000
    assert len(l) == 1000
