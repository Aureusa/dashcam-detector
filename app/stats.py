"""Thread-safe rolling rate and latency meters."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


class RateMeter:
    """Events-per-second over a sliding time window (default 2 s).

    Args:
        window_s: Sliding window length in seconds.
        clock: Monotonic clock function (injectable for tests).
    """

    def __init__(self, window_s: float = 2.0, clock: Callable[[], float] = time.monotonic) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self._window = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._ts: deque[float] = deque()
        self._count = 0
        self._last: Optional[float] = None

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()

    def tick(self) -> None:
        """Record one event at the current time."""
        now = self._clock()
        with self._lock:
            self._ts.append(now)
            self._count += 1
            self._last = now
            self._trim(now)

    def rate(self) -> float:
        """Events per second over the window (0.0 if fewer than 2 events or stale)."""
        now = self._clock()
        with self._lock:
            self._trim(now)
            n = len(self._ts)
            if n < 2:
                return 0.0
            span = self._ts[-1] - self._ts[0]
            avg_dt = span / (n - 1)
            # If ticks stop, let the rate decay instead of reporting the last steady value.
            span = max(span, (now - self._ts[0]) - avg_dt)
            if span <= 0:
                return 0.0
            return (n - 1) / span

    @property
    def count(self) -> int:
        """Total number of ticks since creation."""
        with self._lock:
            return self._count

    @property
    def last_ts(self) -> Optional[float]:
        """Monotonic timestamp of the last tick, or None."""
        with self._lock:
            return self._last


class LatencyMeter:
    """Mean / p95 over the last ``window`` samples (milliseconds)."""

    def __init__(self, window: int = 100) -> None:
        if window <= 0:
            raise ValueError("window must be > 0")
        self._lock = threading.Lock()
        self._samples: deque[float] = deque(maxlen=window)

    def add(self, ms: float) -> None:
        """Add one latency sample in milliseconds (non-finite values are ignored)."""
        if not math.isfinite(ms):
            return
        with self._lock:
            self._samples.append(float(ms))

    def mean(self) -> float:
        """Mean of the window (0.0 if empty)."""
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    def p95(self) -> float:
        """95th percentile of the window using nearest-rank (0.0 if empty)."""
        return self.percentile(95.0)

    def percentile(self, q: float) -> float:
        """Nearest-rank percentile ``q`` in [0, 100] (0.0 if empty)."""
        with self._lock:
            data = sorted(self._samples)
        if not data:
            return 0.0
        k = max(0, min(len(data) - 1, math.ceil(q / 100.0 * len(data)) - 1))
        return data[k]

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def snapshot(self) -> dict[str, float]:
        """Return ``{"mean": ..., "p95": ...}`` rounded to 0.1 ms."""
        return {"mean": round(self.mean(), 1), "p95": round(self.p95(), 1)}


@dataclass
class PipelineMeters:
    """All meters shared between capture, inference worker, server and HUD."""

    capture_fps: RateMeter = field(default_factory=RateMeter)
    inference_fps: RateMeter = field(default_factory=RateMeter)
    inference_ms: LatencyMeter = field(default_factory=LatencyMeter)
    e2e_ms: LatencyMeter = field(default_factory=LatencyMeter)
