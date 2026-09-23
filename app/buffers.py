"""Thread-safe single-slot "latest value" buffer.

Used between pipeline stages instead of a queue: a producer overwrites the
slot, so slow consumers always see the newest value and stale values are
dropped. This keeps end-to-end latency bounded.
"""

from __future__ import annotations

import threading
import time
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class LatestValue(Generic[T]):
    """Thread-safe single-slot buffer. ``put()`` overwrites; readers can wait for a newer version.

    Every ``put`` increments a version counter (starting at 0 for "empty").
    Any number of readers may call :meth:`wait_newer` concurrently, each
    tracking its own last-seen version.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._version: int = 0
        self._value: Optional[T] = None
        self._closed = False

    def put(self, value: T) -> int:
        """Store ``value`` (overwriting any previous one) and wake all waiters.

        Returns:
            The new version number.
        """
        with self._cond:
            self._version += 1
            self._value = value
            self._cond.notify_all()
            return self._version

    def get(self) -> tuple[int, Optional[T]]:
        """Return ``(version, value)`` without blocking. ``(0, None)`` if nothing was put yet."""
        with self._cond:
            return self._version, self._value

    def wait_newer(self, last_version: int, timeout: float) -> tuple[int, Optional[T]]:
        """Block until the version is greater than ``last_version`` or ``timeout`` elapses.

        Returns:
            ``(version, value)``. On timeout the current (possibly unchanged)
            version/value is returned; callers detect this via ``version <= last_version``.
            Returns immediately (without waiting) once the buffer is closed.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while self._version <= last_version and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self._version, self._value

    def close(self) -> None:
        """Mark the buffer closed (shutdown): current and future ``wait_newer`` calls return immediately.

        ``put``/``get`` keep working so late producers do not fail.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        with self._cond:
            return self._closed

    @property
    def version(self) -> int:
        """Current version number (0 = nothing put yet)."""
        with self._cond:
            return self._version
