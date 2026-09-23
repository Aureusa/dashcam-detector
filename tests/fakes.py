"""Test doubles that mimic ``app.detector`` without torch/transformers.

``FakeDetector`` implements the same interface as ``RTDetrDetector`` (detect, warmup,
set_threshold, set_class_filter, threshold, class_filter, labels, device, model_name,
last_timings) so the pipeline, server and headless runner can be exercised quickly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

COCO_LABELS: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class FakeDetection:
    """Same fields / to_dict() as ``app.detector.Detection``."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str

    def to_dict(self) -> dict:
        return asdict(self)


class FakeDetector:
    """Deterministic detector: returns a few fixed boxes relative to the frame size.

    Args:
        delay_s: Artificial per-call latency (simulates a slow model).
        threshold: Initial score threshold.
        classes: Initial class filter ([] = all).
        fail_every: If > 0, every N-th ``detect`` call raises ``RuntimeError``.
    """

    #: (relative box, score, class_id)
    TEMPLATE = [
        ((0.10, 0.50, 0.30, 0.80), 0.92, 2),   # car
        ((0.60, 0.40, 0.70, 0.85), 0.81, 0),   # person
        ((0.45, 0.10, 0.48, 0.20), 0.64, 9),   # traffic light
        ((0.80, 0.55, 0.99, 0.95), 0.35, 7),   # truck (low score)
        ((0.00, 0.00, 0.05, 0.05), 0.55, 15),  # cat (non-driving class)
    ]

    def __init__(self, delay_s: float = 0.0, threshold: float = 0.5,
                 classes: Optional[list[str]] = None, fail_every: int = 0) -> None:
        self.delay_s = delay_s
        self.fail_every = fail_every
        self._lock = threading.Lock()
        self._labels = {i: n for i, n in enumerate(COCO_LABELS)}
        self._threshold = threshold
        self._classes: list[str] = []
        self.calls = 0
        self.warmed_up: Optional[tuple[int, int]] = None
        self._timings: dict[str, float] = {}
        if classes:
            self.set_class_filter(classes)

    # interface ------------------------------------------------------------
    def warmup(self, height: int, width: int) -> None:
        self.warmed_up = (height, width)

    def detect(self, frame_bgr: np.ndarray) -> list[FakeDetection]:
        t0 = time.perf_counter()
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("fake detector failure")
        if self.delay_s:
            time.sleep(self.delay_s)
        h, w = frame_bgr.shape[:2]
        with self._lock:
            thr = self._threshold
            allowed = {c.lower() for c in self._classes}
        out = []
        for (rx1, ry1, rx2, ry2), score, cid in self.TEMPLATE:
            label = self._labels[cid]
            if score < thr or (allowed and label.lower() not in allowed):
                continue
            out.append(FakeDetection(rx1 * w, ry1 * h, rx2 * w, ry2 * h, score, cid, label))
        out.sort(key=lambda d: d.score, reverse=True)
        total = (time.perf_counter() - t0) * 1000.0
        self._timings = {"preprocess": 0.0, "forward": total, "postprocess": 0.0, "total": total}
        return out

    def set_threshold(self, t: float) -> None:
        t = float(t)
        if not 0.0 <= t <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        with self._lock:
            self._threshold = t

    def set_class_filter(self, names: list[str]) -> None:
        canon = {n.lower(): n for n in self._labels.values()}
        bad = [n for n in names if n.lower() not in canon]
        if bad:
            raise ValueError(f"unknown class names: {bad}")
        with self._lock:
            self._classes = [canon[n.lower()] for n in names]

    @property
    def threshold(self) -> float:
        with self._lock:
            return self._threshold

    @property
    def class_filter(self) -> list[str]:
        with self._lock:
            return list(self._classes)

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._labels)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "fake-detector"

    @property
    def last_timings(self) -> dict[str, float]:
        return dict(self._timings)


def make_frame(h: int = 360, w: int = 640, value: int = 90) -> np.ndarray:
    """Solid BGR test frame."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    img[:, : w // 2, 2] = 200  # some color so JPEGs are not trivial
    return img
