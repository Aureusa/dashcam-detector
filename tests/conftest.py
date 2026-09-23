"""Shared pytest setup: put the project root (and tests/) on sys.path, common fixtures."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (PROJECT_ROOT, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fakes import FakeDetector  # noqa: E402

CLIP_PATH = os.path.join(PROJECT_ROOT, "samples", "clip.mp4")


@pytest.fixture
def fake_detector() -> FakeDetector:
    """Fast fake detector (no torch)."""
    return FakeDetector()


@pytest.fixture
def tiny_video(tmp_path) -> str:
    """A 20-frame 160x120 @ 20 fps MJPG AVI with a moving square (frame i has square at x=4*i)."""
    path = str(tmp_path / "tiny.avi")
    w, h, n = 160, 120, 20
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (w, h))
    assert vw.isOpened(), "OpenCV cannot write MJPG AVI"
    for i in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.rectangle(img, (4 * i, 40), (4 * i + 30, 70), (0, 200, 255), -1)
        vw.write(img)
    vw.release()
    return path
