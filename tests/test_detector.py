"""Tests for app.detector (RT-DETR wrapper).

Skipped entirely if the checkpoint cannot be loaded (e.g. offline without cache).
Checkpoint defaults to the fast r18 model; override with RTDETR_TEST_CHECKPOINT.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from app.config import ModelConfig  # noqa: E402
from app.detector import Detection, RTDetrDetector  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = PROJECT_ROOT / "samples" / "coco"
CATS = COCO_DIR / "000000039769.jpg"
CHECKPOINT = os.environ.get("RTDETR_TEST_CHECKPOINT", "PekingU/rtdetr_v2_r18vd")

DET_KEYS = {"x1", "y1", "x2", "y2", "score", "class_id", "label"}


def _cfg(**kw) -> ModelConfig:
    base = dict(checkpoint=CHECKPOINT, device="auto", fp16=True, score_threshold=0.5, classes=[], warmup_iters=2)
    base.update(kw)
    return ModelConfig(**base)


def _load(**kw) -> RTDetrDetector:
    try:
        return RTDetrDetector(_cfg(**kw.pop("cfg", {})), **kw)
    except (OSError, EnvironmentError, ValueError) as exc:  # weights not downloadable/cached
        pytest.skip(f"cannot load checkpoint {CHECKPOINT}: {exc}")


@pytest.fixture(scope="module")
def detector() -> RTDetrDetector:
    """Fast-path detector shared by the module (filter = all classes, threshold 0.5)."""
    return _load()


@pytest.fixture(scope="module")
def reference_detector() -> RTDetrDetector:
    """Detector using the HF image processor for preprocessing (parity baseline)."""
    return _load(use_reference_preprocess=True)


@pytest.fixture(autouse=True)
def _reset(request):
    """Restore threshold/filter after each test that uses the shared detector."""
    yield
    if "detector" in request.fixturenames:
        det = request.getfixturevalue("detector")
        det.set_threshold(0.5)
        det.set_class_filter([])


@pytest.fixture(scope="module")
def cats() -> np.ndarray:
    img = cv2.imread(str(CATS))
    if img is None:
        pytest.skip(f"missing sample image {CATS}")
    return img


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------- basics


def test_device_and_metadata(detector: RTDetrDetector) -> None:
    if torch.cuda.is_available():
        assert detector.device.type == "cuda"
    assert detector.model_name == CHECKPOINT
    assert len(detector.labels) == 80
    assert "cat" in detector.labels.values()
    assert detector.threshold == pytest.approx(0.5)
    assert detector.class_filter == []


def test_two_cats(detector: RTDetrDetector, cats: np.ndarray) -> None:
    dets = detector.detect(cats)
    cats_found = [d for d in dets if d.label == "cat"]
    assert len(cats_found) == 2
    assert all(d.score > 0.8 for d in cats_found)
    labels = {d.label for d in dets}
    assert "remote" in labels
    assert labels & {"sofa", "couch"}
    scores = [d.score for d in dets]
    assert scores == sorted(scores, reverse=True)


def test_detection_fields_valid(detector: RTDetrDetector) -> None:
    detector.set_threshold(0.3)
    names = set(detector.labels.values())
    n = 0
    for path in sorted(COCO_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        for d in detector.detect(img):
            n += 1
            assert 0.0 <= d.x1 < d.x2 <= w
            assert 0.0 <= d.y1 < d.y2 <= h
            assert 0.0 <= d.score <= 1.0
            assert d.score >= 0.3
            assert d.label in names
            assert detector.labels[d.class_id] == d.label
    assert n > 0


def test_to_dict(detector: RTDetrDetector, cats: np.ndarray) -> None:
    d = detector.detect(cats)[0]
    out = d.to_dict()
    assert set(out) == DET_KEYS
    assert isinstance(out["class_id"], int) and isinstance(out["label"], str)
    assert out["score"] == round(d.score, 4)
    assert out["x1"] == round(d.x1, 1)
    manual = Detection(1.26, 2.0, 3.04, 4.0, 0.123456, 15, "cat").to_dict()
    assert manual == {"x1": 1.3, "y1": 2.0, "x2": 3.0, "y2": 4.0, "score": 0.1235, "class_id": 15, "label": "cat"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.score = 0.0  # type: ignore[misc]


def test_last_timings(detector: RTDetrDetector, cats: np.ndarray) -> None:
    detector.detect(cats)
    t = detector.last_timings
    assert set(t) == {"preprocess", "forward", "postprocess", "total"}
    assert all(v >= 0 for v in t.values())
    assert t["total"] >= t["forward"]


def test_warmup_and_other_sizes(detector: RTDetrDetector) -> None:
    detector.warmup(720, 1280)
    blank = np.zeros((720, 1280, 3), np.uint8)
    assert isinstance(detector.detect(blank), list)
    tiny = np.full((17, 23, 3), 127, np.uint8)
    for d in detector.detect(tiny):
        assert 0 <= d.x1 <= d.x2 <= 23 and 0 <= d.y1 <= d.y2 <= 17


def test_bad_input_rejected(detector: RTDetrDetector) -> None:
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10, 3), np.float32))


# ---------------------------------------------------------------------------- live settings


def test_class_filter(detector: RTDetrDetector, cats: np.ndarray) -> None:
    detector.set_class_filter(["cat"])
    assert detector.class_filter == ["cat"]
    dets = detector.detect(cats)
    assert len(dets) == 2 and {d.label for d in dets} == {"cat"}

    detector.set_class_filter(["CAT", " Remote "])  # case-insensitive, trimmed
    assert detector.class_filter == ["cat", "remote"]
    assert {d.label for d in detector.detect(cats)} == {"cat", "remote"}

    detector.set_class_filter(["person"])
    assert detector.detect(cats) == []

    detector.set_class_filter([])
    assert detector.class_filter == []
    assert len(detector.detect(cats)) >= 4


def test_class_filter_aliases(detector: RTDetrDetector) -> None:
    # PekingU checkpoints spell COCO names VOC-style ("motorbike", "sofa"); COCO spellings are accepted.
    detector.set_class_filter(["motorcycle", "couch"])
    assert set(detector.class_filter) <= set(detector.labels.values())
    assert len(detector.class_filter) == 2


def test_class_filter_unknown_raises_and_keeps_state(detector: RTDetrDetector) -> None:
    detector.set_class_filter(["cat"])
    with pytest.raises(ValueError, match="unicorn"):
        detector.set_class_filter(["cat", "unicorn"])
    assert detector.class_filter == ["cat"]
    with pytest.raises(ValueError):
        detector.set_class_filter("cat")  # type: ignore[arg-type]


def test_unknown_configured_classes_are_dropped() -> None:
    det = _load(cfg={"classes": ["car", "flying saucer", "person"], "warmup_iters": 0})
    assert det.class_filter == ["car", "person"]


def test_threshold(detector: RTDetrDetector, cats: np.ndarray) -> None:
    detector.set_threshold(0.3)
    low = detector.detect(cats)
    detector.set_threshold(0.9)
    assert detector.threshold == pytest.approx(0.9)
    high = detector.detect(cats)
    assert 0 < len(high) < len(low)
    assert all(d.score >= 0.9 for d in high)
    low_keys = {(d.label, round(d.x1), round(d.y1), round(d.x2), round(d.y2)) for d in low}
    for d in high:
        assert (d.label, round(d.x1), round(d.y1), round(d.x2), round(d.y2)) in low_keys


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf"), "0.5", None, True])
def test_threshold_invalid(detector: RTDetrDetector, bad) -> None:
    with pytest.raises(ValueError):
        detector.set_threshold(bad)
    assert detector.threshold == pytest.approx(0.5)


def test_detect_from_other_thread(detector: RTDetrDetector, cats: np.ndarray) -> None:
    results: list = []
    errors: list = []

    def worker() -> None:
        try:
            for _ in range(5):
                results.append(detector.detect(cats))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for _ in range(20):  # concurrent live updates from "UI" thread
        detector.set_threshold(0.5)
        detector.set_class_filter([])
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert len(results) == 10 and all(len(r) >= 2 for r in results)


# ---------------------------------------------------------------------------- parity


def _coco_images() -> list[Path]:
    images = sorted(COCO_DIR.glob("*.jpg"))
    if len(images) < 5:
        pytest.skip("need >= 5 sample images in samples/coco")
    return images


def test_preprocess_tensor_parity(detector: RTDetrDetector, reference_detector: RTDetrDetector) -> None:
    """The GPU preprocessing must reproduce the HF processor's pixel_values (up to uint8 rounding)."""
    worst_max, worst_mean = 0.0, 0.0
    with torch.inference_mode():
        for path in _coco_images():
            img = cv2.imread(str(path))
            fast = detector._preprocess_fast(img).float().cpu()
            ref = reference_detector._preprocess_reference(img).float().cpu()
            assert fast.shape == ref.shape == (1, 3, 640, 640)
            diff = (fast - ref).abs()
            worst_max, worst_mean = max(worst_max, float(diff.max())), max(worst_mean, float(diff.mean()))
    print(f"\npreprocess parity: max |diff| {worst_max:.5f} (1/255={1 / 255:.5f}), worst mean {worst_mean:.2e}")
    assert worst_max <= 1 / 255 + 1e-6  # at most one uint8 rounding step
    assert worst_mean < 1e-3  # a BGR/RGB swap or missing/extra normalization gives >> 1e-2


def test_parity_fast_vs_reference(detector: RTDetrDetector, reference_detector: RTDetrDetector) -> None:
    """Detections from the fast path match the HF-processor path (PLAN 7.8).

    Criterion per matched top detection: same label, IoU > 0.9, |score diff| < 0.05. RT-DETR's
    discrete top-K query selection makes a few ambiguous objects (overlapping duplicates) sensitive
    to 1/255-level input/numeric noise: with larger checkpoints the same deltas appear between the
    reference path in fp16 vs fp32 on identical inputs, and even between processes (cudnn.benchmark
    algorithm choice), e.g. r50's motorbike+sidecar in 000000007386.jpg scores 0.52 or 0.69. So we
    require the strict criterion for every image's top detection and for >= 90% of all matches, and
    for the rest only that the object is found (same label, IoU > 0.5). Exact preprocessing parity is
    asserted separately in test_preprocess_tensor_parity. Fast-path candidates use threshold 0.3 so a
    reference detection at ~0.5 cannot vanish just by crossing the threshold.
    """
    detector.set_threshold(0.3)
    matched, strict_ok, min_iou, max_dscore, worst = 0, 0, 1.0, 0.0, ""
    for path in _coco_images():
        img = cv2.imread(str(path))
        fast = detector.detect(img)
        ref = reference_detector.detect(img)  # threshold 0.5
        assert ref, f"reference found nothing in {path.name}"
        for rank, r in enumerate(ref[:10]):
            cands = [f for f in fast if f.label == r.label]
            assert cands, f"{path.name}: no fast-path {r.label} for reference {r}"
            best = max(cands, key=lambda f: _iou(f, r))
            iou, ds = _iou(best, r), abs(best.score - r.score)
            ok = iou > 0.9 and ds < 0.05
            assert iou > 0.5, f"{path.name}: {r.label} IoU {iou:.3f} score {best.score:.3f} vs {r.score:.3f}"
            if rank == 0:
                assert ok, f"{path.name}: top detection {r.label} IoU {iou:.3f} score {best.score:.3f} vs {r.score:.3f}"
            matched += 1
            strict_ok += ok
            min_iou = min(min_iou, iou)
            if ds > max_dscore:
                max_dscore, worst = ds, f"{path.name} {r.label} {best.score:.3f} vs {r.score:.3f}"
    print(f"\nparity [{CHECKPOINT}]: {matched} matched detections, strict (IoU>0.9 & dScore<0.05) "
          f"{strict_ok}/{matched}, min IoU {min_iou:.4f}, max |score diff| {max_dscore:.4f} ({worst})")
    assert strict_ok >= 0.9 * matched
