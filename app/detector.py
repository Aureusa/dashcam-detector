"""RT-DETR object detector wrapper (Hugging Face ``transformers``) with a fast GPU preprocessing path.

The HF image processor is kept as a reference path (``use_reference_preprocess=True``) for
parity testing; the default path reimplements the processor's resize/rescale/normalize with
tensor ops on the inference device, which is much faster for real-time video.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from app.config import ModelConfig

__all__ = ["Detection", "RTDetrDetector"]

logger = logging.getLogger(__name__)

# COCO spellings vs. the VOC-style spellings used by the PekingU RT-DETR checkpoints' id2label
# (e.g. "motorbike", "sofa"). Used in both directions, only when the requested spelling is not
# itself a label of the loaded checkpoint.
_LABEL_ALIASES: tuple[tuple[str, str], ...] = (
    ("motorcycle", "motorbike"),
    ("airplane", "aeroplane"),
    ("couch", "sofa"),
    ("potted plant", "pottedplant"),
    ("dining table", "diningtable"),
    ("tv", "tvmonitor"),
)


@dataclass(frozen=True)
class Detection:
    """One detected object. Coordinates are pixels in the original frame, clamped to its bounds."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str

    def to_dict(self) -> dict:
        """JSON-friendly dict with keys x1, y1, x2, y2, score, class_id, label."""
        return {
            "x1": round(float(self.x1), 1),
            "y1": round(float(self.y1), 1),
            "x2": round(float(self.x2), 1),
            "y2": round(float(self.y2), 1),
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "label": str(self.label),
        }


@dataclass
class _DetOutputs:
    """Minimal outputs-like object accepted by ``post_process_object_detection``."""

    logits: torch.Tensor
    pred_boxes: torch.Tensor


def _resolve_device(name: str) -> torch.device:
    """Resolve ``auto|cpu|cuda|cuda:N`` to a torch.device, with clear errors/warnings."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning(
            "CUDA is not available; running RT-DETR on CPU. Expect only a few FPS. "
            "Check the NVIDIA driver and that a CUDA build of torch is installed."
        )
        return torch.device("cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"model.device={name!r} requested but CUDA is not available (use 'auto' or 'cpu')")
    return dev


def _check_threshold(t: Any) -> float:
    if isinstance(t, bool) or not isinstance(t, (int, float, np.floating, np.integer)):
        raise ValueError(f"threshold must be a real number in [0, 1], got {t!r}")
    v = float(t)
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"threshold must be a real number in [0, 1], got {t!r}")
    return v


class RTDetrDetector:
    """RT-DETR / RT-DETRv2 detector.

    Thread-safety: ``set_threshold``/``set_class_filter`` may be called from any thread while
    ``detect`` runs in another; ``detect``/``warmup`` are serialized by an internal lock and may
    be called from a thread other than the one that constructed the detector.
    """

    def __init__(self, cfg: ModelConfig, use_reference_preprocess: bool = False) -> None:
        self._cfg = cfg
        self._use_reference = use_reference_preprocess
        self._device = _resolve_device(cfg.device)
        self._fp16 = bool(cfg.fp16) and self._device.type == "cuda"
        if cfg.fp16 and not self._fp16:
            logger.info("fp16 requested but device is %s; running in fp32", self._device)

        t0 = time.perf_counter()
        self._processor = AutoImageProcessor.from_pretrained(cfg.checkpoint)
        model = AutoModelForObjectDetection.from_pretrained(cfg.checkpoint)
        model = model.to(self._device).eval()
        self._model = model
        self._forward = model
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        logger.info(
            "Loaded %s (%s) on %s in %.1f s (fp16 autocast=%s)",
            cfg.checkpoint, type(model).__name__, self._device, time.perf_counter() - t0, self._fp16,
        )

        self._labels: dict[int, str] = {int(k): str(v) for k, v in model.config.id2label.items()}
        self._num_classes = len(self._labels)
        self._use_focal_loss = bool(getattr(model.config, "use_focal_loss", True))
        self._lookup = self._build_label_lookup()

        # --- processor config -> fast preprocessing parameters
        p = self._processor
        size = getattr(p, "size", None) or {}
        size = dict(size) if not isinstance(size, dict) else size
        self._do_resize = bool(getattr(p, "do_resize", True))
        self._do_rescale = bool(getattr(p, "do_rescale", True))
        self._rescale_factor = float(getattr(p, "rescale_factor", 1.0 / 255.0))
        self._do_normalize = bool(getattr(p, "do_normalize", False))
        self._do_pad = bool(getattr(p, "do_pad", False))
        logger.info(
            "Processor %s: size=%s do_resize=%s resample=%s do_rescale=%s rescale_factor=%.6f "
            "do_normalize=%s image_mean=%s image_std=%s do_pad=%s",
            type(p).__name__, size, self._do_resize, getattr(p, "resample", None), self._do_rescale,
            self._rescale_factor, self._do_normalize, getattr(p, "image_mean", None),
            getattr(p, "image_std", None), self._do_pad,
        )
        proc_hw = (size.get("height"), size.get("width"))
        if not (proc_hw[0] and proc_hw[1]):
            logger.warning("Processor size %s is not a fixed height/width; using input_size=%d", size, cfg.input_size)
        elif proc_hw != (cfg.input_size, cfg.input_size):
            logger.warning(
                "model.input_size=%d differs from processor size %s; using %dx%d for both paths",
                cfg.input_size, size, cfg.input_size, cfg.input_size,
            )
        if self._do_pad:
            logger.warning("Processor has do_pad=True; fixed-size resize makes padding a no-op in the fast path")
        self._input_hw = (int(cfg.input_size), int(cfg.input_size))
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        if self._do_normalize:
            self._mean = torch.tensor(p.image_mean, dtype=torch.float32, device=self._device).view(1, 3, 1, 1)
            self._std = torch.tensor(p.image_std, dtype=torch.float32, device=self._device).view(1, 3, 1, 1)

        self._pinned: torch.Tensor | None = None  # reusable pinned host buffer for frame uploads

        # --- live-tunable settings (guarded by _settings_lock)
        self._settings_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._threshold = _check_threshold(cfg.score_threshold)
        self._class_filter: list[str] = []
        self._class_mask: torch.Tensor | None = None  # bool [num_classes] on device, None = all

        logger.info("Available labels (%d): %s", self._num_classes, ", ".join(self._labels.values()))
        resolved, unknown = self._resolve_names(cfg.classes)
        if unknown:
            logger.warning(
                "Ignoring configured class name(s) not in the checkpoint's id2label: %s", ", ".join(unknown)
            )
        if cfg.classes and not resolved:
            logger.warning("None of the configured classes exist; falling back to ALL classes")
        self._apply_filter(resolved)
        logger.info("Class filter: %s", ", ".join(self._class_filter) if self._class_filter else "ALL")

        if cfg.compile:
            if self._device.type != "cuda":
                logger.warning("model.compile=true ignored on CPU")
            else:
                logger.warning(
                    "torch.compile(mode='reduce-overhead') enabled (experimental): the first warmup "
                    "passes take ~20-60 s while compiling/capturing CUDA graphs"
                )
                self._forward = torch.compile(model, mode="reduce-overhead", dynamic=False)
        self._compiled = self._forward is not model
        self._warmed_threads: set[int] = set()

        self._last_timings: dict[str, float] = {"preprocess": 0.0, "forward": 0.0, "postprocess": 0.0, "total": 0.0}

    # ------------------------------------------------------------------ properties

    @property
    def threshold(self) -> float:
        """Current score threshold."""
        with self._settings_lock:
            return self._threshold

    @property
    def class_filter(self) -> list[str]:
        """Active class filter in canonical id2label spellings; [] means all classes."""
        with self._settings_lock:
            return list(self._class_filter)

    @property
    def labels(self) -> dict[int, str]:
        """The checkpoint's id2label mapping (copy)."""
        return dict(self._labels)

    @property
    def device(self) -> torch.device:
        """Inference device."""
        return self._device

    @property
    def model_name(self) -> str:
        """Checkpoint id."""
        return self._cfg.checkpoint

    @property
    def last_timings(self) -> dict[str, float]:
        """Stage timings (ms) of the last ``detect()`` call."""
        with self._settings_lock:
            return dict(self._last_timings)

    # ------------------------------------------------------------------ settings

    def set_threshold(self, t: float) -> None:
        """Set the score threshold (thread-safe). Raises ValueError unless a real number in [0, 1]."""
        v = _check_threshold(t)
        with self._settings_lock:
            self._threshold = v
        logger.info("Score threshold set to %.3f", v)

    def set_class_filter(self, names: list[str]) -> None:
        """Restrict detections to ``names`` (case-insensitive; [] = all). Thread-safe.

        Raises:
            ValueError: if any name is unknown (the current filter is left unchanged).
        """
        if isinstance(names, str) or not isinstance(names, (list, tuple)):
            raise ValueError(f"class filter must be a list of class names, got {names!r}")
        if not all(isinstance(n, str) for n in names):
            raise ValueError("class filter must contain only strings")
        resolved, unknown = self._resolve_names(list(names))
        if unknown:
            raise ValueError(f"unknown class name(s): {', '.join(unknown)}")
        self._apply_filter(resolved)
        logger.info("Class filter set to %s", ", ".join(resolved) if resolved else "ALL")

    def _build_label_lookup(self) -> dict[str, int]:
        lookup = {name.strip().lower(): cid for cid, name in self._labels.items()}
        for a, b in _LABEL_ALIASES:
            if a in lookup and b not in lookup:
                lookup[b] = lookup[a]
            elif b in lookup and a not in lookup:
                lookup[a] = lookup[b]
        return lookup

    def _resolve_names(self, names: list[str]) -> tuple[list[str], list[str]]:
        """Map names to canonical labels. Returns (resolved unique canonical names, unknown names)."""
        resolved: list[str] = []
        unknown: list[str] = []
        for n in names:
            cid = self._lookup.get(str(n).strip().lower())
            if cid is None:
                unknown.append(str(n))
                continue
            canon = self._labels[cid]
            if canon.lower() != str(n).strip().lower():
                logger.debug("Class name %r resolved to checkpoint label %r", n, canon)
            if canon not in resolved:
                resolved.append(canon)
        return resolved, unknown

    def _apply_filter(self, canonical: list[str]) -> None:
        mask: torch.Tensor | None = None
        if canonical:
            ids = [cid for cid, name in self._labels.items() if name in canonical]
            mask = torch.zeros(self._num_classes, dtype=torch.bool, device=self._device)
            mask[ids] = True
        with self._settings_lock:
            self._class_filter = list(canonical)
            self._class_mask = mask

    # ------------------------------------------------------------------ inference

    def warmup(self, height: int, width: int) -> None:
        """Run ``warmup_iters`` full detect() passes on a dummy frame of the given size."""
        iters = int(self._cfg.warmup_iters)
        # CUDA/cuDNN handles are per thread: the first forward in a new thread costs ~0.5 s,
        # so warmup must run in the thread that will call detect().
        self._warmed_threads.add(threading.get_ident())
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 256, size=(int(height), int(width), 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(iters):
            self.detect(dummy)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        logger.info("Warmup: %d iters at %dx%d in %.2f s", iters, width, height, time.perf_counter() - t0)

    def _sync(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _upload(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Copy the HxWx3 uint8 frame to the device (via a reusable pinned buffer on CUDA)."""
        if self._device.type != "cuda":
            return torch.from_numpy(np.ascontiguousarray(frame_bgr))
        if self._pinned is None or tuple(self._pinned.shape) != frame_bgr.shape:
            self._pinned = torch.empty(frame_bgr.shape, dtype=torch.uint8, pin_memory=True)
        # Safe to reuse: the previous detect() ended with a blocking device->host copy.
        self._pinned.numpy()[...] = frame_bgr
        return self._pinned.to(self._device, non_blocking=True)

    def _preprocess_fast(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """GPU reimplementation of the HF processor: BGR->RGB, resize (bilinear, antialias, uint8-rounded), rescale[, normalize]."""
        t = self._upload(frame_bgr)
        t = t.permute(2, 0, 1).flip(0).unsqueeze(0).float()  # 1x3xHxW RGB, 0..255
        if self._do_resize and tuple(t.shape[-2:]) != self._input_hw:
            t = F.interpolate(t, size=self._input_hw, mode="bilinear", align_corners=False, antialias=True)
            # The HF processor resizes uint8 images and gets uint8 back; rounding reproduces that
            # quantization (bit-exact for most pixels) and noticeably tightens score parity.
            t = t.round_().clamp_(0.0, 255.0)
        if self._do_rescale:
            t = t.mul_(self._rescale_factor)
        if self._do_normalize and self._mean is not None and self._std is not None:
            t = t.sub_(self._mean).div_(self._std)
        return t.contiguous()

    def _preprocess_reference(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Reference preprocessing through the Hugging Face image processor (CPU)."""
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h, w = self._input_hw
        inputs = self._processor(images=rgb, size={"height": h, "width": w}, return_tensors="pt")
        return inputs["pixel_values"].to(self._device)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect objects in an HxWx3 uint8 BGR frame. Returns detections sorted by score (desc)."""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR ndarray, got {getattr(frame_bgr, 'shape', type(frame_bgr))}")
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame, got {frame_bgr.dtype}")
        h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        tid = threading.get_ident()
        if tid not in self._warmed_threads:
            self._warmed_threads.add(tid)
            logger.warning(
                "first detect() on thread %r without warmup() in that thread; expect a one-off "
                "latency spike (per-thread CUDA/cuDNN init). Call warmup() from the inference thread.",
                threading.current_thread().name,
            )
        with self._settings_lock:
            threshold = self._threshold
            mask = self._class_mask

        with self._infer_lock, torch.inference_mode():
            t0 = time.perf_counter()
            if self._use_reference:
                pixel_values = self._preprocess_reference(frame_bgr)
            else:
                pixel_values = self._preprocess_fast(frame_bgr)
            self._sync()
            t1 = time.perf_counter()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._fp16):
                outputs = self._forward(pixel_values=pixel_values)
            logits = outputs.logits.float()
            boxes = outputs.pred_boxes.float()
            if self._compiled:
                # CUDA-graph output buffers are overwritten by the next replay.
                logits, boxes = logits.clone(), boxes.clone()
            self._sync()
            t2 = time.perf_counter()

            if mask is not None:
                # Mask filtered-out classes before top-K so they cannot starve allowed classes.
                logits = logits.masked_fill(~mask.view(1, 1, -1), float("-inf"))
            res = self._processor.post_process_object_detection(
                _DetOutputs(logits=logits, pred_boxes=boxes),
                threshold=threshold,
                target_sizes=[(h, w)],
                use_focal_loss=self._use_focal_loss,
            )[0]
            scores, labels, bxs = res["scores"], res["labels"], res["boxes"]
            bxs = torch.stack(
                (bxs[:, 0].clamp(0, w), bxs[:, 1].clamp(0, h), bxs[:, 2].clamp(0, w), bxs[:, 3].clamp(0, h)), dim=1
            )
            packed = torch.cat((bxs, scores[:, None], labels[:, None].to(bxs.dtype)), dim=1)
            order = torch.argsort(packed[:, 4], descending=True)
            rows = packed[order].cpu().tolist()  # single device->host transfer
            t3 = time.perf_counter()

        dets = [
            Detection(x1=r[0], y1=r[1], x2=r[2], y2=r[3], score=r[4], class_id=int(r[5]),
                      label=self._labels.get(int(r[5]), str(int(r[5]))))
            for r in rows
        ]
        timings = {
            "preprocess": (t1 - t0) * 1e3,
            "forward": (t2 - t1) * 1e3,
            "postprocess": (t3 - t2) * 1e3,
            "total": (time.perf_counter() - t0) * 1e3,
        }
        with self._settings_lock:
            self._last_timings = timings
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "detect %dx%d: %d dets | pre %.2f fwd %.2f post %.2f total %.2f ms",
                w, h, len(dets), timings["preprocess"], timings["forward"], timings["postprocess"], timings["total"],
            )
        return dets
