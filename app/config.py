"""Typed application configuration: dataclasses, YAML loading, validation and CLI overrides.

The YAML file (``config.yaml``) maps 1:1 onto :class:`AppConfig`. Unknown keys are
rejected so that typos surface immediately instead of being silently ignored.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AppConfig",
    "CameraConfig",
    "ConfigError",
    "LoggingConfig",
    "ModelConfig",
    "RenderConfig",
    "ServerConfig",
    "add_cli_args",
    "apply_overrides",
    "load_config",
    "setup_logging",
    "validate_config",
]

VALID_BACKENDS = ("auto", "v4l2", "ffmpeg", "gstreamer")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def _default_classes() -> list[str]:
    """Driving-relevant COCO subset used as the default class filter."""
    return ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]


class ConfigError(ValueError):
    """Raised for invalid configuration files or values."""


@dataclass
class CameraConfig:
    """Video source settings (device, file or URL)."""

    source: str | int = "/dev/video0"  # digit-only strings are converted to int
    backend: str = "auto"  # auto | v4l2 | ffmpeg | gstreamer
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"  # 4 chars, or "" to not set
    buffer_size: int = 1
    loop_file: bool = True
    pace_file: bool = True
    reconnect_delay_s: float = 2.0
    max_consecutive_failures: int = 30
    first_frame_timeout_s: float = 10.0


@dataclass
class ModelConfig:
    """Detector settings."""

    checkpoint: str = "PekingU/rtdetr_v2_r50vd"  # 53.4 COCO AP; ~19 ms/frame full pipeline on RTX 5060 Ti
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    fp16: bool = True
    input_size: int = 640
    score_threshold: float = 0.5
    classes: list[str] = field(default_factory=_default_classes)  # [] = all classes
    compile: bool = False
    warmup_iters: int = 10


@dataclass
class RenderConfig:
    """Annotation drawing settings."""

    line_thickness: int = 2
    font_scale: float = 0.5
    draw_hud: bool = True
    output_width: int = 0  # 0 = keep capture width


@dataclass
class ServerConfig:
    """Flask server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 80
    max_stream_fps: int = 30


@dataclass
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"


@dataclass
class AppConfig:
    """Top-level application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


_SECTIONS: dict[str, type] = {
    "camera": CameraConfig,
    "model": ModelConfig,
    "render": RenderConfig,
    "server": ServerConfig,
    "logging": LoggingConfig,
}


# --------------------------------------------------------------------------- helpers


def _normalize_source(source: Any) -> str | int:
    """Convert digit-only strings (e.g. ``"0"``) to an int device index."""
    if isinstance(source, bool):
        raise ConfigError("camera.source must be a string or int, got bool")
    if isinstance(source, int):
        return source
    if isinstance(source, str):
        s = source.strip()
        if s.isdigit():
            return int(s)
        return s
    raise ConfigError(f"camera.source must be a string or int, got {type(source).__name__}")


def _coerce(section: str, name: str, value: Any, default: Any) -> Any:
    """Coerce a YAML value to the type of the dataclass default, with clear errors."""
    key = f"{section}.{name}"
    if section == "camera" and name == "source":
        return _normalize_source(value)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{key} must be a boolean (true/false), got {value!r}")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise ConfigError(f"{key} must be an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{key} must be a string, got {value!r}")
        return value
    if isinstance(default, list):
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{key} must be a list of strings, got {value!r}")
        return list(value)
    return value


def _build_section(section: str, cls: type, data: Any) -> Any:
    """Instantiate one section dataclass from a mapping, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"section '{section}' must be a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in section '{section}': {', '.join(map(str, unknown))}; "
            f"valid keys: {', '.join(fields)}"
        )
    defaults = cls()
    kwargs = {
        name: _coerce(section, name, value, getattr(defaults, name)) for name, value in data.items()
    }
    return cls(**kwargs)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _valid_device(device: str) -> bool:
    if device in ("auto", "cpu", "cuda"):
        return True
    if device.startswith("cuda:"):
        return device[5:].isdigit()
    return False


# --------------------------------------------------------------------------- public API


def validate_config(cfg: AppConfig) -> None:
    """Validate all values; raise :class:`ConfigError` with a clear message on the first problem."""
    c = cfg.camera
    _check(
        _is_int(c.source) or (isinstance(c.source, str) and c.source.strip() != ""),
        "camera.source must be a non-empty string or an int device index",
    )
    _check(not _is_int(c.source) or c.source >= 0, "camera.source device index must be >= 0")
    _check(c.backend in VALID_BACKENDS, f"camera.backend must be one of {VALID_BACKENDS}, got {c.backend!r}")
    for name in ("width", "height", "fps"):
        v = getattr(c, name)
        _check(_is_int(v) and v > 0, f"camera.{name} must be a positive integer, got {v!r}")
    _check(
        isinstance(c.fourcc, str) and len(c.fourcc) in (0, 4),
        f"camera.fourcc must be exactly 4 characters or empty, got {c.fourcc!r}",
    )
    _check(_is_int(c.buffer_size) and c.buffer_size >= 1, f"camera.buffer_size must be >= 1, got {c.buffer_size!r}")
    _check(isinstance(c.loop_file, bool), "camera.loop_file must be a boolean")
    _check(isinstance(c.pace_file, bool), "camera.pace_file must be a boolean")
    _check(
        _is_num(c.reconnect_delay_s) and c.reconnect_delay_s >= 0,
        f"camera.reconnect_delay_s must be >= 0, got {c.reconnect_delay_s!r}",
    )
    _check(
        _is_int(c.max_consecutive_failures) and c.max_consecutive_failures >= 1,
        f"camera.max_consecutive_failures must be a positive integer, got {c.max_consecutive_failures!r}",
    )
    _check(
        _is_num(c.first_frame_timeout_s) and c.first_frame_timeout_s > 0,
        f"camera.first_frame_timeout_s must be > 0, got {c.first_frame_timeout_s!r}",
    )

    m = cfg.model
    _check(isinstance(m.checkpoint, str) and m.checkpoint.strip() != "", "model.checkpoint must be a non-empty string")
    _check(
        isinstance(m.device, str) and _valid_device(m.device),
        f"model.device must be one of auto | cpu | cuda | cuda:N, got {m.device!r}",
    )
    _check(isinstance(m.fp16, bool), "model.fp16 must be a boolean")
    _check(_is_int(m.input_size) and m.input_size > 0, f"model.input_size must be a positive integer, got {m.input_size!r}")
    _check(
        _is_num(m.score_threshold) and 0.0 <= m.score_threshold <= 1.0,
        f"model.score_threshold must be in [0, 1], got {m.score_threshold!r}",
    )
    _check(
        isinstance(m.classes, list) and all(isinstance(x, str) and x.strip() for x in m.classes),
        "model.classes must be a list of non-empty strings ([] = all classes)",
    )
    _check(isinstance(m.compile, bool), "model.compile must be a boolean")
    _check(_is_int(m.warmup_iters) and m.warmup_iters >= 0, f"model.warmup_iters must be >= 0, got {m.warmup_iters!r}")

    r = cfg.render
    _check(_is_int(r.line_thickness) and r.line_thickness > 0, f"render.line_thickness must be a positive integer, got {r.line_thickness!r}")
    _check(_is_num(r.font_scale) and r.font_scale > 0, f"render.font_scale must be > 0, got {r.font_scale!r}")
    _check(isinstance(r.draw_hud, bool), "render.draw_hud must be a boolean")
    _check(_is_int(r.output_width) and r.output_width >= 0, f"render.output_width must be >= 0, got {r.output_width!r}")

    s = cfg.server
    _check(isinstance(s.host, str) and s.host.strip() != "", "server.host must be a non-empty string")
    _check(_is_int(s.port) and 1 <= s.port <= 65535, f"server.port must be in 1..65535, got {s.port!r}")
    _check(_is_int(s.jpeg_quality) and 1 <= s.jpeg_quality <= 100, f"server.jpeg_quality must be in 1..100, got {s.jpeg_quality!r}")
    _check(_is_num(s.max_stream_fps) and s.max_stream_fps > 0, f"server.max_stream_fps must be > 0, got {s.max_stream_fps!r}")

    lv = cfg.logging.level
    _check(
        isinstance(lv, str) and lv.upper() in VALID_LOG_LEVELS,
        f"logging.level must be one of {VALID_LOG_LEVELS}, got {lv!r}",
    )


def load_config(path: str | os.PathLike | None) -> AppConfig:
    """Load and validate a YAML config file. ``None`` returns validated defaults.

    Raises:
        ConfigError: if the file is missing, not valid YAML, contains unknown keys
            or invalid values.
    """
    if path is None:
        cfg = AppConfig()
        validate_config(cfg)
        return cfg
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {p} must be a mapping")
    unknown = sorted(set(data) - set(_SECTIONS))
    if unknown:
        raise ConfigError(
            f"unknown section(s) in {p}: {', '.join(map(str, unknown))}; valid: {', '.join(_SECTIONS)}"
        )
    sections = {name: _build_section(name, cls, data.get(name)) for name, cls in _SECTIONS.items()}
    cfg = AppConfig(**sections)
    validate_config(cfg)
    return cfg


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard config/override CLI options to ``parser`` (all default to None)."""
    parser.add_argument("--config", default="config.yaml", help="path to YAML config (default: config.yaml)")
    parser.add_argument("--source", default=None, help="camera source: index, /dev/videoN, file, rtsp:// or http:// URL")
    parser.add_argument("--checkpoint", default=None, help="Hugging Face checkpoint id or local path")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--host", default=None, help="server bind host")
    parser.add_argument("--port", type=int, default=None, help="server port")
    parser.add_argument("--threshold", type=float, default=None, help="score threshold in [0, 1]")
    parser.add_argument("--log-level", dest="log_level", default=None, help="DEBUG | INFO | WARNING | ERROR")


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply non-None CLI overrides onto ``cfg`` (in place), re-validate and return it."""
    mapping = {
        "source": ("camera", "source"),
        "checkpoint": ("model", "checkpoint"),
        "device": ("model", "device"),
        "host": ("server", "host"),
        "port": ("server", "port"),
        "threshold": ("model", "score_threshold"),
        "log_level": ("logging", "level"),
    }
    for arg_name, (section, attr) in mapping.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        if arg_name == "source":
            value = _normalize_source(value)
        elif arg_name == "log_level":
            value = str(value).upper()
        setattr(getattr(cfg, section), attr, value)
    validate_config(cfg)
    return cfg


def setup_logging(level: str) -> None:
    """Configure root logging with timestamps and thread names."""
    lv = str(level).upper()
    if lv not in VALID_LOG_LEVELS:
        raise ConfigError(f"invalid log level {level!r}; valid: {VALID_LOG_LEVELS}")
    logging.basicConfig(level=getattr(logging, lv), format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    # Hugging Face Hub HTTP chatter (one line per HEAD request) is noise unless debugging.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING if lv != "DEBUG" else logging.DEBUG)
