"""Engine tuning from rep_counter.toml, with environment overrides.

Loads ``rep_counter.toml`` (cwd or parents), optional ``FLEXIBLE_REP_COUNTER_CONFIG``,
then bundled ``default_rep_counter.toml``. Safe to import without a VM URL
(unlike the local app shim in ``app.config``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_env_path = _ROOT / ".env"

try:
    from dotenv import load_dotenv

    load_dotenv(_env_path, override=True)
except ImportError:
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"')


def _toml_load_file(f: Any) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.load(f)
    import tomli

    return tomli.load(f)


def _resolve_rep_counter_toml() -> Any | None:
    env = os.environ.get("FLEXIBLE_REP_COUNTER_CONFIG", "").strip()
    if env:
        p = Path(env).expanduser()
        return p.resolve() if p.is_file() else None
    here = Path.cwd()
    for d in [here, *here.parents]:
        cand = d / "rep_counter.toml"
        if cand.is_file():
            return cand.resolve()
    try:
        import importlib.resources as ir

        ref = ir.files("flexible_rep_counter") / "default_rep_counter.toml"
        if ref.is_file():
            return ref
    except (OSError, TypeError, ModuleNotFoundError):
        pass
    return None


def _load_rep_counter_cfg() -> dict:
    src = _resolve_rep_counter_toml()
    if src is None:
        return {}
    try:
        if isinstance(src, Path):
            with src.open("rb") as f:
                return _toml_load_file(f)
        with src.open("rb") as f:
            return _toml_load_file(f)
    except OSError:
        return {}


_REP_COUNTER_CFG = _load_rep_counter_cfg()


def _env_val(key: str) -> Optional[str]:
    v = os.environ.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _toml_val(*keys: str) -> Any:
    cur: Any = _REP_COUNTER_CFG
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _pick_float(env_key: str, toml_path: tuple[str, ...], default: float) -> float:
    e = _env_val(env_key)
    if e is not None:
        return float(e)
    t = _toml_val(*toml_path)
    if isinstance(t, (int, float)):
        return float(t)
    return default


def _pick_int(env_key: str, toml_path: tuple[str, ...], default: int) -> int:
    e = _env_val(env_key)
    if e is not None:
        return int(float(e))
    t = _toml_val(*toml_path)
    if isinstance(t, bool):
        return int(t)
    if isinstance(t, (int, float)):
        return int(t)
    return default


def _pick_vm_url() -> Optional[str]:
    for k in ("YOLO_VM_DIRECT_URL", "YOLO_VM_TARGET_URL"):
        e = _env_val(k)
        if e is not None:
            return e
    t = _toml_val("vm", "direct_url")
    if isinstance(t, str):
        s = t.strip()
        if s:
            return s
    return None


YOLO_VM_TARGET_URL = _pick_vm_url()
VM_TIMEOUT_SEC = float(os.environ.get("VM_TIMEOUT_SEC", "5.0"))
VM_HEALTH_TIMEOUT_SEC = float(os.environ.get("VM_HEALTH_TIMEOUT_SEC", "5.0"))
PREDICT_RESIZE_WIDTH = int(os.environ.get("PREDICT_RESIZE_WIDTH", "0"))
PREDICT_JPEG_QUALITY = int(os.environ.get("PREDICT_JPEG_QUALITY", "85"))
PREDICT_VALIDATE_RESPONSE = os.environ.get("PREDICT_VALIDATE_RESPONSE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
DEBUG_CONSOLE_ENABLED = LOG_LEVEL == "DEBUG"

DEFAULT_HYSTERESIS = _pick_float("REP_HYSTERESIS", ("rep", "hysteresis"), 5.0)
DEFAULT_MIN_PEAK_DISTANCE = int(os.environ.get("REP_MIN_PEAK_DISTANCE", "5"))
DEFAULT_SMOOTHING_FACTOR = float(os.environ.get("REP_SMOOTHING_FACTOR", "0.45"))
DEFAULT_PEAK_MARGIN = _pick_float("REP_PEAK_MARGIN", ("rep", "peak_margin"), 15.0)
DEFAULT_VALLEY_MARGIN = _pick_float("REP_VALLEY_MARGIN", ("rep", "valley_margin"), 15.0)
DEFAULT_MIN_RANGE_GATE = float(os.environ.get("REP_MIN_RANGE_GATE", "15"))
DEFAULT_RANGE_WINDOW_FRAMES = int(os.environ.get("REP_RANGE_WINDOW_FRAMES", "90"))
DEFAULT_RANGE_MIN_SAMPLES = int(os.environ.get("REP_RANGE_MIN_SAMPLES", "12"))
DEFAULT_ANGLE_DELTA_DEADBAND = float(os.environ.get("REP_ANGLE_DELTA_DEADBAND", "0"))
DEFAULT_CALIBRATION_REPS = _pick_int("REP_CALIBRATION_REPS", ("rep", "calibration_reps"), 3)
DEFAULT_CALIBRATION_CERTAINTY = float(os.environ.get("REP_CALIBRATION_CERTAINTY", "0.5"))
DEFAULT_CALIBRATION_FORCE_EXTRA_REPS = int(os.environ.get("REP_CALIBRATION_FORCE_EXTRA_REPS", "2"))
DEFAULT_MIN_REP_INTERVAL_MS = _pick_float("REP_MIN_INTERVAL_MS", ("rep", "min_interval_ms"), 400.0)

ANGLE_SELECTION_MIN_SEC = float(os.environ.get("ANGLE_SELECTION_MIN_SEC", "7.0"))
ANGLE_SELECTION_MIN_FRAMES = int(os.environ.get("ANGLE_SELECTION_MIN_FRAMES", "70"))
ANGLE_SELECTION_MAX_BUFFER_FRAMES = int(os.environ.get("ANGLE_SELECTION_MAX_BUFFER_FRAMES", "400"))
ANGLE_SELECTION_RETRY_INTERVAL_SEC = float(os.environ.get("ANGLE_SELECTION_RETRY_INTERVAL_SEC", "2.0"))
ANGLE_SELECTION_DOMINANCE_FRACTION = float(
    os.environ.get("ANGLE_SELECTION_DOMINANCE_FRACTION", str(2.0 / 3.0))
)
ANGLE_SELECTION_MIN_LEADING_REPS = int(os.environ.get("ANGLE_SELECTION_MIN_LEADING_REPS", "2"))
ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES = int(
    os.environ.get("ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES", "36")
)
ANGLE_SELECTION_VARIANCE_FALLBACK_SEC = float(
    os.environ.get("ANGLE_SELECTION_VARIANCE_FALLBACK_SEC", "14")
)
ANGLE_SELECTION_MIN_VARIANCE = float(os.environ.get("ANGLE_SELECTION_MIN_VARIANCE", "6.0"))
ANGLE_SELECTION_MIN_RANGE_DEG = float(os.environ.get("ANGLE_SELECTION_MIN_RANGE_DEG", "16.0"))
ANGLE_SELECTION_SECOND_BEST_RATIO = float(os.environ.get("ANGLE_SELECTION_SECOND_BEST_RATIO", "1.15"))


def get_angle_selection_joint_thresholds(angle_key: str) -> dict[str, float]:
    """Per-joint gates: env ``ANGLE_SELECTION_*_<ANGLE_KEY>``, then ``rep_counter.toml``, then globals."""
    suf = f"_{angle_key}"
    joint = _toml_val("angle_selection", "joints", angle_key)
    if not isinstance(joint, dict):
        joint = {}

    def _one(env_name: str, joint_key: str, fallback: float) -> float:
        v = _env_val(f"{env_name}{suf}")
        if v is not None:
            return float(v)
        raw = joint.get(joint_key)
        if raw is not None and isinstance(raw, (int, float)):
            return float(raw)
        return fallback

    return {
        "min_variance": _one("ANGLE_SELECTION_MIN_VARIANCE", "min_variance", ANGLE_SELECTION_MIN_VARIANCE),
        "min_range_deg": _one("ANGLE_SELECTION_MIN_RANGE_DEG", "min_range_deg", ANGLE_SELECTION_MIN_RANGE_DEG),
        "second_best_ratio": _one(
            "ANGLE_SELECTION_SECOND_BEST_RATIO",
            "second_best_ratio",
            ANGLE_SELECTION_SECOND_BEST_RATIO,
        ),
    }


def get_default_tuning_params() -> dict:
    return {
        "hysteresis": DEFAULT_HYSTERESIS,
        "minPeakDistance": DEFAULT_MIN_PEAK_DISTANCE,
        "smoothingFactor": DEFAULT_SMOOTHING_FACTOR,
        "peakMargin": DEFAULT_PEAK_MARGIN,
        "valleyMargin": DEFAULT_VALLEY_MARGIN,
        "minRangeGate": DEFAULT_MIN_RANGE_GATE,
        "rangeWindowFrames": DEFAULT_RANGE_WINDOW_FRAMES,
        "rangeMinSamples": DEFAULT_RANGE_MIN_SAMPLES,
        "angleDeltaDeadband": DEFAULT_ANGLE_DELTA_DEADBAND,
        "calibrationReps": DEFAULT_CALIBRATION_REPS,
        "calibrationCertainty": DEFAULT_CALIBRATION_CERTAINTY,
        "calibrationForceExtraReps": DEFAULT_CALIBRATION_FORCE_EXTRA_REPS,
        "minRepIntervalMs": DEFAULT_MIN_REP_INTERVAL_MS,
    }
