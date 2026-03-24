"""Engine tuning from rep_counter.toml (see FLEXIBLE_REP_COUNTER_CONFIG or cwd/parents)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from ipaddress import ip_address
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_env_path = _ROOT / ".env"
load_dotenv(_env_path, override=True)


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
    return None


def _load_rep_counter_cfg() -> dict:
    src = _resolve_rep_counter_toml()
    if src is None:
        return {}
    try:
        with src.open("rb") as f:
            return _toml_load_file(f)
    except OSError:
        return {}


_REP_COUNTER_CFG = _load_rep_counter_cfg()


def _toml_val(*keys: str) -> Any:
    cur: Any = _REP_COUNTER_CFG
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _toml_float(*keys: str, default: float) -> float:
    v = _toml_val(*keys)
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _toml_int(*keys: str, default: int) -> int:
    v = _toml_val(*keys)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    return default


def _toml_bool(*keys: str, default: bool) -> bool:
    v = _toml_val(*keys)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "no", "")
    return default


def _vm_netloc_host_port(hostname: str, port: int) -> str:
    try:
        if ip_address(hostname).version == 6:
            return f"[{hostname}]:{port}"
    except ValueError:
        pass
    return f"{hostname}:{port}"


def _normalize_vm_base_url(url: str) -> str:
    s = url.strip().rstrip("/")
    if not s:
        return s
    p = urlparse(s)
    if p.scheme:
        if p.scheme == "http" and p.hostname is not None and p.port is None:
            netloc = _vm_netloc_host_port(p.hostname, 8000)
            out = urlunparse(("http", netloc, p.path, p.params, p.query, p.fragment))
            return out.rstrip("/") or out
        return s
    synthetic = f"http://{s}"
    p2 = urlparse(synthetic)
    if p2.port is not None:
        return synthetic
    if p2.hostname is None:
        return synthetic
    netloc = _vm_netloc_host_port(p2.hostname, 8000)
    out = urlunparse(("http", netloc, p2.path, p2.params, p2.query, p2.fragment))
    return out.rstrip("/") or out


def _vm_url_from_toml() -> Optional[str]:
    t = _toml_val("vm", "direct_url")
    if isinstance(t, str):
        s = t.strip()
        if s:
            return _normalize_vm_base_url(s)
    return None


VM_BASE_URL = _vm_url_from_toml()
VM_TIMEOUT_SEC = _toml_float("vm", "timeout_sec", default=5.0)
VM_HEALTH_TIMEOUT_SEC = _toml_float("vm", "health_timeout_sec", default=5.0)
PREDICT_RESIZE_WIDTH = _toml_int("predict", "resize_width", default=0)
PREDICT_JPEG_QUALITY = _toml_int("predict", "jpeg_quality", default=85)
PREDICT_VALIDATE_RESPONSE = _toml_bool("predict", "validate_response", default=True)

LOG_LEVEL = str(_toml_val("app", "log_level") or "DEBUG").strip().upper() or "DEBUG"
DEBUG_CONSOLE_ENABLED = LOG_LEVEL == "DEBUG"

DEFAULT_HYSTERESIS = _toml_float("rep", "hysteresis", default=5.0)
DEFAULT_MIN_PEAK_DISTANCE = _toml_int("rep", "min_peak_distance", default=5)
DEFAULT_SMOOTHING_FACTOR = _toml_float("rep", "smoothing_factor", default=0.45)
DEFAULT_PEAK_MARGIN = _toml_float("rep", "peak_margin", default=15.0)
DEFAULT_VALLEY_MARGIN = _toml_float("rep", "valley_margin", default=15.0)
DEFAULT_MIN_RANGE_GATE = _toml_float("rep", "min_range_gate", default=15.0)
DEFAULT_RANGE_WINDOW_FRAMES = _toml_int("rep", "range_window_frames", default=90)
DEFAULT_RANGE_MIN_SAMPLES = _toml_int("rep", "range_min_samples", default=12)
DEFAULT_ANGLE_DELTA_DEADBAND = _toml_float("rep", "angle_delta_deadband", default=0.0)
DEFAULT_CALIBRATION_REPS = _toml_int("rep", "calibration_reps", default=3)
DEFAULT_CALIBRATION_CERTAINTY = _toml_float("rep", "calibration_certainty", default=0.5)
DEFAULT_CALIBRATION_FORCE_EXTRA_REPS = _toml_int("rep", "calibration_force_extra_reps", default=2)
DEFAULT_MIN_REP_INTERVAL_MS = _toml_float("rep", "min_interval_ms", default=400.0)

ANGLE_SELECTION_MIN_SEC = _toml_float("angle_selection", "min_sec", default=7.0)
ANGLE_SELECTION_MIN_FRAMES = _toml_int("angle_selection", "min_frames", default=70)
ANGLE_SELECTION_MAX_BUFFER_FRAMES = _toml_int("angle_selection", "max_buffer_frames", default=400)
ANGLE_SELECTION_RETRY_INTERVAL_SEC = _toml_float("angle_selection", "retry_interval_sec", default=2.0)
ANGLE_SELECTION_DOMINANCE_FRACTION = _toml_float("angle_selection", "dominance_fraction", default=2.0 / 3.0)
ANGLE_SELECTION_MIN_LEADING_REPS = _toml_int("angle_selection", "min_leading_reps", default=2)
ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES = _toml_int(
    "angle_selection", "dominance_streak_frames", default=36
)
ANGLE_SELECTION_VARIANCE_FALLBACK_SEC = _toml_float(
    "angle_selection", "variance_fallback_sec", default=14.0
)
ANGLE_SELECTION_MIN_VARIANCE = _toml_float("angle_selection", "min_variance", default=6.0)
ANGLE_SELECTION_MIN_RANGE_DEG = _toml_float("angle_selection", "min_range_deg", default=16.0)
ANGLE_SELECTION_SECOND_BEST_RATIO = _toml_float("angle_selection", "second_best_ratio", default=1.15)
ANGLE_SELECTION_MIN_ACTIVE_WINDOWS = _toml_int("angle_selection", "min_active_windows", default=4)
ANGLE_SELECTION_SMOOTH_WINDOW = _toml_int("angle_selection", "smooth_window", default=5)


def get_angle_selection_joint_thresholds(angle_key: str) -> dict[str, float]:
    """Per-joint gates from ``[angle_selection.joints.<KEY>]``, else ``[angle_selection]`` globals."""
    joint = _toml_val("angle_selection", "joints", angle_key)
    if not isinstance(joint, dict):
        joint = {}

    def _local(key: str, global_val: float) -> float:
        raw = joint.get(key)
        if raw is not None and isinstance(raw, (int, float)):
            return float(raw)
        return global_val

    return {
        "min_variance": _local("min_variance", ANGLE_SELECTION_MIN_VARIANCE),
        "min_range_deg": _local("min_range_deg", ANGLE_SELECTION_MIN_RANGE_DEG),
        "second_best_ratio": _local("second_best_ratio", ANGLE_SELECTION_SECOND_BEST_RATIO),
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
