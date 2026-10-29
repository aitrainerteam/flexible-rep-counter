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
# Fixed default; use `python main.py --no-validate-response` to skip JSON shape checks.
PREDICT_VALIDATE_RESPONSE = True

LOG_LEVEL = str(_toml_val("app", "log_level") or "DEBUG").strip().upper() or "DEBUG"
DEBUG_CONSOLE_ENABLED = LOG_LEVEL == "DEBUG"

DEFAULT_HYSTERESIS = _toml_float("rep", "hysteresis", default=5.0)
DEFAULT_MIN_PEAK_DISTANCE = _toml_int("rep", "min_peak_distance", default=5)
# EMA on angle stream; not exposed in TOML (tune with hysteresis / margins).
DEFAULT_SMOOTHING_FACTOR = 0.70
DEFAULT_PEAK_MARGIN_PCT = _toml_float("rep", "peak_margin_pct", default=0.50)
DEFAULT_VALLEY_MARGIN_PCT = _toml_float("rep", "valley_margin_pct", default=0.50)
DEFAULT_MIN_RANGE_GATE = _toml_float("rep", "min_range_gate", default=15.0)
DEFAULT_RANGE_WINDOW_FRAMES = _toml_int("rep", "range_window_frames", default=90)
DEFAULT_RANGE_MIN_SAMPLES = _toml_int("rep", "range_min_samples", default=12)
DEFAULT_ANGLE_DELTA_DEADBAND = _toml_float("rep", "angle_delta_deadband", default=0.0)
DEFAULT_CALIBRATION_REPS = _toml_int("rep", "calibration_reps", default=3)
DEFAULT_CALIBRATION_CERTAINTY = _toml_float("rep", "calibration_certainty", default=0.5)
DEFAULT_CALIBRATION_FORCE_EXTRA_REPS = _toml_int("rep", "calibration_force_extra_reps", default=2)
DEFAULT_MIN_REP_INTERVAL_MS = _toml_float("rep", "min_interval_ms", default=300.0)

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
ANGLE_SELECTION_REEVALUATE_EVERY_SEC = _toml_float(
    "angle_selection", "reevaluate_every_sec", default=0.75
)
ANGLE_SELECTION_SWITCH_MIN_SEC = _toml_float(
    "angle_selection", "switch_min_sec", default=1.5
)
ANGLE_SELECTION_SWITCH_MIN_REPS = _toml_int(
    "angle_selection", "switch_min_reps", default=10
)
ANGLE_SELECTION_SWITCH_VARIANCE_RATIO = _toml_float(
    "angle_selection", "switch_variance_ratio", default=1.2
)
ANGLE_SELECTION_MIN_VARIANCE = _toml_float("angle_selection", "min_variance", default=6.0)
ANGLE_SELECTION_MIN_RANGE_DEG = _toml_float("angle_selection", "min_range_deg", default=16.0)
ANGLE_SELECTION_SECOND_BEST_RATIO = _toml_float("angle_selection", "second_best_ratio", default=1.15)
ANGLE_SELECTION_MIN_ACTIVE_WINDOWS = _toml_int("angle_selection", "min_active_windows", default=3)
ANGLE_SELECTION_SMOOTH_WINDOW = _toml_int("angle_selection", "smooth_window", default=5)
DYNAMIC_RECALIBRATION_POST_LOCK_MIN_RAW_REPS = _toml_int(
    "dynamic_recalibration", "post_lock_min_raw_reps", default=5
)
DYNAMIC_RECALIBRATION_PRIMARY_RECOVERY_BYPASS_REP_COOLDOWN = _toml_bool(
    "dynamic_recalibration", "primary_recovery_bypass_rep_cooldown", default=False
)
DYNAMIC_RECALIBRATION_PRIMARY_RECOVERY_FORCE_AFTER_STALE_REEVALS = _toml_int(
    "dynamic_recalibration", "primary_recovery_force_after_stale_reevals", default=8
)
DYNAMIC_RECALIBRATION_PRIMARY_RECOVERY_SKIP_SCORE_MARGIN = _toml_bool(
    "dynamic_recalibration", "primary_recovery_skip_score_margin", default=False
)
# Recent ROM below this fraction of calibrated ROM marks the tracked joint collapsed.
DYNAMIC_RECALIBRATION_RANGE_COLLAPSE_RATIO = _toml_float(
    "dynamic_recalibration", "range_collapse_ratio", default=0.50
)
FALLBACK_Y_ARM_WINDOW_SEC = _toml_float("fallback_y_point", "arm_window_sec", default=1.2)
FALLBACK_Y_LOW_SCORE_THRESHOLD = _toml_float(
    "fallback_y_point", "low_score_threshold", default=0.40
)
FALLBACK_Y_ARMING_MIN_SCORE = _toml_float(
    "fallback_y_point", "arming_min_score", default=0.62
)
FALLBACK_Y_PRIMARY_RECOVERY_SCORE = _toml_float(
    "fallback_y_point", "primary_recovery_score", default=0.60
)
FALLBACK_Y_MIN_ACTIVITY_SCORE = _toml_float(
    "fallback_y_point", "min_activity_score", default=0.50
)
FALLBACK_Y_MIN_POSE_SCORE = _toml_float(
    "fallback_y_point", "min_pose_score", default=0.45
)
FALLBACK_Y_MIN_ROM_SCORE = _toml_float("fallback_y_point", "min_rom_score", default=0.50)
FALLBACK_Y_MIN_EXTREMA_SCORE = _toml_float(
    "fallback_y_point", "min_extrema_score", default=0.40
)
FALLBACK_Y_MIN_COMPLETED_CYCLES = _toml_int(
    "fallback_y_point", "min_completed_cycles", default=2
)
FALLBACK_Y_BASELINE_MIN_SAMPLES = _toml_int(
    "fallback_y_point", "baseline_min_samples", default=20
)
FALLBACK_Y_BASELINE_WINDOW_FRAMES = _toml_int(
    "fallback_y_point", "baseline_window_frames", default=90
)
FALLBACK_Y_BASELINE_SHORT_WINDOW_FRAMES = _toml_int(
    "fallback_y_point", "baseline_short_window_frames", default=24
)
FALLBACK_Y_BASELINE_MAX_SLEW_PX_PER_SEC = _toml_float(
    "fallback_y_point", "baseline_max_slew_px_per_sec", default=6.0
)
FALLBACK_Y_BASELINE_JUMP_FRAC = _toml_float(
    "fallback_y_point", "baseline_jump_frac", default=0.25
)
FALLBACK_Y_MIN_CADENCE_SCORE = _toml_float(
    "fallback_y_point", "min_cadence_score", default=0.45
)
FALLBACK_Y_MIN_EVIDENCE_RANGE_PX = _toml_float(
    "fallback_y_point", "min_evidence_range_px", default=8.0
)
FALLBACK_Y_REQUIRE_BASELINE = bool(
    _toml_val("fallback_y_point", "require_baseline") is not False
)
FALLBACK_Y_STREAK_DECAY_GRACE_SEC = _toml_float(
    "fallback_y_point", "streak_decay_grace_sec", default=0.5
)

DEPTH_RECALIBRATION_ENABLED = bool(_toml_val("depth_recalibration", "enabled") is not False)
DEPTH_RECALIBRATION_SCALE_CHANGE_PCT = _toml_float(
    "depth_recalibration", "scale_change_pct", default=0.18
)
DEPTH_RECALIBRATION_BASELINE_JUMP_FRAC = _toml_float(
    "depth_recalibration", "baseline_jump_frac", default=0.25
)
DEPTH_RECALIBRATION_COOLDOWN_SEC = _toml_float(
    "depth_recalibration", "cooldown_sec", default=2.0
)
DEPTH_RECALIBRATION_OBSERVATION_REPS = _toml_int(
    "depth_recalibration", "observation_reps", default=2
)
DEPTH_RECALIBRATION_TRIGGER_ON_RANGE_COLLAPSE = _toml_bool(
    "depth_recalibration", "trigger_on_range_collapse", default=True
)
DEPTH_RECALIBRATION_DEFER_WHEN_PRIMARY_RECOVERED = _toml_bool(
    "depth_recalibration", "defer_when_primary_recovered", default=False
)
REP_VERTICAL_PX_SCALE_RELATIVE_RANGE_GATE = _toml_bool(
    "rep", "vertical_px", "scale_relative_range_gate", default=False
)
REP_VERTICAL_PX_MIN_SCALE_RATIO = _toml_float(
    "rep", "vertical_px", "min_scale_ratio", default=0.75
)

LOW_FPS_SAFE_MODE_ENABLED = bool(_toml_val("low_fps_safe_mode", "enabled") is not False)
LOW_FPS_INTERVAL_WINDOW_FRAMES = _toml_int(
    "low_fps_safe_mode", "interval_window_frames", default=12
)
LOW_FPS_MIN_SAMPLES = _toml_int("low_fps_safe_mode", "min_samples", default=8)
LOW_FPS_ENTER_P50_MS = _toml_float("low_fps_safe_mode", "enter_p50_ms", default=75.0)
LOW_FPS_ENTER_P90_MS = _toml_float("low_fps_safe_mode", "enter_p90_ms", default=200.0)
LOW_FPS_EXIT_P50_MS = _toml_float("low_fps_safe_mode", "exit_p50_ms", default=67.0)
LOW_FPS_EXIT_P90_MS = _toml_float("low_fps_safe_mode", "exit_p90_ms", default=100.0)
LOW_FPS_ENTER_STREAK_FRAMES = _toml_int(
    "low_fps_safe_mode", "enter_streak_frames", default=6
)
LOW_FPS_EXIT_STREAK_FRAMES = _toml_int(
    "low_fps_safe_mode", "exit_streak_frames", default=10
)
LOW_FPS_RECALIBRATION_ATTEMPTS_PER_ENTRY = _toml_int(
    "low_fps_safe_mode", "recalibration_attempts_per_entry", default=1
)


def _signal_modality_section(signal_unit: str) -> dict[str, Any]:
    """Modality overrides from ``[rep.<modality>]`` or ``[angle_selection.<modality>]``."""
    if signal_unit == "px":
        section = _toml_val("rep", "vertical_px")
    else:
        section = _toml_val("rep", "angle_deg")
    return section if isinstance(section, dict) else {}


def get_rep_modality_tuning_overrides(*, signal_unit: str) -> dict[str, Any]:
    """Peak-detector overrides by signal modality (``angle_deg`` or ``vertical_px``), not joint name."""
    section = _signal_modality_section(signal_unit)
    if not section:
        return {}

    def _f(key: str) -> Optional[float]:
        raw = section.get(key)
        if raw is not None and isinstance(raw, (int, float)):
            return float(raw)
        return None

    out: dict[str, Any] = {}
    mapping = {
        "hysteresis": "hysteresis",
        "min_peak_distance": "minPeakDistance",
        "min_range_gate": "minRangeGate",
        "range_window_frames": "rangeWindowFrames",
        "range_min_samples": "rangeMinSamples",
        "min_interval_ms": "minRepIntervalMs",
        "calibration_reps": "calibrationReps",
    }
    for toml_key, tp_key in mapping.items():
        val = _f(toml_key)
        if val is not None:
            if tp_key in ("minPeakDistance", "rangeWindowFrames", "rangeMinSamples", "calibrationReps"):
                out[tp_key] = int(val)
            else:
                out[tp_key] = val
    return out


def get_dynamic_recalibration_vertical_px_thresholds() -> dict[str, float]:
    """Stale-switch and range-health thresholds for vertical_px incumbents."""
    section_raw = _toml_val("dynamic_recalibration", "vertical_px")
    section = section_raw if isinstance(section_raw, dict) else {}

    def _local_int(key: str, default: int) -> int:
        raw = section.get(key)
        if raw is not None and isinstance(raw, (int, float)):
            return int(raw)
        return default

    def _local_float(key: str, default: float) -> float:
        raw = section.get(key)
        if raw is not None and isinstance(raw, (int, float)):
            return float(raw)
        return default

    return {
        "min_closed_streak": _local_int("min_closed_streak", 10),
        "force_after_stale_reevals": _local_int("force_after_stale_reevals", 8),
        "max_recent_range_px": _local_float("max_recent_range_px", 14.0),
    }


def get_angle_selection_modality_thresholds(*, signal_unit: str) -> dict[str, float]:
    """Selection gates by signal modality, not joint name."""
    if signal_unit == "px":
        section_raw = _toml_val("angle_selection", "vertical_px")
    else:
        section_raw = _toml_val("angle_selection", "angle_deg")
    section = section_raw if isinstance(section_raw, dict) else {}

    def _local(key: str, global_val: float) -> float:
        raw = section.get(key)
        if raw is not None and isinstance(raw, (int, float)):
            return float(raw)
        return global_val

    return {
        "min_variance": _local("min_variance", ANGLE_SELECTION_MIN_VARIANCE),
        "min_range_deg": _local("min_range_deg", ANGLE_SELECTION_MIN_RANGE_DEG),
        "min_variance_px2": _local("min_variance", ANGLE_SELECTION_MIN_VARIANCE),
        "min_range_px": _local("min_range", ANGLE_SELECTION_MIN_RANGE_DEG),
        "second_best_ratio": _local("second_best_ratio", ANGLE_SELECTION_SECOND_BEST_RATIO),
    }


def get_default_tuning_params() -> dict:
    return {
        "hysteresis": DEFAULT_HYSTERESIS,
        "minPeakDistance": DEFAULT_MIN_PEAK_DISTANCE,
        "smoothingFactor": DEFAULT_SMOOTHING_FACTOR,
        "peakMarginPct": DEFAULT_PEAK_MARGIN_PCT,
        "valleyMarginPct": DEFAULT_VALLEY_MARGIN_PCT,
        "minRangeGate": DEFAULT_MIN_RANGE_GATE,
        "rangeWindowFrames": DEFAULT_RANGE_WINDOW_FRAMES,
        "rangeMinSamples": DEFAULT_RANGE_MIN_SAMPLES,
        "angleDeltaDeadband": DEFAULT_ANGLE_DELTA_DEADBAND,
        "calibrationReps": DEFAULT_CALIBRATION_REPS,
        "calibrationCertainty": DEFAULT_CALIBRATION_CERTAINTY,
        "calibrationForceExtraReps": DEFAULT_CALIBRATION_FORCE_EXTRA_REPS,
        "minRepIntervalMs": DEFAULT_MIN_REP_INTERVAL_MS,
    }
