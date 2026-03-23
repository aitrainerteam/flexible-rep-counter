"""Engine tuning and selection thresholds from environment.

Safe to import without a VM URL (unlike the local app shim in ``app.config``).
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env: repo root = parents from src/flexible_rep_counter/core/settings.py
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

# VM (optional for library-only use; app.config enforces when running local VM client)
YOLO_VM_TARGET_URL = (
    os.environ.get("YOLO_VM_DIRECT_URL", "").strip()
    or os.environ.get("YOLO_VM_TARGET_URL", "").strip()
    or None
)
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

DEFAULT_HYSTERESIS = float(os.environ.get("REP_HYSTERESIS", "5"))
DEFAULT_MIN_PEAK_DISTANCE = int(os.environ.get("REP_MIN_PEAK_DISTANCE", "5"))
DEFAULT_SMOOTHING_FACTOR = float(os.environ.get("REP_SMOOTHING_FACTOR", "0.45"))
DEFAULT_PEAK_MARGIN = float(os.environ.get("REP_PEAK_MARGIN", "15"))
DEFAULT_VALLEY_MARGIN = float(os.environ.get("REP_VALLEY_MARGIN", "15"))
DEFAULT_MIN_RANGE_GATE = float(os.environ.get("REP_MIN_RANGE_GATE", "15"))
DEFAULT_RANGE_WINDOW_FRAMES = int(os.environ.get("REP_RANGE_WINDOW_FRAMES", "90"))
DEFAULT_RANGE_MIN_SAMPLES = int(os.environ.get("REP_RANGE_MIN_SAMPLES", "12"))
DEFAULT_ANGLE_DELTA_DEADBAND = float(os.environ.get("REP_ANGLE_DELTA_DEADBAND", "0"))
DEFAULT_CALIBRATION_REPS = int(os.environ.get("REP_CALIBRATION_REPS", "3"))
DEFAULT_CALIBRATION_CERTAINTY = float(os.environ.get("REP_CALIBRATION_CERTAINTY", "0.5"))
DEFAULT_CALIBRATION_FORCE_EXTRA_REPS = int(os.environ.get("REP_CALIBRATION_FORCE_EXTRA_REPS", "2"))
DEFAULT_MIN_REP_INTERVAL_MS = float(os.environ.get("REP_MIN_INTERVAL_MS", "400"))

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
