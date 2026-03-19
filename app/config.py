"""Load configuration from .env and expose for VM URL and tuning."""
import os
from pathlib import Path

# Load .env from project root (parent of app/)
_ROOT = Path(__file__).resolve().parent.parent
_env_path = _ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=True)
except ImportError:
    # Minimal .env parse when python-dotenv not installed
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"')

# VM inference: YOLO_VM_DIRECT_URL or YOLO_VM_TARGET_URL in .env
YOLO_VM_TARGET_URL = (
    os.environ.get("YOLO_VM_DIRECT_URL", "").strip()
    or os.environ.get("YOLO_VM_TARGET_URL", "").strip()
    or None
)
if not YOLO_VM_TARGET_URL:
    raise RuntimeError(
        "Set YOLO_VM_DIRECT_URL or YOLO_VM_TARGET_URL in .env (e.g. http://host:8000)"
    )
VM_TIMEOUT_SEC = float(os.environ.get("VM_TIMEOUT_SEC", "5.0"))
VM_HEALTH_TIMEOUT_SEC = float(os.environ.get("VM_HEALTH_TIMEOUT_SEC", "5.0"))
# Optional: resize before upload (0 = full frame; same idea as yolo-deploy camera_pose_client)
PREDICT_RESIZE_WIDTH = int(os.environ.get("PREDICT_RESIZE_WIDTH", "0"))
PREDICT_JPEG_QUALITY = int(os.environ.get("PREDICT_JPEG_QUALITY", "85"))
PREDICT_VALIDATE_RESPONSE = os.environ.get("PREDICT_VALIDATE_RESPONSE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

# Debug console (second window); LOG_LEVEL=DEBUG for dev (default), set LOG_LEVEL=INFO to disable
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
DEBUG_CONSOLE_ENABLED = LOG_LEVEL == "DEBUG"

# Tuning (match limbSelection.js defaults; override via env)
DEFAULT_HYSTERESIS = float(os.environ.get("REP_HYSTERESIS", "5"))
DEFAULT_MIN_PEAK_DISTANCE = int(os.environ.get("REP_MIN_PEAK_DISTANCE", "5"))
DEFAULT_SMOOTHING_FACTOR = float(os.environ.get("REP_SMOOTHING_FACTOR", "0.45"))
# Only count peak/valley when within this many degrees of the average observed max/min (avoids counting micro-movements)
DEFAULT_PEAK_MARGIN = float(os.environ.get("REP_PEAK_MARGIN", "15"))
DEFAULT_VALLEY_MARGIN = float(os.environ.get("REP_VALLEY_MARGIN", "15"))


def get_default_tuning_params():
    return {
        "hysteresis": DEFAULT_HYSTERESIS,
        "minPeakDistance": DEFAULT_MIN_PEAK_DISTANCE,
        "smoothingFactor": DEFAULT_SMOOTHING_FACTOR,
        "peakMargin": DEFAULT_PEAK_MARGIN,
        "valleyMargin": DEFAULT_VALLEY_MARGIN,
    }
