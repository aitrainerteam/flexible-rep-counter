"""Local app config: engine settings from the package + required VM URL for the visualizer."""
from flexible_rep_counter.core.settings import *  # noqa: F401,F403

if not YOLO_VM_TARGET_URL:
    raise RuntimeError(
        "Set YOLO_VM_DIRECT_URL or YOLO_VM_TARGET_URL in .env (e.g. http://host:8000)"
    )
