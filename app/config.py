"""Local app config: engine settings from the package + required VM URL for the visualizer."""
from flexible_rep_counter.core.settings import *  # noqa: F401,F403

if not YOLO_VM_TARGET_URL:
    raise RuntimeError(
        "Set [vm].direct_url in rep_counter.toml (repo root or cwd), or "
        "YOLO_VM_DIRECT_URL / YOLO_VM_TARGET_URL in the environment, or "
        "FLEXIBLE_REP_COUNTER_CONFIG to a TOML file path."
    )
