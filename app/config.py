"""Local app config: engine settings from the package + required VM URL for the visualizer."""
from flexible_rep_counter.core.settings import *  # noqa: F401,F403

if not VM_BASE_URL:
    raise RuntimeError(
        "Set [vm].direct_url in rep_counter.toml (repo root, cwd, or parent), "
        "or FLEXIBLE_REP_COUNTER_CONFIG to a TOML file that defines it."
    )
