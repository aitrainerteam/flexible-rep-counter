from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from flexible_rep_counter.core.variance_angle_selector import (  # noqa: E402
    COMMON_ANGLES,
    _get_top_candidate,
    determine_best_angle,
)


def _variance_row(angle_key: str, median_var: float, range_deg: float, active: int = 4) -> dict:
    return {
        "medianWindowVariance": float(median_var),
        "activeWindowCount": int(active),
        "smoothedRangeDeg": float(range_deg),
        "config": COMMON_ANGLES[angle_key],
    }


def _frame_with_conf(default_conf: float, overrides: dict[int, float]) -> list[dict]:
    points = []
    for i in range(17):
        conf = float(overrides.get(i, default_conf))
        points.append({"x": 0.0, "y": 0.0, "confidence": conf})
    return points


def _frame_buffer(default_conf: float, overrides: dict[int, float], n: int = 24) -> list[list[dict]]:
    return [_frame_with_conf(default_conf, overrides) for _ in range(n)]


def test_unclear_tiebreak_prefers_stable_non_elbow_joint() -> None:
    variances = {
        "LEFT_ELBOW": _variance_row("LEFT_ELBOW", median_var=11.0, range_deg=25.0),
        "LEFT_KNEE": _variance_row("LEFT_KNEE", median_var=10.8, range_deg=24.0),
    }
    top = _get_top_candidate(variances)
    assert top is not None
    assert top["key"] == "LEFT_KNEE"
    assert bool(top.get("__tiebreak_unclear")) is True


def test_unclear_tiebreak_rejects_weak_close_candidates() -> None:
    variances = {
        "LEFT_ELBOW": _variance_row("LEFT_ELBOW", median_var=6.0, range_deg=13.0),
        "LEFT_KNEE": _variance_row("LEFT_KNEE", median_var=6.0, range_deg=15.0),
    }
    top = _get_top_candidate(variances)
    assert top is None


def test_determine_best_angle_blocks_low_confidence_tiebreak_pick() -> None:
    variances = {
        "LEFT_ELBOW": _variance_row("LEFT_ELBOW", median_var=11.0, range_deg=25.0),
        "LEFT_KNEE": _variance_row("LEFT_KNEE", median_var=10.8, range_deg=24.0),
    }
    # Knee landmarks 11/13/15 are above base 0.50 but below tiebreak 0.62 threshold.
    frames = _frame_buffer(default_conf=0.9, overrides={11: 0.55, 13: 0.55, 15: 0.55})
    out = determine_best_angle(frames, variances=variances, include_debug=False)
    assert out["selectedAngle"] is None
    assert out["source"] == "none"


def test_determine_best_angle_allows_high_confidence_tiebreak_pick() -> None:
    variances = {
        "LEFT_ELBOW": _variance_row("LEFT_ELBOW", median_var=11.0, range_deg=25.0),
        "LEFT_KNEE": _variance_row("LEFT_KNEE", median_var=10.8, range_deg=24.0),
    }
    frames = _frame_buffer(default_conf=0.9, overrides={11: 0.9, 13: 0.9, 15: 0.9})
    out = determine_best_angle(frames, variances=variances, include_debug=False)
    assert out["selectedAngle"] == "LEFT_KNEE"
    assert out["source"] == "variance_tiebreak"
