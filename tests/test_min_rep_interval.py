from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from flexible_rep_counter.core.math_engine import PeakDetector


def _calibrated_detector(*, min_rep_interval_ms: float = 500.0) -> PeakDetector:
    det = PeakDetector(
        hysteresis=5.0,
        min_peak_distance=1,
        min_range_gate_degrees=0.0,
        range_window_frames=5,
        range_min_samples=2,
        calibration_reps=1,
        calibration_certainty=0.0,
        min_rep_interval_ms=min_rep_interval_ms,
    )
    det._calibrated = True
    det._calibrated_avg_peak = 100.0
    det._calibrated_avg_valley = 40.0
    det._last_rolling_range = 60.0
    det.rep_count = 7
    det.peaks = [100.0] * 7
    det.valleys = [40.0] * 8
    det._last_rep_time_ms = time.time() * 1000.0
    return det


def test_min_rep_interval_commits_instead_of_blocking() -> None:
    det = _calibrated_detector(min_rep_interval_ms=500.0)
    det.state = "GOING_UP"
    det.current_peak_value = 101.0
    det.frame_count = 100
    det.last_peak_frame = 0

    peak, completed = det._record_peak_on_reversal()

    assert peak == 101.0
    assert completed is True
    assert det.rep_count == 8
    assert len(det.peaks) == 8
