"""PeakDetector stamps client frame_id at the extremum, not at confirmation."""
from __future__ import annotations

from flexible_rep_counter.core.math_engine import PeakDetector


def test_peak_frame_id_is_extremum_not_confirmation() -> None:
    det = PeakDetector(
        smoothing_factor=1.0,
        hysteresis=5.0,
        min_peak_distance=1,
        min_range_gate_degrees=0.0,
        calibration_reps=3,
        calibration_certainty=0.0,
        min_rep_interval_ms=0.0,
    )
    # Climb to peak at frame-A, then reverse past hysteresis on later frames.
    det.update(10.0, frame_id="start")
    det.update(20.0, frame_id="climb-1")
    det.update(30.0, frame_id="frame-A")  # extremum
    det.update(28.0, frame_id="hold")  # still within hysteresis
    out = det.update(24.0, frame_id="frame-B")  # confirmation (30 - 5 = 25 threshold)

    assert out["peak"] == 30.0
    assert det.peak_frame_ids == ["frame-A"]
    assert det.peak_frame_ids[-1] != "frame-B"


def test_valley_frame_id_is_extremum_not_confirmation() -> None:
    det = PeakDetector(
        smoothing_factor=1.0,
        hysteresis=5.0,
        min_peak_distance=1,
        min_range_gate_degrees=0.0,
        calibration_reps=3,
        calibration_certainty=0.0,
        min_rep_interval_ms=0.0,
    )
    det.update(50.0, frame_id="start")
    det.update(40.0, frame_id="down-1")
    det.update(20.0, frame_id="frame-V")  # extremum valley
    det.update(22.0, frame_id="hold")
    out = det.update(26.0, frame_id="frame-confirm")  # confirmation

    assert out["valley"] == 20.0
    assert det.valley_frame_ids == ["frame-V"]
    assert det.valley_frame_ids[-1] != "frame-confirm"


def test_first_rep_needs_both_peak_and_valley_frame_ids() -> None:
    det = PeakDetector(
        smoothing_factor=1.0,
        hysteresis=5.0,
        min_peak_distance=1,
        min_range_gate_degrees=0.0,
        calibration_reps=3,
        calibration_certainty=0.0,
        min_rep_interval_ms=0.0,
    )
    # Peak then valley → first completed cycle.
    det.update(10.0, frame_id="n0")
    det.update(40.0, frame_id="peak-frame")
    det.update(34.0, frame_id="peak-confirm")
    assert det.peak_frame_ids == ["peak-frame"]
    assert det.rep_count == 0

    det.update(10.0, frame_id="valley-frame")
    det.update(16.0, frame_id="valley-confirm")
    assert det.valley_frame_ids == ["valley-frame"]
    assert det.rep_count == 1
    assert det.peak_frame_ids[0] == "peak-frame"
    assert det.valley_frame_ids[0] == "valley-frame"
    assert det.peak_frame_ids[0] != det.valley_frame_ids[0]


def test_reset_clears_frame_ids() -> None:
    det = PeakDetector(
        smoothing_factor=1.0,
        hysteresis=5.0,
        min_peak_distance=1,
        min_range_gate_degrees=0.0,
        calibration_reps=3,
        calibration_certainty=0.0,
        min_rep_interval_ms=0.0,
    )
    det.update(10.0, frame_id="a")
    det.update(40.0, frame_id="b")
    det.update(34.0, frame_id="c")
    assert det.peak_frame_ids
    det.reset()
    assert det.peak_frame_ids == []
    assert det.valley_frame_ids == []
    assert det.current_peak_frame_id is None
    assert det.current_valley_frame_id is None
