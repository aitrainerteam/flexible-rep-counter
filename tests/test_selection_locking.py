from __future__ import annotations

from flexible_rep_counter.session import (
    _adaptive_variance_gate,
    _apply_locked_tracking,
    _family_rep_dominance,
    _peak_detector_from_tuning,
    _select_family_joint_candidate,
    _selection_dominance_thresholds,
)


def test_apply_locked_tracking_drops_selection_rep_debt() -> None:
    tuning = {
        "smoothingFactor": 0.7,
        "hysteresis": 16.0,
        "minPeakDistance": 4,
        "peakMarginPct": 0.5,
        "valleyMarginPct": 0.5,
        "minRangeGate": 15.0,
        "rangeWindowFrames": 90,
        "rangeMinSamples": 12,
        "angleDeltaDeadband": 0.0,
        "calibrationReps": 2,
        "calibrationCertainty": 0.2,
        "calibrationForceExtraReps": 0,
        "minRepIntervalMs": 500.0,
    }
    selection_detector = _peak_detector_from_tuning(tuning)
    selection_detector.rep_count = 28
    selection_detector.peaks = [80.0] * 28
    selection_detector.valleys = [20.0] * 28

    run_state: dict[str, object] = {"tracking_step_count": 0}
    _apply_locked_tracking(
        run_state,
        "LEFT_HIP",
        None,
        tuning,
        selection_detector=selection_detector,
        initial_angle_value=92.0,
    )

    locked = run_state["peak_detector"]
    assert locked is not selection_detector
    assert locked.get_rep_count() == 0


def test_family_dominance_prefers_best_joint_in_family() -> None:
    rep_counts = {
        "LEFT_KNEE": 8,
        "RIGHT_KNEE": 8,
        "LEFT_ELBOW": 1,
    }
    variances = {
        "LEFT_KNEE": {
            "activeWindowCount": 3,
            "medianWindowVariance": 8.0,
            "smoothedRangeDeg": 20.0,
        },
        "RIGHT_KNEE": {
            "activeWindowCount": 3,
            "medianWindowVariance": 11.0,
            "smoothedRangeDeg": 24.0,
        },
        "LEFT_ELBOW": {
            "activeWindowCount": 3,
            "medianWindowVariance": 7.0,
            "smoothedRangeDeg": 18.0,
        },
    }
    family = _family_rep_dominance(rep_counts)
    assert family["leaderFamily"] == "KNEE"
    chosen = _select_family_joint_candidate(
        leader_family="KNEE",
        rep_counts=rep_counts,
        variances=variances,
        elapsed_s=4.0,
    )
    assert chosen == "RIGHT_KNEE"


def test_adaptive_variance_gate_relaxes_after_long_selection() -> None:
    variances = {
        "LEFT_KNEE": {
            "activeWindowCount": 2,
            "medianWindowVariance": 5.0,
            "smoothedRangeDeg": 14.0,
        },
        "RIGHT_KNEE": {
            "activeWindowCount": 2,
            "medianWindowVariance": 5.0,
            "smoothedRangeDeg": 14.0,
        },
    }
    assert _adaptive_variance_gate("LEFT_KNEE", variances, elapsed_s=1.0) is False
    assert _adaptive_variance_gate("LEFT_KNEE", variances, elapsed_s=12.0) is True


def test_selection_thresholds_relax_over_time() -> None:
    early = _selection_dominance_thresholds(1.0)
    late = _selection_dominance_thresholds(12.0)
    assert late["joint_fraction"] < early["joint_fraction"]
    assert late["family_fraction"] < early["family_fraction"]
    assert late["streak_required"] < early["streak_required"]
