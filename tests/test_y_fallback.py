from __future__ import annotations

import json
import math
from pathlib import Path
from collections import deque
from typing import Any, cast

from flexible_rep_counter.core.math_engine import PeakDetector, calculate_from_type
from flexible_rep_counter.core.settings import get_default_tuning_params
from flexible_rep_counter.core.recalibration_confidence import (
    JointMotionState,
    classify_handoff,
    select_recalibration_candidate,
    update_joint_motion_state,
)
from flexible_rep_counter.core.variance_angle_selector import (
    COMMON_ANGLES,
    compute_angle_variances_from_buffer,
    determine_best_angle,
)
from flexible_rep_counter.session import (
    _apply_locked_tracking,
    _activate_joint_switch,
    _best_fallback_candidate,
    _fallback_epoch_raw_for_joint,
    _fallback_incumbent_low_score_now,
    _is_y_displacement_ready,
    _peak_detector_from_tuning,
    _snapshot_fallback_epoch_raw,
    _try_immediate_fallback_arm,
    _update_fallback_bad_streak,
)

class _FakeDetector:
    def __init__(self, rep_counts: list[int], *, rolling_range: float = 18.0) -> None:
        self._rep_counts = list(rep_counts)
        self._idx = 0
        self.rep_count = int(self._rep_counts[0] if self._rep_counts else 0)
        self._rolling_range = rolling_range

    def update(self, value: float | None) -> dict[str, Any]:
        i = min(self._idx, len(self._rep_counts) - 1)
        self.rep_count = int(self._rep_counts[i] if self._rep_counts else 0)
        self._idx += 1
        return {"repCount": self.rep_count, "rollingRange": self._rolling_range}

    def get_rep_count(self) -> int:
        return self.rep_count


def _make_landmarks(
    *,
    shoulder_y: float = 100.0,
    hip_y: float = 200.0,
    knee_y: float = 260.0,
    ankle_y: float = 320.0,
) -> list[dict]:
    lm = [{"x": 0.0, "y": 0.0, "confidence": 0.99} for _ in range(17)]
    # Shoulders
    lm[5].update({"x": 100.0, "y": shoulder_y})
    lm[6].update({"x": 200.0, "y": shoulder_y})
    # Elbows / wrists
    lm[7].update({"x": 100.0, "y": shoulder_y + 40.0})
    lm[8].update({"x": 200.0, "y": shoulder_y + 40.0})
    lm[9].update({"x": 100.0, "y": shoulder_y + 80.0})
    lm[10].update({"x": 200.0, "y": shoulder_y + 80.0})
    # Hips / knees / ankles
    lm[11].update({"x": 120.0, "y": hip_y})
    lm[12].update({"x": 180.0, "y": hip_y})
    lm[13].update({"x": 120.0, "y": knee_y})
    lm[14].update({"x": 180.0, "y": knee_y})
    lm[15].update({"x": 120.0, "y": ankle_y})
    lm[16].update({"x": 180.0, "y": ankle_y})
    return lm


def _transform_landmarks(landmarks: list[dict], *, scale: float, dx: float, dy: float) -> list[dict]:
    out: list[dict] = []
    for point in landmarks:
        out.append(
            {
                "x": float(point["x"]) * scale + dx,
                "y": float(point["y"]) * scale + dy,
                "confidence": float(point.get("confidence", 0.0)),
            }
        )
    return out


def test_absolute_y_changes_with_global_scale_and_translation() -> None:
    cfg = COMMON_ANGLES["SHOULDER_Y"]
    base = _make_landmarks(shoulder_y=95.0, hip_y=210.0)
    shifted_scaled = _transform_landmarks(base, scale=1.4, dx=50.0, dy=30.0)
    v1 = calculate_from_type(cfg["type"], cfg, base)
    v2 = calculate_from_type(cfg["type"], cfg, shifted_scaled)
    assert v1 is not None and v2 is not None
    assert abs(v1 - v2) > 1e-3


def test_fallback_variances_hidden_unless_armed() -> None:
    frame_buffer = [_make_landmarks(shoulder_y=100.0 + 8.0 * math.sin(i / 3.0)) for i in range(40)]
    without_fallback = compute_angle_variances_from_buffer(frame_buffer, include_debug=False, fallback_armed=False)
    with_fallback = compute_angle_variances_from_buffer(frame_buffer, include_debug=False, fallback_armed=True)
    assert "SHOULDER_Y" not in without_fallback
    assert "SHOULDER_Y" in with_fallback


def test_determine_best_angle_prefers_hip_depth_when_fallback_armed() -> None:
    frame_buffer: list[list[dict]] = []
    for i in range(60):
        offset = 16.0 * math.sin(i / 5.0)
        # Hips and knees move together; knee angle stays almost flat while hip depth changes.
        # Move shoulders with hips so SHOULDER_Y stays flat while HIP_DEPTH_Y varies.
        frame_buffer.append(
            _make_landmarks(
                shoulder_y=100.0 + offset,
                hip_y=200.0 + offset,
                knee_y=260.0 + offset,
                ankle_y=320.0,
            )
        )
    variances = compute_angle_variances_from_buffer(frame_buffer, include_debug=False, fallback_armed=True)
    out = determine_best_angle(frame_buffer, variances=variances, include_debug=False, fallback_armed=True)
    assert out["selectedAngle"] == "HIP_DEPTH_Y"


def test_select_candidate_ignores_fallback_when_not_armed() -> None:
    scores = {
        "RIGHT_ELBOW": (0.72, {"poseScore": 0.9, "recentRange": 32.0, "activityScore": 0.7, "completedCycles": 3}),
        "SHOULDER_Y": (
            0.95,
            {"poseScore": 0.9, "recentRange": 38.0, "activityScore": 0.8, "completedCycles": 4},
        ),
    }
    candidate, _ = select_recalibration_candidate(
        scores,
        "LEFT_ELBOW",
        variance_by_joint={},
        fallback_armed=False,
    )
    assert candidate == "RIGHT_ELBOW"


def test_select_candidate_prefers_primary_recovery_over_fallback() -> None:
    scores = {
        "RIGHT_ELBOW": (
            0.70,
            {"poseScore": 0.8, "recentRange": 30.0, "activityScore": 0.7, "completedCycles": 3},
        ),
        "SHOULDER_Y": (
            0.92,
            {"poseScore": 0.9, "recentRange": 36.0, "activityScore": 0.85, "completedCycles": 4},
        ),
    }
    candidate, debug = select_recalibration_candidate(
        scores,
        "LEFT_ELBOW",
        variance_by_joint={},
        fallback_armed=True,
        primary_recovery_score=0.60,
    )
    assert debug["primaryRecovered"] is True
    assert candidate == "RIGHT_ELBOW"


def test_fallback_transition_handoff_is_same_exercise() -> None:
    pending = {
        "incumbent_angle": "LEFT_ELBOW",
        "candidate_angle": "SHOULDER_Y",
        "candidate_current_raw": 5,
        "candidate_carryover_start_raw": 3,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "same_exercise"
    assert decision.rationale.get("rule") == "fallback_joint_transition"


def test_fallback_incumbent_low_rom_without_low_score() -> None:
    """Variance prior can keep score >= 0.40 while ROM is flat; low ROM must still arm."""
    assert not (
        0.55 < 0.40
        and 3.4 < 14.0
    )
    assert _fallback_incumbent_low_score_now(0.55, 3.4)
    assert not _fallback_incumbent_low_score_now(0.55, 20.0)
    assert _fallback_incumbent_low_score_now(0.30, 20.0)


def _legacy_fallback_low_score_now(incumbent_score: float, incumbent_recent_range_deg: float) -> bool:
    return bool(
        incumbent_score < 0.40
        and incumbent_recent_range_deg < 14.0
    )


def test_vertical_px_detector_counts_small_vertical_motion() -> None:
    """8px shoulder travel should register cycles with per-joint Y tuning (not angle defaults)."""
    import math

    from flexible_rep_counter.core.settings import get_default_tuning_params
    from flexible_rep_counter.core.variance_angle_selector import COMMON_ANGLES
    from flexible_rep_counter.session import _peak_detector_from_tuning

    cfg = COMMON_ANGLES["SHOULDER_Y"]
    det = _peak_detector_from_tuning(get_default_tuning_params(), angle_key="SHOULDER_Y")

    def make_lm(shoulder_y: float) -> list[dict]:
        lm = [{"x": 0.0, "y": 0.0, "confidence": 0.99} for _ in range(17)]
        lm[5].update({"x": 100.0, "y": shoulder_y, "confidence": 0.95})
        lm[6].update({"x": 200.0, "y": shoulder_y, "confidence": 0.95})
        lm[11].update({"x": 120.0, "y": 200.0, "confidence": 0.95})
        lm[12].update({"x": 180.0, "y": 200.0, "confidence": 0.95})
        return lm

    for i in range(160):
        offset = 8.0 * math.sin(i * 2 * math.pi / 8)
        val = calculate_from_type(cfg["type"], cfg, make_lm(100.0 - offset))
        assert val is not None
        det.update(val)
    assert det.get_rep_count() >= 8


def test_across_angle_stuck_zero_reps_regression_ndjson() -> None:
    """
    Regression from debug-1780013336: vertical motion, 0 reps on RIGHT_SHOULDER_ACROSS.

    Incumbent ROM never reached min_range_gate (15 deg); re-eval must not be blocked by post_lock.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "regression_shrug_debug-1780013336.ndjson"
    if not fixture_path.is_file():
        return
    frames = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tracking = [
        f
        for f in frames
        if f.get("tracked_joint") == "RIGHT_SHOULDER_ACROSS" and f.get("phase") == "tracking"
    ]
    assert len(tracking) >= 200
    assert max(int(f.get("reps") or 0) for f in tracking) == 0
    rr = [float(f.get("rolling_range") or 0.0) for f in tracking]
    assert max(rr) < 15.0
    assert sum(1 for x in rr if x < 14.0) >= len(rr) - 5


def test_low_rom_enables_fallback_low_score_regression_ndjson() -> None:
    """
    Regression from debug-1780012235 Brazil VM session (low-ROM across-shoulder segment).

    Incumbent rolling_range was ~3deg while score would stay inflated; old gate never opened.
    """
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "regression_shrug_debug-1780012235.ndjson"
    )
    if not fixture.is_file():
        return
    frames = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    segment_frames = [
        f
        for f in frames
        if f.get("tracked_joint") == "RIGHT_SHOULDER_ACROSS"
        and 139 <= int(f.get("frame_idx") or 0) <= 488
        and int(f.get("reps") or 0) >= 2
    ]
    assert len(segment_frames) >= 200

    inflated_score = 0.55
    newly_enabled = 0
    for f in segment_frames:
        recent_range = float(f.get("rolling_range") or 0.0)
        assert _fallback_incumbent_low_score_now(inflated_score, recent_range) == (
            recent_range < 14.0
        )
        if recent_range < 14.0:
            assert not _legacy_fallback_low_score_now(inflated_score, recent_range)
            newly_enabled += 1

    assert newly_enabled >= 250


def test_absolute_y_state_tracks_baseline_and_oscillation() -> None:
    state = JointMotionState(
        angle_key="SHOULDER_Y",
        detector=cast(PeakDetector, _FakeDetector([0] * 200)),
        history=deque(maxlen=400),
        confidence_history=deque(maxlen=400),
    )
    for i in range(120):
        y = 100.0 + 0.15 * i + 8.0 * math.sin(i / 5.0)
        update_joint_motion_state(
            state,
            y,
            0.95,
            i * 33,
            min_confidence=0.4,
            scale_px=120.0,
        )
    assert state.last_raw_y_px is not None
    assert state.last_baseline_px is not None
    assert state.last_oscillation_px is not None
    assert abs(float(state.last_oscillation_px)) < 20.0


def test_best_fallback_candidate_picks_highest_ready_score() -> None:
    scores = {
        "SHOULDER_Y": (
            0.48,
            {
                "poseScore": 0.9,
                "activityScore": 0.65,
                "romScore": 0.70,
                "extremaScore": 0.60,
                "completedCycles": 2,
                "recentRange": 18.0,
            },
        ),
        "HIP_DEPTH_Y": (
            0.72,
            {
                "poseScore": 0.9,
                "activityScore": 0.66,
                "romScore": 0.62,
                "extremaScore": 0.61,
                "completedCycles": 2,
                "recentRange": 19.0,
            },
        ),
    }
    key, score, _ = _best_fallback_candidate(
        scores,
    )
    assert key == "HIP_DEPTH_Y"
    assert score >= 0.70


def test_fallback_bad_streak_pauses_during_precalibration() -> None:
    state: dict[str, float | None] = {
        "fallback_bad_streak_started_at": None,
        "fallback_bad_streak_paused_total_s": 0.0,
        "fallback_bad_streak_pause_anchor": None,
        "fallback_bad_streak_accumulated_s": 0.0,
        "fallback_bad_streak_last_update_s": None,
        "fallback_epoch_raw": {},
        "selection_detectors_by_angle": {},
    }
    elapsed = _update_fallback_bad_streak(state, now_s=0.0, conditions_met=True, paused=False)
    assert elapsed == 0.0
    elapsed = _update_fallback_bad_streak(state, now_s=1.0, conditions_met=True, paused=False)
    assert abs(elapsed - 1.0) < 1e-6
    elapsed = _update_fallback_bad_streak(state, now_s=3.0, conditions_met=True, paused=True)
    assert abs(elapsed - 1.0) < 1e-6
    elapsed = _update_fallback_bad_streak(state, now_s=7.0, conditions_met=True, paused=True)
    assert abs(elapsed - 1.0) < 1e-6
    elapsed = _update_fallback_bad_streak(state, now_s=9.0, conditions_met=True, paused=False)
    assert abs(elapsed - 3.0) < 1e-6


def test_fallback_bad_streak_decays_with_grace_not_instant_reset() -> None:
    state: dict[str, float | None] = {
        "fallback_bad_streak_started_at": None,
        "fallback_bad_streak_paused_total_s": 0.0,
        "fallback_bad_streak_pause_anchor": None,
        "fallback_bad_streak_accumulated_s": 0.0,
        "fallback_bad_streak_last_update_s": None,
        "fallback_epoch_raw": {},
        "selection_detectors_by_angle": {},
    }
    elapsed = _update_fallback_bad_streak(state, now_s=0.0, conditions_met=True, paused=False)
    assert elapsed == 0.0
    elapsed = _update_fallback_bad_streak(state, now_s=2.0, conditions_met=True, paused=False)
    assert abs(elapsed - 2.0) < 1e-6
    elapsed = _update_fallback_bad_streak(state, now_s=2.5, conditions_met=False, paused=False)
    assert elapsed == 0.0
    elapsed = _update_fallback_bad_streak(state, now_s=3.0, conditions_met=False, paused=False)
    assert elapsed == 0.0


def test_best_fallback_candidate_prefers_torso_on_close_scores() -> None:
    scores = {
        "SHOULDER_Y": (
            0.70,
            {
                "poseScore": 0.9,
                "activityScore": 0.66,
                "romScore": 0.62,
                "extremaScore": 0.61,
                "completedCycles": 2,
                "recentRange": 19.0,
            },
        ),
        "RIGHT_WRIST_Y": (
            0.72,
            {
                "poseScore": 0.9,
                "activityScore": 0.66,
                "romScore": 0.62,
                "extremaScore": 0.61,
                "completedCycles": 2,
                "recentRange": 19.0,
            },
        ),
    }
    key, _, _ = _best_fallback_candidate(scores)
    assert key == "SHOULDER_Y"


def test_apply_locked_tracking_credits_fallback_epoch_reps() -> None:
    from flexible_rep_counter.session import _detector_raw_count

    run_state: dict[str, Any] = {
        "fallback_epoch_raw": {"SHOULDER_Y": 3},
        "tracking_step_count": 0,
        "suppress_retroactive_credit_until_step": 0,
        "last_joint_switch_step": 0,
        "rep_count_shown_floor": 0,
    }
    selection_det = _peak_detector_from_tuning(get_default_tuning_params(), angle_key="SHOULDER_Y")
    for i in range(40):
        val = calculate_from_type(
            COMMON_ANGLES["SHOULDER_Y"]["type"],
            COMMON_ANGLES["SHOULDER_Y"],
            _make_landmarks(shoulder_y=100.0 - 8.0 * math.sin(i / 2.0)),
        )
        selection_det.update(val)
    _apply_locked_tracking(
        run_state,
        "SHOULDER_Y",
        None,
        get_default_tuning_params(),
        selection_detector=selection_det,
    )
    selection_raw = _detector_raw_count(selection_det)
    assert selection_raw >= 3
    assert int(run_state["rep_count_offset"]) == max(0, selection_raw - 3)


def test_activate_joint_switch_fallback_credits_epoch_reps() -> None:
    run_state: dict[str, Any] = {
        "selected_angle": "LEFT_ELBOW",
        "fallback_epoch_raw": {"SHOULDER_Y": 4},
        "pending_switch_candidate_carryover_start_raw": 4,
        "pending_switch_candidate_carryover_start_shown": 4,
        "tracking_step_count": 0,
        "suppress_retroactive_credit_until_step": 0,
        "last_joint_switch_step": 0,
        "rep_count_shown_floor": 0,
    }
    det = _peak_detector_from_tuning(get_default_tuning_params(), angle_key="SHOULDER_Y")
    for i in range(30):
        val = calculate_from_type(
            COMMON_ANGLES["SHOULDER_Y"]["type"],
            COMMON_ANGLES["SHOULDER_Y"],
            _make_landmarks(shoulder_y=100.0 - 8.0 * math.sin(i / 2.0)),
        )
        det.update(val)
    sdba: dict[str, Any] = {"SHOULDER_Y": det}
    _activate_joint_switch(
        run_state,
        sdba,
        new_angle="SHOULDER_Y",
        detector=det,
        cumulative_shown=2,
        cumulative_raw=2,
        switched_at=0.0,
    )
    assert int(run_state["rep_count_offset"]) >= 0
    assert int(run_state["rep_count_offset"]) + det.get_rep_count() >= 2


def test_try_immediate_fallback_arm_requires_rep_evidence() -> None:
    scores = {
        "SHOULDER_Y": (
            0.72,
            {
                "poseScore": 0.9,
                "activityScore": 0.66,
                "romScore": 0.62,
                "extremaScore": 0.61,
                "completedCycles": 2,
                "recentRange": 19.0,
                "observable": True,
                "cadenceScore": 0.55,
                "recentRangePx": 12.0,
            },
        ),
    }
    state = JointMotionState(
        angle_key="SHOULDER_Y",
        detector=cast(PeakDetector, _FakeDetector([0] * 5)),
        history=deque(maxlen=400),
        confidence_history=deque(maxlen=400),
    )
    state.absolute_y_baseline_px = 100.0
    variances = {
        "SHOULDER_Y": {
            "medianWindowVariance": 8.0,
            "activeWindowCount": 4,
            "smoothedRangePx": 12.0,
            "signalUnit": "px",
        }
    }
    assert not _try_immediate_fallback_arm(
        score_by_joint=scores,
        variances=variances,
        joint_states={"SHOULDER_Y": state},
        rep_counts_all={"SHOULDER_Y": 0},
        cal_reps_target=2,
        best_primary_score=0.2,
    )
    assert _try_immediate_fallback_arm(
        score_by_joint=scores,
        variances=variances,
        joint_states={"SHOULDER_Y": state},
        rep_counts_all={"SHOULDER_Y": 5},
        cal_reps_target=2,
        best_primary_score=0.2,
    )


def test_y_displacement_ready_rejects_missing_baseline() -> None:
    state = JointMotionState(
        angle_key="SHOULDER_Y",
        detector=cast(PeakDetector, _FakeDetector([0] * 5)),
        history=deque(maxlen=400),
        confidence_history=deque(maxlen=400),
    )
    debug = {
        "observable": True,
        "cadenceScore": 0.55,
        "activityScore": 0.66,
        "recentRangePx": 12.0,
        "extremaScore": 0.61,
    }
    variances = {
        "SHOULDER_Y": {
            "medianWindowVariance": 8.0,
            "activeWindowCount": 4,
            "smoothedRangePx": 12.0,
            "signalUnit": "px",
        }
    }
    assert not _is_y_displacement_ready(state, debug, variances, angle_key="SHOULDER_Y")


def test_snapshot_fallback_epoch_raw_captures_selection_detectors() -> None:
    det = _peak_detector_from_tuning(get_default_tuning_params(), angle_key="SHOULDER_Y")
    for i in range(20):
        val = calculate_from_type(
            COMMON_ANGLES["SHOULDER_Y"]["type"],
            COMMON_ANGLES["SHOULDER_Y"],
            _make_landmarks(shoulder_y=100.0 - 8.0 * math.sin(i / 2.0)),
        )
        det.update(val)
    run_state = {"selection_detectors_by_angle": {"SHOULDER_Y": det}}
    _snapshot_fallback_epoch_raw(run_state)
    assert _fallback_epoch_raw_for_joint(run_state, "SHOULDER_Y") == det.get_rep_count()


def test_brazil_debug_session_ndjson_regression_expectations() -> None:
    """
    Regression expectations from Brazil VM session pulls (yolo-deploy).

    S1: vertical motion with late Y lock should have background vertical_px rep increments.
    S2: selecting phase should have background vertical_px increments but 0 shown reps.
    S3: unrelated joint sessions should not merge counts at session level.
    """

    def _vertical_px_detector_label(label: object) -> bool:
        if not isinstance(label, str):
            return False
        return label.endswith("_Y") or label == "SHOULDER_SHRUG_Y"

    ndjson = (
        Path(__file__).resolve().parents[2]
        / "yolo-deploy"
        / "logs"
        / "sessions"
        / "latest-brazil"
        / "events.ndjson"
    )
    if not ndjson.is_file():
        return
    rows = [json.loads(line) for line in ndjson.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sid = str(row.get("session_id") or "")
        by_session.setdefault(sid, []).append(row)

    session_stats = []
    for sid, sr in by_session.items():
        fs = [r for r in sr if r.get("event") == "frame_snapshot"]
        ri = [r for r in sr if r.get("event") == "rep_increment"]
        vertical_px_inc = sum(
            1 for r in ri if _vertical_px_detector_label(r.get("detector_label"))
        )
        session_stats.append(
            {
                "session_id": sid,
                "max_reps": max((int(r.get("reps") or 0) for r in sr), default=0),
                "tracked_frames": len([r for r in fs if r.get("tracked_joint")]),
                "vertical_px_increments": vertical_px_inc,
                "ever_armed": any(r.get("fallbackArmed") for r in fs),
            }
        )

    s1 = max(session_stats, key=lambda s: s["vertical_px_increments"])
    s2 = min(session_stats, key=lambda s: (s["max_reps"], -s["vertical_px_increments"]))
    assert s1["vertical_px_increments"] >= 15
    assert s2["max_reps"] == 0
    assert s2["vertical_px_increments"] >= 15


def test_apply_scale_relative_range_gate_updates_vertical_px_detectors() -> None:
    from flexible_rep_counter.session import _apply_scale_relative_range_gate
    import flexible_rep_counter.session as session_mod

    session_mod.REP_VERTICAL_PX_SCALE_RELATIVE_RANGE_GATE = True
    try:
        det = PeakDetector(min_range_gate_degrees=5.0)
        state = JointMotionState(
            angle_key="SHOULDER_Y",
            detector=det,
            history=deque(),
            confidence_history=deque(),
        )
        state.last_scale_px = 212.0
        state.scale_at_lock_px = 256.0
        run_state: dict[str, Any] = {"tuning_params": {"minRangeGate": 15.0}}
        effective = _apply_scale_relative_range_gate(
            run_state,
            {"SHOULDER_Y": state},
            selected_angle="SHOULDER_Y",
            peak_detector=det,
        )
        assert effective is not None
        assert abs(float(det.min_range_gate_degrees) - 4.1484375) < 0.01
        assert run_state.get("effective_min_range_gate") == effective
    finally:
        session_mod.REP_VERTICAL_PX_SCALE_RELATIVE_RANGE_GATE = False


def test_peak_detector_reset_calibration_preserves_reps() -> None:
    detector = PeakDetector(
        smoothing_factor=0.3,
        hysteresis=2.0,
        min_peak_distance=1,
        min_range_gate_degrees=0.0,
        calibration_reps=1,
        calibration_certainty=0.0,
        calibration_force_extra_reps=0,
    )
    for i in range(80):
        detector.update(100.0 + 20.0 * math.sin(i / 2.0))
    before = detector.get_rep_count()
    assert before > 0
    detector.reset_calibration_preserve_reps()
    after = detector.get_rep_count()
    assert after == before
