from __future__ import annotations

import math

from flexible_rep_counter.core.math_engine import calculate_from_type
from flexible_rep_counter.core.recalibration_confidence import (
    classify_handoff,
    select_recalibration_candidate,
)
from flexible_rep_counter.core.variance_angle_selector import (
    COMMON_ANGLES,
    compute_angle_variances_from_buffer,
    determine_best_angle,
)
from flexible_rep_counter.session import _update_fallback_bad_streak


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


def test_relational_y_is_invariant_to_global_scale_and_translation() -> None:
    cfg = COMMON_ANGLES["SHOULDER_SHRUG_Y"]
    base = _make_landmarks(shoulder_y=95.0, hip_y=210.0)
    shifted_scaled = _transform_landmarks(base, scale=1.4, dx=50.0, dy=30.0)
    v1 = calculate_from_type(cfg["type"], cfg, base)
    v2 = calculate_from_type(cfg["type"], cfg, shifted_scaled)
    assert v1 is not None and v2 is not None
    assert abs(v1 - v2) < 1e-6


def test_fallback_variances_hidden_unless_armed() -> None:
    frame_buffer = [_make_landmarks(shoulder_y=100.0 + 8.0 * math.sin(i / 3.0)) for i in range(40)]
    without_fallback = compute_angle_variances_from_buffer(frame_buffer, include_debug=False, fallback_armed=False)
    with_fallback = compute_angle_variances_from_buffer(frame_buffer, include_debug=False, fallback_armed=True)
    assert "SHOULDER_SHRUG_Y" not in without_fallback
    assert "SHOULDER_SHRUG_Y" in with_fallback


def test_determine_best_angle_prefers_hip_depth_when_fallback_armed() -> None:
    frame_buffer: list[list[dict]] = []
    for i in range(60):
        offset = 16.0 * math.sin(i / 5.0)
        # Hips and knees move together; knee angle stays almost flat while hip depth changes.
        frame_buffer.append(
            _make_landmarks(
                shoulder_y=100.0,
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
        "SHOULDER_SHRUG_Y": (
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
        "SHOULDER_SHRUG_Y": (
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
        "candidate_angle": "SHOULDER_SHRUG_Y",
        "candidate_current_raw": 5,
        "candidate_carryover_start_raw": 3,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "same_exercise"
    assert decision.rationale.get("rule") == "fallback_joint_transition"


def test_fallback_bad_streak_pauses_during_precalibration() -> None:
    state: dict[str, float | None] = {
        "fallback_bad_streak_started_at": None,
        "fallback_bad_streak_paused_total_s": 0.0,
        "fallback_bad_streak_pause_anchor": None,
    }
    elapsed = _update_fallback_bad_streak(state, now_s=0.0, conditions_met=True, paused=False)
    assert elapsed == 0.0
    elapsed = _update_fallback_bad_streak(state, now_s=2.0, conditions_met=True, paused=True)
    assert abs(elapsed - 2.0) < 1e-6
    elapsed = _update_fallback_bad_streak(state, now_s=6.0, conditions_met=True, paused=True)
    assert abs(elapsed - 2.0) < 1e-6
    elapsed = _update_fallback_bad_streak(state, now_s=8.0, conditions_met=True, paused=False)
    assert abs(elapsed - 2.0) < 1e-6
