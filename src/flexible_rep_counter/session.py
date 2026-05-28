"""Session state machine: landmark stream -> rep metrics (same logic as legacy webcam loop)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import replace
from typing import Any, Literal, Optional

from flexible_rep_counter.instrumentation import (
    RepInstrumentationSink,
    RepInstrumentationSettings,
    merge_trace,
)

from flexible_rep_counter.core.math_engine import (
    PeakDetector,
    calculate_from_type,
    create_peak_detector,
    get_min_confidence_for_landmarks,
)
from flexible_rep_counter.core.recalibration_confidence import (
    HandoffDecision,
    JointMotionState,
    should_run_full_recalibration,
    classify_handoff,
    compute_joint_recalibration_score,
    cycle_sync_score_last_4s,
    is_mirrored_pair,
    is_same_joint_family,
    median_cycle_rom_deg,
    median_recent_range_from_score_debug,
    select_recalibration_candidate,
    should_switch_to_candidate,
    update_joint_motion_state,
)
from flexible_rep_counter.core.pose_filters import PoseFilterPipeline
from flexible_rep_counter.core.settings import (
    ANGLE_SELECTION_DOMINANCE_FRACTION,
    ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES,
    ANGLE_SELECTION_MAX_BUFFER_FRAMES,
    ANGLE_SELECTION_MIN_ACTIVE_WINDOWS,
    ANGLE_SELECTION_MIN_FRAMES,
    ANGLE_SELECTION_MIN_LEADING_REPS,
    ANGLE_SELECTION_MIN_SEC,
    ANGLE_SELECTION_RETRY_INTERVAL_SEC,
    ANGLE_SELECTION_REEVALUATE_EVERY_SEC,
    ANGLE_SELECTION_SWITCH_MIN_REPS,
    ANGLE_SELECTION_SWITCH_MIN_SEC,
    ANGLE_SELECTION_VARIANCE_FALLBACK_SEC,
    DYNAMIC_RECALIBRATION_POST_LOCK_MIN_RAW_REPS,
    FALLBACK_Y_ARM_WINDOW_SEC as CFG_FALLBACK_Y_ARM_WINDOW_SEC,
    FALLBACK_Y_ARMING_MIN_SCORE as CFG_FALLBACK_Y_ARMING_MIN_SCORE,
    FALLBACK_Y_LOW_SCORE_THRESHOLD as CFG_FALLBACK_Y_LOW_SCORE_THRESHOLD,
    FALLBACK_Y_MIN_ACTIVITY_SCORE as CFG_FALLBACK_Y_MIN_ACTIVITY_SCORE,
    FALLBACK_Y_MIN_COMPLETED_CYCLES as CFG_FALLBACK_Y_MIN_COMPLETED_CYCLES,
    FALLBACK_Y_MIN_EXTREMA_SCORE as CFG_FALLBACK_Y_MIN_EXTREMA_SCORE,
    FALLBACK_Y_MIN_POSE_SCORE as CFG_FALLBACK_Y_MIN_POSE_SCORE,
    FALLBACK_Y_MIN_ROM_SCORE as CFG_FALLBACK_Y_MIN_ROM_SCORE,
    FALLBACK_Y_PRIMARY_RECOVERY_SCORE as CFG_FALLBACK_Y_PRIMARY_RECOVERY_SCORE,
    get_default_tuning_params,
    LOW_FPS_ENTER_P50_MS as CFG_LOW_FPS_ENTER_P50_MS,
    LOW_FPS_ENTER_P90_MS as CFG_LOW_FPS_ENTER_P90_MS,
    LOW_FPS_ENTER_STREAK_FRAMES as CFG_LOW_FPS_ENTER_STREAK_FRAMES,
    LOW_FPS_EXIT_P50_MS as CFG_LOW_FPS_EXIT_P50_MS,
    LOW_FPS_EXIT_P90_MS as CFG_LOW_FPS_EXIT_P90_MS,
    LOW_FPS_EXIT_STREAK_FRAMES as CFG_LOW_FPS_EXIT_STREAK_FRAMES,
    LOW_FPS_INTERVAL_WINDOW_FRAMES as CFG_LOW_FPS_INTERVAL_WINDOW_FRAMES,
    LOW_FPS_MIN_SAMPLES as CFG_LOW_FPS_MIN_SAMPLES,
    LOW_FPS_SAFE_MODE_ENABLED as CFG_LOW_FPS_SAFE_MODE_ENABLED,
)
from flexible_rep_counter.core.variance_angle_selector import (
    COMMON_ANGLES,
    FRAME_MIN_CONFIDENCE,
    compute_angle_variances_from_buffer,
    determine_best_angle,
    dominance_conditions_met,
    is_fallback_angle,
    passes_consistent_variance_gate,
    summarize_rep_dominance,
)
from flexible_rep_counter.types import StepResult

DEFAULT_TUNING_PARAMS = get_default_tuning_params()

# Match yolo-deploy / angles.py gate so pose_dropped aligns with omitted angles in JSON.
_MIN_KEYPOINT_CONF_FOR_ANGLE = 0.3
# Temporary kill-switch: disable runtime variance-based joint recalibration/switching,
# pending handoff observation, and classify_handoff carryover.
# Keep reevaluation code in place so it can be re-enabled later by flipping this.
DYNAMIC_RECALIBRATION_ENABLED = True
STALE_SWITCH_SELECTED_RECENT_WINDOW = 16
STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG = 14.0
STALE_SWITCH_MIN_CLOSED_STREAK = 10
STALE_SWITCH_FORCE_AFTER_STALE_REEVALS = 8
JOINT_SWITCH_MIN_REPS_SINCE_LAST = max(1, int(ANGLE_SELECTION_SWITCH_MIN_REPS))
PENDING_SWITCH_MIN_OBSERVATION_MS = 300
PENDING_SWITCH_MAX_OBSERVATION_MS = 600
HANDOFF_OBSERVATION_MIN_COMPLETED_CYCLES = 2
HANDOFF_OBSERVATION_MIN_ACTIVITY_SCORE = 0.45
HANDOFF_OBSERVATION_MIN_POSE_SCORE = 0.45
HANDOFF_OBSERVATION_INCUMBENT_POSE_WEAK_SCORE = 0.45
POST_LOCK_MIN_RAW_REPS = max(0, int(DYNAMIC_RECALIBRATION_POST_LOCK_MIN_RAW_REPS))
LOW_FPS_SAFE_MODE_ENABLED = CFG_LOW_FPS_SAFE_MODE_ENABLED
LOW_FPS_INTERVAL_WINDOW_FRAMES = max(1, int(CFG_LOW_FPS_INTERVAL_WINDOW_FRAMES))
LOW_FPS_MIN_SAMPLES = max(2, int(CFG_LOW_FPS_MIN_SAMPLES))
LOW_FPS_ENTER_P50_MS = float(CFG_LOW_FPS_ENTER_P50_MS)
LOW_FPS_ENTER_P90_MS = float(CFG_LOW_FPS_ENTER_P90_MS)
LOW_FPS_EXIT_P50_MS = float(CFG_LOW_FPS_EXIT_P50_MS)
LOW_FPS_EXIT_P90_MS = float(CFG_LOW_FPS_EXIT_P90_MS)
LOW_FPS_ENTER_STREAK = max(1, int(CFG_LOW_FPS_ENTER_STREAK_FRAMES))
LOW_FPS_EXIT_STREAK = max(1, int(CFG_LOW_FPS_EXIT_STREAK_FRAMES))
LOW_FPS_GAP_SPIKE_MS = 200.0
JOINT_SWITCH_SUPPRESS_RETROACTIVE_STEPS = 45
REFERENCE_FPS_FOR_PEAK_DISTANCE = 12.0
MIN_PEAK_DISTANCE_FLOOR = 2
SELECTION_RELAX_MAX_SEC = 6.0
SELECTION_MIN_DOMINANCE_FRACTION = 0.54
SELECTION_MIN_FAMILY_DOMINANCE_FRACTION = 0.50
SELECTION_MIN_STREAK_FRAMES = 8
SELECTION_MIN_ACTIVE_WINDOWS_RELAXED = 2
SELECTION_MIN_MEDIAN_VARIANCE_ABS = 3.0
SELECTION_MIN_RANGE_DEG_ABS = 10.0
FALLBACK_Y_ARM_WINDOW_SEC = max(0.0, float(CFG_FALLBACK_Y_ARM_WINDOW_SEC))
FALLBACK_Y_LOW_SCORE_THRESHOLD = float(CFG_FALLBACK_Y_LOW_SCORE_THRESHOLD)
FALLBACK_Y_ARMING_MIN_SCORE = float(CFG_FALLBACK_Y_ARMING_MIN_SCORE)
FALLBACK_Y_PRIMARY_RECOVERY_SCORE = float(CFG_FALLBACK_Y_PRIMARY_RECOVERY_SCORE)
FALLBACK_Y_MIN_ACTIVITY_SCORE = float(CFG_FALLBACK_Y_MIN_ACTIVITY_SCORE)
FALLBACK_Y_MIN_POSE_SCORE = float(CFG_FALLBACK_Y_MIN_POSE_SCORE)
FALLBACK_Y_MIN_ROM_SCORE = float(CFG_FALLBACK_Y_MIN_ROM_SCORE)
FALLBACK_Y_MIN_EXTREMA_SCORE = float(CFG_FALLBACK_Y_MIN_EXTREMA_SCORE)
FALLBACK_Y_MIN_COMPLETED_CYCLES = max(1, int(CFG_FALLBACK_Y_MIN_COMPLETED_CYCLES))


def _diagnose_missing_angle(
    cfg: dict[str, Any], landmarks: list[dict]
) -> tuple[str, dict[str, Any]]:
    """Why ``calculate_from_type`` may return None for this joint configuration."""
    detail: dict[str, Any] = {"landmark_indices": list(cfg.get("landmarks") or [])}
    idxs = list(cfg.get("landmarks") or [])
    confs: list[dict[str, Any]] = []
    for i in idxs:
        if i >= len(landmarks):
            detail["keypoints"] = confs
            return "missing_keypoint_index", detail
        p = landmarks[i]
        c = float(p.get("confidence", 0.0))
        confs.append({"index": i, "confidence": c})
        if c < _MIN_KEYPOINT_CONF_FOR_ANGLE:
            detail["keypoints"] = confs
            detail["min_conf_required"] = _MIN_KEYPOINT_CONF_FOR_ANGLE
            return "low_keypoint_confidence", detail
    detail["keypoints"] = confs
    return "geometry_unavailable", detail


def _peak_detector_from_tuning(tuning_params: dict[str, Any]) -> PeakDetector:
    d = DEFAULT_TUNING_PARAMS
    tp = tuning_params or {}
    return create_peak_detector(
        smoothing_factor=float(tp.get("smoothingFactor", d["smoothingFactor"])),
        hysteresis=float(tp.get("hysteresis", d["hysteresis"])),
        min_peak_distance=int(tp.get("minPeakDistance", d["minPeakDistance"])),
        peak_margin_pct=float(tp.get("peakMarginPct", d["peakMarginPct"])),
        valley_margin_pct=float(tp.get("valleyMarginPct", d["valleyMarginPct"])),
        min_range_gate_degrees=float(tp.get("minRangeGate", d["minRangeGate"])),
        range_window_frames=int(tp.get("rangeWindowFrames", d["rangeWindowFrames"])),
        range_min_samples=int(tp.get("rangeMinSamples", d["rangeMinSamples"])),
        delta_deadband_degrees=float(tp.get("angleDeltaDeadband", d["angleDeltaDeadband"])),
        calibration_reps=int(tp.get("calibrationReps", d["calibrationReps"])),
        calibration_certainty=float(tp.get("calibrationCertainty", d["calibrationCertainty"])),
        calibration_force_extra_reps=int(
            tp.get("calibrationForceExtraReps", d["calibrationForceExtraReps"])
        ),
        min_rep_interval_ms=float(tp.get("minRepIntervalMs", d["minRepIntervalMs"])),
    )


def _normalized_angle_family(angle_key: str) -> str:
    if is_fallback_angle(angle_key):
        return angle_key
    base = angle_key
    if base.startswith("LEFT_"):
        base = base[len("LEFT_") :]
    elif base.startswith("RIGHT_"):
        base = base[len("RIGHT_") :]
    if base.endswith("_ACROSS"):
        base = base[: -len("_ACROSS")]
    return base


def _angle_landmarks(cfg: dict[str, Any]) -> list[int]:
    landmarks = cfg.get("landmarks")
    if isinstance(landmarks, list) and landmarks:
        return [int(i) for i in landmarks]
    moving = [int(i) for i in (cfg.get("moving") or [])]
    reference = [int(i) for i in (cfg.get("reference") or [])]
    merged = moving + reference
    deduped: list[int] = []
    for idx in merged:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def _angle_confidence_threshold(cfg: dict[str, Any]) -> float:
    if str(cfg.get("type") or "").strip().lower() == "relational_y_displacement":
        return float(cfg.get("min_conf", 0.4) or 0.4)
    return FRAME_MIN_CONFIDENCE


def _reset_fallback_streak(run_state: dict[str, Any]) -> None:
    run_state["fallback_bad_streak_started_at"] = None
    run_state["fallback_bad_streak_paused_total_s"] = 0.0
    run_state["fallback_bad_streak_pause_anchor"] = None


def _update_fallback_bad_streak(
    run_state: dict[str, Any],
    *,
    now_s: float,
    conditions_met: bool,
    paused: bool,
) -> float:
    if not conditions_met:
        _reset_fallback_streak(run_state)
        return 0.0
    started = run_state.get("fallback_bad_streak_started_at")
    if not isinstance(started, (int, float)):
        started = now_s
        run_state["fallback_bad_streak_started_at"] = started
    paused_total = float(run_state.get("fallback_bad_streak_paused_total_s") or 0.0)
    pause_anchor = run_state.get("fallback_bad_streak_pause_anchor")
    if paused:
        if not isinstance(pause_anchor, (int, float)):
            run_state["fallback_bad_streak_pause_anchor"] = now_s
            pause_anchor = now_s
    elif isinstance(pause_anchor, (int, float)):
        paused_total += max(0.0, now_s - float(pause_anchor))
        run_state["fallback_bad_streak_paused_total_s"] = paused_total
        run_state["fallback_bad_streak_pause_anchor"] = None
        pause_anchor = None
    live_pause = max(0.0, now_s - float(pause_anchor)) if isinstance(pause_anchor, (int, float)) else 0.0
    effective_elapsed = max(0.0, now_s - float(started) - paused_total - live_pause)
    return effective_elapsed


def _is_fallback_candidate_ready(score: float, debug: dict[str, Any]) -> bool:
    return bool(
        score >= FALLBACK_Y_ARMING_MIN_SCORE
        and float(debug.get("activityScore") or 0.0) >= FALLBACK_Y_MIN_ACTIVITY_SCORE
        and float(debug.get("poseScore") or 0.0) >= FALLBACK_Y_MIN_POSE_SCORE
        and float(debug.get("romScore") or 0.0) >= FALLBACK_Y_MIN_ROM_SCORE
        and float(debug.get("extremaScore") or 0.0) >= FALLBACK_Y_MIN_EXTREMA_SCORE
        and int(debug.get("completedCycles") or 0) >= FALLBACK_Y_MIN_COMPLETED_CYCLES
    )


def _best_fallback_candidate(
    score_by_joint: dict[str, tuple[float, dict[str, Any]]],
) -> tuple[Optional[str], float, dict[str, Any]]:
    best_key: Optional[str] = None
    best_score = -1.0
    best_debug: dict[str, Any] = {}
    for angle_key, (score, debug) in score_by_joint.items():
        if not is_fallback_angle(angle_key):
            continue
        if _is_fallback_candidate_ready(float(score), debug) and float(score) > best_score:
            best_key = angle_key
            best_score = float(score)
            best_debug = debug
    return best_key, best_score, best_debug


def _has_primary_recovery_candidate(
    score_by_joint: dict[str, tuple[float, dict[str, Any]]],
) -> bool:
    for angle_key, (score, debug) in score_by_joint.items():
        if is_fallback_angle(angle_key):
            continue
        if (
            float(score) >= FALLBACK_Y_PRIMARY_RECOVERY_SCORE
            and float(debug.get("activityScore") or 0.0) >= FALLBACK_Y_MIN_ACTIVITY_SCORE
            and float(debug.get("poseScore") or 0.0) >= FALLBACK_Y_MIN_POSE_SCORE
            and int(debug.get("completedCycles") or 0) >= FALLBACK_Y_MIN_COMPLETED_CYCLES
        ):
            return True
    return False


def _selection_relax_progress(elapsed_s: float) -> float:
    if elapsed_s <= ANGLE_SELECTION_MIN_SEC:
        return 0.0
    return min(
        1.0,
        max(0.0, (elapsed_s - ANGLE_SELECTION_MIN_SEC) / max(0.1, SELECTION_RELAX_MAX_SEC)),
    )


def _selection_dominance_thresholds(elapsed_s: float) -> dict[str, Any]:
    progress = _selection_relax_progress(elapsed_s)
    joint_fraction = max(
        SELECTION_MIN_DOMINANCE_FRACTION,
        float(ANGLE_SELECTION_DOMINANCE_FRACTION) - 0.12 * progress,
    )
    family_fraction = max(
        SELECTION_MIN_FAMILY_DOMINANCE_FRACTION,
        min(joint_fraction, float(ANGLE_SELECTION_DOMINANCE_FRACTION) - 0.16 * progress),
    )
    streak_required = int(
        round(
            max(
                float(SELECTION_MIN_STREAK_FRAMES),
                float(ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES)
                - (float(ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES) - float(SELECTION_MIN_STREAK_FRAMES))
                * progress,
            )
        )
    )
    min_leading_reps = max(
        1,
        int(round(float(ANGLE_SELECTION_MIN_LEADING_REPS) - progress)),
    )
    return {
        "progress": progress,
        "joint_fraction": joint_fraction,
        "family_fraction": family_fraction,
        "streak_required": streak_required,
        "min_leading_reps": min_leading_reps,
    }


def _adaptive_variance_gate(
    angle_key: str,
    variances: dict[str, dict[str, Any]],
    *,
    elapsed_s: float,
) -> bool:
    if passes_consistent_variance_gate(variances, angle_key):
        return True
    row = variances.get(angle_key) or {}
    active_windows = int(row.get("activeWindowCount") or 0)
    median_var = float(row.get("medianWindowVariance") or 0.0)
    range_deg = float(row.get("smoothedRangeDeg") or 0.0)
    if active_windows <= 0 or median_var <= 0.0 or range_deg <= 0.0:
        return False
    progress = _selection_relax_progress(elapsed_s)
    all_variances = [
        float(v.get("medianWindowVariance") or 0.0)
        for v in variances.values()
        if float(v.get("medianWindowVariance") or 0.0) > 0.0
    ]
    all_ranges = [
        float(v.get("smoothedRangeDeg") or 0.0)
        for v in variances.values()
        if float(v.get("smoothedRangeDeg") or 0.0) > 0.0
    ]
    med_variance = sorted(all_variances)[len(all_variances) // 2] if all_variances else 0.0
    med_range = sorted(all_ranges)[len(all_ranges) // 2] if all_ranges else 0.0
    min_windows = max(
        SELECTION_MIN_ACTIVE_WINDOWS_RELAXED,
        int(round(ANGLE_SELECTION_MIN_ACTIVE_WINDOWS - progress)),
    )
    required_variance = max(
        SELECTION_MIN_MEDIAN_VARIANCE_ABS,
        med_variance * (0.95 - 0.35 * progress),
    )
    required_range = max(
        SELECTION_MIN_RANGE_DEG_ABS,
        med_range * (0.92 - 0.32 * progress),
    )
    return (
        active_windows >= min_windows
        and median_var >= required_variance
        and range_deg >= required_range
    )


def _family_rep_dominance(rep_counts: dict[str, int]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    for angle_key, reps in rep_counts.items():
        count = int(reps or 0)
        if count <= 0:
            continue
        family = _normalized_angle_family(angle_key)
        by_family[family] = by_family.get(family, 0) + count
    total = sum(by_family.values())
    if total <= 0 or not by_family:
        return {
            "totalReps": 0,
            "leaderFamily": None,
            "leaderReps": 0,
            "leaderShare": 0.0,
        }
    leader_family = max(by_family.keys(), key=lambda k: by_family[k])
    leader_reps = int(by_family[leader_family])
    return {
        "totalReps": int(total),
        "leaderFamily": leader_family,
        "leaderReps": leader_reps,
        "leaderShare": float(leader_reps / total),
    }


def _select_family_joint_candidate(
    *,
    leader_family: str,
    rep_counts: dict[str, int],
    variances: dict[str, dict[str, Any]],
    elapsed_s: float,
) -> Optional[str]:
    best: Optional[tuple[int, float, float, str]] = None
    for angle_key in COMMON_ANGLES:
        if _normalized_angle_family(angle_key) != leader_family:
            continue
        if not _adaptive_variance_gate(angle_key, variances, elapsed_s=elapsed_s):
            continue
        row = variances.get(angle_key) or {}
        reps = int(rep_counts.get(angle_key) or 0)
        var_score = float(row.get("medianWindowVariance") or 0.0)
        range_score = float(row.get("smoothedRangeDeg") or 0.0)
        candidate = (reps, var_score, range_score, angle_key)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return best[3]


def _build_joint_motion_states(
    tuning_params: dict[str, Any],
    *,
    history_maxlen: int = ANGLE_SELECTION_MAX_BUFFER_FRAMES,
) -> dict[str, JointMotionState]:
    states: dict[str, JointMotionState] = {}
    for angle_key in COMMON_ANGLES:
        states[angle_key] = JointMotionState(
            angle_key=angle_key,
            detector=_peak_detector_from_tuning(tuning_params),
            history=deque(maxlen=history_maxlen),
            confidence_history=deque(maxlen=history_maxlen),
        )
    return states


def _apply_locked_tracking(
    run_state: dict[str, Any],
    selected_angle: str,
    buf_list: Optional[list[list[dict]]],
    tuning_params: dict[str, Any],
    *,
    selection_detector: Optional[Any] = None,
    initial_angle_value: Optional[float] = None,
) -> None:
    run_state["selected_angle"] = selected_angle
    run_state["selected_config"] = COMMON_ANGLES[selected_angle]
    run_state["selection_last_attempt"] = None
    run_state["selection_last_reevaluate_at"] = time.time()
    run_state["selection_last_switch_at"] = time.time()
    # Never carry rep debt from selection detectors into tracking lock.
    # Selection may run for many frames while outputs remain hidden (reps=0),
    # so reusing the detector would surface a sudden jump on first lock.
    det: Any = _peak_detector_from_tuning(tuning_params)
    if isinstance(initial_angle_value, (int, float)):
        angle_seed = float(initial_angle_value)
        if angle_seed == angle_seed:
            det.update(angle_seed)
    run_state["peak_detector"] = det
    run_state["rep_count_offset"] = 0
    run_state["rep_count_raw_offset"] = 0
    run_state["tracking_raw_at_joint_lock"] = 0
    _clear_pending_switch(run_state)
    run_state["selection_dominance_key"] = None
    run_state["selection_dominance_streak"] = 0
    _mark_joint_activation_guard(run_state)


def _is_valid_angle_value(value: Optional[float]) -> bool:
    return value is not None and not (isinstance(value, float) and value != value)


def _append_angle_history(
    angle_histories: dict[str, deque],
    angle_key: str,
    value: Optional[float],
    landmarks: list[dict],
) -> None:
    hist = angle_histories.get(angle_key)
    if hist is None:
        return
    cfg = COMMON_ANGLES[angle_key]
    angle_landmarks = _angle_landmarks(cfg)
    min_conf = get_min_confidence_for_landmarks(landmarks, angle_landmarks)
    min_conf_required = _angle_confidence_threshold(cfg)
    if not _is_valid_angle_value(value) or min_conf is None or min_conf < min_conf_required:
        hist.append(None)
        return
    assert value is not None
    hist.append(float(value))


def _update_angle_histories_for_frame(
    angle_histories: dict[str, deque],
    angle_values: dict[str, Optional[float]],
    landmarks: list[dict],
) -> None:
    for angle_key in COMMON_ANGLES:
        _append_angle_history(
            angle_histories,
            angle_key,
            angle_values.get(angle_key),
            landmarks,
        )


def _detector_counts(detector: Optional[Any]) -> tuple[int, int]:
    if detector is None:
        return 0, 0
    raw = int(detector.get_rep_count() or 0)
    shown = raw
    peaks = list(getattr(detector, "peaks", []) or [])
    valleys = list(getattr(detector, "valleys", []) or [])
    if peaks and valleys and len(peaks) != len(valleys):
        shown += 1
    return shown, raw


def _detector_raw_count(detector: Optional[Any]) -> int:
    if detector is None:
        return 0
    return int(detector.get_rep_count() or 0)


def _trim_incomplete_extremum(detector: Optional[Any]) -> None:
    """Drop a trailing half-cycle so handoff offsets are not inflated by peak/valley skew."""
    if detector is None:
        return
    peaks = getattr(detector, "peaks", None)
    valleys = getattr(detector, "valleys", None)
    if not isinstance(peaks, list) or not isinstance(valleys, list):
        return
    if not peaks or not valleys:
        return
    if len(peaks) > len(valleys):
        peaks.pop()
    elif len(valleys) > len(peaks):
        valleys.pop()


def _retroactive_credit_eligible(detector: Optional[Any], run_state: dict[str, Any]) -> bool:
    if detector is None:
        return False
    suppress_until = int(run_state.get("suppress_retroactive_credit_until_step") or 0)
    step = int(run_state.get("tracking_step_count") or 0)
    if step < suppress_until:
        return False
    peaks = list(getattr(detector, "peaks", []) or [])
    valleys = list(getattr(detector, "valleys", []) or [])
    return bool(peaks and valleys and len(peaks) != len(valleys))


def _mark_joint_activation_guard(run_state: dict[str, Any]) -> None:
    step = int(run_state.get("tracking_step_count") or 0)
    run_state["suppress_retroactive_credit_until_step"] = (
        step + JOINT_SWITCH_SUPPRESS_RETROACTIVE_STEPS
    )
    run_state["last_joint_switch_step"] = step


def _apply_monotonic_shown_floor(run_state: dict[str, Any], cumulative_shown: int) -> int:
    floor = int(run_state.get("rep_count_shown_floor") or 0)
    shown = max(int(cumulative_shown), floor)
    if shown > floor:
        run_state["rep_count_shown_floor"] = shown
    return shown


def _ledger_cumulative_reps(
    run_state: dict[str, Any],
    detector: Optional[Any],
    *,
    include_retroactive: bool,
) -> tuple[int, int]:
    raw = _detector_raw_count(detector)
    cumulative_raw = int(run_state.get("rep_count_raw_offset") or 0) + raw
    retro = 1 if include_retroactive and _retroactive_credit_eligible(detector, run_state) else 0
    cumulative_shown = _apply_monotonic_shown_floor(
        run_state,
        int(run_state.get("rep_count_offset") or 0) + raw + retro,
    )
    return cumulative_shown, cumulative_raw


def _effective_min_peak_distance(base: int, run_state: dict[str, Any]) -> int:
    fps = run_state.get("low_fps_effective_fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return base
    scaled = int(round(base * REFERENCE_FPS_FOR_PEAK_DISTANCE / float(fps)))
    return max(MIN_PEAK_DISTANCE_FLOOR, scaled)


def _apply_fps_scaled_peak_distance(
    run_state: dict[str, Any],
    joint_states: dict[str, JointMotionState],
    *,
    peak_detector: Optional[Any] = None,
) -> None:
    base = int(
        (run_state.get("tuning_params") or DEFAULT_TUNING_PARAMS).get(
            "minPeakDistance", DEFAULT_TUNING_PARAMS["minPeakDistance"]
        )
    )
    effective = _effective_min_peak_distance(base, run_state)
    if int(run_state.get("_last_applied_min_peak_distance") or -1) == effective:
        return
    run_state["_last_applied_min_peak_distance"] = effective
    for state in joint_states.values():
        det = state.detector
        if det is not None and hasattr(det, "min_peak_distance"):
            det.min_peak_distance = effective
    active = peak_detector
    if active is not None and hasattr(active, "min_peak_distance"):
        active.min_peak_distance = effective


def _init_ledger_guard_state(run_state: dict[str, Any]) -> None:
    run_state["tracking_step_count"] = 0
    run_state["suppress_retroactive_credit_until_step"] = 0
    run_state["last_joint_switch_step"] = 0
    run_state["rep_count_shown_floor"] = 0
    run_state["_last_applied_min_peak_distance"] = -1


def _percentile_nearest(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def _update_low_fps_health(run_state: dict[str, Any]) -> None:
    run_state["low_fps_mode_changed_pulse"] = False
    now_s = time.monotonic()
    last_s = run_state.get("last_step_monotonic_s")
    run_state["last_step_monotonic_s"] = now_s
    if isinstance(last_s, (int, float)):
        delta_ms = max(0.0, (now_s - float(last_s)) * 1000.0)
        window = run_state.get("frame_interval_window")
        if isinstance(window, deque):
            window.append(delta_ms)
    window = run_state.get("frame_interval_window")
    if not isinstance(window, deque) or len(window) < LOW_FPS_MIN_SAMPLES:
        run_state["low_fps_p50_ms"] = None
        run_state["low_fps_p90_ms"] = None
        run_state["low_fps_gaps_over_200ms"] = 0
        run_state["low_fps_effective_fps"] = None
        return

    sorted_intervals = sorted(float(v) for v in window if isinstance(v, (int, float)))
    if not sorted_intervals:
        run_state["low_fps_p50_ms"] = None
        run_state["low_fps_p90_ms"] = None
        run_state["low_fps_gaps_over_200ms"] = 0
        run_state["low_fps_effective_fps"] = None
        return

    p50_ms = _percentile_nearest(sorted_intervals, 50.0)
    p90_ms = _percentile_nearest(sorted_intervals, 90.0)
    run_state["low_fps_p50_ms"] = p50_ms
    run_state["low_fps_p90_ms"] = p90_ms
    run_state["low_fps_gaps_over_200ms"] = sum(
        1 for v in sorted_intervals if v >= LOW_FPS_GAP_SPIKE_MS
    )
    run_state["low_fps_effective_fps"] = (1000.0 / p50_ms) if p50_ms > 0 else None

    enter_cond = p50_ms >= LOW_FPS_ENTER_P50_MS or p90_ms >= LOW_FPS_ENTER_P90_MS
    exit_cond = p50_ms < LOW_FPS_EXIT_P50_MS and p90_ms < LOW_FPS_EXIT_P90_MS
    if not LOW_FPS_SAFE_MODE_ENABLED:
        run_state["low_fps_mode_active"] = False
        run_state["low_fps_enter_streak"] = 0
        run_state["low_fps_exit_streak"] = 0
        return
    mode_active = bool(run_state.get("low_fps_mode_active"))

    if mode_active:
        run_state["low_fps_enter_streak"] = 0
        if exit_cond:
            run_state["low_fps_exit_streak"] = int(run_state.get("low_fps_exit_streak") or 0) + 1
        else:
            run_state["low_fps_exit_streak"] = 0
        if int(run_state.get("low_fps_exit_streak") or 0) >= LOW_FPS_EXIT_STREAK:
            run_state["low_fps_mode_active"] = False
            run_state["low_fps_mode_changed_pulse"] = True
            run_state["low_fps_exit_streak"] = 0
    else:
        run_state["low_fps_exit_streak"] = 0
        if enter_cond:
            run_state["low_fps_enter_streak"] = int(run_state.get("low_fps_enter_streak") or 0) + 1
        else:
            run_state["low_fps_enter_streak"] = 0
        if int(run_state.get("low_fps_enter_streak") or 0) >= LOW_FPS_ENTER_STREAK:
            run_state["low_fps_mode_active"] = True
            run_state["low_fps_mode_changed_pulse"] = True
            run_state["low_fps_enter_streak"] = 0


def _clear_pending_switch(run_state: dict[str, Any]) -> None:
    run_state["pending_switch_angle"] = None
    run_state["pending_switch_detector"] = None
    run_state["pending_switch_started_at_ms"] = None
    run_state["pending_switch_candidate_shown_start"] = None
    run_state["pending_switch_candidate_raw_start"] = None
    run_state["pending_switch_incumbent_shown_start"] = None
    run_state["pending_switch_incumbent_raw_start"] = None
    run_state["pending_switch_incumbent_pose_score_at_start"] = 0.0
    run_state["pending_switch_incumbent_completed_cycles_at_start"] = 0
    run_state["pending_switch_incumbent_recent_range_at_start"] = 0.0
    run_state["pending_switch_incumbent_last_observed_ts_at_start"] = None
    run_state["pending_switch_candidate_completed_cycles_at_start"] = 0
    run_state["pending_switch_candidate_last_cycle_ts_before_start"] = None
    run_state["pending_switch_candidate_rom_score_at_start"] = 0.0
    run_state["pending_switch_candidate_carryover_start_shown"] = 0
    run_state["pending_switch_candidate_carryover_start_raw"] = 0
    run_state["pending_switch_candidate_carryover_start_ts"] = 0
    run_state["pending_switch_incumbent_cycles_last_4s"] = 0
    run_state["pending_switch_candidate_cycles_last_4s"] = 0
    run_state["pending_switch_cycle_sync_score_last_4s"] = 0.0
    run_state["pending_switch_same_joint_family"] = False
    run_state["pending_switch_mirrored_pair"] = False
    run_state["pending_switch_incumbent_advanced"] = False
    run_state["pending_switch_observed"] = False
    run_state["pending_switch_incumbent_angle_min"] = None
    run_state["pending_switch_incumbent_angle_max"] = None
    run_state["pending_switch_incumbent_motion_span_deg"] = 0.0
    run_state["pending_switch_incumbent_observable_during_pending"] = False
    run_state["pending_switch_incumbent_completed_gated_cycle_during_pending"] = False
    run_state["pending_switch_candidate_advanced_during_pending"] = False
    run_state["pending_switch_candidate_completed_gated_cycle_during_pending"] = False
    run_state["pending_switch_candidate_pending_rom_estimate_deg"] = 0.0
    run_state["pending_switch_forced"] = False


def _clear_handoff_observation(run_state: dict[str, Any]) -> None:
    run_state["handoff_observation_candidate_angle"] = None
    run_state["handoff_observation_selected_angle"] = None
    run_state["handoff_observation_started_at_ms"] = None
    run_state["handoff_observation_candidate_carryover_start_shown"] = 0
    run_state["handoff_observation_candidate_carryover_start_raw"] = 0
    run_state["handoff_observation_candidate_carryover_start_ts"] = 0


def _recent_history_range(series: Optional[deque], window: int) -> Optional[float]:
    if not isinstance(series, deque) or window <= 0:
        return None
    valid_values: list[float] = []
    for value in reversed(series):
        if _is_valid_angle_value(value):
            valid_values.append(float(value))
        if len(valid_values) >= window:
            break
    if len(valid_values) < 2:
        return None
    return max(valid_values) - min(valid_values)


def _update_pending_incumbent_motion(
    run_state: dict[str, Any], current_angle: Optional[float]
) -> None:
    if not isinstance(current_angle, (int, float)):
        return
    if isinstance(current_angle, float) and current_angle != current_angle:
        return
    current = float(current_angle)
    angle_min = run_state.get("pending_switch_incumbent_angle_min")
    angle_max = run_state.get("pending_switch_incumbent_angle_max")
    if not isinstance(angle_min, (int, float)) or not isinstance(angle_max, (int, float)):
        run_state["pending_switch_incumbent_angle_min"] = current
        run_state["pending_switch_incumbent_angle_max"] = current
        return
    min_v = min(float(angle_min), current)
    max_v = max(float(angle_max), current)
    run_state["pending_switch_incumbent_angle_min"] = min_v
    run_state["pending_switch_incumbent_angle_max"] = max_v
    run_state["pending_switch_incumbent_motion_span_deg"] = max_v - min_v


def _is_detector_calibrated(
    detector: Optional[Any], detector_output: Optional[dict[str, Any]] = None
) -> bool:
    if detector is None:
        return False
    if isinstance(detector_output, dict):
        return bool(detector_output.get("calibrationComplete", False))
    try:
        state = detector.get_state()
    except Exception:
        return False
    return bool(state.get("calibrationComplete", False))


def _activate_joint_switch(
    run_state: dict[str, Any],
    detectors_by_angle: dict[str, Any],
    *,
    new_angle: str,
    detector: Any,
    cumulative_shown: int,
    cumulative_raw: int,
    switched_at: float,
) -> None:
    detectors_by_angle[new_angle] = detector
    _trim_incomplete_extremum(detector)
    candidate_current_raw = _detector_raw_count(detector)
    candidate_current_shown, _ = _detector_counts(detector)
    pending_state = {
        "incumbent_angle": run_state.get("selected_angle"),
        "candidate_angle": new_angle,
        "incumbent_advanced": bool(run_state.get("pending_switch_incumbent_advanced")),
        "incumbent_completed_gated_cycle_during_pending": bool(
            run_state.get("pending_switch_incumbent_completed_gated_cycle_during_pending")
        ),
        "incumbent_motion_span_deg": float(run_state.get("pending_switch_incumbent_motion_span_deg") or 0.0),
        "candidate_pending_rom_estimate_deg": float(
            run_state.get("pending_switch_candidate_pending_rom_estimate_deg") or 0.0
        ),
        "incumbent_observable_during_pending": bool(
            run_state.get("pending_switch_incumbent_observable_during_pending")
        ),
        "incumbent_pose_score_at_start": float(
            run_state.get("pending_switch_incumbent_pose_score_at_start") or 0.0
        ),
        "incumbent_last_observed_ts_at_start": run_state.get(
            "pending_switch_incumbent_last_observed_ts_at_start"
        ),
        "pending_start_ts": int(run_state.get("pending_switch_started_at_ms") or 0),
        "candidate_completed_cycles_at_start": int(
            run_state.get("pending_switch_candidate_completed_cycles_at_start") or 0
        ),
        "candidate_last_cycle_ts_before_start": run_state.get(
            "pending_switch_candidate_last_cycle_ts_before_start"
        ),
        "candidate_rom_score_at_start": float(
            run_state.get("pending_switch_candidate_rom_score_at_start") or 0.0
        ),
        "same_joint_family": bool(run_state.get("pending_switch_same_joint_family")),
        "incumbent_cycles_last_4s": int(run_state.get("pending_switch_incumbent_cycles_last_4s") or 0),
        "candidate_cycles_last_4s": int(run_state.get("pending_switch_candidate_cycles_last_4s") or 0),
        "cycle_sync_score_last_4s": float(run_state.get("pending_switch_cycle_sync_score_last_4s") or 0.0),
        "mirrored_pair": bool(run_state.get("pending_switch_mirrored_pair")),
        "candidate_advanced_during_pending": bool(
            run_state.get("pending_switch_candidate_advanced_during_pending")
        ),
        "candidate_completed_gated_cycle_during_pending": bool(
            run_state.get("pending_switch_candidate_completed_gated_cycle_during_pending")
        ),
        "candidate_current_raw": candidate_current_raw,
        "candidate_carryover_start_raw": int(
            run_state.get("pending_switch_candidate_carryover_start_raw") or 0
        ),
        "switch_forced": bool(run_state.get("pending_switch_forced")),
    }
    decision = classify_handoff(pending_state)

    carryover_start_shown = int(run_state.get("pending_switch_candidate_carryover_start_shown") or 0)
    carryover_start_raw = int(run_state.get("pending_switch_candidate_carryover_start_raw") or 0)
    candidate_delta_shown = max(0, candidate_current_raw - carryover_start_raw)
    candidate_delta_raw = max(0, candidate_current_raw - carryover_start_raw)
    rationale = decision.rationale if isinstance(decision.rationale, dict) else {}
    if decision.kind == "alternate_limb" and LOW_FPS_SAFE_MODE_ENABLED and bool(
        run_state.get("low_fps_mode_active")
    ):
        decision = HandoffDecision(
            kind="same_exercise",
            rationale={
                **rationale,
                "low_fps_safeguard": True,
                "original_kind": "alternate_limb",
            },
        )
        rationale = decision.rationale if isinstance(decision.rationale, dict) else {}
    if decision.kind == "alternate_limb" and rationale.get("rule") in (
        "forced_switch_mirrored_candidate_ready_or_delta",
        "mirrored_incumbent_stopped",
    ) and (
        bool(rationale.get("candidateReadyAtStart"))
        or rationale.get("rule") == "mirrored_incumbent_stopped"
    ):
        # When the mirrored candidate was already ready before pending observation
        # started, or the incumbent stopped on a mirrored limb switch, count the
        # candidate's full calibrated reps as alternate exercise carryover.
        candidate_delta_shown = max(0, candidate_current_raw)
        candidate_delta_raw = max(0, candidate_current_raw)
    if decision.kind == "alternate_limb":
        target_display_shown = cumulative_shown + candidate_delta_shown
        target_display_raw = cumulative_raw + candidate_delta_raw
    else:
        target_display_shown = cumulative_shown
        target_display_raw = cumulative_raw

    # Handoff must never reduce the displayed count, with or without carryover.
    target_display_shown = max(target_display_shown, cumulative_shown)
    target_display_raw = max(target_display_raw, cumulative_raw)

    before_activation_shown = cumulative_shown
    run_state["rep_count_offset"] = target_display_shown - candidate_current_raw
    run_state["rep_count_raw_offset"] = target_display_raw - candidate_current_raw
    after_activation_shown = candidate_current_raw + int(run_state["rep_count_offset"])
    assert after_activation_shown == target_display_shown
    assert after_activation_shown >= before_activation_shown
    if decision.kind in ("same_exercise", "ambiguous"):
        assert after_activation_shown == cumulative_shown

    run_state["pending_switch_handoff_kind"] = decision.kind
    run_state["pending_switch_handoff_rationale"] = decision.rationale
    run_state["pending_switch_handoff_candidate_delta_shown"] = candidate_delta_shown
    run_state["pending_switch_handoff_candidate_delta_raw"] = candidate_delta_raw
    run_state["pending_switch_handoff_candidate_current_shown"] = candidate_current_shown
    run_state["pending_switch_handoff_candidate_current_raw"] = candidate_current_raw
    run_state["pending_switch_handoff_cumulative_shown_before"] = cumulative_shown
    run_state["pending_switch_handoff_cumulative_raw_before"] = cumulative_raw
    run_state["pending_switch_handoff_target_shown"] = target_display_shown
    run_state["pending_switch_handoff_target_raw"] = target_display_raw
    run_state["pending_switch_handoff_incumbent_motion_span_deg"] = float(
        run_state.get("pending_switch_incumbent_motion_span_deg") or 0.0
    )
    run_state["pending_switch_handoff_candidate_pending_rom_estimate_deg"] = float(
        run_state.get("pending_switch_candidate_pending_rom_estimate_deg") or 0.0
    )
    run_state["pending_switch_handoff_switch_forced"] = bool(
        run_state.get("pending_switch_forced")
    )
    run_state["selected_angle"] = new_angle
    run_state["selected_config"] = COMMON_ANGLES[new_angle]
    run_state["peak_detector"] = detector
    run_state["selection_last_switch_at"] = switched_at
    run_state["selection_reps_at_last_recal_switch"] = int(cumulative_shown)
    run_state["tracking_raw_at_joint_lock"] = candidate_current_raw
    _clear_pending_switch(run_state)
    _clear_handoff_observation(run_state)
    _mark_joint_activation_guard(run_state)


def _collect_joint_records(
    detectors_by_angle: dict[str, Any], variances: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for angle_key in COMMON_ANGLES:
        det = detectors_by_angle.get(angle_key)
        det_state = det.get_state() if det is not None else {}
        v = variances.get(angle_key) or {}
        records[angle_key] = {
            "repCount": int(det_state.get("repCount") or 0),
            "avgPeak": (
                float(det_state["calibratedAvgPeak"])
                if det_state.get("calibratedAvgPeak") is not None
                else None
            ),
            "avgValley": (
                float(det_state["calibratedAvgValley"])
                if det_state.get("calibratedAvgValley") is not None
                else None
            ),
            "calibrationComplete": bool(det_state.get("calibrationComplete", False)),
            "medianWindowVariance": float(v.get("medianWindowVariance") or 0.0),
            "activeWindowCount": int(v.get("activeWindowCount") or 0),
            "smoothedRangeDeg": float(v.get("smoothedRangeDeg") or 0.0),
        }
    return records


def _selection_status_message(
    *,
    ready: bool,
    elapsed: float,
    n_frames: int,
    retry_at: Any,
    locked_this_frame: bool,
    selected_angle: Optional[str],
    run_state_selected: Optional[str],
    dom_ok: bool,
    leader_key: Optional[str],
    streak: int,
    rep_dom: dict[str, Any],
) -> str:
    if not ready:
        return (
            f"Observing movement... {n_frames}/{ANGLE_SELECTION_MIN_FRAMES} frames, "
            f"{elapsed:.1f}s / {ANGLE_SELECTION_MIN_SEC:.0f}s min"
        )
    now = time.time()
    if retry_at is not None and (
        now - float(retry_at) < ANGLE_SELECTION_RETRY_INTERVAL_SEC
    ):
        rem = ANGLE_SELECTION_RETRY_INTERVAL_SEC - (now - float(retry_at))
        return f"Unclear motion — retry in {rem:.0f}s (full ROM, steady reps)"
    if selected_angle or locked_this_frame or run_state_selected:
        return "Locked joint — calibrating reps"
    if ready and dom_ok and leader_key:
        share = float(rep_dom.get("leaderShare") or 0.0)
        return (
            f"Confirming primary joint {leader_key}: "
            f"streak {streak}/{ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES}, "
            f"rep share {share:.0%} (need >{ANGLE_SELECTION_DOMINANCE_FRACTION:.0%})"
        )
    if ready:
        total_r = int(rep_dom.get("totalReps") or 0)
        return (
            f"Mapping all joints... {total_r} reps across angles, "
            f"{n_frames} frames — need one joint >{ANGLE_SELECTION_DOMINANCE_FRACTION:.0%} "
            f"of reps and {ANGLE_SELECTION_MIN_LEADING_REPS}+ reps"
        )
    return f"Analyzing... ({n_frames} frames)"


def _format_angle_label(angle_key: str) -> str:
    return angle_key.replace("_", " ").title()


def _calibration_edge_flags(
    run_state: dict[str, Any],
    *,
    calibration_complete: bool,
    phase: str,
) -> tuple[bool, bool]:
    """One-shot flags for calibration start and lock transitions during tracking."""
    prev = run_state.get("_prev_calibration_complete")
    started = False
    locked = False
    if phase == "tracking":
        if prev is not False and not calibration_complete:
            started = True
        if prev is False and calibration_complete:
            locked = True
        run_state["_prev_calibration_complete"] = calibration_complete
    else:
        run_state["_prev_calibration_complete"] = None
    return started, locked


def _low_fps_result_fields(run_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "effective_fps": (
            float(run_state["low_fps_effective_fps"])
            if isinstance(run_state.get("low_fps_effective_fps"), (int, float))
            else None
        ),
        "frame_interval_p50_ms": (
            float(run_state["low_fps_p50_ms"])
            if isinstance(run_state.get("low_fps_p50_ms"), (int, float))
            else None
        ),
        "frame_interval_p90_ms": (
            float(run_state["low_fps_p90_ms"])
            if isinstance(run_state.get("low_fps_p90_ms"), (int, float))
            else None
        ),
        "low_fps_safe_mode_active": bool(run_state.get("low_fps_mode_active")),
        "low_fps_safe_mode_changed": bool(run_state.get("low_fps_mode_changed_pulse")),
    }


def _idle_result(msg: str = "Click Start to begin") -> StepResult:
    d = DEFAULT_TUNING_PARAMS
    return StepResult(
        reps=0,
        reps_raw=0,
        tracked_joint=None,
        angle_3_point_value=None,
        target_landmarks=None,
        tuning_params=d,
        avg_peak=None,
        avg_valley=None,
        calibration_complete=False,
        peak_detector_state="—",
        smoothed_value=None,
        range_gate_open=True,
        rolling_range=None,
        calibration_target_reps=int(d["calibrationReps"]),
        calibration_certainty=0.0,
        calibration_certainty_target=float(d["calibrationCertainty"]),
        phase="idle",
        status_message=msg,
        tracking_detail_message="",
    )


class RepCounterSession:
    """Runs angle selection + peak-detector tracking on a stream of landmark frames."""

    def __init__(
        self,
        *,
        tuning_params: Optional[dict[str, Any]] = None,
        use_pose_filter: bool = True,
        auto_started: bool = False,
        instrumentation_settings: Optional[RepInstrumentationSettings] = None,
    ) -> None:
        self._use_filter = use_pose_filter
        self._pose_pipeline = PoseFilterPipeline() if use_pose_filter else None
        self._auto_started = auto_started
        self._instrumentation_settings = instrumentation_settings
        self._instr_sink: Optional[RepInstrumentationSink] = None
        if instrumentation_settings is not None and instrumentation_settings.enabled:
            self._instr_sink = RepInstrumentationSink(instrumentation_settings)
        self._run_state: dict[str, Any] = {}
        self._last_smoothed_landmarks: Optional[list[dict]] = None
        self.reset(tuning_params=tuning_params)
        if auto_started:
            self.set_started()

    def reset(self, *, tuning_params: Optional[dict[str, Any]] = None) -> None:
        """Full reset (new exercise or second Start in visualizer)."""
        if self._instr_sink is not None:
            self._instr_sink.flush()
        tp = tuning_params if tuning_params is not None else DEFAULT_TUNING_PARAMS
        self._run_state = {
            "started": bool(self._auto_started),
            "started_at": time.time() if self._auto_started else None,
            "frame_buffer": deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES),
            "selection_last_attempt": None,
            "selection_last_reevaluate_at": None,
            "selection_last_switch_at": None,
            "selection_reps_at_last_recal_switch": None,
            "fallback_armed": False,
            "fallback_bad_streak_started_at": None,
            "fallback_bad_streak_paused_total_s": 0.0,
            "fallback_bad_streak_pause_anchor": None,
            "selection_detectors_by_angle": {},
            "joint_motion_states": _build_joint_motion_states(dict(tp)),
            "selection_dominance_key": None,
            "selection_dominance_streak": 0,
            "selected_angle": None,
            "selected_config": None,
            "peak_detector": None,
            "rep_count_offset": 0,
            "rep_count_raw_offset": 0,
            "pending_switch_angle": None,
            "pending_switch_detector": None,
            "selected_raw_last_at_reeval": 0,
            "selected_raw_stale_reeval_streak": 0,
            "tracking_raw_at_joint_lock": 0,
            "selected_range_gate_closed_streak": 0,
            "tuning_params": dict(tp),
            "buffer_list_cache": {"signature": None, "data": []},
            "variance_cache": {"signature": None, "include_debug": False, "data": {}},
            "selection_angle_histories": {
                ak: deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES) for ak in COMMON_ANGLES
            },
            "frame_interval_window": deque(maxlen=LOW_FPS_INTERVAL_WINDOW_FRAMES),
            "last_step_monotonic_s": None,
            "low_fps_mode_active": False,
            "low_fps_enter_streak": 0,
            "low_fps_exit_streak": 0,
            "low_fps_mode_changed_pulse": False,
            "low_fps_p50_ms": None,
            "low_fps_p90_ms": None,
            "low_fps_gaps_over_200ms": 0,
            "low_fps_effective_fps": None,
            "_prev_calibration_complete": None,
        }
        _init_ledger_guard_state(self._run_state)
        _clear_pending_switch(self._run_state)
        _clear_handoff_observation(self._run_state)
        if self._pose_pipeline is not None:
            self._pose_pipeline = PoseFilterPipeline()
        self._sync_detector_instrumentation_flags()

    def set_started(self, wall_time: Optional[float] = None) -> None:
        """Mark session as started (visualizer Start button)."""
        self._run_state["started"] = True
        self._run_state["started_at"] = wall_time if wall_time is not None else time.time()
        self._run_state["selection_last_attempt"] = None
        self._run_state["selection_last_reevaluate_at"] = None
        self._run_state["selection_last_switch_at"] = None
        self._run_state["fallback_armed"] = False
        self._run_state["fallback_bad_streak_started_at"] = None
        self._run_state["fallback_bad_streak_paused_total_s"] = 0.0
        self._run_state["fallback_bad_streak_pause_anchor"] = None
        self._run_state["joint_motion_states"] = _build_joint_motion_states(
            self._run_state["tuning_params"]
        )
        self._run_state["selection_detectors_by_angle"] = {
            k: s.detector for k, s in self._run_state["joint_motion_states"].items()
        }
        self._run_state["selection_dominance_key"] = None
        self._run_state["selection_dominance_streak"] = 0
        self._run_state["rep_count_offset"] = 0
        self._run_state["rep_count_raw_offset"] = 0
        _clear_pending_switch(self._run_state)
        self._run_state["selected_raw_last_at_reeval"] = 0
        self._run_state["selected_raw_stale_reeval_streak"] = 0
        self._run_state["tracking_raw_at_joint_lock"] = 0
        self._run_state["selected_range_gate_closed_streak"] = 0
        self._run_state["buffer_list_cache"] = {"signature": None, "data": []}
        self._run_state["variance_cache"] = {"signature": None, "include_debug": False, "data": {}}
        self._run_state["selection_angle_histories"] = {
            ak: deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES) for ak in COMMON_ANGLES
        }
        self._run_state["frame_interval_window"] = deque(maxlen=LOW_FPS_INTERVAL_WINDOW_FRAMES)
        self._run_state["last_step_monotonic_s"] = None
        self._run_state["low_fps_mode_active"] = False
        self._run_state["low_fps_enter_streak"] = 0
        self._run_state["low_fps_exit_streak"] = 0
        self._run_state["low_fps_mode_changed_pulse"] = False
        self._run_state["low_fps_p50_ms"] = None
        self._run_state["low_fps_p90_ms"] = None
        self._run_state["low_fps_gaps_over_200ms"] = 0
        self._run_state["low_fps_effective_fps"] = None
        _init_ledger_guard_state(self._run_state)
        self._sync_detector_instrumentation_flags()
        _clear_handoff_observation(self._run_state)

    def clear_tracking_keep_started(self) -> None:
        """Second Start click: clear selection/tracking but keep started=True."""
        self._run_state["_prev_calibration_complete"] = None
        self._run_state["selected_angle"] = None
        self._run_state["selected_config"] = None
        self._run_state["peak_detector"] = None
        self._run_state["frame_buffer"] = deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES)
        self._run_state["selection_last_attempt"] = None
        self._run_state["selection_last_reevaluate_at"] = None
        self._run_state["selection_last_switch_at"] = None
        self._run_state["fallback_armed"] = False
        self._run_state["fallback_bad_streak_started_at"] = None
        self._run_state["fallback_bad_streak_paused_total_s"] = 0.0
        self._run_state["fallback_bad_streak_pause_anchor"] = None
        self._run_state["joint_motion_states"] = _build_joint_motion_states(
            self._run_state["tuning_params"]
        )
        self._run_state["selection_detectors_by_angle"] = {
            k: s.detector for k, s in self._run_state["joint_motion_states"].items()
        }
        self._run_state["selection_dominance_key"] = None
        self._run_state["selection_dominance_streak"] = 0
        self._run_state["rep_count_offset"] = 0
        self._run_state["rep_count_raw_offset"] = 0
        _clear_pending_switch(self._run_state)
        self._run_state["selected_raw_last_at_reeval"] = 0
        self._run_state["selected_raw_stale_reeval_streak"] = 0
        self._run_state["tracking_raw_at_joint_lock"] = 0
        self._run_state["selected_range_gate_closed_streak"] = 0
        self._run_state["buffer_list_cache"] = {"signature": None, "data": []}
        self._run_state["variance_cache"] = {"signature": None, "include_debug": False, "data": {}}
        self._run_state["selection_angle_histories"] = {
            ak: deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES) for ak in COMMON_ANGLES
        }
        self._run_state["frame_interval_window"] = deque(maxlen=LOW_FPS_INTERVAL_WINDOW_FRAMES)
        self._run_state["last_step_monotonic_s"] = None
        self._run_state["low_fps_mode_active"] = False
        self._run_state["low_fps_enter_streak"] = 0
        self._run_state["low_fps_exit_streak"] = 0
        self._run_state["low_fps_mode_changed_pulse"] = False
        self._run_state["low_fps_p50_ms"] = None
        self._run_state["low_fps_p90_ms"] = None
        self._run_state["low_fps_gaps_over_200ms"] = 0
        self._run_state["low_fps_effective_fps"] = None
        _init_ledger_guard_state(self._run_state)
        self._sync_detector_instrumentation_flags()
        _clear_handoff_observation(self._run_state)

    @property
    def started(self) -> bool:
        return bool(self._run_state.get("started"))

    @property
    def last_smoothed_landmarks(self) -> Optional[list[dict]]:
        """Landmarks after temporal filtering from the last successful ``step_landmarks`` call."""
        return self._last_smoothed_landmarks

    @staticmethod
    def _buffer_signature(frame_buffer: deque) -> tuple[int, int, int]:
        n = len(frame_buffer)
        if n <= 0:
            return (0, 0, 0)
        return (n, id(frame_buffer[0]), id(frame_buffer[-1]))

    def _buffer_as_list(self, rs: dict[str, Any], frame_buffer: deque) -> list[list[dict]]:
        sig = self._buffer_signature(frame_buffer)
        cache = rs.get("buffer_list_cache") or {}
        if cache.get("signature") == sig:
            data = cache.get("data")
            if isinstance(data, list):
                return data
        data = list(frame_buffer)
        rs["buffer_list_cache"] = {"signature": sig, "data": data}
        return data

    def _get_variances(
        self,
        rs: dict[str, Any],
        frame_buffer: deque,
        *,
        include_debug: bool = False,
        fallback_armed: bool = False,
    ) -> dict[str, dict[str, Any]]:
        sig = (self._buffer_signature(frame_buffer), bool(fallback_armed))
        cache = rs.get("variance_cache") or {}
        if (
            cache.get("signature") == sig
            and (bool(cache.get("include_debug")) or not include_debug)
            and isinstance(cache.get("data"), dict)
        ):
            hit_streak = int(rs.get("debug_variance_cache_hit_streak") or 0) + 1
            rs["debug_variance_cache_hit_streak"] = hit_streak
            rs["debug_last_variance_sig"] = sig
            return cache["data"]
        rs["debug_variance_cache_hit_streak"] = 0
        variances = compute_angle_variances_from_buffer(
            self._buffer_as_list(rs, frame_buffer),
            include_debug=include_debug,
            fallback_armed=fallback_armed,
            angle_histories={
                ak: list(hist)
                for ak, hist in (rs.get("selection_angle_histories") or {}).items()
                if isinstance(hist, deque)
            },
        )
        rs["variance_cache"] = {
            "signature": sig,
            "include_debug": include_debug,
            "data": variances,
        }
        rs["debug_last_variance_sig"] = sig
        return variances

    def _sync_detector_instrumentation_flags(self) -> None:
        rs = self._run_state
        states = rs.get("joint_motion_states") or {}
        sdba = {k: s.detector for k, s in states.items()} if states else rs.get("selection_detectors_by_angle") or {}
        en = self._instr_sink is not None
        for det in sdba.values():
            try:
                det.instrumentation_enabled = bool(en)
            except Exception:
                pass
        pd = rs.get("peak_detector")
        if pd is not None:
            try:
                pd.instrumentation_enabled = bool(en)
            except Exception:
                pass

    def _instrumentation_should_emit(self, trace_context: Optional[dict[str, Any]]) -> bool:
        if self._instr_sink is None or not trace_context:
            return False
        uid = trace_context.get("user_uid")
        if not uid:
            return False
        settings = self._instr_sink.settings
        if not settings.should_emit_for_user(str(uid)):
            return False
        fi = int(trace_context.get("frame_idx", 0) or 0)
        if not settings.should_sample_frame(fi):
            return False
        return True

    def _instr_emit(
        self, trace_context: Optional[dict[str, Any]], event: dict[str, Any]
    ) -> None:
        if not self._instrumentation_should_emit(trace_context):
            return
        if self._instr_sink is None or trace_context is None:
            return
        self._instr_sink.emit(merge_trace(event, trace_context))

    def _drain_detector_instrumentation(
        self, trace_context: Optional[dict[str, Any]], det: Any
    ) -> None:
        if det is None or not self._instrumentation_should_emit(trace_context):
            return
        try:
            events = det.drain_instrumentation_events()
        except Exception:
            return
        for ev in events:
            self._instr_emit(trace_context, ev)

    def _finalize_instrumentation_selection(
        self,
        trace_context: Optional[dict[str, Any]],
        out: StepResult,
        *,
        raw_angle_values: dict[str, Optional[float]],
        angle_values: dict[str, Optional[float]],
        sdba: dict[str, Any],
        leader_key: Optional[str],
    ) -> None:
        if not self._instrumentation_should_emit(trace_context):
            return
        seen: set[int] = set()
        for det in sdba.values():
            i = id(det)
            if i in seen:
                continue
            seen.add(i)
            self._drain_detector_instrumentation(trace_context, det)
        lk = leader_key if isinstance(leader_key, str) else None
        self._instr_emit(
            trace_context,
            {
                "event": "frame_snapshot",
                "phase": out.phase,
                "tracked_joint": out.tracked_joint,
                "leader_key": out.leader_key,
                "reps": out.reps,
                "reps_raw": out.reps_raw,
                "peak_detector_state": out.peak_detector_state,
                "range_gate_open": out.range_gate_open,
                "rolling_range": out.rolling_range,
                "calibration_complete": out.calibration_complete,
                "calibration_started": out.calibration_started,
                "calibration_locked": out.calibration_locked,
                "tracked_joint_changed": out.tracked_joint_changed,
                "calibration_certainty": out.calibration_certainty,
                "calibration_target_reps": out.calibration_target_reps,
                "smoothed_angle": out.smoothed_value,
                "effective_fps": out.effective_fps,
                "frame_interval_p50_ms": out.frame_interval_p50_ms,
                "frame_interval_p90_ms": out.frame_interval_p90_ms,
                "frame_interval_gaps_over_200ms": self._run_state.get("low_fps_gaps_over_200ms"),
                "low_fps_safe_mode_active": out.low_fps_safe_mode_active,
                "low_fps_safe_mode_changed": out.low_fps_safe_mode_changed,
                "raw_angle": raw_angle_values.get(lk) if lk else None,
                "filtered_angle": angle_values.get(lk) if lk else None,
                "deadband_angle": None,
                "angles_raw_compact": {k: v for k, v in raw_angle_values.items() if v is not None},
                "angles_filtered_compact": {
                    k: v for k, v in angle_values.items() if v is not None
                },
            },
        )

    def _finalize_instrumentation_tracking(
        self,
        trace_context: Optional[dict[str, Any]],
        out: StepResult,
        *,
        raw_angle_value: Optional[float],
        filtered_angle_value: Optional[float],
        selected_output: Optional[dict[str, Any]],
        sdba: dict[str, Any],
        peak_detector: Any,
        sel_angle: Optional[str],
    ) -> None:
        if not self._instrumentation_should_emit(trace_context):
            return
        if (
            out.phase == "tracking"
            and sel_angle
            and filtered_angle_value is None
            and self._last_smoothed_landmarks is not None
            and isinstance(sel_angle, str)
            and sel_angle in COMMON_ANGLES
        ):
            reason, detail = _diagnose_missing_angle(
                COMMON_ANGLES[sel_angle], self._last_smoothed_landmarks
            )
            self._instr_emit(
                trace_context,
                {
                    "event": "pose_dropped",
                    "reason": reason,
                    "reason_detail": detail,
                    "tracked_joint": sel_angle,
                },
            )
        seen: set[int] = set()
        for det in sdba.values():
            i = id(det)
            if i in seen:
                continue
            seen.add(i)
            self._drain_detector_instrumentation(trace_context, det)
        if peak_detector is not None and id(peak_detector) not in seen:
            self._drain_detector_instrumentation(trace_context, peak_detector)
        feed_v: Optional[float] = None
        smooth_v: Optional[float] = None
        if selected_output:
            fv = selected_output.get("feedValue")
            sv = selected_output.get("smoothedValue")
            if fv is not None:
                feed_v = float(fv)
            if sv is not None:
                smooth_v = float(sv)
        self._instr_emit(
            trace_context,
            {
                "event": "frame_snapshot",
                "phase": out.phase,
                "tracked_joint": out.tracked_joint,
                "reps": out.reps,
                "reps_raw": out.reps_raw,
                "peak_detector_state": out.peak_detector_state,
                "range_gate_open": out.range_gate_open,
                "rolling_range": out.rolling_range,
                "calibration_complete": out.calibration_complete,
                "calibration_started": out.calibration_started,
                "calibration_locked": out.calibration_locked,
                "tracked_joint_changed": out.tracked_joint_changed,
                "calibration_certainty": out.calibration_certainty,
                "calibration_target_reps": out.calibration_target_reps,
                "effective_fps": out.effective_fps,
                "frame_interval_p50_ms": out.frame_interval_p50_ms,
                "frame_interval_p90_ms": out.frame_interval_p90_ms,
                "frame_interval_gaps_over_200ms": self._run_state.get("low_fps_gaps_over_200ms"),
                "low_fps_safe_mode_active": out.low_fps_safe_mode_active,
                "low_fps_safe_mode_changed": out.low_fps_safe_mode_changed,
                "raw_angle": raw_angle_value,
                "filtered_angle": filtered_angle_value,
                "deadband_angle": feed_v,
                "smoothed_angle": smooth_v,
                "handoff_kind": self._run_state.get("pending_switch_handoff_kind"),
                "handoff_rationale": self._run_state.get("pending_switch_handoff_rationale"),
                "handoff_candidate_delta_shown": self._run_state.get(
                    "pending_switch_handoff_candidate_delta_shown"
                ),
                "handoff_candidate_delta_raw": self._run_state.get(
                    "pending_switch_handoff_candidate_delta_raw"
                ),
                "handoff_candidate_current_shown": self._run_state.get(
                    "pending_switch_handoff_candidate_current_shown"
                ),
                "handoff_candidate_current_raw": self._run_state.get(
                    "pending_switch_handoff_candidate_current_raw"
                ),
                "handoff_cumulative_shown_before": self._run_state.get(
                    "pending_switch_handoff_cumulative_shown_before"
                ),
                "handoff_cumulative_raw_before": self._run_state.get(
                    "pending_switch_handoff_cumulative_raw_before"
                ),
                "handoff_target_shown": self._run_state.get("pending_switch_handoff_target_shown"),
                "handoff_target_raw": self._run_state.get("pending_switch_handoff_target_raw"),
                "handoff_incumbent_motion_span_deg": self._run_state.get(
                    "pending_switch_handoff_incumbent_motion_span_deg"
                ),
                "handoff_candidate_pending_rom_estimate_deg": self._run_state.get(
                    "pending_switch_handoff_candidate_pending_rom_estimate_deg"
                ),
                "handoff_switch_forced": self._run_state.get("pending_switch_handoff_switch_forced"),
            },
        )

    def step_landmarks(
        self,
        landmarks: Optional[list[dict]],
        *,
        timestamp_ms: Optional[float] = None,
        wall_time_s: Optional[float] = None,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> StepResult:
        """
        Process one frame of 17 COCO landmarks (after any resolution scaling).

        ``landmarks`` may be None when no pose is detected.
        """
        if not self._run_state.get("started"):
            return _idle_result()

        now = wall_time_s if wall_time_s is not None else time.time()
        ts = timestamp_ms if timestamp_ms is not None else now * 1000.0
        t_step_start = time.perf_counter()

        rs = self._run_state
        tuning_params = rs["tuning_params"]
        default_tuning = DEFAULT_TUNING_PARAMS
        _update_low_fps_health(rs)

        if not landmarks:
            self._last_smoothed_landmarks = None
            self._instr_emit(
                trace_context,
                {"event": "pose_dropped", "reason": "no_pose", "reason_detail": {}},
            )
            if rs.get("selected_angle") is not None and rs.get("peak_detector") is not None:
                tr = self._build_tracking_step_result(rs, None)
                out = replace(tr, status_message=f"No pose - {tr.status_message}")
                self._finalize_instrumentation_tracking(
                    trace_context,
                    out,
                    raw_angle_value=None,
                    filtered_angle_value=None,
                    selected_output=None,
                    sdba=rs.get("selection_detectors_by_angle") or {},
                    peak_detector=rs.get("peak_detector"),
                    sel_angle=rs.get("selected_angle"),
                )
                return out
            out = self._no_pose_step_result(
                tuning_params=tuning_params,
                tracked_joint=rs.get("selected_angle"),
                default_tuning=default_tuning,
                phase="selecting" if rs.get("selected_angle") is None else "tracking",
                run_state=rs,
            )
            self._instr_emit(
                trace_context,
                {
                    "event": "frame_snapshot",
                    "phase": out.phase,
                    "tracked_joint": out.tracked_joint,
                    "reps": out.reps,
                    "reps_raw": out.reps_raw,
                    "reason": "no_pose",
                    "effective_fps": out.effective_fps,
                    "frame_interval_p50_ms": out.frame_interval_p50_ms,
                    "frame_interval_p90_ms": out.frame_interval_p90_ms,
                    "frame_interval_gaps_over_200ms": rs.get("low_fps_gaps_over_200ms"),
                    "low_fps_safe_mode_active": out.low_fps_safe_mode_active,
                    "low_fps_safe_mode_changed": out.low_fps_safe_mode_changed,
                },
            )
            return out

        raw_landmarks = landmarks

        if self._pose_pipeline is not None:
            lm = self._pose_pipeline.process(landmarks, ts)
        else:
            lm = landmarks
        self._last_smoothed_landmarks = lm

        frame_buffer = rs["frame_buffer"]
        frame_buffer.append(lm)
        started_at = float(rs.get("started_at") or 0.0)
        elapsed = now - started_at
        perf_ms: dict[str, float] = {
            "detector_update_ms": 0.0,
            "variance_ms": 0.0,
            "selection_logic_ms": 0.0,
        }

        joint_states: dict[str, JointMotionState] = rs.get("joint_motion_states") or {}
        if not joint_states:
            joint_states = _build_joint_motion_states(tuning_params)
            rs["joint_motion_states"] = joint_states
        sdba: dict[str, Any] = {k: s.detector for k, s in joint_states.items()}
        rs["selection_detectors_by_angle"] = sdba
        for ak, det in sdba.items():
            try:
                setattr(det, "debug_label", ak)
            except Exception:
                pass
        self._sync_detector_instrumentation_flags()

        selected_angle = rs["selected_angle"]
        if isinstance(selected_angle, str) and selected_angle in sdba:
            selected_detector = sdba[selected_angle]
            if rs.get("peak_detector") is not selected_detector:
                rs["peak_detector"] = selected_detector

        if selected_angle is None:
            selecting_angle_values: dict[str, Optional[float]] = {}
            selecting_raw_angle_values: dict[str, Optional[float]] = {}
            selecting_detector_outputs: dict[str, dict[str, Any]] = {}
            angle_histories: dict[str, deque] = rs.get("selection_angle_histories") or {}
            if not angle_histories:
                angle_histories = {
                    ak: deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES) for ak in COMMON_ANGLES
                }
                rs["selection_angle_histories"] = angle_histories
            t_det = time.perf_counter()
            for ak, cfg in COMMON_ANGLES.items():
                raw_av = calculate_from_type(cfg["type"], cfg, raw_landmarks)
                selecting_raw_angle_values[ak] = raw_av
                val = calculate_from_type(cfg["type"], cfg, lm)
                selecting_angle_values[ak] = val
                conf = get_min_confidence_for_landmarks(lm, _angle_landmarks(cfg))
                upd = update_joint_motion_state(
                    joint_states[ak],
                    val,
                    conf,
                    int(ts),
                    min_confidence=_angle_confidence_threshold(cfg),
                )
                selecting_detector_outputs[ak] = upd.get("detectorOutput") or {}
            _update_angle_histories_for_frame(
                angle_histories,
                selecting_angle_values,
                lm,
            )
            perf_ms["detector_update_ms"] = (time.perf_counter() - t_det) * 1000.0
            _apply_fps_scaled_peak_distance(rs, joint_states)

            rep_counts_sel = {ak: d.get_rep_count() for ak, d in sdba.items()}
            fallback_armed = bool(rs.get("fallback_armed"))
            rep_counts_for_selection = {
                ak: int(v)
                for ak, v in rep_counts_sel.items()
                if fallback_armed or not is_fallback_angle(ak)
            }
            rep_dom = summarize_rep_dominance(rep_counts_for_selection)
            t_var = time.perf_counter()
            variances = self._get_variances(
                rs,
                frame_buffer,
                include_debug=False,
                fallback_armed=fallback_armed,
            )
            perf_ms["variance_ms"] = (time.perf_counter() - t_var) * 1000.0
            joint_records = _collect_joint_records(sdba, variances)
            score_by_joint_all: dict[str, tuple[float, dict[str, Any]]] = {}
            variances_all = self._get_variances(
                rs,
                frame_buffer,
                include_debug=False,
                fallback_armed=True,
            )
            for angle_key, state in joint_states.items():
                score_by_joint_all[angle_key] = compute_joint_recalibration_score(
                    state,
                    variances_all.get(angle_key),
                    int(ts),
                )

            cal_reps_target = int(
                tuning_params.get(
                    "calibrationReps",
                    default_tuning["calibrationReps"],
                )
            )
            total_selection_reps = int(rep_dom.get("totalReps") or 0)
            rep_evidence_ready = total_selection_reps >= cal_reps_target
            frame_ready = len(frame_buffer) >= ANGLE_SELECTION_MIN_FRAMES
            ready = frame_ready and (
                elapsed >= ANGLE_SELECTION_MIN_SEC or rep_evidence_ready
            )
            last_att = rs.get("selection_last_attempt")
            unclear_motion_now = bool(
                last_att is not None
                and (now - float(last_att)) < ANGLE_SELECTION_RETRY_INTERVAL_SEC
            )
            stale_now = bool(elapsed >= ANGLE_SELECTION_MIN_SEC and rs.get("selected_angle") is None)
            best_primary_selection_score = max(
                (
                    float(score)
                    for angle_key, (score, _) in score_by_joint_all.items()
                    if not is_fallback_angle(angle_key)
                ),
                default=0.0,
            )
            low_score_now = bool(best_primary_selection_score < FALLBACK_Y_LOW_SCORE_THRESHOLD)
            not_recalibrating_now = bool(
                rs.get("pending_switch_angle") is None
                and rs.get("handoff_observation_candidate_angle") is None
            )
            fallback_elapsed = _update_fallback_bad_streak(
                rs,
                now_s=now,
                conditions_met=(
                    stale_now
                    and low_score_now
                    and unclear_motion_now
                    and not_recalibrating_now
                ),
                paused=False,
            )
            if bool(rs.get("fallback_armed")):
                if _has_primary_recovery_candidate(score_by_joint_all):
                    rs["fallback_armed"] = False
                    _reset_fallback_streak(rs)
            else:
                fb_key, _, _ = _best_fallback_candidate(score_by_joint_all)
                if (
                    fb_key is not None
                    and fallback_elapsed >= FALLBACK_Y_ARM_WINDOW_SEC
                    and FALLBACK_Y_ARM_WINDOW_SEC > 0.0
                ):
                    rs["fallback_armed"] = True

            fallback_armed = bool(rs.get("fallback_armed"))
            can_try = ready and (
                last_att is None
                or (now - float(last_att)) >= ANGLE_SELECTION_RETRY_INTERVAL_SEC
            )
            sel_thresholds = _selection_dominance_thresholds(elapsed)
            dom_ok_joint = dominance_conditions_met(
                variances,
                rep_dom,
                dominance_fraction=float(sel_thresholds["joint_fraction"]),
                min_leading_reps=int(sel_thresholds["min_leading_reps"]),
                fallback_armed=fallback_armed,
            )
            leader_key: Optional[str] = (
                rep_dom.get("leaderKey") if isinstance(rep_dom.get("leaderKey"), str) else None
            )
            family_dom = _family_rep_dominance(rep_counts_for_selection)
            family_leader = (
                family_dom.get("leaderFamily")
                if isinstance(family_dom.get("leaderFamily"), str)
                else None
            )
            dom_ok_family = bool(
                family_leader
                and float(family_dom.get("leaderShare") or 0.0) > float(sel_thresholds["family_fraction"])
                and int(family_dom.get("leaderReps") or 0) >= int(sel_thresholds["min_leading_reps"])
            )
            selected_from_family = False
            dominance_token: Optional[str] = None
            if dom_ok_joint and leader_key is not None:
                dominance_token = leader_key
            elif dom_ok_family and family_leader is not None:
                family_joint = _select_family_joint_candidate(
                    leader_family=family_leader,
                    rep_counts=rep_counts_sel,
                    variances=variances,
                    elapsed_s=elapsed,
                )
                if family_joint is not None:
                    leader_key = family_joint
                    dominance_token = f"FAMILY::{family_leader}"
                    selected_from_family = True

            if dominance_token and leader_key:
                if rs.get("selection_dominance_key") == dominance_token:
                    rs["selection_dominance_streak"] = int(
                        rs.get("selection_dominance_streak") or 0
                    ) + 1
                else:
                    rs["selection_dominance_key"] = dominance_token
                    rs["selection_dominance_streak"] = 1
            else:
                rs["selection_dominance_key"] = None
                rs["selection_dominance_streak"] = 0

            streak = int(rs.get("selection_dominance_streak") or 0)
            dominance_streak_required = int(sel_thresholds["streak_required"])
            if rep_evidence_ready:
                dominance_streak_required = max(
                    int(sel_thresholds["min_leading_reps"]),
                    dominance_streak_required // 2,
                )
            lock_from_dominance = (
                frame_ready
                and (elapsed >= ANGLE_SELECTION_MIN_SEC or rep_evidence_ready)
                and (dom_ok_joint or selected_from_family)
                and streak >= dominance_streak_required
            )
            # If we already observe enough reps during selection, allow earlier variance lock
            # instead of waiting the full fallback timeout.
            variance_fallback_ready = ready and (
                elapsed >= ANGLE_SELECTION_VARIANCE_FALLBACK_SEC
                or total_selection_reps >= cal_reps_target
            )
            locked_this_frame = False
            t_sel = time.perf_counter()

            if lock_from_dominance and leader_key:
                _apply_locked_tracking(
                    rs,
                    leader_key,
                    None,
                    tuning_params,
                    selection_detector=sdba.get(leader_key),
                    initial_angle_value=selecting_angle_values.get(leader_key),
                )
                locked_this_frame = True
                self._sync_detector_instrumentation_flags()
            elif can_try and variance_fallback_ready:
                buf_list = self._buffer_as_list(rs, frame_buffer)
                result = determine_best_angle(
                    buf_list,
                    variances=variances,
                    include_debug=False,
                    fallback_armed=fallback_armed,
                )
                tuning_params = result.get("tuningParams") or DEFAULT_TUNING_PARAMS
                rs["tuning_params"] = tuning_params
                sel = result.get("selectedAngle")
                src = str(result.get("source") or "")
                selected_key = sel if isinstance(sel, str) and sel in COMMON_ANGLES else None
                variance_ok = selected_key is not None and src == "variance"
                if variance_ok and selected_key is not None:
                    _apply_locked_tracking(
                        rs,
                        selected_key,
                        None,
                        tuning_params,
                        selection_detector=sdba.get(selected_key),
                        initial_angle_value=selecting_angle_values.get(selected_key),
                    )
                    locked_this_frame = True
                    self._sync_detector_instrumentation_flags()
                else:
                    relaxed_family_joint: Optional[str] = None
                    if dom_ok_family and family_leader is not None:
                        relaxed_family_joint = _select_family_joint_candidate(
                            leader_family=family_leader,
                            rep_counts=rep_counts_sel,
                            variances=variances,
                            elapsed_s=elapsed,
                        )
                    if relaxed_family_joint is not None:
                        _apply_locked_tracking(
                            rs,
                            relaxed_family_joint,
                            None,
                            tuning_params,
                            selection_detector=sdba.get(relaxed_family_joint),
                            initial_angle_value=selecting_angle_values.get(relaxed_family_joint),
                        )
                        locked_this_frame = True
                        self._sync_detector_instrumentation_flags()
                    else:
                        rs["selected_angle"] = None
                        rs["selected_config"] = None
                        rs["peak_detector"] = None
                        rs["selection_last_attempt"] = now
            perf_ms["selection_logic_ms"] = (time.perf_counter() - t_sel) * 1000.0

            retry_at = rs.get("selection_last_attempt")
            status = _selection_status_message(
                ready=ready,
                elapsed=elapsed,
                n_frames=len(frame_buffer),
                retry_at=retry_at,
                locked_this_frame=locked_this_frame,
                selected_angle=rs.get("selected_angle"),
                run_state_selected=rs.get("selected_angle"),
                dom_ok=bool(dom_ok_joint or selected_from_family),
                leader_key=leader_key,
                streak=streak,
                rep_dom=rep_dom,
            )
            perf_ms["session_total_ms"] = (time.perf_counter() - t_step_start) * 1000.0
            out = StepResult(
                reps=0,
                reps_raw=0,
                tracked_joint=rs.get("selected_angle"),
                angle_3_point_value=None,
                target_landmarks=None,
                tuning_params=dict(rs["tuning_params"]),
                avg_peak=None,
                avg_valley=None,
                calibration_complete=False,
                peak_detector_state="—",
                smoothed_value=None,
                range_gate_open=True,
                rolling_range=None,
                calibration_target_reps=int(
                    rs["tuning_params"].get(
                        "calibrationReps", default_tuning["calibrationReps"]
                    )
                ),
                calibration_certainty=0.0,
                calibration_certainty_target=float(
                    rs["tuning_params"].get(
                        "calibrationCertainty",
                        default_tuning["calibrationCertainty"],
                    )
                ),
                phase="selecting",
                status_message=status,
                tracking_detail_message="",
                **_low_fps_result_fields(rs),
                leader_key=leader_key if isinstance(leader_key, str) else None,
                selection_debug={
                    "rep_dom": rep_dom,
                    "dom_ok": bool(dom_ok_joint or selected_from_family),
                    "dom_ok_joint": bool(dom_ok_joint),
                    "dom_ok_family": bool(dom_ok_family),
                    "family_rep_dom": family_dom,
                    "selected_from_family": bool(selected_from_family),
                    "dominance_thresholds": sel_thresholds,
                    "joint_records": joint_records,
                    "perf_ms": perf_ms,
                },
            )
            self._finalize_instrumentation_selection(
                trace_context,
                out,
                raw_angle_values=selecting_raw_angle_values,
                angle_values=selecting_angle_values,
                sdba=sdba,
                leader_key=leader_key if isinstance(leader_key, str) else None,
            )
            return out

        # Tracking phase: update all joint motion states every frame.
        rs["tracking_step_count"] = int(rs.get("tracking_step_count") or 0) + 1
        angle_values: dict[str, Optional[float]] = {}
        detector_outputs: dict[str, dict[str, Any]] = {}
        variances: dict[str, dict[str, Any]] = {}
        joint_records: dict[str, Any] = {}
        rep_dom = {
            "totalReps": 0,
            "leaderKey": selected_angle,
            "leaderReps": 0,
            "leaderShare": 0.0,
        }
        active_detector: Optional[Any] = None
        raw_angle_val: Optional[float] = None
        angle_histories = rs.get("selection_angle_histories") or {}
        if not angle_histories:
            angle_histories = {
                ak: deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES) for ak in COMMON_ANGLES
            }
            rs["selection_angle_histories"] = angle_histories
        tracking_angle_values = {
            ak: calculate_from_type(cfg["type"], cfg, lm)
            for ak, cfg in COMMON_ANGLES.items()
        }
        _update_angle_histories_for_frame(angle_histories, tracking_angle_values, lm)
        _apply_fps_scaled_peak_distance(
            rs,
            joint_states,
            peak_detector=rs.get("peak_detector"),
        )
        t_det = time.perf_counter()
        for ak, cfg in COMMON_ANGLES.items():
            val = tracking_angle_values.get(ak)
            conf = get_min_confidence_for_landmarks(lm, _angle_landmarks(cfg))
            upd = update_joint_motion_state(
                joint_states[ak],
                val,
                conf,
                int(ts),
                min_confidence=_angle_confidence_threshold(cfg),
            )
            detector_outputs[ak] = upd.get("detectorOutput") or {}
            angle_values[ak] = val
            if DYNAMIC_RECALIBRATION_ENABLED:
                if isinstance(selected_angle, str) and ak == selected_angle:
                    if upd.get("advanced"):
                        rs["pending_switch_incumbent_advanced"] = True
                    if upd.get("gatedCycle"):
                        rs["pending_switch_incumbent_completed_gated_cycle_during_pending"] = True
                pending_angle_for_updates = rs.get("pending_switch_angle")
                if isinstance(pending_angle_for_updates, str) and ak == pending_angle_for_updates:
                    rs["pending_switch_observed"] = True
                    if upd.get("advanced"):
                        rs["pending_switch_candidate_advanced_during_pending"] = True
                    if upd.get("gatedCycle"):
                        rs["pending_switch_candidate_completed_gated_cycle_during_pending"] = True
                    pending_rolling = (upd.get("detectorOutput") or {}).get("rollingRange")
                    if isinstance(pending_rolling, (int, float)):
                        rs["pending_switch_candidate_pending_rom_estimate_deg"] = max(
                            float(rs.get("pending_switch_candidate_pending_rom_estimate_deg") or 0.0),
                            float(pending_rolling),
                        )
        perf_ms["detector_update_ms"] = (time.perf_counter() - t_det) * 1000.0
        if isinstance(selected_angle, str) and selected_angle in COMMON_ANGLES:
            cfg = COMMON_ANGLES[selected_angle]
            raw_angle_val = calculate_from_type(cfg["type"], cfg, raw_landmarks)
            active_detector = rs.get("peak_detector") or sdba.get(selected_angle)
            if active_detector is None:
                active_detector = sdba[selected_angle]
            rep_value = int(active_detector.get_rep_count() or 0) if active_detector is not None else 0
            rep_dom = {
                "totalReps": rep_value,
                "leaderKey": selected_angle,
                "leaderReps": rep_value,
                "leaderShare": 1.0 if rep_value > 0 else 0.0,
            }
        current_raw = _detector_raw_count(active_detector)
        cumulative_shown, cumulative_raw = _ledger_cumulative_reps(
            rs,
            active_detector,
            include_retroactive=True,
        )
        selected_output_for_eval = (
            detector_outputs.get(selected_angle) if isinstance(selected_angle, str) else None
        )
        selected_pose_score = 0.0
        selected_recent_range_for_gate = 0.0
        if isinstance(selected_angle, str) and selected_angle in joint_states:
            _, selected_pose_debug = compute_joint_recalibration_score(
                joint_states[selected_angle],
                None,
                int(ts),
            )
            selected_pose_score = float(selected_pose_debug.get("poseScore") or 0.0)
            selected_recent_range_for_gate = float(selected_pose_debug.get("recentRange") or 0.0)
        if isinstance(selected_output_for_eval, dict):
            rolling_range = selected_output_for_eval.get("rollingRange")
            if isinstance(rolling_range, (int, float)):
                selected_recent_range_for_gate = max(selected_recent_range_for_gate, float(rolling_range))
        if isinstance(selected_output_for_eval, dict):
            if bool(selected_output_for_eval.get("rangeGateOpen", True)):
                rs["selected_range_gate_closed_streak"] = 0
            else:
                rs["selected_range_gate_closed_streak"] = int(
                    rs.get("selected_range_gate_closed_streak") or 0
                ) + 1

        switched_to: Optional[str] = None
        if DYNAMIC_RECALIBRATION_ENABLED:
            pending_angle = (
                rs.get("pending_switch_angle")
                if isinstance(rs.get("pending_switch_angle"), str)
                else None
            )
            pending_detector = rs.get("pending_switch_detector")
            start_raw = rs.get("pending_switch_incumbent_raw_start")
            if isinstance(start_raw, int) and current_raw > start_raw:
                rs["pending_switch_incumbent_advanced"] = True
            if (
                pending_angle is not None
                and pending_angle != selected_angle
                and pending_angle in COMMON_ANGLES
                and pending_detector is not None
            ):
                if isinstance(selected_angle, str):
                    _update_pending_incumbent_motion(
                        rs,
                        tracking_angle_values.get(selected_angle),
                    )
                    selected_conf = get_min_confidence_for_landmarks(
                        lm, _angle_landmarks(COMMON_ANGLES[selected_angle])
                    )
                    selected_conf_min = _angle_confidence_threshold(COMMON_ANGLES[selected_angle])
                    if isinstance(selected_conf, (int, float)) and selected_conf >= selected_conf_min:
                        rs["pending_switch_incumbent_observable_during_pending"] = True
                if pending_angle in detector_outputs:
                    pending_calibrated = _is_detector_calibrated(
                        pending_detector, detector_outputs.get(pending_angle)
                    )
                else:
                    pending_calibrated = _is_detector_calibrated(pending_detector)
                pending_started_at_ms = int(rs.get("pending_switch_started_at_ms") or int(ts))
                rs["pending_switch_started_at_ms"] = pending_started_at_ms
                pending_elapsed_ms = int(ts) - pending_started_at_ms
                if pending_calibrated and pending_elapsed_ms >= PENDING_SWITCH_MIN_OBSERVATION_MS:
                    _activate_joint_switch(
                        rs,
                        sdba,
                        new_angle=pending_angle,
                        detector=pending_detector,
                        cumulative_shown=cumulative_shown,
                        cumulative_raw=cumulative_raw,
                        switched_at=now,
                    )
                    switched_to = pending_angle
                    selected_angle = pending_angle
                    cfg = COMMON_ANGLES[pending_angle]
                    raw_angle_val = calculate_from_type(cfg["type"], cfg, raw_landmarks)
                    self._sync_detector_instrumentation_flags()
                elif pending_elapsed_ms > PENDING_SWITCH_MAX_OBSERVATION_MS:
                    _activate_joint_switch(
                        rs,
                        sdba,
                        new_angle=pending_angle,
                        detector=pending_detector,
                        cumulative_shown=cumulative_shown,
                        cumulative_raw=cumulative_raw,
                        switched_at=now,
                    )
                    switched_to = pending_angle
                    selected_angle = pending_angle
                    cfg = COMMON_ANGLES[pending_angle]
                    raw_angle_val = calculate_from_type(cfg["type"], cfg, raw_landmarks)
                    self._sync_detector_instrumentation_flags()
            else:
                _clear_pending_switch(rs)
        elif rs.get("pending_switch_angle") is not None or rs.get("handoff_observation_candidate_angle") is not None:
            _clear_pending_switch(rs)
            _clear_handoff_observation(rs)
        last_re_eval = rs.get("selection_last_reevaluate_at")
        timer_re_eval_due = DYNAMIC_RECALIBRATION_ENABLED and (
            ANGLE_SELECTION_REEVALUATE_EVERY_SEC <= 0
            or last_re_eval is None
            or (now - float(last_re_eval)) >= ANGLE_SELECTION_REEVALUATE_EVERY_SEC
        )
        re_eval_due = False
        if timer_re_eval_due:
            rs["selection_last_reevaluate_at"] = now
            last_raw = int(rs.get("selected_raw_last_at_reeval") or 0)
            stale_reevals = int(rs.get("selected_raw_stale_reeval_streak") or 0)
            raw_advanced_since_last_eval = current_raw > last_raw
            if raw_advanced_since_last_eval:
                stale_reevals = 0
            else:
                stale_reevals += 1
            rs["selected_raw_last_at_reeval"] = current_raw
            rs["selected_raw_stale_reeval_streak"] = stale_reevals
            run_full_re_eval = should_run_full_recalibration(
                has_pending_switch=isinstance(rs.get("pending_switch_angle"), str),
                has_handoff_observation=isinstance(rs.get("handoff_observation_candidate_angle"), str),
                current_raw=current_raw,
                tracking_raw_at_joint_lock=int(rs.get("tracking_raw_at_joint_lock") or 0),
                post_lock_min_raw_reps=POST_LOCK_MIN_RAW_REPS,
                raw_advanced_since_last_eval=raw_advanced_since_last_eval,
                selected_recent_range=selected_recent_range_for_gate,
                selected_pose_score=selected_pose_score,
                selected_range_gate_closed_streak=int(rs.get("selected_range_gate_closed_streak") or 0),
                stale_switch_max_selected_recent_range_deg=STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG,
                stale_switch_min_closed_streak=STALE_SWITCH_MIN_CLOSED_STREAK,
            )
            re_eval_due = bool(run_full_re_eval)
        if re_eval_due:
            t_var = time.perf_counter()
            variances = self._get_variances(
                rs,
                frame_buffer,
                include_debug=False,
                fallback_armed=True,
            )
            perf_ms["variance_ms"] = (time.perf_counter() - t_var) * 1000.0
            now_ms = int(ts)
            score_by_joint: dict[str, tuple[float, dict[str, Any]]] = {}
            for angle_key, state in joint_states.items():
                score_by_joint[angle_key] = compute_joint_recalibration_score(
                    state,
                    variances.get(angle_key),
                    now_ms,
                )

            fallback_armed = bool(rs.get("fallback_armed"))
            primary_recovered = _has_primary_recovery_candidate(score_by_joint)
            if fallback_armed and primary_recovered:
                rs["fallback_armed"] = False
                _reset_fallback_streak(rs)
                fallback_armed = False

            selected_score_for_fallback = 0.0
            selected_debug_for_fallback: dict[str, Any] = {}
            if isinstance(selected_angle, str):
                selected_score_for_fallback, selected_debug_for_fallback = score_by_joint.get(
                    selected_angle, (0.0, {})
                )
            selected_range_gate_closed_streak = int(rs.get("selected_range_gate_closed_streak") or 0)
            stale_now = bool(stale_reevals >= 1)
            low_score_now = bool(
                float(selected_score_for_fallback) < FALLBACK_Y_LOW_SCORE_THRESHOLD
                and float(selected_debug_for_fallback.get("recentRange") or 0.0)
                < STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG
            )
            unclear_motion_now = bool(selected_range_gate_closed_streak >= STALE_SWITCH_MIN_CLOSED_STREAK)
            not_recalibrating_now = bool(
                rs.get("pending_switch_angle") is None
                and rs.get("handoff_observation_candidate_angle") is None
            )
            pre_calibrate_active = False
            if isinstance(selected_output_for_eval, dict):
                pre_calibrate_active = not bool(
                    selected_output_for_eval.get("calibrationComplete", False)
                )
            elif active_detector is not None:
                pre_calibrate_active = not _is_detector_calibrated(active_detector)
            fallback_elapsed = _update_fallback_bad_streak(
                rs,
                now_s=now,
                conditions_met=(
                    stale_now
                    and low_score_now
                    and unclear_motion_now
                    and not_recalibrating_now
                ),
                paused=pre_calibrate_active,
            )
            if not fallback_armed:
                fb_key, _, _ = _best_fallback_candidate(score_by_joint)
                if (
                    fb_key is not None
                    and fallback_elapsed >= FALLBACK_Y_ARM_WINDOW_SEC
                    and FALLBACK_Y_ARM_WINDOW_SEC > 0.0
                ):
                    rs["fallback_armed"] = True
                    fallback_armed = True

            candidate, candidate_sel_debug = select_recalibration_candidate(
                score_by_joint, selected_angle if isinstance(selected_angle, str) else None,
                variance_by_joint=variances,
                stale_reevals=stale_reevals,
                stale_switch_force_after_reevals=STALE_SWITCH_FORCE_AFTER_STALE_REEVALS,
                selected_range_gate_closed_streak=int(rs.get("selected_range_gate_closed_streak") or 0),
                stale_switch_max_selected_recent_range_deg=STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG,
                stale_switch_min_closed_streak=STALE_SWITCH_MIN_CLOSED_STREAK,
                fallback_armed=fallback_armed,
                primary_recovery_score=FALLBACK_Y_PRIMARY_RECOVERY_SCORE,
            )

            selected_score = 0.0
            selected_debug: dict[str, Any] = {}
            if isinstance(selected_angle, str):
                selected_score, selected_debug = score_by_joint.get(selected_angle, (0.0, {}))

            candidate_score = 0.0
            candidate_debug: dict[str, Any] = {}
            if isinstance(candidate, str):
                candidate_score, candidate_debug = score_by_joint.get(candidate, (0.0, {}))

            selected_recent_range = float(selected_debug.get("recentRange") or 0.0)
            selected_pose_score = float(selected_debug.get("poseScore") or 0.0)
            candidate_activity_score = float(candidate_debug.get("activityScore") or 0.0)
            candidate_pose_score = float(candidate_debug.get("poseScore") or 0.0)
            candidate_observable = bool(candidate_debug.get("observable"))
            candidate_completed_cycles = int(candidate_debug.get("completedCycles") or 0)
            candidate_recent_range = float(candidate_debug.get("recentRange") or 0.0)
            selected_median_rom_deg = (
                median_cycle_rom_deg(joint_states[selected_angle])
                if isinstance(selected_angle, str) and selected_angle in joint_states
                else 0.0
            )
            candidate_median_rom_deg = (
                median_cycle_rom_deg(joint_states[candidate])
                if isinstance(candidate, str) and candidate in joint_states
                else 0.0
            )
            median_recent_range_all = median_recent_range_from_score_debug(score_by_joint)
            same_joint_family = (
                is_same_joint_family(selected_angle, candidate)
                if isinstance(selected_angle, str) and isinstance(candidate, str)
                else False
            )
            last_switch = rs.get("selection_last_switch_at")
            time_cooldown_ok = (
                last_switch is None or (now - float(last_switch)) >= ANGLE_SELECTION_SWITCH_MIN_SEC
            )
            reps_at_last_switch = rs.get("selection_reps_at_last_recal_switch")
            reps_since_last_switch = (
                cumulative_shown - int(reps_at_last_switch)
                if isinstance(reps_at_last_switch, int)
                else JOINT_SWITCH_MIN_REPS_SINCE_LAST
            )
            rep_cooldown_ok = (
                not isinstance(reps_at_last_switch, int)
                or reps_since_last_switch >= JOINT_SWITCH_MIN_REPS_SINCE_LAST
            )
            cooldown_ok = time_cooldown_ok and rep_cooldown_ok
            should_switch, force_switch, switch_debug = should_switch_to_candidate(
                cooldown_ok=cooldown_ok,
                stale_reevals=stale_reevals,
                stale_switch_force_after_reevals=STALE_SWITCH_FORCE_AFTER_STALE_REEVALS,
                selected_recent_range=selected_recent_range,
                stale_switch_max_selected_recent_range_deg=STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG,
                selected_range_gate_closed_streak=int(rs.get("selected_range_gate_closed_streak") or 0),
                stale_switch_min_closed_streak=STALE_SWITCH_MIN_CLOSED_STREAK,
                selected_score=float(selected_score),
                selected_pose_score=selected_pose_score,
                candidate_score=float(candidate_score),
                candidate_activity_score=candidate_activity_score,
                candidate_pose_score=candidate_pose_score,
                candidate_observable=candidate_observable,
                candidate_completed_cycles=candidate_completed_cycles,
                candidate_recent_range=candidate_recent_range,
                candidate_median_rom_deg=candidate_median_rom_deg,
                selected_median_rom_deg=selected_median_rom_deg,
                median_recent_range_all=median_recent_range_all,
                same_joint_family=same_joint_family,
            )
            switch_debug = {
                **switch_debug,
                "timeCooldownOk": bool(time_cooldown_ok),
                "repCooldownOk": bool(rep_cooldown_ok),
                "repsSinceLastSwitch": int(reps_since_last_switch),
                "minRepsSinceLastSwitch": int(JOINT_SWITCH_MIN_REPS_SINCE_LAST),
                "fallbackArmed": bool(fallback_armed),
                "fallbackElapsedSec": float(fallback_elapsed),
                "fallbackPrimaryRecovered": bool(primary_recovered),
                "fallbackStaleNow": bool(stale_now),
                "fallbackLowScoreNow": bool(low_score_now),
                "fallbackUnclearMotionNow": bool(unclear_motion_now),
                "fallbackNotRecalibratingNow": bool(not_recalibrating_now),
                "fallbackPreCalibratePaused": bool(pre_calibrate_active),
            }
            if (
                isinstance(selected_angle, str)
                and isinstance(candidate, str)
                and candidate in joint_states
                and candidate != selected_angle
                and is_mirrored_pair(selected_angle, candidate)
            ):
                observation_trigger = (
                    candidate_completed_cycles >= HANDOFF_OBSERVATION_MIN_COMPLETED_CYCLES
                    and candidate_activity_score >= HANDOFF_OBSERVATION_MIN_ACTIVITY_SCORE
                    and candidate_pose_score >= HANDOFF_OBSERVATION_MIN_POSE_SCORE
                    and (
                        stale_reevals >= 1
                        or selected_recent_range < STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG
                        or selected_pose_score < HANDOFF_OBSERVATION_INCUMBENT_POSE_WEAK_SCORE
                    )
                )
                if observation_trigger:
                    obs_candidate = rs.get("handoff_observation_candidate_angle")
                    obs_selected = rs.get("handoff_observation_selected_angle")
                    if obs_candidate != candidate or obs_selected != selected_angle:
                        cand_shown, cand_raw = _detector_counts(joint_states[candidate].detector)
                        rs["handoff_observation_candidate_angle"] = candidate
                        rs["handoff_observation_selected_angle"] = selected_angle
                        rs["handoff_observation_started_at_ms"] = int(ts)
                        rs["handoff_observation_candidate_carryover_start_shown"] = cand_raw
                        rs["handoff_observation_candidate_carryover_start_raw"] = cand_raw
                        rs["handoff_observation_candidate_carryover_start_ts"] = int(ts)
            self._instr_emit(
                trace_context,
                {
                    "event": "joint_recalibration_snapshot",
                    "selected_angle": selected_angle,
                    "candidate_angle": candidate,
                    "selected_score": selected_score,
                    "candidate_score": candidate_score,
                    "switch_debug": switch_debug,
                    "candidate_selector_debug": candidate_sel_debug,
                },
            )
            if (
                isinstance(candidate, str)
                and candidate in joint_states
                and candidate != selected_angle
                and (should_switch or force_switch)
            ):
                candidate_state = joint_states[candidate]
                if rs.get("pending_switch_angle") != candidate:
                    rs["pending_switch_started_at_ms"] = int(ts)
                    rs["pending_switch_angle"] = candidate
                    rs["pending_switch_detector"] = candidate_state.detector
                    rs["pending_switch_incumbent_shown_start"] = current_raw
                    rs["pending_switch_incumbent_raw_start"] = current_raw
                    rs["pending_switch_incumbent_advanced"] = False
                    rs["pending_switch_observed"] = False
                    rs["pending_switch_incumbent_angle_min"] = tracking_angle_values.get(selected_angle)
                    rs["pending_switch_incumbent_angle_max"] = tracking_angle_values.get(selected_angle)
                    rs["pending_switch_incumbent_motion_span_deg"] = 0.0
                    rs["pending_switch_incumbent_observable_during_pending"] = False
                    rs["pending_switch_incumbent_completed_gated_cycle_during_pending"] = False
                    rs["pending_switch_candidate_advanced_during_pending"] = False
                    rs["pending_switch_candidate_completed_gated_cycle_during_pending"] = False
                    rs["pending_switch_candidate_pending_rom_estimate_deg"] = float(
                        candidate_debug.get("recentRange") or 0.0
                    )
                    rs["pending_switch_forced"] = bool(force_switch)
                    rs["pending_switch_incumbent_pose_score_at_start"] = selected_pose_score
                    rs["pending_switch_incumbent_completed_cycles_at_start"] = int(
                        selected_debug.get("completedCycles") or 0
                    )
                    rs["pending_switch_incumbent_recent_range_at_start"] = selected_recent_range
                    rs["pending_switch_incumbent_last_observed_ts_at_start"] = joint_states[
                        selected_angle
                    ].last_observed_timestamp_ms if isinstance(selected_angle, str) else None
                    rs["pending_switch_candidate_completed_cycles_at_start"] = candidate_completed_cycles
                    rs["pending_switch_candidate_rom_score_at_start"] = float(
                        candidate_debug.get("romScore") or 0.0
                    )
                    candidate_cycles = list(candidate_state.cycle_log)
                    cycle_before_start = [
                        entry for entry in candidate_cycles if entry[0] < int(rs["pending_switch_started_at_ms"])
                    ]
                    rs["pending_switch_candidate_last_cycle_ts_before_start"] = (
                        int(cycle_before_start[-1][0]) if cycle_before_start else None
                    )
                    cand_shown, cand_raw = _detector_counts(candidate_state.detector)
                    obs_candidate = rs.get("handoff_observation_candidate_angle")
                    obs_selected = rs.get("handoff_observation_selected_angle")
                    obs_shown = rs.get("handoff_observation_candidate_carryover_start_shown")
                    obs_raw = rs.get("handoff_observation_candidate_carryover_start_raw")
                    obs_ts = rs.get("handoff_observation_candidate_carryover_start_ts")
                    if (
                        obs_candidate == candidate
                        and obs_selected == selected_angle
                        and isinstance(obs_shown, int)
                        and isinstance(obs_raw, int)
                    ):
                        rs["pending_switch_candidate_carryover_start_shown"] = obs_shown
                        rs["pending_switch_candidate_carryover_start_raw"] = obs_raw
                        rs["pending_switch_candidate_carryover_start_ts"] = (
                            int(obs_ts) if isinstance(obs_ts, int) else int(ts)
                        )
                    else:
                        rs["pending_switch_candidate_carryover_start_shown"] = cand_raw
                        rs["pending_switch_candidate_carryover_start_raw"] = cand_raw
                        rs["pending_switch_candidate_carryover_start_ts"] = int(ts)
                    now_ms = int(ts)
                    inc_cycles_4s = [
                        ts_cycle
                        for ts_cycle, _ in joint_states[selected_angle].cycle_log
                        if isinstance(selected_angle, str) and ts_cycle >= now_ms - 4000
                    ] if isinstance(selected_angle, str) else []
                    cand_cycles_4s = [
                        ts_cycle for ts_cycle, _ in candidate_state.cycle_log if ts_cycle >= now_ms - 4000
                    ]
                    rs["pending_switch_incumbent_cycles_last_4s"] = len(inc_cycles_4s)
                    rs["pending_switch_candidate_cycles_last_4s"] = len(cand_cycles_4s)
                    rs["pending_switch_cycle_sync_score_last_4s"] = cycle_sync_score_last_4s(
                        joint_states[selected_angle], candidate_state, now_ms
                    ) if isinstance(selected_angle, str) else 0.0
                    rs["pending_switch_same_joint_family"] = is_same_joint_family(selected_angle, candidate)
                    rs["pending_switch_mirrored_pair"] = is_mirrored_pair(selected_angle, candidate)
            joint_records = _collect_joint_records(sdba, variances)

        angle_value = angle_values.get(selected_angle) if isinstance(selected_angle, str) else None
        selected_output = (
            detector_outputs.get(selected_angle) if isinstance(selected_angle, str) else None
        )
        perf_ms["session_total_ms"] = (time.perf_counter() - t_step_start) * 1000.0
        out = self._build_tracking_step_result(
            rs,
            angle_value,
            detector_output=selected_output,
            tracked_joint_changed=switched_to is not None,
            selection_debug={
                "rep_dom": rep_dom,
                "joint_records": joint_records,
                "switched_to": switched_to,
                "pending_switch": rs.get("pending_switch_angle"),
                "perf_ms": perf_ms,
            },
        )
        self._finalize_instrumentation_tracking(
            trace_context,
            out,
            raw_angle_value=raw_angle_val,
            filtered_angle_value=angle_value,
            selected_output=selected_output,
            sdba=sdba,
            peak_detector=rs.get("peak_detector"),
            sel_angle=selected_angle if isinstance(selected_angle, str) else None,
        )
        return out

    def _build_tracking_step_result(
        self,
        rs: dict[str, Any],
        angle_value: Optional[float],
        *,
        detector_output: Optional[dict[str, Any]] = None,
        tracked_joint_changed: bool = False,
        selection_debug: Optional[dict[str, Any]] = None,
    ) -> StepResult:
        tuning_params = rs["tuning_params"]
        peak_detector = rs["peak_detector"]
        d_tuning = DEFAULT_TUNING_PARAMS
        rep_count = 0
        state_str = "—"
        smoothed_value = None
        range_gate_open = True
        rolling_range: Optional[float] = None
        calibration_complete = True
        cal_target = int(tuning_params.get("calibrationReps", d_tuning["calibrationReps"]))
        cal_certainty_target = float(
            tuning_params.get("calibrationCertainty", d_tuning["calibrationCertainty"])
        )
        cal_certainty = 0.0
        primary_rep_count = 0
        avg_peak: Optional[float] = None
        avg_valley: Optional[float] = None

        if peak_detector is not None:
            out = detector_output if detector_output is not None else peak_detector.update(angle_value)
            rep_count = int(out.get("repCount", 0) or 0)
            primary_rep_count = rep_count
            state_str = str(out.get("state", "—"))
            smoothed_value = out.get("smoothedValue")
            if smoothed_value is not None:
                smoothed_value = float(smoothed_value)
            range_gate_open = bool(out.get("rangeGateOpen", True))
            r = out.get("rollingRange")
            rolling_range = float(r) if r is not None else None
            calibration_complete = bool(out.get("calibrationComplete", False))
            cal_target = int(out.get("calibrationTargetReps", cal_target))
            cal_certainty = float(out.get("calibrationCertainty", 0.0) or 0.0)
            cal_certainty_target = float(
                out.get("calibrationCertaintyTarget", cal_certainty_target)
            )
            st = peak_detector.get_state()
            avg_peak = st.get("calibratedAvgPeak")
            avg_valley = st.get("calibratedAvgValley")
            if avg_peak is not None:
                avg_peak = float(avg_peak)
            if avg_valley is not None:
                avg_valley = float(avg_valley)

        # Retroactive first-rep credit: once both a peak and valley are
        # established, count an unmatched trailing half-cycle as a complete
        # rep.  This compensates for the last rep whose closing turn is never
        # detected when the user stops moving at the end of a set.
        if (
            peak_detector is not None
            and _retroactive_credit_eligible(peak_detector, rs)
        ):
            rep_count += 1

        shown_rep_count = _apply_monotonic_shown_floor(
            rs,
            rep_count + int(rs.get("rep_count_offset") or 0),
        )
        shown_rep_count = max(0, shown_rep_count)
        shown_rep_count_raw = primary_rep_count + int(rs.get("rep_count_raw_offset") or 0)
        shown_rep_count_raw = max(0, shown_rep_count_raw)
        sel_ang = rs.get("selected_angle")
        if isinstance(sel_ang, str):
            tlm = list(_angle_landmarks(COMMON_ANGLES[sel_ang]))
        else:
            tlm = None

        cal_detail = ""
        if not calibration_complete:
            cal_detail = (
                f"Calibrating... reps {primary_rep_count}/{cal_target}  "
                f"certainty {cal_certainty * 100:.0f}%/{cal_certainty_target * 100:.0f}%"
            )
            status = cal_detail
        else:
            label = _format_angle_label(sel_ang) if sel_ang else "Joint"
            if sel_ang and sel_ang.startswith("LEFT_"):
                status = f"Tracking {label} - left side only"
            elif sel_ang and sel_ang.startswith("RIGHT_"):
                status = f"Tracking {label} - right side only"
            else:
                status = f"Tracking {label}"

        need_rg = float(tuning_params.get("minRangeGate", d_tuning["minRangeGate"]))
        if rolling_range is not None and not range_gate_open and need_rg > 0:
            if not calibration_complete:
                status = (
                    f"{status} - Move more: ~{rolling_range:.0f} deg so far "
                    f"(need at least {need_rg:.0f} deg for a rep)"
                )
            else:
                status = (
                    f"{status} - Range ~{rolling_range:.0f} deg, "
                    f"need at least {need_rg:.0f} deg"
                )

        calibration_started, calibration_locked = _calibration_edge_flags(
            rs,
            calibration_complete=calibration_complete,
            phase="tracking",
        )

        return StepResult(
            reps=shown_rep_count,
            reps_raw=shown_rep_count_raw,
            tracked_joint=sel_ang if isinstance(sel_ang, str) else None,
            tracked_joint_changed=tracked_joint_changed,
            calibration_started=calibration_started,
            calibration_locked=calibration_locked,
            angle_3_point_value=float(angle_value) if angle_value is not None else None,
            target_landmarks=tlm,
            tuning_params=dict(rs["tuning_params"]),
            avg_peak=avg_peak,
            avg_valley=avg_valley,
            calibration_complete=calibration_complete,
            peak_detector_state=state_str,
            smoothed_value=smoothed_value,
            range_gate_open=range_gate_open,
            rolling_range=rolling_range,
            calibration_target_reps=cal_target,
            calibration_certainty=cal_certainty,
            calibration_certainty_target=cal_certainty_target,
            phase="tracking",
            status_message=status,
            tracking_detail_message=cal_detail if not calibration_complete else "",
            **_low_fps_result_fields(rs),
            selection_debug=dict(selection_debug or {}),
        )

    def _no_pose_step_result(
        self,
        *,
        tuning_params: dict[str, Any],
        tracked_joint: Optional[str],
        default_tuning: dict[str, Any],
        phase: Literal["idle", "selecting", "tracking"],
        run_state: Optional[dict[str, Any]] = None,
    ) -> StepResult:
        low_fps_fields = _low_fps_result_fields(run_state) if isinstance(run_state, dict) else {}
        return StepResult(
            reps=0,
            reps_raw=0,
            tracked_joint=tracked_joint,
            angle_3_point_value=None,
            target_landmarks=None,
            tuning_params=dict(tuning_params),
            avg_peak=None,
            avg_valley=None,
            calibration_complete=False,
            peak_detector_state="—",
            smoothed_value=None,
            range_gate_open=True,
            rolling_range=None,
            calibration_target_reps=int(
                tuning_params.get("calibrationReps", default_tuning["calibrationReps"])
            ),
            calibration_certainty=0.0,
            calibration_certainty_target=float(
                tuning_params.get(
                    "calibrationCertainty",
                    default_tuning["calibrationCertainty"],
                )
            ),
            phase=phase,
            status_message="No pose",
            tracking_detail_message="",
            **low_fps_fields,
        )
