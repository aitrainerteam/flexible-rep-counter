from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Literal, Optional

from flexible_rep_counter.core.math_engine import PeakDetector
from flexible_rep_counter.core.settings import (
    FALLBACK_Y_BASELINE_MAX_SLEW_PX_PER_SEC,
    FALLBACK_Y_BASELINE_MIN_SAMPLES,
    FALLBACK_Y_BASELINE_SHORT_WINDOW_FRAMES,
    FALLBACK_Y_BASELINE_WINDOW_FRAMES,
)
from flexible_rep_counter.core.variance_angle_selector import (
    FRAME_MIN_CONFIDENCE,
    angle_signal_unit,
    is_fallback_angle,
)

_RECENT_RANGE_WINDOW = 45
_POSE_WINDOW = 30
_SYNC_WINDOW_MS = 4000

MIN_EVIDENCE_ROM_DEG = 12.0
MIN_EVIDENCE_RANGE_DEG = 12.0
GOOD_RANGE_DEG = 35.0
MIN_RANGE_DEG = 12.0
MIN_EVIDENCE_ROM_PX = 8.0
MIN_EVIDENCE_RANGE_PX = 8.0
GOOD_RANGE_PX = 24.0
MIN_RANGE_PX = 8.0

UNOBSERVABLE_POSE_SCORE = 0.30
INCUMBENT_ACTIVE_MOTION_MIN_SPAN_DEG = 18.0
INCUMBENT_ACTIVE_MOTION_ROM_RATIO = 0.60

CANDIDATE_MIN_SCORE = 0.62
CANDIDATE_MIN_ACTIVITY = 0.45
CANDIDATE_MIN_POSE_SCORE = 0.45
CANDIDATE_MIN_COMPLETED_CYCLES = 2

FORCE_SWITCH_MIN_SCORE = 0.50
FORCE_SWITCH_MIN_ACTIVITY = 0.50
FORCE_SWITCH_MIN_POSE_SCORE = 0.45

# Incumbent must go stale for several re-evals before a healthy-range joint is "bad".
STALE_REEVAL_INCUMBENT_BAD = 3
# Candidate must exceed these motion thresholds vs incumbent / field (not just cadence).
CANDIDATE_MIN_MEDIAN_RANGE_RATIO = 0.90
SAME_FAMILY_MIN_RANGE_RATIO = 0.92
CROSS_FAMILY_MIN_RANGE_RATIO = 1.05
SAME_FAMILY_ROM_DOMINANCE_RATIO = 0.95
CROSS_FAMILY_ROM_DOMINANCE_RATIO = 1.0
SAME_FAMILY_SCORE_MARGIN = 0.18
CROSS_FAMILY_SCORE_MARGIN = 0.25
SAME_FAMILY_SCORE_RATIO = 1.35
CROSS_FAMILY_SCORE_RATIO = 1.45
CANDIDATE_MIN_MEDIAN_WINDOW_VARIANCE = 6.0
PRIMARY_RECOVERY_SCORE = 0.60


@dataclass
class JointMotionState:
    angle_key: str
    detector: PeakDetector
    history: deque[Optional[float]]
    confidence_history: deque[tuple[int, float]]

    last_raw_rep_count: int = 0
    last_rep_timestamp_ms: int | None = None
    last_observed_timestamp_ms: int | None = None

    recent_roms: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_peaks: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_valleys: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_intervals_ms: deque[int] = field(default_factory=lambda: deque(maxlen=8))
    cycle_log: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=16))
    absolute_y_long_window: deque[float] = field(
        default_factory=lambda: deque(maxlen=max(8, int(FALLBACK_Y_BASELINE_WINDOW_FRAMES)))
    )
    absolute_y_short_window: deque[float] = field(
        default_factory=lambda: deque(maxlen=max(4, int(FALLBACK_Y_BASELINE_SHORT_WINDOW_FRAMES)))
    )
    absolute_y_baseline_px: float | None = None
    absolute_y_last_baseline_update_ms: int | None = None
    last_raw_y_px: float | None = None
    last_baseline_px: float | None = None
    last_oscillation_px: float | None = None
    last_scale_px: float | None = None
    scale_at_lock_px: float | None = None

    last_score: float = 0.0
    last_score_debug: dict[str, float] = field(default_factory=dict)


@dataclass
class JointRecalibrationScore:
    angle_key: str
    score: float
    debug: dict[str, Any]


@dataclass
class HandoffDecision:
    kind: Literal["alternate_limb", "same_exercise", "ambiguous"]
    rationale: dict[str, Any]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = clamp(q / 100.0, 0.0, 1.0) * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def robust_std(values: list[float]) -> float:
    if not values:
        return 0.0
    med = median(values)
    mad = median(abs(v - med) for v in values)
    return 1.4826 * mad


def robust_cv(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    med = median(values)
    return robust_std(values) / max(abs(med), 1e-6)


def _recent_range(state: JointMotionState) -> float:
    recent_values = [v for v in list(state.history)[-_RECENT_RANGE_WINDOW:] if v is not None]
    if len(recent_values) < 8:
        return 0.0
    return _percentile(recent_values, 95.0) - _percentile(recent_values, 5.0)


def median_recent_range_from_score_debug(
    scores: dict[str, tuple[float, dict[str, Any]]],
    *,
    signal_unit: Optional[str] = None,
) -> float:
    ranges = [
        float(debug.get("recentRange") or 0.0)
        for angle_key, (_, debug) in scores.items()
        if float(debug.get("recentRange") or 0.0) > 0.0
        and (signal_unit is None or angle_signal_unit(angle_key) == signal_unit)
    ]
    return float(median(ranges)) if ranges else 0.0


def median_cycle_rom_deg(state: JointMotionState) -> float:
    roms = [float(v) for v in list(state.recent_roms) if isinstance(v, (int, float))]
    return float(median(roms)) if roms else 0.0


def signal_unit(angle_key: str) -> str:
    return angle_signal_unit(angle_key)


def _activity_thresholds(unit: str) -> tuple[float, float]:
    if unit == "px":
        return MIN_RANGE_PX, GOOD_RANGE_PX
    return MIN_RANGE_DEG, GOOD_RANGE_DEG


def _evidence_thresholds(unit: str) -> tuple[float, float]:
    if unit == "px":
        return MIN_EVIDENCE_ROM_PX, MIN_EVIDENCE_RANGE_PX
    return MIN_EVIDENCE_ROM_DEG, MIN_EVIDENCE_RANGE_DEG


def _apply_absolute_y_baseline(
    state: JointMotionState,
    *,
    raw_value: float,
    timestamp_ms: int,
) -> float:
    state.absolute_y_long_window.append(raw_value)
    state.absolute_y_short_window.append(raw_value)
    state.last_raw_y_px = float(raw_value)
    long_values = list(state.absolute_y_long_window)
    if state.absolute_y_baseline_px is None:
        if len(long_values) >= max(3, int(FALLBACK_Y_BASELINE_MIN_SAMPLES)):
            state.absolute_y_baseline_px = float(median(long_values))
            state.absolute_y_last_baseline_update_ms = int(timestamp_ms)
        state.last_baseline_px = state.absolute_y_baseline_px
        state.last_oscillation_px = 0.0
        return 0.0

    target_baseline = float(median(long_values)) if long_values else float(state.absolute_y_baseline_px)
    last_update = (
        int(state.absolute_y_last_baseline_update_ms)
        if isinstance(state.absolute_y_last_baseline_update_ms, int)
        else int(timestamp_ms)
    )
    dt_s = max(0.0, float(timestamp_ms - last_update) / 1000.0)
    max_slew = max(0.0, float(FALLBACK_Y_BASELINE_MAX_SLEW_PX_PER_SEC)) * dt_s
    delta = target_baseline - float(state.absolute_y_baseline_px)
    if max_slew <= 0.0:
        adjusted_baseline = float(state.absolute_y_baseline_px)
    else:
        adjusted_baseline = float(state.absolute_y_baseline_px) + max(-max_slew, min(max_slew, delta))
    state.absolute_y_baseline_px = adjusted_baseline
    state.absolute_y_last_baseline_update_ms = int(timestamp_ms)
    state.last_baseline_px = adjusted_baseline
    state.last_oscillation_px = float(raw_value - adjusted_baseline)
    return float(raw_value - adjusted_baseline)


def _latest_cycle_rom_deg(state: JointMotionState, detector_output: dict[str, Any]) -> float:
    peak = detector_output.get("reversalPeak")
    valley = detector_output.get("reversalValley")
    if not isinstance(peak, (int, float)):
        peak = detector_output.get("peak")
    if not isinstance(valley, (int, float)):
        valley = detector_output.get("valley")
    if isinstance(peak, (int, float)) and isinstance(valley, (int, float)):
        return abs(float(peak) - float(valley))
    recent_peaks = list(state.recent_peaks)
    recent_valleys = list(state.recent_valleys)
    if recent_peaks and recent_valleys:
        return abs(float(recent_peaks[-1]) - float(recent_valleys[-1]))
    peaks = list(getattr(state.detector, "peaks", []) or [])
    valleys = list(getattr(state.detector, "valleys", []) or [])
    if peaks and valleys:
        return abs(float(peaks[-1]) - float(valleys[-1]))
    rolling = detector_output.get("rollingRange")
    if isinstance(rolling, (int, float)):
        return float(rolling)
    return 0.0


def cycle_is_evidence(state: JointMotionState, rom: float, recent_range: float) -> bool:
    min_rom, min_range = _evidence_thresholds(signal_unit(state.angle_key))
    return (
        rom >= min_rom
        and recent_range >= min_range
        and state.last_observed_timestamp_ms is not None
    )


def update_joint_motion_state(
    state: JointMotionState,
    val: Optional[float],
    conf: Optional[float],
    timestamp_ms: int,
    min_confidence: float = FRAME_MIN_CONFIDENCE,
    scale_px: Optional[float] = None,
) -> dict[str, Any]:
    confidence = float(conf) if isinstance(conf, (int, float)) else 0.0
    state.confidence_history.append((timestamp_ms, confidence))
    if val is None or conf is None or conf < float(min_confidence):
        state.history.append(None)
        return {"advanced": False, "detectorOutput": state.detector.update(None), "gatedCycle": False}

    raw_value = float(val)
    value = raw_value
    if signal_unit(state.angle_key) == "px":
        value = _apply_absolute_y_baseline(
            state,
            raw_value=raw_value,
            timestamp_ms=timestamp_ms,
        )
    state.history.append(value)
    state.last_scale_px = float(scale_px) if isinstance(scale_px, (int, float)) else None
    if state.scale_at_lock_px is None and state.last_scale_px is not None:
        state.scale_at_lock_px = float(state.last_scale_px)
    state.last_observed_timestamp_ms = timestamp_ms

    prev_raw = int(state.detector.get_rep_count() or 0)
    detector_output = state.detector.update(value)
    new_raw = int(state.detector.get_rep_count() or 0)
    advanced = new_raw > prev_raw
    gated_cycle = False

    peak = detector_output.get("peak")
    valley = detector_output.get("valley")
    reversal_peak = detector_output.get("reversalPeak")
    reversal_valley = detector_output.get("reversalValley")
    score_peak = peak if isinstance(peak, (int, float)) else reversal_peak
    score_valley = valley if isinstance(valley, (int, float)) else reversal_valley
    if isinstance(score_peak, (int, float)):
        state.recent_peaks.append(float(score_peak))
    if isinstance(score_valley, (int, float)):
        state.recent_valleys.append(float(score_valley))

    reversal_observed = isinstance(score_peak, (int, float)) or isinstance(score_valley, (int, float))
    if advanced or reversal_observed:
        if advanced and state.last_rep_timestamp_ms is not None:
            state.recent_intervals_ms.append(max(1, timestamp_ms - state.last_rep_timestamp_ms))
        if advanced:
            state.last_rep_timestamp_ms = timestamp_ms
        rom = _latest_cycle_rom_deg(state, detector_output)
        recent_range = _recent_range(state)
        if cycle_is_evidence(state, rom, recent_range):
            if not advanced and state.last_rep_timestamp_ms is not None:
                state.recent_intervals_ms.append(max(1, timestamp_ms - state.last_rep_timestamp_ms))
            state.recent_roms.append(float(rom))
            state.cycle_log.append((timestamp_ms, float(rom)))
            gated_cycle = True
            if not advanced:
                state.last_rep_timestamp_ms = timestamp_ms

    state.last_raw_rep_count = new_raw
    return {"advanced": advanced, "detectorOutput": detector_output, "gatedCycle": gated_cycle}


def normalize_variance_prior(variance_data: Optional[dict[str, Any]]) -> float:
    data = variance_data or {}
    unit = str(data.get("signalUnit") or "deg")
    variance = float(data.get("medianWindowVariance") or 0.0)
    active_windows = float(data.get("activeWindowCount") or 0.0)
    range_raw = data.get("smoothedRangePx") if unit == "px" else data.get("smoothedRangeDeg")
    smoothed_range = float(range_raw or 0.0)
    min_range, good_range = _activity_thresholds(unit)
    variance_score = clamp(variance / 120.0, 0.0, 1.0)
    active_score = clamp(active_windows / 4.0, 0.0, 1.0)
    range_score = smoothstep(min_range, good_range, smoothed_range)
    return 0.45 * variance_score + 0.20 * active_score + 0.35 * range_score


def compute_joint_recalibration_score(
    state: JointMotionState,
    variance_data: Optional[dict[str, Any]],
    now_ms: int,
) -> tuple[float, dict[str, Any]]:
    unit = signal_unit(state.angle_key)
    recent_values = [v for v in list(state.history)[-_RECENT_RANGE_WINDOW:] if v is not None]
    recent_range = (
        _percentile(recent_values, 95.0) - _percentile(recent_values, 5.0)
        if len(recent_values) >= 8
        else 0.0
    )

    variance_prior = normalize_variance_prior(variance_data)
    min_range, good_range = _activity_thresholds(unit)
    activity_score = smoothstep(min_range, good_range, recent_range)

    roms = list(state.recent_roms)
    peaks = list(state.recent_peaks)
    valleys = list(state.recent_valleys)
    intervals = list(state.recent_intervals_ms)

    completed_cycles = len(roms)
    evidence = 1.0 - math.exp(-completed_cycles / 2.0)
    rom_cv = robust_cv(roms) if completed_cycles >= 2 else None
    rom_score = math.exp(-rom_cv / 0.25) if rom_cv is not None else 0.0

    extrema_jitter = None
    if len(peaks) >= 2 and len(valleys) >= 2 and roms:
        extrema_jitter = (robust_std(peaks) + robust_std(valleys)) / max(median(roms), 1e-6)
        extrema_score = math.exp(-extrema_jitter / 0.35)
    else:
        extrema_score = 0.0

    interval_cv = robust_cv(intervals) if len(intervals) >= 2 else None
    cadence_score = (
        math.exp(-interval_cv / 0.40)
        if interval_cv is not None
        else (0.5 if completed_cycles >= 1 else 0.0)
    )

    recent_confs = [c for _, c in list(state.confidence_history)[-_POSE_WINDOW:]]
    avg_conf = mean(recent_confs) if recent_confs else 0.0
    visible_fraction = (
        sum(c >= FRAME_MIN_CONFIDENCE for c in recent_confs) / len(recent_confs)
        if recent_confs
        else 0.0
    )
    pose_score = clamp((avg_conf - 0.35) / 0.40, 0.0, 1.0)
    observable = (
        visible_fraction >= 0.50
        and state.last_observed_timestamp_ms is not None
        and (now_ms - state.last_observed_timestamp_ms) <= 750
        and pose_score >= UNOBSERVABLE_POSE_SCORE
    )

    cycle_quality = (
        0.25 * activity_score
        + 0.30 * rom_score
        + 0.15 * extrema_score
        + 0.15 * cadence_score
        + 0.15 * pose_score
    )
    score = evidence * cycle_quality + (1.0 - evidence) * variance_prior
    debug = {
        "score": score,
        "signalUnit": unit,
        "evidence": evidence,
        "completedCycles": completed_cycles,
        "activityScore": activity_score,
        "variancePrior": variance_prior,
        "romScore": rom_score,
        "romCv": rom_cv,
        "extremaScore": extrema_score,
        "extremaJitter": extrema_jitter,
        "cadenceScore": cadence_score,
        "intervalCv": interval_cv,
        "poseScore": pose_score,
        "visibleFraction": visible_fraction,
        "recentRange": recent_range,
        "recentRangePx": recent_range if unit == "px" else None,
        "recentRangeDeg": recent_range if unit == "deg" else None,
        "observable": observable,
    }
    state.last_score = float(score)
    state.last_score_debug = {
        "activityScore": float(activity_score),
        "romScore": float(rom_score),
        "cadenceScore": float(cadence_score),
        "poseScore": float(pose_score),
        "evidence": float(evidence),
    }
    return score, debug


def compute_incumbent_health(
    *,
    stale_reevals: int,
    selected_recent_range: float,
    stale_switch_max_selected_recent_range_deg: float,
    selected_range_gate_closed_streak: int,
    stale_switch_min_closed_streak: int,
    selected_score: float,
    selected_pose_score: float,
) -> dict[str, Any]:
    low_range = selected_recent_range < stale_switch_max_selected_recent_range_deg
    gate_closed = selected_range_gate_closed_streak >= stale_switch_min_closed_streak
    low_score = selected_score < 0.35
    low_pose = selected_pose_score < UNOBSERVABLE_POSE_SCORE
    incumbent_range_healthy = (
        not low_range
        and not gate_closed
        and not low_pose
    )
    stale_threshold = STALE_REEVAL_INCUMBENT_BAD if incumbent_range_healthy else 1
    stale_incumbent_bad = stale_reevals >= stale_threshold
    incumbent_bad = (
        stale_incumbent_bad
        or low_range
        or gate_closed
        or low_score
        or low_pose
    )
    return {
        "incumbentBad": bool(incumbent_bad),
        "incumbentRangeHealthy": bool(incumbent_range_healthy),
        "staleThreshold": int(stale_threshold),
        "staleIncumbentBad": bool(stale_incumbent_bad),
        "lowRange": bool(low_range),
        "gateClosed": bool(gate_closed),
        "lowScore": bool(low_score),
        "lowPose": bool(low_pose),
    }


def allow_cross_family_rescue(
    *,
    incumbent_health: dict[str, Any],
    stale_reevals: int,
    stale_switch_force_after_reevals: int,
) -> bool:
    if not bool(incumbent_health.get("incumbentBad")):
        return False
    return bool(
        incumbent_health.get("lowRange")
        or incumbent_health.get("gateClosed")
        or incumbent_health.get("lowScore")
        or incumbent_health.get("lowPose")
        or stale_reevals >= stale_switch_force_after_reevals
    )


def should_run_full_recalibration(
    *,
    has_pending_switch: bool,
    has_handoff_observation: bool,
    current_raw: int,
    tracking_raw_at_joint_lock: int,
    post_lock_min_raw_reps: int,
    raw_advanced_since_last_eval: bool,
    selected_recent_range: float,
    selected_pose_score: float,
    selected_range_gate_closed_streak: int,
    stale_switch_max_selected_recent_range_deg: float,
    stale_switch_min_closed_streak: int,
    selected_score: float = 1.0,
) -> bool:
    if has_pending_switch or has_handoff_observation:
        return True
    incumbent_health = compute_incumbent_health(
        stale_reevals=0,
        selected_recent_range=selected_recent_range,
        stale_switch_max_selected_recent_range_deg=stale_switch_max_selected_recent_range_deg,
        selected_range_gate_closed_streak=selected_range_gate_closed_streak,
        stale_switch_min_closed_streak=stale_switch_min_closed_streak,
        selected_score=float(selected_score),
        selected_pose_score=selected_pose_score,
    )
    # Keep post-lock warmup for healthy incumbents, but do not block re-evals when
    # the currently tracked joint is already stale/bad (low ROM, closed range gate,
    # weak score, or weak pose). This allows generic fallback recovery across joints.
    within_post_lock_warmup = (
        current_raw < tracking_raw_at_joint_lock + max(0, int(post_lock_min_raw_reps))
    )
    if within_post_lock_warmup and bool(incumbent_health.get("incumbentRangeHealthy")):
        return False
    if raw_advanced_since_last_eval and bool(incumbent_health.get("incumbentRangeHealthy")):
        return False
    return True


def select_recalibration_candidate(
    scores: dict[str, tuple[float, dict[str, Any]]],
    selected_angle: Optional[str],
    *,
    variance_by_joint: Optional[dict[str, dict[str, Any]]] = None,
    stale_reevals: int = 0,
    stale_switch_force_after_reevals: int = 8,
    selected_range_gate_closed_streak: int = 0,
    stale_switch_max_selected_recent_range_deg: float = 14.0,
    stale_switch_min_closed_streak: int = 10,
    fallback_armed: bool = False,
    primary_recovery_score: float = PRIMARY_RECOVERY_SCORE,
) -> tuple[str | None, dict[str, Any]]:
    variance_by_joint = variance_by_joint or {}
    selected_unit = (
        signal_unit(selected_angle) if isinstance(selected_angle, str) else None
    )
    field_median_range = median_recent_range_from_score_debug(
        scores,
        signal_unit=selected_unit,
    )
    variance_values = [
        float((variance_by_joint.get(angle_key) or {}).get("medianWindowVariance") or 0.0)
        for angle_key in scores
        if float((variance_by_joint.get(angle_key) or {}).get("medianWindowVariance") or 0.0) > 0.0
    ]
    field_median_variance = float(median(variance_values)) if variance_values else 0.0
    min_range = field_median_range * CANDIDATE_MIN_MEDIAN_RANGE_RATIO if field_median_range > 0.0 else 0.0
    min_variance = (
        max(CANDIDATE_MIN_MEDIAN_WINDOW_VARIANCE, field_median_variance * CANDIDATE_MIN_MEDIAN_RANGE_RATIO)
        if field_median_variance > 0.0
        else CANDIDATE_MIN_MEDIAN_WINDOW_VARIANCE
    )
    selected_score = 0.0
    selected_debug: dict[str, Any] = {}
    if isinstance(selected_angle, str):
        selected_score, selected_debug = scores.get(selected_angle, (0.0, {}))
    selected_health = compute_incumbent_health(
        stale_reevals=stale_reevals,
        selected_recent_range=float(selected_debug.get("recentRange") or 0.0),
        stale_switch_max_selected_recent_range_deg=stale_switch_max_selected_recent_range_deg,
        selected_range_gate_closed_streak=selected_range_gate_closed_streak,
        stale_switch_min_closed_streak=stale_switch_min_closed_streak,
        selected_score=float(selected_score),
        selected_pose_score=float(selected_debug.get("poseScore") or 0.0),
    )
    allow_cross_family = allow_cross_family_rescue(
        incumbent_health=selected_health,
        stale_reevals=stale_reevals,
        stale_switch_force_after_reevals=stale_switch_force_after_reevals,
    )

    best_key: Optional[str] = None
    best_score = -1.0
    best_debug: dict[str, Any] = {}
    best_primary_key: Optional[str] = None
    best_primary_score = -1.0
    best_primary_debug: dict[str, Any] = {}
    for angle_key, (score, debug) in scores.items():
        if angle_key == selected_angle:
            continue
        if is_fallback_angle(angle_key) and not fallback_armed:
            continue
        if (
            isinstance(selected_angle, str)
            and not is_same_joint_family(selected_angle, angle_key)
            and not allow_cross_family
        ):
            continue
        if float(debug.get("poseScore") or 0.0) < UNOBSERVABLE_POSE_SCORE:
            continue
        candidate_unit = signal_unit(angle_key)
        range_floor = min_range
        if candidate_unit != selected_unit:
            range_floor = 0.0
        recent_range = float(debug.get("recentRange") or 0.0)
        if range_floor > 0.0 and recent_range < range_floor:
            continue
        joint_variance = float(
            (variance_by_joint.get(angle_key) or {}).get("medianWindowVariance") or 0.0
        )
        if joint_variance > 0.0 and joint_variance < min_variance:
            continue
        if score > best_score:
            best_key = angle_key
            best_score = float(score)
            best_debug = debug
        if (not is_fallback_angle(angle_key)) and score > best_primary_score:
            best_primary_key = angle_key
            best_primary_score = float(score)
            best_primary_debug = debug

    primary_recovered = bool(
        best_primary_key
        and best_primary_score >= float(primary_recovery_score)
        and float(best_primary_debug.get("poseScore") or 0.0) >= CANDIDATE_MIN_POSE_SCORE
        and float(best_primary_debug.get("activityScore") or 0.0) >= CANDIDATE_MIN_ACTIVITY
        and int(best_primary_debug.get("completedCycles") or 0) >= CANDIDATE_MIN_COMPLETED_CYCLES
    )
    if (
        fallback_armed
        and isinstance(best_key, str)
        and is_fallback_angle(best_key)
        and primary_recovered
        and isinstance(best_primary_key, str)
    ):
        best_key = best_primary_key
        best_score = best_primary_score
        best_debug = best_primary_debug

    return best_key, {
        "candidateScore": best_score,
        "candidateDebug": best_debug,
        "bestPrimaryCandidate": best_primary_key,
        "bestPrimaryScore": best_primary_score,
        "primaryRecovered": primary_recovered,
        "fallbackArmed": bool(fallback_armed),
        "fieldMedianRangeUnit": selected_unit,
        "fieldMedianRange": field_median_range,
        "fieldMedianVariance": field_median_variance,
        "allowCrossFamily": bool(allow_cross_family),
        "incumbentHealth": selected_health,
    }


def should_switch_to_candidate(
    *,
    cooldown_ok: bool,
    stale_reevals: int,
    stale_switch_force_after_reevals: int,
    selected_recent_range: float,
    stale_switch_max_selected_recent_range_deg: float,
    selected_range_gate_closed_streak: int,
    stale_switch_min_closed_streak: int,
    selected_score: float,
    selected_pose_score: float,
    candidate_score: float,
    candidate_activity_score: float,
    candidate_pose_score: float,
    candidate_observable: bool,
    candidate_completed_cycles: int,
    candidate_recent_range: float = 0.0,
    candidate_median_rom_deg: float = 0.0,
    selected_median_rom_deg: float = 0.0,
    median_recent_range_all: float = 0.0,
    same_joint_family: bool = False,
    selected_signal_unit: str = "deg",
    candidate_signal_unit: str = "deg",
    primary_recovery_cross_modality: bool = False,
    primary_recovery_force_after_stale_reevals: int = 8,
    primary_recovery_skip_score_margin: bool = False,
    primary_recovery_score: float = PRIMARY_RECOVERY_SCORE,
    incumbent_is_fallback: bool = False,
) -> tuple[bool, bool, dict[str, Any]]:
    incumbent_health = compute_incumbent_health(
        stale_reevals=stale_reevals,
        selected_recent_range=selected_recent_range,
        stale_switch_max_selected_recent_range_deg=stale_switch_max_selected_recent_range_deg,
        selected_range_gate_closed_streak=selected_range_gate_closed_streak,
        stale_switch_min_closed_streak=stale_switch_min_closed_streak,
        selected_score=selected_score,
        selected_pose_score=selected_pose_score,
    )
    incumbent_bad = bool(incumbent_health.get("incumbentBad"))

    range_ratio = SAME_FAMILY_MIN_RANGE_RATIO if same_joint_family else CROSS_FAMILY_MIN_RANGE_RATIO
    rom_ratio = SAME_FAMILY_ROM_DOMINANCE_RATIO if same_joint_family else CROSS_FAMILY_ROM_DOMINANCE_RATIO
    score_margin = SAME_FAMILY_SCORE_MARGIN if same_joint_family else CROSS_FAMILY_SCORE_MARGIN
    score_ratio = SAME_FAMILY_SCORE_RATIO if same_joint_family else CROSS_FAMILY_SCORE_RATIO

    compare_motion_ranges = selected_signal_unit == candidate_signal_unit
    candidate_range_vs_incumbent = (
        not compare_motion_ranges
        or candidate_recent_range <= 0.0
        or selected_recent_range <= 0.0
        or candidate_recent_range >= selected_recent_range * range_ratio
    )
    candidate_range_vs_field = (
        not compare_motion_ranges
        or median_recent_range_all <= 0.0
        or candidate_recent_range <= 0.0
        or candidate_recent_range >= median_recent_range_all * CANDIDATE_MIN_MEDIAN_RANGE_RATIO
    )
    incumbent_range_stale = bool(
        incumbent_health.get("lowRange") or incumbent_health.get("gateClosed")
    )
    # When the tracked joint's recent ROM has collapsed, compare against its live
    # sliding-window range—not peak cycle ROM from before the limb went quiet.
    selected_rom_for_dominance = (
        selected_recent_range
        if incumbent_range_stale and selected_recent_range > 0.0
        else selected_median_rom_deg
    )
    candidate_rom_dominates = (
        not compare_motion_ranges
        or candidate_median_rom_deg <= 0.0
        or selected_rom_for_dominance <= 0.0
        or candidate_median_rom_deg >= selected_rom_for_dominance * rom_ratio
    )
    candidate_motion_ok = (
        candidate_range_vs_incumbent and candidate_range_vs_field and candidate_rom_dominates
    )

    candidate_good = (
        candidate_score >= CANDIDATE_MIN_SCORE
        and candidate_activity_score >= CANDIDATE_MIN_ACTIVITY
        and candidate_pose_score >= CANDIDATE_MIN_POSE_SCORE
        and candidate_observable
        and candidate_completed_cycles >= CANDIDATE_MIN_COMPLETED_CYCLES
        and candidate_motion_ok
    )
    if (
        primary_recovery_cross_modality
        and incumbent_is_fallback
        and incumbent_bad
        and candidate_score >= float(primary_recovery_score)
        and candidate_score >= FORCE_SWITCH_MIN_SCORE
        and candidate_activity_score >= FORCE_SWITCH_MIN_ACTIVITY
        and candidate_pose_score >= FORCE_SWITCH_MIN_POSE_SCORE
        and candidate_observable
        and candidate_completed_cycles >= CANDIDATE_MIN_COMPLETED_CYCLES
        and candidate_motion_ok
    ):
        candidate_good = True
    candidate_clearly_better = (
        candidate_score >= selected_score + score_margin
        or candidate_score >= selected_score * score_ratio
    )
    if (
        primary_recovery_skip_score_margin
        and primary_recovery_cross_modality
        and incumbent_is_fallback
        and incumbent_bad
        and candidate_good
        and candidate_score >= float(primary_recovery_score)
    ):
        candidate_clearly_better = True
    should_switch = cooldown_ok and incumbent_bad and candidate_good and candidate_clearly_better

    force_after = (
        int(primary_recovery_force_after_stale_reevals)
        if primary_recovery_cross_modality
        else int(stale_switch_force_after_reevals)
    )
    force_switch = (
        stale_reevals >= force_after
        and candidate_score >= FORCE_SWITCH_MIN_SCORE
        and candidate_activity_score >= FORCE_SWITCH_MIN_ACTIVITY
        and candidate_pose_score >= FORCE_SWITCH_MIN_POSE_SCORE
        and candidate_completed_cycles >= CANDIDATE_MIN_COMPLETED_CYCLES
        and candidate_motion_ok
    )
    return bool(should_switch), bool(force_switch), {
        "incumbentBad": bool(incumbent_bad),
        "incumbentRangeHealthy": bool(incumbent_health.get("incumbentRangeHealthy")),
        "staleThreshold": int(incumbent_health.get("staleThreshold") or 0),
        "candidateGood": bool(candidate_good),
        "candidateMotionOk": bool(candidate_motion_ok),
        "candidateRangeVsIncumbent": bool(candidate_range_vs_incumbent),
        "candidateRangeVsField": bool(candidate_range_vs_field),
        "candidateRomDominates": bool(candidate_rom_dominates),
        "incumbentRangeStale": bool(incumbent_range_stale),
        "selectedRomForDominance": float(selected_rom_for_dominance),
        "candidateObservable": bool(candidate_observable),
        "candidateClearlyBetter": bool(candidate_clearly_better),
        "compareMotionRanges": bool(compare_motion_ranges),
        "selectedSignalUnit": selected_signal_unit,
        "candidateSignalUnit": candidate_signal_unit,
        "sameJointFamily": bool(same_joint_family),
        "shouldSwitch": bool(should_switch),
        "forceSwitch": bool(force_switch),
        "primaryRecoveryCrossModality": bool(primary_recovery_cross_modality),
        "forceAfterStaleReevals": int(force_after),
    }


def is_same_joint_family(a: Optional[str], b: Optional[str]) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if is_fallback_angle(a) or is_fallback_angle(b):
        return a == b
    base_a = a[5:] if a.startswith("LEFT_") else a[6:] if a.startswith("RIGHT_") else a
    base_b = b[5:] if b.startswith("LEFT_") else b[6:] if b.startswith("RIGHT_") else b
    return base_a == base_b


def is_mirrored_pair(a: Optional[str], b: Optional[str]) -> bool:
    if isinstance(a, str) and is_fallback_angle(a):
        return False
    if isinstance(b, str) and is_fallback_angle(b):
        return False
    if not is_same_joint_family(a, b):
        return False
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return (a.startswith("LEFT_") and b.startswith("RIGHT_")) or (
        a.startswith("RIGHT_") and b.startswith("LEFT_")
    )


def cycle_sync_score_last_4s(
    incumbent: JointMotionState,
    candidate: JointMotionState,
    now_ms: int,
) -> float:
    lo = now_ms - _SYNC_WINDOW_MS
    inc_cycles = [ts for ts, _ in incumbent.cycle_log if ts >= lo]
    cand_cycles = [ts for ts, _ in candidate.cycle_log if ts >= lo]
    if not inc_cycles or not cand_cycles:
        return 0.0
    count_ratio = min(len(inc_cycles), len(cand_cycles)) / max(len(inc_cycles), len(cand_cycles))
    ts_gap = abs(inc_cycles[-1] - cand_cycles[-1])
    ts_score = 1.0 - clamp(ts_gap / 1500.0, 0.0, 1.0)
    return 0.6 * count_ratio + 0.4 * ts_score


def classify_handoff(pending_state: dict[str, Any]) -> HandoffDecision:
    incumbent_angle = pending_state.get("incumbent_angle")
    candidate_angle = pending_state.get("candidate_angle")
    if isinstance(incumbent_angle, str) and isinstance(candidate_angle, str):
        if is_fallback_angle(incumbent_angle) or is_fallback_angle(candidate_angle):
            return HandoffDecision(
                kind="same_exercise",
                rationale={"rule": "fallback_joint_transition"},
            )
    incumbent_motion_span = float(pending_state.get("incumbent_motion_span_deg") or 0.0)
    candidate_pending_rom = float(pending_state.get("candidate_pending_rom_estimate_deg") or 0.0)
    incumbent_active_motion_threshold = max(
        INCUMBENT_ACTIVE_MOTION_MIN_SPAN_DEG,
        INCUMBENT_ACTIVE_MOTION_ROM_RATIO * candidate_pending_rom,
    )
    incumbent_stopped = (
        not bool(pending_state.get("incumbent_completed_gated_cycle_during_pending"))
        and incumbent_motion_span < incumbent_active_motion_threshold
    )
    candidate_delta_since_observation_start = max(
        0,
        int(pending_state.get("candidate_current_raw") or 0)
        - int(pending_state.get("candidate_carryover_start_raw") or 0),
    )
    candidate_has_background_reps = int(pending_state.get("candidate_current_raw") or 0) > 0
    candidate_active_now = (
        candidate_delta_since_observation_start >= 1
        or bool(pending_state.get("candidate_completed_gated_cycle_during_pending"))
    )
    if (
        bool(pending_state.get("mirrored_pair"))
        and bool(pending_state.get("same_joint_family"))
        and incumbent_stopped
        and (candidate_active_now or candidate_has_background_reps)
    ):
        return HandoffDecision(
            kind="alternate_limb",
            rationale={
                "rule": "mirrored_incumbent_stopped",
                "candidateDeltaSinceObservationStart": candidate_delta_since_observation_start,
                "candidateCurrentRaw": int(pending_state.get("candidate_current_raw") or 0),
            },
        )

    prior_synchronized_same_exercise = (
        bool(pending_state.get("same_joint_family"))
        and int(pending_state.get("incumbent_cycles_last_4s") or 0) >= 2
        and int(pending_state.get("candidate_cycles_last_4s") or 0) >= 2
        and float(pending_state.get("cycle_sync_score_last_4s") or 0.0) >= 0.60
        and not incumbent_stopped
    )
    if prior_synchronized_same_exercise:
        return HandoffDecision(
            kind="same_exercise",
            rationale={"rule": "prior_synchronized_same_exercise"},
        )
    if (
        bool(pending_state.get("incumbent_completed_gated_cycle_during_pending"))
        or incumbent_motion_span >= incumbent_active_motion_threshold
    ):
        return HandoffDecision(
            kind="same_exercise",
            rationale={
                "rule": "incumbent_active_during_pending",
                "incumbentMotionSpanDeg": incumbent_motion_span,
                "incumbentActiveMotionThresholdDeg": incumbent_active_motion_threshold,
            },
        )

    incumbent_last_obs = pending_state.get("incumbent_last_observed_ts_at_start")
    pending_start_ts = int(pending_state.get("pending_start_ts") or 0)
    incumbent_disappeared = (
        not bool(pending_state.get("incumbent_observable_during_pending"))
        or float(pending_state.get("incumbent_pose_score_at_start") or 0.0) < UNOBSERVABLE_POSE_SCORE
        or (
            isinstance(incumbent_last_obs, int)
            and (pending_start_ts - incumbent_last_obs) > 750
        )
    )
    candidate_last_cycle = pending_state.get("candidate_last_cycle_ts_before_start")
    candidate_had_prior_cycles = (
        int(pending_state.get("candidate_completed_cycles_at_start") or 0) >= 2
        and isinstance(candidate_last_cycle, int)
        and (pending_start_ts - candidate_last_cycle) <= _SYNC_WINDOW_MS
        and float(pending_state.get("candidate_rom_score_at_start") or 0.0) >= 0.50
        and bool(pending_state.get("same_joint_family"))
        and float(pending_state.get("cycle_sync_score_last_4s") or 0.0) >= 0.60
        and int(pending_state.get("candidate_cycles_last_4s") or 0) >= 2
    )
    if (
        incumbent_disappeared
        and candidate_had_prior_cycles
        and not (
            bool(pending_state.get("mirrored_pair"))
            and bool(pending_state.get("same_joint_family"))
            and incumbent_stopped
        )
    ):
        return HandoffDecision(
            kind="same_exercise",
            rationale={"rule": "incumbent_disappeared_with_prior_sync_cycles"},
        )

    incumbent_quiet_but_visible = (
        bool(pending_state.get("incumbent_observable_during_pending"))
        and incumbent_motion_span < 12.0
        and not bool(pending_state.get("incumbent_completed_gated_cycle_during_pending"))
    )
    forced_mirrored_candidate_ready_at_start = (
        bool(pending_state.get("switch_forced"))
        and bool(pending_state.get("mirrored_pair"))
        and bool(pending_state.get("same_joint_family"))
        and int(pending_state.get("candidate_completed_cycles_at_start") or 0) >= 2
        and float(pending_state.get("candidate_rom_score_at_start") or 0.0) >= 0.50
    )
    if (
        bool(pending_state.get("switch_forced"))
        and bool(pending_state.get("mirrored_pair"))
        and (
            candidate_delta_since_observation_start >= 1
            or forced_mirrored_candidate_ready_at_start
        )
        and not bool(pending_state.get("incumbent_completed_gated_cycle_during_pending"))
        and not prior_synchronized_same_exercise
    ):
        return HandoffDecision(
            kind="alternate_limb",
            rationale={
                "rule": "forced_switch_mirrored_candidate_ready_or_delta",
                "candidateDeltaSinceObservationStart": candidate_delta_since_observation_start,
                "candidateReadyAtStart": forced_mirrored_candidate_ready_at_start,
            },
        )
    if (
        incumbent_quiet_but_visible
        and bool(pending_state.get("mirrored_pair"))
        and candidate_delta_since_observation_start >= 1
        and not prior_synchronized_same_exercise
    ):
        return HandoffDecision(
            kind="alternate_limb",
            rationale={
                "rule": "mirrored_quiet_incumbent_candidate_delta_since_observation_start",
                "candidateDeltaSinceObservationStart": candidate_delta_since_observation_start,
            },
        )

    return HandoffDecision(kind="ambiguous", rationale={"rule": "insufficient_or_cross_family_evidence"})
