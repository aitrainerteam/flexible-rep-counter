"""Variance-based angle selection: pick the angle with most consistent movement (median window variance)."""
from __future__ import annotations

from typing import Any, Optional

from flexible_rep_counter.core.settings import (
    ANGLE_SELECTION_MIN_FRAMES,
    ANGLE_SELECTION_MIN_ACTIVE_WINDOWS,
    ANGLE_SELECTION_SMOOTH_WINDOW,
    get_angle_selection_joint_thresholds,
    get_default_tuning_params,
)
from flexible_rep_counter.core.math_engine import (
    calculate_from_type,
    calculate_variance,
    compute_consistent_variance_score,
    compute_robust_variance,
    get_average_confidence_for_landmarks,
    get_min_confidence_for_landmarks,
    smooth_angle_series,
)

# COCO indices: 0 Nose, 1-4 eyes/ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles
# *_ACROSS: partial-body framing — shoulder pair uses elbow–shoulder–opposite shoulder (no hips); hip pair uses
# knee–hip–opposite hip (no shoulders/torso).

COMMON_ANGLES: dict[str, dict[str, Any]] = {
    "LEFT_KNEE": {"type": "angle_3_point", "landmarks": [11, 13, 15], "eligibility": "primary", "modality": "angle_deg"},
    "RIGHT_KNEE": {"type": "angle_3_point", "landmarks": [12, 14, 16], "eligibility": "primary", "modality": "angle_deg"},
    "LEFT_ELBOW": {"type": "angle_3_point", "landmarks": [5, 7, 9], "eligibility": "primary", "modality": "angle_deg"},
    "RIGHT_ELBOW": {"type": "angle_3_point", "landmarks": [6, 8, 10], "eligibility": "primary", "modality": "angle_deg"},
    "LEFT_SHOULDER": {"type": "angle_3_point", "landmarks": [11, 5, 7], "eligibility": "primary", "modality": "angle_deg"},
    "RIGHT_SHOULDER": {"type": "angle_3_point", "landmarks": [12, 6, 8], "eligibility": "primary", "modality": "angle_deg"},
    "LEFT_SHOULDER_ACROSS": {"type": "angle_3_point", "landmarks": [7, 5, 6], "eligibility": "primary", "modality": "angle_deg"},
    "RIGHT_SHOULDER_ACROSS": {"type": "angle_3_point", "landmarks": [8, 6, 5], "eligibility": "primary", "modality": "angle_deg"},
    "LEFT_HIP": {"type": "angle_3_point", "landmarks": [5, 11, 13], "eligibility": "primary", "modality": "angle_deg"},
    "RIGHT_HIP": {"type": "angle_3_point", "landmarks": [6, 12, 14], "eligibility": "primary", "modality": "angle_deg"},
    "LEFT_HIP_ACROSS": {"type": "angle_3_point", "landmarks": [13, 11, 12], "eligibility": "primary", "modality": "angle_deg"},
    "RIGHT_HIP_ACROSS": {"type": "angle_3_point", "landmarks": [14, 12, 11], "eligibility": "primary", "modality": "angle_deg"},
    "SHOULDER_SHRUG_Y": {
        "type": "absolute_y_position",
        "landmarks": [5, 6, 11, 12],
        "sample_landmarks": [5, 6],
        "scale": {"kind": "shoulder_width", "landmarks": [5, 6]},
        "min_conf": 0.4,
        "eligibility": "fallback",
        "modality": "vertical_px",
    },
    "LEFT_WRIST_Y": {
        "type": "absolute_y_position",
        "landmarks": [9, 5, 6, 11, 12],
        "sample_landmarks": [9],
        "scale": {"kind": "torso_height", "landmarks": [5, 6, 11, 12]},
        "min_conf": 0.4,
        "eligibility": "fallback",
        "modality": "vertical_px",
    },
    "RIGHT_WRIST_Y": {
        "type": "absolute_y_position",
        "landmarks": [10, 6, 5, 11, 12],
        "sample_landmarks": [10],
        "scale": {"kind": "torso_height", "landmarks": [5, 6, 11, 12]},
        "min_conf": 0.4,
        "eligibility": "fallback",
        "modality": "vertical_px",
    },
    "HIP_DEPTH_Y": {
        "type": "absolute_y_position",
        "landmarks": [11, 12, 15, 16],
        "sample_landmarks": [11, 12],
        "scale": {"kind": "hip_width", "landmarks": [11, 12]},
        "min_conf": 0.4,
        "eligibility": "fallback",
        "modality": "vertical_px",
    },
    "ANKLE_LIFT_Y": {
        "type": "absolute_y_position",
        "landmarks": [15, 16, 11, 12],
        "sample_landmarks": [15, 16],
        "scale": {"kind": "hip_width", "landmarks": [11, 12]},
        "min_conf": 0.4,
        "eligibility": "fallback",
        "modality": "vertical_px",
    },
}

LOW_CONFIDENCE_THRESHOLD = 0.5
FRAME_MIN_CONFIDENCE = 0.5
MIN_ACTIVE_WINDOWS = ANGLE_SELECTION_MIN_ACTIVE_WINDOWS
SMOOTH_WINDOW = ANGLE_SELECTION_SMOOTH_WINDOW
SHOULDER_ACROSS_TO_ELBOW_MIN_SCORE_RATIO = 0.55
# Across-body angles (elbow–shoulder–opposite shoulder, etc.) must beat same-side
# non-across alternatives by this median-window-variance ratio to stay selected.
ACROSS_DOMINANCE_MIN_RATIO = 1.32
# When the #2 candidate is hip/shoulder *across* and #1 is a limb joint, median-window
# variances are not directly comparable; use at most this ratio (vs global second_best).
SECOND_BEST_RELAXED_WHEN_RUNNERUP_ACROSS = 1.06
UNCLEAR_TIEBREAK_POOL_RATIO = 0.96
UNCLEAR_TIEBREAK_MIN_CONFIDENCE = 0.62
UNCLEAR_TIEBREAK_MIN_VARIANCE_FACTOR = 1.10
UNCLEAR_TIEBREAK_MIN_ACTIVE_WINDOWS = 3
UNCLEAR_TIEBREAK_MIN_RANGE_FLOOR_DEG = 18.0


def is_fallback_angle(angle_key: str) -> bool:
    cfg = COMMON_ANGLES.get(angle_key) or {}
    return str(cfg.get("eligibility") or "primary").strip().lower() == "fallback"


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


def _angle_min_conf_threshold(cfg: dict[str, Any]) -> float:
    if str(cfg.get("type") or "").strip().lower() in {"absolute_y_position"}:
        return float(cfg.get("min_conf", 0.4) or 0.4)
    return FRAME_MIN_CONFIDENCE


def angle_modality(angle_key: str) -> str:
    cfg = COMMON_ANGLES.get(angle_key) or {}
    modality = str(cfg.get("modality") or "").strip().lower()
    return modality if modality else "angle_deg"


def angle_signal_unit(angle_key: str) -> str:
    return "px" if angle_modality(angle_key) == "vertical_px" else "deg"


def _angle_side(angle_key: str) -> str:
    if angle_key.startswith("LEFT_"):
        return "LEFT"
    if angle_key.startswith("RIGHT_"):
        return "RIGHT"
    return ""


def _angle_base(angle_key: str) -> str:
    base = angle_key
    if base.startswith("LEFT_"):
        base = base[len("LEFT_") :]
    elif base.startswith("RIGHT_"):
        base = base[len("RIGHT_") :]
    if base.endswith("_ACROSS"):
        base = base[: -len("_ACROSS")]
    return base


def angle_keys_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """True when angle keys are the same logical limb/joint family."""
    if not a or not b:
        return False
    if a == b:
        return True
    side_a, side_b = _angle_side(a), _angle_side(b)
    if side_a and side_b and side_a != side_b:
        return False
    return _angle_base(a) == _angle_base(b)


def _angle_selection_thresholds(angle_key: str) -> dict[str, float]:
    """
    Per-common-angle gates from env, ``rep_counter.toml`` ``[angle_selection.joints.<KEY>]``,
    then global defaults (see ``get_angle_selection_joint_thresholds``).
    """
    return get_angle_selection_joint_thresholds(angle_key)


def _variance_eligibility(angle_key: str, data: dict[str, Any]) -> tuple[bool, float]:
    """Same gates as top-candidate scoring: (passes, median_window_variance_or_zero)."""
    t = _angle_selection_thresholds(angle_key)
    consistent_var = float(data.get("medianWindowVariance") or 0.0)
    active_windows = int(data.get("activeWindowCount") or 0)
    signal_unit = angle_signal_unit(angle_key)
    span_raw = data.get("smoothedRangePx") if signal_unit == "px" else data.get("smoothedRangeDeg")
    span = float(span_raw or 0.0)
    if active_windows < MIN_ACTIVE_WINDOWS:
        return False, 0.0
    min_variance = float(t["min_variance_px2"] if signal_unit == "px" else t["min_variance"])
    if consistent_var < min_variance:
        return False, 0.0
    min_range = float(t["min_range_px"] if signal_unit == "px" else t["min_range_deg"])
    if span < min_range:
        return False, 0.0
    return True, consistent_var


def compute_angle_variances_from_buffer(
    frame_buffer: list[list[dict]],
    *,
    include_debug: bool = True,
    angle_histories: Optional[dict[str, list[Optional[float]]]] = None,
    fallback_armed: bool = False,
) -> dict[str, dict[str, Any]]:
    if not frame_buffer:
        return {}
    variances: dict[str, dict[str, Any]] = {}
    for angle_key, config in COMMON_ANGLES.items():
        if is_fallback_angle(angle_key) and not fallback_armed:
            continue
        history: list[float] = []
        precomputed = angle_histories.get(angle_key) if angle_histories else None
        if precomputed is not None:
            for value in precomputed:
                if value is None:
                    continue
                if isinstance(value, float) and value != value:
                    continue
                history.append(float(value))
        else:
            for landmarks in frame_buffer:
                if not landmarks:
                    continue
                angle_landmarks = _angle_landmarks(config)
                min_conf_required = _angle_min_conf_threshold(config)
                min_conf = get_min_confidence_for_landmarks(landmarks, angle_landmarks)
                if min_conf is None or min_conf < min_conf_required:
                    continue
                value = calculate_from_type(config["type"], config, landmarks)
                if value is not None and not (isinstance(value, float) and value != value):
                    history.append(value)
        if len(history) >= 10:
            smoothed = smooth_angle_series(history, window=SMOOTH_WINDOW)
            min_ws = 15 if len(smoothed) >= 90 else 12
            stats = calculate_variance(smoothed)
            robust = compute_robust_variance(smoothed)
            consistent = compute_consistent_variance_score(smoothed, min_window_size=min_ws)
            span_deg = max(smoothed) - min(smoothed) if len(smoothed) >= 2 else 0.0
            row: dict[str, Any] = {
                **stats,
                "robustVariance": robust["variance"],
                "medianWindowVariance": consistent["medianWindowVariance"],
                "activeWindowCount": consistent["activeWindowCount"],
                "signalUnit": angle_signal_unit(angle_key),
                "smoothedRange": span_deg,
                "smoothedRangeDeg": span_deg,
                "smoothedRangePx": span_deg if angle_signal_unit(angle_key) == "px" else None,
                "config": config,
            }
            if include_debug:
                row["windowVariances"] = consistent["windowVariances"]
                row["history"] = history
            variances[angle_key] = row
    return variances


# Backwards-compatible name for internal use
_calculate_all_variances = compute_angle_variances_from_buffer


def summarize_rep_dominance(rep_counts: dict[str, int]) -> dict[str, Any]:
    """Rep distribution across joints during selection (only joints with rep_count > 0)."""
    positive = {k: int(v) for k, v in rep_counts.items() if int(v) > 0}
    total = sum(positive.values())
    if total <= 0 or not positive:
        return {
            "totalReps": 0,
            "leaderKey": None,
            "leaderReps": 0,
            "leaderShare": 0.0,
        }
    leader_key = max(positive.keys(), key=lambda k: positive[k])
    leader_reps = positive[leader_key]
    return {
        "totalReps": total,
        "leaderKey": leader_key,
        "leaderReps": leader_reps,
        "leaderShare": leader_reps / total,
    }


def dominance_conditions_met(
    variances: dict[str, dict[str, Any]],
    rep_dom: dict[str, Any],
    *,
    dominance_fraction: float,
    min_leading_reps: int,
    fallback_armed: bool = False,
) -> bool:
    """
    True when one joint leads rep count by more than `dominance_fraction` of total reps, has at least
    `min_leading_reps` reps, passes the same variance/ROM gate used for selection, and matches the
    variance top candidate when one exists (avoids locking to rep noise on the wrong limb).
    """
    leader_key = rep_dom.get("leaderKey")
    if not leader_key or rep_dom.get("totalReps", 0) <= 0:
        return False
    share = float(rep_dom.get("leaderShare") or 0.0)
    if share <= dominance_fraction:
        return False
    if int(rep_dom.get("leaderReps") or 0) < min_leading_reps:
        return False
    if not passes_consistent_variance_gate(variances, leader_key, fallback_armed=fallback_armed):
        return False
    top = _get_top_candidate(variances, fallback_armed=fallback_armed)
    if top is not None and not angle_keys_compatible(str(top.get("key")), str(leader_key)):
        return False
    return True


def passes_consistent_variance_gate(
    variances: dict[str, dict[str, Any]], angle_key: str, *, fallback_armed: bool = False
) -> bool:
    """
    True if this angle shows the same multi-window activity + ROM pattern used to
    accept a candidate in angle selection (see _get_top_candidate), without the
    second-best margin rule.
    """
    data = variances.get(angle_key)
    if not data:
        return False
    if is_fallback_angle(angle_key) and not fallback_armed:
        return False
    ok, _ = _variance_eligibility(angle_key, data)
    return ok


def _candidate_score_if_eligible(
    angle_key: str, data: dict[str, Any]
) -> Optional[float]:
    ok, score = _variance_eligibility(angle_key, data)
    return score if ok else None


def _non_across_alternatives_order(across_key: str) -> list[str]:
    """Same-side joints to try before accepting a shoulder/hip *across* angle."""
    if across_key == "LEFT_SHOULDER_ACROSS":
        return ["LEFT_ELBOW", "LEFT_SHOULDER"]
    if across_key == "RIGHT_SHOULDER_ACROSS":
        return ["RIGHT_ELBOW", "RIGHT_SHOULDER"]
    if across_key == "LEFT_HIP_ACROSS":
        return ["LEFT_KNEE", "LEFT_HIP"]
    if across_key == "RIGHT_HIP_ACROSS":
        return ["RIGHT_KNEE", "RIGHT_HIP"]
    return []


def _prefer_same_side_alt_over_across(
    across_key: str,
    across_score: float,
    alt_key: str,
    alt_score: float,
) -> bool:
    if across_score <= 0.0 or alt_score <= 0.0:
        return False
    if across_key.endswith("_SHOULDER_ACROSS") and alt_key.endswith("_ELBOW"):
        if alt_score >= across_score * SHOULDER_ACROSS_TO_ELBOW_MIN_SCORE_RATIO:
            return True
    return across_score < alt_score * ACROSS_DOMINANCE_MIN_RATIO


def _joint_stability_rank(angle_key: str) -> int:
    base = _angle_base(angle_key)
    if base == "KNEE":
        return 4
    if base == "HIP":
        return 3
    if base == "SHOULDER":
        return 2
    if base == "ELBOW":
        return 1
    return 0


def _is_strong_unclear_tiebreak_candidate(
    angle_key: str,
    score: float,
    data: dict[str, Any],
) -> bool:
    thresholds = _angle_selection_thresholds(angle_key)
    unit = angle_signal_unit(angle_key)
    min_score = float(
        thresholds["min_variance_px2"] if unit == "px" else thresholds["min_variance"]
    ) * UNCLEAR_TIEBREAK_MIN_VARIANCE_FACTOR
    min_range = (
        float(thresholds["min_range_px"])
        if unit == "px"
        else max(float(thresholds["min_range_deg"]), UNCLEAR_TIEBREAK_MIN_RANGE_FLOOR_DEG)
    )
    active_windows = int(data.get("activeWindowCount") or 0)
    range_raw = data.get("smoothedRangePx") if unit == "px" else data.get("smoothedRangeDeg")
    range_deg = float(range_raw or 0.0)
    return bool(
        score >= min_score
        and active_windows >= UNCLEAR_TIEBREAK_MIN_ACTIVE_WINDOWS
        and range_deg >= min_range
    )


def _resolve_unclear_tie_break_candidate(
    ranked: list[tuple[float, str, dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    if len(ranked) < 2:
        return None
    top_score = float(ranked[0][0])
    if top_score <= 0.0:
        return None
    pool: list[tuple[float, str, dict[str, Any]]] = []
    for score, key, data in ranked:
        if float(score) < top_score * UNCLEAR_TIEBREAK_POOL_RATIO:
            break
        if _is_strong_unclear_tiebreak_candidate(key, float(score), data):
            pool.append((float(score), key, data))
    if len(pool) < 2:
        return None
    winner_score, winner_key, winner_data = max(
        pool,
        key=lambda x: (
            _joint_stability_rank(x[1]),
            float(x[2].get("smoothedRangeDeg") or 0.0),
            x[0],
            int(x[2].get("activeWindowCount") or 0),
            x[1],
        ),
    )
    return {
        "key": winner_key,
        "__tiebreak_unclear": True,
        "__tiebreak_pool_size": len(pool),
        "__tiebreak_winner_score": winner_score,
        **winner_data,
    }


def _get_top_candidate(
    variances: dict[str, dict[str, Any]],
    *,
    fallback_armed: bool = False,
) -> Optional[dict[str, Any]]:
    """Pick the clearest winner: multi-window activity, meaningful ROM, and margin over the runner-up."""
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for key, data in variances.items():
        if is_fallback_angle(key) and not fallback_armed:
            continue
        score = _candidate_score_if_eligible(key, data)
        if score is None:
            continue
        ranked.append((score, key, data))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        return None
    top_score, top_key, top_data = ranked[0]
    if top_key.endswith("_SHOULDER_ACROSS") or top_key.endswith("_HIP_ACROSS"):
        for alt_key in _non_across_alternatives_order(top_key):
            alt_data = variances.get(alt_key)
            if alt_data is None:
                continue
            alt_score = _candidate_score_if_eligible(alt_key, alt_data)
            if alt_score is None:
                continue
            prefer_alt = _prefer_same_side_alt_over_across(
                top_key, top_score, alt_key, alt_score
            )
            if prefer_alt:
                return {"key": alt_key, **alt_data}
    if len(ranked) >= 2:
        second_score, second_key, _ = ranked[1]
        ratio = _angle_selection_thresholds(top_key)["second_best_ratio"]
        if second_key.endswith("_ACROSS") and not top_key.endswith("_ACROSS"):
            ratio = min(ratio, SECOND_BEST_RELAXED_WHEN_RUNNERUP_ACROSS)
        if second_score > 0 and top_score < second_score * ratio:
            tie_break = _resolve_unclear_tie_break_candidate(ranked)
            if tie_break is not None:
                return tie_break
            return None
    return {"key": top_key, **top_data}


def _get_angle_confidence(frame_buffer: list[list[dict]], angle_config: Optional[dict]) -> float:
    if not angle_config:
        return 0.0
    landmarks = _angle_landmarks(angle_config)
    if not landmarks:
        return 0.0
    return get_average_confidence_for_landmarks(frame_buffer, landmarks)


def determine_best_angle(
    frame_buffer: list[list[dict]],
    *,
    variances: Optional[dict[str, dict[str, Any]]] = None,
    include_debug: bool = True,
    fallback_armed: bool = False,
) -> dict[str, Any]:
    """
    Pick the best angle to track from a buffer of frames.
    Returns { selectedAngle, source, tuningParams, debug }.
    """
    debug: dict[str, Any] = {
        "frameCount": len(frame_buffer) if frame_buffer else 0,
        "variances": {},
        "topCandidate": None,
        "avgConfidence": 0,
    }
    default_result = {
        "selectedAngle": None,
        "source": "none",
        "tuningParams": get_default_tuning_params(),
        "debug": debug,
    }

    if not frame_buffer or len(frame_buffer) < ANGLE_SELECTION_MIN_FRAMES:
        return default_result

    variance_rows = variances or _calculate_all_variances(
        frame_buffer, include_debug=include_debug, fallback_armed=fallback_armed
    )
    if include_debug:
        debug["variances"] = {
            k: {
                "variance": v.get("variance"),
                "robustVariance": v.get("robustVariance"),
                "medianWindowVariance": v.get("medianWindowVariance"),
                "activeWindowCount": v.get("activeWindowCount"),
                "signalUnit": v.get("signalUnit"),
                "smoothedRange": v.get("smoothedRange"),
                "smoothedRangePx": v.get("smoothedRangePx"),
                "smoothedRangeDeg": v.get("smoothedRangeDeg"),
                "mean": v.get("mean"),
                "thresholds": _angle_selection_thresholds(k),
            }
            for k, v in variance_rows.items()
        }

    top_candidate = _get_top_candidate(variance_rows, fallback_armed=fallback_armed)
    debug["topCandidate"] = (
        {
            "key": top_candidate["key"],
            "variance": top_candidate.get("variance"),
            "medianWindowVariance": top_candidate.get("medianWindowVariance"),
            "activeWindowCount": top_candidate.get("activeWindowCount"),
        }
        if top_candidate
        else None
    )

    if not top_candidate:
        return default_result

    effective_variance = top_candidate.get("medianWindowVariance") or top_candidate.get("variance") or 0.0
    top_thresholds = _angle_selection_thresholds(top_candidate["key"])
    if effective_variance < top_thresholds["min_variance"]:
        return default_result

    avg_confidence = _get_angle_confidence(frame_buffer, top_candidate.get("config"))
    debug["avgConfidence"] = avg_confidence

    min_conf_for_pick = (
        UNCLEAR_TIEBREAK_MIN_CONFIDENCE
        if bool(top_candidate.get("__tiebreak_unclear"))
        else LOW_CONFIDENCE_THRESHOLD
    )
    if avg_confidence < min_conf_for_pick:
        return default_result

    return {
        "selectedAngle": top_candidate["key"],
        "source": "variance_tiebreak" if bool(top_candidate.get("__tiebreak_unclear")) else "variance",
        "tuningParams": get_default_tuning_params(),
        "debug": debug,
    }
