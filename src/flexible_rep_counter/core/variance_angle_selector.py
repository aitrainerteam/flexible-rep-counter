"""Variance-based angle selection: pick the angle with most consistent movement (median window variance)."""
from __future__ import annotations

from typing import Any, Optional

from flexible_rep_counter.core.settings import (
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
    "LEFT_KNEE": {"type": "angle_3_point", "landmarks": [11, 13, 15]},
    "RIGHT_KNEE": {"type": "angle_3_point", "landmarks": [12, 14, 16]},
    "LEFT_ELBOW": {"type": "angle_3_point", "landmarks": [5, 7, 9]},
    "RIGHT_ELBOW": {"type": "angle_3_point", "landmarks": [6, 8, 10]},
    "LEFT_SHOULDER": {"type": "angle_3_point", "landmarks": [11, 5, 7]},
    "RIGHT_SHOULDER": {"type": "angle_3_point", "landmarks": [12, 6, 8]},
    "LEFT_SHOULDER_ACROSS": {"type": "angle_3_point", "landmarks": [7, 5, 6]},
    "RIGHT_SHOULDER_ACROSS": {"type": "angle_3_point", "landmarks": [8, 6, 5]},
    "LEFT_HIP": {"type": "angle_3_point", "landmarks": [5, 11, 13]},
    "RIGHT_HIP": {"type": "angle_3_point", "landmarks": [6, 12, 14]},
    "LEFT_HIP_ACROSS": {"type": "angle_3_point", "landmarks": [13, 11, 12]},
    "RIGHT_HIP_ACROSS": {"type": "angle_3_point", "landmarks": [14, 12, 11]},
}

LOW_CONFIDENCE_THRESHOLD = 0.5
FRAME_MIN_CONFIDENCE = 0.5
MIN_ACTIVE_WINDOWS = ANGLE_SELECTION_MIN_ACTIVE_WINDOWS
SMOOTH_WINDOW = ANGLE_SELECTION_SMOOTH_WINDOW
ISOMETRIC_FALLBACK_ANGLE = "LEFT_HIP"
SHOULDER_ACROSS_TO_ELBOW_MIN_SCORE_RATIO = 0.6


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


def _exercise_looks_isometric(exercise: Optional[dict[str, Any]]) -> bool:
    """True when caller passed exercise metadata indicating a static hold (optional API)."""
    if not exercise:
        return False
    if exercise.get("isIsometric") is True:
        return True
    if (exercise.get("category") or "").lower() == "core" and "isometric" in (
        (exercise.get("description") or "").lower()
    ):
        return True
    name = (exercise.get("name") or "").lower()
    return any(x in name for x in ("plank", "hold", "isometric"))


def _angle_selection_thresholds(angle_key: str) -> dict[str, float]:
    """
    Per-common-angle gates from env, ``rep_counter.toml`` ``[angle_selection.joints.<KEY>]``,
    then global defaults (see ``get_angle_selection_joint_thresholds``).
    """
    return get_angle_selection_joint_thresholds(angle_key)


def compute_angle_variances_from_buffer(
    frame_buffer: list[list[dict]],
) -> dict[str, dict[str, Any]]:
    if not frame_buffer:
        return {}
    variances: dict[str, dict[str, Any]] = {}
    for angle_key, config in COMMON_ANGLES.items():
        history: list[float] = []
        for landmarks in frame_buffer:
            if not landmarks:
                continue
            min_conf = get_min_confidence_for_landmarks(landmarks, config["landmarks"])
            if min_conf is None or min_conf < FRAME_MIN_CONFIDENCE:
                continue
            value = calculate_from_type(config["type"], config["landmarks"], landmarks)
            if value is not None and not (isinstance(value, float) and value != value):
                history.append(value)
        if len(history) >= 10:
            smoothed = smooth_angle_series(history, window=SMOOTH_WINDOW)
            min_ws = 15 if len(smoothed) >= 90 else 12
            stats = calculate_variance(smoothed)
            robust = compute_robust_variance(smoothed)
            consistent = compute_consistent_variance_score(smoothed, min_window_size=min_ws)
            span_deg = max(smoothed) - min(smoothed) if len(smoothed) >= 2 else 0.0
            variances[angle_key] = {
                **stats,
                "robustVariance": robust["variance"],
                "medianWindowVariance": consistent["medianWindowVariance"],
                "activeWindowCount": consistent["activeWindowCount"],
                "windowVariances": consistent["windowVariances"],
                "smoothedRangeDeg": span_deg,
                "history": history,
                "config": config,
            }
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
    if not passes_consistent_variance_gate(variances, leader_key):
        return False
    top = _get_top_candidate(variances)
    if top is not None and not angle_keys_compatible(str(top.get("key")), str(leader_key)):
        return False
    return True


def passes_consistent_variance_gate(
    variances: dict[str, dict[str, Any]], angle_key: str
) -> bool:
    """
    True if this angle shows the same multi-window activity + ROM pattern used to
    accept a candidate in angle selection (see _get_top_candidate), without the
    second-best margin rule.
    """
    data = variances.get(angle_key)
    if not data:
        return False
    t = _angle_selection_thresholds(angle_key)
    consistent_var = float(data.get("medianWindowVariance") or 0.0)
    active_windows = int(data.get("activeWindowCount") or 0)
    span_deg = float(data.get("smoothedRangeDeg") or 0.0)
    if active_windows < MIN_ACTIVE_WINDOWS:
        return False
    if consistent_var < t["min_variance"]:
        return False
    if span_deg < t["min_range_deg"]:
        return False
    return True


def _candidate_score_if_eligible(
    angle_key: str, data: dict[str, Any]
) -> Optional[float]:
    t = _angle_selection_thresholds(angle_key)
    consistent_var = float(data.get("medianWindowVariance") or 0.0)
    active_windows = int(data.get("activeWindowCount") or 0)
    span_deg = float(data.get("smoothedRangeDeg") or 0.0)
    if active_windows < MIN_ACTIVE_WINDOWS:
        return None
    if consistent_var < t["min_variance"]:
        return None
    if span_deg < t["min_range_deg"]:
        return None
    return consistent_var


def _get_top_candidate(variances: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick the clearest winner: multi-window activity, meaningful ROM, and margin over the runner-up."""
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for key, data in variances.items():
        score = _candidate_score_if_eligible(key, data)
        if score is None:
            continue
        ranked.append((score, key, data))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        return None
    top_score, top_key, top_data = ranked[0]
    if top_key.endswith("_SHOULDER_ACROSS"):
        preferred_elbow = top_key.replace("SHOULDER_ACROSS", "ELBOW")
        elbow_data = variances.get(preferred_elbow)
        if elbow_data is not None:
            elbow_score = _candidate_score_if_eligible(preferred_elbow, elbow_data)
            # Across-shoulder angles can spike from torso sway during curls.
            # If the same-side elbow is also clearly active, prefer elbow tracking.
            if (
                elbow_score is not None
                and top_score > 0.0
                and elbow_score >= top_score * SHOULDER_ACROSS_TO_ELBOW_MIN_SCORE_RATIO
            ):
                return {"key": preferred_elbow, **elbow_data}
    if len(ranked) >= 2:
        second_score = ranked[1][0]
        ratio = _angle_selection_thresholds(top_key)["second_best_ratio"]
        if second_score > 0 and top_score < second_score * ratio:
            return None
    return {"key": top_key, **top_data}


def _get_angle_confidence(frame_buffer: list[list[dict]], angle_config: Optional[dict]) -> float:
    if not angle_config or not angle_config.get("landmarks"):
        return 0.0
    return get_average_confidence_for_landmarks(frame_buffer, angle_config["landmarks"])


def determine_best_angle(
    frame_buffer: list[list[dict]],
    exercise: Optional[dict[str, Any]] = None,
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

    if not frame_buffer or len(frame_buffer) < 40:
        return default_result

    is_isometric = _exercise_looks_isometric(exercise)

    variances = _calculate_all_variances(frame_buffer)
    debug["variances"] = {
        k: {
            "variance": v.get("variance"),
            "robustVariance": v.get("robustVariance"),
            "medianWindowVariance": v.get("medianWindowVariance"),
            "activeWindowCount": v.get("activeWindowCount"),
            "smoothedRangeDeg": v.get("smoothedRangeDeg"),
            "mean": v.get("mean"),
            "thresholds": _angle_selection_thresholds(k),
        }
        for k, v in variances.items()
    }

    top_candidate = _get_top_candidate(variances)
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
        if is_isometric and ISOMETRIC_FALLBACK_ANGLE in COMMON_ANGLES:
            fallback_config = COMMON_ANGLES[ISOMETRIC_FALLBACK_ANGLE]
            avg_conf = _get_angle_confidence(frame_buffer, fallback_config)
            debug["avgConfidence"] = avg_conf
            return {
                "selectedAngle": ISOMETRIC_FALLBACK_ANGLE,
                "source": "isometric_fallback",
                "tuningParams": get_default_tuning_params(),
                "debug": debug,
            }
        return default_result

    effective_variance = top_candidate.get("medianWindowVariance") or top_candidate.get("variance") or 0.0
    top_thresholds = _angle_selection_thresholds(top_candidate["key"])
    if effective_variance < top_thresholds["min_variance"]:
        if is_isometric and ISOMETRIC_FALLBACK_ANGLE in COMMON_ANGLES:
            fallback_config = COMMON_ANGLES[ISOMETRIC_FALLBACK_ANGLE]
            avg_conf = _get_angle_confidence(frame_buffer, fallback_config)
            debug["avgConfidence"] = avg_conf
            return {
                "selectedAngle": ISOMETRIC_FALLBACK_ANGLE,
                "source": "isometric_fallback",
                "tuningParams": get_default_tuning_params(),
                "debug": debug,
            }
        return default_result

    avg_confidence = _get_angle_confidence(frame_buffer, top_candidate.get("config"))
    debug["avgConfidence"] = avg_confidence

    if avg_confidence < LOW_CONFIDENCE_THRESHOLD:
        return default_result

    return {
        "selectedAngle": top_candidate["key"],
        "source": "variance",
        "tuningParams": get_default_tuning_params(),
        "debug": debug,
    }
