"""Variance-based angle selection: pick the angle with most consistent movement (median window variance)."""
from __future__ import annotations

from typing import Any, Optional

from app.config import get_default_tuning_params
from app.math_engine import (
    MIN_VARIANCE_THRESHOLD,
    calculate_from_type,
    calculate_variance,
    compute_consistent_variance_score,
    compute_robust_variance,
    get_average_confidence_for_landmarks,
    get_min_confidence_for_landmarks,
)

# COCO indices: 0 Nose, 1-4 eyes/ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles

COMMON_ANGLES: dict[str, dict[str, Any]] = {
    "LEFT_KNEE": {"type": "angle_3_point", "landmarks": [11, 13, 15]},
    "RIGHT_KNEE": {"type": "angle_3_point", "landmarks": [12, 14, 16]},
    "LEFT_ELBOW": {"type": "angle_3_point", "landmarks": [5, 7, 9]},
    "RIGHT_ELBOW": {"type": "angle_3_point", "landmarks": [6, 8, 10]},
    "LEFT_SHOULDER": {"type": "angle_3_point", "landmarks": [11, 5, 7]},
    "RIGHT_SHOULDER": {"type": "angle_3_point", "landmarks": [12, 6, 8]},
    "LEFT_HIP": {"type": "angle_3_point", "landmarks": [5, 11, 13]},
    "RIGHT_HIP": {"type": "angle_3_point", "landmarks": [6, 12, 14]},
}

MIN_VARIANCE_THRESHOLD_SELECTOR = 5.0
LOW_CONFIDENCE_THRESHOLD = 0.5
FRAME_MIN_CONFIDENCE = 0.5
MIN_ACTIVE_WINDOWS = 2
ISOMETRIC_FALLBACK_ANGLE = "LEFT_HIP"


def _calculate_all_variances(frame_buffer: list[list[dict]]) -> dict[str, dict[str, Any]]:
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
            stats = calculate_variance(history)
            robust = compute_robust_variance(history)
            consistent = compute_consistent_variance_score(history)
            variances[angle_key] = {
                **stats,
                "robustVariance": robust["variance"],
                "medianWindowVariance": consistent["medianWindowVariance"],
                "activeWindowCount": consistent["activeWindowCount"],
                "windowVariances": consistent["windowVariances"],
                "history": history,
                "config": config,
            }
    return variances


def _get_top_candidate(variances: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    best_score = 0.0
    top = None
    for key, data in variances.items():
        consistent_var = data.get("medianWindowVariance") or 0.0
        active_windows = data.get("activeWindowCount") or 0
        if active_windows < MIN_ACTIVE_WINDOWS:
            continue
        if consistent_var < MIN_VARIANCE_THRESHOLD_SELECTOR:
            continue
        if consistent_var > best_score:
            best_score = consistent_var
            top = {"key": key, **data}
    return top


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

    if not frame_buffer or len(frame_buffer) < 10:
        return default_result

    is_isometric = bool(
        (exercise or {}).get("isIsometric") is True
        or (
            (exercise or {}).get("category", "").lower() == "core"
            and "isometric" in ((exercise or {}).get("description") or "").lower()
        )
        or any(
            x in ((exercise or {}).get("name") or "").lower()
            for x in ("plank", "hold", "isometric")
        )
    )

    variances = _calculate_all_variances(frame_buffer)
    debug["variances"] = {
        k: {
            "variance": v.get("variance"),
            "robustVariance": v.get("robustVariance"),
            "medianWindowVariance": v.get("medianWindowVariance"),
            "activeWindowCount": v.get("activeWindowCount"),
            "mean": v.get("mean"),
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
    if effective_variance < MIN_VARIANCE_THRESHOLD_SELECTOR:
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
