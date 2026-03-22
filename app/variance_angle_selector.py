"""Variance-based angle selection: pick the angle with most consistent movement (median window variance)."""
from __future__ import annotations

import os
from typing import Any, Optional

from app.config import (
    ANGLE_SELECTION_MIN_RANGE_DEG,
    ANGLE_SELECTION_MIN_VARIANCE,
    ANGLE_SELECTION_SECOND_BEST_RATIO,
    get_default_tuning_params,
)
from app.math_engine import (
    MIN_VARIANCE_THRESHOLD,
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
# Require several time segments with real variance (not one accidental spike or jitter).
MIN_ACTIVE_WINDOWS = 3
SMOOTH_WINDOW = int(os.environ.get("ANGLE_SELECTION_SMOOTH_WINDOW", "5"))
ISOMETRIC_FALLBACK_ANGLE = "LEFT_HIP"


def _angle_selection_thresholds(angle_key: str) -> dict[str, float]:
    """
    Per-common-angle gates from env: ANGLE_SELECTION_<NAME>_<KEY>, e.g.
    ANGLE_SELECTION_MIN_RANGE_DEG_LEFT_ELBOW, ANGLE_SELECTION_MIN_VARIANCE_RIGHT_KNEE.
    Falls back to global ANGLE_SELECTION_* from config.
    """
    suf = f"_{angle_key}"
    return {
        "min_variance": float(
            os.environ.get(f"ANGLE_SELECTION_MIN_VARIANCE{suf}", str(ANGLE_SELECTION_MIN_VARIANCE))
        ),
        "min_range_deg": float(
            os.environ.get(f"ANGLE_SELECTION_MIN_RANGE_DEG{suf}", str(ANGLE_SELECTION_MIN_RANGE_DEG))
        ),
        "second_best_ratio": float(
            os.environ.get(
                f"ANGLE_SELECTION_SECOND_BEST_RATIO{suf}", str(ANGLE_SELECTION_SECOND_BEST_RATIO)
            )
        ),
    }


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


def _get_top_candidate(variances: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick the clearest winner: multi-window activity, meaningful ROM, and margin over the runner-up."""
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for key, data in variances.items():
        t = _angle_selection_thresholds(key)
        consistent_var = float(data.get("medianWindowVariance") or 0.0)
        active_windows = int(data.get("activeWindowCount") or 0)
        span_deg = float(data.get("smoothedRangeDeg") or 0.0)
        if active_windows < MIN_ACTIVE_WINDOWS:
            continue
        if consistent_var < t["min_variance"]:
            continue
        if span_deg < t["min_range_deg"]:
            continue
        ranked.append((consistent_var, key, data))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        return None
    top_score, top_key, top_data = ranked[0]
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
