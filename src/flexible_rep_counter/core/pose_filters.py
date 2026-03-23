# Temporal smoothing and filtering for pose keypoints.
# Port of feedback-program poseFilters: One Euro, Simple EMA, velocity clamping,
# confidence-weighted interpolation.

from __future__ import annotations

import time
import math
from typing import Optional

# COCO 17 keypoint indices -> body part for per-joint smoothing
KEYPOINT_BODY_PARTS = [
    "face", "face", "face", "face", "face",   # 0-4: nose, eyes, ears
    "torso", "torso",                          # 5-6: shoulders
    "arms", "arms", "arms", "arms",            # 7-10: elbows, wrists
    "torso", "torso",                          # 11-12: hips
    "legs", "legs", "legs", "legs",            # 13-16: knees, ankles
]

# EMA alpha: higher = less smoothing (more responsive)
EMA_ALPHA = {
    "face": 0.4,
    "torso": 0.5,
    "arms": 0.6,
    "legs": 0.5,
}

# One Euro (optional): minCutoff, beta, dCutoff
ONE_EURO_CONFIG = {
    "face": {"min_cutoff": 2.0, "beta": 0.02, "d_cutoff": 1.0},
    "torso": {"min_cutoff": 3.0, "beta": 0.01, "d_cutoff": 1.0},
    "arms": {"min_cutoff": 4.0, "beta": 0.02, "d_cutoff": 1.0},
    "legs": {"min_cutoff": 3.0, "beta": 0.015, "d_cutoff": 1.0},
}

HISTORY_SIZE = 5
CONFIDENCE_THRESHOLD = 0.3
MAX_VELOCITY_PER_FRAME = 50.0


# -----------------------------------------------------------------------------
# Simple EMA
# -----------------------------------------------------------------------------

class SimpleEMA:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._last: Optional[float] = None

    def filter(self, value: float) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return self._last if self._last is not None else 0.0
        if self._last is None:
            self._last = value
            return value
        self._last = self.alpha * value + (1.0 - self.alpha) * self._last
        return self._last

    def reset(self) -> None:
        self._last = None


def create_simple_keypoint_filters() -> list[dict]:
    """One EMA for x and one for y per keypoint, with body-part alpha."""
    return [
        {"x": SimpleEMA(EMA_ALPHA[KEYPOINT_BODY_PARTS[i]]), "y": SimpleEMA(EMA_ALPHA[KEYPOINT_BODY_PARTS[i]])}
        for i in range(17)
    ]


def filter_keypoints_simple(keypoints: list[dict], filters: list[dict]) -> list[dict]:
    if not keypoints or not filters or len(keypoints) != 17 or len(filters) != 17:
        return keypoints
    out = []
    for i, kp in enumerate(keypoints):
        x = kp.get("x", 0.0)
        y = kp.get("y", 0.0)
        c = kp.get("confidence", 1.0)
        out.append({
            "x": filters[i]["x"].filter(x),
            "y": filters[i]["y"].filter(y),
            "confidence": c,
        })
    return out


def reset_simple_keypoint_filters(filters: list[dict]) -> None:
    if not filters:
        return
    for f in filters:
        f["x"].reset()
        f["y"].reset()


# -----------------------------------------------------------------------------
# One Euro Filter
# -----------------------------------------------------------------------------

class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_filter: Optional[float] = None
        self._dx_filter: Optional[float] = None
        self._last_time: Optional[float] = None
        self._last_value: Optional[float] = None

    @staticmethod
    def _low_pass(alpha: float, x: float, prev_x: float) -> float:
        return alpha * x + (1.0 - alpha) * prev_x

    @staticmethod
    def _alpha_from_cutoff(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt) if dt > 0 else 1.0

    def filter(self, value: float, timestamp_ms: Optional[float] = None) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return self._x_filter if self._x_filter is not None else 0.0
        t = timestamp_ms if timestamp_ms is not None else time.time() * 1000.0
        if self._last_time is None:
            self._last_time = t
            self._last_value = value
            self._x_filter = value
            return value
        dt = max((t - self._last_time) / 1000.0, 0.001)
        self._last_time = t
        dx = (value - (self._last_value or value)) / dt
        self._last_value = value
        alpha_d = self._alpha_from_cutoff(self.d_cutoff, dt)
        self._dx_filter = dx if self._dx_filter is None else self._low_pass(alpha_d, dx, self._dx_filter)
        cutoff = self.min_cutoff + self.beta * abs(self._dx_filter)
        alpha = self._alpha_from_cutoff(cutoff, dt)
        self._x_filter = value if self._x_filter is None else self._low_pass(alpha, value, self._x_filter)
        return self._x_filter

    def reset(self) -> None:
        self._x_filter = None
        self._dx_filter = None
        self._last_time = None
        self._last_value = None


def create_one_euro_keypoint_filters() -> list[dict]:
    """One Euro for x, y (and optional conf) per keypoint."""
    return [
        {
            "x": OneEuroFilter(**ONE_EURO_CONFIG[KEYPOINT_BODY_PARTS[i]]),
            "y": OneEuroFilter(**ONE_EURO_CONFIG[KEYPOINT_BODY_PARTS[i]]),
        }
        for i in range(17)
    ]


def filter_keypoints_one_euro(
    keypoints: list[dict], filters: list[dict], timestamp_ms: Optional[float] = None
) -> list[dict]:
    if not keypoints or not filters or len(keypoints) != 17 or len(filters) != 17:
        return keypoints
    t = timestamp_ms if timestamp_ms is not None else time.time() * 1000.0
    return [
        {
            "x": filters[i]["x"].filter(kp.get("x", 0.0), t),
            "y": filters[i]["y"].filter(kp.get("y", 0.0), t),
            "confidence": kp.get("confidence", 1.0),
        }
        for i, kp in enumerate(keypoints)
    ]


def reset_one_euro_filters(filters: list[dict]) -> None:
    if not filters:
        return
    for f in filters:
        f["x"].reset()
        f["y"].reset()


# -----------------------------------------------------------------------------
# Velocity clamping
# -----------------------------------------------------------------------------

def clamp_velocity(
    current: dict, previous: Optional[dict], max_velocity: float = MAX_VELOCITY_PER_FRAME
) -> dict:
    if not previous or previous.get("x") is None or previous.get("y") is None:
        return dict(current)
    if current.get("x") is None or current.get("y") is None:
        return dict(previous)
    dx = current["x"] - previous["x"]
    dy = current["y"] - previous["y"]
    dist = math.hypot(dx, dy)
    if dist <= max_velocity or dist == 0:
        return dict(current)
    scale = max_velocity / dist
    return {
        "x": previous["x"] + dx * scale,
        "y": previous["y"] + dy * scale,
        "confidence": current.get("confidence", 1.0),
    }


# -----------------------------------------------------------------------------
# Confidence-weighted interpolation
# -----------------------------------------------------------------------------

def interpolate_keypoint(
    current: dict, history: list[dict], confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> dict:
    if not current or current.get("confidence", 0) >= confidence_threshold:
        return dict(current)
    if not history:
        return dict(current)
    for h in reversed(history):
        if h and h.get("confidence", 0) >= confidence_threshold:
            w = current.get("confidence", 0) / max(confidence_threshold, 1e-6)
            return {
                "x": current.get("x", 0) * w + h.get("x", 0) * (1 - w),
                "y": current.get("y", 0) * w + h.get("y", 0) * (1 - w),
                "confidence": current.get("confidence", 0),
            }
    return dict(current)


# -----------------------------------------------------------------------------
# Pipeline state and process
# -----------------------------------------------------------------------------

class PoseFilterPipeline:
    """Temporal smoothing pipeline: EMA or One Euro, then velocity clamp, then optional interpolation."""

    def __init__(self, use_one_euro: bool = False):
        self.use_one_euro = use_one_euro
        self._simple = create_simple_keypoint_filters()
        self._one_euro = create_one_euro_keypoint_filters() if use_one_euro else None
        self._previous: Optional[list[dict]] = None
        self._history: list[list[dict]] = []

    def reset(self) -> None:
        reset_simple_keypoint_filters(self._simple)
        if self._one_euro:
            reset_one_euro_filters(self._one_euro)
        self._previous = None
        self._history.clear()

    def process(self, keypoints: list[dict], timestamp_ms: Optional[float] = None) -> list[dict]:
        if not keypoints or len(keypoints) != 17:
            return keypoints
        # 1. Temporal smoothing
        if self.use_one_euro and self._one_euro:
            smoothed = filter_keypoints_one_euro(keypoints, self._one_euro, timestamp_ms)
        else:
            smoothed = filter_keypoints_simple(keypoints, self._simple)
        # 2. Velocity clamping
        if self._previous is not None:
            smoothed = [clamp_velocity(smoothed[i], self._previous[i]) for i in range(17)]
        self._previous = smoothed
        # 3. Confidence-weighted interpolation (per-keypoint history)
        if self._history:
            keypoint_history = [[f[i] for f in self._history] for i in range(17)]
            smoothed = [
                interpolate_keypoint(smoothed[i], keypoint_history[i])
                for i in range(17)
            ]
        self._history.append(smoothed)
        if len(self._history) > HISTORY_SIZE:
            self._history.pop(0)
        return smoothed
