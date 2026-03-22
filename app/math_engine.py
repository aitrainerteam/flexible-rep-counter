# COCO 17-keypoint indices (yolo26n-pose)
# 0: Nose, 1: L Eye, 2: R Eye, 3: L Ear, 4: R Ear
# 5: L Shoulder, 6: R Shoulder, 7: L Elbow, 8: R Elbow, 9: L Wrist, 10: R Wrist
# 11: L Hip, 12: R Hip, 13: L Knee, 14: R Knee, 15: L Ankle, 16: R Ankle

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

MIN_VARIANCE_THRESHOLD = 5.0
RULE_EVALUATION_MIN_CONFIDENCE = 0.3

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


def _get_xy(p: dict) -> tuple[float, float]:
    return (float(p["x"]), float(p["y"]))


def calculate_angle(a: dict, b: dict, c: dict) -> Optional[float]:
    """Inner angle at B for A-B-C, 0-180 degrees."""
    if not a or not b or not c:
        return None
    ax, ay = _get_xy(a)
    bx, by = _get_xy(b)
    cx, cy = _get_xy(c)
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba == 0 or mag_bc == 0:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def calculate_vertical_angle(a: dict, b: dict) -> Optional[float]:
    """Angle relative to vertical Y-axis; 0 when pointing down, 180 up. Screen Y down."""
    if not a or not b:
        return None
    ax, ay = _get_xy(a)
    bx, by = _get_xy(b)
    dx, dy = bx - ax, by - ay
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg


def calculate_horizontal_angle(a: dict, b: dict) -> Optional[float]:
    """Angle relative to horizontal X-axis; -180 to 180."""
    if not a or not b:
        return None
    ax, ay = _get_xy(a)
    bx, by = _get_xy(b)
    dx, dy = bx - ax, by - ay
    return math.degrees(math.atan2(dy, dx))


def calculate_distance(a: dict, b: dict, scale: float = 1.0) -> Optional[float]:
    if not a or not b:
        return None
    ax, ay = _get_xy(a)
    bx, by = _get_xy(b)
    d = math.hypot(bx - ax, by - ay)
    return d / scale if scale > 0 else d


def calculate_body_scale(landmarks: list) -> float:
    if not landmarks or len(landmarks) < 17:
        return 1.0
    nose = landmarks[0]
    left_ankle = landmarks[15]
    right_ankle = landmarks[16]
    if not nose or not left_ankle or not right_ankle:
        return 1.0
    _, ny = _get_xy(nose)
    _, ly = _get_xy(left_ankle)
    _, ry = _get_xy(right_ankle)
    height = (ly + ry) / 2 - ny
    return abs(height) or 1.0


def calculate_from_type(
    calc_type: str,
    target_landmarks: list[int],
    landmarks: list[dict],
) -> Optional[float]:
    if not landmarks or not target_landmarks:
        return None
    points = [landmarks[i] if i < len(landmarks) else None for i in target_landmarks]
    if any(p is None for p in points):
        return None
    if calc_type == "angle_3_point" and len(points) >= 3:
        return calculate_angle(points[0], points[1], points[2])
    if calc_type == "angle_vertical_axis" and len(points) >= 2:
        return calculate_vertical_angle(points[0], points[1])
    if calc_type == "angle_horizontal_axis" and len(points) >= 2:
        return calculate_horizontal_angle(points[0], points[1])
    if calc_type == "normalized_distance_2_point" and len(points) >= 2:
        scale = calculate_body_scale(landmarks)
        return calculate_distance(points[0], points[1], scale)
    return None


# -----------------------------------------------------------------------------
# Peak detector
# -----------------------------------------------------------------------------

PEAK_STATE_NEUTRAL = "neutral"
PEAK_STATE_GOING_DOWN = "going_down"
PEAK_STATE_GOING_UP = "going_up"


def _rolling_spread_percentile(
    values: Sequence[float],
    low_pct: float,
    high_pct: float,
) -> float:
    """Spread of smoothed angle in the rolling window (robust to single-frame spikes)."""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, high_pct) - np.percentile(arr, low_pct))


def _stddev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr))


class PeakDetector:
    """State machine: NEUTRAL -> GOING_UP/GOING_DOWN; peaks/valleys with hysteresis and min distance.
    First `calibration_reps` reps record peaks/valleys without margin checks; then averages are locked
    and subsequent peaks/valleys must fall within peak_margin/valley_margin of those baselines.
    Optional rolling-window range gate: reps are not recorded until recent p95–p5 spread exceeds
    min_range_gate_degrees (avoids false reps from tiny oscillations; window is not monotonic global min/max).
    Optional per-frame delta deadband: if |angle - last_passed| is below the threshold, the previous
    passed value is reused before EMA smoothing (suppresses sub-threshold jitter from entering the smoother).
    Optional min_rep_interval_ms: suppresses rep increments when two reps would be counted too close in time."""

    def __init__(
        self,
        smoothing_factor: float = 0.3,
        hysteresis: float = 3.0,
        min_peak_distance: int = 10,
        peak_margin: float = 15.0,
        valley_margin: float = 15.0,
        min_range_gate_degrees: float = 15.0,
        range_window_frames: int = 90,
        range_min_samples: int = 12,
        range_percentile_low: float = 5.0,
        range_percentile_high: float = 95.0,
        delta_deadband_degrees: float = 0.0,
        calibration_reps: int = 3,
        calibration_certainty: float = 0.65,
        calibration_force_extra_reps: int = 2,
        min_rep_interval_ms: float = 400.0,
    ):
        self.smoothing_factor = smoothing_factor
        self.hysteresis = hysteresis
        self.min_peak_distance = min_peak_distance
        self.peak_margin = peak_margin
        self.valley_margin = valley_margin
        self.calibration_reps = max(1, int(calibration_reps))
        self.calibration_certainty = max(0.0, min(1.0, float(calibration_certainty)))
        self.calibration_force_extra_reps = max(0, int(calibration_force_extra_reps))
        self.min_rep_interval_ms = max(0.0, float(min_rep_interval_ms))
        self.min_range_gate_degrees = min_range_gate_degrees
        self.delta_deadband_degrees = max(0.0, float(delta_deadband_degrees))
        self.range_window_frames = max(1, int(range_window_frames))
        self.range_min_samples = max(2, min(int(range_min_samples), self.range_window_frames))
        self.range_percentile_low = float(range_percentile_low)
        self.range_percentile_high = float(range_percentile_high)
        self._value_window: deque[float] = deque(maxlen=self.range_window_frames)
        self._last_rolling_range: float = 0.0
        self._last_range_gate_open: bool = True
        self.state = PEAK_STATE_NEUTRAL
        self.smoothed_value: Optional[float] = None
        self.last_peak_frame = -min_peak_distance
        self.frame_count = 0
        self.rep_count = 0
        self.neutral_frame_count = 0
        self.peaks: list[float] = []
        self.valleys: list[float] = []
        self.current_peak_value: Optional[float] = None
        self.current_valley_value: Optional[float] = None
        self._last_debanded_pass: Optional[float] = None
        # True after calibration_reps reps: locked baselines for strict margin checks
        self._calibrated: bool = False
        self._calibrated_avg_peak: Optional[float] = None
        self._calibrated_avg_valley: Optional[float] = None
        self._last_rep_time_ms: Optional[float] = None

    def _calibration_stats(self) -> dict[str, float | None]:
        if not self.peaks or not self.valleys:
            return {
                "avgPeak": None,
                "avgValley": None,
                "peakStd": None,
                "valleyStd": None,
                "certainty": 0.0,
                "amplitude": None,
            }
        avg_peak = sum(self.peaks) / len(self.peaks)
        avg_valley = sum(self.valleys) / len(self.valleys)
        amplitude = avg_peak - avg_valley
        if amplitude <= 0:
            return {
                "avgPeak": avg_peak,
                "avgValley": avg_valley,
                "peakStd": None,
                "valleyStd": None,
                "certainty": 0.0,
                "amplitude": amplitude,
            }
        peak_std = _stddev(self.peaks)
        valley_std = _stddev(self.valleys)
        combined_jitter_ratio = (peak_std + valley_std) / max(amplitude, 1e-6)
        # certainty=1 when extrema are very consistent relative to movement amplitude.
        # Denominator 1.0 is more forgiving than 0.5 (small ROM / few samples still usable).
        certainty = max(0.0, min(1.0, 1.0 - (combined_jitter_ratio / 1.0)))
        return {
            "avgPeak": avg_peak,
            "avgValley": avg_valley,
            "peakStd": peak_std,
            "valleyStd": valley_std,
            "certainty": certainty,
            "amplitude": amplitude,
        }

    def _maybe_lock_calibration(self) -> None:
        if self._calibrated:
            return
        if self.rep_count < self.calibration_reps:
            return
        stats = self._calibration_stats()
        certainty = float(stats.get("certainty") or 0.0)
        force_at = self.calibration_reps + self.calibration_force_extra_reps
        # REP_CALIBRATION_CERTAINTY=0 disables certainty gate (lock on min reps only).
        certainty_ok = self.calibration_certainty <= 0 or certainty >= self.calibration_certainty
        force_ok = self.rep_count >= force_at
        if certainty_ok or force_ok:
            self._calibrated = True
            self._calibrated_avg_peak = float(stats["avgPeak"]) if stats.get("avgPeak") is not None else None
            self._calibrated_avg_valley = float(stats["avgValley"]) if stats.get("avgValley") is not None else None

    def _pass_through_deadband(self, raw_value: float) -> float:
        """Hold previous angle when frame-to-frame change is below threshold; else accept new sample."""
        if self.delta_deadband_degrees <= 0:
            self._last_debanded_pass = raw_value
            return raw_value
        if self._last_debanded_pass is None:
            self._last_debanded_pass = raw_value
            return raw_value
        if abs(raw_value - self._last_debanded_pass) < self.delta_deadband_degrees:
            return self._last_debanded_pass
        self._last_debanded_pass = raw_value
        return raw_value

    def _update_rolling_range(self) -> None:
        if self.smoothed_value is None:
            self._last_rolling_range = 0.0
            self._last_range_gate_open = self.min_range_gate_degrees <= 0
            return
        self._value_window.append(self.smoothed_value)
        spread = _rolling_spread_percentile(
            self._value_window,
            self.range_percentile_low,
            self.range_percentile_high,
        )
        self._last_rolling_range = spread
        if self.min_range_gate_degrees <= 0:
            self._last_range_gate_open = True
        elif len(self._value_window) < self.range_min_samples:
            self._last_range_gate_open = False
        else:
            self._last_range_gate_open = spread >= self.min_range_gate_degrees

    def _range_gate_allows_rep_recording(self) -> bool:
        return self._last_range_gate_open

    def update(self, raw_value: Optional[float]) -> dict[str, Any]:
        if raw_value is None:
            self._last_debanded_pass = None
            stats = self._calibration_stats()
            return {
                "repCompleted": False,
                "peak": None,
                "valley": None,
                "smoothedValue": self.smoothed_value,
                "state": self.state,
                "repCount": self.rep_count,
                "rollingRange": self._last_rolling_range,
                "rangeGateOpen": self._last_range_gate_open,
                "calibrationComplete": self._calibrated,
                "calibrationTargetReps": self.calibration_reps,
                "calibrationCertainty": float(stats.get("certainty") or 0.0),
                "calibrationCertaintyTarget": self.calibration_certainty,
            }

        feed = self._pass_through_deadband(raw_value)
        self.frame_count += 1
        if self.smoothed_value is None:
            self.smoothed_value = feed
        else:
            self.smoothed_value = (
                self.smoothing_factor * feed
                + (1 - self.smoothing_factor) * self.smoothed_value
            )
        self._update_rolling_range()

        rep_completed = False
        detected_peak = None
        detected_valley = None

        if self.state == PEAK_STATE_NEUTRAL:
            if self.current_peak_value is None or self.current_valley_value is None:
                self.current_peak_value = self.smoothed_value
                self.current_valley_value = self.smoothed_value
            self.neutral_frame_count += 1
            if self.neutral_frame_count > 30:
                drift = 0.05
                self.current_peak_value = (1 - drift) * self.current_peak_value + drift * self.smoothed_value
                self.current_valley_value = (1 - drift) * self.current_valley_value + drift * self.smoothed_value
            if self.smoothed_value > self.current_peak_value + self.hysteresis:
                self.state = PEAK_STATE_GOING_UP
                self.current_peak_value = self.smoothed_value
                self.neutral_frame_count = 0
            elif self.smoothed_value < self.current_valley_value - self.hysteresis:
                self.state = PEAK_STATE_GOING_DOWN
                self.current_valley_value = self.smoothed_value
                self.neutral_frame_count = 0

        elif self.state == PEAK_STATE_GOING_UP:
            if self.smoothed_value > self.current_peak_value:
                self.current_peak_value = self.smoothed_value
            elif self.smoothed_value < self.current_peak_value - self.hysteresis:
                if self.frame_count - self.last_peak_frame >= self.min_peak_distance:
                    if not self._calibrated:
                        within_margin = True
                    else:
                        within_margin = (
                            self._calibrated_avg_peak is not None
                            and self.current_peak_value >= self._calibrated_avg_peak - self.peak_margin
                        )
                    if within_margin and self._range_gate_allows_rep_recording():
                        new_rep_count_if = min(len(self.peaks) + 1, len(self.valleys))
                        rep_would_increment = new_rep_count_if > self.rep_count
                        now_ms = time.time() * 1000.0
                        interval_ok = True
                        if rep_would_increment and self.min_rep_interval_ms > 0:
                            if self._last_rep_time_ms is not None and (
                                now_ms - self._last_rep_time_ms
                            ) < self.min_rep_interval_ms:
                                interval_ok = False
                        if interval_ok:
                            detected_peak = self.current_peak_value
                            self.peaks.append(self.current_peak_value)
                            self.last_peak_frame = self.frame_count
                            new_rep_count = min(len(self.peaks), len(self.valleys))
                            if new_rep_count > self.rep_count:
                                self.rep_count = new_rep_count
                                rep_completed = True
                                self._last_rep_time_ms = now_ms
                            self._maybe_lock_calibration()
                self.state = PEAK_STATE_GOING_DOWN
                self.current_valley_value = self.smoothed_value

        elif self.state == PEAK_STATE_GOING_DOWN:
            if self.smoothed_value < self.current_valley_value:
                self.current_valley_value = self.smoothed_value
            elif self.smoothed_value > self.current_valley_value + self.hysteresis:
                if self.frame_count - self.last_peak_frame >= self.min_peak_distance:
                    if not self._calibrated:
                        within_margin = True
                    else:
                        within_margin = (
                            self._calibrated_avg_valley is not None
                            and self.current_valley_value <= self._calibrated_avg_valley + self.valley_margin
                        )
                    if within_margin and self._range_gate_allows_rep_recording():
                        new_rep_count_if = min(len(self.peaks), len(self.valleys) + 1)
                        rep_would_increment = new_rep_count_if > self.rep_count
                        now_ms = time.time() * 1000.0
                        interval_ok = True
                        if rep_would_increment and self.min_rep_interval_ms > 0:
                            if self._last_rep_time_ms is not None and (
                                now_ms - self._last_rep_time_ms
                            ) < self.min_rep_interval_ms:
                                interval_ok = False
                        if interval_ok:
                            detected_valley = self.current_valley_value
                            self.valleys.append(self.current_valley_value)
                            self.last_peak_frame = self.frame_count
                            new_rep_count = min(len(self.peaks), len(self.valleys))
                            if new_rep_count > self.rep_count:
                                self.rep_count = new_rep_count
                                rep_completed = True
                                self._last_rep_time_ms = now_ms
                            self._maybe_lock_calibration()
                self.state = PEAK_STATE_GOING_UP
                self.current_peak_value = self.smoothed_value

        stats = self._calibration_stats()
        return {
            "repCompleted": rep_completed,
            "peak": detected_peak,
            "valley": detected_valley,
            "smoothedValue": self.smoothed_value,
            "state": self.state,
            "repCount": self.rep_count,
            "rollingRange": self._last_rolling_range,
            "rangeGateOpen": self._last_range_gate_open,
            "calibrationComplete": self._calibrated,
            "calibrationTargetReps": self.calibration_reps,
            "calibrationCertainty": float(stats.get("certainty") or 0.0),
            "calibrationCertaintyTarget": self.calibration_certainty,
        }

    def get_rep_count(self) -> int:
        return self.rep_count

    def get_peaks(self) -> list[float]:
        return list(self.peaks)

    def get_valleys(self) -> list[float]:
        return list(self.valleys)

    def get_extremes(self) -> dict[str, Optional[float]]:
        return {
            "minPeak": min(self.peaks) if self.peaks else None,
            "maxPeak": max(self.peaks) if self.peaks else None,
            "minValley": min(self.valleys) if self.valleys else None,
            "maxValley": max(self.valleys) if self.valleys else None,
        }

    def reset(self) -> None:
        self.state = PEAK_STATE_NEUTRAL
        self.smoothed_value = None
        self.last_peak_frame = -self.min_peak_distance
        self.frame_count = 0
        self.rep_count = 0
        self.neutral_frame_count = 0
        self.peaks.clear()
        self.valleys.clear()
        self.current_peak_value = None
        self.current_valley_value = None
        self._value_window.clear()
        self._last_rolling_range = 0.0
        self._last_range_gate_open = self.min_range_gate_degrees <= 0
        self._last_debanded_pass = None
        self._calibrated = False
        self._calibrated_avg_peak = None
        self._calibrated_avg_valley = None
        self._last_rep_time_ms = None

    def get_state(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "smoothedValue": self.smoothed_value,
            "frameCount": self.frame_count,
            "repCount": self.rep_count,
            "currentPeakValue": self.current_peak_value,
            "currentValleyValue": self.current_valley_value,
            "calibrationComplete": self._calibrated,
            "calibrationTargetReps": self.calibration_reps,
            "calibrationCertainty": float(self._calibration_stats().get("certainty") or 0.0),
            "calibrationCertaintyTarget": self.calibration_certainty,
            "calibratedAvgPeak": self._calibrated_avg_peak,
            "calibratedAvgValley": self._calibrated_avg_valley,
        }


def create_peak_detector(
    smoothing_factor: float = 0.3,
    hysteresis: float = 3.0,
    min_peak_distance: int = 10,
    peak_margin: float = 15.0,
    valley_margin: float = 15.0,
    min_range_gate_degrees: float = 15.0,
    range_window_frames: int = 90,
    range_min_samples: int = 12,
    range_percentile_low: float = 5.0,
    range_percentile_high: float = 95.0,
    delta_deadband_degrees: float = 0.0,
    calibration_reps: int = 3,
    calibration_certainty: float = 0.65,
    calibration_force_extra_reps: int = 2,
    min_rep_interval_ms: float = 400.0,
) -> PeakDetector:
    return PeakDetector(
        smoothing_factor=smoothing_factor,
        hysteresis=hysteresis,
        min_peak_distance=min_peak_distance,
        peak_margin=peak_margin,
        valley_margin=valley_margin,
        min_range_gate_degrees=min_range_gate_degrees,
        range_window_frames=range_window_frames,
        range_min_samples=range_min_samples,
        range_percentile_low=range_percentile_low,
        range_percentile_high=range_percentile_high,
        delta_deadband_degrees=delta_deadband_degrees,
        calibration_reps=calibration_reps,
        calibration_certainty=calibration_certainty,
        calibration_force_extra_reps=calibration_force_extra_reps,
        min_rep_interval_ms=min_rep_interval_ms,
    )


def replay_angle_series_on_peak_detector(
    detector: PeakDetector,
    values: Sequence[Optional[float]],
) -> dict[str, Any]:
    """Feed a calibration-window angle series through the detector (same logic as live updates).

    Includes None samples like the main loop so frame_count and deadband behavior match live.
    Populates peaks, valleys, rep_count, and locks calibration when rep_count reaches calibration_reps.
    """
    last: dict[str, Any] = {}
    for v in values:
        last = detector.update(v)
    return last


# -----------------------------------------------------------------------------
# Variance and robust stats
# -----------------------------------------------------------------------------


def smooth_angle_series(values: list[float], window: int = 5) -> list[float]:
    """Light moving average on angle samples to suppress single-frame jitter before variance/range stats."""
    if not values:
        return []
    if len(values) < 2 or window <= 1:
        return list(values)
    w = min(window, len(values))
    half = w // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), lo + w)
        lo = max(0, hi - w)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def calculate_variance(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "variance": 0.0, "stdDev": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = math.sqrt(variance)
    return {"mean": mean, "variance": variance, "stdDev": std_dev}


def compute_robust_variance(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "variance": 0.0, "stdDev": 0.0, "filteredCount": 0, "outlierCount": 0}
    if len(values) < 4:
        out = calculate_variance(values)
        out["filteredCount"] = len(values)
        out["outlierCount"] = 0
        return out
    sorted_vals = sorted(values)
    q1_idx = int(len(sorted_vals) * 0.25)
    q3_idx = int(len(sorted_vals) * 0.75)
    q1, q3 = sorted_vals[q1_idx], sorted_vals[q3_idx]
    iqr = max(q3 - q1, 1e-6)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [v for v in values if lo <= v <= hi]
    if len(filtered) < 2:
        out = calculate_variance(values)
        out["filteredCount"] = len(values)
        out["outlierCount"] = 0
        return out
    mean = sum(filtered) / len(filtered)
    variance = sum((v - mean) ** 2 for v in filtered) / len(filtered)
    std_dev = math.sqrt(variance)
    return {
        "mean": mean,
        "variance": variance,
        "stdDev": std_dev,
        "filteredCount": len(filtered),
        "outlierCount": len(values) - len(filtered),
    }


def compute_consistent_variance_score(
    values: list[float],
    min_window_size: int = 12,
) -> dict[str, Any]:
    if not values or len(values) < min_window_size * 2:
        return {"medianWindowVariance": 0.0, "activeWindowCount": 0, "windowVariances": []}
    num_windows = min(4, max(2, len(values) // min_window_size))
    window_size = len(values) // num_windows
    window_variances = []
    for i in range(num_windows):
        start = i * window_size
        end = len(values) if i == num_windows - 1 else start + window_size
        window = values[start:end]
        if len(window) < 5:
            continue
        window_variances.append(calculate_variance(window)["variance"])
    if not window_variances:
        return {"medianWindowVariance": 0.0, "activeWindowCount": 0, "windowVariances": []}
    sorted_wv = sorted(window_variances)
    mid = len(sorted_wv) // 2
    median_var = (sorted_wv[mid - 1] + sorted_wv[mid]) / 2 if len(sorted_wv) % 2 == 0 else sorted_wv[mid]
    active_count = sum(1 for v in window_variances if v >= MIN_VARIANCE_THRESHOLD)
    return {
        "medianWindowVariance": median_var,
        "activeWindowCount": active_count,
        "windowVariances": window_variances,
    }


def get_average_confidence_for_landmarks(
    frame_buffer: list[list[dict]],
    landmark_indices: list[int],
) -> float:
    if not frame_buffer or not landmark_indices:
        return 0.0
    total, count = 0.0, 0
    for landmarks in frame_buffer:
        if not landmarks:
            continue
        for idx in landmark_indices:
            if idx >= len(landmarks):
                continue
            p = landmarks[idx]
            if p and isinstance(p.get("confidence"), (int, float)):
                total += float(p["confidence"])
                count += 1
    return total / count if count else 0.0


def get_min_confidence_for_landmarks(
    landmarks: list[dict],
    landmark_indices: list[int],
) -> Optional[float]:
    if not landmarks or not landmark_indices:
        return None
    min_conf = 1.0
    for idx in landmark_indices:
        if idx >= len(landmarks):
            return None
        p = landmarks[idx]
        if not p:
            return None
        c = float(p["confidence"]) if isinstance(p.get("confidence"), (int, float)) else 1.0
        min_conf = min(min_conf, c)
    return min_conf
