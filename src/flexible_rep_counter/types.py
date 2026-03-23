"""Public result types for :class:`flexible_rep_counter.session.RepCounterSession`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class StepResult:
    """Per-frame output after processing smoothed landmarks."""

    reps: int
    """Rep count shown to the user (0 until calibration completes)."""

    reps_raw: int
    """Internal rep count from the peak detector (includes calibration reps)."""

    tracked_joint: Optional[str]
    angle_3_point_value: Optional[float]
    target_landmarks: Optional[list[int]]
    tuning_params: dict[str, Any]
    avg_peak: Optional[float]
    avg_valley: Optional[float]
    calibration_complete: bool
    peak_detector_state: str
    smoothed_value: Optional[float]
    range_gate_open: bool
    rolling_range: Optional[float]
    calibration_target_reps: int
    calibration_certainty: float
    calibration_certainty_target: float
    phase: Literal["idle", "selecting", "tracking"]
    status_message: str
    tracking_detail_message: str
    """Second line for tracking phase (calibration / range gate text)."""

    leader_key: Optional[str] = None
    selection_debug: dict[str, Any] = field(default_factory=dict)
