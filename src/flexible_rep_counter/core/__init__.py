"""Core rep engine: geometry, peak detection, variance selection, pose filters."""

from flexible_rep_counter.core.math_engine import (
    PeakDetector,
    calculate_angle,
    calculate_from_type,
    create_peak_detector,
    replay_angle_series_on_peak_detector,
)
from flexible_rep_counter.core.pose_filters import PoseFilterPipeline
from flexible_rep_counter.core.settings import get_default_tuning_params
from flexible_rep_counter.core.variance_angle_selector import COMMON_ANGLES

__all__ = [
    "COMMON_ANGLES",
    "PeakDetector",
    "PoseFilterPipeline",
    "calculate_angle",
    "calculate_from_type",
    "create_peak_detector",
    "get_default_tuning_params",
    "replay_angle_series_on_peak_detector",
]
