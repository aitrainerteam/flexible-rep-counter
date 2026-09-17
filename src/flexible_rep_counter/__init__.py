"""Importable rep-counter engine and types."""

__version__ = "3.0.1"

from flexible_rep_counter.angle_session import AngleRepCounterSession
from flexible_rep_counter.core.math_engine import calculate_angle_3d
from flexible_rep_counter.core.settings_3d import Angle3dConfig, load_3d_config
from flexible_rep_counter.landmark_utils import (
    keypoints_numpy_to_landmarks,
    scale_landmarks_to_display,
)
from flexible_rep_counter.instrumentation import RepInstrumentationSettings
from flexible_rep_counter.session import RepCounterSession
from flexible_rep_counter.types import AngleSample, AngleStepResult, StepResult

__all__ = [
    "__version__",
    "AngleRepCounterSession",
    "AngleSample",
    "AngleStepResult",
    "Angle3dConfig",
    "calculate_angle_3d",
    "load_3d_config",
    "RepCounterSession",
    "RepInstrumentationSettings",
    "StepResult",
    "keypoints_numpy_to_landmarks",
    "scale_landmarks_to_display",
]
