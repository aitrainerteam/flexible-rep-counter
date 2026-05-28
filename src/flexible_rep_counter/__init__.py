"""Importable rep-counter engine and types."""

__version__ = "2.0.0"

from flexible_rep_counter.landmark_utils import (
    keypoints_numpy_to_landmarks,
    scale_landmarks_to_display,
)
from flexible_rep_counter.instrumentation import RepInstrumentationSettings
from flexible_rep_counter.session import RepCounterSession
from flexible_rep_counter.types import StepResult

__all__ = [
    "__version__",
    "RepCounterSession",
    "RepInstrumentationSettings",
    "StepResult",
    "keypoints_numpy_to_landmarks",
    "scale_landmarks_to_display",
]
