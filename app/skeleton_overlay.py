# COCO 17 skeleton overlay on OpenCV frames.
# Connections match feedback-program usePoseEstimation SKELETON_CONNECTIONS.

from __future__ import annotations

import cv2
from typing import Optional

# COCO 17: 0 nose, 1 L eye, 2 R eye, 3 L ear, 4 R ear,
# 5 L shoulder, 6 R shoulder, 7 L elbow, 8 R elbow, 9 L wrist, 10 R wrist,
# 11 L hip, 12 R hip, 13 L knee, 14 R knee, 15 L ankle, 16 R ankle
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # face
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_LINE_COLOR = (0, 255, 0)    # BGR green
DEFAULT_POINT_COLOR = (0, 0, 255)   # BGR red
DEFAULT_LINE_THICKNESS = 2
DEFAULT_POINT_RADIUS = 4


def _get_xy(kp: dict, frame_shape: Optional[tuple] = None) -> Optional[tuple[int, int]]:
    """Return (x, y) as integers; optionally clamp to frame. Returns None if low confidence."""
    x = kp.get("x")
    y = kp.get("y")
    conf = kp.get("confidence", 0.0)
    if x is None or y is None or conf < DEFAULT_CONFIDENCE_THRESHOLD:
        return None
    ix, iy = int(round(x)), int(round(y))
    if frame_shape and len(frame_shape) >= 2:
        h, w = frame_shape[0], frame_shape[1]
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
    return (ix, iy)


def draw_skeleton(
    frame,
    landmarks: list[dict],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    line_color: tuple = DEFAULT_LINE_COLOR,
    point_color: tuple = DEFAULT_POINT_COLOR,
    line_thickness: int = DEFAULT_LINE_THICKNESS,
    point_radius: int = DEFAULT_POINT_RADIUS,
) -> None:
    """
    Draw COCO 17 skeleton and keypoints on an OpenCV BGR frame in-place.
    landmarks: list of 17 dicts with x, y, confidence.
    """
    if not landmarks or len(landmarks) < 17:
        return
    h, w = frame.shape[:2]

    for (i, j) in SKELETON_CONNECTIONS:
        pt1 = _get_xy(landmarks[i], frame.shape)
        pt2 = _get_xy(landmarks[j], frame.shape)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)

    for kp in landmarks:
        pt = _get_xy(kp, frame.shape)
        if pt:
            cv2.circle(frame, pt, point_radius, point_color, -1, cv2.LINE_AA)
