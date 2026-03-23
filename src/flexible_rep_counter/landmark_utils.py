"""Pure landmark helpers (no OpenCV)."""
from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def scale_landmarks_to_display(
    landmarks: list[dict],
    sent_hw: tuple[int, int] | None,
    display_hw: tuple[int, int],
) -> list[dict]:
    """Scale keypoints from encoded image size to the current display frame."""
    if not landmarks or not sent_hw:
        return landmarks
    sh, sw = sent_hw[0], sent_hw[1]
    dh, dw = display_hw[0], display_hw[1]
    if sw <= 0 or sh <= 0:
        return landmarks
    if (sh, sw) == (dh, dw):
        return landmarks
    sx, sy = dw / sw, dh / sh
    out: list[dict] = []
    for p in landmarks:
        out.append(
            {
                "x": float(p["x"]) * sx,
                "y": float(p["y"]) * sy,
                "confidence": float(p.get("confidence", 0.0)),
            }
        )
    return out


def keypoints_numpy_to_landmarks(keypoints: Any) -> Optional[list[dict]]:
    """Convert YOLO-style ``(17, 3)`` array to list of 17 ``{x, y, confidence}`` dicts."""
    if np is None or keypoints is None:
        return None
    try:
        arr = np.asarray(keypoints, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (17, 3):
        return None
    out: list[dict] = []
    for i in range(17):
        out.append(
            {
                "x": float(arr[i, 0]),
                "y": float(arr[i, 1]),
                "confidence": float(arr[i, 2]),
            }
        )
    return out
