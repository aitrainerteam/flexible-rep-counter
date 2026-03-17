"""Send frames to YOLO VM and return COCO 17-keypoint landmarks."""
import logging
from typing import Any, Optional

import cv2
import requests

from app.config import YOLO_VM_TARGET_URL, VM_TIMEOUT_SEC

logger = logging.getLogger(__name__)

# COCO 17 keypoint order (matches yolo-deploy / Ultralytics pose)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17


def _landmark_from_xyc(x: float, y: float, c: float) -> dict:
    return {"x": float(x), "y": float(y), "confidence": float(c)}


def _person_keypoints_to_list(kp_dict: dict) -> Optional[list[dict]]:
    """Convert VM response keypoints dict (name -> {x, y, conf}) to list of 17 in COCO order."""
    if not isinstance(kp_dict, dict):
        return None
    out = []
    for name in COCO_KEYPOINT_NAMES:
        pt = kp_dict.get(name)
        if isinstance(pt, dict):
            x = pt.get("x", pt.get("x_center", 0))
            y = pt.get("y", pt.get("y_center", 0))
            c = pt.get("conf", pt.get("confidence", 1.0))
            out.append(_landmark_from_xyc(x, y, c))
        else:
            return None
    return out


def _parse_keypoints(data: Any) -> Optional[list[dict]]:
    """Parse JSON response into list of 17 landmarks {x, y, confidence}. Returns None if invalid."""
    if data is None:
        return None

    # yolo-deploy format: person_1, person_2, ... with keypoints as dict (name -> {x, y, conf})
    if isinstance(data, dict):
        for key in sorted(data.keys()):
            if key.startswith("person_") and isinstance(data[key], dict):
                kp = data[key].get("keypoints")
                parsed = _person_keypoints_to_list(kp)
                if parsed is not None:
                    return parsed

    # Shape: list of 17 items with x, y, confidence
    if isinstance(data, list) and len(data) >= NUM_KEYPOINTS:
        out = []
        for i in range(NUM_KEYPOINTS):
            p = data[i]
            if isinstance(p, dict):
                x = p.get("x", p.get("x_center", 0))
                y = p.get("y", p.get("y_center", 0))
                c = p.get("confidence", p.get("conf", 1.0))
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                x, y, c = p[0], p[1], p[2]
            else:
                return None
            out.append(_landmark_from_xyc(x, y, c))
        return out

    # Nested: results[0].keypoints or predictions[0].keypoints
    for key in ("results", "predictions", "keypoints"):
        arr = data.get(key) if isinstance(data, dict) else None
        if isinstance(arr, list) and len(arr) > 0:
            kp = arr[0].get("keypoints", arr[0]) if isinstance(arr[0], dict) else arr[0]
            parsed = _parse_keypoints(kp)
            if parsed is not None:
                return parsed
        elif isinstance(data, dict) and key == "keypoints":
            parsed = _person_keypoints_to_list(data["keypoints"])
            if parsed is not None:
                return parsed

    # Single object with keypoints key (dict format)
    if isinstance(data, dict) and "keypoints" in data:
        parsed = _person_keypoints_to_list(data["keypoints"])
        if parsed is not None:
            return parsed
        return _parse_keypoints(data["keypoints"])

    return None


def send_frame(frame_bgr) -> Optional[list[dict]]:
    """
    Send a BGR frame to the VM /predict endpoint (multipart file upload) and return 17 landmarks or None.
    Matches the yolo-deploy API: POST /predict with file=image.
    """
    if frame_bgr is None:
        return None
    try:
        _, buf = cv2.imencode(".jpg", frame_bgr)
        if not buf.any():
            return None
        jpeg_bytes = buf.tobytes()
    except Exception:
        return None

    base_url = YOLO_VM_TARGET_URL.rstrip("/")
    predict_url = f"{base_url}/predict"
    try:
        r = requests.post(
            predict_url,
            files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            timeout=VM_TIMEOUT_SEC,
        )
    except requests.RequestException as e:
        logger.debug("VM predict request failed: %s", e)
        return None
    if r.status_code != 200:
        logger.debug("VM predict status=%s body=%s", r.status_code, r.text[:200] if r.text else "")
        return None
    try:
        body = r.json()
    except Exception as e:
        logger.debug("VM predict JSON decode failed: %s", e)
        return None

    parsed = _parse_keypoints(body)
    if parsed is not None:
        return parsed
    if isinstance(body, dict):
        parsed = _parse_keypoints(body.get("keypoints") or body.get("data"))
    if parsed is None:
        logger.debug("VM predict parse failed, keys=%s", list(body.keys()) if isinstance(body, dict) else type(body).__name__)
    return parsed
