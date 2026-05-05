"""Send frames to YOLO VM and return COCO 17-keypoint landmarks."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import requests

from app.config import VM_BASE_URL, VM_TIMEOUT_SEC

logger = logging.getLogger(__name__)

# COCO 17 keypoint order (matches yolo-deploy / Ultralytics pose).
# Face slots are kept for index compatibility, but may be absent in payloads.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17
FACE_KEYPOINT_COUNT = 5
BODY_KEYPOINT_NAMES = COCO_KEYPOINT_NAMES[FACE_KEYPOINT_COUNT:]
VM_PARSE_LEGACY_FORMATS = os.environ.get("VM_PARSE_LEGACY_FORMATS", "false").lower() in (
    "1",
    "true",
    "yes",
)


@dataclass
class VmPredictOutcome:
    """Result of one /predict call including client-side timing (yolo-deploy-style).

    benchmark keys: encode_ms, roundtrip_ms, upload_ms, payload_kb, inference_ms (server),
    response_time (perf_counter). sent_hw is (height, width) of the encoded JPEG for display scaling.
    """

    landmarks: Optional[list[dict]]
    benchmark: Optional[dict[str, Any]] = None
    sent_hw: Optional[tuple[int, int]] = None
    validation_issues: list[str] = field(default_factory=list)
    rep_counter: Optional[dict] = None
    rep_session_reset: Optional[dict] = None


def _landmark_from_xyc(x: float, y: float, c: float) -> dict:
    return {"x": float(x), "y": float(y), "confidence": float(c)}


def _empty_landmark() -> dict:
    return {"x": 0.0, "y": 0.0, "confidence": 0.0}


def validate_predict_response(data: Any) -> tuple[bool, list[str]]:
    """Sanity-check API JSON (landmarks shape). Angles optional for rep counter."""
    issues: list[str] = []
    if not isinstance(data, dict):
        return False, ["response is not a dict"]
    if "inference_ms" not in data:
        issues.append("missing inference_ms")
    persons = [k for k in data if k.startswith("person_")]
    if not persons:
        return len(issues) == 0, issues
    for p in persons:
        obj = data.get(p)
        if not isinstance(obj, dict):
            issues.append(f"{p} is not a dict")
            continue
        if "keypoints" not in obj:
            issues.append(f"{p} missing keypoints")
            continue
        kp = obj["keypoints"]
        if not isinstance(kp, dict):
            issues.append(f"{p}.keypoints is not a dict")
            continue
        # Require only body joints. Face keypoints are optional.
        for name in BODY_KEYPOINT_NAMES:
            if name not in kp:
                issues.append(f"{p}.keypoints missing body joint '{name}'")
                break
            pt = kp[name]
            if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                issues.append(f"{p}.keypoints['{name}'] bad shape")
                break
            if "conf" not in pt and "confidence" not in pt:
                issues.append(f"{p}.keypoints['{name}'] missing conf")
                break
    return len(issues) == 0, issues


def check_vm_health(
    base_url: Optional[str] = None,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 5.0,
) -> tuple[bool, Any]:
    """
    GET /health on direct VM URL. Expects JSON with status ok and model_loaded (yolo-deploy).
    Returns (ok, parsed_json_or_error_str).
    """
    url = (base_url or VM_BASE_URL or "").strip().rstrip("/")
    if not url:
        return False, "no base URL"
    health_url = f"{url}/health"
    sess = session or requests
    try:
        r = sess.get(health_url, timeout=timeout)
        r.raise_for_status()
        body = r.json()
    except requests.RequestException as e:
        logger.warning("VM health request failed: %s", e)
        return False, str(e)
    except ValueError as e:
        logger.warning("VM health JSON invalid: %s", e)
        return False, f"invalid JSON: {e}"
    if body.get("status") != "ok":
        return False, body
    if not body.get("model_loaded"):
        return False, body
    logger.info("VM health OK (model_loaded) at %s", health_url)
    return True, body


def _person_keypoints_to_list(kp_dict: dict) -> Optional[list[dict]]:
    """Convert VM response keypoints dict (name -> {x, y, conf}) to list of 17 in COCO order."""
    if not isinstance(kp_dict, dict):
        return None
    out = []
    for idx, name in enumerate(COCO_KEYPOINT_NAMES):
        pt = kp_dict.get(name)
        if isinstance(pt, dict):
            x = pt.get("x", pt.get("x_center", 0))
            y = pt.get("y", pt.get("y_center", 0))
            c = pt.get("conf", pt.get("confidence", 1.0))
            out.append(_landmark_from_xyc(x, y, c))
        elif idx < FACE_KEYPOINT_COUNT:
            out.append(_empty_landmark())
        else:
            return None
    return out


def _parse_keypoints_vm_schema(data: Any) -> Optional[list[dict]]:
    """Parse production VM schema (person_n -> keypoints map)."""
    if not isinstance(data, dict):
        return None
    person_1 = data.get("person_1")
    if isinstance(person_1, dict):
        parsed = _person_keypoints_to_list(person_1.get("keypoints"))
        if parsed is not None:
            return parsed
    for key, value in data.items():
        if key == "person_1" or not key.startswith("person_") or not isinstance(value, dict):
            continue
        parsed = _person_keypoints_to_list(value.get("keypoints"))
        if parsed is not None:
            return parsed
    return None


def _parse_keypoints_legacy(data: Any) -> Optional[list[dict]]:
    """Optional compatibility parser for legacy payload formats."""
    if data is None:
        return None

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

    if isinstance(data, dict):
        for key in ("results", "predictions"):
            arr = data.get(key)
            if isinstance(arr, list) and len(arr) > 0:
                first = arr[0]
                if isinstance(first, dict):
                    nested = first.get("keypoints", first)
                else:
                    nested = first
                parsed = _parse_keypoints_legacy(nested)
                if parsed is not None:
                    return parsed

        nested = data.get("keypoints")
        if nested is not None:
            if isinstance(nested, dict):
                parsed = _person_keypoints_to_list(nested)
                if parsed is not None:
                    return parsed
            return _parse_keypoints_legacy(nested)

        nested_data = data.get("data")
        if nested_data is not None:
            return _parse_keypoints_legacy(nested_data)

    return None


def send_frame(
    frame_bgr,
    *,
    session: Optional[requests.Session] = None,
    resize_width: int = 0,
    jpeg_quality: int = 85,
    validate: bool = True,
    timeout: Optional[float] = None,
    params: Optional[dict[str, str]] = None,
    parse_rep_counter: bool = False,
) -> VmPredictOutcome:
    """
    Send a BGR frame to the VM /predict endpoint (multipart file upload).
    Reuses TCP via optional requests.Session. Records encode / upload+server / roundtrip ms.
    """
    if frame_bgr is None:
        return VmPredictOutcome(landmarks=None)

    to_send = frame_bgr
    if resize_width > 0 and frame_bgr.shape[1] > resize_width:
        h, w = frame_bgr.shape[:2]
        new_w = resize_width
        new_h = int(h * new_w / w)
        to_send = cv2.resize(to_send, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    try:
        t_encode = time.perf_counter()
        _, buf = cv2.imencode(".jpg", to_send, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not buf.any():
            return VmPredictOutcome(landmarks=None)
        jpeg_bytes = buf.tobytes()
        t_post = time.perf_counter()
    except Exception as e:
        logger.debug("JPEG encode failed: %s", e)
        return VmPredictOutcome(landmarks=None)

    base_url = VM_BASE_URL.rstrip("/")
    predict_url = f"{base_url}/predict"
    sess = session or requests
    tout = float(timeout if timeout is not None else VM_TIMEOUT_SEC)
    try:
        r = sess.post(
            predict_url,
            files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            params=params or None,
            timeout=tout,
        )
    except requests.RequestException as e:
        logger.debug("VM predict request failed: %s", e)
        return VmPredictOutcome(landmarks=None)

    t_done = time.perf_counter()
    encode_ms = (t_post - t_encode) * 1000
    roundtrip_ms = (t_done - t_encode) * 1000
    upload_ms = (t_done - t_post) * 1000
    payload_kb = len(jpeg_bytes) / 1024
    sent_h, sent_w = int(to_send.shape[0]), int(to_send.shape[1])

    benchmark: dict[str, Any] = {
        "encode_ms": encode_ms,
        "roundtrip_ms": roundtrip_ms,
        "upload_ms": upload_ms,
        "payload_kb": payload_kb,
        "response_time": t_done,
        "inference_ms": None,
    }

    if r.status_code != 200:
        logger.debug("VM predict status=%s body=%s", r.status_code, r.text[:200] if r.text else "")
        return VmPredictOutcome(
            landmarks=None,
            benchmark=benchmark,
            sent_hw=(sent_h, sent_w),
            rep_counter=None,
            rep_session_reset=None,
        )

    try:
        body = r.json()
    except Exception as e:
        logger.debug("VM predict JSON decode failed: %s", e)
        return VmPredictOutcome(
            landmarks=None,
            benchmark=benchmark,
            sent_hw=(sent_h, sent_w),
            rep_counter=None,
            rep_session_reset=None,
        )

    if isinstance(body, dict) and "inference_ms" in body:
        try:
            benchmark["inference_ms"] = float(body["inference_ms"])
        except (TypeError, ValueError):
            benchmark["inference_ms"] = body.get("inference_ms")

    rep_counter_dict = None
    rep_session_reset_dict = None
    if parse_rep_counter and isinstance(body, dict):
        rc = body.get("rep_counter")
        if isinstance(rc, dict):
            rep_counter_dict = rc
        rr = body.get("rep_session_reset")
        if isinstance(rr, dict):
            rep_session_reset_dict = rr

    issues: list[str] = []
    if validate:
        _ok, issues = validate_predict_response(body)
        if issues:
            logger.debug("predict validation ok=%s issues=%s", _ok, issues)

    parsed = _parse_keypoints_vm_schema(body)
    if parsed is None and VM_PARSE_LEGACY_FORMATS:
        parsed = _parse_keypoints_legacy(body)

    logger.debug(
        "VM predict ok roundtrip=%.1fms encode=%.1fms upload=%.1fms payload=%.1fKB server_inf=%s",
        roundtrip_ms,
        encode_ms,
        upload_ms,
        payload_kb,
        benchmark.get("inference_ms"),
    )

    if parsed is None:
        logger.debug("VM predict parse failed, keys=%s", list(body.keys()) if isinstance(body, dict) else type(body).__name__)
        return VmPredictOutcome(
            landmarks=None,
            benchmark=benchmark,
            sent_hw=(sent_h, sent_w),
            validation_issues=issues,
            rep_counter=rep_counter_dict,
            rep_session_reset=rep_session_reset_dict,
        )

    return VmPredictOutcome(
        landmarks=parsed,
        benchmark=benchmark,
        sent_hw=(sent_h, sent_w),
        validation_issues=issues,
        rep_counter=rep_counter_dict,
        rep_session_reset=rep_session_reset_dict,
    )


def send_reset(
    *,
    session: Optional[requests.Session] = None,
    user_uid: str,
    timeout: Optional[float] = None,
) -> Optional[dict]:
    """POST /predict?user_uid=<uid>&reset_session=true (sem imagem).

    Retorna o dict {"requested": bool, "had_session": bool, "reason": str} ou None em falha.
    """
    if not user_uid:
        return None
    base_url = VM_BASE_URL.rstrip("/")
    predict_url = f"{base_url}/predict"
    sess = session or requests
    tout = float(timeout if timeout is not None else VM_TIMEOUT_SEC)
    try:
        r = sess.post(
            predict_url,
            params={"user_uid": user_uid, "reset_session": "true"},
            timeout=tout,
        )
    except requests.RequestException as e:
        logger.warning("VM reset request failed: %s", e)
        return None
    if r.status_code != 200:
        logger.warning("VM reset status=%s body=%s", r.status_code, r.text[:200] if r.text else "")
        return None
    try:
        body = r.json()
    except Exception:
        return None
    rr = body.get("rep_session_reset") if isinstance(body, dict) else None
    if isinstance(rr, dict):
        logger.info("VM session reset for uid=%s: %s", user_uid, rr)
        return rr
    return None
