"""Webcam capture loop: calibration buffer, angle selection, peak detector, skeleton overlay."""
from __future__ import annotations

import sys
import cv2
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import Any, Optional

from app.config import (
    ANGLE_SELECTION_MAX_BUFFER_FRAMES,
    ANGLE_SELECTION_MIN_FRAMES,
    ANGLE_SELECTION_MIN_SEC,
    ANGLE_SELECTION_RETRY_INTERVAL_SEC,
    PREDICT_JPEG_QUALITY,
    PREDICT_RESIZE_WIDTH,
    PREDICT_VALIDATE_RESPONSE,
    VM_HEALTH_TIMEOUT_SEC,
    VM_TIMEOUT_SEC,
    get_default_tuning_params,
)
from app.vm_client import agent_debug_log, check_vm_health
from app.debug_console import (
    ensure_console_window,
    get_logger,
    setup_logging,
    update_console_window,
)
from app.math_engine import (
    calculate_from_type,
    calculate_variance,
    create_peak_detector,
    replay_angle_series_on_peak_detector,
    smooth_angle_series,
)
from app.pose_filters import PoseFilterPipeline
from app.skeleton_overlay import draw_skeleton
from app.variance_angle_selector import COMMON_ANGLES, determine_best_angle

logger = get_logger(__name__)


def _peak_detector_from_tuning(tuning_params: dict[str, Any]) -> Any:
    d = get_default_tuning_params()
    tp = tuning_params or {}
    return create_peak_detector(
        smoothing_factor=float(tp.get("smoothingFactor", d["smoothingFactor"])),
        hysteresis=float(tp.get("hysteresis", d["hysteresis"])),
        min_peak_distance=int(tp.get("minPeakDistance", d["minPeakDistance"])),
        peak_margin=float(tp.get("peakMargin", d["peakMargin"])),
        valley_margin=float(tp.get("valleyMargin", d["valleyMargin"])),
        min_range_gate_degrees=float(tp.get("minRangeGate", d["minRangeGate"])),
        range_window_frames=int(tp.get("rangeWindowFrames", d["rangeWindowFrames"])),
        range_min_samples=int(tp.get("rangeMinSamples", d["rangeMinSamples"])),
        delta_deadband_degrees=float(tp.get("angleDeltaDeadband", d["angleDeltaDeadband"])),
        calibration_reps=int(tp.get("calibrationReps", d["calibrationReps"])),
        calibration_certainty=float(tp.get("calibrationCertainty", d["calibrationCertainty"])),
        calibration_force_extra_reps=int(tp.get("calibrationForceExtraReps", d["calibrationForceExtraReps"])),
        min_rep_interval_ms=float(tp.get("minRepIntervalMs", d["minRepIntervalMs"])),
    )


OVERLAY_FONT = cv2.FONT_HERSHEY_DUPLEX  # Bold, readable
OVERLAY_SCALE = 0.7
OVERLAY_THICKNESS = 2
# Black text for readability; use with transparent grey boxes
OVERLAY_COLOR = (0, 0, 0)
OVERLAY_COLOR_DIM = (40, 40, 40)
OVERLAY_COLOR_STATUS = (0, 0, 0)
OVERLAY_OUTLINE = (255, 255, 255)  # White outline for readability

# Semi-transparent grey for UI background boxes (BGR)
OVERLAY_BOX_COLOR = (200, 200, 200)
OVERLAY_BOX_ALPHA = 0.6

# Start button: green when idle, yellow for a few seconds after click
# Min size so button stays clickable when window is small; scale up on large frames
BUTTON_W_MIN = 80
BUTTON_H_MIN = 36
BUTTON_W = 120
BUTTON_H = 44
BUTTON_MARGIN = 12
BUTTON_COLOR = (60, 180, 80)
BUTTON_COLOR_JUST_CLICKED = (0, 255, 255)  # Yellow (BGR) for a few seconds after click
BUTTON_TEXT_COLOR = (0, 0, 0)  # Black, caps, bold
BUTTON_YELLOW_SECONDS = 2.5


def _counterpart_angle_key(angle_key: str) -> Optional[str]:
    """Return mirrored LEFT/RIGHT angle key for the same joint when available."""
    if angle_key.startswith("LEFT_"):
        candidate = "RIGHT_" + angle_key[len("LEFT_") :]
    elif angle_key.startswith("RIGHT_"):
        candidate = "LEFT_" + angle_key[len("RIGHT_") :]
    else:
        return None
    return candidate if candidate in COMMON_ANGLES else None


def _format_angle_label(angle_key: str) -> str:
    return angle_key.replace("_", " ").title()


def _compute_angle_series(frame_buffer: list[list[dict]], angle_key: str) -> list[Optional[float]]:
    cfg = COMMON_ANGLES.get(angle_key)
    if not cfg:
        return []
    return [
        calculate_from_type(cfg["type"], cfg["landmarks"], lm)
        for lm in frame_buffer
    ]


def _series_activity_stats(values: list[Optional[float]]) -> dict[str, float]:
    valid = [float(v) for v in values if v is not None]
    if len(valid) < 15:
        return {"validSamples": float(len(valid)), "variance": 0.0, "range": 0.0}
    smoothed = smooth_angle_series(valid, window=5)
    var = float(calculate_variance(smoothed).get("variance") or 0.0)
    rng = float(max(smoothed) - min(smoothed))
    return {"validSamples": float(len(valid)), "variance": var, "range": rng}


def _select_tracking_mode(
    primary_angle_key: str,
    frame_buffer: list[list[dict]],
    tuning_params: dict[str, Any],
) -> tuple[list[str], str]:
    tracked = [primary_angle_key]
    counterpart = _counterpart_angle_key(primary_angle_key)
    if not counterpart:
        return tracked, "single"

    primary_series = _compute_angle_series(frame_buffer, primary_angle_key)
    counterpart_series = _compute_angle_series(frame_buffer, counterpart)
    primary_stats = _series_activity_stats(primary_series)
    counterpart_stats = _series_activity_stats(counterpart_series)
    min_range_gate = float(tuning_params.get("minRangeGate", 15.0) or 0.0)
    min_rom = max(14.0, min_range_gate * 0.85) if min_range_gate > 0 else 14.0
    counterpart_active = (
        counterpart_stats["validSamples"] >= 15
        and counterpart_stats["variance"] >= 7.0
        and counterpart_stats["range"] >= min_rom
        and (
            primary_stats["range"] <= 0
            or counterpart_stats["range"] >= (primary_stats["range"] * 0.52)
        )
    )
    if counterpart_active:
        tracked.append(counterpart)
        return tracked, "dual"
    return tracked, "single"


def _draw_start_button(frame: Any, run_state: dict[str, Any]) -> None:
    """Draw Start button (green when idle, yellow for a few seconds after click); store rect for click detection.
    Button size scales with frame size so it stays clickable at any window size."""
    h, w = frame.shape[:2]
    run_state["frame_shape"] = (h, w)
    # Scale button with frame so it stays a usable target when window is small
    scale = min(1.0, w / 400, h / 300)
    scale = max(scale, 0.5)
    bw = max(BUTTON_W_MIN, int(BUTTON_W * scale))
    bh = max(BUTTON_H_MIN, int(BUTTON_H * scale))
    margin = max(6, int(BUTTON_MARGIN * scale))
    x = w - bw - margin
    y = margin
    run_state["button_rect"] = (x, y, bw, bh)
    started = run_state.get("started", False)
    started_at = run_state.get("started_at", 0.0)
    just_clicked = started and (time.time() - started_at) < BUTTON_YELLOW_SECONDS
    color = BUTTON_COLOR_JUST_CLICKED if just_clicked else BUTTON_COLOR
    border = (0, 200, 255) if just_clicked else (80, 200, 100)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, -1)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), border, 2)
    label = "START"
    font_scale = max(0.4, 0.6 * scale)
    (tw, th), _ = cv2.getTextSize(label, OVERLAY_FONT, font_scale, 2)
    tx = x + (bw - tw) // 2
    ty = y + (bh + th) // 2
    cv2.putText(frame, label, (tx, ty), OVERLAY_FONT, font_scale, BUTTON_TEXT_COLOR, 2)


def _get_display_scale_and_offset(state: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Get scale and offset for frame display in window. Returns (scale, offset_x, offset_y, win_w) or None.
    Mouse callback (x,y) are window-relative (client area), so we do not use window position on screen."""
    frame_shape = state.get("frame_shape", (0, 0))
    if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
        return None
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    try:
        _wx, _wy, win_w, win_h = cv2.getWindowImageRect("Rep Counter")
    except Exception:
        return None
    if win_w <= 0 or win_h <= 0:
        return None
    scale = min(win_w / frame_w, win_h / frame_h)
    if scale <= 0:
        return None
    offset_x = (win_w - frame_w * scale) / 2
    offset_y = (win_h - frame_h * scale) / 2
    return (scale, offset_x, offset_y, win_w)


def _window_to_image_coords(state: dict[str, Any], win_x: int, win_y: int) -> tuple[int, int]:
    """Convert window-relative mouse coordinates to image coordinates (handles any window size)."""
    display = _get_display_scale_and_offset(state)
    if display is None:
        return (win_x, win_y)
    scale, offset_x, offset_y, _ = display
    frame_shape = state.get("frame_shape", (0, 0))
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    # (win_x, win_y) are relative to window client area; do not subtract window position
    img_x = int((win_x - offset_x) / scale)
    img_y = int((win_y - offset_y) / scale)
    return (max(0, min(frame_w - 1, img_x)), max(0, min(frame_h - 1, img_y)))


def _on_mouse(event: int, x: int, y: int, _flags: int, param: dict[str, Any]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    state = param
    rect = state.get("button_rect", (0, 0, 0, 0))
    rx, ry, rw, rh = rect
    in_rect = False
    # 1) Hit-test in image coordinates (window-relative conversion)
    img_x, img_y = _window_to_image_coords(state, x, y)
    in_rect = rx <= img_x <= rx + rw and ry <= img_y <= ry + rh
    if not in_rect:
        # 2) Hit-test in window coordinates: map button rect to display rect
        display = _get_display_scale_and_offset(state)
        if display is not None:
            scale, offset_x, offset_y, _ = display
            bx1 = offset_x + rx * scale
            by1 = offset_y + ry * scale
            bx2 = offset_x + (rx + rw) * scale
            by2 = offset_y + (ry + rh) * scale
            in_rect = bx1 <= x <= bx2 and by1 <= y <= by2
    if not in_rect:
        return
    if not state.get("started", False):
        state["started"] = True
        state["started_at"] = time.time()
        state["selection_last_attempt"] = None
    else:
        state["selected_angle"] = None
        state["selected_config"] = None
        state["peak_detector"] = None
        state["tracked_angles"] = []
        state["detectors_by_angle"] = {}
        state["rep_counts_by_angle"] = {}
        state["tracking_mode"] = "single"
        state["frame_buffer"] = deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES)
        state["selection_last_attempt"] = None


def _draw_transparent_box(frame: Any, x1: int, y1: int, x2: int, y2: int) -> None:
    """Draw a semi-transparent grey box for UI readability."""
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), OVERLAY_BOX_COLOR, -1)
    cv2.addWeighted(overlay, OVERLAY_BOX_ALPHA, roi, 1 - OVERLAY_BOX_ALPHA, 0, roi)


def _put_text_readable(frame: Any, text: str, pos: tuple[int, int], font: int, scale: float, color: tuple, thickness: int) -> None:
    """Draw text in CAPS with outline for readability on grey box background."""
    x, y = pos
    text = text.upper()
    cv2.putText(frame, text, (x, y), font, scale, OVERLAY_OUTLINE, thickness + 1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def _merge_benchmark_peaks(peaks: dict[str, float | None], b: dict[str, Any]) -> None:
    """Session maxima for timing fields (yolo-deploy camera_pose_client style)."""
    for key in ("roundtrip_ms", "upload_ms", "encode_ms"):
        v = b.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        prev = peaks.get(key)
        peaks[key] = fv if prev is None else max(prev, fv)
    inf = b.get("inference_ms")
    if isinstance(inf, (int, float)):
        fi = float(inf)
        prev_i = peaks.get("inference_ms")
        peaks["inference_ms"] = fi if prev_i is None else max(prev_i, fi)


def _scale_landmarks_to_display(
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


def _draw_vm_benchmark_hud(
    frame: Any,
    benchmark: Optional[dict[str, Any]],
    peaks: dict[str, float | None],
    cam_fps: float,
    inf_fps: float,
    validation_issues: Optional[list[str]],
) -> None:
    """Bottom-right overlay: roundtrip, upload, encode, server inference, payload, FPS peaks."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if benchmark:
        rt = float(benchmark.get("roundtrip_ms") or 0)
        max_inf = (1000.0 / rt) if rt > 0 else 0.0
        srv = benchmark.get("inference_ms")
        line1 = f"rt {rt:.0f}ms  up {float(benchmark.get('upload_ms') or 0):.0f}ms  enc {float(benchmark.get('encode_ms') or 0):.0f}ms"
        lines.append((line1, (220, 220, 220)))
        if srv is not None:
            try:
                sm = float(srv)
                lines.append((f"server {sm:.0f}ms  ~max {max_inf:.1f} inf/s", (180, 255, 180)))
            except (TypeError, ValueError):
                lines.append((f"server {srv!s}", (180, 255, 180)))
        lines.append((f"payload {float(benchmark.get('payload_kb') or 0):.1f} KB", (200, 200, 200)))
    lines.append((f"cam {cam_fps:.0f} fps   infer {inf_fps:.1f} /s", (200, 200, 255)))
    prt, pinf = peaks.get("roundtrip_ms"), peaks.get("inference_ms")
    peak_bits = []
    if prt is not None:
        peak_bits.append(f"rt_peak {prt:.0f}")
    if pinf is not None:
        peak_bits.append(f"srv_peak {pinf:.0f}")
    if peak_bits:
        lines.append(("  ".join(peak_bits), (180, 180, 255)))
    if validation_issues:
        msg = "API: " + "; ".join(validation_issues[:3])
        if len(validation_issues) > 3:
            msg += "…"
        lines.append((msg, (60, 60, 255)))

    line_h = 18
    max_tw = 0
    for txt, _c in lines:
        tw, _th = cv2.getTextSize(txt, font, scale, thick)[0]
        max_tw = max(max_tw, tw)
    margin = 8
    box_w = min(w - 2 * margin, max_tw + 2 * margin)
    box_h = len(lines) * line_h + margin
    x1 = w - box_w - margin
    y1 = h - box_h - margin
    x2, y2 = w - margin, h - margin
    _draw_transparent_box(frame, x1, y1, x2, y2)
    y = y1 + line_h
    for txt, col in lines:
        cv2.putText(frame, txt, (x1 + 6, y), font, scale, (40, 40, 40), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, txt, (x1 + 6, y), font, scale, col, thick, cv2.LINE_AA)
        y += line_h


def _pose_worker(
    frame_queue: Queue,
    result_holder: list,
    stop_event: threading.Event,
    pose_options: dict[str, Any],
) -> None:
    """Background thread: send frames to VM; store latest landmarks + benchmark per /predict."""
    import requests

    from app.vm_client import send_frame

    session = requests.Session()
    resize_width = int(pose_options.get("resize_width", 0))
    jpeg_quality = int(pose_options.get("jpeg_quality", 85))
    validate = bool(pose_options.get("validate", True))
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        if frame is None:
            break
        outcome = send_frame(
            frame,
            session=session,
            resize_width=resize_width,
            jpeg_quality=jpeg_quality,
            validate=validate,
        )
        result_holder[0] = {
            "landmarks": outcome.landmarks,
            "benchmark": outcome.benchmark,
            "sent_hw": outcome.sent_hw,
            "validation_issues": outcome.validation_issues,
        }
        # #region agent log
        try:
            agent_debug_log(
                "H2",
                "loop.py:_pose_worker",
                "worker finished send_frame",
                {
                    "landmarks_present": outcome.landmarks is not None,
                    "worker_ts_ms": int(time.time() * 1000),
                },
            )
        except Exception:
            pass
        # #endregion


def _draw_overlay(
    frame: Any,
    selected_angle: Optional[str],
    smoothed_value: Optional[float],
    rep_count: int,
    state: str,
    status: str,
    rep_counts_by_angle: Optional[dict[str, int]] = None,
    tracking_mode: str = "single",
) -> None:
    margin = 10
    y = 30
    line_height = 30
    box_x1 = margin
    box_y1 = 12
    # Count lines to size the grey box
    lines = 3  # Reps, State, status
    if selected_angle:
        lines += 1
    if smoothed_value is not None:
        lines += 1
    if rep_counts_by_angle:
        lines += len(rep_counts_by_angle)
    box_x2 = min(frame.shape[1] - margin, 560)
    box_y2 = box_y1 + lines * line_height + 8
    _draw_transparent_box(frame, box_x1, box_y1, box_x2, box_y2)
    if selected_angle:
        _put_text_readable(
            frame, f"Angle: {selected_angle}", (margin, y), OVERLAY_FONT, OVERLAY_SCALE, OVERLAY_COLOR, OVERLAY_THICKNESS
        )
        y += line_height
    if smoothed_value is not None:
        _put_text_readable(
            frame, f"Value: {smoothed_value:.1f}", (margin, y), OVERLAY_FONT, OVERLAY_SCALE, OVERLAY_COLOR_DIM, OVERLAY_THICKNESS
        )
        y += line_height
    rep_label = "Matched Reps" if tracking_mode == "dual" else "Reps"
    _put_text_readable(
        frame,
        f"{rep_label}: {rep_count}",
        (margin, y),
        OVERLAY_FONT,
        OVERLAY_SCALE,
        OVERLAY_COLOR,
        OVERLAY_THICKNESS,
    )
    y += line_height
    if rep_counts_by_angle:
        for angle_key, count in rep_counts_by_angle.items():
            _put_text_readable(
                frame,
                f"{_format_angle_label(angle_key)}: {count}",
                (margin, y),
                OVERLAY_FONT,
                0.6,
                OVERLAY_COLOR_DIM,
                2,
            )
            y += line_height
    _put_text_readable(frame, f"State: {state}", (margin, y), OVERLAY_FONT, 0.6, OVERLAY_COLOR_DIM, 2)
    y += line_height
    _put_text_readable(frame, status, (margin, y), OVERLAY_FONT, 0.6, OVERLAY_COLOR_STATUS, 2)


def run_webcam_loop(
    *,
    skip_health_check: bool = False,
    benchmark_log_path: Optional[str] = None,
    resize_width: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    validate_response: Optional[bool] = None,
) -> None:
    setup_logging()

    rw = PREDICT_RESIZE_WIDTH if resize_width is None else int(resize_width)
    jq = PREDICT_JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality)
    val = PREDICT_VALIDATE_RESPONSE if validate_response is None else bool(validate_response)

    if not skip_health_check:
        ok, info = check_vm_health(timeout=VM_HEALTH_TIMEOUT_SEC)
        if not ok:
            raise RuntimeError(f"VM health check failed: {info}")

    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    run_state: dict[str, Any] = {
        "started": False,
        "frame_buffer": deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES),
        "selection_last_attempt": None,
        "selected_angle": None,
        "selected_config": None,
        "peak_detector": None,
        "tracked_angles": [],
        "detectors_by_angle": {},
        "rep_counts_by_angle": {},
        "tracking_mode": "single",
        "tuning_params": get_default_tuning_params(),
        "button_rect": (0, 0, BUTTON_W_MIN, BUTTON_H_MIN),
        "frame_shape": (0, 0),
    }
    cv2.namedWindow("Rep Counter", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Rep Counter", _on_mouse, run_state)
    ensure_console_window()
    pose_pipeline = PoseFilterPipeline(use_one_euro=True)
    logger.debug(
        "webcam opened, VM predict resize_width=%s jpeg_q=%s validate=%s",
        rw,
        jq,
        val,
    )
    if benchmark_log_path:
        logger.info("Benchmark log: appending inference/roundtrip to %s", benchmark_log_path)

    pose_options: dict[str, Any] = {
        "resize_width": rw,
        "jpeg_quality": jq,
        "validate": val,
    }
    frame_queue: Queue = Queue(maxsize=1)
    latest_pose: list = [
        {
            "landmarks": None,
            "benchmark": None,
            "sent_hw": None,
            "validation_issues": [],
        }
    ]
    stop_worker = threading.Event()
    worker = threading.Thread(
        target=_pose_worker,
        args=(frame_queue, latest_pose, stop_worker, pose_options),
        daemon=True,
    )
    worker.start()

    _dbg_overlay_samples = 0
    fps_times: list[float] = []
    fps_window = 30
    inference_response_times: list[float] = []
    inference_fps_window = 20
    last_benchmark: Optional[dict[str, Any]] = None
    prev_resp_t: Any = None
    benchmark_peaks: dict[str, float | None] = {
        "roundtrip_ms": None,
        "upload_ms": None,
        "encode_ms": None,
        "inference_ms": None,
    }

    def _update_vm_metrics(snap: dict[str, Any]) -> tuple[dict[str, Any] | None, float, list[str]]:
        """Merge timing peaks, optional benchmark log line; return (display_benchmark, inf_fps, issues)."""
        nonlocal last_benchmark, prev_resp_t
        b = snap.get("benchmark")
        issues = list(snap.get("validation_issues") or [])
        if b:
            last_benchmark = b
            _merge_benchmark_peaks(benchmark_peaks, b)
            rt_key = b.get("response_time")
            if rt_key is not None and rt_key != prev_resp_t:
                prev_resp_t = rt_key
                inference_response_times.append(float(rt_key))
                if len(inference_response_times) > inference_fps_window:
                    inference_response_times.pop(0)
                if benchmark_log_path and b.get("inference_ms") is not None:
                    try:
                        with open(benchmark_log_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"{time.time():.3f}\t{float(b['inference_ms']):.2f}\t{float(b['roundtrip_ms']):.2f}\t{float(b['upload_ms']):.2f}\n"
                            )
                    except OSError as e:
                        logger.debug("benchmark log write failed: %s", e)
        inf_fps = 0.0
        if len(inference_response_times) > 1:
            dt = inference_response_times[-1] - inference_response_times[0]
            if dt > 0:
                inf_fps = (len(inference_response_times) - 1) / dt
        return (b or last_benchmark), inf_fps, issues

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            t_now = time.perf_counter()
            fps_times.append(t_now)
            if len(fps_times) > fps_window:
                fps_times.pop(0)
            cam_fps = (
                (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                if len(fps_times) > 1 and fps_times[-1] > fps_times[0]
                else 0.0
            )

            _draw_start_button(frame_bgr, run_state)

            if not run_state["started"]:
                cx, cy = 10, frame_bgr.shape[0] // 2
                _draw_transparent_box(frame_bgr, cx - 4, cy - 28, cx + 280, cy + 12)
                _put_text_readable(
                    frame_bgr, "Click Start to begin",
                    (cx, cy),
                    OVERLAY_FONT, 0.7, OVERLAY_COLOR, 2,
                )
                cv2.imshow("Rep Counter", frame_bgr)
                update_console_window()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            try:
                frame_queue.put_nowait(frame_bgr.copy())
            except Full:
                pass
            snap = latest_pose[0]
            disp_b, inf_fps, val_issues = _update_vm_metrics(snap)
            raw_landmarks = snap.get("landmarks")
            sent_hw = snap.get("sent_hw")
            logger.debug("frame sent, got landmarks=%s", raw_landmarks is not None)

            if raw_landmarks is not None:
                _dbg_overlay_samples += 1
                if _dbg_overlay_samples <= 25:
                    # #region agent log
                    try:
                        agent_debug_log(
                            "H2",
                            "loop.py:run_webcam_loop",
                            "main thread read landmarks for overlay",
                            {
                                "sample_i": _dbg_overlay_samples,
                                "main_ts_ms": int(time.time() * 1000),
                                "frame_hw": [int(frame_bgr.shape[0]), int(frame_bgr.shape[1])],
                            },
                        )
                    except Exception:
                        pass
                    # #endregion

            if raw_landmarks is None:
                status = "No pose"
                _draw_overlay(
                    frame_bgr, run_state["selected_angle"], None, 0, "—", status
                )
                _draw_vm_benchmark_hud(
                    frame_bgr, disp_b, benchmark_peaks, cam_fps, inf_fps, val_issues
                )
                cv2.imshow("Rep Counter", frame_bgr)
                update_console_window()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            disp_h, disp_w = frame_bgr.shape[0], frame_bgr.shape[1]
            raw_scaled = _scale_landmarks_to_display(
                raw_landmarks,
                sent_hw if isinstance(sent_hw, tuple) else None,
                (disp_h, disp_w),
            )
            timestamp_ms = time.time() * 1000.0
            landmarks = pose_pipeline.process(raw_scaled, timestamp_ms)
            draw_skeleton(frame_bgr, landmarks)

            frame_buffer = run_state["frame_buffer"]
            selected_angle = run_state["selected_angle"]
            selected_config = run_state["selected_config"]
            peak_detector = run_state["peak_detector"]
            tracked_angles: list[str] = list(run_state.get("tracked_angles") or [])
            detectors_by_angle: dict[str, Any] = dict(run_state.get("detectors_by_angle") or {})
            rep_counts_by_angle: dict[str, int] = dict(run_state.get("rep_counts_by_angle") or {})
            tracking_mode: str = str(run_state.get("tracking_mode") or "single")
            tuning_params = run_state["tuning_params"]

            if selected_angle is None:
                frame_buffer.append(landmarks)
                started_at = float(run_state.get("started_at") or 0.0)
                elapsed = time.time() - started_at
                ready = (
                    len(frame_buffer) >= ANGLE_SELECTION_MIN_FRAMES
                    and elapsed >= ANGLE_SELECTION_MIN_SEC
                )
                last_att = run_state.get("selection_last_attempt")
                can_try = ready and (
                    last_att is None
                    or (time.time() - float(last_att)) >= ANGLE_SELECTION_RETRY_INTERVAL_SEC
                )
                if can_try:
                    result = determine_best_angle(frame_buffer, exercise=None)
                    selected_angle = result.get("selectedAngle")
                    tuning_params = result.get("tuningParams") or get_default_tuning_params()
                    run_state["tuning_params"] = tuning_params
                    if selected_angle and selected_angle in COMMON_ANGLES:
                        run_state["selected_angle"] = selected_angle
                        run_state["selected_config"] = COMMON_ANGLES[selected_angle]
                        run_state["selection_last_attempt"] = None
                        tracked, tracking_mode = _select_tracking_mode(
                            selected_angle,
                            frame_buffer,
                            tuning_params,
                        )
                        run_state["tracked_angles"] = tracked
                        run_state["tracking_mode"] = tracking_mode
                        detectors_by_angle = {
                            angle_key: _peak_detector_from_tuning(tuning_params)
                            for angle_key in tracked
                        }
                        # Retroactive counting: replay calibration buffer through each PeakDetector
                        # so reps already completed in the buffer are counted and peaks/valleys injected.
                        cal_landmarks = list(frame_buffer)
                        rep_counts_by_angle: dict[str, int] = {}
                        for angle_key in tracked:
                            cfg = COMMON_ANGLES.get(angle_key)
                            det = detectors_by_angle.get(angle_key)
                            if not cfg or det is None:
                                continue
                            series = [
                                calculate_from_type(cfg["type"], cfg["landmarks"], lm)
                                for lm in cal_landmarks
                            ]
                            replay_angle_series_on_peak_detector(det, series)
                            rep_counts_by_angle[angle_key] = det.get_rep_count()
                        run_state["detectors_by_angle"] = detectors_by_angle
                        run_state["rep_counts_by_angle"] = rep_counts_by_angle
                        run_state["peak_detector"] = detectors_by_angle[selected_angle]
                        frame_buffer.clear()
                    else:
                        run_state["selected_angle"] = None
                        run_state["selected_config"] = None
                        run_state["tracked_angles"] = []
                        run_state["detectors_by_angle"] = {}
                        run_state["rep_counts_by_angle"] = {}
                        run_state["tracking_mode"] = "single"
                        run_state["selection_last_attempt"] = time.time()
                retry_at = run_state.get("selection_last_attempt")
                if not ready:
                    status = (
                        f"Observing movement... {len(frame_buffer)}/{ANGLE_SELECTION_MIN_FRAMES} frames, "
                        f"{elapsed:.1f}s / {ANGLE_SELECTION_MIN_SEC:.0f}s min"
                    )
                elif retry_at is not None and (
                    time.time() - float(retry_at) < ANGLE_SELECTION_RETRY_INTERVAL_SEC
                ):
                    rem = ANGLE_SELECTION_RETRY_INTERVAL_SEC - (time.time() - float(retry_at))
                    status = f"Unclear motion — retry in {rem:.0f}s (full ROM, steady reps)"
                elif selected_angle:
                    status = "Locked joint — calibrating reps"
                else:
                    status = f"Analyzing... ({len(frame_buffer)} frames)"
                _draw_overlay(frame_bgr, None, None, 0, "—", status)
                _draw_vm_benchmark_hud(
                    frame_bgr, disp_b, benchmark_peaks, cam_fps, inf_fps, val_issues
                )
                cv2.imshow("Rep Counter", frame_bgr)
                update_console_window()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            angle_value = None
            if selected_config:
                angle_value = calculate_from_type(
                    selected_config["type"],
                    selected_config["landmarks"],
                    landmarks,
                )
            rep_count = 0
            state_str = "—"
            smoothed_value = None
            range_gate_open = True
            rolling_range: Optional[float] = None
            d_tuning = get_default_tuning_params()
            calibration_complete = True
            cal_target = int(tuning_params.get("calibrationReps", d_tuning["calibrationReps"]))
            cal_certainty_target = float(
                tuning_params.get("calibrationCertainty", d_tuning["calibrationCertainty"])
            )
            cal_certainty = 0.0
            primary_rep_count = 0
            if detectors_by_angle:
                primary_angle = selected_angle
                for angle_key in tracked_angles:
                    detector = detectors_by_angle.get(angle_key)
                    config = COMMON_ANGLES.get(angle_key)
                    if detector is None or not config:
                        continue
                    value = calculate_from_type(
                        config["type"],
                        config["landmarks"],
                        landmarks,
                    )
                    out = detector.update(value)
                    reps_for_angle = int(out.get("repCount", 0) or 0)
                    rep_counts_by_angle[angle_key] = reps_for_angle
                    if primary_angle == angle_key:
                        state_str = out.get("state", "—")
                        smoothed_value = out.get("smoothedValue")
                        range_gate_open = bool(out.get("rangeGateOpen", True))
                        r = out.get("rollingRange")
                        rolling_range = float(r) if r is not None else None
                        calibration_complete = bool(out.get("calibrationComplete", False))
                        cal_target = int(out.get("calibrationTargetReps", cal_target))
                        cal_certainty = float(out.get("calibrationCertainty", 0.0) or 0.0)
                        cal_certainty_target = float(
                            out.get("calibrationCertaintyTarget", cal_certainty_target)
                        )
                        primary_rep_count = reps_for_angle
                counts = [int(rep_counts_by_angle.get(k, 0) or 0) for k in tracked_angles]
                if tracking_mode == "dual" and len(counts) >= 2:
                    rep_count = min(counts)
                else:
                    rep_count = primary_rep_count
                run_state["detectors_by_angle"] = detectors_by_angle
                run_state["rep_counts_by_angle"] = rep_counts_by_angle
            elif peak_detector is not None:
                out = peak_detector.update(angle_value)
                rep_count = int(out.get("repCount", 0) or 0)
                primary_rep_count = rep_count
                state_str = out.get("state", "—")
                smoothed_value = out.get("smoothedValue")
                range_gate_open = bool(out.get("rangeGateOpen", True))
                r = out.get("rollingRange")
                rolling_range = float(r) if r is not None else None
                calibration_complete = bool(out.get("calibrationComplete", False))
                cal_target = int(out.get("calibrationTargetReps", cal_target))
                cal_certainty = float(out.get("calibrationCertainty", 0.0) or 0.0)
                cal_certainty_target = float(
                    out.get("calibrationCertaintyTarget", cal_certainty_target)
                )

            shown_rep_count = rep_count if calibration_complete else 0
            if not calibration_complete:
                status = (
                    f"Calibrating... reps {primary_rep_count}/{cal_target}  "
                    f"certainty {cal_certainty * 100:.0f}%/{cal_certainty_target * 100:.0f}%"
                )
            elif tracked_angles and tracking_mode == "dual":
                status = "Tracking dual limbs (counting only matched reps)"
            else:
                status = "Tracking single limb (locked to calibrated side)"
            if tracking_mode == "dual" and len(tracked_angles) >= 2:
                ordered = sorted(
                    [
                        (k, int(rep_counts_by_angle.get(k, 0) or 0))
                        for k in tracked_angles
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                if ordered and ordered[0][1] != ordered[-1][1]:
                    lead_key, lead_reps = ordered[0]
                    lag_key, lag_reps = ordered[-1]
                    status = (
                        f"{status} - WARNING: {_format_angle_label(lag_key)} "
                        f"is behind ({lag_reps} vs {lead_reps}); no extra matched rep counted"
                    )
            # OpenCV fonts often mangle Unicode (e.g. >= and degree); use plain ASCII.
            if rolling_range is not None and not range_gate_open:
                need = float(tuning_params.get("minRangeGate", get_default_tuning_params()["minRangeGate"]))
                if need > 0:
                    if not calibration_complete:
                        status = (
                            f"{status} — Move more: ~{rolling_range:.0f} deg so far "
                            f"(need at least {need:.0f} deg for a rep)"
                        )
                    else:
                        status = (
                            f"{status} — Range ~{rolling_range:.0f} deg, "
                            f"need at least {need:.0f} deg"
                        )
            _draw_overlay(
                frame_bgr,
                selected_angle,
                smoothed_value,
                shown_rep_count,
                state_str,
                status,
                {
                    k: int(rep_counts_by_angle.get(k, 0) or 0)
                    for k in tracked_angles
                } if tracked_angles else None,
                tracking_mode,
            )
            _draw_vm_benchmark_hud(
                frame_bgr, disp_b, benchmark_peaks, cam_fps, inf_fps, val_issues
            )
            logger.debug("angle=%s smoothed=%.1f reps=%d state=%s", selected_angle, smoothed_value or 0, rep_count, state_str)
            cv2.imshow("Rep Counter", frame_bgr)
            update_console_window()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        stop_worker.set()
        try:
            frame_queue.put_nowait(None)
        except Full:
            pass
        worker.join(timeout=VM_TIMEOUT_SEC + 1.0)
        cap.release()
        cv2.destroyAllWindows()
        if last_benchmark:
            print("\nLast VM request (benchmark):")
            print(
                f"  roundtrip: {last_benchmark.get('roundtrip_ms', 0):.0f} ms  "
                f"upload: {last_benchmark.get('upload_ms', 0):.0f} ms  "
                f"encode: {last_benchmark.get('encode_ms', 0):.0f} ms"
            )
            if last_benchmark.get("inference_ms") is not None:
                print(f"  server inference: {last_benchmark['inference_ms']}")
            print(f"  payload: {last_benchmark.get('payload_kb', 0):.1f} KB")
        if any(v is not None for v in benchmark_peaks.values()):
            print("\nSession VM peaks (ms):")
            for k in ("roundtrip_ms", "upload_ms", "encode_ms", "inference_ms"):
                v = benchmark_peaks.get(k)
                if v is not None:
                    print(f"  {k}: {v:.1f}")
