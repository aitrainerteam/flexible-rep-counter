"""Webcam capture loop: calibration buffer, angle selection, peak detector, skeleton overlay."""
from __future__ import annotations

import cv2
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Optional

from app.config import VM_POSE_MAX_AGE_SEC, VM_TIMEOUT_SEC, get_default_tuning_params
from app.vm_client import agent_debug_log
from app.debug_console import (
    ensure_console_window,
    get_logger,
    setup_logging,
    update_console_window,
)
from app.math_engine import PeakDetector, calculate_from_type, create_peak_detector
from app.pose_filters import PoseFilterPipeline
from app.skeleton_overlay import draw_skeleton
from app.variance_angle_selector import COMMON_ANGLES, determine_best_angle

logger = get_logger(__name__)

CALIBRATION_FRAMES = 60
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
    else:
        state["selected_angle"] = None
        state["selected_config"] = None
        state["peak_detector"] = None
        state["frame_buffer"] = []


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


def _pose_worker(
    frame_queue: Queue,
    result_holder: list,
    stop_event: threading.Event,
) -> None:
    """Background thread: send frames to VM; store {landmarks, received_mono} per completed /predict."""
    from app.vm_client import send_frame
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        if frame is None:
            break
        parsed = send_frame(frame)
        result_holder[0] = {
            "landmarks": parsed,
            "received_mono": time.monotonic(),
        }
        # #region agent log
        try:
            agent_debug_log(
                "H2",
                "loop.py:_pose_worker",
                "worker finished send_frame",
                {
                    "landmarks_present": parsed is not None,
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
    box_x2 = min(frame.shape[1] - margin, 320)
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
    _put_text_readable(frame, f"Reps: {rep_count}", (margin, y), OVERLAY_FONT, OVERLAY_SCALE, OVERLAY_COLOR, OVERLAY_THICKNESS)
    y += line_height
    _put_text_readable(frame, f"State: {state}", (margin, y), OVERLAY_FONT, 0.6, OVERLAY_COLOR_DIM, 2)
    y += line_height
    _put_text_readable(frame, status, (margin, y), OVERLAY_FONT, 0.6, OVERLAY_COLOR_STATUS, 2)


def run_webcam_loop() -> None:
    setup_logging()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    run_state: dict[str, Any] = {
        "started": False,
        "frame_buffer": [],
        "selected_angle": None,
        "selected_config": None,
        "peak_detector": None,
        "tuning_params": get_default_tuning_params(),
        "button_rect": (0, 0, BUTTON_W_MIN, BUTTON_H_MIN),
        "frame_shape": (0, 0),
    }
    cv2.namedWindow("Rep Counter", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Rep Counter", _on_mouse, run_state)
    ensure_console_window()
    pose_pipeline = PoseFilterPipeline(use_one_euro=True)
    logger.debug("webcam opened, loop started")

    # Background thread sends frames to VM so main loop stays responsive
    frame_queue: Queue = Queue(maxsize=1)
    latest_pose: list = [{"landmarks": None, "received_mono": 0.0}]
    stop_worker = threading.Event()
    worker = threading.Thread(
        target=_pose_worker,
        args=(frame_queue, latest_pose, stop_worker),
        daemon=True,
    )
    worker.start()

    _dbg_overlay_samples = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

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

            # Non-blocking: feed latest frame to worker, use last received landmarks
            try:
                frame_queue.put_nowait(frame_bgr.copy())
            except Full:
                pass
            snap = latest_pose[0]
            raw_landmarks = snap.get("landmarks")
            received_mono = float(snap.get("received_mono", 0.0))
            pose_fresh = (
                raw_landmarks is not None
                and (time.monotonic() - received_mono) <= VM_POSE_MAX_AGE_SEC
            )
            logger.debug(
                "frame sent, got landmarks=%s fresh_cloud=%s",
                raw_landmarks is not None,
                pose_fresh,
            )

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
                cv2.imshow("Rep Counter", frame_bgr)
                update_console_window()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            timestamp_ms = time.time() * 1000.0
            landmarks = pose_pipeline.process(raw_landmarks, timestamp_ms)
            if pose_fresh:
                draw_skeleton(frame_bgr, landmarks)

            frame_buffer = run_state["frame_buffer"]
            selected_angle = run_state["selected_angle"]
            selected_config = run_state["selected_config"]
            peak_detector = run_state["peak_detector"]
            tuning_params = run_state["tuning_params"]

            if selected_angle is None:
                frame_buffer.append(landmarks)
                if len(frame_buffer) >= CALIBRATION_FRAMES:
                    result = determine_best_angle(frame_buffer, exercise=None)
                    selected_angle = result.get("selectedAngle")
                    tuning_params = result.get("tuningParams") or get_default_tuning_params()
                    run_state["tuning_params"] = tuning_params
                    if selected_angle and selected_angle in COMMON_ANGLES:
                        run_state["selected_angle"] = selected_angle
                        run_state["selected_config"] = COMMON_ANGLES[selected_angle]
                        run_state["peak_detector"] = create_peak_detector(
                            smoothing_factor=float(tuning_params.get("smoothingFactor", 0.45)),
                            hysteresis=float(tuning_params.get("hysteresis", 5)),
                            min_peak_distance=int(tuning_params.get("minPeakDistance", 5)),
                            peak_margin=float(tuning_params.get("peakMargin", 15)),
                            valley_margin=float(tuning_params.get("valleyMargin", 15)),
                        )
                        frame_buffer.clear()
                    else:
                        run_state["selected_angle"] = None
                        run_state["selected_config"] = None
                        frame_buffer.clear()
                status = f"Calibrating... {len(frame_buffer)}/{CALIBRATION_FRAMES}"
                if not pose_fresh:
                    status = f"{status} | waiting for /predict"
                _draw_overlay(frame_bgr, None, None, 0, "—", status)
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
            if peak_detector is not None:
                out = peak_detector.update(angle_value)
                rep_count = out.get("repCount", 0)
                state_str = out.get("state", "—")
                smoothed_value = out.get("smoothedValue")

            status = "Tracking"
            if not pose_fresh:
                status = f"{status} | waiting for /predict"
            _draw_overlay(
                frame_bgr, selected_angle, smoothed_value, rep_count, state_str, status
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
