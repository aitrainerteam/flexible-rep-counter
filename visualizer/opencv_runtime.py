"""OpenCV webcam frontend (repo root, not part of the installable package).

Uses :mod:`flexible_rep_counter` for :class:`~flexible_rep_counter.session.RepCounterSession`
and landmark helpers; VM I/O stays under ``app``.
"""
from __future__ import annotations

import sys
from collections import deque
import cv2
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Optional

from flexible_rep_counter.landmark_utils import scale_landmarks_to_display
from flexible_rep_counter.session import RepCounterSession
from flexible_rep_counter.types import StepResult

from app.config import (
    PREDICT_JPEG_QUALITY,
    PREDICT_RESIZE_WIDTH,
    PREDICT_VALIDATE_RESPONSE,
    VM_HEALTH_TIMEOUT_SEC,
    VM_TIMEOUT_SEC,
)
from app.vm_client import check_vm_health
from app.debug_console import (
    ensure_console_window,
    get_logger,
    setup_logging,
    update_console_window,
)
from app.skeleton_overlay import draw_skeleton

logger = get_logger(__name__)

# #region agent log
_AGENT_DEBUG_LOG = "/Users/aa/Desktop/flexible-rep-counter/.cursor/debug-0f69d6.log"


def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    import json

    payload = {
        "sessionId": "0f69d6",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as _df:
            _df.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# #endregion

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
BUTTON_W_MIN = 120
BUTTON_H_MIN = 48
BUTTON_W = 168
BUTTON_H = 56
BUTTON_MARGIN = 14
# Extra hit slop in image pixels (HighGUI backends differ on mouse coordinate space)
START_HIT_PAD = 18
BUTTON_COLOR = (60, 180, 80)
BUTTON_COLOR_JUST_CLICKED = (0, 255, 255)  # Yellow (BGR) for a few seconds after click
BUTTON_TEXT_COLOR = (0, 0, 0)  # Black, caps, bold
BUTTON_YELLOW_SECONDS = 2.5


def _ascii_text(text: str) -> str:
    """OpenCV Hershey fonts are ASCII-only; replace common Unicode punctuation."""
    return (
        str(text)
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
        .replace("°", " deg")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .encode("ascii", "replace")
        .decode("ascii")
    )


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


def _hit_in_rect(x: int, y: int, rx: int, ry: int, rw: int, rh: int) -> bool:
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _padded_button_rect(
    rx: int, ry: int, rw: int, rh: int, frame_h: int, frame_w: int
) -> tuple[int, int, int, int]:
    pad = START_HIT_PAD
    x1 = max(0, rx - pad)
    y1 = max(0, ry - pad)
    x2 = min(frame_w - 1, rx + rw + pad)
    y2 = min(frame_h - 1, ry + rh + pad)
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def _start_button_hit(state: dict[str, Any], x: int, y: int) -> bool:
    """True if (x,y) hits the Start button. Tries image-space, scaled window-space, and mapped rect."""
    rect = state.get("button_rect", (0, 0, 0, 0))
    rx, ry, rw, rh = rect
    fs = state.get("frame_shape", (0, 0))
    if len(fs) < 2 or fs[0] <= 0 or fs[1] <= 0:
        return False
    fh, fw = int(fs[0]), int(fs[1])
    px, py, pw, ph = _padded_button_rect(rx, ry, rw, rh, fh, fw)
    # 0) Some backends pass mouse coords already in image space (matches drawn button)
    if _hit_in_rect(x, y, px, py, pw, ph):
        return True
    img_x, img_y = _window_to_image_coords(state, x, y)
    if _hit_in_rect(img_x, img_y, px, py, pw, ph):
        return True
    display = _get_display_scale_and_offset(state)
    if display is not None:
        scale, offset_x, offset_y, _ = display
        bx1 = offset_x + px * scale
        by1 = offset_y + py * scale
        bx2 = offset_x + (px + pw) * scale
        by2 = offset_y + (py + ph) * scale
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return True
    return False


def _trigger_start_toggle(run_state: dict[str, Any]) -> None:
    rs = run_state.get("rep_session")
    if not isinstance(rs, RepCounterSession):
        return
    if not run_state.get("started", False):
        run_state["started"] = True
        run_state["started_at"] = time.time()
        rs.set_started(run_state["started_at"])
    else:
        rs.clear_tracking_keep_started()


def _on_mouse(event: int, x: int, y: int, _flags: int, param: dict[str, Any]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    state = param
    rect = state.get("button_rect", (0, 0, 0, 0))
    rx, ry, rw, rh = rect
    img_x, img_y = _window_to_image_coords(state, x, y)
    fs = state.get("frame_shape", (0, 0))
    fh, fw = (int(fs[0]), int(fs[1])) if len(fs) >= 2 else (0, 0)
    px, py, pw, ph = _padded_button_rect(rx, ry, rw, rh, fh, fw) if fw > 0 and fh > 0 else (0, 0, 0, 0)
    in_direct = _hit_in_rect(x, y, px, py, pw, ph) if pw > 0 else False
    in_mapped_img = _hit_in_rect(img_x, img_y, px, py, pw, ph) if pw > 0 else False
    display = _get_display_scale_and_offset(state)
    in_rect_win = False
    if display is not None and pw > 0:
        sc, ox, oy, _ = display
        bx1, bx2 = ox + px * sc, ox + (px + pw) * sc
        by1, by2 = oy + py * sc, oy + (py + ph) * sc
        in_rect_win = bx1 <= x <= bx2 and by1 <= y <= by2
    in_rect = _start_button_hit(state, x, y)
    # #region agent log
    _rect_dbg = None
    try:
        _rect_dbg = list(cv2.getWindowImageRect("Rep Counter"))
    except Exception:
        _rect_dbg = None
    _agent_dbg(
        "H1",
        "opencv_runtime.py:_on_mouse",
        "lbutton_down",
        {
            "win_xy": [x, y],
            "img_xy": [img_x, img_y],
            "button_rect": [rx, ry, rw, rh],
            "hit_pad_rect": [px, py, pw, ph],
            "frame_shape": list(state.get("frame_shape", ())),
            "display_is_none": display is None,
            "display": list(display) if display is not None else None,
            "in_direct": in_direct,
            "in_rect_img": in_mapped_img,
            "in_rect_win": in_rect_win,
            "in_rect_final": in_rect,
            "win_image_rect": _rect_dbg,
            "started": bool(state.get("started")),
        },
    )
    # #endregion
    if not in_rect:
        return
    _trigger_start_toggle(state)


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
    text = _ascii_text(text).upper()
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
        safe_txt = _ascii_text(txt)
        cv2.putText(frame, safe_txt, (x1 + 6, y), font, scale, (40, 40, 40), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, safe_txt, (x1 + 6, y), font, scale, col, thick, cv2.LINE_AA)
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


def _draw_library_watermark(frame: Any) -> None:
    """Small label so the window is clearly driven by the library, not bundled UI in the wheel."""
    w = frame.shape[1]
    label = "flexible_rep_counter"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    safe = _ascii_text(label)
    tw, th = cv2.getTextSize(safe, font, scale, thick)[0]
    margin = 8
    x1 = max(0, w - tw - margin * 2)
    y1 = th + margin * 2
    _draw_transparent_box(frame, x1 - 4, margin, w - margin, y1 + 4)
    tx, ty = x1, y1
    cv2.putText(frame, safe, (tx, ty), font, scale, (40, 40, 40), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, safe, (tx, ty), font, scale, (220, 220, 255), thick, cv2.LINE_AA)


def _draw_overlay(frame: Any, step: StepResult) -> None:
    margin = 10
    y = 30
    line_height = 30
    box_x1 = margin
    box_y1 = 12
    show_rom = (
        step.phase == "tracking"
        and step.calibration_complete
        and (step.avg_peak is not None or step.avg_valley is not None)
    )
    td = (step.tracking_detail_message or "").strip()
    show_td = bool(td and td != step.status_message)
    sel_dbg = step.selection_debug or {}
    rep_dom_sel = sel_dbg.get("rep_dom") if isinstance(sel_dbg, dict) else None
    show_sel_pulses = step.phase == "selecting" and isinstance(rep_dom_sel, dict)
    lines = 4
    if step.tracked_joint:
        lines += 1
    if step.smoothed_value is not None:
        lines += 1
    if show_rom:
        lines += 1
    if show_td:
        lines += 1
    if show_sel_pulses:
        lines += 1
    box_x2 = min(frame.shape[1] - margin, 620)
    box_y2 = box_y1 + lines * line_height + 8
    _draw_transparent_box(frame, box_x1, box_y1, box_x2, box_y2)
    _put_text_readable(
        frame,
        f"Phase: {step.phase.upper()}",
        (margin, y),
        OVERLAY_FONT,
        0.55,
        OVERLAY_COLOR_DIM,
        2,
    )
    y += line_height
    if step.tracked_joint:
        _put_text_readable(
            frame,
            f"Angle: {step.tracked_joint}",
            (margin, y),
            OVERLAY_FONT,
            OVERLAY_SCALE,
            OVERLAY_COLOR,
            OVERLAY_THICKNESS,
        )
        y += line_height
    if step.smoothed_value is not None:
        _put_text_readable(
            frame,
            f"Value: {step.smoothed_value:.1f}",
            (margin, y),
            OVERLAY_FONT,
            OVERLAY_SCALE,
            OVERLAY_COLOR_DIM,
            OVERLAY_THICKNESS,
        )
        y += line_height
    rep_line = f"Reps: {step.reps}"
    if not step.calibration_complete:
        rep_line += f"  (cal {step.reps_raw}/{step.calibration_target_reps})"
    if show_sel_pulses:
        rep_line += "  — locked reps after angle is chosen"
    _put_text_readable(
        frame,
        rep_line,
        (margin, y),
        OVERLAY_FONT,
        OVERLAY_SCALE,
        OVERLAY_COLOR,
        OVERLAY_THICKNESS,
    )
    y += line_height
    if show_sel_pulses:
        tr = int(rep_dom_sel.get("totalReps") or 0)
        lk = rep_dom_sel.get("leaderKey")
        lk_s = str(lk) if lk else "—"
        _put_text_readable(
            frame,
            f"Motion pulses (all joints): {tr}  leader: {lk_s}",
            (margin, y),
            OVERLAY_FONT,
            0.52,
            OVERLAY_COLOR_DIM,
            2,
        )
        y += line_height
    _put_text_readable(
        frame,
        f"State: {step.peak_detector_state}",
        (margin, y),
        OVERLAY_FONT,
        0.6,
        OVERLAY_COLOR_DIM,
        2,
    )
    y += line_height
    _put_text_readable(
        frame,
        step.status_message,
        (margin, y),
        OVERLAY_FONT,
        0.6,
        OVERLAY_COLOR_STATUS,
        2,
    )
    y += line_height
    if show_rom:
        ap = f"{step.avg_peak:.0f}" if step.avg_peak is not None else "—"
        av = f"{step.avg_valley:.0f}" if step.avg_valley is not None else "—"
        _put_text_readable(
            frame,
            f"ROM ref: peak {ap}  valley {av} deg",
            (margin, y),
            OVERLAY_FONT,
            0.55,
            OVERLAY_COLOR_DIM,
            2,
        )
        y += line_height
    if show_td:
        _put_text_readable(frame, td, (margin, y), OVERLAY_FONT, 0.55, OVERLAY_COLOR_DIM, 2)


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

    rep_session = RepCounterSession(auto_started=False)
    run_state: dict[str, Any] = {
        "started": False,
        "rep_session": rep_session,
        "button_rect": (0, 0, BUTTON_W_MIN, BUTTON_H_MIN),
        "frame_shape": (0, 0),
        "debug_frame_i": 0,
    }
    cv2.namedWindow("Rep Counter", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Rep Counter", _on_mouse, run_state)
    ensure_console_window()
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

    fps_window = 30
    fps_times: deque[float] = deque(maxlen=fps_window)
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
                    frame_bgr, "Click START or press Space to begin",
                    (cx, cy),
                    OVERLAY_FONT, 0.7, OVERLAY_COLOR, 2,
                )
                cv2.imshow("Rep Counter", frame_bgr)
                update_console_window()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord(" "):
                    _trigger_start_toggle(run_state)
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

            rs_sess = run_state["rep_session"]
            timestamp_ms = time.time() * 1000.0
            if raw_landmarks is None:
                step = rs_sess.step_landmarks(None, timestamp_ms=timestamp_ms)
                _draw_overlay(frame_bgr, step)
                _draw_library_watermark(frame_bgr)
                _draw_vm_benchmark_hud(
                    frame_bgr, disp_b, benchmark_peaks, cam_fps, inf_fps, val_issues
                )
                cv2.imshow("Rep Counter", frame_bgr)
                update_console_window()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord(" "):
                    _trigger_start_toggle(run_state)
                continue

            disp_h, disp_w = frame_bgr.shape[0], frame_bgr.shape[1]
            sent_ok = isinstance(sent_hw, tuple) and len(sent_hw) >= 2
            raw_scaled = scale_landmarks_to_display(
                raw_landmarks,
                sent_hw if sent_ok else None,
                (disp_h, disp_w),
            )
            step = rs_sess.step_landmarks(raw_scaled, timestamp_ms=timestamp_ms)
            # #region agent log
            run_state["debug_frame_i"] = int(run_state.get("debug_frame_i", 0)) + 1
            _dfi = run_state["debug_frame_i"]
            if _dfi % 45 == 0:
                _lm0 = raw_scaled[0] if raw_scaled else None
                _agent_dbg(
                    "H3",
                    "opencv_runtime.py:run_webcam_loop",
                    "step_sample",
                    {
                        "frame_i": _dfi,
                        "sent_hw": list(sent_hw) if isinstance(sent_hw, (tuple, list)) else type(sent_hw).__name__,
                        "sent_hw_used": sent_ok,
                        "disp_hw": [disp_h, disp_w],
                        "phase": step.phase,
                        "reps": step.reps,
                        "reps_raw": step.reps_raw,
                        "smoothed_value": step.smoothed_value,
                        "peak_state": step.peak_detector_state,
                        "tracked_joint": step.tracked_joint,
                        "lm0": _lm0,
                        "session_started": rs_sess.started,
                    },
                )
            # #endregion
            sm = rs_sess.last_smoothed_landmarks
            if sm:
                draw_skeleton(frame_bgr, sm)

            _draw_overlay(frame_bgr, step)
            _draw_library_watermark(frame_bgr)
            _draw_vm_benchmark_hud(
                frame_bgr, disp_b, benchmark_peaks, cam_fps, inf_fps, val_issues
            )
            logger.debug(
                "angle=%s smoothed=%s reps=%d state=%s",
                step.tracked_joint,
                step.smoothed_value or 0,
                step.reps,
                step.peak_detector_state,
            )
            cv2.imshow("Rep Counter", frame_bgr)
            update_console_window()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                _trigger_start_toggle(run_state)

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
