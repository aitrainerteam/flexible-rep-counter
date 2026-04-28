"""OpenCV webcam frontend (repo root, not part of the installable package).

Cloud-only: sends frames to the VM and renders landmarks + ``rep_counter`` from
``/predict`` responses. No local :class:`~flexible_rep_counter.session.RepCounterSession``.
Landmark scaling uses :mod:`flexible_rep_counter.landmark_utils`; HTTP stays under ``app``.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import threading
import time
from queue import Empty, Full, Queue
from types import SimpleNamespace
from typing import Any, Optional

# Repo dev layout: allow direct script execution without a prior `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if _path.is_dir():
        _sp = str(_path)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

from flexible_rep_counter.landmark_utils import scale_landmarks_to_display

from app.config import (
    PREDICT_JPEG_QUALITY,
    PREDICT_RESIZE_WIDTH,
    PREDICT_VALIDATE_RESPONSE,
    VM_BASE_URL,
    VM_HEALTH_TIMEOUT_SEC,
    VM_TIMEOUT_SEC,
)
from app.vm_client_2 import check_vm_health, send_frame, send_reset
from app.debug_console import (
    ensure_console_window,
    get_logger,
    setup_logging,
    update_console_window,
)
from app.skeleton_overlay import draw_skeleton

logger = get_logger(__name__)

DEBUG_USER_UID = "debug-opencv-simulator"
SIMULATOR_POST_PARAMS: dict[str, str] = {
    "user_uid": DEBUG_USER_UID,
    "rep_enabled": "true",
    "response_mode": "aux_compat",
}

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
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 960
LOCAL_SEND_FPS_CAP = 12.0
LOCAL_SEND_INTERVAL_SEC = 1.0 / LOCAL_SEND_FPS_CAP


def _camera_backend_candidates(platform: str) -> list[tuple[str, int]]:
    if platform == "darwin":
        raw = [("avfoundation", getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))]
    elif platform == "win32":
        raw = [
            ("dshow", getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
            ("msmf", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
            ("any", cv2.CAP_ANY),
        ]
    else:
        raw = [("any", cv2.CAP_ANY)]

    seen: set[int] = set()
    out: list[tuple[str, int]] = []
    for name, backend in raw:
        if backend in seen:
            continue
        seen.add(backend)
        out.append((name, backend))
    return out


def _open_webcam_capture(camera_index: int) -> tuple[Any, str]:
    tried: list[str] = []
    for backend_name, backend in _camera_backend_candidates(sys.platform):
        tried.append(backend_name)
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened() and _capture_has_usable_frame(cap, backend_name):
            return cap, backend_name
        try:
            cap.release()
        except Exception:
            pass

    tried_str = ", ".join(tried)
    raise RuntimeError(
        f"Could not open webcam (index {camera_index}; tried backends: {tried_str}). "
        "Try a different camera index if your webcam is not device 0."
    )


def _capture_has_usable_frame(
    cap: Any,
    backend_name: str,
    *,
    max_reads: int = 6,
    warmup_delay_sec: float = 0.03,
) -> bool:
    for attempt in range(1, max_reads + 1):
        try:
            ok, frame = cap.read()
        except Exception as exc:
            logger.debug("webcam probe backend=%s attempt=%d read failed: %s", backend_name, attempt, exc)
            ok, frame = False, None
        if ok and _frame_looks_usable(frame):
            if attempt > 1:
                logger.debug("webcam probe backend=%s usable frame after %d attempts", backend_name, attempt)
            return True
        if warmup_delay_sec > 0:
            time.sleep(warmup_delay_sec)
    _log_timed_event(
        logger.warning,
        "webcam backend=%s opened but did not produce a usable frame after %d attempts; trying next backend",
        backend_name,
        max_reads,
    )
    return False


def _frame_looks_usable(frame: Any) -> bool:
    if frame is None:
        return False

    size = getattr(frame, "size", None)
    if size is not None:
        try:
            if int(size) <= 0:
                return False
        except (TypeError, ValueError):
            pass

    shape = getattr(frame, "shape", None)
    if shape is not None:
        try:
            if len(shape) >= 2 and (int(shape[0]) <= 0 or int(shape[1]) <= 0):
                return False
        except (TypeError, ValueError):
            pass

    any_method = getattr(frame, "any", None)
    if callable(any_method):
        try:
            return bool(any_method())
        except Exception:
            pass

    if isinstance(frame, (list, tuple)):
        stack: list[Any] = [frame]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
                continue
            try:
                if float(item) != 0.0:
                    return True
            except (TypeError, ValueError):
                if bool(item):
                    return True
        return False

    return True


def _build_waiting_frame(*, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT) -> Any:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (32, 32, 32)
    cv2.putText(
        frame,
        "Waiting for video...",
        (max(20, width // 2 - 170), max(40, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (220, 220, 220),
        2,
    )
    return frame


def _letterbox_frame_to_canvas(frame: Any, *, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT) -> Any:
    if frame is None or getattr(frame, "shape", None) is None or len(frame.shape) < 2:
        return _build_waiting_frame(width=width, height=height)

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return _build_waiting_frame(width=width, height=height)

    canvas = np.zeros((height, width, 3), dtype=frame.dtype if hasattr(frame, "dtype") else np.uint8)
    scale = min(width / max(1, w), height / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled
    return canvas


class _DisplayWindow:
    def __init__(
        self,
        *,
        window_name: str,
        run_state: dict[str, Any],
        width: int = DISPLAY_WIDTH,
        height: int = DISPLAY_HEIGHT,
    ) -> None:
        self._window_name = window_name
        self._run_state = run_state
        self._width = width
        self._height = height
        self._frame_lock = threading.Lock()
        self._latest_frame: Any = None
        self._frames_rendered_total = 0
        self._running = False
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None

    def publish_frame(self, frame: Any) -> None:
        if frame is None:
            return
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    @property
    def frames_rendered_total(self) -> int:
        return int(self._frames_rendered_total)

    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _display_loop(self) -> None:
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, self._width, self._height)
            cv2.setMouseCallback(self._window_name, _on_mouse, self._run_state)
            while self._running:
                with self._frame_lock:
                    frame = self._latest_frame.copy() if self._latest_frame is not None else None
                display_frame = _letterbox_frame_to_canvas(frame, width=self._width, height=self._height)
                cv2.imshow(self._window_name, display_frame)
                self._frames_rendered_total += 1
                update_console_window()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self._stop_requested.set()
                    self._running = False
                elif key == ord(" "):
                    _trigger_start_toggle(self._run_state)
        except Exception as exc:
            self._stop_requested.set()
            _log_timed_event(logger.error, "display loop error: %s", exc)
        finally:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass


class _RecoveringCapture:
    def __init__(
        self,
        *,
        camera_index: int,
        open_capture: Any = None,
        max_bad_frames: int = 5,
    ) -> None:
        self._camera_index = int(camera_index)
        self._open_capture = open_capture or _open_webcam_capture
        self._max_bad_frames = max(1, int(max_bad_frames))
        self._bad_frame_streak = 0
        self.reopen_count = 0
        self.read_failures_total = 0
        self.bad_frames_total = 0
        self.good_frames_total = 0
        self.last_good_frame_monotonic: float | None = None
        self._cap: Any = None
        self.backend_name = "unknown"
        self._reopen(initial=True)

    def _reopen(self, *, initial: bool = False) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap, self.backend_name = self._open_capture(self._camera_index)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._bad_frame_streak = 0
        if not initial:
            self.reopen_count += 1
        level = logger.info if initial else logger.warning
        _log_timed_event(
            level,
            "webcam capture %s backend=%s index=%s",
            "opened" if initial else "reopened",
            self.backend_name,
            self._camera_index,
        )

    def read(self) -> tuple[bool, Any]:
        if self._cap is None:
            self._reopen()
        try:
            ok, frame = self._cap.read()
        except Exception as exc:
            _log_timed_event(logger.warning, "webcam read failed backend=%s index=%s: %s", self.backend_name, self._camera_index, exc)
            self.read_failures_total += 1
            ok, frame = False, None
        if ok and _frame_looks_usable(frame):
            self._bad_frame_streak = 0
            self.good_frames_total += 1
            self.last_good_frame_monotonic = time.monotonic()
            return True, frame

        self._bad_frame_streak += 1
        self.bad_frames_total += 1
        if self._bad_frame_streak >= self._max_bad_frames:
            _log_timed_event(
                logger.warning,
                "webcam backend=%s index=%s produced %d consecutive unusable frames; reopening capture",
                self.backend_name,
                self._camera_index,
                self._bad_frame_streak,
            )
            self._reopen()
            try:
                ok2, frame2 = self._cap.read()
            except Exception as exc:
                _log_timed_event(
                    logger.warning,
                    "webcam read after reopen failed backend=%s index=%s: %s",
                    self.backend_name,
                    self._camera_index,
                    exc,
                )
                self.read_failures_total += 1
                return False, None
            if ok2 and _frame_looks_usable(frame2):
                self.good_frames_total += 1
                self.last_good_frame_monotonic = time.monotonic()
                return True, frame2
            self.bad_frames_total += 1
        return False, None

    def release(self) -> None:
        if self._cap is None:
            return
        try:
            self._cap.release()
        finally:
            self._cap = None


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


def _unix_ts_ms() -> float:
    return time.time() * 1000.0


def _format_timed_event(message: str, *, ts_ms: float | None = None) -> str:
    event_ts_ms = _unix_ts_ms() if ts_ms is None else float(ts_ms)
    return f"ts_ms={event_ts_ms:.4f} {message}"


def _log_timed_event(log_method: Any, message: str, *args: Any) -> None:
    rendered = message % args if args else message
    log_method(_format_timed_event(rendered))


def _print_timed_event(message: str, *, file: Any = None) -> None:
    print(_format_timed_event(message), file=file or sys.stdout, flush=True)


def _cloud_rep_count(vm_rep_counter: Any) -> int:
    if isinstance(vm_rep_counter, dict):
        try:
            return max(0, int(vm_rep_counter.get("reps", 0) or 0))
        except (TypeError, ValueError):
            return 0
    return 0


class _RuntimeDiagnostics:
    def __init__(
        self,
        *,
        logger_obj: Any = None,
        heartbeat_sec: float = 5.0,
        vm_stall_sec: float = 2.5,
        start_monotonic: float | None = None,
    ) -> None:
        self._logger = logger_obj or logger
        self._heartbeat_sec = max(1.0, float(heartbeat_sec))
        self._vm_stall_sec = max(0.5, float(vm_stall_sec))
        self._started_monotonic = time.monotonic() if start_monotonic is None else float(start_monotonic)
        self._last_heartbeat_monotonic = self._started_monotonic
        self._first_enqueue_monotonic: float | None = None
        self._last_vm_response_monotonic: float | None = None
        self._last_vm_token: int | None = None
        self._last_display_frames_total = 0
        self._queue_drop_active = False
        self._queue_drop_streak = 0
        self._vm_stall_active = False
        self._display_stall_active = False
        self._display_dead_logged = False
        self._capture_ok_interval = 0
        self._capture_bad_interval = 0
        self._published_interval = 0
        self._queue_enqueued_interval = 0
        self._queue_dropped_interval = 0
        self._vm_responses_interval = 0
        self._vm_landmarks_interval = 0
        self._last_cloud_reps: int | None = None

    def _emit(self, level: str, message: str, *args: Any) -> None:
        log_method = getattr(self._logger, level)
        _log_timed_event(log_method, message, *args)

    def note_capture_result(self, *, ok: bool) -> None:
        if ok:
            self._capture_ok_interval += 1
        else:
            self._capture_bad_interval += 1

    def note_frame_published(self) -> None:
        self._published_interval += 1

    def note_frame_enqueued(self, *, now_monotonic: float) -> None:
        self._queue_enqueued_interval += 1
        if self._first_enqueue_monotonic is None:
            self._first_enqueue_monotonic = now_monotonic
        if self._queue_drop_active:
            self._emit("info", "diag pose queue recovered after %d dropped frames", self._queue_drop_streak)
            self._queue_drop_active = False
            self._queue_drop_streak = 0

    def note_frame_dropped(self) -> None:
        self._queue_dropped_interval += 1
        self._queue_drop_streak += 1
        if self._queue_drop_active:
            return
        self._queue_drop_active = True
        self._emit("warning", "diag pose queue saturated; dropping frames until the VM worker catches up")

    def note_vm_snapshot(self, snap: dict[str, Any], *, now_monotonic: float) -> None:
        if not isinstance(snap, dict):
            return
        if "rep_counter" not in snap and "rep_session_reset" not in snap and "sent_hw" not in snap:
            return
        token = id(snap)
        if token == self._last_vm_token:
            return

        idle_for = None
        if self._last_vm_response_monotonic is not None:
            idle_for = now_monotonic - self._last_vm_response_monotonic
        elif self._first_enqueue_monotonic is not None:
            idle_for = now_monotonic - self._first_enqueue_monotonic

        self._last_vm_token = token
        self._last_vm_response_monotonic = now_monotonic
        self._vm_responses_interval += 1
        if snap.get("landmarks") is not None:
            self._vm_landmarks_interval += 1
        if self._vm_stall_active:
            self._emit("info", "diag VM responses recovered after %.1fs idle", max(0.0, float(idle_for or 0.0)))
            self._vm_stall_active = False

    def note_cloud_rep_count(self, *, cloud_reps: int) -> None:
        """Log when VM ``rep_counter.reps`` increases (no local rep counter)."""
        cloud_now = max(0, int(cloud_reps))
        if self._last_cloud_reps is None:
            self._last_cloud_reps = cloud_now
            return
        if cloud_now > self._last_cloud_reps:
            self._emit("info", "event=cloud_rep_counted cloud_reps=%d", cloud_now)
        self._last_cloud_reps = cloud_now

    def maybe_log(
        self,
        *,
        now_monotonic: float,
        started: bool,
        cam_fps: float,
        inf_fps: float,
        capture: Any,
        display: Any,
    ) -> None:
        self._maybe_log_vm_stall(now_monotonic=now_monotonic, started=started)

        display_alive = bool(getattr(display, "is_alive", lambda: False)())
        if not display_alive:
            if not self._display_dead_logged:
                self._emit("warning", "diag display thread is not alive")
                self._display_dead_logged = True
        elif self._display_dead_logged:
            self._emit("info", "diag display thread is alive again")
            self._display_dead_logged = False

        elapsed = now_monotonic - self._last_heartbeat_monotonic
        if elapsed < self._heartbeat_sec:
            return

        display_total = int(getattr(display, "frames_rendered_total", 0) or 0)
        display_interval = max(0, display_total - self._last_display_frames_total)
        self._last_display_frames_total = display_total

        if started and self._published_interval > 0 and display_interval == 0:
            if not self._display_stall_active:
                self._emit(
                    "warning",
                    "diag display rendered 0 frames in the last %.1fs while %d frames were published",
                    elapsed,
                    self._published_interval,
                )
                self._display_stall_active = True
        elif display_interval > 0 and self._display_stall_active:
            self._emit("info", "diag display rendering recovered")
            self._display_stall_active = False

        if self._vm_responses_interval > 0:
            landmark_ratio = f"{(100.0 * self._vm_landmarks_interval / self._vm_responses_interval):.0f}%"
        else:
            landmark_ratio = "n/a"
        if self._last_vm_response_monotonic is None:
            last_vm_age = "none"
        else:
            last_vm_age = f"{max(0.0, now_monotonic - self._last_vm_response_monotonic):.1f}s"

        self._emit(
            "info",
            "diag %.1fs backend=%s cam=%.1ffps infer=%.1f/s cap ok=%d bad=%d pub=%d ui=%d q enq=%d drop=%d vm rsp=%d lm=%s reopens=%d display=%s last_vm=%s",
            elapsed,
            getattr(capture, "backend_name", "unknown"),
            cam_fps,
            inf_fps,
            self._capture_ok_interval,
            self._capture_bad_interval,
            self._published_interval,
            display_interval,
            self._queue_enqueued_interval,
            self._queue_dropped_interval,
            self._vm_responses_interval,
            landmark_ratio,
            int(getattr(capture, "reopen_count", 0) or 0),
            "alive" if display_alive else "dead",
            last_vm_age,
        )
        self._reset_interval(now_monotonic)

    def _maybe_log_vm_stall(self, *, now_monotonic: float, started: bool) -> None:
        if not started or self._vm_stall_active or self._first_enqueue_monotonic is None:
            return
        last_seen = self._last_vm_response_monotonic
        idle_since = last_seen if last_seen is not None else self._first_enqueue_monotonic
        idle_for = now_monotonic - idle_since
        if idle_for < self._vm_stall_sec:
            return
        if last_seen is None:
            self._emit("warning", "diag no VM response received %.1fs after frames started enqueuing", idle_for)
        else:
            self._emit("warning", "diag no new VM response for %.1fs", idle_for)
        self._vm_stall_active = True

    def _reset_interval(self, now_monotonic: float) -> None:
        self._last_heartbeat_monotonic = now_monotonic
        self._capture_ok_interval = 0
        self._capture_bad_interval = 0
        self._published_interval = 0
        self._queue_enqueued_interval = 0
        self._queue_dropped_interval = 0
        self._vm_responses_interval = 0
        self._vm_landmarks_interval = 0


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
    """Toggle streaming + HUD; rep state lives only on the VM."""
    if not run_state.get("started", False):
        run_state["started"] = True
        run_state["started_at"] = time.time()
    else:
        # Previously cleared local RepCounterSession tracking; cloud state is unchanged here.
        run_state["started_at"] = time.time()


def _on_mouse(event: int, x: int, y: int, _flags: int, param: dict[str, Any]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    state = param
    in_rect = _start_button_hit(state, x, y)
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
    for key in (
        "roundtrip_ms",
        "upload_ms",
        "encode_ms",
        "session_ms",
        "detector_ms",
        "variance_ms",
    ):
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
        session_ms = benchmark.get("session_ms")
        detector_ms = benchmark.get("detector_ms")
        variance_ms = benchmark.get("variance_ms")
        if session_ms is not None or detector_ms is not None or variance_ms is not None:
            lines.append(
                (
                    f"cpu sess {float(session_ms or 0):.1f}ms  det {float(detector_ms or 0):.1f}ms  var {float(variance_ms or 0):.1f}ms",
                    (200, 230, 255),
                )
            )
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
    p_session = peaks.get("session_ms")
    if p_session is not None:
        peak_bits.append(f"cpu_peak {p_session:.1f}")
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


def _draw_overlay_from_vm(
    frame: Any,
    vm_rep_counter: Optional[dict],
    reset_info: Optional[dict],
) -> None:
    """Secondary overlay showing VM-side rep_counter state (source of truth for NDJSON).

    Drawn at bottom-left to avoid overlapping the benchmark HUD
    (_draw_vm_benchmark_hud, bottom-right).
    """
    h, w = frame.shape[:2]
    margin = 10
    box_x1 = margin
    box_x2 = min(w // 2, 420)

    vm = SimpleNamespace(**vm_rep_counter) if isinstance(vm_rep_counter, dict) else None

    lines: list[str] = ["VM REP COUNTER"]
    if vm is None:
        lines.append("(waiting for /predict response)")
    else:
        phase = str(getattr(vm, "phase", "idle"))
        reps = getattr(vm, "reps", 0)
        reps_raw = getattr(vm, "reps_raw", None)
        tracked = getattr(vm, "tracked_joint", None)
        state = getattr(vm, "peak_detector_state", None)
        smoothed = getattr(vm, "smoothed_value", None)
        avg_p = getattr(vm, "avg_peak", None)
        avg_v = getattr(vm, "avg_valley", None)
        calib_ok = bool(getattr(vm, "calibration_complete", False))
        gate_open = bool(getattr(vm, "range_gate_open", False))
        status = str(getattr(vm, "status_message", "") or "")

        lines.append(f"Phase: {phase.upper()}")
        if reps_raw is not None and reps_raw != reps:
            lines.append(f"Reps: {reps}  (raw {reps_raw})")
        else:
            lines.append(f"Reps: {reps}")
        if tracked:
            lines.append(f"Angle: {tracked}")
        if state:
            lines.append(f"State: {state}")
        if smoothed is not None:
            try:
                lines.append(f"Value: {float(smoothed):.1f}")
            except (TypeError, ValueError):
                pass
        if calib_ok and (avg_p is not None or avg_v is not None):
            lines.append(f"ROM: peak={avg_p} valley={avg_v}")
        lines.append(f"Range gate: {'open' if gate_open else 'closed'}")
        if status:
            lines.append(f"Status: {status}")

    if reset_info is not None:
        lines.append(
            f"(startup reset: had_session={reset_info.get('had_session')})"
        )

    line_h = 22
    box_y2 = h - margin
    box_y1 = box_y2 - (len(lines) * line_h + 8)
    _draw_transparent_box(frame, box_x1, box_y1, box_x2, box_y2)
    y = box_y1 + line_h
    for txt in lines:
        _put_text_readable(
            frame,
            txt,
            (box_x1 + 6, y),
            OVERLAY_FONT,
            0.5,
            OVERLAY_COLOR,
            2,
        )
        y += line_h


def _pose_worker(
    frame_queue: Queue,
    result_holder: list,
    stop_event: threading.Event,
    pose_options: dict[str, Any],
) -> None:
    """Background thread: send frames to VM; store latest landmarks + benchmark per /predict."""
    import requests

    from app.vm_client_2 import send_frame

    session = pose_options.get("http_session") or requests.Session()
    resize_width = int(pose_options.get("resize_width", 0))
    jpeg_quality = int(pose_options.get("jpeg_quality", 85))
    validate = bool(pose_options.get("validate", True))
    send_params = pose_options.get("send_params") or None
    parse_rep_counter = bool(pose_options.get("parse_rep_counter", False))
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
            params=send_params,
            parse_rep_counter=parse_rep_counter,
        )
        result_holder[0] = {
            "landmarks": outcome.landmarks,
            "benchmark": outcome.benchmark,
            "sent_hw": outcome.sent_hw,
            "validation_issues": outcome.validation_issues,
            "rep_counter": outcome.rep_counter,
            "rep_session_reset": outcome.rep_session_reset,
        }


def _draw_library_watermark(frame: Any) -> None:
    """Small label: this window reflects VM /predict state only."""
    w = frame.shape[1]
    label = "cloud state (VM)"
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


def run_webcam_loop(
    *,
    skip_health_check: bool = False,
    benchmark_log_path: Optional[str] = None,
    resize_width: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    validate_response: Optional[bool] = None,
    camera_index: int = 0,
) -> None:
    setup_logging()

    rw = PREDICT_RESIZE_WIDTH if resize_width is None else int(resize_width)
    jq = PREDICT_JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality)
    val = PREDICT_VALIDATE_RESPONSE if validate_response is None else bool(validate_response)

    if not skip_health_check:
        ok, info = check_vm_health(timeout=VM_HEALTH_TIMEOUT_SEC)
        if not ok:
            raise RuntimeError(f"VM health check failed: {info}")

    import requests as _rq

    http_session = _rq.Session()
    reset_info: Optional[dict] = send_reset(
        session=http_session,
        user_uid=DEBUG_USER_UID,
        timeout=VM_HEALTH_TIMEOUT_SEC,
    )
    if reset_info is None:
        _log_timed_event(logger.warning, "startup session reset failed or VM returned no rep_session_reset")
    else:
        _log_timed_event(logger.info, "startup reset: %s", reset_info)

    capture = _RecoveringCapture(camera_index=camera_index)

    run_state: dict[str, Any] = {
        "started": False,
        "button_rect": (0, 0, BUTTON_W_MIN, BUTTON_H_MIN),
        "frame_shape": (0, 0),
        "reset_info": reset_info,
    }
    ensure_console_window()
    display = _DisplayWindow(window_name="Rep Counter", run_state=run_state)
    display.start()
    diag = _RuntimeDiagnostics()
    logger.debug(
        "webcam opened index=%s backend=%s, VM predict resize_width=%s jpeg_q=%s validate=%s",
        camera_index,
        capture.backend_name,
        rw,
        jq,
        val,
    )
    if benchmark_log_path:
        _log_timed_event(logger.info, "Benchmark log: appending inference/roundtrip to %s", benchmark_log_path)

    pose_options: dict[str, Any] = {
        "resize_width": rw,
        "jpeg_quality": jq,
        "validate": val,
        "http_session": http_session,
        "send_params": SIMULATOR_POST_PARAMS,
        "parse_rep_counter": True,
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
    cam_fps = 0.0
    inf_fps = 0.0
    benchmark_peaks: dict[str, float | None] = {
        "roundtrip_ms": None,
        "upload_ms": None,
        "encode_ms": None,
        "inference_ms": None,
        "session_ms": None,
        "detector_ms": None,
        "variance_ms": None,
    }
    next_send_monotonic = 0.0

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
        while not display.stop_requested():
            ret, frame_bgr = capture.read()
            loop_now = time.monotonic()
            diag.note_capture_result(ok=bool(ret and frame_bgr is not None))
            if not ret or frame_bgr is None:
                diag.maybe_log(
                    now_monotonic=loop_now,
                    started=bool(run_state["started"]),
                    cam_fps=cam_fps,
                    inf_fps=inf_fps,
                    capture=capture,
                    display=display,
                )
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
                display.publish_frame(frame_bgr)
                diag.note_frame_published()
                diag.maybe_log(
                    now_monotonic=loop_now,
                    started=False,
                    cam_fps=cam_fps,
                    inf_fps=inf_fps,
                    capture=capture,
                    display=display,
                )
                continue

            # Local-only sender cap: reduce upload cadence without touching VM-side limits.
            if loop_now >= next_send_monotonic:
                next_send_monotonic = loop_now + LOCAL_SEND_INTERVAL_SEC
                try:
                    frame_queue.put_nowait(frame_bgr.copy())
                    diag.note_frame_enqueued(now_monotonic=loop_now)
                except Full:
                    diag.note_frame_dropped()
            snap = latest_pose[0]
            diag.note_vm_snapshot(snap, now_monotonic=loop_now)
            disp_b, inf_fps, val_issues = _update_vm_metrics(snap)
            raw_landmarks = snap.get("landmarks")
            sent_hw = snap.get("sent_hw")
            logger.debug("frame sent, got landmarks=%s", raw_landmarks is not None)
            cloud_reps = _cloud_rep_count(snap.get("rep_counter"))
            diag.note_cloud_rep_count(cloud_reps=cloud_reps)

            if raw_landmarks is None:
                _draw_overlay_from_vm(frame_bgr, snap.get("rep_counter"), run_state.get("reset_info"))
                _draw_library_watermark(frame_bgr)
                _draw_vm_benchmark_hud(
                    frame_bgr, disp_b, benchmark_peaks, cam_fps, inf_fps, val_issues
                )
                display.publish_frame(frame_bgr)
                diag.note_frame_published()
                diag.maybe_log(
                    now_monotonic=loop_now,
                    started=True,
                    cam_fps=cam_fps,
                    inf_fps=inf_fps,
                    capture=capture,
                    display=display,
                )
                continue

            disp_h, disp_w = frame_bgr.shape[0], frame_bgr.shape[1]
            sent_ok = isinstance(sent_hw, tuple) and len(sent_hw) >= 2
            raw_scaled = scale_landmarks_to_display(
                raw_landmarks,
                sent_hw if sent_ok else None,
                (disp_h, disp_w),
            )
            draw_skeleton(frame_bgr, raw_scaled)

            _draw_overlay_from_vm(frame_bgr, snap.get("rep_counter"), run_state.get("reset_info"))
            _draw_library_watermark(frame_bgr)
            _draw_vm_benchmark_hud(
                frame_bgr, disp_b, benchmark_peaks, cam_fps, inf_fps, val_issues
            )
            vm_rc = snap.get("rep_counter")
            if isinstance(vm_rc, dict):
                logger.debug(
                    "angle=%s smoothed=%s reps=%d state=%s",
                    vm_rc.get("tracked_joint"),
                    vm_rc.get("smoothed_value") or 0,
                    vm_rc.get("reps", 0),
                    vm_rc.get("peak_detector_state"),
                )
            display.publish_frame(frame_bgr)
            diag.note_frame_published()
            diag.maybe_log(
                now_monotonic=loop_now,
                started=True,
                cam_fps=cam_fps,
                inf_fps=inf_fps,
                capture=capture,
                display=display,
            )

    finally:
        stop_worker.set()
        try:
            frame_queue.put_nowait(None)
        except Full:
            pass
        worker.join(timeout=VM_TIMEOUT_SEC + 1.0)
        capture.release()
        display.stop()
        cv2.destroyAllWindows()
        if last_benchmark:
            _print_timed_event("Last VM request (benchmark):")
            _print_timed_event(
                f"roundtrip={last_benchmark.get('roundtrip_ms', 0):.0f}ms "
                f"upload={last_benchmark.get('upload_ms', 0):.0f}ms "
                f"encode={last_benchmark.get('encode_ms', 0):.0f}ms"
            )
            if last_benchmark.get("inference_ms") is not None:
                _print_timed_event(f"server_inference={last_benchmark['inference_ms']}")
            _print_timed_event(f"payload={last_benchmark.get('payload_kb', 0):.1f}KB")
        if any(v is not None for v in benchmark_peaks.values()):
            _print_timed_event("Session VM peaks (ms):")
            for k in (
                "roundtrip_ms",
                "upload_ms",
                "encode_ms",
                "inference_ms",
                "session_ms",
                "detector_ms",
                "variance_ms",
            ):
                v = benchmark_peaks.get(k)
                if v is not None:
                    _print_timed_event(f"{k}={v:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam rep simulator (VM pose API + VM rep counter)")
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip GET /health (status ok + model_loaded) before opening the camera",
    )
    parser.add_argument(
        "--benchmark-log",
        metavar="FILE",
        help="Append tab-separated timestamp, server inference_ms, roundtrip_ms, upload_ms per VM response",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        metavar="W",
        help="Resize frame to this width before JPEG upload (0=no resize). Default: [predict].resize_width in rep_counter.toml",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        metavar="Q",
        help="JPEG quality 1-100 for uploads. Default: [predict].jpeg_quality in rep_counter.toml",
    )
    parser.add_argument(
        "--no-validate-response",
        action="store_true",
        help="Skip JSON shape checks on /predict (inference_ms, keypoint names)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        metavar="N",
        help="Webcam device index to open. Default: 0",
    )
    args = parser.parse_args()

    _print_timed_event(f"VM rep simulator starting (VM: {VM_BASE_URL})")
    try:
        run_webcam_loop(
            skip_health_check=args.no_health_check,
            benchmark_log_path=args.benchmark_log,
            resize_width=args.resize_width,
            jpeg_quality=args.jpeg_quality,
            validate_response=False if args.no_validate_response else None,
            camera_index=args.camera_index,
        )
    except KeyboardInterrupt:
        _print_timed_event("Stopped by user.")
        sys.exit(0)
    except Exception as exc:
        _print_timed_event(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
