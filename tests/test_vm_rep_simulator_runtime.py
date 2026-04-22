from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = ModuleType("cv2")
    cv2_stub.CAP_ANY = 0
    cv2_stub.CAP_DSHOW = 700
    cv2_stub.CAP_MSMF = 1400
    cv2_stub.FONT_HERSHEY_DUPLEX = 0
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.LINE_AA = 0
    cv2_stub.WINDOW_NORMAL = 0
    cv2_stub.EVENT_LBUTTONDOWN = 1
    cv2_stub.IMWRITE_JPEG_QUALITY = 1
    cv2_stub.VideoCapture = lambda *args, **kwargs: None
    sys.modules["cv2"] = cv2_stub

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy_stub = ModuleType("numpy")
    numpy_stub.uint8 = "uint8"
    numpy_stub.ndarray = object
    numpy_stub.zeros = lambda *args, **kwargs: [[0]]
    sys.modules["numpy"] = numpy_stub

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv_stub

from visualizer import vm_rep_simulator


MODULE_PATH = ROOT / "visualizer" / "vm_rep_simulator.py"


class _FakeCapture:
    def __init__(self, opened: bool, read_results: list[tuple[bool, object]] | None = None) -> None:
        self._opened = opened
        self._has_explicit_reads = read_results is not None
        self._read_results = list(read_results or [])
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, object]:
        if self._read_results:
            return self._read_results.pop(0)
        if self._opened and not self._has_explicit_reads:
            return True, [[1]]
        return False, None

    def release(self) -> None:
        self.released = True


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def info(self, message: str, *args: object) -> None:
        self.events.append(("info", message % args if args else message))

    def warning(self, message: str, *args: object) -> None:
        self.events.append(("warning", message % args if args else message))

    def error(self, message: str, *args: object) -> None:
        self.events.append(("error", message % args if args else message))


class VmRepSimulatorRuntimeTests(unittest.TestCase):
    def test_format_timed_event_prefixes_unix_ms_timestamp(self) -> None:
        message = vm_rep_simulator._format_timed_event(
            "event=local_rep_counted local_reps=3 cloud_reps=2",
            ts_ms=1776780004773.1301,
        )

        self.assertEqual(
            "ts_ms=1776780004773.1301 event=local_rep_counted local_reps=3 cloud_reps=2",
            message,
        )

    def test_runtime_diagnostics_heartbeat_summarizes_pipeline_state(self) -> None:
        fake_logger = _FakeLogger()
        diag = vm_rep_simulator._RuntimeDiagnostics(
            logger_obj=fake_logger,
            heartbeat_sec=5.0,
            vm_stall_sec=10.0,
            start_monotonic=0.0,
        )
        capture = SimpleNamespace(
            backend_name="dshow",
            reopen_count=1,
            bad_frame_streak=0,
        )
        display = SimpleNamespace(frames_rendered_total=18)

        diag.note_capture_result(ok=True)
        diag.note_frame_published()
        diag.note_frame_enqueued(now_monotonic=0.5)
        diag.note_vm_snapshot(
            {
                "benchmark": {"roundtrip_ms": 61.0},
                "landmarks": [[1, 2], [3, 4]],
                "rep_counter": None,
            },
            now_monotonic=0.6,
        )

        diag.maybe_log(
            now_monotonic=5.2,
            started=True,
            cam_fps=27.5,
            inf_fps=16.2,
            capture=capture,
            display=display,
        )

        info_messages = [message for level, message in fake_logger.events if level == "info"]
        self.assertEqual(1, len(info_messages))
        summary = info_messages[0]
        self.assertRegex(summary, r"^ts_ms=\d+\.\d{4} ")
        self.assertIn("diag 5.2s", summary)
        self.assertIn("backend=dshow", summary)
        self.assertIn("cam=27.5fps", summary)
        self.assertIn("infer=16.2/s", summary)
        self.assertIn("pub=1", summary)
        self.assertIn("ui=18", summary)
        self.assertIn("q enq=1 drop=0", summary)
        self.assertIn("vm rsp=1", summary)
        self.assertIn("lm=100%", summary)
        self.assertIn("reopens=1", summary)

    def test_runtime_diagnostics_logs_queue_drop_transition_once_and_recovery(self) -> None:
        fake_logger = _FakeLogger()
        diag = vm_rep_simulator._RuntimeDiagnostics(
            logger_obj=fake_logger,
            heartbeat_sec=5.0,
            vm_stall_sec=2.0,
            start_monotonic=0.0,
        )

        diag.note_frame_dropped()
        diag.note_frame_dropped()
        diag.note_frame_enqueued(now_monotonic=1.0)

        self.assertEqual(2, len(fake_logger.events))
        self.assertEqual("warning", fake_logger.events[0][0])
        self.assertIn("diag pose queue saturated; dropping frames until the VM worker catches up", fake_logger.events[0][1])
        self.assertEqual("info", fake_logger.events[1][0])
        self.assertIn("diag pose queue recovered after 2 dropped frames", fake_logger.events[1][1])

    def test_runtime_diagnostics_logs_cloud_rep_events_only(self) -> None:
        fake_logger = _FakeLogger()
        diag = vm_rep_simulator._RuntimeDiagnostics(
            logger_obj=fake_logger,
            heartbeat_sec=5.0,
            vm_stall_sec=2.0,
            start_monotonic=0.0,
        )

        diag.note_cloud_rep_count(cloud_reps=0)
        diag.note_cloud_rep_count(cloud_reps=0)
        diag.note_cloud_rep_count(cloud_reps=1)
        diag.note_cloud_rep_count(cloud_reps=1)
        diag.note_cloud_rep_count(cloud_reps=2)

        info_messages = [message for level, message in fake_logger.events if level == "info"]
        self.assertEqual(2, len(info_messages))
        self.assertIn("event=cloud_rep_counted cloud_reps=1", info_messages[0])
        self.assertIn("event=cloud_rep_counted cloud_reps=2", info_messages[1])

    def test_recovering_capture_reopens_after_consecutive_bad_frames(self) -> None:
        first = _FakeCapture(
            opened=True,
            read_results=[
                (True, [[0, 0], [0, 0]]),
                (True, [[0, 0], [0, 0]]),
            ],
        )
        second = _FakeCapture(opened=True, read_results=[(True, [[1, 2], [3, 4]])])
        opened: list[tuple[int, _FakeCapture]] = []

        def fake_open(camera_index: int) -> tuple[_FakeCapture, str]:
            cap = first if not opened else second
            opened.append((camera_index, cap))
            return cap, "fake"

        capture = vm_rep_simulator._RecoveringCapture(
            camera_index=7,
            open_capture=fake_open,
            max_bad_frames=2,
        )

        ok1, frame1 = capture.read()
        ok2, frame2 = capture.read()

        self.assertFalse(ok1)
        self.assertIsNone(frame1)
        self.assertTrue(ok2)
        self.assertEqual([[1, 2], [3, 4]], frame2)
        self.assertEqual([7, 7], [camera_index for camera_index, _ in opened])
        self.assertTrue(first.released)
        capture.release()

    def test_run_webcam_loop_does_not_call_cv2_imshow_or_waitkey_directly(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

        run_webcam_loop = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run_webcam_loop":
                run_webcam_loop = node
                break

        self.assertIsNotNone(run_webcam_loop, "run_webcam_loop should exist")

        disallowed_calls: list[str] = []
        assert run_webcam_loop is not None
        for node in ast.walk(run_webcam_loop):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name) or func.value.id != "cv2":
                continue
            if func.attr in {"imshow", "waitKey"}:
                disallowed_calls.append(func.attr)

        self.assertEqual(
            [],
            disallowed_calls,
            "run_webcam_loop should publish frames to a display worker instead of calling cv2.imshow/waitKey directly",
        )

    def test_windows_skips_backend_that_opens_but_only_returns_black_frames(self) -> None:
        captures: list[tuple[int, int, _FakeCapture]] = []

        def fake_video_capture(index: int, backend: int) -> _FakeCapture:
            if backend == 11:
                cap = _FakeCapture(
                    opened=True,
                    read_results=[
                        (True, [[0, 0], [0, 0]]),
                        (True, [[0, 0], [0, 0]]),
                        (True, [[0, 0], [0, 0]]),
                    ],
                )
            elif backend == 22:
                cap = _FakeCapture(opened=True, read_results=[(True, [[0, 1], [2, 3]])])
            else:
                cap = _FakeCapture(opened=False)
            captures.append((index, backend, cap))
            return cap

        with (
            patch.object(vm_rep_simulator.sys, "platform", "win32"),
            patch.object(vm_rep_simulator.cv2, "VideoCapture", side_effect=fake_video_capture),
            patch.object(vm_rep_simulator.cv2, "CAP_DSHOW", 11, create=True),
            patch.object(vm_rep_simulator.cv2, "CAP_MSMF", 22, create=True),
            patch.object(vm_rep_simulator.cv2, "CAP_ANY", 33),
        ):
            cap, backend_name = vm_rep_simulator._open_webcam_capture(camera_index=1)

        self.assertEqual("msmf", backend_name)
        self.assertEqual([(1, 11), (1, 22)], [(idx, backend) for idx, backend, _ in captures])
        self.assertTrue(captures[0][2].released)
        self.assertFalse(cap.released)


if __name__ == "__main__":
    unittest.main()
