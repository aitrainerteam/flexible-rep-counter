from __future__ import annotations

import sys
from pathlib import Path
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

from visualizer import opencv_runtime


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


class OpenCvRuntimeCameraOpeningTests(unittest.TestCase):
    def test_windows_falls_back_to_second_backend_and_releases_failed_capture(self) -> None:
        captures: list[tuple[int, int, _FakeCapture]] = []

        def fake_video_capture(index: int, backend: int) -> _FakeCapture:
            opened = backend == 22
            cap = _FakeCapture(opened=opened)
            captures.append((index, backend, cap))
            return cap

        with (
            patch.object(opencv_runtime.sys, "platform", "win32"),
            patch.object(opencv_runtime.cv2, "VideoCapture", side_effect=fake_video_capture),
            patch.object(opencv_runtime.cv2, "CAP_DSHOW", 11, create=True),
            patch.object(opencv_runtime.cv2, "CAP_MSMF", 22, create=True),
            patch.object(opencv_runtime.cv2, "CAP_ANY", 33),
        ):
            cap, backend_name = opencv_runtime._open_webcam_capture(camera_index=2)

        self.assertEqual(backend_name, "msmf")
        self.assertEqual([(2, 11), (2, 22)], [(idx, backend) for idx, backend, _ in captures])
        self.assertTrue(captures[0][2].released)
        self.assertFalse(cap.released)

    def test_open_webcam_capture_raises_with_tried_backends(self) -> None:
        def fake_video_capture(index: int, backend: int) -> _FakeCapture:
            return _FakeCapture(opened=False)

        with (
            patch.object(opencv_runtime.sys, "platform", "win32"),
            patch.object(opencv_runtime.cv2, "VideoCapture", side_effect=fake_video_capture),
            patch.object(opencv_runtime.cv2, "CAP_DSHOW", 11, create=True),
            patch.object(opencv_runtime.cv2, "CAP_MSMF", 22, create=True),
            patch.object(opencv_runtime.cv2, "CAP_ANY", 33),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                opencv_runtime._open_webcam_capture(camera_index=3)

        msg = str(ctx.exception)
        self.assertIn("index 3", msg)
        self.assertIn("dshow", msg)
        self.assertIn("msmf", msg)
        self.assertIn("any", msg)

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
                cap = _FakeCapture(
                    opened=True,
                    read_results=[
                        (True, [[0, 1], [2, 3]]),
                    ],
                )
            else:
                cap = _FakeCapture(opened=False)
            captures.append((index, backend, cap))
            return cap

        with (
            patch.object(opencv_runtime.sys, "platform", "win32"),
            patch.object(opencv_runtime.cv2, "VideoCapture", side_effect=fake_video_capture),
            patch.object(opencv_runtime.cv2, "CAP_DSHOW", 11, create=True),
            patch.object(opencv_runtime.cv2, "CAP_MSMF", 22, create=True),
            patch.object(opencv_runtime.cv2, "CAP_ANY", 33),
        ):
            cap, backend_name = opencv_runtime._open_webcam_capture(camera_index=1)

        self.assertEqual("msmf", backend_name)
        self.assertEqual([(1, 11), (1, 22)], [(idx, backend) for idx, backend, _ in captures])
        self.assertTrue(captures[0][2].released)
        self.assertFalse(cap.released)


if __name__ == "__main__":
    unittest.main()
