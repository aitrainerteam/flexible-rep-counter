from __future__ import annotations

import ast
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from visualizer import opencv_runtime


MODULE_PATH = ROOT / "visualizer" / "opencv_runtime.py"


class OpenCvRuntimeDisplayTests(unittest.TestCase):
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

    def test_letterbox_frame_to_canvas_returns_target_shape(self) -> None:
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)

        out = opencv_runtime._letterbox_frame_to_canvas(frame, width=1280, height=960)

        self.assertEqual((960, 1280, 3), out.shape)
        self.assertTrue(np.any(out))


if __name__ == "__main__":
    unittest.main()
