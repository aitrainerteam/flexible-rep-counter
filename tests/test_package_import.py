"""Package must import without VM env or OpenCV."""
from __future__ import annotations

import os


def test_import_without_vm_env(monkeypatch):
    monkeypatch.delenv("YOLO_VM_DIRECT_URL", raising=False)
    monkeypatch.delenv("YOLO_VM_TARGET_URL", raising=False)
    from flexible_rep_counter import RepCounterSession, StepResult, keypoints_numpy_to_landmarks

    assert RepCounterSession is not None
    assert StepResult is not None
    assert keypoints_numpy_to_landmarks is not None


def test_numpy_keypoints_conversion():
    import numpy as np

    from flexible_rep_counter import keypoints_numpy_to_landmarks

    arr = np.zeros((17, 3), dtype=float)
    arr[:, 2] = 1.0
    lm = keypoints_numpy_to_landmarks(arr)
    assert lm is not None
    assert len(lm) == 17
    assert lm[0]["confidence"] == 1.0
