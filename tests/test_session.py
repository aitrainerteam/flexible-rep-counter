"""RepCounterSession smoke tests (no VM)."""
from __future__ import annotations

import time

import numpy as np

from flexible_rep_counter import RepCounterSession, keypoints_numpy_to_landmarks


def _neutral_pose() -> list[dict]:
    """Synthetic upright-ish pose (not necessarily geometrically valid)."""
    pts = []
    for i in range(17):
        pts.append({"x": 100.0 + i * 2, "y": 200.0 + i, "confidence": 1.0})
    return pts


def test_session_idle_before_started():
    s = RepCounterSession(auto_started=False)
    r = s.step_landmarks(_neutral_pose())
    assert r.phase == "idle"


def test_session_auto_started_runs_selection():
    s = RepCounterSession(auto_started=True, use_pose_filter=False)
    for _ in range(5):
        s.step_landmarks(_neutral_pose(), timestamp_ms=time.time() * 1000.0)
    r = s.step_landmarks(_neutral_pose())
    assert r.phase in ("selecting", "tracking")


def test_session_accepts_numpy_keypoints():
    s = RepCounterSession(auto_started=True, use_pose_filter=False)
    arr = np.random.rand(17, 3).astype(float) * 100
    arr[:, 2] = 0.99
    lm = keypoints_numpy_to_landmarks(arr)
    assert lm is not None
    r = s.step_landmarks(lm)
    assert r.reps >= 0
