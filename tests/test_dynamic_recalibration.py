from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from app import vm_client_2
import flexible_rep_counter.session as session_mod
from flexible_rep_counter.core.variance_angle_selector import COMMON_ANGLES
from flexible_rep_counter.session import RepCounterSession, _peak_detector_from_tuning


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _point(x: float, y: float, conf: float = 0.99) -> dict[str, float]:
    return {"x": x, "y": y, "confidence": conf}


def _build_pose_frame(left_elbow_deg: float = 120.0, right_elbow_deg: float = 120.0) -> list[dict]:
    left_shoulder = (0.32, 0.40)
    left_elbow = (0.32, 0.56)
    right_shoulder = (0.68, 0.40)
    right_elbow = (0.68, 0.56)
    length = 0.18

    left_th = math.radians(left_elbow_deg)
    right_th = math.radians(right_elbow_deg)
    left_wrist = (
        left_elbow[0] + length * math.cos(left_th),
        left_elbow[1] - length * math.sin(left_th),
    )
    right_wrist = (
        right_elbow[0] + length * math.cos(right_th),
        right_elbow[1] - length * math.sin(right_th),
    )

    points: dict[str, dict[str, float]] = {
        name: _point(0.0, 0.0, 0.0) for name in COCO_KEYPOINT_NAMES
    }
    defaults = {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_hip": (0.44, 0.72),
        "right_hip": (0.56, 0.72),
        "left_knee": (0.44, 0.90),
        "right_knee": (0.56, 0.90),
        "left_ankle": (0.44, 1.05),
        "right_ankle": (0.56, 1.05),
    }
    for key, (x, y) in defaults.items():
        points[key] = _point(x, y, 0.99)
    return [points[name] for name in COCO_KEYPOINT_NAMES]


class _FakeFrame:
    shape = (120, 160, 3)


class _FakeBuffer:
    def any(self) -> bool:
        return True

    def tobytes(self) -> bytes:
        return b"jpeg"


class _FakeResponse:
    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._body

    @property
    def text(self) -> str:
        return str(self._body)


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return self._response


def test_send_frame_preserves_rep_payload_when_landmarks_fail_parse(monkeypatch) -> None:
    body = {
        "inference_ms": 12.3,
        "rep_counter": {
            "reps": 7,
            "tracked_joint": "RIGHT_ELBOW",
            "tracked_joint_changed": True,
        },
        "rep_session_reset": {"requested": True, "had_session": False},
    }
    fake_session = _FakeSession(_FakeResponse(body))
    monkeypatch.setattr(vm_client_2.cv2, "imencode", lambda *_args, **_kwargs: (True, _FakeBuffer()))

    outcome = vm_client_2.send_frame(
        _FakeFrame(),
        session=fake_session,
        validate=False,
        parse_rep_counter=True,
    )

    assert outcome.landmarks is None
    assert outcome.rep_counter == body["rep_counter"]
    assert outcome.rep_session_reset == body["rep_session_reset"]


def test_tracking_keeps_all_angle_histories_fresh() -> None:
    session = RepCounterSession(auto_started=True, use_pose_filter=False)
    rs = session._run_state
    rs["selected_angle"] = "RIGHT_ELBOW"
    rs["selected_config"] = COMMON_ANGLES["RIGHT_ELBOW"]
    rs["peak_detector"] = _peak_detector_from_tuning(rs["tuning_params"])

    _ = session.step_landmarks(_build_pose_frame(), timestamp_ms=0.0, wall_time_s=0.0)
    histories = rs["selection_angle_histories"]

    assert set(histories.keys()) == set(COMMON_ANGLES.keys())
    assert all(len(hist) == 1 for hist in histories.values())


def test_switch_waits_for_candidate_calibration_and_keeps_reps_monotonic(monkeypatch) -> None:
    session = RepCounterSession(auto_started=True, use_pose_filter=False)
    rs = session._run_state
    rs["selected_angle"] = "RIGHT_ELBOW"
    rs["selected_config"] = COMMON_ANGLES["RIGHT_ELBOW"]
    current_detector = _peak_detector_from_tuning(rs["tuning_params"])
    current_detector.rep_count = 8
    rs["peak_detector"] = current_detector
    rs["selection_detectors_by_angle"] = {"RIGHT_ELBOW": current_detector}
    histories = rs["selection_angle_histories"]
    for key in COMMON_ANGLES:
        histories[key] = deque(maxlen=400)
    for i in range(24):
        histories["LEFT_ELBOW"].append(float(45 + (i % 6) * 18))

    def _fake_get_variances(_self, _rs, _frame_buffer, include_debug=False):
        return {
            "RIGHT_ELBOW": {"medianWindowVariance": 1.0},
            "LEFT_ELBOW": {"medianWindowVariance": 3.0},
        }

    monkeypatch.setattr(
        session_mod.RepCounterSession,
        "_get_variances",
        _fake_get_variances,
    )
    monkeypatch.setattr(
        session_mod,
        "determine_best_angle",
        lambda *_args, **_kwargs: {
            "selectedAngle": "LEFT_ELBOW",
            "source": "variance",
            "tuningParams": rs["tuning_params"],
        },
    )
    monkeypatch.setattr(
        session_mod,
        "passes_consistent_variance_gate",
        lambda _variances, _angle_key: False,
    )

    class _PendingDetector:
        def __init__(self) -> None:
            self.rep_count = 0
            self.peaks: list[float] = []
            self.valleys: list[float] = []
            self._updates = 0

        def update(self, _value: float | None) -> dict[str, Any]:
            self._updates += 1
            calibrated = self._updates >= 2
            return {
                "repCount": self.rep_count,
                "state": "GOING_DOWN",
                "smoothedValue": _value,
                "rangeGateOpen": True,
                "rollingRange": 70.0,
                "calibrationComplete": calibrated,
                "calibrationTargetReps": 3,
                "calibrationCertainty": 1.0 if calibrated else 0.2,
                "calibrationCertaintyTarget": 0.8,
            }

        def get_state(self) -> dict[str, Any]:
            return {
                "calibratedAvgPeak": None,
                "calibratedAvgValley": None,
                "calibrationComplete": self._updates >= 2,
            }

        def get_rep_count(self) -> int:
            return self.rep_count

    monkeypatch.setattr(
        session_mod,
        "_rebuild_detector_from_history",
        lambda *_args, **_kwargs: _PendingDetector(),
    )

    out1 = session.step_landmarks(
        _build_pose_frame(left_elbow_deg=65.0, right_elbow_deg=120.0),
        timestamp_ms=0.0,
        wall_time_s=0.0,
    )
    out2 = session.step_landmarks(
        _build_pose_frame(left_elbow_deg=90.0, right_elbow_deg=120.0),
        timestamp_ms=1000.0,
        wall_time_s=1.0,
    )
    out3 = session.step_landmarks(
        _build_pose_frame(left_elbow_deg=110.0, right_elbow_deg=120.0),
        timestamp_ms=2000.0,
        wall_time_s=2.0,
    )

    assert out1.tracked_joint == "RIGHT_ELBOW"
    assert out1.tracked_joint_changed is False
    assert out1.reps >= 8
    assert out2.tracked_joint == "RIGHT_ELBOW"
    assert out2.tracked_joint_changed is False
    assert out2.reps >= out1.reps
    assert out3.tracked_joint == "LEFT_ELBOW"
    assert out3.tracked_joint_changed is True
    assert out3.reps >= out2.reps
