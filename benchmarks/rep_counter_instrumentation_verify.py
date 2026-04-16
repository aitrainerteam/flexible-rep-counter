#!/usr/bin/env python3
"""
Controlled validation: run a short RepCounterSession with instrumentation enabled,
then assert NDJSON contains required fields (local / CI substitute for VM shoulder test).
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flexible_rep_counter.core.variance_angle_selector import COMMON_ANGLES
from flexible_rep_counter.instrumentation import RepInstrumentationSettings
from flexible_rep_counter.session import RepCounterSession

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


def _frame(theta_deg: float) -> list[dict]:
    sh = (0.7, 0.4)
    el = (0.7, 0.55)
    length = 0.18
    th = math.radians(theta_deg)
    wr = (el[0] + length * math.cos(th), el[1] - length * math.sin(th))

    points = {name: {"x": 0.0, "y": 0.0, "confidence": 0.0} for name in COCO_KEYPOINT_NAMES}
    defaults = {
        "left_shoulder": (0.3, 0.4),
        "left_elbow": (0.3, 0.55),
        "left_wrist": (0.3, 0.73),
        "right_shoulder": sh,
        "right_elbow": el,
        "right_wrist": wr,
        "left_hip": (0.4, 0.72),
        "right_hip": (0.6, 0.72),
        "left_knee": (0.42, 0.9),
        "right_knee": (0.58, 0.9),
        "left_ankle": (0.42, 1.05),
        "right_ankle": (0.58, 1.05),
    }
    for key, (x, y) in defaults.items():
        points[key] = {"x": x, "y": y, "confidence": 0.99}
    return [points[name] for name in COCO_KEYPOINT_NAMES]


def main() -> int:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".ndjson", delete=False) as tmp:
        path = tmp.name

    try:
        settings = RepInstrumentationSettings(
            enabled=True,
            output_path=path,
            sample_every_n_frames=1,
            flush_every_n_records=1,
            user_allowlist=frozenset({"test-user"}),
        )
        session = RepCounterSession(
            auto_started=True,
            use_pose_filter=False,
            instrumentation_settings=settings,
        )
        from flexible_rep_counter.session import _peak_detector_from_tuning

        tp = session._run_state["tuning_params"]
        sdba = {ak: _peak_detector_from_tuning(tp) for ak in COMMON_ANGLES}
        for ak, det in sdba.items():
            try:
                setattr(det, "debug_label", ak)
            except Exception:
                pass
        session._run_state["selection_detectors_by_angle"] = sdba
        session._run_state["selected_angle"] = "RIGHT_ELBOW"
        session._run_state["selected_config"] = COMMON_ANGLES["RIGHT_ELBOW"]
        session._run_state["peak_detector"] = sdba["RIGHT_ELBOW"]
        session._sync_detector_instrumentation_flags()

        for i, ang in enumerate([35, 85, 145, 85, 35, 85, 145]):
            t = i * (1.0 / 30.0)
            session.step_landmarks(
                _frame(ang),
                timestamp_ms=t * 1000.0,
                wall_time_s=t,
                trace_context={
                    "user_uid": "test-user",
                    "session_id": "verify-session",
                    "request_ts_ms": t * 1000.0,
                    "frame_idx": i + 1,
                },
            )

        if session._instr_sink is not None:
            session._instr_sink.flush()

        text = Path(path).read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert lines, "expected at least one NDJSON line"

        required_keys = {"event", "user_uid", "session_id", "request_ts_ms", "frame_idx", "ts_ms"}
        for ln in lines:
            obj = json.loads(ln)
            missing = required_keys - set(obj.keys())
            assert not missing, f"missing keys {missing} in {obj!r}"

        events = {json.loads(ln)["event"] for ln in lines}
        assert "frame_snapshot" in events, f"expected frame_snapshot in {events}"

        print("rep_counter_instrumentation_verify: OK", f"({len(lines)} events)")
        return 0
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
