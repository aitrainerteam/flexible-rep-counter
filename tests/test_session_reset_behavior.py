from __future__ import annotations

from flexible_rep_counter.core.variance_angle_selector import COMMON_ANGLES
from flexible_rep_counter.session import RepCounterSession


def test_clear_tracking_keep_started_preserves_started_and_rebuilds_tracking_state() -> None:
    session = RepCounterSession(auto_started=False, use_pose_filter=False)
    session.set_started(123.0)
    rs = session._run_state

    rs["selected_angle"] = "LEFT_ELBOW"
    rs["selection_dominance_key"] = "LEFT_ELBOW"
    rs["selection_dominance_streak"] = 6
    rs["rep_count_offset"] = 9
    rs["rep_count_raw_offset"] = 7
    rs["pending_switch_angle"] = "RIGHT_ELBOW"
    rs["low_fps_mode_active"] = True
    rs["low_fps_mode_changed_pulse"] = True
    rs["frame_buffer"].append([{"x": 0.0, "y": 0.0, "confidence": 1.0}] * 17)

    session.clear_tracking_keep_started()

    rs = session._run_state
    assert session.started is True
    assert rs["started"] is True
    assert rs["started_at"] == 123.0
    assert rs["selected_angle"] is None
    assert rs["peak_detector"] is None
    assert len(rs["frame_buffer"]) == 0
    assert rs["selection_dominance_key"] is None
    assert rs["selection_dominance_streak"] == 0
    assert rs["rep_count_offset"] == 0
    assert rs["rep_count_raw_offset"] == 0
    assert rs["pending_switch_angle"] is None
    assert rs["low_fps_mode_active"] is False
    assert rs["low_fps_mode_changed_pulse"] is False

    state_keys = set(rs["joint_motion_states"].keys())
    detector_keys = set(rs["selection_detectors_by_angle"].keys())
    assert state_keys == set(COMMON_ANGLES.keys())
    assert detector_keys == set(COMMON_ANGLES.keys())
