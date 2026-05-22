from __future__ import annotations

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

from flexible_rep_counter.session import (
    LOW_FPS_ENTER_STREAK,
    LOW_FPS_EXIT_STREAK,
    LOW_FPS_INTERVAL_WINDOW_FRAMES,
    _activate_joint_switch,
    _clear_pending_switch,
    _update_low_fps_health,
)


class _StubDetector:
    def __init__(self, rep_count: int) -> None:
        self.rep_count = int(rep_count)
        self.peaks = [80.0, 82.0]
        self.valleys = [30.0, 32.0]

    def get_rep_count(self) -> int:
        return self.rep_count


def _new_health_state() -> dict[str, Any]:
    return {
        "frame_interval_window": deque(maxlen=LOW_FPS_INTERVAL_WINDOW_FRAMES),
        "last_step_monotonic_s": None,
        "low_fps_mode_active": False,
        "low_fps_enter_streak": 0,
        "low_fps_exit_streak": 0,
        "low_fps_mode_changed_pulse": False,
        "low_fps_p50_ms": None,
        "low_fps_p90_ms": None,
        "low_fps_gaps_over_200ms": 0,
        "low_fps_effective_fps": None,
    }


def _drive_intervals(
    run_state: dict[str, Any], monkeypatch: Any, intervals_ms: list[float]
) -> list[bool]:
    now = [1000.0]

    def _fake_monotonic() -> float:
        return now[0]

    monkeypatch.setattr("flexible_rep_counter.session.time.monotonic", _fake_monotonic)
    pulses: list[bool] = []
    _update_low_fps_health(run_state)  # establish baseline
    pulses.append(bool(run_state.get("low_fps_mode_changed_pulse")))
    for delta_ms in intervals_ms:
        now[0] += float(delta_ms) / 1000.0
        _update_low_fps_health(run_state)
        pulses.append(bool(run_state.get("low_fps_mode_changed_pulse")))
    return pulses


def _new_pending_run_state(*, low_fps_mode_active: bool) -> dict[str, Any]:
    rs: dict[str, Any] = {
        "rep_count_offset": 0,
        "rep_count_raw_offset": 0,
        "selection_last_switch_at": None,
        "selected_angle": "RIGHT_ELBOW",
        "selected_config": None,
        "peak_detector": None,
        "low_fps_mode_active": low_fps_mode_active,
    }
    _clear_pending_switch(rs)
    rs.update(
        {
            "pending_switch_incumbent_advanced": False,
            "pending_switch_incumbent_completed_gated_cycle_during_pending": False,
            "pending_switch_incumbent_motion_span_deg": 10.0,
            "pending_switch_candidate_pending_rom_estimate_deg": 45.0,
            "pending_switch_incumbent_observable_during_pending": True,
            "pending_switch_incumbent_pose_score_at_start": 0.8,
            "pending_switch_incumbent_last_observed_ts_at_start": 1200,
            "pending_switch_started_at_ms": 1800,
            "pending_switch_candidate_completed_cycles_at_start": 3,
            "pending_switch_candidate_last_cycle_ts_before_start": 1600,
            "pending_switch_candidate_rom_score_at_start": 0.8,
            "pending_switch_same_joint_family": True,
            "pending_switch_incumbent_cycles_last_4s": 1,
            "pending_switch_candidate_cycles_last_4s": 1,
            "pending_switch_cycle_sync_score_last_4s": 0.2,
            "pending_switch_mirrored_pair": True,
            "pending_switch_candidate_advanced_during_pending": True,
            "pending_switch_candidate_completed_gated_cycle_during_pending": True,
            "pending_switch_candidate_carryover_start_shown": 5,
            "pending_switch_candidate_carryover_start_raw": 5,
            "pending_switch_forced": True,
        }
    )
    return rs


def test_low_fps_mode_enters_on_sustained_slow_intervals(monkeypatch: Any) -> None:
    rs = _new_health_state()
    pulses = _drive_intervals(rs, monkeypatch, [160.0] * 40)
    assert rs["low_fps_mode_active"] is True
    assert any(pulses)
    assert rs["low_fps_p50_ms"] is not None and rs["low_fps_p50_ms"] >= 150.0
    assert rs["low_fps_effective_fps"] is not None and rs["low_fps_effective_fps"] < 8.0


def test_low_fps_mode_exits_after_clean_recovery(monkeypatch: Any) -> None:
    rs = _new_health_state()
    _drive_intervals(rs, monkeypatch, [160.0] * (LOW_FPS_ENTER_STREAK + 20))
    assert rs["low_fps_mode_active"] is True
    pulses = _drive_intervals(rs, monkeypatch, [35.0] * (LOW_FPS_EXIT_STREAK + 20))
    assert rs["low_fps_mode_active"] is False
    assert any(pulses)


def test_low_fps_mode_enters_on_spike_heavy_stream(monkeypatch: Any) -> None:
    rs = _new_health_state()
    pattern = [30.0] * 5 + [300.0] + [30.0] * 4 + [300.0]
    _drive_intervals(rs, monkeypatch, pattern * 3)
    assert rs["low_fps_mode_active"] is True
    assert int(rs["low_fps_gaps_over_200ms"]) >= 2


def test_low_fps_mode_does_not_flap_on_borderline_stream(monkeypatch: Any) -> None:
    rs = _new_health_state()
    jitter = [65.0, 70.0, 75.0, 68.0, 72.0, 74.0] * 20
    pulses = _drive_intervals(rs, monkeypatch, jitter)
    assert rs["low_fps_mode_active"] is False
    assert not any(pulses)


def test_alternate_limb_carryover_is_suppressed_in_safe_mode() -> None:
    detectors: dict[str, Any] = {}
    detector = _StubDetector(rep_count=9)
    run_state = _new_pending_run_state(low_fps_mode_active=True)
    _activate_joint_switch(
        run_state,
        detectors,
        new_angle="LEFT_ELBOW",
        detector=detector,
        cumulative_shown=10,
        cumulative_raw=10,
        switched_at=1.0,
    )
    assert run_state["pending_switch_handoff_kind"] == "same_exercise"
    rationale = run_state["pending_switch_handoff_rationale"]
    assert isinstance(rationale, dict)
    assert rationale.get("low_fps_safeguard") is True
    assert rationale.get("original_kind") == "alternate_limb"
    assert int(run_state["pending_switch_handoff_target_shown"]) == 10
    assert int(run_state["rep_count_offset"]) == 1


def test_alternate_limb_carryover_remains_when_safe_mode_inactive() -> None:
    detectors: dict[str, Any] = {}
    detector = _StubDetector(rep_count=9)
    run_state = _new_pending_run_state(low_fps_mode_active=False)
    _activate_joint_switch(
        run_state,
        detectors,
        new_angle="LEFT_ELBOW",
        detector=detector,
        cumulative_shown=10,
        cumulative_raw=10,
        switched_at=1.0,
    )
    assert run_state["pending_switch_handoff_kind"] == "alternate_limb"
    assert int(run_state["pending_switch_handoff_target_shown"]) == 19
    assert int(run_state["rep_count_offset"]) == 10
