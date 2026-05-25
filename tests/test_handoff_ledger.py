from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from flexible_rep_counter.session import (
    JOINT_SWITCH_SUPPRESS_RETROACTIVE_STEPS,
    _activate_joint_switch,
    _apply_monotonic_shown_floor,
    _clear_pending_switch,
    _effective_min_peak_distance,
    _ledger_cumulative_reps,
    _mark_joint_activation_guard,
    _retroactive_credit_eligible,
)


class _HalfCycleDetector:
    def __init__(self, rep_count: int) -> None:
        self.rep_count = int(rep_count)
        self.peaks = [80.0, 82.0]
        self.valleys = [30.0]

    def get_rep_count(self) -> int:
        return self.rep_count


def _run_state(*, offset: int = 0, step: int = 5) -> dict[str, Any]:
    rs: dict[str, Any] = {
        "rep_count_offset": offset,
        "rep_count_raw_offset": 0,
        "tracking_step_count": step,
        "suppress_retroactive_credit_until_step": 0,
    }
    _clear_pending_switch(rs)
    return rs


def test_ledger_ignores_half_cycle_for_handoff_cumulative() -> None:
    detector = _HalfCycleDetector(rep_count=1)
    rs = _run_state(offset=0)
    shown, raw = _ledger_cumulative_reps(rs, detector, include_retroactive=False)
    assert shown == 1
    assert raw == 1


def test_same_exercise_handoff_preserves_display_without_half_cycle_inflation() -> None:
    incumbent = _HalfCycleDetector(rep_count=1)
    candidate = _HalfCycleDetector(rep_count=1)
    rs = _run_state()
    cumulative_shown, cumulative_raw = _ledger_cumulative_reps(
        rs, incumbent, include_retroactive=False
    )
    assert cumulative_shown == 1
    assert cumulative_raw == 1

    detectors: dict[str, Any] = {}
    _activate_joint_switch(
        rs,
        detectors,
        new_angle="RIGHT_HIP",
        detector=candidate,
        cumulative_shown=cumulative_shown,
        cumulative_raw=cumulative_raw,
        switched_at=1.0,
    )

    assert rs["pending_switch_handoff_kind"] in ("same_exercise", "ambiguous")
    assert int(rs["rep_count_offset"]) == 0
    display = candidate.rep_count + int(rs["rep_count_offset"])
    assert display == 1


def test_retroactive_credit_suppressed_after_joint_activation() -> None:
    detector = _HalfCycleDetector(rep_count=1)
    rs = _run_state(step=10)
    _mark_joint_activation_guard(rs)
    assert _retroactive_credit_eligible(detector, rs) is False
    assert int(rs["suppress_retroactive_credit_until_step"]) == (
        10 + JOINT_SWITCH_SUPPRESS_RETROACTIVE_STEPS
    )


def test_handoff_never_decreases_when_display_includes_retroactive_credit() -> None:
    incumbent = _HalfCycleDetector(rep_count=9)
    candidate = _HalfCycleDetector(rep_count=3)
    rs = _run_state(offset=0)
    rs.update(
        {
            "pending_switch_mirrored_pair": True,
            "pending_switch_same_joint_family": True,
            "pending_switch_incumbent_motion_span_deg": 4.0,
            "pending_switch_candidate_pending_rom_estimate_deg": 42.0,
            "pending_switch_incumbent_observable_during_pending": True,
            "pending_switch_incumbent_completed_gated_cycle_during_pending": False,
            "pending_switch_incumbent_pose_score_at_start": 0.8,
            "pending_switch_incumbent_last_observed_ts_at_start": 1200,
            "pending_switch_started_at_ms": 1800,
            "pending_switch_candidate_completed_cycles_at_start": 3,
            "pending_switch_candidate_rom_score_at_start": 0.8,
            "pending_switch_incumbent_cycles_last_4s": 3,
            "pending_switch_candidate_cycles_last_4s": 3,
            "pending_switch_cycle_sync_score_last_4s": 0.85,
            "pending_switch_candidate_carryover_start_raw": 3,
            "pending_switch_candidate_carryover_start_shown": 3,
        }
    )
    cumulative_shown, cumulative_raw = _ledger_cumulative_reps(
        rs, incumbent, include_retroactive=True
    )
    assert cumulative_shown == 10
    assert cumulative_raw == 9

    detectors: dict[str, Any] = {}
    _activate_joint_switch(
        rs,
        detectors,
        new_angle="LEFT_ELBOW",
        detector=candidate,
        cumulative_shown=cumulative_shown,
        cumulative_raw=cumulative_raw,
        switched_at=1.0,
    )

    assert rs["pending_switch_handoff_kind"] == "alternate_limb"
    display = candidate.rep_count + int(rs["rep_count_offset"])
    assert display >= cumulative_shown
    assert display == 13


def test_effective_min_peak_distance_scales_with_effective_fps() -> None:
    rs = {"low_fps_effective_fps": 6.0}
    assert _effective_min_peak_distance(4, rs) == 8
    rs_fast = {"low_fps_effective_fps": 24.0}
    assert _effective_min_peak_distance(4, rs_fast) == 2


def test_monotonic_shown_floor_prevents_display_snap_back() -> None:
    rs = _run_state(offset=0)
    assert _apply_monotonic_shown_floor(rs, 8) == 8
    assert int(rs["rep_count_shown_floor"]) == 8
    assert _apply_monotonic_shown_floor(rs, 7) == 8

    detector = _HalfCycleDetector(rep_count=7)
    shown, raw = _ledger_cumulative_reps(rs, detector, include_retroactive=False)
    assert shown == 8
    assert raw == 7


def test_monotonic_shown_floor_resets_with_ledger_guard() -> None:
    rs = _run_state()
    _apply_monotonic_shown_floor(rs, 5)
    from flexible_rep_counter.session import _init_ledger_guard_state

    _init_ledger_guard_state(rs)
    assert int(rs["rep_count_shown_floor"]) == 0
    assert _apply_monotonic_shown_floor(rs, 2) == 2
