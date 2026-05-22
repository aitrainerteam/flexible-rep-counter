from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from flexible_rep_counter.core.math_engine import PeakDetector
from flexible_rep_counter.core.recalibration_confidence import (
    HandoffDecision,
    JointMotionState,
    classify_handoff,
    compute_joint_recalibration_score,
    should_switch_to_candidate,
    update_joint_motion_state,
)


class _ScriptedDetector:
    def __init__(
        self,
        rep_counts: list[int],
        *,
        peaks: list[float] | None = None,
        valleys: list[float] | None = None,
        rolling_range_deg: float = 40.0,
    ) -> None:
        self._rep_counts = list(rep_counts)
        self._idx = 0
        self.rep_count = int(self._rep_counts[0] if self._rep_counts else 0)
        self.peaks = list(peaks or [80.0, 82.0, 81.0])
        self.valleys = list(valleys or [30.0, 32.0, 31.0])
        self._rolling_range_deg = rolling_range_deg

    def update(self, value: float | None) -> dict[str, Any]:
        i = min(self._idx, len(self._rep_counts) - 1)
        self.rep_count = int(self._rep_counts[i] if self._rep_counts else 0)
        self._idx += 1
        return {
            "repCount": self.rep_count,
            "peak": self.peaks[-1] if self.peaks else None,
            "valley": self.valleys[-1] if self.valleys else None,
            "rollingRange": self._rolling_range_deg,
            "rangeGateOpen": True,
            "calibrationComplete": True,
            "smoothedValue": value,
            "state": "GOING_DOWN",
            "calibrationCertainty": 1.0,
            "calibrationTargetReps": 3,
            "calibrationCertaintyTarget": 0.8,
        }

    def get_rep_count(self) -> int:
        return self.rep_count


def _make_state(
    angle_key: str,
    detector: _ScriptedDetector | None = None,
    history_values: list[float] | None = None,
    confidence: float = 0.99,
) -> JointMotionState:
    state = JointMotionState(
        angle_key=angle_key,
        detector=cast(PeakDetector, detector or _ScriptedDetector([0])),
        history=deque(maxlen=400),
        confidence_history=deque(maxlen=400),
    )
    for i, value in enumerate(history_values or []):
        state.history.append(float(value))
        state.confidence_history.append((i * 33, confidence))
    if history_values:
        state.last_observed_timestamp_ms = len(history_values) * 33
    return state


def _activation_targets(
    decision: HandoffDecision,
    *,
    cumulative_shown: int,
    cumulative_raw: int,
    candidate_current_shown: int,
    candidate_current_raw: int,
    carryover_start_shown: int,
    carryover_start_raw: int,
) -> tuple[int, int]:
    candidate_delta_shown = max(0, candidate_current_shown - carryover_start_shown)
    candidate_delta_raw = max(0, candidate_current_raw - carryover_start_raw)
    rationale = decision.rationale if isinstance(decision.rationale, dict) else {}
    if (
        decision.kind == "alternate_limb"
        and rationale.get("rule") == "forced_switch_mirrored_candidate_ready_or_delta"
        and bool(rationale.get("candidateReadyAtStart"))
    ):
        candidate_delta_shown = max(0, candidate_current_shown)
        candidate_delta_raw = max(0, candidate_current_raw)
    if decision.kind == "alternate_limb":
        return cumulative_shown + candidate_delta_shown, cumulative_raw + candidate_delta_raw
    return cumulative_shown, cumulative_raw


def test_alternate_limb_adds_candidate_pending_reps() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 6.0,
        "candidate_pending_rom_estimate_deg": 50.0,
        "incumbent_observable_during_pending": True,
        "incumbent_pose_score_at_start": 0.75,
        "incumbent_last_observed_ts_at_start": 1200,
        "pending_start_ts": 1800,
        "candidate_completed_cycles_at_start": 3,
        "candidate_last_cycle_ts_before_start": 1600,
        "candidate_rom_score_at_start": 0.85,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 1,
        "candidate_cycles_last_4s": 1,
        "cycle_sync_score_last_4s": 0.30,
        "mirrored_pair": True,
        "candidate_advanced_during_pending": True,
        "candidate_completed_gated_cycle_during_pending": True,
        "candidate_current_raw": 8,
        "candidate_carryover_start_raw": 4,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "alternate_limb"
    shown, raw = _activation_targets(
        decision,
        cumulative_shown=10,
        cumulative_raw=10,
        candidate_current_shown=8,
        candidate_current_raw=8,
        carryover_start_shown=4,
        carryover_start_raw=4,
    )
    assert shown == 14
    assert raw == 14


def test_forced_mirrored_handoff_with_candidate_delta_is_alternate_limb() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 22.0,
        "candidate_pending_rom_estimate_deg": 42.0,
        "incumbent_observable_during_pending": True,
        "incumbent_pose_score_at_start": 0.80,
        "incumbent_last_observed_ts_at_start": 1200,
        "pending_start_ts": 1800,
        "candidate_completed_cycles_at_start": 3,
        "candidate_last_cycle_ts_before_start": 1600,
        "candidate_rom_score_at_start": 0.80,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 1,
        "candidate_cycles_last_4s": 1,
        "cycle_sync_score_last_4s": 0.25,
        "mirrored_pair": True,
        "switch_forced": True,
        "candidate_advanced_during_pending": True,
        "candidate_completed_gated_cycle_during_pending": True,
        "candidate_current_raw": 9,
        "candidate_carryover_start_raw": 5,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "alternate_limb"
    shown, raw = _activation_targets(
        decision,
        cumulative_shown=10,
        cumulative_raw=10,
        candidate_current_shown=9,
        candidate_current_raw=9,
        carryover_start_shown=5,
        carryover_start_raw=5,
    )
    assert shown == 19
    assert raw == 19


def test_forced_mirrored_handoff_carries_full_ready_candidate_when_delta_zero() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 10.0,
        "candidate_pending_rom_estimate_deg": 45.0,
        "incumbent_observable_during_pending": True,
        "incumbent_pose_score_at_start": 0.80,
        "incumbent_last_observed_ts_at_start": 1200,
        "pending_start_ts": 1800,
        "candidate_completed_cycles_at_start": 3,
        "candidate_last_cycle_ts_before_start": 1600,
        "candidate_rom_score_at_start": 0.80,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 1,
        "candidate_cycles_last_4s": 1,
        "cycle_sync_score_last_4s": 0.25,
        "mirrored_pair": True,
        "switch_forced": True,
        "candidate_advanced_during_pending": False,
        "candidate_completed_gated_cycle_during_pending": False,
        "candidate_current_raw": 9,
        "candidate_carryover_start_raw": 9,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "alternate_limb"
    shown, raw = _activation_targets(
        decision,
        cumulative_shown=10,
        cumulative_raw=10,
        candidate_current_shown=9,
        candidate_current_raw=9,
        carryover_start_shown=9,
        carryover_start_raw=9,
    )
    assert shown == 19
    assert raw == 19


def test_incumbent_disappears_without_sync_is_ambiguous() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 0.0,
        "candidate_pending_rom_estimate_deg": 45.0,
        "incumbent_observable_during_pending": False,
        "incumbent_pose_score_at_start": 0.10,
        "incumbent_last_observed_ts_at_start": 1000,
        "pending_start_ts": 1800,
        "candidate_completed_cycles_at_start": 3,
        "candidate_last_cycle_ts_before_start": 1600,
        "candidate_rom_score_at_start": 0.80,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 0,
        "candidate_cycles_last_4s": 1,
        "cycle_sync_score_last_4s": 0.20,
        "mirrored_pair": True,
        "candidate_advanced_during_pending": False,
        "candidate_completed_gated_cycle_during_pending": False,
        "candidate_current_raw": 8,
        "candidate_carryover_start_raw": 4,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "ambiguous"
    shown, raw = _activation_targets(
        decision,
        cumulative_shown=10,
        cumulative_raw=10,
        candidate_current_shown=8,
        candidate_current_raw=8,
        carryover_start_shown=4,
        carryover_start_raw=4,
    )
    assert shown == 10
    assert raw == 10


def test_same_exercise_better_joint_continues_linearly() -> None:
    pending = {
        "incumbent_advanced": True,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 20.0,
        "candidate_pending_rom_estimate_deg": 45.0,
        "incumbent_observable_during_pending": True,
        "pending_start_ts": 0,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "same_exercise"
    shown, _ = _activation_targets(
        decision,
        cumulative_shown=10,
        cumulative_raw=10,
        candidate_current_shown=14,
        candidate_current_raw=14,
        carryover_start_shown=9,
        carryover_start_raw=9,
    )
    assert shown == 10


def test_fov_disappearance_with_prior_candidate_cycles_continues_linearly() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 0.0,
        "candidate_pending_rom_estimate_deg": 42.0,
        "incumbent_observable_during_pending": False,
        "incumbent_pose_score_at_start": 0.10,
        "incumbent_last_observed_ts_at_start": 1000,
        "pending_start_ts": 1900,
        "candidate_completed_cycles_at_start": 4,
        "candidate_last_cycle_ts_before_start": 1700,
        "candidate_rom_score_at_start": 0.70,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 2,
        "candidate_cycles_last_4s": 3,
        "cycle_sync_score_last_4s": 0.75,
        "mirrored_pair": True,
        "candidate_advanced_during_pending": True,
        "candidate_completed_gated_cycle_during_pending": True,
        "candidate_current_raw": 10,
        "candidate_carryover_start_raw": 7,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "same_exercise"


def test_fov_disappearance_without_prior_candidate_cycles_is_ambiguous() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 0.0,
        "candidate_pending_rom_estimate_deg": 35.0,
        "incumbent_observable_during_pending": False,
        "incumbent_pose_score_at_start": 0.15,
        "incumbent_last_observed_ts_at_start": 1000,
        "pending_start_ts": 1900,
        "candidate_completed_cycles_at_start": 0,
        "candidate_last_cycle_ts_before_start": None,
        "candidate_rom_score_at_start": 0.0,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 0,
        "candidate_cycles_last_4s": 0,
        "cycle_sync_score_last_4s": 0.0,
        "mirrored_pair": True,
        "candidate_advanced_during_pending": True,
        "candidate_completed_gated_cycle_during_pending": False,
        "candidate_current_raw": 13,
        "candidate_carryover_start_raw": 10,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "ambiguous"
    shown, _ = _activation_targets(
        decision,
        cumulative_shown=10,
        cumulative_raw=10,
        candidate_current_shown=13,
        candidate_current_raw=13,
        carryover_start_shown=10,
        carryover_start_raw=10,
    )
    assert shown == 10


def test_cycle_gating_rejects_low_range_cycles() -> None:
    detector = _ScriptedDetector([0, 1, 1], rolling_range_deg=30.0)
    state = _make_state(
        "LEFT_ELBOW",
        detector=detector,
        history_values=[100.0, 101.0, 102.0, 101.5, 100.5, 101.0] * 8,
    )
    for i in range(3):
        update_joint_motion_state(state, 101.0 + i * 0.05, 0.99, 1000 + i * 33)
    assert len(state.recent_roms) == 0
    score, debug = compute_joint_recalibration_score(state, {"medianWindowVariance": 10.0}, now_ms=1200)
    assert debug["completedCycles"] == 0
    assert score < 0.62


def test_force_stale_requires_two_gated_cycles() -> None:
    can_switch, force_switch, _ = should_switch_to_candidate(
        cooldown_ok=True,
        stale_reevals=8,
        stale_switch_force_after_reevals=8,
        selected_recent_range=4.0,
        stale_switch_max_selected_recent_range_deg=14.0,
        selected_range_gate_closed_streak=10,
        stale_switch_min_closed_streak=10,
        selected_score=0.30,
        selected_pose_score=0.9,
        candidate_score=0.60,
        candidate_activity_score=0.55,
        candidate_pose_score=0.60,
        candidate_observable=True,
        candidate_completed_cycles=1,
    )
    assert not can_switch
    assert not force_switch

    _, force_switch_after_two, _ = should_switch_to_candidate(
        cooldown_ok=True,
        stale_reevals=8,
        stale_switch_force_after_reevals=8,
        selected_recent_range=4.0,
        stale_switch_max_selected_recent_range_deg=14.0,
        selected_range_gate_closed_streak=10,
        stale_switch_min_closed_streak=10,
        selected_score=0.30,
        selected_pose_score=0.9,
        candidate_score=0.60,
        candidate_activity_score=0.55,
        candidate_pose_score=0.60,
        candidate_observable=True,
        candidate_completed_cycles=2,
    )
    assert force_switch_after_two


def test_no_displayed_count_regression_on_any_handoff() -> None:
    before = 10
    for kind in ("alternate_limb", "same_exercise", "ambiguous"):
        decision = HandoffDecision(kind=kind, rationale={})
        shown, _ = _activation_targets(
            decision,
            cumulative_shown=before,
            cumulative_raw=before,
            candidate_current_shown=13,
            candidate_current_raw=13,
            carryover_start_shown=9,
            carryover_start_raw=9,
        )
        assert shown >= before


def test_same_exercise_never_jumps_forward_on_activation() -> None:
    for kind in ("same_exercise", "ambiguous"):
        decision = HandoffDecision(kind=kind, rationale={})
        shown, _ = _activation_targets(
            decision,
            cumulative_shown=10,
            cumulative_raw=10,
            candidate_current_shown=22,
            candidate_current_raw=22,
            carryover_start_shown=12,
            carryover_start_raw=12,
        )
        assert shown == 10


def test_cross_family_switch_cannot_be_alternate_limb() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 2.0,
        "candidate_pending_rom_estimate_deg": 50.0,
        "incumbent_observable_during_pending": True,
        "pending_start_ts": 1900,
        "candidate_completed_cycles_at_start": 3,
        "candidate_last_cycle_ts_before_start": 1750,
        "candidate_rom_score_at_start": 0.8,
        "same_joint_family": False,
        "incumbent_cycles_last_4s": 1,
        "candidate_cycles_last_4s": 2,
        "cycle_sync_score_last_4s": 0.2,
        "mirrored_pair": False,
        "candidate_advanced_during_pending": True,
        "candidate_completed_gated_cycle_during_pending": True,
        "candidate_current_raw": 9,
        "candidate_carryover_start_raw": 7,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "ambiguous"


def test_same_exercise_sync_during_pause_does_not_become_alternate_limb() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 2.0,
        "candidate_pending_rom_estimate_deg": 45.0,
        "incumbent_observable_during_pending": True,
        "incumbent_pose_score_at_start": 0.80,
        "pending_start_ts": 2000,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 3,
        "candidate_cycles_last_4s": 3,
        "cycle_sync_score_last_4s": 0.85,
        "mirrored_pair": True,
        "candidate_current_raw": 12,
        "candidate_carryover_start_raw": 10,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "same_exercise"


def test_alternate_limb_uses_observation_start_not_pending_start() -> None:
    pending = {
        "incumbent_advanced": False,
        "incumbent_completed_gated_cycle_during_pending": False,
        "incumbent_motion_span_deg": 4.0,
        "candidate_pending_rom_estimate_deg": 50.0,
        "incumbent_observable_during_pending": True,
        "incumbent_pose_score_at_start": 0.70,
        "incumbent_last_observed_ts_at_start": 1500,
        "pending_start_ts": 1900,
        "candidate_completed_cycles_at_start": 3,
        "candidate_last_cycle_ts_before_start": 1700,
        "candidate_rom_score_at_start": 0.80,
        "same_joint_family": True,
        "incumbent_cycles_last_4s": 1,
        "candidate_cycles_last_4s": 1,
        "cycle_sync_score_last_4s": 0.25,
        "mirrored_pair": True,
        "candidate_advanced_during_pending": False,
        "candidate_completed_gated_cycle_during_pending": False,
        "candidate_current_raw": 9,
        "candidate_carryover_start_raw": 6,
    }
    decision = classify_handoff(pending)
    assert decision.kind == "alternate_limb"
