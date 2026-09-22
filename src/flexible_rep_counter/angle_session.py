"""Skeleton-agnostic 3D angle session: eight named scalars, no landmark indices."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from flexible_rep_counter.core.math_engine import PeakDetector
from flexible_rep_counter.core.scalar_angle_selector import (
    PRODUCT_ANGLE_KEYS,
    compute_variances_from_histories,
    dominance_lock_ready,
    observed_fraction,
    require_product_samples,
    summarize_rep_dominance,
    top_variance_candidate,
)
from flexible_rep_counter.core.settings_3d import Angle3dConfig, load_3d_config
from flexible_rep_counter.types import AngleSample, AngleStepResult

# Mirrored-limb observation window. Same durations as the 2D pending switch.
_MIRROR_PENDING_MIN_MS = 300.0
_MIRROR_PENDING_MAX_MS = 600.0
_INCUMBENT_ACTIVE_MIN_SPAN_DEG = 18.0
_INCUMBENT_ACTIVE_ROM_RATIO = 0.60
_SYNC_WINDOW_MS = 4000.0
_SYNC_SCORE_MIN = 0.60
_SYNC_MIN_CYCLES = 2
_CYCLE_LOG_LIMIT = 16


def mirrored_partner(angle_key: str) -> Optional[str]:
    """Opposite side of the same joint family. ``left_elbow`` ↔ ``right_elbow``.

    Prefix-based, same rule as 2D ``is_mirrored_pair``. Product keys have no
    ``*_Y`` fallbacks, so those are never a mirrored pair.
    """
    if angle_key.startswith("left_"):
        partner = "right_" + angle_key[len("left_") :]
    elif angle_key.startswith("right_"):
        partner = "left_" + angle_key[len("right_") :]
    else:
        return None
    if partner not in PRODUCT_ANGLE_KEYS or partner == angle_key:
        return None
    return partner


def cycle_sync_score(incumbent_ts: list[float], candidate_ts: list[float]) -> float:
    """How aligned two limbs' recent cycles are. 1.0 is simultaneous.

    ``0.6 * count_ratio + 0.4 * timestamp_proximity``, matching the 2D score.
    Empty either side is 0 (a limb that has been quiet for the whole window).
    """
    if not incumbent_ts or not candidate_ts:
        return 0.0
    count_ratio = min(len(incumbent_ts), len(candidate_ts)) / max(len(incumbent_ts), len(candidate_ts))
    ts_gap = abs(incumbent_ts[-1] - candidate_ts[-1])
    ts_score = 1.0 - max(0.0, min(1.0, ts_gap / 1500.0))
    return 0.6 * count_ratio + 0.4 * ts_score


def mirror_display_offset(
    *,
    incumbent_displayed: int,
    candidate_raw: int,
    candidate_raw_at_new_lock: int,
    credited_raw: int = 0,
) -> int:
    """Offset so ``candidate_raw_now + offset`` is the post-switch display.

    Stopped-incumbent credit adds the candidate's raw reps that are not already
    on the display. On the first handoff ``credited_raw`` is 0, so that is the
    detector's full raw count (selection plus the quiet period). The result is
    clamped so the display cannot fall below ``incumbent_displayed``.
    """
    added = max(0, int(candidate_raw) - int(credited_raw))
    target = int(incumbent_displayed) + added
    target = max(target, int(incumbent_displayed))
    return target - int(candidate_raw_at_new_lock)


def classify_mirrored_handoff(
    *,
    incumbent_motion_span_deg: float,
    candidate_rom_deg: float,
    incumbent_completed_gated_cycle: bool,
    candidate_raw: int,
    candidate_active: bool,
    cycle_sync_score_last_4s: float,
    incumbent_cycles_last_4s: int,
    candidate_cycles_last_4s: int,
) -> tuple[str, str]:
    """Classify a mirrored-partner window. Returns ``(kind, rule)``.

    Only the stopped-mirror case adds reps. Sync and a still-moving incumbent
    add nothing. There is no cross-family or fallback-angle branch.
    """
    threshold = max(
        _INCUMBENT_ACTIVE_MIN_SPAN_DEG,
        _INCUMBENT_ACTIVE_ROM_RATIO * float(candidate_rom_deg),
    )
    incumbent_stopped = (
        not incumbent_completed_gated_cycle and float(incumbent_motion_span_deg) < threshold
    )
    if incumbent_stopped and (candidate_active or int(candidate_raw) > 0):
        return "alternate_limb", "mirrored_incumbent_stopped"
    synchronized = (
        int(incumbent_cycles_last_4s) >= _SYNC_MIN_CYCLES
        and int(candidate_cycles_last_4s) >= _SYNC_MIN_CYCLES
        and float(cycle_sync_score_last_4s) >= _SYNC_SCORE_MIN
        and not incumbent_stopped
    )
    if synchronized:
        return "same_exercise", "prior_synchronized_same_exercise"
    if incumbent_completed_gated_cycle or float(incumbent_motion_span_deg) >= threshold:
        return "same_exercise", "incumbent_active_during_pending"
    return "ambiguous", "insufficient_evidence"


@dataclass
class _MirrorPending:
    candidate: str
    started_ms: float
    incumbent_raw_start: int
    candidate_raw_start: int
    candidate_calibrated_at_start: bool
    candidate_rom: float
    incumbent_completed_cycle: bool
    candidate_completed_cycle: bool
    angle_min: Optional[float] = None
    angle_max: Optional[float] = None
    incumbent_observed: bool = False

    @property
    def motion_span_deg(self) -> float:
        if self.angle_min is None or self.angle_max is None:
            return 0.0
        return float(self.angle_max) - float(self.angle_min)

    def fold_angle(self, value: Optional[float]) -> None:
        if value is None:
            return
        self.incumbent_observed = True
        current = float(value)
        if self.angle_min is None or self.angle_max is None:
            self.angle_min = current
            self.angle_max = current
            return
        self.angle_min = min(self.angle_min, current)
        self.angle_max = max(self.angle_max, current)


class AngleRepCounterSession:
    """selecting -> fixed joint -> calibration -> counting.

    ``process_frame`` takes exactly eight product :class:`AngleSample` values.
    ``None``/unknown frames hold detector state. Predicted values may continue
    tracking after lock but do not count toward selection evidence.

    On limb lock the winning selection detector is carried into tracking. Displayed
    reps stay 0 until that detector completes one more validated cycle, then jump
    to the full raw count (pre-lock cycles plus the new one).

    After lock the other seven detectors keep updating. If the locked limb goes
    quiet and its mirrored partner is the one cycling, the display becomes the
    incumbent shown count plus that partner's not-yet-shown raw reps, and the
    partner detector is carried the same way as the first lock.
    """

    def __init__(self, config: Angle3dConfig | None = None, *, fps_hint: float = 30.0) -> None:
        self.cfg = config if config is not None else load_3d_config()
        self.fps_hint = float(fps_hint) if fps_hint and fps_hint > 0 else 30.0
        self._reset_state()

    def _reset_state(self) -> None:
        self.phase: str = "idle"
        self.tracked_joint: Optional[str] = None
        self._histories: dict[str, list[Optional[float]]] = {k: [] for k in PRODUCT_ANGLE_KEYS}
        self._evidence: dict[str, list[str]] = {k: [] for k in PRODUCT_ANGLE_KEYS}
        self._select_detectors: dict[str, PeakDetector] = {
            k: PeakDetector(**self.cfg.peak_detector_kwargs()) for k in PRODUCT_ANGLE_KEYS
        }
        self._count_detector: Optional[PeakDetector] = None
        # Raw detector count at limb lock. Display stays 0 until the carried
        # detector completes one more validated cycle, then exposes the full raw count.
        self._raw_at_lock: Optional[int] = None
        self._credit_revealed: bool = False
        # Set on a mirrored switch. Display is ``raw + offset`` and is not re-hidden.
        self._rep_offset: Optional[int] = None
        self._shown_floor: int = 0
        # Raw reps of each joint already included in the displayed total.
        self._credited_raw: dict[str, int] = {k: 0 for k in PRODUCT_ANGLE_KEYS}
        self._raw_seen: dict[str, int] = {k: 0 for k in PRODUCT_ANGLE_KEYS}
        self._rose_this_frame: dict[str, bool] = {k: False for k in PRODUCT_ANGLE_KEYS}
        self._cycle_log: dict[str, deque[tuple[float, float]]] = {
            k: deque(maxlen=_CYCLE_LOG_LIMIT) for k in PRODUCT_ANGLE_KEYS
        }
        self._pending: Optional[_MirrorPending] = None
        self._rearm_raw: Optional[int] = None
        self._last_handoff_kind: Optional[str] = None
        self._last_handoff_rule: Optional[str] = None
        self._select_started_ms: Optional[float] = None
        self._select_frames: int = 0
        self._dominance_streak: int = 0
        self._last_timestamp_ms: Optional[float] = None
        self._frame_index: int = 0
        self._last_result = self._result(
            reps=0,
            tracked_joint=None,
            angle_3d_value=None,
            detector=None,
            phase="idle",
            status_message="waiting for 3D angles",
        )

    def reset(self) -> None:
        self._reset_state()

    def _note_cycles(self, key: str, detector: PeakDetector, timestamp_ms: float) -> None:
        raw = int(detector.rep_count)
        prev = int(self._raw_seen[key])
        self._rose_this_frame[key] = raw > prev
        if raw > prev:
            rom = float(detector._last_rolling_range)
            for _ in range(raw - prev):
                self._cycle_log[key].append((float(timestamp_ms), rom))
        self._raw_seen[key] = raw

    def _cycles_since(self, key: str, now_ms: float) -> list[float]:
        lo = float(now_ms) - _SYNC_WINDOW_MS
        return [ts for ts, _ in self._cycle_log[key] if ts >= lo]

    def _displayed_reps(self, detector: PeakDetector) -> int:
        """Hide pre-lock selection cycles until the next completed post-lock rep.

        After a mirrored switch, display is ``raw + offset`` immediately. That
        credit is the other limb's raw count, not this hide-until-next-cycle rule.
        """
        raw = int(detector.rep_count)
        if self._rep_offset is not None:
            shown = raw + int(self._rep_offset)
            if shown < self._shown_floor:
                shown = self._shown_floor
            self._shown_floor = shown
            return shown
        if self._credit_revealed:
            return raw
        if self._raw_at_lock is None:
            return raw
        if raw > int(self._raw_at_lock):
            self._credit_revealed = True
            return raw
        return 0

    def _mark_live_joint_credited(self, joint: str, detector: PeakDetector) -> None:
        """Reps currently on the display are already credited for this joint."""
        if self._rep_offset is not None or self._credit_revealed:
            self._credited_raw[joint] = int(detector.rep_count)

    def _trim(self) -> None:
        cap = self.cfg.max_buffer_frames
        for key in PRODUCT_ANGLE_KEYS:
            if len(self._histories[key]) > cap:
                self._histories[key] = self._histories[key][-cap:]
                self._evidence[key] = self._evidence[key][-cap:]

    def _observed_fractions(self) -> dict[str, float]:
        return {key: observed_fraction(self._evidence[key]) for key in PRODUCT_ANGLE_KEYS}

    def _elapsed_sec(self, timestamp_ms: float) -> float:
        if self._select_started_ms is None:
            return 0.0
        return max(0.0, (timestamp_ms - self._select_started_ms) / 1000.0)

    def _result(
        self,
        *,
        reps: int,
        tracked_joint: Optional[str],
        angle_3d_value: Optional[float],
        detector: Optional[PeakDetector],
        phase: str,
        status_message: str,
        tracked_joint_changed: bool = False,
        calibration_started: bool = False,
        calibration_locked: bool = False,
        leader_key: Optional[str] = None,
    ) -> AngleStepResult:
        stats: dict = {}
        if detector is not None:
            stats = detector._calibration_stats()
        return AngleStepResult(
            reps=reps,
            tracked_joint=tracked_joint,
            angle_3d_value=angle_3d_value,
            calibration_complete=bool(detector._calibrated) if detector is not None else False,
            peak_detector_state=detector.state if detector is not None else "NEUTRAL",
            smoothed_value=detector.smoothed_value if detector is not None else None,
            range_gate_open=bool(detector._last_range_gate_open) if detector is not None else False,
            rolling_range=float(detector._last_rolling_range) if detector is not None else None,
            calibration_target_reps=self.cfg.calibration_reps,
            calibration_certainty=float(stats.get("certainty") or 0.0) if stats else 0.0,
            calibration_certainty_target=self.cfg.calibration_certainty,
            phase=phase,  # type: ignore[arg-type]
            status_message=status_message,
            tracked_joint_changed=tracked_joint_changed,
            calibration_started=calibration_started,
            calibration_locked=calibration_locked,
            leader_key=leader_key,
            avg_peak=stats.get("avgPeak") if stats else None,
            avg_valley=stats.get("avgValley") if stats else None,
        )

    def _lock(self, joint: str, timestamp_ms: float) -> AngleStepResult:
        if joint not in PRODUCT_ANGLE_KEYS:
            raise ValueError(f"cannot lock unknown angle {joint!r}")
        self.tracked_joint = joint
        self.phase = "tracking"
        # Carry the winning selection detector so ROM / calibration / cycles survive lock.
        carried = self._select_detectors[joint]
        self._count_detector = carried
        self._raw_at_lock = int(carried.rep_count)
        self._credit_revealed = False
        self._rep_offset = None
        self._shown_floor = 0
        self._pending = None
        self._rearm_raw = None
        self._last_handoff_kind = None
        self._last_handoff_rule = None
        sample_value = None
        if self._histories[joint]:
            sample_value = self._histories[joint][-1]
        status = f"locked {joint}"
        # Truthful calibration metadata from the carried detector. Display stays 0.
        already_calibrated = bool(carried._calibrated)
        result = self._result(
            reps=0,
            tracked_joint=joint,
            angle_3d_value=sample_value,
            detector=self._count_detector,
            phase="tracking",
            status_message=status,
            tracked_joint_changed=True,
            calibration_started=True,
            calibration_locked=already_calibrated,
            leader_key=joint,
        )
        self._last_result = result
        self._last_timestamp_ms = timestamp_ms
        return result

    def _try_lock(self, timestamp_ms: float) -> Optional[str]:
        elapsed = self._elapsed_sec(timestamp_ms)
        if elapsed < self.cfg.min_sec or self._select_frames < self.cfg.min_frames:
            return None
        variances = compute_variances_from_histories(self._histories, self.cfg)
        fractions = self._observed_fractions()
        rep_counts = {k: self._select_detectors[k].rep_count for k in PRODUCT_ANGLE_KEYS}
        rep_dom = summarize_rep_dominance(rep_counts)
        dominant = dominance_lock_ready(variances, rep_dom, self.cfg, observed_fractions=fractions)
        if dominant is not None:
            self._dominance_streak += 1
            if self._dominance_streak >= self.cfg.dominance_streak_frames:
                return dominant
        else:
            self._dominance_streak = 0
        if elapsed >= self.cfg.variance_fallback_sec:
            return top_variance_candidate(variances, self.cfg, observed_fractions=fractions)
        return None

    def _activate_mirror(self, incumbent: str, candidate_key: str, incumbent_shown: int) -> None:
        candidate = self._select_detectors[candidate_key]
        candidate_raw = int(candidate.rep_count)
        if self._credit_revealed or self._rep_offset is not None:
            self._credited_raw[incumbent] = int(self._count_detector.rep_count)  # type: ignore[union-attr]
        offset = mirror_display_offset(
            incumbent_displayed=incumbent_shown,
            candidate_raw=candidate_raw,
            candidate_raw_at_new_lock=candidate_raw,
            credited_raw=self._credited_raw[candidate_key],
        )
        target = candidate_raw + offset
        target = max(target, incumbent_shown)
        offset = target - candidate_raw
        self._rep_offset = offset
        self._shown_floor = max(self._shown_floor, target)
        self._credited_raw[candidate_key] = candidate_raw
        # Carry the partner detector. ROM and calibration stay on that object.
        self._count_detector = candidate
        self.tracked_joint = candidate_key
        self._credit_revealed = True
        self._pending = None
        partner = mirrored_partner(candidate_key)
        self._rearm_raw = int(self._select_detectors[partner].rep_count) if partner is not None else None

    def _mirror_handoff(
        self,
        joint: str,
        parsed: dict[str, AngleSample],
        timestamp_ms: float,
    ) -> bool:
        """Observe only the mirrored partner. Switch when that side is cycling and this side has stopped."""
        mirror = mirrored_partner(joint)
        if mirror is None or self._count_detector is None:
            self._pending = None
            return False
        mirror_det = self._select_detectors[mirror]
        if self._pending is not None and self._pending.candidate != mirror:
            self._pending = None
        if self._pending is None:
            rose = bool(self._rose_this_frame.get(mirror))
            raw_now = int(mirror_det.rep_count)
            if not rose:
                return False
            if self._rearm_raw is not None and raw_now <= int(self._rearm_raw):
                return False
            self._pending = _MirrorPending(
                candidate=mirror,
                started_ms=float(timestamp_ms),
                incumbent_raw_start=int(self._count_detector.rep_count),
                candidate_raw_start=raw_now,
                candidate_calibrated_at_start=bool(mirror_det._calibrated),
                candidate_rom=float(mirror_det._last_rolling_range),
                incumbent_completed_cycle=bool(self._rose_this_frame.get(joint)),
                candidate_completed_cycle=True,
            )
        pending = self._pending
        pending.fold_angle(parsed[joint].value)
        if int(self._count_detector.rep_count) > pending.incumbent_raw_start:
            pending.incumbent_completed_cycle = True
        if int(mirror_det.rep_count) > pending.candidate_raw_start:
            pending.candidate_completed_cycle = True
        pending.candidate_rom = max(pending.candidate_rom, float(mirror_det._last_rolling_range))
        elapsed = float(timestamp_ms) - pending.started_ms
        # Already-calibrated partner can close the window at 300ms. A partner that
        # was not calibrated when observation started waits out the full 600ms.
        ready = (
            pending.candidate_calibrated_at_start and elapsed >= _MIRROR_PENDING_MIN_MS
        ) or elapsed > _MIRROR_PENDING_MAX_MS
        if not ready:
            return False
        if not pending.incumbent_observed:
            self._last_handoff_kind = "ambiguous"
            self._last_handoff_rule = "incumbent_unobserved"
            self._pending = None
            self._rearm_raw = int(mirror_det.rep_count)
            return False
        inc_ts = self._cycles_since(joint, timestamp_ms)
        cand_ts = self._cycles_since(mirror, timestamp_ms)
        sync = cycle_sync_score(inc_ts, cand_ts)
        candidate_raw = int(mirror_det.rep_count)
        kind, rule = classify_mirrored_handoff(
            incumbent_motion_span_deg=pending.motion_span_deg,
            candidate_rom_deg=pending.candidate_rom,
            incumbent_completed_gated_cycle=pending.incumbent_completed_cycle,
            candidate_raw=candidate_raw,
            candidate_active=pending.candidate_completed_cycle or candidate_raw > pending.candidate_raw_start,
            cycle_sync_score_last_4s=sync,
            incumbent_cycles_last_4s=len(inc_ts),
            candidate_cycles_last_4s=len(cand_ts),
        )
        self._last_handoff_kind = kind
        self._last_handoff_rule = rule
        self._pending = None
        if kind != "alternate_limb":
            self._rearm_raw = candidate_raw
            return False
        shown = self._displayed_reps(self._count_detector)
        self._activate_mirror(joint, mirror, shown)
        return True

    def process_frame(
        self,
        samples: dict[str, AngleSample],
        timestamp_ms: float | None = None,
    ) -> AngleStepResult:
        parsed = require_product_samples(samples)
        self._frame_index += 1
        self._rose_this_frame = {k: False for k in PRODUCT_ANGLE_KEYS}
        ts = float(timestamp_ms) if timestamp_ms is not None else (
            (self._last_timestamp_ms or 0.0) + (1000.0 / self.fps_hint)
        )
        usable = {k: s for k, s in parsed.items() if s.value is not None}
        if not usable:
            # Missing 3D samples are not stillness. Don't let the gap expire a
            # mirrored window and read an empty span as "the limb stopped".
            if self._pending is not None and self._last_timestamp_ms is not None:
                self._pending.started_ms += max(0.0, ts - float(self._last_timestamp_ms))
            held = AngleStepResult(**{**self._last_result.__dict__, "tracked_joint_changed": False,
                                      "calibration_started": False, "calibration_locked": False})
            held.status_message = "holding (no 3D angles)"
            self._last_result = held
            self._last_timestamp_ms = ts
            return held

        if self.phase == "idle":
            self.phase = "selecting"
            self._select_started_ms = ts

        for key, sample in parsed.items():
            self._histories[key].append(sample.value)
            self._evidence[key].append(sample.evidence)
        self._trim()

        if self.phase == "selecting":
            self._select_frames += 1
            for key, sample in parsed.items():
                self._select_detectors[key].update(sample.value)
                self._note_cycles(key, self._select_detectors[key], ts)
            lock_key = self._try_lock(ts)
            if lock_key is not None:
                return self._lock(lock_key, ts)
            variances = compute_variances_from_histories(self._histories, self.cfg)
            fractions = self._observed_fractions()
            leader = top_variance_candidate(variances, self.cfg, observed_fractions=fractions)
            result = self._result(
                reps=0,
                tracked_joint=None,
                angle_3d_value=None,
                detector=None,
                phase="selecting",
                status_message="selecting angle",
                leader_key=leader,
            )
            self._last_result = result
            self._last_timestamp_ms = ts
            return result

        joint = self.tracked_joint
        assert joint is not None and self._count_detector is not None
        cal_before = {key: bool(det._calibrated) for key, det in self._select_detectors.items()}
        # The locked detector is updated below. The other seven must keep counting
        # so a mirrored partner has a raw total to add.
        for key, sample in parsed.items():
            if key == joint:
                continue
            self._select_detectors[key].update(sample.value)
            self._note_cycles(key, self._select_detectors[key], ts)
        prev_reps = int(self._count_detector.rep_count)
        self._count_detector.update(parsed[joint].value)
        self._note_cycles(joint, self._count_detector, ts)
        if int(self._count_detector.rep_count) < prev_reps:
            raise RuntimeError("count detector lost reps")
        switched = self._mirror_handoff(joint, parsed, ts)
        joint = self.tracked_joint
        assert joint is not None and self._count_detector is not None
        displayed = self._displayed_reps(self._count_detector)
        self._mark_live_joint_credited(joint, self._count_detector)
        cal_locked = (not cal_before[joint]) and bool(self._count_detector._calibrated)
        result = self._result(
            reps=displayed,
            tracked_joint=joint,
            angle_3d_value=parsed[joint].value,
            detector=self._count_detector,
            phase="tracking",
            status_message=f"counting {joint}",
            tracked_joint_changed=switched,
            calibration_started=False,
            calibration_locked=cal_locked,
            leader_key=joint,
        )
        self._last_result = result
        self._last_timestamp_ms = ts
        return result
