"""Skeleton-agnostic 3D angle session: eight named scalars, no landmark indices."""
from __future__ import annotations

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


class AngleRepCounterSession:
    """selecting -> fixed joint -> calibration -> counting.

    ``process_frame`` takes exactly eight product :class:`AngleSample` values.
    ``None``/unknown frames hold detector state. Predicted values may continue
    tracking after lock but do not count toward selection evidence.

    On limb lock the winning selection detector is carried into tracking. Displayed
    reps stay 0 until that detector completes one more validated cycle, then jump
    to the full raw count (pre-lock cycles plus the new one).
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

    def _displayed_reps(self, detector: PeakDetector) -> int:
        """Hide pre-lock selection cycles until the next completed post-lock rep."""
        raw = int(detector.rep_count)
        if self._credit_revealed:
            return raw
        if self._raw_at_lock is None:
            return raw
        if raw > int(self._raw_at_lock):
            self._credit_revealed = True
            return raw
        return 0

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

    def process_frame(
        self,
        samples: dict[str, AngleSample],
        timestamp_ms: float | None = None,
    ) -> AngleStepResult:
        parsed = require_product_samples(samples)
        self._frame_index += 1
        ts = float(timestamp_ms) if timestamp_ms is not None else (
            (self._last_timestamp_ms or 0.0) + (1000.0 / self.fps_hint)
        )
        usable = {k: s for k, s in parsed.items() if s.value is not None}
        if not usable:
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
        sample = parsed[joint]
        prev_cal = self._count_detector._calibrated
        prev_reps = self._count_detector.rep_count
        self._count_detector.update(sample.value)
        cal_locked = (not prev_cal) and self._count_detector._calibrated
        cal_started = False
        displayed = self._displayed_reps(self._count_detector)
        result = self._result(
            reps=displayed,
            tracked_joint=joint,
            angle_3d_value=sample.value,
            detector=self._count_detector,
            phase="tracking",
            status_message=f"counting {joint}",
            calibration_started=cal_started,
            calibration_locked=cal_locked,
            leader_key=joint,
        )
        if self._count_detector.rep_count < prev_reps:
            raise RuntimeError("count detector lost reps")
        self._last_result = result
        self._last_timestamp_ms = ts
        return result
