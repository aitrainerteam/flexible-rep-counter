"""Session state machine: landmark stream -> rep metrics (same logic as legacy webcam loop)."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from flexible_rep_counter.core.math_engine import (
    calculate_from_type,
    create_peak_detector,
    replay_angle_series_on_peak_detector,
)
from flexible_rep_counter.core.pose_filters import PoseFilterPipeline
from flexible_rep_counter.core.settings import (
    ANGLE_SELECTION_DOMINANCE_FRACTION,
    ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES,
    ANGLE_SELECTION_MAX_BUFFER_FRAMES,
    ANGLE_SELECTION_MIN_FRAMES,
    ANGLE_SELECTION_MIN_LEADING_REPS,
    ANGLE_SELECTION_MIN_SEC,
    ANGLE_SELECTION_RETRY_INTERVAL_SEC,
    ANGLE_SELECTION_REEVALUATE_EVERY_SEC,
    ANGLE_SELECTION_SWITCH_MIN_SEC,
    ANGLE_SELECTION_SWITCH_VARIANCE_RATIO,
    ANGLE_SELECTION_VARIANCE_FALLBACK_SEC,
    get_default_tuning_params,
)
from flexible_rep_counter.core.variance_angle_selector import (
    COMMON_ANGLES,
    compute_angle_variances_from_buffer,
    determine_best_angle,
    dominance_conditions_met,
    passes_consistent_variance_gate,
    summarize_rep_dominance,
)
from flexible_rep_counter.types import StepResult

_DEBUG_LOG_PATH = Path("/Users/aa/Desktop/flexible-rep-counter/.cursor/debug-75e746.log")
_DEBUG_SESSION_ID = "75e746"
_DEBUG_RUN_ID = "post-fix"


def _agent_debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _peak_detector_from_tuning(tuning_params: dict[str, Any]) -> Any:
    d = get_default_tuning_params()
    tp = tuning_params or {}
    return create_peak_detector(
        smoothing_factor=float(tp.get("smoothingFactor", d["smoothingFactor"])),
        hysteresis=float(tp.get("hysteresis", d["hysteresis"])),
        min_peak_distance=int(tp.get("minPeakDistance", d["minPeakDistance"])),
        peak_margin_pct=float(tp.get("peakMarginPct", d["peakMarginPct"])),
        valley_margin_pct=float(tp.get("valleyMarginPct", d["valleyMarginPct"])),
        min_range_gate_degrees=float(tp.get("minRangeGate", d["minRangeGate"])),
        range_window_frames=int(tp.get("rangeWindowFrames", d["rangeWindowFrames"])),
        range_min_samples=int(tp.get("rangeMinSamples", d["rangeMinSamples"])),
        delta_deadband_degrees=float(tp.get("angleDeltaDeadband", d["angleDeltaDeadband"])),
        calibration_reps=int(tp.get("calibrationReps", d["calibrationReps"])),
        calibration_certainty=float(tp.get("calibrationCertainty", d["calibrationCertainty"])),
        calibration_force_extra_reps=int(
            tp.get("calibrationForceExtraReps", d["calibrationForceExtraReps"])
        ),
        min_rep_interval_ms=float(tp.get("minRepIntervalMs", d["minRepIntervalMs"])),
    )


def _apply_locked_tracking(
    run_state: dict[str, Any],
    selected_angle: str,
    buf_list: list[list[dict]],
    tuning_params: dict[str, Any],
    *,
    selection_detector: Optional[Any] = None,
) -> None:
    run_state["selected_angle"] = selected_angle
    run_state["selected_config"] = COMMON_ANGLES[selected_angle]
    run_state["selection_last_attempt"] = None
    run_state["selection_last_reevaluate_at"] = time.time()
    run_state["selection_last_switch_at"] = time.time()
    det: Any
    if selection_detector is not None:
        det = selection_detector
    else:
        det = _peak_detector_from_tuning(tuning_params)
        cfg = COMMON_ANGLES[selected_angle]
        series = [
            calculate_from_type(cfg["type"], cfg["landmarks"], lm) for lm in buf_list
        ]
        replay_angle_series_on_peak_detector(det, series)
    run_state["peak_detector"] = det
    det_state = det.get_state() if det is not None else {}
    # region agent log
    _agent_debug_log(
        run_id=_DEBUG_RUN_ID,
        hypothesis_id="H1",
        location="session.py:_apply_locked_tracking",
        message="Tracking lock applied with detector state carryover",
        data={
            "selectedAngle": str(selected_angle),
            "detectorRepCount": int(det_state.get("repCount") or 0),
            "calibrationComplete": bool(det_state.get("calibrationComplete", False)),
            "calibratedAvgPeak": (
                float(det_state["calibratedAvgPeak"])
                if det_state.get("calibratedAvgPeak") is not None
                else None
            ),
            "calibratedAvgValley": (
                float(det_state["calibratedAvgValley"])
                if det_state.get("calibratedAvgValley") is not None
                else None
            ),
        },
    )
    # endregion
    run_state["selection_dominance_key"] = None
    run_state["selection_dominance_streak"] = 0


def _collect_joint_records(
    detectors_by_angle: dict[str, Any], variances: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for angle_key in COMMON_ANGLES:
        det = detectors_by_angle.get(angle_key)
        det_state = det.get_state() if det is not None else {}
        v = variances.get(angle_key) or {}
        records[angle_key] = {
            "repCount": int(det_state.get("repCount") or 0),
            "avgPeak": (
                float(det_state["calibratedAvgPeak"])
                if det_state.get("calibratedAvgPeak") is not None
                else None
            ),
            "avgValley": (
                float(det_state["calibratedAvgValley"])
                if det_state.get("calibratedAvgValley") is not None
                else None
            ),
            "calibrationComplete": bool(det_state.get("calibrationComplete", False)),
            "medianWindowVariance": float(v.get("medianWindowVariance") or 0.0),
            "activeWindowCount": int(v.get("activeWindowCount") or 0),
            "smoothedRangeDeg": float(v.get("smoothedRangeDeg") or 0.0),
        }
    return records


def _selection_status_message(
    *,
    ready: bool,
    elapsed: float,
    n_frames: int,
    retry_at: Any,
    locked_this_frame: bool,
    selected_angle: Optional[str],
    run_state_selected: Optional[str],
    dom_ok: bool,
    leader_key: Optional[str],
    streak: int,
    rep_dom: dict[str, Any],
) -> str:
    if not ready:
        return (
            f"Observing movement... {n_frames}/{ANGLE_SELECTION_MIN_FRAMES} frames, "
            f"{elapsed:.1f}s / {ANGLE_SELECTION_MIN_SEC:.0f}s min"
        )
    now = time.time()
    if retry_at is not None and (
        now - float(retry_at) < ANGLE_SELECTION_RETRY_INTERVAL_SEC
    ):
        rem = ANGLE_SELECTION_RETRY_INTERVAL_SEC - (now - float(retry_at))
        return f"Unclear motion — retry in {rem:.0f}s (full ROM, steady reps)"
    if selected_angle or locked_this_frame or run_state_selected:
        return "Locked joint — calibrating reps"
    if ready and dom_ok and leader_key:
        share = float(rep_dom.get("leaderShare") or 0.0)
        return (
            f"Confirming primary joint {leader_key}: "
            f"streak {streak}/{ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES}, "
            f"rep share {share:.0%} (need >{ANGLE_SELECTION_DOMINANCE_FRACTION:.0%})"
        )
    if ready:
        total_r = int(rep_dom.get("totalReps") or 0)
        return (
            f"Mapping all joints... {total_r} reps across angles, "
            f"{n_frames} frames — need one joint >{ANGLE_SELECTION_DOMINANCE_FRACTION:.0%} "
            f"of reps and {ANGLE_SELECTION_MIN_LEADING_REPS}+ reps"
        )
    return f"Analyzing... ({n_frames} frames)"


def _format_angle_label(angle_key: str) -> str:
    return angle_key.replace("_", " ").title()


def _idle_result(msg: str = "Click Start to begin") -> StepResult:
    d = get_default_tuning_params()
    return StepResult(
        reps=0,
        reps_raw=0,
        tracked_joint=None,
        angle_3_point_value=None,
        target_landmarks=None,
        tuning_params=d,
        avg_peak=None,
        avg_valley=None,
        calibration_complete=False,
        peak_detector_state="—",
        smoothed_value=None,
        range_gate_open=True,
        rolling_range=None,
        calibration_target_reps=int(d["calibrationReps"]),
        calibration_certainty=0.0,
        calibration_certainty_target=float(d["calibrationCertainty"]),
        phase="idle",
        status_message=msg,
        tracking_detail_message="",
    )


class RepCounterSession:
    """Runs angle selection + peak-detector tracking on a stream of landmark frames."""

    def __init__(
        self,
        *,
        tuning_params: Optional[dict[str, Any]] = None,
        use_pose_filter: bool = True,
        auto_started: bool = False,
    ) -> None:
        self._use_filter = use_pose_filter
        self._pose_pipeline = PoseFilterPipeline() if use_pose_filter else None
        self._auto_started = auto_started
        self._run_state: dict[str, Any] = {}
        self._last_smoothed_landmarks: Optional[list[dict]] = None
        self.reset(tuning_params=tuning_params)
        if auto_started:
            self.set_started()

    def reset(self, *, tuning_params: Optional[dict[str, Any]] = None) -> None:
        """Full reset (new exercise or second Start in visualizer)."""
        tp = tuning_params if tuning_params is not None else get_default_tuning_params()
        self._run_state = {
            "started": bool(self._auto_started),
            "started_at": time.time() if self._auto_started else None,
            "frame_buffer": deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES),
            "selection_last_attempt": None,
            "selection_last_reevaluate_at": None,
            "selection_last_switch_at": None,
            "selection_detectors_by_angle": {},
            "selection_dominance_key": None,
            "selection_dominance_streak": 0,
            "selected_angle": None,
            "selected_config": None,
            "peak_detector": None,
            "tuning_params": dict(tp),
            "debug_last_selection_log_bucket": -1,
            "debug_last_rep_count": 0,
            "debug_last_calibration_complete": None,
            "debug_last_range_gate_open": None,
            "debug_last_peak_state": None,
        }
        if self._pose_pipeline is not None:
            self._pose_pipeline = PoseFilterPipeline()

    def set_started(self, wall_time: Optional[float] = None) -> None:
        """Mark session as started (visualizer Start button)."""
        self._run_state["started"] = True
        self._run_state["started_at"] = wall_time if wall_time is not None else time.time()
        self._run_state["selection_last_attempt"] = None
        self._run_state["selection_last_reevaluate_at"] = None
        self._run_state["selection_last_switch_at"] = None
        self._run_state["selection_detectors_by_angle"] = {}
        self._run_state["selection_dominance_key"] = None
        self._run_state["selection_dominance_streak"] = 0
        self._run_state["debug_last_selection_log_bucket"] = -1
        self._run_state["debug_last_rep_count"] = 0
        self._run_state["debug_last_calibration_complete"] = None
        self._run_state["debug_last_range_gate_open"] = None
        self._run_state["debug_last_peak_state"] = None

    def clear_tracking_keep_started(self) -> None:
        """Second Start click: clear selection/tracking but keep started=True."""
        self._run_state["selected_angle"] = None
        self._run_state["selected_config"] = None
        self._run_state["peak_detector"] = None
        self._run_state["frame_buffer"] = deque(maxlen=ANGLE_SELECTION_MAX_BUFFER_FRAMES)
        self._run_state["selection_last_attempt"] = None
        self._run_state["selection_last_reevaluate_at"] = None
        self._run_state["selection_last_switch_at"] = None
        self._run_state["selection_detectors_by_angle"] = {}
        self._run_state["selection_dominance_key"] = None
        self._run_state["selection_dominance_streak"] = 0
        self._run_state["debug_last_selection_log_bucket"] = -1
        self._run_state["debug_last_rep_count"] = 0
        self._run_state["debug_last_calibration_complete"] = None
        self._run_state["debug_last_range_gate_open"] = None
        self._run_state["debug_last_peak_state"] = None

    @property
    def started(self) -> bool:
        return bool(self._run_state.get("started"))

    @property
    def last_smoothed_landmarks(self) -> Optional[list[dict]]:
        """Landmarks after temporal filtering from the last successful ``step_landmarks`` call."""
        return self._last_smoothed_landmarks

    def step_landmarks(
        self,
        landmarks: Optional[list[dict]],
        *,
        timestamp_ms: Optional[float] = None,
        wall_time_s: Optional[float] = None,
    ) -> StepResult:
        """
        Process one frame of 17 COCO landmarks (after any resolution scaling).

        ``landmarks`` may be None when no pose is detected.
        """
        if not self._run_state.get("started"):
            return _idle_result()

        now = wall_time_s if wall_time_s is not None else time.time()
        ts = timestamp_ms if timestamp_ms is not None else now * 1000.0

        rs = self._run_state
        tuning_params = rs["tuning_params"]

        if not landmarks:
            self._last_smoothed_landmarks = None
            if rs.get("selected_angle") is not None and rs.get("peak_detector") is not None:
                tr = self._build_tracking_step_result(rs, None)
                return replace(tr, status_message=f"No pose - {tr.status_message}")
            return StepResult(
                reps=0,
                reps_raw=0,
                tracked_joint=rs.get("selected_angle"),
                angle_3_point_value=None,
                target_landmarks=None,
                tuning_params=dict(tuning_params),
                avg_peak=None,
                avg_valley=None,
                calibration_complete=False,
                peak_detector_state="—",
                smoothed_value=None,
                range_gate_open=True,
                rolling_range=None,
                calibration_target_reps=int(
                    tuning_params.get("calibrationReps", get_default_tuning_params()["calibrationReps"])
                ),
                calibration_certainty=0.0,
                calibration_certainty_target=float(
                    tuning_params.get(
                        "calibrationCertainty",
                        get_default_tuning_params()["calibrationCertainty"],
                    )
                ),
                phase="selecting" if rs.get("selected_angle") is None else "tracking",
                status_message="No pose",
                tracking_detail_message="",
            )

        if self._pose_pipeline is not None:
            lm = self._pose_pipeline.process(landmarks, ts)
        else:
            lm = landmarks
        self._last_smoothed_landmarks = lm

        frame_buffer = rs["frame_buffer"]
        frame_buffer.append(lm)
        buf_list = list(frame_buffer)
        started_at = float(rs.get("started_at") or 0.0)
        elapsed = now - started_at

        sdba: dict[str, Any] = rs.get("selection_detectors_by_angle") or {}
        if not sdba:
            sdba = {ak: _peak_detector_from_tuning(tuning_params) for ak in COMMON_ANGLES}
            for ak, det in sdba.items():
                try:
                    setattr(det, "debug_label", ak)
                except Exception:
                    pass
            rs["selection_detectors_by_angle"] = sdba

        angle_values: dict[str, Optional[float]] = {}
        detector_outputs: dict[str, dict[str, Any]] = {}
        for ak, cfg in COMMON_ANGLES.items():
            val = calculate_from_type(cfg["type"], cfg["landmarks"], lm)
            angle_values[ak] = val
            detector_outputs[ak] = sdba[ak].update(val)  # type: ignore[union-attr]

        rep_counts_sel = {ak: d.get_rep_count() for ak, d in sdba.items()}
        rep_dom = summarize_rep_dominance(rep_counts_sel)
        variances = compute_angle_variances_from_buffer(buf_list)
        joint_records = _collect_joint_records(sdba, variances)

        selected_angle = rs["selected_angle"]
        if selected_angle is not None and rs.get("peak_detector") is None:
            rs["peak_detector"] = sdba.get(selected_angle)

        if selected_angle is None:
            ready = (
                len(frame_buffer) >= ANGLE_SELECTION_MIN_FRAMES
                and elapsed >= ANGLE_SELECTION_MIN_SEC
            )
            last_att = rs.get("selection_last_attempt")
            can_try = ready and (
                last_att is None
                or (now - float(last_att)) >= ANGLE_SELECTION_RETRY_INTERVAL_SEC
            )
            dom_ok = dominance_conditions_met(
                variances,
                rep_dom,
                dominance_fraction=ANGLE_SELECTION_DOMINANCE_FRACTION,
                min_leading_reps=ANGLE_SELECTION_MIN_LEADING_REPS,
            )
            leader_key = rep_dom.get("leaderKey")
            if dom_ok and leader_key:
                if rs.get("selection_dominance_key") == leader_key:
                    rs["selection_dominance_streak"] = int(
                        rs.get("selection_dominance_streak") or 0
                    ) + 1
                else:
                    rs["selection_dominance_key"] = leader_key
                    rs["selection_dominance_streak"] = 1
            else:
                rs["selection_dominance_key"] = None
                rs["selection_dominance_streak"] = 0

            streak = int(rs.get("selection_dominance_streak") or 0)
            lock_from_dominance = (
                ready and dom_ok and streak >= ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES
            )
            # If we already observe enough reps during selection, allow earlier variance lock
            # instead of waiting the full fallback timeout.
            cal_reps_target = int(
                tuning_params.get(
                    "calibrationReps",
                    get_default_tuning_params()["calibrationReps"],
                )
            )
            total_selection_reps = int(rep_dom.get("totalReps") or 0)
            variance_fallback_ready = ready and (
                elapsed >= ANGLE_SELECTION_VARIANCE_FALLBACK_SEC
                or total_selection_reps >= cal_reps_target
            )
            log_bucket = int(elapsed * 2.0)
            if (
                can_try
                or dom_ok
                or rs.get("debug_last_selection_log_bucket", -1) != log_bucket
            ):
                rs["debug_last_selection_log_bucket"] = log_bucket
                # region agent log
                _agent_debug_log(
                    run_id=_DEBUG_RUN_ID,
                    hypothesis_id="H2_H5",
                    location="session.py:selection_snapshot",
                    message="Selection state snapshot",
                    data={
                        "elapsedSec": round(float(elapsed), 3),
                        "bufferFrames": len(frame_buffer),
                        "ready": bool(ready),
                        "canTry": bool(can_try),
                        "domOk": bool(dom_ok),
                        "leaderKey": str(leader_key) if leader_key else None,
                        "streak": int(streak),
                        "totalSelectionReps": int(total_selection_reps),
                        "varianceFallbackReady": bool(variance_fallback_ready),
                    },
                )
                # endregion

            locked_this_frame = False
            selected_angle_local: Optional[str] = None

            if lock_from_dominance and leader_key:
                # region agent log
                _agent_debug_log(
                    run_id=_DEBUG_RUN_ID,
                    hypothesis_id="H1",
                    location="session.py:lock_from_dominance",
                    message="Angle locked from dominance",
                    data={
                        "selectedAngle": str(leader_key),
                        "leaderShare": float(rep_dom.get("leaderShare") or 0.0),
                        "leaderReps": int(rep_dom.get("leaderReps") or 0),
                        "totalSelectionReps": int(rep_dom.get("totalReps") or 0),
                        "streak": int(streak),
                    },
                )
                # endregion
                _apply_locked_tracking(
                    rs,
                    leader_key,
                    buf_list,
                    tuning_params,
                    selection_detector=sdba.get(leader_key),
                )
                locked_this_frame = True
            elif can_try and variance_fallback_ready:
                result = determine_best_angle(buf_list, exercise=None)
                tuning_params = result.get("tuningParams") or get_default_tuning_params()
                rs["tuning_params"] = tuning_params
                sel = result.get("selectedAngle")
                src = str(result.get("source") or "")
                # region agent log
                _agent_debug_log(
                    run_id=_DEBUG_RUN_ID,
                    hypothesis_id="H5",
                    location="session.py:variance_attempt",
                    message="Variance selection attempt",
                    data={
                        "selectedAngle": str(sel) if sel else None,
                        "source": src,
                        "topCandidate": (
                            (result.get("debug") or {}).get("topCandidate")
                            if isinstance(result.get("debug"), dict)
                            else None
                        ),
                    },
                )
                # endregion
                variance_ok = sel and sel in COMMON_ANGLES and src == "variance"
                if variance_ok:
                    # region agent log
                    _agent_debug_log(
                        run_id=_DEBUG_RUN_ID,
                        hypothesis_id="H2",
                        location="session.py:variance_lock_carryover",
                        message="Variance lock carryover snapshot",
                        data={
                            "selectedAngle": str(sel),
                            "selectionRepCountForAngle": int(
                                sdba.get(str(sel)).get_rep_count() if sdba.get(str(sel)) else 0
                            ),
                            "totalSelectionReps": int(rep_dom.get("totalReps") or 0),
                        },
                    )
                    # endregion
                    _apply_locked_tracking(
                        rs,
                        sel,
                        buf_list,
                        tuning_params,
                        selection_detector=sdba.get(sel),
                    )
                    locked_this_frame = True
                else:
                    rs["selected_angle"] = None
                    rs["selected_config"] = None
                    rs["peak_detector"] = None
                    rs["selection_last_attempt"] = now

            retry_at = rs.get("selection_last_attempt")
            status = _selection_status_message(
                ready=ready,
                elapsed=elapsed,
                n_frames=len(frame_buffer),
                retry_at=retry_at,
                locked_this_frame=locked_this_frame,
                selected_angle=selected_angle_local,
                run_state_selected=rs.get("selected_angle"),
                dom_ok=dom_ok,
                leader_key=leader_key,
                streak=streak,
                rep_dom=rep_dom,
            )
            return StepResult(
                reps=0,
                reps_raw=0,
                tracked_joint=rs.get("selected_angle"),
                angle_3_point_value=None,
                target_landmarks=None,
                tuning_params=dict(rs["tuning_params"]),
                avg_peak=None,
                avg_valley=None,
                calibration_complete=False,
                peak_detector_state="—",
                smoothed_value=None,
                range_gate_open=True,
                rolling_range=None,
                calibration_target_reps=int(
                    rs["tuning_params"].get(
                        "calibrationReps", get_default_tuning_params()["calibrationReps"]
                    )
                ),
                calibration_certainty=0.0,
                calibration_certainty_target=float(
                    rs["tuning_params"].get(
                        "calibrationCertainty",
                        get_default_tuning_params()["calibrationCertainty"],
                    )
                ),
                phase="selecting",
                status_message=status,
                tracking_detail_message="",
                leader_key=leader_key if isinstance(leader_key, str) else None,
                selection_debug={
                    "rep_dom": rep_dom,
                    "dom_ok": dom_ok,
                    "joint_records": joint_records,
                },
            )

        # Tracking phase: keep reassessing based on recent variance while preserving per-joint records.
        switched_to: Optional[str] = None
        last_re_eval = rs.get("selection_last_reevaluate_at")
        re_eval_due = (
            ANGLE_SELECTION_REEVALUATE_EVERY_SEC <= 0
            or last_re_eval is None
            or (now - float(last_re_eval)) >= ANGLE_SELECTION_REEVALUATE_EVERY_SEC
        )
        if re_eval_due:
            rs["selection_last_reevaluate_at"] = now
            result = determine_best_angle(buf_list, exercise=None)
            cand = result.get("selectedAngle")
            src = str(result.get("source") or "")
            candidate = cand if isinstance(cand, str) and cand in COMMON_ANGLES else None
            if candidate and src == "variance" and candidate != selected_angle:
                cur_ok = passes_consistent_variance_gate(variances, selected_angle)
                cand_var = float((variances.get(candidate) or {}).get("medianWindowVariance") or 0.0)
                cur_var = float((variances.get(selected_angle) or {}).get("medianWindowVariance") or 0.0)
                stronger = (cur_var <= 0.0 and cand_var > 0.0) or (
                    cur_var > 0.0 and cand_var >= cur_var * ANGLE_SELECTION_SWITCH_VARIANCE_RATIO
                )
                last_switch = rs.get("selection_last_switch_at")
                cooldown_ok = (
                    last_switch is None
                    or (now - float(last_switch)) >= ANGLE_SELECTION_SWITCH_MIN_SEC
                )
                if cooldown_ok and ((not cur_ok) or stronger):
                    rs["selected_angle"] = candidate
                    rs["selected_config"] = COMMON_ANGLES[candidate]
                    rs["peak_detector"] = sdba.get(candidate)
                    rs["selection_last_switch_at"] = now
                    switched_to = candidate
                    selected_angle = candidate
                    _agent_debug_log(
                        run_id="pre-fix",
                        hypothesis_id="H6",
                        location="session.py:tracking_reselection",
                        message="Switched tracked joint during tracking",
                        data={
                            "newSelectedAngle": candidate,
                            "currentVariance": round(cur_var, 5),
                            "candidateVariance": round(cand_var, 5),
                            "currentPassesGate": bool(cur_ok),
                            "switchRatio": float(ANGLE_SELECTION_SWITCH_VARIANCE_RATIO),
                        },
                    )

        angle_value = angle_values.get(selected_angle) if isinstance(selected_angle, str) else None
        selected_output = (
            detector_outputs.get(selected_angle) if isinstance(selected_angle, str) else None
        )
        return self._build_tracking_step_result(
            rs,
            angle_value,
            detector_output=selected_output,
            selection_debug={
                "rep_dom": rep_dom,
                "joint_records": joint_records,
                "switched_to": switched_to,
            },
        )

    def _build_tracking_step_result(
        self,
        rs: dict[str, Any],
        angle_value: Optional[float],
        *,
        detector_output: Optional[dict[str, Any]] = None,
        selection_debug: Optional[dict[str, Any]] = None,
    ) -> StepResult:
        tuning_params = rs["tuning_params"]
        peak_detector = rs["peak_detector"]
        d_tuning = get_default_tuning_params()
        rep_count = 0
        state_str = "—"
        smoothed_value = None
        range_gate_open = True
        rolling_range: Optional[float] = None
        calibration_complete = True
        cal_target = int(tuning_params.get("calibrationReps", d_tuning["calibrationReps"]))
        cal_certainty_target = float(
            tuning_params.get("calibrationCertainty", d_tuning["calibrationCertainty"])
        )
        cal_certainty = 0.0
        primary_rep_count = 0
        avg_peak: Optional[float] = None
        avg_valley: Optional[float] = None

        if peak_detector is not None:
            out = detector_output if detector_output is not None else peak_detector.update(angle_value)
            rep_count = int(out.get("repCount", 0) or 0)
            primary_rep_count = rep_count
            state_str = str(out.get("state", "—"))
            smoothed_value = out.get("smoothedValue")
            if smoothed_value is not None:
                smoothed_value = float(smoothed_value)
            range_gate_open = bool(out.get("rangeGateOpen", True))
            r = out.get("rollingRange")
            rolling_range = float(r) if r is not None else None
            calibration_complete = bool(out.get("calibrationComplete", False))
            cal_target = int(out.get("calibrationTargetReps", cal_target))
            cal_certainty = float(out.get("calibrationCertainty", 0.0) or 0.0)
            cal_certainty_target = float(
                out.get("calibrationCertaintyTarget", cal_certainty_target)
            )
            last_rep = int(rs.get("debug_last_rep_count") or 0)
            last_cal = rs.get("debug_last_calibration_complete")
            last_gate = rs.get("debug_last_range_gate_open")
            last_peak_state = rs.get("debug_last_peak_state")
            rep_inc = rep_count > last_rep
            cal_changed = (last_cal is None) or (bool(last_cal) != bool(calibration_complete))
            gate_changed = (last_gate is None) or (bool(last_gate) != bool(range_gate_open))
            state_changed = (last_peak_state is None) or (str(last_peak_state) != state_str)
            if rep_inc or cal_changed or gate_changed:
                # region agent log
                _agent_debug_log(
                    run_id=_DEBUG_RUN_ID,
                    hypothesis_id="H3_H4",
                    location="session.py:tracking_state",
                    message="Tracking detector transition",
                    data={
                        "trackedJoint": str(rs.get("selected_angle") or ""),
                        "repCountRaw": int(rep_count),
                        "shownRepCount": int(rep_count),
                        "calibrationComplete": bool(calibration_complete),
                        "calibrationTargetReps": int(cal_target),
                        "calibrationCertainty": round(float(cal_certainty), 5),
                        "calibrationCertaintyTarget": round(float(cal_certainty_target), 5),
                        "rangeGateOpen": bool(range_gate_open),
                        "rollingRange": (
                            round(float(rolling_range), 4)
                            if rolling_range is not None
                            else None
                        ),
                        "state": state_str,
                    },
                )
                # endregion
            if state_changed:
                st_for_state = peak_detector.get_state()
                # region agent log
                _agent_debug_log(
                    run_id=_DEBUG_RUN_ID,
                    hypothesis_id="H6",
                    location="session.py:tracking_state_change",
                    message="Tracked joint peak-detector state changed",
                    data={
                        "trackedJoint": str(rs.get("selected_angle") or ""),
                        "state": state_str,
                        "repCountRaw": int(rep_count),
                        "smoothedValue": (
                            round(float(smoothed_value), 5)
                            if smoothed_value is not None
                            else None
                        ),
                        "currentPeakValue": (
                            round(float(st_for_state.get("currentPeakValue")), 5)
                            if st_for_state.get("currentPeakValue") is not None
                            else None
                        ),
                        "currentValleyValue": (
                            round(float(st_for_state.get("currentValleyValue")), 5)
                            if st_for_state.get("currentValleyValue") is not None
                            else None
                        ),
                        "hysteresis": float(
                            tuning_params.get(
                                "hysteresis",
                                get_default_tuning_params()["hysteresis"],
                            )
                        ),
                    },
                )
                # endregion
            rs["debug_last_rep_count"] = rep_count
            rs["debug_last_calibration_complete"] = calibration_complete
            rs["debug_last_range_gate_open"] = range_gate_open
            rs["debug_last_peak_state"] = state_str
            st = peak_detector.get_state()
            avg_peak = st.get("calibratedAvgPeak")
            avg_valley = st.get("calibratedAvgValley")
            if avg_peak is not None:
                avg_peak = float(avg_peak)
            if avg_valley is not None:
                avg_valley = float(avg_valley)

        shown_rep_count = rep_count
        sel_ang = rs.get("selected_angle")
        if isinstance(sel_ang, str):
            tlm = list(COMMON_ANGLES[sel_ang]["landmarks"])
        else:
            tlm = None

        cal_detail = ""
        if not calibration_complete:
            cal_detail = (
                f"Calibrating... reps {primary_rep_count}/{cal_target}  "
                f"certainty {cal_certainty * 100:.0f}%/{cal_certainty_target * 100:.0f}%"
            )
            status = cal_detail
        else:
            label = _format_angle_label(sel_ang) if sel_ang else "Joint"
            if sel_ang and sel_ang.startswith("LEFT_"):
                status = f"Tracking {label} - left side only"
            elif sel_ang and sel_ang.startswith("RIGHT_"):
                status = f"Tracking {label} - right side only"
            else:
                status = f"Tracking {label}"

        need_rg = float(tuning_params.get("minRangeGate", d_tuning["minRangeGate"]))
        if rolling_range is not None and not range_gate_open and need_rg > 0:
            if not calibration_complete:
                status = (
                    f"{status} - Move more: ~{rolling_range:.0f} deg so far "
                    f"(need at least {need_rg:.0f} deg for a rep)"
                )
            else:
                status = (
                    f"{status} - Range ~{rolling_range:.0f} deg, "
                    f"need at least {need_rg:.0f} deg"
                )

        return StepResult(
            reps=shown_rep_count,
            reps_raw=primary_rep_count,
            tracked_joint=sel_ang if isinstance(sel_ang, str) else None,
            angle_3_point_value=float(angle_value) if angle_value is not None else None,
            target_landmarks=tlm,
            tuning_params=dict(rs["tuning_params"]),
            avg_peak=avg_peak,
            avg_valley=avg_valley,
            calibration_complete=calibration_complete,
            peak_detector_state=state_str,
            smoothed_value=smoothed_value,
            range_gate_open=range_gate_open,
            rolling_range=rolling_range,
            calibration_target_reps=cal_target,
            calibration_certainty=cal_certainty,
            calibration_certainty_target=cal_certainty_target,
            phase="tracking",
            status_message=status,
            tracking_detail_message=cal_detail if not calibration_complete else "",
            selection_debug=dict(selection_debug or {}),
        )
