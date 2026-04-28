"""Session state machine: landmark stream -> rep metrics (same logic as legacy webcam loop)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import replace
from typing import Any, Optional

from flexible_rep_counter.instrumentation import (
    RepInstrumentationSink,
    RepInstrumentationSettings,
    merge_trace,
)

from flexible_rep_counter.core.math_engine import (
    PeakDetector,
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

DEFAULT_TUNING_PARAMS = get_default_tuning_params()

# Match yolo-deploy / angles.py gate so pose_dropped aligns with omitted angles in JSON.
_MIN_KEYPOINT_CONF_FOR_ANGLE = 0.3
# Temporary kill-switch: disable runtime variance-based joint recalibration/switching.
# Keep reevaluation code in place so it can be re-enabled later by flipping this.
DYNAMIC_RECALIBRATION_ENABLED = False


def _diagnose_missing_angle(
    cfg: dict[str, Any], landmarks: list[dict]
) -> tuple[str, dict[str, Any]]:
    """Why ``calculate_from_type`` may return None for this joint configuration."""
    detail: dict[str, Any] = {"landmark_indices": list(cfg.get("landmarks") or [])}
    idxs = list(cfg.get("landmarks") or [])
    confs: list[dict[str, Any]] = []
    for i in idxs:
        if i >= len(landmarks):
            detail["keypoints"] = confs
            return "missing_keypoint_index", detail
        p = landmarks[i]
        c = float(p.get("confidence", 0.0))
        confs.append({"index": i, "confidence": c})
        if c < _MIN_KEYPOINT_CONF_FOR_ANGLE:
            detail["keypoints"] = confs
            detail["min_conf_required"] = _MIN_KEYPOINT_CONF_FOR_ANGLE
            return "low_keypoint_confidence", detail
    detail["keypoints"] = confs
    return "geometry_unavailable", detail


def _peak_detector_from_tuning(tuning_params: dict[str, Any]) -> PeakDetector:
    d = DEFAULT_TUNING_PARAMS
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
    buf_list: Optional[list[list[dict]]],
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
        series = []
        for lm in buf_list or []:
            series.append(calculate_from_type(cfg["type"], cfg["landmarks"], lm))
        replay_angle_series_on_peak_detector(det, series)
    run_state["peak_detector"] = det
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
    d = DEFAULT_TUNING_PARAMS
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
        instrumentation_settings: Optional[RepInstrumentationSettings] = None,
    ) -> None:
        self._use_filter = use_pose_filter
        self._pose_pipeline = PoseFilterPipeline() if use_pose_filter else None
        self._auto_started = auto_started
        self._instrumentation_settings = instrumentation_settings
        self._instr_sink: Optional[RepInstrumentationSink] = None
        if instrumentation_settings is not None and instrumentation_settings.enabled:
            self._instr_sink = RepInstrumentationSink(instrumentation_settings)
        self._run_state: dict[str, Any] = {}
        self._last_smoothed_landmarks: Optional[list[dict]] = None
        self.reset(tuning_params=tuning_params)
        if auto_started:
            self.set_started()

    def reset(self, *, tuning_params: Optional[dict[str, Any]] = None) -> None:
        """Full reset (new exercise or second Start in visualizer)."""
        if self._instr_sink is not None:
            self._instr_sink.flush()
        tp = tuning_params if tuning_params is not None else DEFAULT_TUNING_PARAMS
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
            "buffer_list_cache": {"signature": None, "data": []},
            "variance_cache": {"signature": None, "include_debug": False, "data": {}},
        }
        if self._pose_pipeline is not None:
            self._pose_pipeline = PoseFilterPipeline()
        self._sync_detector_instrumentation_flags()

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
        self._run_state["buffer_list_cache"] = {"signature": None, "data": []}
        self._run_state["variance_cache"] = {"signature": None, "include_debug": False, "data": {}}
        self._sync_detector_instrumentation_flags()

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
        self._run_state["buffer_list_cache"] = {"signature": None, "data": []}
        self._run_state["variance_cache"] = {"signature": None, "include_debug": False, "data": {}}
        self._sync_detector_instrumentation_flags()

    @property
    def started(self) -> bool:
        return bool(self._run_state.get("started"))

    @property
    def last_smoothed_landmarks(self) -> Optional[list[dict]]:
        """Landmarks after temporal filtering from the last successful ``step_landmarks`` call."""
        return self._last_smoothed_landmarks

    @staticmethod
    def _buffer_signature(frame_buffer: deque) -> tuple[int, int, int]:
        n = len(frame_buffer)
        if n <= 0:
            return (0, 0, 0)
        return (n, id(frame_buffer[0]), id(frame_buffer[-1]))

    def _buffer_as_list(self, rs: dict[str, Any], frame_buffer: deque) -> list[list[dict]]:
        sig = self._buffer_signature(frame_buffer)
        cache = rs.get("buffer_list_cache") or {}
        if cache.get("signature") == sig:
            data = cache.get("data")
            if isinstance(data, list):
                return data
        data = list(frame_buffer)
        rs["buffer_list_cache"] = {"signature": sig, "data": data}
        return data

    def _get_variances(
        self,
        rs: dict[str, Any],
        frame_buffer: deque,
        *,
        include_debug: bool = False,
    ) -> dict[str, dict[str, Any]]:
        sig = self._buffer_signature(frame_buffer)
        cache = rs.get("variance_cache") or {}
        if (
            cache.get("signature") == sig
            and (bool(cache.get("include_debug")) or not include_debug)
            and isinstance(cache.get("data"), dict)
        ):
            return cache["data"]
        variances = compute_angle_variances_from_buffer(
            self._buffer_as_list(rs, frame_buffer),
            include_debug=include_debug,
        )
        rs["variance_cache"] = {
            "signature": sig,
            "include_debug": include_debug,
            "data": variances,
        }
        return variances

    def _sync_detector_instrumentation_flags(self) -> None:
        rs = self._run_state
        sdba = rs.get("selection_detectors_by_angle") or {}
        en = self._instr_sink is not None
        for det in sdba.values():
            try:
                det.instrumentation_enabled = bool(en)
            except Exception:
                pass
        pd = rs.get("peak_detector")
        if pd is not None:
            try:
                pd.instrumentation_enabled = bool(en)
            except Exception:
                pass

    def _instrumentation_should_emit(self, trace_context: Optional[dict[str, Any]]) -> bool:
        if self._instr_sink is None or not trace_context:
            return False
        uid = trace_context.get("user_uid")
        if not uid:
            return False
        settings = self._instr_sink.settings
        if not settings.should_emit_for_user(str(uid)):
            return False
        fi = int(trace_context.get("frame_idx", 0) or 0)
        if not settings.should_sample_frame(fi):
            return False
        return True

    def _instr_emit(
        self, trace_context: Optional[dict[str, Any]], event: dict[str, Any]
    ) -> None:
        if not self._instrumentation_should_emit(trace_context):
            return
        self._instr_sink.emit(merge_trace(event, trace_context))

    def _drain_detector_instrumentation(
        self, trace_context: Optional[dict[str, Any]], det: Any
    ) -> None:
        if det is None or not self._instrumentation_should_emit(trace_context):
            return
        try:
            events = det.drain_instrumentation_events()
        except Exception:
            return
        for ev in events:
            self._instr_emit(trace_context, ev)

    def _finalize_instrumentation_selection(
        self,
        trace_context: Optional[dict[str, Any]],
        out: StepResult,
        *,
        raw_angle_values: dict[str, Optional[float]],
        angle_values: dict[str, Optional[float]],
        sdba: dict[str, Any],
        leader_key: Optional[str],
    ) -> None:
        if not self._instrumentation_should_emit(trace_context):
            return
        seen: set[int] = set()
        for det in sdba.values():
            i = id(det)
            if i in seen:
                continue
            seen.add(i)
            self._drain_detector_instrumentation(trace_context, det)
        lk = leader_key if isinstance(leader_key, str) else None
        self._instr_emit(
            trace_context,
            {
                "event": "frame_snapshot",
                "phase": out.phase,
                "tracked_joint": out.tracked_joint,
                "leader_key": out.leader_key,
                "reps": out.reps,
                "reps_raw": out.reps_raw,
                "peak_detector_state": out.peak_detector_state,
                "range_gate_open": out.range_gate_open,
                "rolling_range": out.rolling_range,
                "calibration_complete": out.calibration_complete,
                "calibration_certainty": out.calibration_certainty,
                "calibration_target_reps": out.calibration_target_reps,
                "smoothed_angle": out.smoothed_value,
                "raw_angle": raw_angle_values.get(lk) if lk else None,
                "filtered_angle": angle_values.get(lk) if lk else None,
                "deadband_angle": None,
                "angles_raw_compact": {k: v for k, v in raw_angle_values.items() if v is not None},
                "angles_filtered_compact": {
                    k: v for k, v in angle_values.items() if v is not None
                },
            },
        )

    def _finalize_instrumentation_tracking(
        self,
        trace_context: Optional[dict[str, Any]],
        out: StepResult,
        *,
        raw_angle_value: Optional[float],
        filtered_angle_value: Optional[float],
        selected_output: Optional[dict[str, Any]],
        sdba: dict[str, Any],
        peak_detector: Any,
        sel_angle: Optional[str],
    ) -> None:
        if not self._instrumentation_should_emit(trace_context):
            return
        if (
            out.phase == "tracking"
            and sel_angle
            and filtered_angle_value is None
            and self._last_smoothed_landmarks is not None
            and isinstance(sel_angle, str)
            and sel_angle in COMMON_ANGLES
        ):
            reason, detail = _diagnose_missing_angle(
                COMMON_ANGLES[sel_angle], self._last_smoothed_landmarks
            )
            self._instr_emit(
                trace_context,
                {
                    "event": "pose_dropped",
                    "reason": reason,
                    "reason_detail": detail,
                    "tracked_joint": sel_angle,
                },
            )
        seen: set[int] = set()
        for det in sdba.values():
            i = id(det)
            if i in seen:
                continue
            seen.add(i)
            self._drain_detector_instrumentation(trace_context, det)
        if peak_detector is not None and id(peak_detector) not in seen:
            self._drain_detector_instrumentation(trace_context, peak_detector)
        feed_v: Optional[float] = None
        smooth_v: Optional[float] = None
        if selected_output:
            fv = selected_output.get("feedValue")
            sv = selected_output.get("smoothedValue")
            if fv is not None:
                feed_v = float(fv)
            if sv is not None:
                smooth_v = float(sv)
        self._instr_emit(
            trace_context,
            {
                "event": "frame_snapshot",
                "phase": out.phase,
                "tracked_joint": out.tracked_joint,
                "reps": out.reps,
                "reps_raw": out.reps_raw,
                "peak_detector_state": out.peak_detector_state,
                "range_gate_open": out.range_gate_open,
                "rolling_range": out.rolling_range,
                "calibration_complete": out.calibration_complete,
                "calibration_certainty": out.calibration_certainty,
                "calibration_target_reps": out.calibration_target_reps,
                "raw_angle": raw_angle_value,
                "filtered_angle": filtered_angle_value,
                "deadband_angle": feed_v,
                "smoothed_angle": smooth_v,
            },
        )

    def step_landmarks(
        self,
        landmarks: Optional[list[dict]],
        *,
        timestamp_ms: Optional[float] = None,
        wall_time_s: Optional[float] = None,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> StepResult:
        """
        Process one frame of 17 COCO landmarks (after any resolution scaling).

        ``landmarks`` may be None when no pose is detected.
        """
        if not self._run_state.get("started"):
            return _idle_result()

        now = wall_time_s if wall_time_s is not None else time.time()
        ts = timestamp_ms if timestamp_ms is not None else now * 1000.0
        t_step_start = time.perf_counter()

        rs = self._run_state
        tuning_params = rs["tuning_params"]
        default_tuning = DEFAULT_TUNING_PARAMS

        if not landmarks:
            self._last_smoothed_landmarks = None
            self._instr_emit(
                trace_context,
                {"event": "pose_dropped", "reason": "no_pose", "reason_detail": {}},
            )
            if rs.get("selected_angle") is not None and rs.get("peak_detector") is not None:
                tr = self._build_tracking_step_result(rs, None)
                out = replace(tr, status_message=f"No pose - {tr.status_message}")
                self._finalize_instrumentation_tracking(
                    trace_context,
                    out,
                    raw_angle_value=None,
                    filtered_angle_value=None,
                    selected_output=None,
                    sdba=rs.get("selection_detectors_by_angle") or {},
                    peak_detector=rs.get("peak_detector"),
                    sel_angle=rs.get("selected_angle"),
                )
                return out
            out = self._no_pose_step_result(
                tuning_params=tuning_params,
                tracked_joint=rs.get("selected_angle"),
                default_tuning=default_tuning,
                phase="selecting" if rs.get("selected_angle") is None else "tracking",
            )
            self._instr_emit(
                trace_context,
                {
                    "event": "frame_snapshot",
                    "phase": out.phase,
                    "tracked_joint": out.tracked_joint,
                    "reps": out.reps,
                    "reps_raw": out.reps_raw,
                    "reason": "no_pose",
                },
            )
            return out

        raw_landmarks = landmarks

        if self._pose_pipeline is not None:
            lm = self._pose_pipeline.process(landmarks, ts)
        else:
            lm = landmarks
        self._last_smoothed_landmarks = lm

        frame_buffer = rs["frame_buffer"]
        frame_buffer.append(lm)
        started_at = float(rs.get("started_at") or 0.0)
        elapsed = now - started_at
        perf_ms: dict[str, float] = {
            "detector_update_ms": 0.0,
            "variance_ms": 0.0,
            "selection_logic_ms": 0.0,
        }

        sdba: dict[str, Any] = rs.get("selection_detectors_by_angle") or {}
        if not sdba:
            sdba = {ak: _peak_detector_from_tuning(tuning_params) for ak in COMMON_ANGLES}
            for ak, det in sdba.items():
                try:
                    setattr(det, "debug_label", ak)
                except Exception:
                    pass
            rs["selection_detectors_by_angle"] = sdba
        self._sync_detector_instrumentation_flags()

        selected_angle = rs["selected_angle"]
        if selected_angle is not None and rs.get("peak_detector") is None:
            rs["peak_detector"] = sdba.get(selected_angle)

        if selected_angle is None:
            angle_values: dict[str, Optional[float]] = {}
            raw_angle_values: dict[str, Optional[float]] = {}
            detector_outputs: dict[str, dict[str, Any]] = {}
            t_det = time.perf_counter()
            for ak, cfg in COMMON_ANGLES.items():
                raw_av = calculate_from_type(cfg["type"], cfg["landmarks"], raw_landmarks)
                raw_angle_values[ak] = raw_av
                val = calculate_from_type(cfg["type"], cfg["landmarks"], lm)
                angle_values[ak] = val
                detector_outputs[ak] = sdba[ak].update(val)  # type: ignore[union-attr]
            perf_ms["detector_update_ms"] = (time.perf_counter() - t_det) * 1000.0

            rep_counts_sel = {ak: d.get_rep_count() for ak, d in sdba.items()}
            rep_dom = summarize_rep_dominance(rep_counts_sel)
            t_var = time.perf_counter()
            variances = self._get_variances(rs, frame_buffer, include_debug=False)
            perf_ms["variance_ms"] = (time.perf_counter() - t_var) * 1000.0
            joint_records = _collect_joint_records(sdba, variances)

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
                    default_tuning["calibrationReps"],
                )
            )
            total_selection_reps = int(rep_dom.get("totalReps") or 0)
            variance_fallback_ready = ready and (
                elapsed >= ANGLE_SELECTION_VARIANCE_FALLBACK_SEC
                or total_selection_reps >= cal_reps_target
            )
            locked_this_frame = False
            t_sel = time.perf_counter()

            if lock_from_dominance and leader_key:
                _apply_locked_tracking(
                    rs,
                    leader_key,
                    None,
                    tuning_params,
                    selection_detector=sdba.get(leader_key),
                )
                locked_this_frame = True
                self._sync_detector_instrumentation_flags()
            elif can_try and variance_fallback_ready:
                buf_list = self._buffer_as_list(rs, frame_buffer)
                result = determine_best_angle(
                    buf_list,
                    variances=variances,
                    include_debug=False,
                )
                tuning_params = result.get("tuningParams") or DEFAULT_TUNING_PARAMS
                rs["tuning_params"] = tuning_params
                sel = result.get("selectedAngle")
                src = str(result.get("source") or "")
                variance_ok = sel and sel in COMMON_ANGLES and src == "variance"
                if variance_ok:
                    _apply_locked_tracking(
                        rs,
                        sel,
                        None,
                        tuning_params,
                        selection_detector=sdba.get(sel),
                    )
                    locked_this_frame = True
                    self._sync_detector_instrumentation_flags()
                else:
                    rs["selected_angle"] = None
                    rs["selected_config"] = None
                    rs["peak_detector"] = None
                    rs["selection_last_attempt"] = now
            perf_ms["selection_logic_ms"] = (time.perf_counter() - t_sel) * 1000.0

            retry_at = rs.get("selection_last_attempt")
            status = _selection_status_message(
                ready=ready,
                elapsed=elapsed,
                n_frames=len(frame_buffer),
                retry_at=retry_at,
                locked_this_frame=locked_this_frame,
                selected_angle=rs.get("selected_angle"),
                run_state_selected=rs.get("selected_angle"),
                dom_ok=dom_ok,
                leader_key=leader_key,
                streak=streak,
                rep_dom=rep_dom,
            )
            perf_ms["session_total_ms"] = (time.perf_counter() - t_step_start) * 1000.0
            out = StepResult(
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
                        "calibrationReps", default_tuning["calibrationReps"]
                    )
                ),
                calibration_certainty=0.0,
                calibration_certainty_target=float(
                    rs["tuning_params"].get(
                        "calibrationCertainty",
                        default_tuning["calibrationCertainty"],
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
                    "perf_ms": perf_ms,
                },
            )
            self._finalize_instrumentation_selection(
                trace_context,
                out,
                raw_angle_values=raw_angle_values,
                angle_values=angle_values,
                sdba=sdba,
                leader_key=leader_key if isinstance(leader_key, str) else None,
            )
            return out

        # Tracking phase: update only the selected detector each frame.
        angle_values: dict[str, Optional[float]] = {}
        detector_outputs: dict[str, dict[str, Any]] = {}
        variances: dict[str, dict[str, Any]] = {}
        joint_records: dict[str, Any] = {}
        rep_dom = {
            "totalReps": 0,
            "leaderKey": selected_angle,
            "leaderReps": 0,
            "leaderShare": 0.0,
        }
        raw_angle_val: Optional[float] = None
        if isinstance(selected_angle, str) and selected_angle in COMMON_ANGLES:
            cfg = COMMON_ANGLES[selected_angle]
            t_det = time.perf_counter()
            raw_angle_val = calculate_from_type(cfg["type"], cfg["landmarks"], raw_landmarks)
            val = calculate_from_type(cfg["type"], cfg["landmarks"], lm)
            angle_values[selected_angle] = val
            detector_outputs[selected_angle] = sdba[selected_angle].update(val)  # type: ignore[index]
            perf_ms["detector_update_ms"] = (time.perf_counter() - t_det) * 1000.0
            active_detector = sdba.get(selected_angle) or rs.get("peak_detector")
            rep_value = int(active_detector.get_rep_count() or 0) if active_detector is not None else 0
            rep_dom = {
                "totalReps": rep_value,
                "leaderKey": selected_angle,
                "leaderReps": rep_value,
                "leaderShare": 1.0 if rep_value > 0 else 0.0,
            }

        switched_to: Optional[str] = None
        last_re_eval = rs.get("selection_last_reevaluate_at")
        re_eval_due = DYNAMIC_RECALIBRATION_ENABLED and (
            ANGLE_SELECTION_REEVALUATE_EVERY_SEC <= 0
            or last_re_eval is None
            or (now - float(last_re_eval)) >= ANGLE_SELECTION_REEVALUATE_EVERY_SEC
        )
        if re_eval_due:
            rs["selection_last_reevaluate_at"] = now
            t_var = time.perf_counter()
            variances = self._get_variances(rs, frame_buffer, include_debug=False)
            perf_ms["variance_ms"] = (time.perf_counter() - t_var) * 1000.0
            t_sel = time.perf_counter()
            buf_list = self._buffer_as_list(rs, frame_buffer)
            result = determine_best_angle(
                buf_list,
                variances=variances,
                include_debug=False,
            )
            perf_ms["selection_logic_ms"] = (time.perf_counter() - t_sel) * 1000.0
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
                    cfg = COMMON_ANGLES[candidate]
                    raw_angle_val = calculate_from_type(
                        cfg["type"], cfg["landmarks"], raw_landmarks
                    )
                    val = calculate_from_type(cfg["type"], cfg["landmarks"], lm)
                    angle_values[candidate] = val
                    detector_outputs[candidate] = sdba[candidate].update(val)  # type: ignore[index]
                    self._sync_detector_instrumentation_flags()
            joint_records = _collect_joint_records(sdba, variances)

        angle_value = angle_values.get(selected_angle) if isinstance(selected_angle, str) else None
        selected_output = (
            detector_outputs.get(selected_angle) if isinstance(selected_angle, str) else None
        )
        perf_ms["session_total_ms"] = (time.perf_counter() - t_step_start) * 1000.0
        out = self._build_tracking_step_result(
            rs,
            angle_value,
            detector_output=selected_output,
            tracked_joint_changed=switched_to is not None,
            selection_debug={
                "rep_dom": rep_dom,
                "joint_records": joint_records,
                "switched_to": switched_to,
                "perf_ms": perf_ms,
            },
        )
        self._finalize_instrumentation_tracking(
            trace_context,
            out,
            raw_angle_value=raw_angle_val,
            filtered_angle_value=angle_value,
            selected_output=selected_output,
            sdba=sdba,
            peak_detector=rs.get("peak_detector"),
            sel_angle=selected_angle if isinstance(selected_angle, str) else None,
        )
        return out

    def _build_tracking_step_result(
        self,
        rs: dict[str, Any],
        angle_value: Optional[float],
        *,
        detector_output: Optional[dict[str, Any]] = None,
        tracked_joint_changed: bool = False,
        selection_debug: Optional[dict[str, Any]] = None,
    ) -> StepResult:
        tuning_params = rs["tuning_params"]
        peak_detector = rs["peak_detector"]
        d_tuning = DEFAULT_TUNING_PARAMS
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
            st = peak_detector.get_state()
            avg_peak = st.get("calibratedAvgPeak")
            avg_valley = st.get("calibratedAvgValley")
            if avg_peak is not None:
                avg_peak = float(avg_peak)
            if avg_valley is not None:
                avg_valley = float(avg_valley)

        # Retroactive first-rep credit: once both a peak and valley are
        # established, count an unmatched trailing half-cycle as a complete
        # rep.  This compensates for the last rep whose closing turn is never
        # detected when the user stops moving at the end of a set.
        if (
            peak_detector is not None
            and peak_detector.peaks
            and peak_detector.valleys
            and len(peak_detector.peaks) != len(peak_detector.valleys)
        ):
            rep_count += 1

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
            tracked_joint_changed=tracked_joint_changed,
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

    def _no_pose_step_result(
        self,
        *,
        tuning_params: dict[str, Any],
        tracked_joint: Optional[str],
        default_tuning: dict[str, Any],
        phase: str,
    ) -> StepResult:
        return StepResult(
            reps=0,
            reps_raw=0,
            tracked_joint=tracked_joint,
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
                tuning_params.get("calibrationReps", default_tuning["calibrationReps"])
            ),
            calibration_certainty=0.0,
            calibration_certainty_target=float(
                tuning_params.get(
                    "calibrationCertainty",
                    default_tuning["calibrationCertainty"],
                )
            ),
            phase=phase,
            status_message="No pose",
            tracking_detail_message="",
        )
