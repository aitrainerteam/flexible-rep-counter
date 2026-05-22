# Architecture: data flow and execution order

This document describes how the flexible rep counter moves data from the webcam to a rep count on screen, and how the math/selection layers fit together.

## High-level pipeline

```text
Webcam (OpenCV)
    -> main thread: BGR frame
    -> Queue(maxsize=1) -> background thread -> HTTP POST /predict (JPEG multipart)
    -> latest_pose snapshot { landmarks, benchmark, sent_hw, validation_issues }
    -> scale landmarks to display resolution if sent size differs
    -> PoseFilterPipeline.process (One Euro + velocity clamp + confidence blend)
    -> draw_skeleton (overlay)
    -> [selection phase OR tracking phase]
         selection: buffer landmarks, run N PeakDetectors on all COMMON_ANGLES, compute variances, lock one angle
         tracking: active joint drives displayed reps, while all joint motion states keep updating for recalibration confidence
    -> UI overlay (reps, state, status) + benchmark HUD
```

**Threads**

| Thread        | Role |
|---------------|------|
| Main          | `cap.read`, pose filtering, angle math, rep logic, frame composition |
| `_pose_worker`| Dequeue latest frame, `send_frame` to VM, write `latest_pose[0]` |
| `_DisplayWindow` | `imshow`/`waitKey`, keyboard + click interactions, non-blocking rendering |

Only one frame is queued at a time (`Queue(maxsize=1)`), so the VM always sees a recent frame without backlog.

---

## Entry and runtime setup

1. **`main.main()`** ([`main.py`](main.py)) parses CLI args and calls **`run_webcam_loop`** ([`visualizer/opencv_runtime.py`](visualizer/opencv_runtime.py)).
2. **`run_webcam_loop`**
   - Loads tuning via **`get_default_tuning_params()`** ([`app/config.py`](app/config.py)).
   - Optionally **`check_vm_health`** ([`app/vm_client.py`](app/vm_client.py)) → `GET {base}/health`.
   - Opens camera with backend probing + bad-frame rejection and starts a display worker thread.
   - Builds **`run_state`** and a `RepCounterSession` instance (the session owns selection/tracking internals).
   - Starts **`_pose_worker`** thread.
   - Enters the **per-frame loop** (below).

---

## Per-frame loop (main thread) — call order

For each frame after the user clicks **Start**:

1. **`frame_queue.put_nowait(frame_bgr.copy())`** — non-blocking; drops if full.
2. Read **`snap = latest_pose[0]`** (may be stale by one frame; that is intentional for latency).
3. **`_update_vm_metrics(snap)`** — updates FPS-style inference rate, optional benchmark log file.
4. If **`snap["landmarks"]` is None** → draw “No pose”, continue.
5. **`_scale_landmarks_to_display`** — if the JPEG sent to the VM was resized, scale x/y back to the display frame.
6. **`PoseFilterPipeline.process(raw_scaled, timestamp_ms)`** ([`src/flexible_rep_counter/core/pose_filters.py`](src/flexible_rep_counter/core/pose_filters.py)) — temporal smoothing per keypoint, then velocity clamp, then short history interpolation for low-confidence points.
7. **`draw_skeleton(frame_bgr, landmarks)`** ([`app/skeleton_overlay.py`](app/skeleton_overlay.py)).

Then branch (inside `RepCounterSession.step_landmarks`):

### A) Selection phase — `phase == "selecting"`

Goal: observe motion, then lock one active tracked joint while still collecting all-joint motion stats.

1. Buffer landmarks and maintain one lightweight detector per candidate joint.
2. Compute per-joint dominance and variance gates from `COMMON_ANGLES`.
3. Lock when dominance/variance converge and streak thresholds are satisfied.
4. If dominance remains ambiguous past the configured fallback window, allow variance fallback.

### B) Tracking phase — `phase == "tracking"`

1. Active joint `PeakDetector` emits displayed reps and calibration status.
2. All joint `JointMotionState` records continue updating to support reevaluation and handoff confidence.
3. Reevaluation can trigger immediate or pending switch to a stronger candidate joint (subject to cooldown and confidence gates).
4. `StepResult` emits edge telemetry (`tracked_joint_changed`, calibration start/lock flags, low-fps fields) for UI and instrumentation.

### Reset (second Start click)

Mouse handler clears selection, `peak_detector`, and the frame buffer so a new session can pick a (possibly different) limb.

---

## Math engine ([`src/flexible_rep_counter/core/math_engine.py`](src/flexible_rep_counter/core/math_engine.py))

### Geometry

- **`calculate_angle(a, b, c)`** — interior angle at **B** for points A–B–C, degrees 0–180. Used for all **`angle_3_point`** entries in `COMMON_ANGLES` (elbow, knee, shoulder, hip, etc.).
- **`calculate_from_type`** — dispatches on `calc_type` (`angle_3_point`, vertical/horizontal axis, normalized distance). The live app uses **`angle_3_point`** only for `COMMON_ANGLES`.

### PeakDetector (rep execution)

State machine:

- **States**: `neutral` → `going_up` / `going_down` alternating as the **smoothed** angle moves by more than **`hysteresis`** from the running extremum.
- **Smoothing**: EMA on the input angle; optional **delta deadband** before EMA to drop tiny jitter.
- **Peak / valley events**: When reversing direction, if **`min_peak_distance`** frames have passed since the last event, a peak or valley may be recorded.
- **Range gate**: Recent window of smoothed values must have p95−p5 spread ≥ **`min_range_gate_degrees`** before reps count (stops micro-bounces from counting).
- **Calibration**: First **`calibration_reps`** completed rep cycles record peaks/valleys **without** strict margin checks; then **average peak** and **average valley** are **locked**. After lock, new peaks must stay near the average peak (within **`peak_margin_pct`** × calibrated ROM) and valleys near average valley (within **`valley_margin_pct`** × ROM), so ROM drift does not duplicate counts.  Margins are expressed as fractions (0–1) of the calibrated ROM (`avg_peak − avg_valley`).
- **Certainty**: From spread of recorded peaks/valleys vs amplitude; can delay lock until **`calibration_force_extra_reps`** if certainty is low.
- **Rep count**: Derived from paired peak/valley sequences (`min(len(peaks), len(valleys))` style progression inside the implementation); **`min_rep_interval_ms`** caps how fast reps can increment.

**`replay_angle_series_on_peak_detector`** feeds a stored list of angles (including `None` samples) through the same `update` path so post-lock behavior matches what would have happened if tracking had started at buffer start.

### Variance / consistency helpers (selection only)

- **`smooth_angle_series`** — short moving average before variance stats.
- **`calculate_variance`**, **`compute_robust_variance`** — classical and IQR-filtered variance on the smoothed series.
- **`compute_consistent_variance_score`** — splits the series into a few time windows, variances per window, **median** of window variances and count of “active” windows (variance above `MIN_VARIANCE_THRESHOLD`). Favors **rhythmic, repeated** motion over one spike.

---

## Angle tracking and joint decision ([`src/flexible_rep_counter/core/variance_angle_selector.py`](src/flexible_rep_counter/core/variance_angle_selector.py))

### `COMMON_ANGLES`

Fixed map of logical names → `{ type: "angle_3_point", landmarks: [i,j,k] }` using **COCO 17** indices. Each name encodes **side** (`LEFT_` / `RIGHT_`) so locking **`LEFT_ELBOW`** never counts **`RIGHT_ELBOW`**.

### `compute_angle_variances_from_buffer`

For each angle key:

1. Walk frames; skip frames where any involved landmark has confidence &lt; **`FRAME_MIN_CONFIDENCE`**.
2. Collect valid **`calculate_from_type`** values into `history`.
3. If enough samples: smooth → **`compute_consistent_variance_score`**, ROM = max−min of smoothed series, plus raw/robust variance for debugging.

### `_get_top_candidate`

Ranks angles by **median window variance** after passing **min active windows**, **min variance**, **min ROM**, and **second-best ratio** (winner must be sufficiently ahead of runner-up). This is the “most regular / consistent” joint in a statistical sense.

### `dominance_conditions_met`

Ensures the **rep leader** during selection matches that statistical winner when a unique top candidate exists—so the limb that is **actually cycling** wins, not a noisy idle joint.

### `determine_best_angle`

Returns `{ selectedAngle, source, tuningParams, debug }`. Used for **variance fallback** and **tracking re-eval** in the session.

---

## VM client ([`app/vm_client.py`](app/vm_client.py))

- **`send_frame`**: encode JPEG, `POST .../predict` multipart `file`, parse JSON into 17 landmarks.
- **`validate_predict_response`**: optional shape checks.
- **`_parse_keypoints`**: tolerant parsing of several server JSON layouts.

---

## Configuration source of truth

Tuning is loaded from **`rep_counter.toml`** (see **[`README.md`](README.md)**) in **[`flexible_rep_counter/core/settings.py`](src/flexible_rep_counter/core/settings.py)**; **[`app/config.py`](app/config.py)** re-exports those values and requires a non-empty **`[vm].direct_url`** for the visualizer.

---

## Possible future improvements (universal counter)

- **Temporal consistency**: penalize switching `selected_angle` after lock unless user explicitly resets.
- **3D / multi-view**: 2D angles are view-dependent; elevation or dual camera would reduce ambiguity.
- **Torso-normalized angles**: express limb angles relative to hip–shoulder axis for camera roll invariance.
- **Explicit “which side” UI**: rare ambiguity when variance ties; ask user once instead of auto-guess.
- **Async**: second thread for VM is usually enough; extra threads for CPU math rarely beat cost of GIL/contention unless moving heavy work to NumPy/C extensions in batch.
