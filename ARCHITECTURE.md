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
         tracking: one PeakDetector.update on the locked angle only
    -> UI overlay (reps, state, status) + benchmark HUD
```

**Threads**

| Thread        | Role |
|---------------|------|
| Main          | `cap.read`, UI (`imshow`, `waitKey`), pose filtering, angle math, rep logic |
| `_pose_worker`| Dequeue latest frame, `send_frame` to VM, write `latest_pose[0]` |

Only one frame is queued at a time (`Queue(maxsize=1)`), so the VM always sees a recent frame without backlog.

---

## Entry and runtime setup

1. **`main.main()`** ([`main.py`](main.py)) parses CLI args and calls **`run_webcam_loop`** ([`visualizer/opencv_runtime.py`](visualizer/opencv_runtime.py)).
2. **`run_webcam_loop`**
   - Loads tuning via **`get_default_tuning_params()`** ([`app/config.py`](app/config.py)).
   - Optionally **`check_vm_health`** ([`app/vm_client.py`](app/vm_client.py)) → `GET {base}/health`.
   - Opens camera, builds **`run_state`** (selection buffer, selection detectors, dominance streak, `selected_angle`, `peak_detector`, etc.).
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
6. **`PoseFilterPipeline.process(raw_scaled, timestamp_ms)`** ([`app/pose_filters.py`](app/pose_filters.py)) — temporal smoothing per keypoint, then velocity clamp, then short history interpolation for low-confidence points.
7. **`draw_skeleton(frame_bgr, landmarks)`** ([`app/skeleton_overlay.py`](app/skeleton_overlay.py)).

Then branch:

### A) Selection phase — `run_state["selected_angle"] is None`

Goal: observe motion, then **lock exactly one** entry of **`COMMON_ANGLES`** (one joint / one side). The opposite limb is **never** tracked or displayed as a second counter.

1. **`frame_buffer.append(landmarks)`** — bounded by `ANGLE_SELECTION_MAX_BUFFER_FRAMES`.
2. **Selection PeakDetectors** — lazy-init `selection_detectors_by_angle`: one **`PeakDetector`** per key in **`COMMON_ANGLES`** ([`app/variance_angle_selector.py`](app/variance_angle_selector.py)).
3. For **each** candidate angle key, each frame:
   - **`calculate_from_type(type, landmark_indices, landmarks)`** ([`app/math_engine.py`](app/math_engine.py)) → scalar angle (or None).
   - **`detector.update(value)`** — same peak/valley machine as live tracking (see “PeakDetector” below). This produces **per-joint rep-like events** used only for **dominance**, not the final user count yet.
4. **`summarize_rep_dominance(rep_counts_sel)`** — among joints with rep_count &gt; 0, find the **leader** (max count) and its **share** of total selection-phase rep events.
5. **`compute_angle_variances_from_buffer(buf_list)`** — builds per-angle statistics from the buffered landmark sequence (see “Angle variance / joint scoring”).
6. **`dominance_conditions_met(variances, rep_dom, ...)`** — true when:
   - leader share &gt; `ANGLE_SELECTION_DOMINANCE_FRACTION`,
   - leader has at least `ANGLE_SELECTION_MIN_LEADING_REPS` reps,
   - leader passes **`passes_consistent_variance_gate`** (multi-window activity + ROM),
   - and **`_get_top_candidate(variances)`** agrees with the leader (variance winner == rep leader), so noise on the idle arm does not steal the lock.
7. **Dominance streak** — if conditions hold for the same `leader_key` for `ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES` consecutive frames (and wall/frame minimums are met), **lock**:
   - **`_apply_locked_tracking(..., selection_detector=sdba.get(leader_key))`** — reuse the **same** `PeakDetector` instance that observed the selection window so counts stay continuous.
8. **Variance fallback** (only if dominance never stabilizes): when **`can_try`** (retry interval) and elapsed ≥ `ANGLE_SELECTION_VARIANCE_FALLBACK_SEC`, call **`determine_best_angle(buf_list)`**. If it returns `source=="variance"`:
   - **Limb alignment**: if any selection-phase rep events exist (`totalReps > 0`), require **`selectedAngle == leader_key`** so the high-variance pick cannot be the **idle** arm while the user reps on the other side.
   - **`_apply_locked_tracking(..., selection_detector=None)`** — new detector + **`replay_angle_series_on_peak_detector`** over the buffer.
9. **`_selection_status_message`** drives the overlay string during this phase.

### B) Tracking phase — `selected_angle` is set

1. **`angle_value = calculate_from_type(selected_config["type"], selected_config["landmarks"], landmarks)`**.
2. **`out = peak_detector.update(angle_value)`** — single detector, single joint; the other arm’s angles are **not** fed into any detector.
3. **Displayed rep count** — `rep_count` from `out["repCount"]`, but shown as **0** until **`calibrationComplete`** (user sees calibration progress without flashing false totals).
4. Status line reminds that only **left** or **right** side is locked (for `LEFT_*` / `RIGHT_*` keys).

### Reset (second Start click)

Mouse handler clears selection, `peak_detector`, and the frame buffer so a new session can pick a (possibly different) limb.

---

## Math engine ([`app/math_engine.py`](app/math_engine.py))

### Geometry

- **`calculate_angle(a, b, c)`** — interior angle at **B** for points A–B–C, degrees 0–180. Used for all **`angle_3_point`** entries in `COMMON_ANGLES` (elbow, knee, shoulder, hip, etc.).
- **`calculate_from_type`** — dispatches on `calc_type` (`angle_3_point`, vertical/horizontal axis, normalized distance). The live app uses **`angle_3_point`** only for `COMMON_ANGLES`.

### PeakDetector (rep execution)

State machine:

- **States**: `neutral` → `going_up` / `going_down` alternating as the **smoothed** angle moves by more than **`hysteresis`** from the running extremum.
- **Smoothing**: EMA on the input angle; optional **delta deadband** before EMA to drop tiny jitter.
- **Peak / valley events**: When reversing direction, if **`min_peak_distance`** frames have passed since the last event, a peak or valley may be recorded.
- **Range gate**: Recent window of smoothed values must have p95−p5 spread ≥ **`min_range_gate_degrees`** before reps count (stops micro-bounces from counting).
- **Calibration**: First **`calibration_reps`** completed rep cycles record peaks/valleys **without** strict margin checks; then **average peak** and **average valley** are **locked**. After lock, new peaks must stay near the average peak (within **`peak_margin`**) and valleys near average valley (within **`valley_margin`**), so ROM drift does not duplicate counts.
- **Certainty**: From spread of recorded peaks/valleys vs amplitude; can delay lock until **`calibration_force_extra_reps`** if certainty is low.
- **Rep count**: Derived from paired peak/valley sequences (`min(len(peaks), len(valleys))` style progression inside the implementation); **`min_rep_interval_ms`** caps how fast reps can increment.

**`replay_angle_series_on_peak_detector`** feeds a stored list of angles (including `None` samples) through the same `update` path so post-lock behavior matches what would have happened if tracking had started at buffer start.

### Variance / consistency helpers (selection only)

- **`smooth_angle_series`** — short moving average before variance stats.
- **`calculate_variance`**, **`compute_robust_variance`** — classical and IQR-filtered variance on the smoothed series.
- **`compute_consistent_variance_score`** — splits the series into a few time windows, variances per window, **median** of window variances and count of “active” windows (variance above `MIN_VARIANCE_THRESHOLD`). Favors **rhythmic, repeated** motion over one spike.

---

## Angle tracking and joint decision ([`app/variance_angle_selector.py`](app/variance_angle_selector.py))

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

Returns `{ selectedAngle, source, tuningParams, debug }`. Used for **variance fallback** only in the loop; **`exercise`** metadata can enable isometric fallback paths when passed (webcam loop passes `exercise=None`).

---

## VM client ([`app/vm_client.py`](app/vm_client.py))

- **`send_frame`**: encode JPEG, `POST .../predict` multipart `file`, parse JSON into 17 landmarks.
- **`validate_predict_response`**: optional shape checks.
- **`_parse_keypoints`**: tolerant parsing of several server JSON layouts.

---

## Configuration source of truth

All thresholds are documented in **[`README.md`](README.md)** and loaded in **[`app/config.py`](app/config.py)** from environment (and `.env`).

---

## Possible future improvements (universal counter)

- **Temporal consistency**: penalize switching `selected_angle` after lock unless user explicitly resets.
- **3D / multi-view**: 2D angles are view-dependent; elevation or dual camera would reduce ambiguity.
- **Torso-normalized angles**: express limb angles relative to hip–shoulder axis for camera roll invariance.
- **Explicit “which side” UI**: rare ambiguity when variance ties; ask user once instead of auto-guess.
- **Async**: second thread for VM is usually enough; extra threads for CPU math rarely beat cost of GIL/contention unless moving heavy work to NumPy/C extensions in batch.
