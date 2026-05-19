# Flexible Rep Counter

AI-powered fitness rep counter using computer vision pose estimation. It analyzes joint-angle motion to pick the limb that best matches your movement, then counts repetitions with peak/valley detection—no exercise name or preset required.

## Architecture overview

- **Webcam capture**: OpenCV frames; optional resize/JPEG tuning before upload.
- **Pose inference**: YOLO pose on a remote VM; client in [`app/vm_client.py`](app/vm_client.py) (`GET /health`, `POST /predict` multipart JPEG, benchmarking hooks).
- **Runtime timing HUD**: Bottom-right panel shows VM timings (`roundtrip/upload/encode/inference`) plus local CPU stage timings from the rep engine (`session`, `detector`, `variance`) so you can watch jitter spikes in real time.
- **Concurrency**: Main thread runs capture, overlay, and angle math; one background **worker thread** sends the latest frame to the VM (queue size 1) so slow network does not block the UI loop.
- **Angle selection**: [`src/flexible_rep_counter/core/variance_angle_selector.py`](src/flexible_rep_counter/core/variance_angle_selector.py) scores per-joint angle variance over a buffer; the main loop also tracks **rep dominance** across joints (which angle’s peak detector counts the most reps) and can lock the leader after a streak. If dominance stays ambiguous, after `angle_selection.variance_fallback_sec` (in `rep_counter.toml`) the session may lock using pure **variance** selection when the retry window allows.
- **Tracking**: **One joint only** (one `COMMON_ANGLES` key, e.g. `LEFT_ELBOW` or `RIGHT_KNEE`). The opposite limb is not tracked and does not contribute to the count.
- **Rep counting**: `PeakDetector` in [`src/flexible_rep_counter/core/math_engine.py`](src/flexible_rep_counter/core/math_engine.py)—hysteresis, peak/valley margins after calibration, rolling range gate, **retroactive replay** of the observation buffer through the detector, and certainty-based locking.
- **Importable package**: Core logic lives under [`src/flexible_rep_counter/`](src/flexible_rep_counter/). OpenCV UI is in [`visualizer/opencv_runtime.py`](visualizer/opencv_runtime.py) (repo-only; run via [`main.py`](main.py)).
- **Docs**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full pipeline, call order, and math/selection details.
- **UI**: OpenCV overlay; log verbosity follows `[app].log_level` in `rep_counter.toml` (stderr when `DEBUG`).

## Use as a library (other projects)

**This package is not published on PyPI.** `pip install flexible-rep-counter` / `pip index versions flexible-rep-counter` will not find it until you publish it yourself. Install from a **local clone**, a **built wheel**, or a **Git URL** instead.

```bash
# Editable (recommended while developing both repos)
pip install -e "/path/to/flexible-rep-counter"

# Or build a wheel in this repo (python3 -m build), then:
pip install /path/to/flexible-rep-counter/dist/flexible_rep_counter-*-py3-none-any.whl

# Optional OpenCV + requests for the demo app
pip install -e "/path/to/flexible-rep-counter[viz]"
```

**`requirements.txt`** (PEP 508 direct URL or path):

```text
flexible-rep-counter @ file:///absolute/path/to/flexible-rep-counter
```

Or a relative path from the file that contains the requirement (same parent layout as your project).

**`pyproject.toml` dependency** (e.g. sibling checkout):

```toml
[project]
dependencies = [
  "flexible-rep-counter @ file:///absolute/path/to/flexible-rep-counter",
]
# or: { path = "../flexible-rep-counter", editable = true }  # tool-specific
```

```python
from flexible_rep_counter import RepCounterSession, keypoints_numpy_to_landmarks

session = RepCounterSession(auto_started=True, use_pose_filter=False)
landmarks = keypoints_numpy_to_landmarks(keypoints_np)  # (17, 3) float array
step = session.step_landmarks(landmarks, timestamp_ms=...)
# step.reps, step.tracked_joint, step.angle_3_point_value, step.avg_peak, step.avg_valley, ...
```

**Matching this repo’s angle math** (COCO 17 landmarks as `list[dict]` with `x`, `y`, `confidence`): use the same entry points as the session—[`calculate_from_type`](src/flexible_rep_counter/core/math_engine.py) plus joint configs in [`COMMON_ANGLES`](src/flexible_rep_counter/core/variance_angle_selector.py) (e.g. `angle_3_point` and landmark index triples). Example:

```python
from flexible_rep_counter.core import COMMON_ANGLES, calculate_from_type

cfg = COMMON_ANGLES["LEFT_ELBOW"]
deg = calculate_from_type(cfg["type"], cfg["landmarks"], landmarks)
```

**ai-personal-trainer** (sibling checkout): `pyproject.toml` includes an editable `flexible-rep-counter` path dependency; run `uv sync` or `pip install -e ../flexible-rep-counter` if needed.

## Python local app (webcam + VM)

Runs on your machine: frames go to the VM for keypoints; angles and reps are computed locally with an on-screen overlay.

### Setup

```bash
cd /path/to/flexible-rep-counter
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .          # register the flexible_rep_counter package
```

### Run

```bash
python main.py
```

Optional flags: `--no-health-check`, `--benchmark-log FILE`, `--resize-width W`, `--jpeg-quality Q`, `--no-validate-response` (defaults follow `[predict]` in `rep_counter.toml` for resize/quality; validation defaults to on unless this flag is set).

### Session flow

1. **Angle selection**: Buffer frames (cap `angle_selection.max_buffer_frames`) until `angle_selection.min_sec` and `angle_selection.min_frames`. Per-frame, lightweight peak detectors on all candidate angles feed **rep dominance** stats; variance/ROM gates pick a stable joint. Lock when dominance + variance agree for `angle_selection.dominance_streak_frames`, or after `angle_selection.variance_fallback_sec` via pure variance selection if needed. Retries use `angle_selection.retry_interval_sec`.
2. **Calibration**: First reps establish peak/valley baselines; the buffer is **replayed** through the same detectors so displayed reps include motion from the selection window. Locking uses `rep.calibration_certainty` and optionally `rep.calibration_force_extra_reps`.
3. **Tracking**: Counts with margins, range gate, and `rep.min_interval_ms`. Quit with `q` or Escape in the video window.

## Dynamic recalibration internals

This section is the shortest "how it works" map for first-joint selection, confidence math, and runtime handoff to a recalibrated candidate joint.

### 1) First joint selection (variance-window regulator)

- The selector computes per-joint smoothed angle histories and scores each joint with:
  - `medianWindowVariance` (consistent movement across windows)
  - `activeWindowCount` (how many windows are truly active)
  - `smoothedRangeDeg` (effective ROM in degrees)
- A joint is eligible only when all gates pass:
  - `activeWindowCount >= angle_selection.min_active_windows`
  - `medianWindowVariance >= angle_selection.min_variance` (or joint override)
  - `smoothedRangeDeg >= angle_selection.min_range_deg` (or joint override)
- The top candidate must also beat runner-up ambiguity using `second_best_ratio` (with a relaxed ratio for across-body runner-ups).
- Final lock also requires keypoint confidence:
  - `avg_confidence(selected_joint_landmarks) >= 0.5`
- During initial selection, rep-dominance can also lock the joint when one joint leads rep share for enough frames (`dominance_fraction`, `dominance_streak_frames`, `min_leading_reps`).

### 2) Math-engine confidence and calibration lock

- The peak detector computes calibration certainty from extrema stability:
  - `amplitude = avgPeak - avgValley`
  - `combined_jitter_ratio = (peakStd + valleyStd) / amplitude`
  - `certainty = clamp(1 - combined_jitter_ratio, 0, 1)`
- Calibration locks when either condition passes:
  - `rep_count >= calibration_reps` **and** `certainty >= calibration_certainty`
  - or force-lock at `rep_count >= calibration_reps + calibration_force_extra_reps`
- After calibration, rep acceptance is stricter:
  - Peak/valley must pass margin gates (`peak_margin_pct`, `valley_margin_pct`) relative to calibrated ROM.
  - Rolling range gate must be open: p95-p5 spread over `range_window_frames` is at least `min_range_gate`.
  - Fast duplicate reps are blocked by `min_interval_ms`.

### 3) Runtime candidate recalibration and switch

- Reevaluation runs every `angle_selection.reevaluate_every_sec`.
- Candidate source is usually variance, with two fallback paths:
  - **stale-selected fallback**: selected joint is stale for repeated reevaluations and opposite side shows consistent variance activity.
  - **range-gate fallback**: selected joint range gate stays closed and opposite side variance is stronger.
- Pre-switch gate requires cooldown and strength checks:
  - `now - last_switch >= angle_selection.switch_min_sec`
  - Candidate variance must be stronger by `angle_selection.switch_variance_ratio` (unless stale override path applies).
- Candidate detector is rebuilt by replaying its recent history. If already calibrated, switch is immediate; otherwise candidate becomes pending and is updated until calibration completes.

### 4) Candidate pre-calibrated reps and bend-switch fine tuning

- While a candidate is pending, the session stores both baselines:
  - incumbent shown/raw counts at pending start
  - candidate shown/raw counts at pending start
- On activation, pending candidate reps are added only when all are true:
  - candidate was observed
  - incumbent did **not** advance rep count during pending
  - incumbent did **not** show meaningful motion (`pending_switch_incumbent_moving == False`)
- "Meaningful motion" is flagged when incumbent angle span during pending reaches:
  - `PENDING_SWITCH_INCUMBENT_MOVEMENT_DEG` (default `12.0` deg)
- Stale/bend inactivity forcing variables:
  - `STALE_SWITCH_MIN_STALE_REEVALS`
  - `STALE_SWITCH_SELECTED_RECENT_WINDOW`
  - `STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG`
  - `STALE_SWITCH_MIN_CLOSED_STREAK`
  - `STALE_SWITCH_FORCE_AFTER_STALE_REEVALS`
  - `STALE_SWITCH_FORCE_MIN_OPPOSITE_VAR`

These gates are covered by `tests/test_dynamic_recalibration.py`, including monotonic-count handoff, pending-candidate rep carryover, and stale/range-gate fallback switching.

## Configuration

**Source of truth:** [`rep_counter.toml`](rep_counter.toml) in the repo root, or the current working directory, or any parent directory. Set `FLEXIBLE_REP_COUNTER_CONFIG` to an absolute path to use a different file.

Sections:

| Section | Role |
|---------|------|
| `[app]` | `log_level` (`DEBUG`, `INFO`, …) |
| `[vm]` | `direct_url` (required for the visualizer), `timeout_sec`, `health_timeout_sec` |
| `[predict]` | `resize_width`, `jpeg_quality` |
| `[rep]` | Peak detector tuning: hysteresis, margins, calibration, `min_interval_ms`, etc. |
| `[angle_selection]` | Selection window, dominance, `variance_fallback_sec`, global variance/range thresholds |
| `[angle_selection.joints.<NAME>]` | Per-joint overrides (e.g. `LEFT_ELBOW`) for `min_variance`, `min_range_deg`, `second_best_ratio` |

The root `.env` is loaded for compatibility (e.g. `FLEXIBLE_REP_COUNTER_CONFIG`); tuning keys live in TOML, not duplicate env vars. Response validation for `/predict` is on by default in code; use `--no-validate-response` to disable.

### VM API (client contract)

The client sends **`POST {base_url}/predict`** with multipart form field **`file`** = JPEG bytes (`image/jpeg`). The JSON response should include top-level `inference_ms` (optional) and either:

- `person_*` objects with `keypoints` as a dict of COCO-style names → `{x, y, conf|confidence}`, or  
- a structure parseable by [`_parse_keypoints`](app/vm_client.py) (see that function for supported shapes).

Keypoints are converted to a list of 17 `{x, y, confidence}` entries in COCO order for local angle math.
