# Flexible Rep Counter

AI-powered fitness rep counter using computer vision pose estimation. It analyzes joint-angle motion to pick the limb that best matches your movement, then counts repetitions with peak/valley detection—no exercise name or preset required.

## Architecture overview

- **Webcam capture**: OpenCV frames; optional resize/JPEG tuning before upload.
- **Pose inference**: YOLO pose on a remote VM; client in [`app/vm_client.py`](app/vm_client.py) (`GET /health`, `POST /predict` multipart JPEG, benchmarking hooks).
- **Concurrency**: Main thread runs capture, overlay, and angle math; one background **worker thread** sends the latest frame to the VM (queue size 1) so slow network does not block the UI loop.
- **Angle selection**: [`app/variance_angle_selector.py`](app/variance_angle_selector.py) scores per-joint angle variance over a buffer; the main loop also tracks **rep dominance** across joints (which angle’s peak detector counts the most reps) and can lock the leader after a streak. If dominance stays ambiguous, after `ANGLE_SELECTION_VARIANCE_FALLBACK_SEC` it may lock using the variance winner **only if** that winner matches the rep leader when any reps exist (so the idle arm does not get locked).
- **Tracking**: **One joint only** (one `COMMON_ANGLES` key, e.g. `LEFT_ELBOW` or `RIGHT_KNEE`). The opposite limb is not tracked and does not contribute to the count.
- **Rep counting**: `PeakDetector` in [`src/flexible_rep_counter/core/math_engine.py`](src/flexible_rep_counter/core/math_engine.py) (also re-exported from [`app/math_engine.py`](app/math_engine.py))—hysteresis, peak/valley margins after calibration, rolling range gate, **retroactive replay** of the observation buffer through the detector, and certainty-based locking.
- **Importable package**: Core logic lives under [`src/flexible_rep_counter/`](src/flexible_rep_counter/). OpenCV UI is in [`visualizer/opencv_runtime.py`](visualizer/opencv_runtime.py) (repo-only; run via [`main.py`](main.py)).
- **Docs**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full pipeline, call order, and math/selection details.
- **UI**: OpenCV overlay; logs go to stderr when `LOG_LEVEL=DEBUG`.

## Use as a library (other projects)

```bash
pip install -e ".[viz]"   # from this repo root; [viz] adds OpenCV + requests for the demo app
```

```python
from flexible_rep_counter import RepCounterSession, keypoints_numpy_to_landmarks

session = RepCounterSession(auto_started=True, use_pose_filter=False)
landmarks = keypoints_numpy_to_landmarks(keypoints_np)  # (17, 3) float array
step = session.step_landmarks(landmarks, timestamp_ms=...)
# step.reps, step.tracked_joint, step.angle_3_point_value, step.avg_peak, step.avg_valley, ...
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
pip install -e .          # register the flexible_rep_counter package (needed for app shims)
```

### Run

```bash
python main.py
```

Optional flags: `--no-health-check`, `--benchmark-log FILE`, `--resize-width W`, `--jpeg-quality Q`, `--no-validate-response` (see [`main.py`](main.py)).

### Session flow

1. **Angle selection**: Buffer frames (cap `ANGLE_SELECTION_MAX_BUFFER_FRAMES`) until `ANGLE_SELECTION_MIN_SEC` and `ANGLE_SELECTION_MIN_FRAMES` are met. Per-frame, lightweight peak detectors on all candidate angles feed **rep dominance** stats; variance/ROM gates pick a stable joint. Lock when dominance + variance agree for `ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES`, or after `ANGLE_SELECTION_VARIANCE_FALLBACK_SEC` via pure variance selection if needed. Retries use `ANGLE_SELECTION_RETRY_INTERVAL_SEC`.
2. **Calibration**: First reps establish peak/valley baselines; the buffer is **replayed** through the same detectors so displayed reps include motion from the selection window. Locking uses `REP_CALIBRATION_CERTAINTY` and optionally `REP_CALIBRATION_FORCE_EXTRA_REPS`.
3. **Tracking**: Counts with margins, range gate, and `REP_MIN_INTERVAL_MS`. Quit with `q` or Escape in the video window.

## Configuration

Create a `.env` in the project root. The app loads it automatically ([`app/config.py`](app/config.py)).

### VM

| Variable | Description |
|----------|-------------|
| `YOLO_VM_DIRECT_URL` | Primary VM base URL (used if set) |
| `YOLO_VM_TARGET_URL` | Fallback VM base URL if `YOLO_VM_DIRECT_URL` is empty |
| `VM_TIMEOUT_SEC` | Request timeout for pose calls (default `5.0`) |
| `VM_HEALTH_TIMEOUT_SEC` | Health check timeout (default `5.0`) |
| `PREDICT_RESIZE_WIDTH` | Resize width before JPEG (`0` = full frame) |
| `PREDICT_JPEG_QUALITY` | JPEG quality `1`–`100` (default `85`) |
| `PREDICT_VALIDATE_RESPONSE` | `1`/`0` — validate JSON shape on responses |

At least one of `YOLO_VM_DIRECT_URL` or `YOLO_VM_TARGET_URL` must be set or the app exits on startup.

### Rep counting

| Variable | Description | Default |
|----------|-------------|---------|
| `REP_HYSTERESIS` | Min angle change (°) to register a direction change | `5` |
| `REP_MIN_PEAK_DISTANCE` | Min frames between peak/valley events | `5` |
| `REP_SMOOTHING_FACTOR` | EMA alpha for angle smoothing (0–1) | `0.45` |
| `REP_PEAK_MARGIN` | Peak must be within this many ° of session max avg | `15` |
| `REP_VALLEY_MARGIN` | Valley must be within this many ° of session min avg | `15` |
| `REP_MIN_RANGE_GATE` | Rolling motion span must exceed this (°) before reps count; `0` off | `15` |
| `REP_RANGE_WINDOW_FRAMES` | Frames in rolling span window | `90` |
| `REP_RANGE_MIN_SAMPLES` | Min samples before span is trusted | `12` |
| `REP_ANGLE_DELTA_DEADBAND` | Ignore sub-threshold ° changes before EMA; `0` off | `0` |
| `REP_CALIBRATION_REPS` | Reps without strict margins before lock | `3` |
| `REP_CALIBRATION_CERTAINTY` | Min certainty (0–1) to lock baselines; `0` = lock after min reps only | `0.5` |
| `REP_CALIBRATION_FORCE_EXTRA_REPS` | Extra counted reps if certainty still low after min reps | `2` |
| `REP_MIN_INTERVAL_MS` | Min time between reps; `0` off | `400` |

### Angle selection (before rep calibration)

| Variable | Description | Default |
|----------|-------------|---------|
| `ANGLE_SELECTION_MIN_SEC` | Minimum wall time for selection window | `7.0` |
| `ANGLE_SELECTION_MIN_FRAMES` | Minimum frames in window | `70` |
| `ANGLE_SELECTION_MAX_BUFFER_FRAMES` | Cap on buffered frames | `400` |
| `ANGLE_SELECTION_RETRY_INTERVAL_SEC` | Retry if gates not met | `2.0` |
| `ANGLE_SELECTION_DOMINANCE_FRACTION` | Leader’s rep share must exceed this (strict `>` in code) | `~0.667` |
| `ANGLE_SELECTION_MIN_LEADING_REPS` | Minimum reps on leader joint | `2` |
| `ANGLE_SELECTION_DOMINANCE_STREAK_FRAMES` | Frames dominance + variance must hold before lock | `36` |
| `ANGLE_SELECTION_VARIANCE_FALLBACK_SEC` | After this many seconds, allow pure-variance lock | `14.0` |
| `ANGLE_SELECTION_MIN_VARIANCE` | Min variance (global; per-joint: `_LEFT_ELBOW`, etc.) | `6.0` |
| `ANGLE_SELECTION_MIN_RANGE_DEG` | Min observed range (°) for a candidate | `16.0` |
| `ANGLE_SELECTION_SECOND_BEST_RATIO` | Top candidate must beat runner-up by this ratio | `1.15` |
| `ANGLE_SELECTION_MIN_ACTIVE_WINDOWS` | Active variance windows required (see code) | `4` |
| `ANGLE_SELECTION_SMOOTH_WINDOW` | Smoothing window for selection series | `5` |

### Debugging

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `DEBUG` |

When `LOG_LEVEL=DEBUG`, extra log lines are written to stderr (see [`app/debug_console.py`](app/debug_console.py)).

### VM API (client contract)

The client sends **`POST {base_url}/predict`** with multipart form field **`file`** = JPEG bytes (`image/jpeg`). The JSON response should include top-level `inference_ms` (optional) and either:

- `person_*` objects with `keypoints` as a dict of COCO-style names → `{x, y, conf|confidence}`, or  
- a structure parseable by [`_parse_keypoints`](app/vm_client.py) (see that function for supported shapes).

Keypoints are converted to a list of 17 `{x, y, confidence}` entries in COCO order for local angle math.
