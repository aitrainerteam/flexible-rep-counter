# Flexible Rep Counter

AI-powered fitness rep counter using computer vision pose estimation. During startup it analyzes joint-angle motion to pick the limb that best matches your movement, then counts repetitions with peak/valley detection—no exercise name or preset required.

## Architecture overview

- **Webcam capture**: OpenCV frames, optional resize/JPEG tuning before upload.
- **Pose inference**: YOLO pose on a remote VM; client in `app/vm_client.py` (health check, `/predict`-style endpoints, benchmarking hooks).
- **Angle selection**: Variance-based scoring over a calibration window (`app/variance_angle_selector.py`), with smoothed angle series (`smooth_angle_series` in `app/math_engine.py`), minimum wall time and frame counts, retries, and per-joint variance/range gates so the chosen angle stays stable.
- **Tracking mode**: Single- or dual-angle tracking from angle activity; overlay can show matched reps and per-angle counts.
- **Rep counting**: `PeakDetector` in `app/math_engine.py` with hysteresis, margins, rolling range gate, calibration reps, **retroactive replay** of calibration angles through the detector, and **certainty-based locking** (with optional extra reps if certainty is low).
- **UI**: OpenCV overlay for reps, state, calibration feedback, and debug info; optional debug console when `LOG_LEVEL=DEBUG`.

## Python local app (webcam + VM)

Runs on your machine: frames go to the VM for keypoints; angles and reps are computed locally with an on-screen overlay.

### Setup

```bash
cd /path/to/flexible-rep-counter
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Optional flags: `--no-health-check`, `--benchmark-log FILE`, `--resize-width W`, `--jpeg-quality Q`, `--no-validate-response` (see `main.py`).

### Session flow

- **Angle selection**: Collects motion for at least `ANGLE_SELECTION_MIN_SEC` and `ANGLE_SELECTION_MIN_FRAMES` (up to `ANGLE_SELECTION_MAX_BUFFER_FRAMES`), uses smoothed angles and variance/range gates; may retry on `ANGLE_SELECTION_RETRY_INTERVAL_SEC` if nothing passes.
- **Calibration**: First reps establish peak/valley baselines; calibration data can be **replayed** through the peak detector so counts stay consistent with locked logic. Locking respects `REP_CALIBRATION_CERTAINTY` and, if needed, `REP_CALIBRATION_FORCE_EXTRA_REPS` after the minimum calibration reps.
- **Tracking**: Counts peaks/valleys with margins, range gate, and `REP_MIN_INTERVAL_MS`; quit with `q` or Escape in the video window.

## Configuration

Create a `.env` in the project root. The app loads it automatically (`app/config.py`).

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
| `ANGLE_SELECTION_MIN_SEC` | Minimum wall time for selection window | `5.0` |
| `ANGLE_SELECTION_MIN_FRAMES` | Minimum frames in window | `50` |
| `ANGLE_SELECTION_MAX_BUFFER_FRAMES` | Cap on buffered frames | `400` |
| `ANGLE_SELECTION_RETRY_INTERVAL_SEC` | Retry if gates not met | `2.0` |
| `ANGLE_SELECTION_MIN_VARIANCE` | Min variance to accept a joint (global; per-joint overrides like `_LEFT_ELBOW` exist) | `6.0` |
| `ANGLE_SELECTION_MIN_RANGE_DEG` | Min observed range (°) for a candidate | `16.0` |
| `ANGLE_SELECTION_SECOND_BEST_RATIO` | Second-best must be worse by this ratio | `1.15` |

### Debugging

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `DEBUG` |
| `DEBUG_CONSOLE_ENABLED` | Second debug window | on when `LOG_LEVEL=DEBUG` |

### VM API

The client POSTs JSON `{"image": "<base64-jpeg>"}`. It tries the base URL and paths `/predict`, `/infer`, `/pose`. Responses should include COCO-17-style keypoints (`x`, `y`, confidence or equivalent).

---

## Recent commits (last 5)

| Commit | Summary |
|--------|---------|
| `df37d11` | Angle selection: new variance/range thresholds in config; loop uses consistent variance gates; `variance_angle_selector` refactored with angle-specific thresholds and stronger variance logic. |
| `a92f54a` | Angle selection timing and frame requirements; smoothed angle series in `math_engine`; loop uses retries and smoother series; variance selector uses smoothed data and tuned thresholds. |
| `3b98c32` | Angle series and activity stats; `_select_tracking_mode` for single vs dual tracking; overlay shows matched reps and per-angle rep counts. |
| `4aa610d` | `REP_CALIBRATION_CERTAINTY` and `REP_CALIBRATION_FORCE_EXTRA_REPS`; `PeakDetector` certainty and extra-rep locking; calibration status reflects certainty. |
| `65e15b5` | Retroactive rep counting via `replay_angle_series_on_peak_detector`; calibration replay through peak detectors for consistent state. |
