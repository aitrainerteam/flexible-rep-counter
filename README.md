# Flexible Rep Counter

AI-powered fitness rep counter using computer vision pose estimation. It analyzes joint-angle motion to pick the limb that best matches your movement, then counts repetitions with peak/valley detection—no exercise name or preset required.

## Architecture overview

- **Webcam capture**: OpenCV frames; optional resize/JPEG tuning before upload.
- **Pose inference**: YOLO pose on a remote VM; client in [`app/vm_client.py`](app/vm_client.py) (`GET /health`, `POST /predict` multipart JPEG, benchmarking hooks).
- **Concurrency**: Main thread runs capture, overlay, and angle math; one background **worker thread** sends the latest frame to the VM (queue size 1) so slow network does not block the UI loop.
- **Angle selection**: [`app/variance_angle_selector.py`](app/variance_angle_selector.py) scores per-joint angle variance over a buffer; the main loop also tracks **rep dominance** across joints (which angle’s peak detector counts the most reps) and can lock the leader after a streak. If dominance stays ambiguous, after `angle_selection.variance_fallback_sec` (in `rep_counter.toml`) the session may lock using pure **variance** selection when the retry window allows.
- **Tracking**: **One joint only** (one `COMMON_ANGLES` key, e.g. `LEFT_ELBOW` or `RIGHT_KNEE`). The opposite limb is not tracked and does not contribute to the count.
- **Rep counting**: `PeakDetector` in [`src/flexible_rep_counter/core/math_engine.py`](src/flexible_rep_counter/core/math_engine.py) (also re-exported from [`app/math_engine.py`](app/math_engine.py))—hysteresis, peak/valley margins after calibration, rolling range gate, **retroactive replay** of the observation buffer through the detector, and certainty-based locking.
- **Importable package**: Core logic lives under [`src/flexible_rep_counter/`](src/flexible_rep_counter/). OpenCV UI is in [`visualizer/opencv_runtime.py`](visualizer/opencv_runtime.py) (repo-only; run via [`main.py`](main.py)).
- **Docs**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full pipeline, call order, and math/selection details.
- **UI**: OpenCV overlay; log verbosity follows `[app].log_level` in `rep_counter.toml` (stderr when `DEBUG`).

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

Optional flags: `--no-health-check`, `--benchmark-log FILE`, `--resize-width W`, `--jpeg-quality Q`, `--no-validate-response` (defaults follow `[predict]` in `rep_counter.toml` when omitted).

### Session flow

1. **Angle selection**: Buffer frames (cap `angle_selection.max_buffer_frames`) until `angle_selection.min_sec` and `angle_selection.min_frames`. Per-frame, lightweight peak detectors on all candidate angles feed **rep dominance** stats; variance/ROM gates pick a stable joint. Lock when dominance + variance agree for `angle_selection.dominance_streak_frames`, or after `angle_selection.variance_fallback_sec` via pure variance selection if needed. Retries use `angle_selection.retry_interval_sec`.
2. **Calibration**: First reps establish peak/valley baselines; the buffer is **replayed** through the same detectors so displayed reps include motion from the selection window. Locking uses `rep.calibration_certainty` and optionally `rep.calibration_force_extra_reps`.
3. **Tracking**: Counts with margins, range gate, and `rep.min_interval_ms`. Quit with `q` or Escape in the video window.

## Configuration

**Source of truth:** [`rep_counter.toml`](rep_counter.toml) in the repo root, or the current working directory, or any parent directory. Set `FLEXIBLE_REP_COUNTER_CONFIG` to an absolute path to use a different file.

Sections:

| Section | Role |
|---------|------|
| `[app]` | `log_level` (`DEBUG`, `INFO`, …) |
| `[vm]` | `direct_url` (required for the visualizer), `timeout_sec`, `health_timeout_sec` |
| `[predict]` | `resize_width`, `jpeg_quality`, `validate_response` |
| `[rep]` | Peak detector tuning: hysteresis, margins, calibration, `min_interval_ms`, etc. |
| `[angle_selection]` | Selection window, dominance, `variance_fallback_sec`, global variance/range thresholds |
| `[angle_selection.joints.<NAME>]` | Per-joint overrides (e.g. `LEFT_ELBOW`) for `min_variance`, `min_range_deg`, `second_best_ratio` |

The root `.env` is loaded for compatibility (e.g. `FLEXIBLE_REP_COUNTER_CONFIG`); tuning keys live in TOML, not duplicate env vars.

### VM API (client contract)

The client sends **`POST {base_url}/predict`** with multipart form field **`file`** = JPEG bytes (`image/jpeg`). The JSON response should include top-level `inference_ms` (optional) and either:

- `person_*` objects with `keypoints` as a dict of COCO-style names → `{x, y, conf|confidence}`, or  
- a structure parseable by [`_parse_keypoints`](app/vm_client.py) (see that function for supported shapes).

Keypoints are converted to a list of 17 `{x, y, confidence}` entries in COCO order for local angle math.
