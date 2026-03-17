# Flexible Rep Counter

AI-powered fitness rep counter using computer vision pose estimation. Automatically detects exercise type by analyzing joint angle variance during initial calibration, then uses real-time peak/valley detection to count repetitions without requiring prior exercise knowledge.

## Architecture Overview

The system consists of several key components:

- **Webcam Capture**: OpenCV-based frame capture with configurable resolution
- **Pose Estimation**: YOLO pose model running on remote VM for real-time keypoint detection
- **Angle Selection**: Automatic exercise detection via joint angle variance analysis during calibration
- **Rep Counting**: State machine-based peak/valley detection with configurable hysteresis and smoothing
- **Real-time UI**: OpenCV overlay displaying rep count, selected angle, tracking state, and debug information

## Python Local App (Webcam + VM)

Runs locally on your machine: webcam frames are sent to a remote YOLO pose inference VM; keypoints are processed to automatically select the most active joint angle and count reps with an on-screen overlay.

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

- **Calibration**: First ~60 frames are used to choose the most consistently moving joint angle.
- **Tracking**: Selected angle is tracked; reps increment on peak/valley pairs.
- **Quit**: Press `q` or Escape in the video window.

## Configuration

Create a `.env` file in the project root directory to configure the application:

### VM Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `YOLO_VM_DIRECT_URL` | Primary VM endpoint URL | Required (no default) |
| `YOLO_VM_TARGET_URL` | Alternative VM endpoint URL | `http://127.0.0.1:8000` |
| `VM_TIMEOUT_SEC` | Timeout in seconds for VM requests | `5.0` |

### Rep Counting Tuning
| Variable | Description | Default |
|----------|-------------|---------|
| `REP_HYSTERESIS` | Minimum angle change (degrees) to register direction change | `5` |
| `REP_MIN_PEAK_DISTANCE` | Minimum frames between peak/valley events | `5` |
| `REP_SMOOTHING_FACTOR` | Exponential moving average alpha for angle smoothing (0–1) | `0.45` |
| `REP_PEAK_MARGIN` | Only count a peak (e.g. arm extended) when within this many degrees of the average observed max | `15` |
| `REP_VALLEY_MARGIN` | Only count a valley (e.g. arm flexed) when within this many degrees of the average observed min | `15` |

### Debugging
| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `DEBUG` |
| `DEBUG_CONSOLE_ENABLED` | Enable debug console window | `true` when LOG_LEVEL=DEBUG |

Create a `.env` in the project root to set these (e.g. `YOLO_VM_TARGET_URL=http://your-vm:8000`). The app loads `.env` automatically.

### VM API

The client POSTs JSON `{"image": "<base64-jpeg>"}` to the VM. It tries, in order: base URL, `/predict`, `/infer`, `/pose`. The response is expected to include keypoints in COCO 17 format (list of 17 items with `x`, `y`, `confidence` or equivalent).
