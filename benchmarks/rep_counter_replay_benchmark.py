#!/usr/bin/env python3
from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flexible_rep_counter.session import RepCounterSession

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def frame(theta_deg: float) -> list[dict]:
    sh = (0.7, 0.4)
    el = (0.7, 0.55)
    length = 0.18
    th = math.radians(theta_deg)
    wr = (el[0] + length * math.cos(th), el[1] - length * math.sin(th))

    points = {name: {"x": 0.0, "y": 0.0, "confidence": 0.0} for name in COCO_KEYPOINT_NAMES}
    defaults = {
        "left_shoulder": (0.3, 0.4),
        "left_elbow": (0.3, 0.55),
        "left_wrist": (0.3, 0.73),
        "right_shoulder": sh,
        "right_elbow": el,
        "right_wrist": wr,
        "left_hip": (0.4, 0.72),
        "right_hip": (0.6, 0.72),
        "left_knee": (0.42, 0.9),
        "right_knee": (0.58, 0.9),
        "left_ankle": (0.42, 1.05),
        "right_ankle": (0.58, 1.05),
    }
    for key, (x, y) in defaults.items():
        points[key] = {"x": x, "y": y, "confidence": 0.99}
    return [points[name] for name in COCO_KEYPOINT_NAMES]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(round((len(values) - 1) * q))))
    return values[idx]


def main() -> int:
    angles = []
    for _ in range(24):
        angles += [35, 45, 60, 85, 115, 145, 160, 145, 115, 85, 60, 45]

    session = RepCounterSession(auto_started=False, use_pose_filter=False)
    session.set_started(wall_time=0.0)

    latencies_ms: list[float] = []
    for i, angle in enumerate(angles):
        t = i * (1.0 / 30.0)
        t0 = time.perf_counter()
        session.step_landmarks(frame(angle), timestamp_ms=t * 1000.0, wall_time_s=t)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms.sort()
    print(f"frames={len(latencies_ms)}")
    print(f"p50_ms={percentile(latencies_ms, 0.50):.4f}")
    print(f"p95_ms={percentile(latencies_ms, 0.95):.4f}")
    print(f"p99_ms={percentile(latencies_ms, 0.99):.4f}")
    print(f"mean_ms={statistics.fmean(latencies_ms):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
