#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

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


def _point(x: float, y: float, conf: float = 0.99) -> dict[str, float]:
    return {"x": x, "y": y, "confidence": conf}


def frame_right_elbow(theta_deg: float) -> list[dict[str, float]]:
    sh = (0.70, 0.40)
    el = (0.70, 0.55)
    length = 0.18
    th = math.radians(theta_deg)
    wr = (el[0] + length * math.cos(th), el[1] - length * math.sin(th))

    points: dict[str, dict[str, float]] = {
        name: _point(0.0, 0.0, 0.0) for name in COCO_KEYPOINT_NAMES
    }
    defaults = {
        "left_shoulder": (0.30, 0.40),
        "left_elbow": (0.30, 0.55),
        "left_wrist": (0.30, 0.73),
        "right_shoulder": sh,
        "right_elbow": el,
        "right_wrist": wr,
        "left_hip": (0.40, 0.72),
        "right_hip": (0.60, 0.72),
        "left_knee": (0.42, 0.90),
        "right_knee": (0.58, 0.90),
        "left_ankle": (0.42, 1.05),
        "right_ankle": (0.58, 1.05),
    }
    for key, (x, y) in defaults.items():
        points[key] = _point(x, y, 0.99)
    return [points[name] for name in COCO_KEYPOINT_NAMES]


def build_stream() -> list[list[dict[str, float]]]:
    # 1) Warm idle/noise (user moving around a little).
    idle = [95, 98, 92, 99, 94, 97] * 6
    # 2) Exercise reps.
    rep_cycle = [25, 35, 50, 75, 105, 135, 165, 150, 120, 90, 60, 40]
    reps = rep_cycle * 30
    return [frame_right_elbow(v) for v in (idle + reps)]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def _load_repo_modules(repo_root: Path) -> tuple[Any, Any]:
    src = repo_root / "src"
    if not src.is_dir():
        raise RuntimeError(f"missing src directory: {src}")
    sys.path.insert(0, str(src))
    math_engine = importlib.import_module("flexible_rep_counter.core.math_engine")
    selector = importlib.import_module("flexible_rep_counter.core.variance_angle_selector")
    return math_engine, selector


def run(repo_root: Path, fps: float) -> int:
    _math_engine, _selector = _load_repo_modules(repo_root)
    session_mod = importlib.import_module("flexible_rep_counter.session")
    RepCounterSession = getattr(session_mod, "RepCounterSession")

    timings: dict[str, float] = {
        "variance_ms": 0.0,
        "selection_logic_ms": 0.0,
        "detector_update_ms": 0.0,
    }
    counters: dict[str, int] = {
        "variance_calls": 0,
        "selection_calls": 0,
        "detector_updates": 0,
    }

    orig_compute_variances = _selector.compute_angle_variances_from_buffer
    orig_determine_best = _selector.determine_best_angle
    orig_peak_update = _math_engine.PeakDetector.update
    orig_session_compute = getattr(session_mod, "compute_angle_variances_from_buffer", None)
    orig_session_determine = getattr(session_mod, "determine_best_angle", None)

    def timed_compute(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_compute_variances(*args, **kwargs)
        finally:
            timings["variance_ms"] += (time.perf_counter() - t0) * 1000.0
            counters["variance_calls"] += 1

    def timed_determine(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_determine_best(*args, **kwargs)
        finally:
            timings["selection_logic_ms"] += (time.perf_counter() - t0) * 1000.0
            counters["selection_calls"] += 1

    def timed_peak_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_peak_update(self, *args, **kwargs)
        finally:
            timings["detector_update_ms"] += (time.perf_counter() - t0) * 1000.0
            counters["detector_updates"] += 1

    _selector.compute_angle_variances_from_buffer = timed_compute
    _selector.determine_best_angle = timed_determine
    _math_engine.PeakDetector.update = timed_peak_update
    if orig_session_compute is not None:
        setattr(session_mod, "compute_angle_variances_from_buffer", timed_compute)
    if orig_session_determine is not None:
        setattr(session_mod, "determine_best_angle", timed_determine)

    stream = build_stream()
    session = RepCounterSession(auto_started=False, use_pose_filter=False)
    session.set_started(wall_time=0.0)
    latencies_ms: list[float] = []

    first_selected_frame: int | None = None
    first_calibrated_frame: int | None = None
    selected_joint: str | None = None
    last_out: Any = None

    for i, landmarks in enumerate(stream):
        t_s = i * (1.0 / fps)
        t0 = time.perf_counter()
        out = session.step_landmarks(
            landmarks,
            timestamp_ms=t_s * 1000.0,
            wall_time_s=t_s,
        )
        last_out = out
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if out.tracked_joint and first_selected_frame is None:
            first_selected_frame = i
            selected_joint = out.tracked_joint
        if out.calibration_complete and first_calibrated_frame is None:
            first_calibrated_frame = i

        perf_ms = out.selection_debug.get("perf_ms") if isinstance(out.selection_debug, dict) else None
        if isinstance(perf_ms, dict):
            timings["variance_ms"] += float(perf_ms.get("variance_ms") or 0.0)
            timings["selection_logic_ms"] += float(perf_ms.get("selection_logic_ms") or 0.0)
            timings["detector_update_ms"] += float(perf_ms.get("detector_update_ms") or 0.0)

    latencies_sorted = sorted(latencies_ms)
    mean_latency = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    p95 = percentile(latencies_sorted, 0.95)
    p99 = percentile(latencies_sorted, 0.99)

    def _frame_to_sec(f: int | None) -> float | None:
        if f is None:
            return None
        return f / fps

    print(f"repo={repo_root}")
    print(f"frames={len(stream)} fps={fps:.2f}")
    print(f"selected_joint={selected_joint}")
    print(f"first_selected_frame={first_selected_frame} first_selected_sec={_frame_to_sec(first_selected_frame)}")
    print(
        f"first_calibrated_frame={first_calibrated_frame} first_calibrated_sec={_frame_to_sec(first_calibrated_frame)}"
    )
    print(f"latency_mean_ms={mean_latency:.4f} latency_p95_ms={p95:.4f} latency_p99_ms={p99:.4f}")
    print(
        "timing_totals_ms="
        f"detector_update:{timings['detector_update_ms']:.4f}, "
        f"variance:{timings['variance_ms']:.4f}, "
        f"selection_logic:{timings['selection_logic_ms']:.4f}"
    )
    print(
        "timing_calls="
        f"detector_updates:{counters['detector_updates']}, "
        f"variance_calls:{counters['variance_calls']}, "
        f"selection_calls:{counters['selection_calls']}"
    )
    if last_out is not None:
        print(
            "final_state="
            f"phase:{last_out.phase}, tracked_joint:{last_out.tracked_joint}, reps:{last_out.reps}, "
            f"reps_raw:{last_out.reps_raw}, calibration_complete:{last_out.calibration_complete}, "
            f"calibration_certainty:{last_out.calibration_certainty:.3f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile rep-counter selection/calibration speed.")
    parser.add_argument("--repo", required=True, help="Repo root containing src/flexible_rep_counter")
    parser.add_argument("--fps", type=float, default=12.0, help="Synthetic stream FPS")
    args = parser.parse_args()
    return run(Path(args.repo).resolve(), args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
