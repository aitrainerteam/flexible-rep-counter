#!/usr/bin/env python3
"""Entrypoint for local rep counter: webcam -> VM pose -> rep count overlay."""
import argparse
import sys

from app.config import YOLO_VM_TARGET_URL
from app.loop import run_webcam_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam rep counter (VM pose API)")
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip GET /health (status ok + model_loaded) before opening the camera",
    )
    parser.add_argument(
        "--benchmark-log",
        metavar="FILE",
        help="Append tab-separated timestamp, server inference_ms, roundtrip_ms, upload_ms per VM response",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        metavar="W",
        help="Resize frame to this width before JPEG upload (0=no resize). Default: PREDICT_RESIZE_WIDTH in .env",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        metavar="Q",
        help="JPEG quality 1-100 for uploads. Default: PREDICT_JPEG_QUALITY in .env",
    )
    parser.add_argument(
        "--no-validate-response",
        action="store_true",
        help="Skip JSON shape checks on /predict (inference_ms, keypoint names)",
    )
    args = parser.parse_args()

    print("Rep counter starting (VM:", YOLO_VM_TARGET_URL, ")")
    try:
        run_webcam_loop(
            skip_health_check=args.no_health_check,
            benchmark_log_path=args.benchmark_log,
            resize_width=args.resize_width,
            jpeg_quality=args.jpeg_quality,
            validate_response=False if args.no_validate_response else None,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
