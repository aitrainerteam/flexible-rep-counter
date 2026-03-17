#!/usr/bin/env python3
"""Entrypoint for local rep counter: webcam -> VM pose -> rep count overlay."""
import sys

from app.config import YOLO_VM_TARGET_URL
from app.loop import run_webcam_loop


def main():
    print("Rep counter starting (VM:", YOLO_VM_TARGET_URL, ")")
    try:
        run_webcam_loop()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
