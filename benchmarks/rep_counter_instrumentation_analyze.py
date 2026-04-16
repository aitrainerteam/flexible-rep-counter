#!/usr/bin/env python3
"""Summarize rep-counter NDJSON instrumentation (stdin or file path)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def _first_frame(events: list[dict[str, Any]], event: str, key: str = "frame_idx") -> Optional[int]:
    for e in events:
        if e.get("event") == event and key in e:
            try:
                return int(e[key])
            except (TypeError, ValueError):
                continue
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default="-",
        help="NDJSON file (default: stdin)",
    )
    p.add_argument("--json-summary", action="store_true", help="Print JSON summary to stdout")
    args = p.parse_args()

    lines: list[str]
    path = args.path
    if path == "-":
        if not sys.stdin.isatty():
            lines = sys.stdin.read().splitlines()
        else:
            # TTY + "-" : avoid blocking on stdin.read(); use server default path.
            default = os.environ.get("REP_COUNTER_INSTRUMENTATION_PATH", "").strip()
            if not default:
                default = str(Path.home() / "yolo-deploy" / "logs" / "rep-counter.ndjson")
            path = default
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except OSError as e:
                print(f"ERROR: could not read {path}: {e}", file=sys.stderr)
                print(
                    "Pass a file path, pipe NDJSON on stdin, or set REP_COUNTER_INSTRUMENTATION_PATH.",
                    file=sys.stderr,
                )
                return 1
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    events: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    reasons = Counter()
    for e in events:
        if e.get("event") == "rep_block":
            reasons[str(e.get("reason", ""))] += 1
        if e.get("event") == "pose_dropped":
            reasons[f"pose_dropped:{e.get('reason', '')}"] += 1

    summary: dict[str, Any] = {
        "line_count": len(lines),
        "parsed_event_count": len(events),
        "first_frame_snapshot": _first_frame(events, "frame_snapshot"),
        "first_state_transition": _first_frame(events, "state_transition"),
        "first_range_gate_open_frame": None,
        "first_rep_increment": _first_frame(events, "rep_increment"),
        "rep_block_reason_counts": dict(reasons),
    }

    for e in events:
        if e.get("event") == "frame_snapshot" and e.get("range_gate_open") is True:
            try:
                summary["first_range_gate_open_frame"] = int(e.get("frame_idx", -1))
            except (TypeError, ValueError):
                summary["first_range_gate_open_frame"] = None
            break

    if args.json_summary:
        print(json.dumps(summary, indent=2))
        return 0

    print("Rep counter instrumentation summary")
    print("-------------------------------------")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
