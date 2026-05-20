#!/usr/bin/env python3
"""
Stream-filter rep-counter NDJSON by frame_idx, session_id, and event type.

Designed for large logs: reads line-by-line (no full-file buffer).

``range_gate`` in --events is treated as an alias for ``frame_snapshot`` (there is no
separate ``range_gate`` event; gate state lives on ``frame_snapshot`` rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional


def _parse_events(raw: str) -> Optional[frozenset[str]]:
    if not raw.strip():
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        return None
    out: set[str] = set()
    for p in parts:
        if p == "range_gate":
            out.add("frame_snapshot")
        else:
            out.add(p)
    return frozenset(out)


def _frame_idx(obj: dict[str, Any]) -> Optional[int]:
    v = obj.get("frame_idx")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "log_path",
        help="Path to append-only NDJSON log (e.g. ~/yolo-deploy/logs/rep-counter.ndjson)",
    )
    p.add_argument(
        "--start",
        type=int,
        default=1,
        help="Minimum frame_idx to emit (inclusive). Default: 1",
    )
    p.add_argument(
        "--end",
        type=int,
        default=60,
        help="Maximum frame_idx to emit (inclusive). Default: 60",
    )
    p.add_argument(
        "--session-id",
        default="",
        help="If set, only lines whose session_id equals this string",
    )
    p.add_argument(
        "--user-uid",
        default="",
        help="If set, only lines whose user_uid equals this string",
    )
    p.add_argument(
        "--events",
        default="",
        help="Comma-separated event types to keep (empty = all). "
        "Known: frame_snapshot, state_transition, rep_block, rep_increment, pose_dropped. "
        "range_gate is accepted as an alias for frame_snapshot.",
    )
    args = p.parse_args()

    event_filter = _parse_events(args.events)
    sid = args.session_id.strip()
    uid = args.user_uid.strip()
    lo = int(args.start)
    hi = int(args.end)
    if hi < lo:
        print("ERROR: --end must be >= --start", file=sys.stderr)
        return 1

    try:
        f = open(args.log_path, "r", encoding="utf-8")
    except OSError as e:
        print(f"ERROR: could not open {args.log_path!r}: {e}", file=sys.stderr)
        return 1

    with f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if sid and str(obj.get("session_id", "")) != sid:
                continue
            if uid and str(obj.get("user_uid", "")) != uid:
                continue
            ev = str(obj.get("event", ""))
            if event_filter is not None and ev not in event_filter:
                continue
            fi = _frame_idx(obj)
            if fi is not None and (fi < lo or fi > hi):
                continue
            sys.stdout.write(line if line.endswith("\n") else line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
