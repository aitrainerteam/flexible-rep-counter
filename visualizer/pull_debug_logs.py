#!/usr/bin/env python3
"""Pull the most recent rep-counter NDJSON + access log from the YOLO VM.

The NDJSON is filtered down to the latest session for a single ``user_uid``
(default ``debug-opencv-simulator`` — the UID emitted by
``visualizer/vm_rep_simulator.py``) and written to fixed filenames next to
this script so each run overwrites the previous output.

Platform notes
--------------
* ``gcloud`` is only installed on Windows, so this script always invokes it
  through ``powershell.exe``. That works both when launched from WSL (which
  spawns the Windows shell through interop) and from native Windows.
* When launched from WSL, the output path is converted to a Windows path via
  ``wslpath -w`` so ``gcloud compute scp`` can write to it. On native Windows
  the path is used as-is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

VM_INSTANCE = "yolo-deployed-brazil"
VM_ZONE = "southamerica-east1-a"
VM_PROJECT = "ai-trainer-a44a3"
VM_USER = "aa"
REMOTE_NDJSON = "/home/aa/yolo-deploy/logs/rep-counter.ndjson"
DEFAULT_USER_UID = "debug-opencv-simulator"

_OUT_DIR = Path(__file__).resolve().parent
OUT_NDJSON_RAW = _OUT_DIR / "latest-rep-counter-full.ndjson"
OUT_NDJSON_LATEST = _OUT_DIR / "latest-rep-counter-session.ndjson"
OUT_ACCESS_LOG = _OUT_DIR / "latest-access.log"
OUT_METADATA = _OUT_DIR / "latest-metadata.json"


def _run_pwsh(cmd: str, timeout: float = 120.0) -> tuple[int, str, str]:
    """Run a PowerShell command and return ``(returncode, stdout, stderr)``."""
    try:
        p = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", "powershell.exe not found (running outside WSL?)"


def _to_windows_path(p: Path) -> str:
    """Return a Windows-style path for ``p``.

    On native Windows the path is returned as-is. On WSL ``wslpath -w`` is
    used; if that fails (unusual layouts, missing tool) the raw string path
    is returned as a best-effort fallback.
    """
    if sys.platform == "win32":
        return str(p)
    try:
        return subprocess.check_output(
            ["wslpath", "-w", str(p)],
            text=True,
            timeout=5.0,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return str(p)


def _scp_ndjson() -> int:
    """Copy the remote NDJSON into ``OUT_NDJSON_RAW``. Returns 0 on success."""
    local_raw_win = _to_windows_path(OUT_NDJSON_RAW)
    scp_cmd = (
        f"gcloud compute scp "
        f"--project {VM_PROJECT} --zone {VM_ZONE} "
        f'{VM_USER}@{VM_INSTANCE}:{REMOTE_NDJSON} "{local_raw_win}"'
    )
    rc, _out, err = _run_pwsh(scp_cmd)
    if rc != 0:
        err_lower = err.lower()
        if (
            "TERMINATED" in err
            or "could not SSH" in err
            or "connection timed out" in err_lower
        ):
            print(
                "ERROR: VM may be TERMINATED or unreachable. To start the VM:\n"
                '  powershell.exe -NoProfile -Command "gcloud compute instances start '
                'yolo-deployed-brazil --zone=southamerica-east1-a --project=ai-trainer-a44a3"',
                file=sys.stderr,
            )
        else:
            print(
                f"ERROR: gcloud scp failed (rc={rc}): {err[:500]}",
                file=sys.stderr,
            )
        return rc if rc != 0 else 1
    return 0


def _filter_latest_session(
    path_raw: Path, user_uid: str
) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Return ``(session_id, events)`` for the most recent session of ``user_uid``.

    Events are grouped by ``session_id`` and the group whose maximum ``ts_ms``
    is highest wins. Events inside the winning group are returned in
    chronological order.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_ts: dict[str, float] = {}
    with open(path_raw, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("user_uid") != user_uid:
                continue
            sid = ev.get("session_id")
            if not sid:
                continue
            ts = float(ev.get("ts_ms") or 0.0)
            groups[sid].append(ev)
            if ts > max_ts.get(sid, -1.0):
                max_ts[sid] = ts
    if not max_ts:
        return None, []
    latest_sid = max(max_ts, key=max_ts.get)
    events = sorted(groups[latest_sid], key=lambda e: float(e.get("ts_ms") or 0))
    return latest_sid, events


def _write_latest_ndjson(events: list[dict[str, Any]]) -> None:
    with open(OUT_NDJSON_LATEST, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _pull_access_log(events: list[dict[str, Any]], minutes: int) -> None:
    """Pull ``yolo-server`` journal via SSH, anchored on the NDJSON start ts."""
    if events:
        first_ts_ms = float(events[0].get("ts_ms", 0))
        since = datetime.fromtimestamp((first_ts_ms / 1000.0) - 5.0, tz=timezone.utc)
        since_str = since.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        since_str = f"{minutes} minutes ago"

    remote_cmd = (
        f"journalctl -u yolo-server --since '{since_str}' --no-pager -o short-iso"
    )
    ssh_cmd = (
        f"gcloud compute ssh "
        f"--project {VM_PROJECT} --zone {VM_ZONE} "
        f'{VM_USER}@{VM_INSTANCE} --command "{remote_cmd}"'
    )
    rc, out, err = _run_pwsh(ssh_cmd)
    if rc == 0:
        OUT_ACCESS_LOG.write_text(out, encoding="utf-8")
    else:
        print(
            f"WARNING: journalctl SSH failed (rc={rc}): {err[:300]}",
            file=sys.stderr,
        )


def _write_metadata(
    user_uid: str,
    latest_sid: Optional[str],
    events: list[dict[str, Any]],
) -> None:
    meta = {
        "user_uid": user_uid,
        "latest_session_id": latest_sid,
        "event_count": len(events),
        "first_ts_ms": events[0].get("ts_ms") if events else None,
        "last_ts_ms": events[-1].get("ts_ms") if events else None,
        "vm_instance": VM_INSTANCE,
        "vm_zone": VM_ZONE,
        "vm_project": VM_PROJECT,
        "event_type_counts": dict(
            Counter((e.get("event") or "unknown") for e in events)
        ),
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT_METADATA.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _print_summary(
    user_uid: str,
    latest_sid: Optional[str],
    events: list[dict[str, Any]],
) -> None:
    if events:
        print(
            f"Pulled logs from VM {VM_INSTANCE} ({VM_ZONE}):\n"
            f"  Latest session: {OUT_NDJSON_LATEST} "
            f"({len(events)} events, session_id={latest_sid})\n"
            f"  Access log: {OUT_ACCESS_LOG}\n"
            f"  Metadata: {OUT_METADATA}"
        )
    else:
        print(
            f"WARNING: no NDJSON events for user_uid={user_uid}.\n"
            "Check VM systemd env:\n"
            "  powershell.exe gcloud compute ssh aa@yolo-deployed-brazil --zone ... "
            "--command \\\n"
            "    \"systemctl show yolo-server.service --property=Environment | "
            "tr ' ' '\\n' | grep REP_COUNTER\"\n"
            "Expected: REP_COUNTER_INSTRUMENTATION_ENABLED=true"
        )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull the latest rep-counter NDJSON and yolo-server access log "
            "from the YOLO VM, filter to the most recent session for a given "
            "user_uid, and save fixed files next to this script."
        ),
    )
    parser.add_argument(
        "--user-uid",
        default=DEFAULT_USER_UID,
        help=f"user_uid to filter NDJSON events (default: {DEFAULT_USER_UID}).",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=15,
        help=(
            "Fallback access-log window when the NDJSON has no events for "
            "the given user_uid (default: 15)."
        ),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the raw NDJSON copy (default: delete after filtering).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    rc = _scp_ndjson()
    if rc != 0:
        return 1

    latest_sid, events = _filter_latest_session(OUT_NDJSON_RAW, args.user_uid)
    if events:
        _write_latest_ndjson(events)
    else:
        OUT_NDJSON_LATEST.write_text("", encoding="utf-8")

    _pull_access_log(events, args.minutes)
    _write_metadata(args.user_uid, latest_sid, events)

    if not args.keep_intermediate:
        OUT_NDJSON_RAW.unlink(missing_ok=True)

    _print_summary(args.user_uid, latest_sid, events)
    return 0


if __name__ == "__main__":
    sys.exit(main())
