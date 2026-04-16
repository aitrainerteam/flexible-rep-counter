"""Buffered NDJSON sink for rep-counter instrumentation (observation-only)."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RepInstrumentationSettings:
    """Server-controlled instrumentation (typically from environment variables)."""

    enabled: bool
    output_path: str
    sample_every_n_frames: int = 1
    flush_every_n_records: int = 20
    user_allowlist: Optional[frozenset[str]] = None

    def should_emit_for_user(self, user_uid: Optional[str]) -> bool:
        if not self.enabled or not user_uid:
            return False
        if self.user_allowlist is not None and user_uid not in self.user_allowlist:
            return False
        return True

    def should_sample_frame(self, frame_idx: int) -> bool:
        n = max(1, int(self.sample_every_n_frames))
        return (int(frame_idx) % n) == 0


class RepInstrumentationSink:
    """Thread-safe append-only NDJSON writer with simple buffering."""

    def __init__(self, settings: RepInstrumentationSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._buf: list[str] = []
        self._since_flush = 0
        self._disabled_due_to_error = False

    @property
    def settings(self) -> RepInstrumentationSettings:
        return self._settings

    def emit(self, record: dict[str, Any]) -> None:
        if self._disabled_due_to_error or not self._settings.enabled:
            return
        try:
            line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        except (TypeError, ValueError):
            return
        flush_every = max(1, int(self._settings.flush_every_n_records))
        with self._lock:
            self._buf.append(line)
            self._since_flush += 1
            if self._since_flush >= flush_every:
                self._flush_unlocked()

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def close(self) -> None:
        self.flush()

    def _flush_unlocked(self) -> None:
        if not self._buf:
            self._since_flush = 0
            return
        try:
            parent = os.path.dirname(self._settings.output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._settings.output_path, "a", encoding="utf-8") as f:
                f.writelines(self._buf)
        except OSError:
            self._disabled_due_to_error = True
            self._buf.clear()
            self._since_flush = 0
            return
        self._buf.clear()
        self._since_flush = 0


def build_trace_context(
    *,
    user_uid: Optional[str],
    session_id: Optional[str],
    request_ts_ms: float,
    frame_idx: int,
) -> dict[str, Any]:
    return {
        "user_uid": user_uid or "",
        "session_id": session_id or "",
        "request_ts_ms": float(request_ts_ms),
        "frame_idx": int(frame_idx),
    }


def merge_trace(record: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out.update(trace)
    out.setdefault("ts_ms", time.time() * 1000.0)
    return out
