"""Buffered NDJSON sink for rep-counter instrumentation (observation-only)."""
from __future__ import annotations

import atexit
import json
import os
import queue
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
    max_queue_records: int = 10000
    flush_interval_sec: float = 0.25

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
    """Thread-safe append-only NDJSON writer that keeps disk IO off the hot path."""

    _writers: dict[tuple[str, int, int, float], "_AsyncNdjsonWriter"] = {}
    _writers_lock = threading.Lock()

    def __init__(self, settings: RepInstrumentationSettings) -> None:
        self._settings = settings
        self._writer = self._get_writer(settings)

    @property
    def settings(self) -> RepInstrumentationSettings:
        return self._settings

    def emit(self, record: dict[str, Any]) -> None:
        if not self._settings.enabled:
            return
        try:
            line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        except (TypeError, ValueError):
            return
        self._writer.enqueue(line)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self.flush()

    @classmethod
    def _get_writer(cls, settings: RepInstrumentationSettings) -> "_AsyncNdjsonWriter":
        key = (
            os.path.abspath(settings.output_path),
            max(1, int(settings.flush_every_n_records)),
            max(1, int(settings.max_queue_records)),
            max(0.01, float(settings.flush_interval_sec)),
        )
        with cls._writers_lock:
            writer = cls._writers.get(key)
            if writer is None:
                writer = _AsyncNdjsonWriter(
                    output_path=key[0],
                    flush_every_n_records=key[1],
                    max_queue_records=key[2],
                    flush_interval_sec=key[3],
                )
                cls._writers[key] = writer
            return writer


class _AsyncNdjsonWriter:
    """Single background writer for one instrumentation output file."""

    def __init__(
        self,
        *,
        output_path: str,
        flush_every_n_records: int,
        max_queue_records: int,
        flush_interval_sec: float,
    ) -> None:
        self._output_path = output_path
        self._flush_every = max(1, int(flush_every_n_records))
        self._flush_interval = max(0.01, float(flush_interval_sec))
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=max(1, int(max_queue_records)))
        self._buf: list[str] = []
        self._write_lock = threading.Lock()
        self._disabled_due_to_error = False
        self._closed = False
        self._dropped_records = 0
        self._thread = threading.Thread(
            target=self._run,
            name="rep-instrumentation-writer",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def enqueue(self, line: str) -> None:
        if self._closed or self._disabled_due_to_error:
            return
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self._dropped_records += 1
            return

    def flush(self) -> None:
        if self._closed:
            return
        self._queue.join()
        with self._write_lock:
            self._flush_locked()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self.flush()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                return
        self._closed = True
        self._thread.join(timeout=2.0)

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    def _run(self) -> None:
        while True:
            try:
                line = self._queue.get(timeout=self._flush_interval)
            except queue.Empty:
                with self._write_lock:
                    self._flush_locked()
                continue

            try:
                if line is None:
                    with self._write_lock:
                        self._flush_locked()
                    return
                self._buf.append(line)
                if len(self._buf) >= self._flush_every:
                    with self._write_lock:
                        self._flush_locked()
            finally:
                self._queue.task_done()

    def _flush_locked(self) -> None:
        if not self._buf or self._disabled_due_to_error:
            self._buf.clear()
            return
        try:
            parent = os.path.dirname(self._output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._output_path, "a", encoding="utf-8") as f:
                f.writelines(self._buf)
        except OSError:
            self._disabled_due_to_error = True
        self._buf.clear()


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
