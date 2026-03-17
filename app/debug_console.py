"""Debug console: stream log output at DEBUG level to terminal (stderr) so it can be copied."""
from __future__ import annotations

import logging
import sys

from app.config import DEBUG_CONSOLE_ENABLED, LOG_LEVEL


class _ConsoleHandler(logging.Handler):
    """Write log records to stderr as terminal text so the user can copy/paste."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            line = f"[{record.levelname}] {msg}\n"
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    """Configure root logger with DEBUG level when debug console enabled, and add console handler."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(logging.StreamHandler(sys.stderr))
    if DEBUG_CONSOLE_ENABLED:
        handler = _ConsoleHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given name (e.g. __name__)."""
    return logging.getLogger(name)


def ensure_console_window() -> None:
    """No-op: debug console is terminal text only (no second window)."""
    pass


def update_console_window() -> None:
    """No-op: debug output is streamed to stderr in real time."""
    pass
