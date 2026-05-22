from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from flexible_rep_counter.session import _calibration_edge_flags


def test_calibration_started_on_first_uncalibrated_tracking_frame() -> None:
    rs: dict = {}
    started, locked = _calibration_edge_flags(
        rs, calibration_complete=False, phase="tracking"
    )
    assert started is True
    assert locked is False
    started2, locked2 = _calibration_edge_flags(
        rs, calibration_complete=False, phase="tracking"
    )
    assert started2 is False
    assert locked2 is False


def test_calibration_locked_on_complete_transition() -> None:
    rs: dict = {"_prev_calibration_complete": False}
    started, locked = _calibration_edge_flags(
        rs, calibration_complete=True, phase="tracking"
    )
    assert started is False
    assert locked is True


def test_calibration_started_after_switch_from_locked_joint() -> None:
    rs: dict = {"_prev_calibration_complete": True}
    started, locked = _calibration_edge_flags(
        rs, calibration_complete=False, phase="tracking"
    )
    assert started is True
    assert locked is False


def test_no_flags_outside_tracking() -> None:
    rs: dict = {}
    started, locked = _calibration_edge_flags(
        rs, calibration_complete=False, phase="selecting"
    )
    assert started is False
    assert locked is False
    assert rs["_prev_calibration_complete"] is None
