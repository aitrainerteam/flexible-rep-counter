"""Tests for the skeleton-agnostic 3D angle session."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from flexible_rep_counter import (
    AngleRepCounterSession,
    AngleSample,
    calculate_angle_3d,
    load_3d_config,
)
from flexible_rep_counter.core.scalar_angle_selector import PRODUCT_ANGLE_KEYS
from flexible_rep_counter.core.settings_3d import load_3d_config as load_cfg


def _unknown() -> AngleSample:
    return AngleSample(value=None, evidence="unknown")


def _obs(value: float) -> AngleSample:
    return AngleSample(value=value, evidence="observed")


def _pred(value: float) -> AngleSample:
    return AngleSample(value=value, evidence="predicted")


def _frame(active: str, value: float, *, evidence: str = "observed") -> dict[str, AngleSample]:
    sample = AngleSample(value=value, evidence=evidence)  # type: ignore[arg-type]
    out = {k: _unknown() for k in PRODUCT_ANGLE_KEYS}
    out[active] = sample
    for k in PRODUCT_ANGLE_KEYS:
        if k != active:
            out[k] = _obs(90.0)
    return out


def _all_unknown() -> dict[str, AngleSample]:
    return {k: _unknown() for k in PRODUCT_ANGLE_KEYS}


def _sine(i: int, *, amp: float = 50.0, mid: float = 100.0, period: int = 20) -> float:
    return mid + amp * math.sin(2 * math.pi * i / period)


@pytest.fixture
def fast_cfg(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "rep_counter_3d.toml"
    text = src.read_text()
    text = text.replace("min_sec = 1.6", "min_sec = 0.3")
    text = text.replace("min_frames = 24", "min_frames = 10")
    text = text.replace("dominance_streak_frames = 18", "dominance_streak_frames = 4")
    text = text.replace("variance_fallback_sec = 10.0", "variance_fallback_sec = 1.2")
    dest = tmp_path / "rep_counter_3d.toml"
    dest.write_text(text)
    return load_cfg(dest)


def test_calculate_angle_3d_rotation_invariance():
    a = {"x": 0.0, "y": 1.0, "z": 0.0}
    b = {"x": 0.0, "y": 0.0, "z": 0.0}
    c = {"x": 1.0, "y": 0.0, "z": 0.0}
    base = calculate_angle_3d(a, b, c)
    assert abs(base - 90.0) < 1e-6

    def rot_z(p, deg):
        r = math.radians(deg)
        return {
            "x": p["x"] * math.cos(r) - p["y"] * math.sin(r),
            "y": p["x"] * math.sin(r) + p["y"] * math.cos(r),
            "z": p["z"],
        }

    def rot_y(p, deg):
        r = math.radians(deg)
        return {
            "x": p["x"] * math.cos(r) + p["z"] * math.sin(r),
            "y": p["y"],
            "z": -p["x"] * math.sin(r) + p["z"] * math.cos(r),
        }

    for deg in (15, 90, 180, 247):
        a2, b2, c2 = rot_z(a, deg), rot_z(b, deg), rot_z(c, deg)
        a3, b3, c3 = rot_y(a2, deg), rot_y(b2, deg), rot_y(c2, deg)
        assert abs(calculate_angle_3d(a3, b3, c3) - base) < 1e-6


def test_calculate_angle_3d_rejects_degenerate_and_nonfinite():
    origin = {"x": 0.0, "y": 0.0, "z": 0.0}
    with pytest.raises(ValueError, match="non-zero"):
        calculate_angle_3d(origin, origin, {"x": 1.0, "y": 0.0, "z": 0.0})
    with pytest.raises(ValueError, match="finite"):
        calculate_angle_3d({"x": float("nan"), "y": 0, "z": 0}, origin, {"x": 1, "y": 0, "z": 0})
    with pytest.raises(ValueError, match="3 coordinates"):
        calculate_angle_3d((0.0, 1.0), origin, (1.0, 0.0, 0.0))


def test_load_3d_config_strict_errors(tmp_path: Path):
    good = (Path(__file__).resolve().parents[1] / "rep_counter_3d.toml").read_text()
    extra_section = tmp_path / "extra_section.toml"
    extra_section.write_text(good + "\n[fallback_y_point]\nmin_sec = 1\n")
    with pytest.raises(ValueError, match="unknown sections"):
        load_3d_config(extra_section)
    extra = tmp_path / "extra.toml"
    extra.write_text(good + "\n[rep.vertical_px]\nmin_range_gate = 1\n")
    with pytest.raises(ValueError, match="nested tables"):
        load_3d_config(extra)
    missing = tmp_path / "missing.toml"
    missing.write_text("[rep]\npeak_margin_pct = 0.5\n")
    with pytest.raises(ValueError, match="missing required"):
        load_3d_config(missing)
    bad_key = tmp_path / "bad_key.toml"
    bad_key.write_text(good.replace("smoothing_factor = 0.70", "smoothing_factor = 0.70\nshrug_gain = 1"))
    with pytest.raises(ValueError, match="unknown keys"):
        load_3d_config(bad_key)


def test_synthetic_zero_at_lock_then_cumulative_credit(fast_cfg):
    """Lock stays at 0; first post-lock completed cycle reveals full raw count."""
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    lock_frame: int | None = None
    raw_at_lock: int | None = None
    last = None
    for i in range(200):
        last = session.process_frame(_frame("left_elbow", _sine(i)), timestamp_ms=i * (1000 / 30))
        if last.tracked_joint_changed:
            lock_frame = i
            assert last.reps == 0
            assert last.tracked_joint == "left_elbow"
            assert last.phase == "tracking"
            assert last.calibration_started is True
            assert session._count_detector is session._select_detectors["left_elbow"]
            raw_at_lock = int(session._count_detector.rep_count)
            break
    assert lock_frame is not None and raw_at_lock is not None and last is not None

    revealed = False
    prev_displayed = 0
    for i in range(lock_frame + 1, lock_frame + 120):
        last = session.process_frame(_frame("left_elbow", _sine(i)), timestamp_ms=i * (1000 / 30))
        raw = int(session._count_detector.rep_count)  # type: ignore[union-attr]
        if not revealed:
            if raw > raw_at_lock:
                assert last.reps == raw
                revealed = True
                prev_displayed = last.reps
            else:
                assert last.reps == 0
        else:
            assert last.reps >= prev_displayed
            prev_displayed = last.reps
    assert revealed
    assert last.reps >= raw_at_lock + 1
    assert last.tracked_joint == "left_elbow"


def test_credit_reveal_jumps_from_prelock_to_next(fast_cfg):
    """Selection cycles stay hidden; next completed cycle reveals cumulative total."""
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    lock_i = None
    last = None
    for i in range(250):
        last = session.process_frame(
            _frame("left_elbow", _sine(i, period=16, amp=55.0)),
            timestamp_ms=i * (1000 / 30),
        )
        if last.tracked_joint_changed:
            lock_i = i
            raw = int(session._count_detector.rep_count)  # type: ignore[union-attr]
            assert raw >= 1
            assert last.reps == 0
            break
    assert lock_i is not None
    raw_at_lock = int(session._count_detector.rep_count)  # type: ignore[union-attr]

    for i in range(lock_i + 1, lock_i + 100):
        last = session.process_frame(
            _frame("left_elbow", _sine(i, period=16, amp=55.0)),
            timestamp_ms=i * (1000 / 30),
        )
        raw = int(session._count_detector.rep_count)  # type: ignore[union-attr]
        if raw > raw_at_lock:
            assert last.reps == raw
            assert last.reps == raw_at_lock + 1
            break
        assert last.reps == 0
    else:
        pytest.fail("never revealed cumulative credit after lock")


def test_dominance_lock_credits_selection_reps(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    lock_i = None
    last = None
    for i in range(120):
        samples = {k: _obs(90.0) for k in PRODUCT_ANGLE_KEYS}
        samples["right_knee"] = _obs(_sine(i, amp=55.0, mid=110.0, period=18))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        if last.tracked_joint_changed:
            lock_i = i
            assert last.tracked_joint == "right_knee"
            assert last.reps == 0
            break
    assert lock_i is not None and last is not None
    raw_at_lock = int(session._count_detector.rep_count)  # type: ignore[union-attr]
    assert session._count_detector is session._select_detectors["right_knee"]

    for i in range(lock_i + 1, lock_i + 100):
        samples = {k: _obs(90.0) for k in PRODUCT_ANGLE_KEYS}
        samples["right_knee"] = _obs(_sine(i, amp=55.0, mid=110.0, period=18))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        raw = int(session._count_detector.rep_count)  # type: ignore[union-attr]
        if raw > raw_at_lock:
            assert last.reps == raw
            break
        assert last.reps == 0
    else:
        pytest.fail("dominance path never revealed selection credit")


def test_variance_timeout_lock_credits_selection_reps(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    lock_i = None
    last = None
    for i in range(120):
        samples = {k: _obs(90.0 + 0.2 * math.sin(i / 7.0)) for k in PRODUCT_ANGLE_KEYS}
        samples["left_hip"] = _obs(_sine(i, amp=40.0, mid=95.0, period=22))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        if last.tracked_joint_changed:
            lock_i = i
            assert last.tracked_joint == "left_hip"
            assert last.reps == 0
            break
    assert lock_i is not None and last is not None
    raw_at_lock = int(session._count_detector.rep_count)  # type: ignore[union-attr]

    for i in range(lock_i + 1, lock_i + 100):
        samples = {k: _obs(90.0 + 0.2 * math.sin(i / 7.0)) for k in PRODUCT_ANGLE_KEYS}
        samples["left_hip"] = _obs(_sine(i, amp=40.0, mid=95.0, period=22))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        raw = int(session._count_detector.rep_count)  # type: ignore[union-attr]
        if raw > raw_at_lock:
            assert last.reps == raw
            break
        assert last.reps == 0
    else:
        pytest.fail("variance-timeout path never revealed selection credit")


def test_zero_prelock_reps_still_starts_at_zero(fast_cfg):
    """If lock happens with raw=0, display stays 0 until first completed cycle (shows 1)."""
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    lock_i = None
    last = None
    for i in range(80):
        samples = {k: _obs(90.0 + 0.2 * math.sin(i / 7.0)) for k in PRODUCT_ANGLE_KEYS}
        samples["left_hip"] = _obs(_sine(i, amp=40.0, mid=95.0, period=22))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        if last.tracked_joint_changed:
            lock_i = i
            break
    assert lock_i is not None
    raw_at_lock = int(session._count_detector.rep_count)  # type: ignore[union-attr]
    assert last.reps == 0

    for i in range(lock_i + 1, lock_i + 100):
        samples = {k: _obs(90.0) for k in PRODUCT_ANGLE_KEYS}
        samples["left_hip"] = _obs(_sine(i, amp=40.0, mid=95.0, period=22))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        raw = int(session._count_detector.rep_count)  # type: ignore[union-attr]
        if raw > raw_at_lock:
            assert last.reps == raw
            if raw_at_lock == 0:
                assert last.reps == 1
            break
        assert last.reps == 0


def test_carried_calibration_metadata_at_lock(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    last = None
    for i in range(200):
        last = session.process_frame(
            _frame("left_elbow", _sine(i, period=16, amp=55.0)),
            timestamp_ms=i * (1000 / 30),
        )
        if last.tracked_joint_changed:
            det = session._count_detector
            assert det is not None
            assert last.calibration_started is True
            assert last.calibration_complete is bool(det._calibrated)
            assert last.calibration_locked is bool(det._calibrated)
            assert last.smoothed_value == det.smoothed_value
            break
    else:
        pytest.fail("never locked")


def test_unknown_hold_while_credit_pending(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    lock_i = None
    last = None
    for i in range(200):
        last = session.process_frame(_frame("left_elbow", _sine(i)), timestamp_ms=i * (1000 / 30))
        if last.tracked_joint_changed:
            lock_i = i
            assert last.reps == 0
            break
    assert lock_i is not None
    raw_at_lock = int(session._count_detector.rep_count)  # type: ignore[union-attr]

    for j in range(8):
        held = session.process_frame(_all_unknown(), timestamp_ms=(lock_i + 1 + j) * (1000 / 30))
        assert held.reps == 0
        assert held.tracked_joint == "left_elbow"
        assert held.tracked_joint_changed is False
        assert int(session._count_detector.rep_count) == raw_at_lock  # type: ignore[union-attr]

    resume = lock_i + 10
    for i in range(resume, resume + 100):
        last = session.process_frame(_frame("left_elbow", _sine(i)), timestamp_ms=i * (1000 / 30))
        raw = int(session._count_detector.rep_count)  # type: ignore[union-attr]
        if raw > raw_at_lock:
            assert last.reps == raw
            break
        assert last.reps == 0
    else:
        pytest.fail("credit never revealed after hold")


def test_dominance_lock(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    last = None
    for i in range(80):
        samples = {k: _obs(90.0) for k in PRODUCT_ANGLE_KEYS}
        samples["right_knee"] = _obs(_sine(i, amp=55.0, mid=110.0, period=18))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        if last.phase == "tracking":
            break
    assert last is not None
    assert last.tracked_joint == "right_knee"


def test_variance_timeout_lock(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    last = None
    for i in range(80):
        samples = {k: _obs(90.0 + 0.2 * math.sin(i / 7.0)) for k in PRODUCT_ANGLE_KEYS}
        samples["left_hip"] = _obs(_sine(i, amp=40.0, mid=95.0, period=22))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        if last.phase == "tracking":
            break
    assert last is not None
    assert last.tracked_joint == "left_hip"


def test_none_state_holding(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    last = None
    for i in range(70):
        last = session.process_frame(_frame("left_elbow", _sine(i)), timestamp_ms=i * (1000 / 30))
    assert last.phase == "tracking"
    # Advance past credit-pending so hold asserts on a stable revealed count.
    start = 70
    for i in range(start, start + 80):
        last = session.process_frame(_frame("left_elbow", _sine(i)), timestamp_ms=i * (1000 / 30))
        if last.reps > 0:
            break
    assert last.reps > 0
    reps = last.reps
    smoothed = last.smoothed_value
    for j in range(10):
        held = session.process_frame(_all_unknown(), timestamp_ms=(start + 80 + j) * (1000 / 30))
        assert held.reps == reps
        assert held.tracked_joint == "left_elbow"
        assert held.smoothed_value == smoothed
        assert held.tracked_joint_changed is False


def test_predicted_evidence_does_not_lock_until_observed(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    last = None
    for i in range(60):
        samples = {k: _pred(90.0) for k in PRODUCT_ANGLE_KEYS}
        samples["left_shoulder"] = _pred(_sine(i, amp=48.0))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
    assert last.phase == "selecting"
    assert last.tracked_joint is None
    for i in range(60, 130):
        samples = {k: _obs(90.0) for k in PRODUCT_ANGLE_KEYS}
        samples["left_shoulder"] = _obs(_sine(i, amp=48.0))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        if last.phase == "tracking":
            break
    assert last.tracked_joint == "left_shoulder"


def test_tracked_joint_immutable_after_lock(fast_cfg):
    session = AngleRepCounterSession(fast_cfg, fps_hint=30.0)
    last = None
    for i in range(70):
        last = session.process_frame(_frame("left_elbow", _sine(i)), timestamp_ms=i * (1000 / 30))
    assert last.tracked_joint == "left_elbow"
    for i in range(70, 120):
        samples = {k: _obs(90.0) for k in PRODUCT_ANGLE_KEYS}
        samples["left_elbow"] = _obs(_sine(i, amp=20.0))
        samples["right_hip"] = _obs(_sine(i, amp=70.0, period=16))
        last = session.process_frame(samples, timestamp_ms=i * (1000 / 30))
        assert last.tracked_joint == "left_elbow"
        assert last.tracked_joint_changed is False


def test_samples_must_be_exactly_eight_product_keys(fast_cfg):
    session = AngleRepCounterSession(fast_cfg)
    with pytest.raises(ValueError, match="exactly the 8"):
        session.process_frame({"left_elbow": _obs(90.0)})
