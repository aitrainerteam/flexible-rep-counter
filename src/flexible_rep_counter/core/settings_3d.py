"""Strict loader for ``rep_counter_3d.toml``."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_ALLOWED_SECTIONS = frozenset({"rep", "angle_selection"})
_REP_KEYS = {
    "peak_margin_pct": float,
    "valley_margin_pct": float,
    "hysteresis": float,
    "min_peak_distance": int,
    "min_range_gate": float,
    "range_window_frames": int,
    "range_min_samples": int,
    "angle_delta_deadband": float,
    "calibration_reps": int,
    "calibration_certainty": float,
    "calibration_force_extra_reps": int,
    "min_interval_ms": float,
    "smoothing_factor": float,
}
_SELECTION_KEYS = {
    "min_sec": float,
    "min_frames": int,
    "max_buffer_frames": int,
    "retry_interval_sec": float,
    "dominance_fraction": float,
    "min_leading_reps": int,
    "dominance_streak_frames": int,
    "variance_fallback_sec": float,
    "min_variance": float,
    "min_range_deg": float,
    "second_best_ratio": float,
    "min_active_windows": int,
    "smooth_window": int,
    "min_observed_fraction": float,
}


def _toml_load_file(f: Any) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.load(f)
    import tomli

    return tomli.load(f)


def resolve_3d_config_path() -> Path:
    env = os.environ.get("FLEXIBLE_REP_COUNTER_3D_CONFIG", "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"FLEXIBLE_REP_COUNTER_3D_CONFIG is not a file: {p}")
        return p.resolve()
    candidates = [_ROOT / "rep_counter_3d.toml"]
    here = Path.cwd()
    for d in [here, *here.parents]:
        candidates.append(d / "rep_counter_3d.toml")
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "rep_counter_3d.toml not found. Set FLEXIBLE_REP_COUNTER_3D_CONFIG or place the file next to the package."
    )


def _require_number(section: str, key: str, raw: Any, kind: type) -> float | int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"[{section}].{key} must be a number, got {raw!r}")
    if kind is int:
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"[{section}].{key} must be an integer, got {raw!r}")
        return int(raw)
    return float(raw)


def _parse_section(name: str, raw: Any, schema: dict[str, type]) -> dict[str, float | int]:
    if not isinstance(raw, dict):
        raise ValueError(f"[{name}] must be a table")
    nested = [key for key, value in raw.items() if isinstance(value, dict)]
    if nested:
        raise ValueError(f"[{name}] nested tables are not allowed: {sorted(nested)}")
    extra = set(raw) - set(schema)
    missing = set(schema) - set(raw)
    if extra:
        raise ValueError(f"[{name}] has unknown keys: {sorted(extra)}")
    if missing:
        raise ValueError(f"[{name}] missing required keys: {sorted(missing)}")
    return {key: _require_number(name, key, raw[key], kind) for key, kind in schema.items()}


@dataclass(frozen=True)
class Angle3dConfig:
    source_path: Path
    peak_margin_pct: float
    valley_margin_pct: float
    hysteresis: float
    min_peak_distance: int
    min_range_gate: float
    range_window_frames: int
    range_min_samples: int
    angle_delta_deadband: float
    calibration_reps: int
    calibration_certainty: float
    calibration_force_extra_reps: int
    min_interval_ms: float
    smoothing_factor: float
    min_sec: float
    min_frames: int
    max_buffer_frames: int
    retry_interval_sec: float
    dominance_fraction: float
    min_leading_reps: int
    dominance_streak_frames: int
    variance_fallback_sec: float
    min_variance: float
    min_range_deg: float
    second_best_ratio: float
    min_active_windows: int
    smooth_window: int
    min_observed_fraction: float

    def peak_detector_kwargs(self) -> dict[str, Any]:
        return {
            "smoothing_factor": self.smoothing_factor,
            "hysteresis": self.hysteresis,
            "min_peak_distance": self.min_peak_distance,
            "peak_margin_pct": self.peak_margin_pct,
            "valley_margin_pct": self.valley_margin_pct,
            "min_range_gate_degrees": self.min_range_gate,
            "range_window_frames": self.range_window_frames,
            "range_min_samples": self.range_min_samples,
            "delta_deadband_degrees": self.angle_delta_deadband,
            "calibration_reps": self.calibration_reps,
            "calibration_certainty": self.calibration_certainty,
            "calibration_force_extra_reps": self.calibration_force_extra_reps,
            "min_rep_interval_ms": self.min_interval_ms,
        }


def load_3d_config(path: Path | None = None) -> Angle3dConfig:
    src = path.resolve() if path is not None else resolve_3d_config_path()
    with src.open("rb") as f:
        raw = _toml_load_file(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{src} must contain TOML tables")
    extra_sections = set(raw) - _ALLOWED_SECTIONS
    missing_sections = _ALLOWED_SECTIONS - set(raw)
    if extra_sections:
        raise ValueError(f"{src} has unknown sections: {sorted(extra_sections)}")
    if missing_sections:
        raise ValueError(f"{src} missing required sections: {sorted(missing_sections)}")
    rep = _parse_section("rep", raw["rep"], _REP_KEYS)
    sel = _parse_section("angle_selection", raw["angle_selection"], _SELECTION_KEYS)
    if not (0.0 <= float(rep["peak_margin_pct"]) <= 1.0):
        raise ValueError("[rep].peak_margin_pct must be in [0, 1]")
    if not (0.0 <= float(rep["valley_margin_pct"]) <= 1.0):
        raise ValueError("[rep].valley_margin_pct must be in [0, 1]")
    if not (0.0 < float(rep["smoothing_factor"]) <= 1.0):
        raise ValueError("[rep].smoothing_factor must be in (0, 1]")
    if not (0.0 <= float(sel["min_observed_fraction"]) <= 1.0):
        raise ValueError("[angle_selection].min_observed_fraction must be in [0, 1]")
    if not (0.0 < float(sel["dominance_fraction"]) < 1.0):
        raise ValueError("[angle_selection].dominance_fraction must be in (0, 1)")
    return Angle3dConfig(
        source_path=src,
        peak_margin_pct=float(rep["peak_margin_pct"]),
        valley_margin_pct=float(rep["valley_margin_pct"]),
        hysteresis=float(rep["hysteresis"]),
        min_peak_distance=int(rep["min_peak_distance"]),
        min_range_gate=float(rep["min_range_gate"]),
        range_window_frames=int(rep["range_window_frames"]),
        range_min_samples=int(rep["range_min_samples"]),
        angle_delta_deadband=float(rep["angle_delta_deadband"]),
        calibration_reps=int(rep["calibration_reps"]),
        calibration_certainty=float(rep["calibration_certainty"]),
        calibration_force_extra_reps=int(rep["calibration_force_extra_reps"]),
        min_interval_ms=float(rep["min_interval_ms"]),
        smoothing_factor=float(rep["smoothing_factor"]),
        min_sec=float(sel["min_sec"]),
        min_frames=int(sel["min_frames"]),
        max_buffer_frames=int(sel["max_buffer_frames"]),
        retry_interval_sec=float(sel["retry_interval_sec"]),
        dominance_fraction=float(sel["dominance_fraction"]),
        min_leading_reps=int(sel["min_leading_reps"]),
        dominance_streak_frames=int(sel["dominance_streak_frames"]),
        variance_fallback_sec=float(sel["variance_fallback_sec"]),
        min_variance=float(sel["min_variance"]),
        min_range_deg=float(sel["min_range_deg"]),
        second_best_ratio=float(sel["second_best_ratio"]),
        min_active_windows=int(sel["min_active_windows"]),
        smooth_window=int(sel["smooth_window"]),
        min_observed_fraction=float(sel["min_observed_fraction"]),
    )
