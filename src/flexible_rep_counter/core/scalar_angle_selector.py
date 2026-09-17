"""Variance-based selection over named scalar angle series (no landmark indices)."""
from __future__ import annotations

from typing import Any, Optional

from flexible_rep_counter.core.math_engine import (
    MIN_VARIANCE_THRESHOLD,
    calculate_variance,
    compute_consistent_variance_score,
    compute_robust_variance,
    smooth_angle_series,
)
from flexible_rep_counter.core.settings_3d import Angle3dConfig
from flexible_rep_counter.types import AngleSample

PRODUCT_ANGLE_KEYS: tuple[str, ...] = (
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)


def require_product_samples(samples: dict[str, AngleSample]) -> dict[str, AngleSample]:
    if not isinstance(samples, dict):
        raise TypeError("samples must be a dict of product angle keys")
    extra = set(samples) - set(PRODUCT_ANGLE_KEYS)
    missing = set(PRODUCT_ANGLE_KEYS) - set(samples)
    if extra or missing:
        raise ValueError(
            "samples must contain exactly the 8 product angles "
            f"{list(PRODUCT_ANGLE_KEYS)}; extra={sorted(extra)} missing={sorted(missing)}"
        )
    out: dict[str, AngleSample] = {}
    for key in PRODUCT_ANGLE_KEYS:
        sample = samples[key]
        if not isinstance(sample, AngleSample):
            raise TypeError(f"{key} must be an AngleSample, got {type(sample)!r}")
        if sample.evidence not in ("observed", "predicted", "unknown"):
            raise ValueError(f"{key}.evidence must be observed|predicted|unknown, got {sample.evidence!r}")
        if sample.value is not None:
            value = float(sample.value)
            if value != value:
                raise ValueError(f"{key}.value is NaN")
            if sample.evidence == "unknown":
                raise ValueError(f"{key} cannot have a value when evidence is unknown")
            out[key] = AngleSample(value=value, evidence=sample.evidence)
        else:
            if sample.evidence != "unknown":
                raise ValueError(f"{key} missing value must use evidence='unknown'")
            out[key] = sample
    return out


def summarize_rep_dominance(rep_counts: dict[str, int]) -> dict[str, Any]:
    positive = {k: int(v) for k, v in rep_counts.items() if int(v) > 0}
    total = sum(positive.values())
    if total <= 0 or not positive:
        return {"totalReps": 0, "leaderKey": None, "leaderReps": 0, "leaderShare": 0.0}
    leader_key = max(positive.keys(), key=lambda k: (positive[k], k))
    leader_reps = positive[leader_key]
    return {
        "totalReps": total,
        "leaderKey": leader_key,
        "leaderReps": leader_reps,
        "leaderShare": leader_reps / total,
    }


def compute_variances_from_histories(
    histories: dict[str, list[Optional[float]]],
    cfg: Angle3dConfig,
) -> dict[str, dict[str, Any]]:
    variances: dict[str, dict[str, Any]] = {}
    for key in PRODUCT_ANGLE_KEYS:
        series = [float(v) for v in histories.get(key) or [] if v is not None]
        if len(series) < 10:
            continue
        smoothed = smooth_angle_series(series, window=cfg.smooth_window)
        min_ws = 15 if len(smoothed) >= 90 else 12
        stats = calculate_variance(smoothed)
        robust = compute_robust_variance(smoothed)
        consistent = compute_consistent_variance_score(smoothed, min_window_size=min_ws)
        span = max(smoothed) - min(smoothed) if len(smoothed) >= 2 else 0.0
        variances[key] = {
            **stats,
            "robustVariance": robust["variance"],
            "medianWindowVariance": consistent["medianWindowVariance"],
            "activeWindowCount": consistent["activeWindowCount"],
            "smoothedRangeDeg": span,
        }
    return variances


def _eligibility(cfg: Angle3dConfig, data: dict[str, Any]) -> tuple[bool, float]:
    consistent_var = float(data.get("medianWindowVariance") or 0.0)
    active_windows = int(data.get("activeWindowCount") or 0)
    span = float(data.get("smoothedRangeDeg") or 0.0)
    if active_windows < cfg.min_active_windows:
        return False, 0.0
    if consistent_var < cfg.min_variance:
        return False, 0.0
    if span < cfg.min_range_deg:
        return False, 0.0
    return True, consistent_var


def observed_fraction(evidence_history: list[str]) -> float:
    if not evidence_history:
        return 0.0
    n = sum(1 for ev in evidence_history if ev == "observed")
    return n / len(evidence_history)


def top_variance_candidate(
    variances: dict[str, dict[str, Any]],
    cfg: Angle3dConfig,
    *,
    observed_fractions: dict[str, float],
) -> Optional[str]:
    ranked: list[tuple[float, str]] = []
    for key, data in variances.items():
        if observed_fractions.get(key, 0.0) < cfg.min_observed_fraction:
            continue
        ok, score = _eligibility(cfg, data)
        if not ok:
            continue
        ranked.append((score, key))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if not ranked:
        return None
    top_score, top_key = ranked[0]
    if len(ranked) >= 2:
        second_score, _ = ranked[1]
        if second_score > 0 and top_score < second_score * cfg.second_best_ratio:
            return None
    return top_key


def dominance_lock_ready(
    variances: dict[str, dict[str, Any]],
    rep_dom: dict[str, Any],
    cfg: Angle3dConfig,
    *,
    observed_fractions: dict[str, float],
) -> Optional[str]:
    leader = rep_dom.get("leaderKey")
    if not isinstance(leader, str):
        return None
    if float(rep_dom.get("leaderShare") or 0.0) <= cfg.dominance_fraction:
        return None
    if int(rep_dom.get("leaderReps") or 0) < cfg.min_leading_reps:
        return None
    if observed_fractions.get(leader, 0.0) < cfg.min_observed_fraction:
        return None
    data = variances.get(leader)
    if not data:
        return None
    ok, _ = _eligibility(cfg, data)
    if not ok:
        return None
    variance_winner = top_variance_candidate(
        variances, cfg, observed_fractions=observed_fractions
    )
    if variance_winner is not None and variance_winner != leader:
        return None
    return leader


def active_window_count_from_series(values: list[float], min_window_size: int = 12) -> int:
    score = compute_consistent_variance_score(values, min_window_size=min_window_size)
    return int(score.get("activeWindowCount") or 0)


def variance_threshold() -> float:
    return MIN_VARIANCE_THRESHOLD
