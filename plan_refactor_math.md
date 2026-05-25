# Recalibration math refactor

Refactor the codebase into a **two-score system + a three-way handoff classifier**:

1. **Fast-start score (first lock)**: keep `medianWindowVariance` + `activeWindowCount` + `smoothedRangeDeg` + landmark confidence.
2. **Recalibration score (after first lock)**: replace variance-driven selection with a rep-cycle quality score over per-joint motion state that stays alive in the background.
3. **Handoff classifier**: when a switch is requested, decide whether it is **alternate-limb** (add candidate pre-calibrated reps), **dual-limb same-exercise** (continue linearly), or **ambiguous** (wait).

The two target scenarios this plan must satisfy:

- **Scenario 1 — alternate-limb biceps curl.** Right arm does 10 reps, left arm does 10 reps. Reps from the *candidate* limb that accumulated since **handoff observation start** (not just formal pending-start) are added on activation. Visible count progresses right side `~4 → 10`, then jumps to `~14`, then progresses to `~20`.
- **Scenario 2 — same-exercise, better joint (e.g. bodyweight squat).** Both knees track the same cycles. When the second joint is selected, the count must continue **linearly** with no addition and no skip.
- **Scenario 3-FOV — same-exercise, first joint disappears.** User steps so the originally-selected joint loses pose confidence. If the second joint was *already cycling on the same rhythm before the disappearance*, treat as same-exercise (linear continuation). If not, the switch is ambiguous and should not add reps.

---

# 0. Anchoring in the current code

The orchestration this plan replaces lives in `src/flexible_rep_counter/session.py`. The pieces that matter:

- Per-joint angle histories: `selection_angle_histories` (deque-per-joint) — kept alive, but no live `PeakDetector` per joint after the first lock.
- Background detectors during *selection only*: `selection_detectors_by_angle` is created the first time `step_landmarks` runs and is reused inside the selecting phase; after lock, `peak_detector` is the only one being fed in tracking.
- Re-evaluation entry: the `re_eval_due` branch (≈ `session.py:1162-1499`) computes candidates from `determine_best_angle` plus the two fallback paths `stale_selected_fallback` and `range_gate_fallback`.
- Pending switch state: `pending_switch_angle`, `pending_switch_detector`, `pending_switch_incumbent_*`, `pending_switch_candidate_*`, `pending_switch_observed`, `pending_switch_incumbent_moving`, all cleared by `_clear_pending_switch`.
- Activation: `_activate_joint_switch` (`session.py:279-321`) decides whether to add candidate pending reps via `include_candidate_pending_reps`.
- Replay fallback for cold candidates: `_rebuild_detector_from_history` (`session.py:245-262`) — needs `SWITCH_REPLAY_MIN_VALID_SAMPLES = 10` valid samples, which is brittle.

After this refactor, `selection_detectors_by_angle` becomes a permanent fixture (now `joint_motion_states`), `_rebuild_detector_from_history` becomes a fallback only, and `_activate_joint_switch` consults the new handoff classifier rather than the bare `incumbent_moving` boolean.

---

# 1. Keep background detectors alive (always, all joints)

After first lock, every joint in `COMMON_ANGLES` keeps a live `JointMotionState`. The per-frame cost of `PeakDetector.update` is negligible compared to YOLO pose inference, so do not optimize this by sub-sampling joints — uniform per-frame updates simplify cadence / ROM statistics enormously.

```python
@dataclass
class JointMotionState:
    angle_key: str
    detector: PeakDetector
    history: deque[float]                       # smoothed angle, fixed maxlen
    confidence_history: deque[tuple[int, float]]  # (timestamp_ms, avg_landmark_conf)

    last_raw_rep_count: int = 0
    last_rep_timestamp_ms: int | None = None
    last_observed_timestamp_ms: int | None = None  # last frame with usable angle

    recent_roms: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_peaks: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_valleys: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_intervals_ms: deque[int] = field(default_factory=lambda: deque(maxlen=8))

    # NEW: timestamped cycle log so we can ask "did this joint complete cycles
    # BEFORE timestamp T?" during handoff classification.
    cycle_log: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=16))
    # Each entry: (timestamp_ms_of_cycle_completion, ROM_deg)

    last_score: float = 0.0
    last_score_debug: dict[str, float] = field(default_factory=dict)
```

Per-frame update (every joint, every frame):

```python
for angle_key, cfg in COMMON_ANGLES.items():
    val = calculate_from_type(cfg["type"], cfg["landmarks"], lm)
    conf = get_min_confidence_for_landmarks(lm, cfg["landmarks"])

    state = joint_states[angle_key]
    if val is None or conf is None or conf < FRAME_MIN_CONFIDENCE:
        # Do NOT advance the detector; just log the unobservable frame.
        state.history.append(None)
        state.confidence_history.append((timestamp_ms, conf or 0.0))
        continue

    state.history.append(val)
    state.confidence_history.append((timestamp_ms, conf))
    state.last_observed_timestamp_ms = timestamp_ms

    prev_raw = state.detector.get_rep_count()
    result = state.detector.update(val)
    new_raw = state.detector.get_rep_count()

    update_joint_motion_stats(state, result, timestamp_ms, advanced=(new_raw > prev_raw))
```

**Background detectors never affect the displayed rep count.** They only feed the candidate scoring and the handoff classifier. The displayed count is still owned by the joint at `selected_angle`.

---

# 2. Cycle gating — when does a cycle "count" as evidence?

This is the new rule that prevents a quiet limb from looking falsely "consistent":

A completed ROM is appended to `recent_roms` / `cycle_log` only if:

```python
def cycle_is_evidence(state: JointMotionState, rom: float, recent_range_deg: float) -> bool:
    return (
        rom >= MIN_EVIDENCE_ROM_DEG          # e.g. 12.0
        and recent_range_deg >= MIN_EVIDENCE_RANGE_DEG  # e.g. 12.0
        and state.last_observed_timestamp_ms is not None
    )
```

Where `recent_range_deg = p95(history[-N:]) - p5(history[-N:])` over the last ~45 frames.

Why this matters: during scenario 1's right-arm phase, the *left* joint's history is mostly flat. Without gating, a single noisy peak/valley would silently add a "cycle" to `recent_roms` and inflate the candidate's score. With gating, the candidate must actually be moving with rep-sized amplitude before any cycle counts as evidence — and `completed_cycles` honestly reflects "real reps observed by this joint".

---

# 3. Cycle-quality components

```python
score =
    0.25 * activity_score +
    0.30 * rom_consistency_score +
    0.15 * extrema_stability_score +
    0.15 * cadence_score +
    0.15 * pose_confidence_score
```

Evidence-weighted blend:

```python
completed_cycles = len(state.recent_roms)
evidence = 1.0 - exp(-completed_cycles / 2.0)

recalibration_score = (
    evidence * cycle_quality_score
    + (1.0 - evidence) * variance_prior_score
)
```

So:

```text
0 reps  → mostly variance prior
1 rep   → mix
2 reps  → ~0.63 cycle quality, ~0.37 variance
3 reps  → ~0.78 cycle quality
4+ reps → strong cycle-quality lead
```

## A) Activity score

```python
recent_range = p95(history[-45:]) - p5(history[-45:])
activity_score = smoothstep(MIN_RANGE_DEG, GOOD_RANGE_DEG, recent_range)
# MIN_RANGE_DEG = 12, GOOD_RANGE_DEG = 35
```

## B) ROM consistency score

```python
def robust_cv(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    med = median(values)
    mad = median(abs(v - med) for v in values)
    robust_std = 1.4826 * mad
    return robust_std / max(abs(med), 1e-6)

rom_score = exp(-robust_cv(roms) / 0.25) if len(roms) >= 2 else 0.0
```

This is the main recalibration signal.

## C) Extrema stability score (normalized by ROM)

```python
extrema_jitter = (robust_std(peaks) + robust_std(valleys)) / max(median(roms), 1e-6)
extrema_score = exp(-extrema_jitter / 0.35)
```

## D) Cadence score

```python
cadence_score = exp(-robust_cv(recent_intervals_ms) / 0.40)
```

Not too strict — real exercises vary tempo.

## E) Pose confidence score

```python
recent_confs = [c for _, c in state.confidence_history[-N:]]  # includes zeros
avg_conf = mean(recent_confs) if recent_confs else 0.0
visible_fraction = (
    sum(c >= FRAME_MIN_CONFIDENCE for c in recent_confs) / len(recent_confs)
    if recent_confs else 0.0
)
 pose_score = clamp((avg_conf - 0.35) / 0.40, 0.0, 1.0)

observable = (
    visible_fraction >= 0.50
    and state.last_observed_timestamp_ms is not None
    and (now_ms - state.last_observed_timestamp_ms) <= 750
)
```

Treat any joint with `pose_score < 0.30` as **unobservable** for handoff purposes (see §6).

---

# 4. Candidate scoring utility

```python
def compute_joint_recalibration_score(
    state: JointMotionState,
    variance_data: dict,
    now_ms: int,
) -> tuple[float, dict]:
    recent_values = [v for v in list(state.history)[-45:] if v is not None]
    recent_range = (
        percentile(recent_values, 95) - percentile(recent_values, 5)
        if len(recent_values) >= 8 else 0.0
    )

    variance_prior = normalize_variance_prior(variance_data)
    activity_score = smoothstep(12.0, 35.0, recent_range)

    roms     = list(state.recent_roms)
    peaks    = list(state.recent_peaks)
    valleys  = list(state.recent_valleys)
    intervals = list(state.recent_intervals_ms)

    completed_cycles = len(roms)
    evidence = 1.0 - math.exp(-completed_cycles / 2.0)

    rom_score = math.exp(-robust_cv(roms) / 0.25) if completed_cycles >= 2 else 0.0

    if len(peaks) >= 2 and len(valleys) >= 2 and roms:
        extrema_jitter = (robust_std(peaks) + robust_std(valleys)) / max(median(roms), 1e-6)
        extrema_score = math.exp(-extrema_jitter / 0.35)
    else:
        extrema_jitter = None
        extrema_score = 0.0

    cadence_score = (
        math.exp(-robust_cv(intervals) / 0.40) if len(intervals) >= 2
        else (0.5 if completed_cycles >= 1 else 0.0)
    )

    recent_confs = [c for _, c in state.confidence_history[-30:]]  # includes zeros
    avg_conf = mean(recent_confs) if recent_confs else 0.0
    visible_fraction = (
        sum(c >= FRAME_MIN_CONFIDENCE for c in recent_confs) / len(recent_confs)
        if recent_confs else 0.0
    )
    pose_score = clamp((avg_conf - 0.35) / 0.40, 0.0, 1.0)

    cycle_quality = (
        0.25 * activity_score +
        0.30 * rom_score +
        0.15 * extrema_score +
        0.15 * cadence_score +
        0.15 * pose_score
    )

    score = evidence * cycle_quality + (1.0 - evidence) * variance_prior

    return score, {
        "score": score, "evidence": evidence,
        "completedCycles": completed_cycles,
        "activityScore": activity_score, "variancePrior": variance_prior,
        "romScore": rom_score, "romCv": robust_cv(roms) if completed_cycles >= 2 else None,
        "extremaScore": extrema_score, "extremaJitter": extrema_jitter,
        "cadenceScore": cadence_score,
        "intervalCv": robust_cv(intervals) if len(intervals) >= 2 else None,
        "poseScore": pose_score, "visibleFraction": visible_fraction, "recentRange": recent_range,
    }
```

---

# 5. Switch decision (which joint wins?)

```python
incumbent_bad = (
    selected_stale                                    # raw rep count flat for K re-eval ticks
    or selected_recent_range < STALE_SWITCH_MAX_SELECTED_RECENT_RANGE_DEG
    or selected_range_gate_closed_streak >= STALE_SWITCH_MIN_CLOSED_STREAK
    or selected_score < 0.35
    or selected_pose_score < 0.30                     # disappeared / occluded
)

candidate_good = (
    candidate_score >= 0.62
    and candidate_activity_score >= 0.45
    and candidate_pose_score >= 0.45
    and candidate.completed_cycles >= 2               # gated cycles only
)

candidate_clearly_better = (
    candidate_score >= selected_score + 0.18
    or candidate_score >= selected_score * 1.35
)

should_switch = cooldown_ok and incumbent_bad and candidate_good and candidate_clearly_better
```

Force-stale path (only when truly stuck):

```python
force_switch = (
    stale_reevals >= STALE_SWITCH_FORCE_AFTER_STALE_REEVALS
    and candidate_score >= 0.50
    and candidate_activity_score >= 0.50
    and candidate_pose_score >= 0.45
    and candidate.completed_cycles >= 2
)
```

**Remove** the current absolute-variance bypass `STALE_SWITCH_FORCE_MIN_OPPOSITE_VAR = 40.0`. Once cycle-quality is the main signal, letting variance overrule cycle quality re-introduces the noisy-but-high-variance failure mode.

---

# 6. Handoff classifier (the new section)

This is the piece that explicitly covers both scenarios and the FOV edge case. It runs only once a switch has been decided (`should_switch == True` or `force_switch == True`) and replaces the binary `pending_switch_incumbent_moving` flag in `_activate_joint_switch`.

## Inputs

At pending-start time, snapshot:

```python
incumbent_pose_score_at_start
incumbent_recent_range_at_start
incumbent_completed_cycles_at_start
incumbent_last_observed_ts_at_start
candidate_completed_cycles_at_start
candidate_last_cycle_ts_before_start   # newest cycle_log entry with ts < pending_start_ts
candidate_rom_score_at_start
pending_start_ts

# NEW: carryover baseline captured earlier at handoff-observation start so
# candidate reps accrued before formal pending are not lost.
candidate_carryover_start_shown
candidate_carryover_start_raw
candidate_carryover_start_ts

# NEW: pre-handoff synchrony snapshot (last ~4 s)
incumbent_cycles_last_4s
candidate_cycles_last_4s
cycle_sync_score_last_4s
same_joint_family
mirrored_pair
```

During pending, accumulate:

```python
incumbent_advanced               # raw rep count increased while pending
incumbent_motion_span_deg        # max(history) - min(history) over pending window (NaN if not observable)
incumbent_observable_during_pending  # any frame in pending with pose_conf >= 0.35
incumbent_completed_gated_cycle_during_pending
candidate_observed               # pending detector saw at least one update
candidate_advanced_during_pending
candidate_completed_gated_cycle_during_pending
candidate_pending_rom_estimate_deg
```

## Three-way decision at activation

```python
def classify_handoff(state) -> Literal["alternate_limb", "same_exercise", "ambiguous"]:

    # 1. Incumbent kept actively repping or completed a gated cycle
    #    during pending → same exercise.
    if (
        state.incumbent_advanced
        or state.incumbent_completed_gated_cycle_during_pending
        or state.incumbent_motion_span_deg >= max(12.0, 0.40 * state.candidate_pending_rom_estimate_deg)
    ):
        return "same_exercise"

    # 2. Incumbent disappeared (lost pose), but candidate had prior synchronized
    #    cycles in the same joint family before disappearance → same exercise.
    incumbent_disappeared = (
        not state.incumbent_observable_during_pending
        or state.incumbent_pose_score_at_start < 0.30
        or (
            state.incumbent_last_observed_ts_at_start is not None
            and (state.pending_start_ts - state.incumbent_last_observed_ts_at_start) > 750
        )
    )
    candidate_had_prior_cycles = (
        state.candidate_completed_cycles_at_start >= 2
        and state.candidate_last_cycle_ts_before_start is not None
        and (state.pending_start_ts - state.candidate_last_cycle_ts_before_start) <= 4000  # ms
        and state.candidate_rom_score_at_start >= 0.50
        and state.same_joint_family
        and state.cycle_sync_score_last_4s >= 0.60
    )
    if incumbent_disappeared and candidate_had_prior_cycles:
        return "same_exercise"

    # 3. Incumbent observable-but-quiet, candidate actually advanced during pending,
    #    and pair is mirrored (e.g. right_elbow <-> left_elbow) → alternate-limb.
    incumbent_quiet_but_visible = (
        state.incumbent_observable_during_pending
        and state.incumbent_motion_span_deg < 12.0
        and not state.incumbent_advanced
    )
    if (
        incumbent_quiet_but_visible
        and state.mirrored_pair
        and (
            state.candidate_advanced_during_pending
            or state.candidate_completed_gated_cycle_during_pending
        )
    ):
        return "alternate_limb"

    # 4. Anything else (cross-family switches, weak evidence, unsynced cycles) is ambiguous.
    return "ambiguous"
```

## How activation uses the classifier

This replaces the `include_candidate_pending_reps` branch in `_activate_joint_switch`:

```python
kind = classify_handoff(state)

candidate_delta_shown = max(0, candidate_current_shown - candidate_carryover_start_shown)
candidate_delta_raw   = max(0, candidate_current_raw   - candidate_carryover_start_raw)

if kind == "alternate_limb":
    target_display_shown = cumulative_shown + candidate_delta_shown
    target_display_raw   = cumulative_raw   + candidate_delta_raw

else:  # "same_exercise" or "ambiguous"
    target_display_shown = cumulative_shown
    target_display_raw   = cumulative_raw

rep_count_offset     = target_display_shown - candidate_current_shown
rep_count_raw_offset = target_display_raw   - candidate_current_raw

if kind in ("same_exercise", "ambiguous"):
    assert candidate_current_shown + rep_count_offset == cumulative_shown
if kind == "alternate_limb":
    assert candidate_current_shown + rep_count_offset == cumulative_shown + candidate_delta_shown
```

This gives:

- **Scenario 1** → `alternate_limb` → visible count jumps `10 → ~14` on activation. ✅
- **Scenario 2** → `same_exercise` (incumbent kept moving) → no jump, no skip. ✅
- **Scenario 3-FOV** → `same_exercise` (incumbent disappeared, but candidate had prior synchronized cycles) → no wrong jump. ✅
- **Truly new movement after long blackout** → `ambiguous` → no jump (safer than a wrong addition). ✅

## Monotonicity invariant

After activation, `displayed_count` must never decrease and must equal:

- `cumulative_shown + delta(candidate_since_handoff_observation_start)` in the alt-limb case;
- `cumulative_shown` in the same-exercise / ambiguous cases.

Add assertions in `_activate_joint_switch` for:
- exact target equality (no wrong forward jump),
- monotonicity (`after >= before`),
- and scenario-specific behavior (`same_exercise`/`ambiguous` must hold count at activation).

---

# 7. Pending switch state machine

Update the pending state to include the fields needed by `classify_handoff`:

```python
pending_switch_started_at_ms: int
pending_switch_angle: str
pending_switch_detector: PeakDetector

# Incumbent snapshot at start
pending_switch_incumbent_pose_score_at_start: float
pending_switch_incumbent_completed_cycles_at_start: int
pending_switch_incumbent_recent_range_at_start: float
pending_switch_incumbent_last_observed_ts_at_start: int | None

# Candidate snapshot at start (NEW)
pending_switch_candidate_completed_cycles_at_start: int
pending_switch_candidate_last_cycle_ts_before_start: int | None
pending_switch_candidate_rom_score_at_start: float

# Running flags
pending_switch_observed: bool
pending_switch_incumbent_advanced: bool
pending_switch_incumbent_motion_span_deg: float       # max - min during pending
pending_switch_incumbent_observable_during_pending: bool
pending_switch_incumbent_completed_gated_cycle_during_pending: bool
pending_switch_candidate_advanced_during_pending: bool
pending_switch_candidate_completed_gated_cycle_during_pending: bool
pending_switch_candidate_pending_rom_estimate_deg: float

# Carryover baseline (captured at handoff-observation start, not pending-start)
pending_switch_candidate_carryover_start_shown: int
pending_switch_candidate_carryover_start_raw: int
pending_switch_candidate_carryover_start_ts: int

# Pre-handoff synchrony descriptors
pending_switch_incumbent_cycles_last_4s: int
pending_switch_candidate_cycles_last_4s: int
pending_switch_cycle_sync_score_last_4s: float
pending_switch_same_joint_family: bool
pending_switch_mirrored_pair: bool
```

Lifecycle:

```text
handoff-observation start trigger
    → incumbent looks stale/quiet and candidate shows meaningful activity
    → snapshot carryover baseline now (candidate_carryover_start_*)
    → keep observing both joints

should_switch && candidate already calibrated
    → run a short observation window (e.g. 300-600 ms) if pending is empty
      so classifier has real incumbent/candidate evidence

should_switch && candidate not calibrated
    → enter pending; both incumbent and candidate continue updating
    → when candidate calibrates OR force_activation_after_seconds elapses,
      run classify_handoff() and apply the matching offset rule
```

Note: with background detectors alive, the "candidate not calibrated" branch is rare — most candidates are already calibrated by the time their score crosses the threshold.

---

# 8. File layout

New module:

```text
src/flexible_rep_counter/core/recalibration_confidence.py
```

Contents:

```python
@dataclass
class JointMotionState: ...

@dataclass
class JointRecalibrationScore: ...

@dataclass
class HandoffDecision:
    kind: Literal["alternate_limb", "same_exercise", "ambiguous"]
    rationale: dict[str, Any]   # for instrumentation

def update_joint_motion_state(state, val, conf, timestamp_ms) -> None: ...
def compute_joint_recalibration_score(state, variance_data, now_ms) -> tuple[float, dict]: ...
def select_recalibration_candidate(states, selected_angle) -> tuple[str | None, dict]: ...
def should_switch_to_candidate(selected_score, candidate_score, ...) -> bool: ...
def classify_handoff(pending_state) -> HandoffDecision: ...
```

`session.py` orchestration becomes:

```python
update_all_joint_motion_states(joint_states, lm, timestamp_ms)

if re_eval_due:
    scores = {k: compute_joint_recalibration_score(s, variances.get(k), now_ms)
              for k, s in joint_states.items()}
    candidate, debug = select_recalibration_candidate(scores, selected_angle)
    if should_switch_to_candidate(...):
        begin_pending_switch(candidate, snapshot_now())

if pending_switch_active:
    update_pending_window(...)
    if candidate_calibrated or pending_window_force_activate:
        decision = classify_handoff(pending_state)
        activate_joint_switch(decision)
```

This makes the entire recalibration pipeline testable without the full session loop.

---

# 9. Readiness timeline

```text
0–3 sec:
    Initial selection runs the existing variance-based path.

After first lock:
    All joints get JointMotionState updates every frame.
    Cycle gating (recent_range >= 12 deg) protects evidence quality.

candidate.completed_cycles == 0:
    Never switch. Candidate can only be monitored.

candidate.completed_cycles == 1:
    Candidate may start handoff-observation bookkeeping, but cannot activate.

candidate.completed_cycles >= 2:
    Candidate may enter pending and may activate if classifier evidence is
    consistent (scenarios 1 / 2 / 3-FOV).

candidate.completed_cycles >= 3 with rom_score >= 0.6:
    Candidate may override faster if clearly better and cooldown allows;
    classifier still controls rep carryover policy.
```

---

# 10. Code paths to retire or simplify

Remove or fold these once the new pipeline is in place:

- `STALE_SWITCH_FORCE_MIN_OPPOSITE_VAR` and the absolute-variance bypass in `session.py:1244-1266`. The new force path uses cycle-quality score, not raw variance.
- `STALE_SWITCH_MIN_VAR_RATIO`. Same reason.
- `_rebuild_detector_from_history` becomes a fallback only (used if `JointMotionState` was reset by `clear_tracking_keep_started`). Background detectors make replay the exception, not the rule.
- `pending_switch_incumbent_moving` (bool). Subsumed by the richer pending-state fields and `classify_handoff`.
- The `range_gate_fallback` re-evaluation source. It overlaps with `incumbent_bad` in §5 (`selected_range_gate_closed_streak >= STALE_SWITCH_MIN_CLOSED_STREAK`) and is no longer needed as a separate path.

Keep:

- `_apply_locked_tracking`, the first-lock entry point.
- The pending switch *idea* (delayed activation until candidate calibrated). Only the bookkeeping fields and classifier change.
- `selection_angle_histories` deques — they remain useful as the cold-start input when a brand-new `JointMotionState` needs to replay a few frames.

---

# 11. Test plan

Extend `tests/test_dynamic_recalibration.py` with these cases. The new tests should drive `JointMotionState` and `classify_handoff` directly so they are not coupled to the full step loop:

1. **`test_alternate_limb_adds_candidate_pending_reps`**
   Right elbow at `repCount=10`, then stops moving (motion span < 12 deg, still observable). Left elbow accrues 4 cycles during pending. `classify_handoff` returns `alternate_limb`. Visible count: `10 → 14` on activation. *(Generalizes the existing `test_stalled_handoff_adds_candidate_precalibration_reps`.)*

2. **`test_same_exercise_better_joint_continues_linearly`**
   Right knee and left knee both cycle in lockstep. Right knee `repCount=10`. Switch to left. `classify_handoff` returns `same_exercise` because `incumbent_advanced` or `incumbent_motion_span_deg >= 12`. Visible count: `10 → 10` (no jump), then continues `11, 12, …` linearly. *(Generalizes the existing `test_pending_switch_with_incumbent_motion_keeps_linear_count`.)*

3. **`test_fov_disappearance_with_prior_candidate_cycles_continues_linearly`** *(new)*
   Both knees cycle for 5 reps. Then right knee's landmarks drop below `FRAME_MIN_CONFIDENCE` (incumbent becomes unobservable). `classify_handoff` returns `same_exercise` because `candidate_had_prior_cycles == True` and `incumbent_disappeared == True`. No rep addition, monotonic continuation.

4. **`test_fov_disappearance_without_prior_candidate_cycles_is_ambiguous`** *(new)*
   Right elbow cycles at `repCount=10`. Left elbow flat (no cycles in `cycle_log`). Right elbow disappears. Switch to left. `classify_handoff` returns `ambiguous`. Offset chosen to keep visible count at 10; no addition.

5. **`test_cycle_gating_rejects_low_range_cycles`** *(new)*
   Left elbow's `recent_range_deg` is below 12 deg, but the detector produces a peak/valley pair from noise. Cycle is not appended to `recent_roms`; `evidence` stays low; the joint cannot win a switch.

6. **`test_force_stale_requires_two_gated_cycles`** *(new)*
   Selected joint stale for `STALE_SWITCH_FORCE_AFTER_STALE_REEVALS` re-evals, candidate has only 1 gated cycle. Force-switch is denied. Once the candidate accumulates a second gated cycle, force-switch proceeds.

7. **`test_no_displayed_count_regression_on_any_handoff`** *(new, property-style)*
   For each handoff kind, assert `out.reps_after_activation >= out.reps_before_activation`.

8. **`test_same_exercise_never_jumps_forward_on_activation`** *(new)*
   Candidate already has a larger raw/display count than incumbent at activation time. `same_exercise` and `ambiguous` must still anchor exactly to `cumulative_shown` (no forward jump).

9. **`test_cross_family_switch_cannot_be_alternate_limb`** *(new)*
   Example: `RIGHT_ELBOW -> LEFT_KNEE`. Even if incumbent is quiet and candidate moves, classifier returns `ambiguous` unless strong same-exercise synchrony evidence exists.

---

# 12. Conceptual summary

Old system asks once:

> "Which joint is moving the most consistently right now?"

New system asks two questions:

> First selection: "Which joint is currently producing the most rep-like variance and ROM in the warm-up window?"
>
> Recalibration: "Which joint is producing the most rep-like, repeatable peak-to-valley cycles right now, and — if I switch — should I treat the candidate's pending reps as a *different* limb or as the *same* exercise observed by a better joint?"

The three handoff outcomes encode the user-visible behaviour explicitly:

| Scenario                                  | Incumbent at handoff               | Candidate prior cycles | `classify_handoff` | Rep addition |
| ----------------------------------------- | ---------------------------------- | ---------------------- | ------------------ | ------------ |
| Alternate limbs (right then left arm)     | observable + quiet (< 12 deg span) | mirrored pair + candidate advanced | `alternate_limb`   | yes          |
| Same exercise, better joint               | observable + still moving (or incumbent gated cycle) | any | `same_exercise`    | no           |
| Same exercise, first joint out of frame   | unobservable                       | ≥ 2 within 4 s + synchronized | `same_exercise`    | no           |
| Genuinely new movement after long blackout| unobservable                       | weak / unsynced prior cycles | `ambiguous`        | no           |

That table is the contract this refactor must satisfy.
