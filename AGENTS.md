# Agent guidelines — flexible-rep-counter

## Exercise-agnostic design (required)

This repository counts reps from **motion patterns**, not named exercises.

- **Do not** add exercise-specific joints, flags, tuning, or debug fields (e.g. shrug, curl, squat).
- **Do not** add per-limb or per-joint overrides in `rep_counter.toml` (`[rep.joints.*]`, `[angle_selection.joints.*]`).
- **Do** configure by **signal modality** only:
  - `angle_deg` — 3-point angle variance (primary candidates)
  - `vertical_px` — absolute Y displacement fallback candidates (`absolute_y_position`)
- **Do** name `COMMON_ANGLES` keys by **anatomy + signal** (e.g. `LEFT_ELBOW`, `SHOULDER_Y`), never by exercise.
- **Do** keep selection/counting logic generic: variance gates, peak/valley detection, handoff, and fallback apply to any modality.

When fixing bugs (missed reps, late lock, margin_fail, handoff loss), fix the **generic pipeline** — not a single exercise or joint.

## Configuration

| Section | Purpose |
|---------|---------|
| `[rep]` | Global peak-detector defaults |
| `[rep.vertical_px]` | Overrides for vertical displacement signals only |
| `[angle_selection]` | Global selection gates (angle_deg) |
| `[angle_selection.vertical_px]` | Selection gates for vertical_px candidates |
| `[fallback_y_point]` | Fallback arming thresholds (modality-level) |

## Tests

Prefer modality-based assertions (`vertical_px`, `angle_deg`, `detector_label.endswith("_Y")`) over exercise names or a single joint constant.
