# Rep Counter Flags

The `rep_counter` payload in `POST /predict` includes one-frame pulse flags:

- `calibration_started`: calibration just entered the uncalibrated phase.
- `calibration_locked`: calibration just transitioned to complete.
- `tracked_joint_changed`: tracked joint switched on this frame.

These are edge events, not continuous state.

## Low-FPS Safe Mode Telemetry

The low-FPS safeguard adds these fields to `rep_counter` and `frame_snapshot` instrumentation:

- `effective_fps` (`number | null`): estimated effective rate (`1000 / frame_interval_p50_ms`).
- `frame_interval_p50_ms` (`number | null`): median inter-frame interval.
- `frame_interval_p90_ms` (`number | null`): p90 inter-frame interval.
- `low_fps_safe_mode_active` (`bool`): continuous safe-mode state.
- `low_fps_safe_mode_changed` (`bool`): one-frame pulse when safe mode toggles.

When `low_fps_safe_mode_active` is true, alternate-limb carryover is suppressed so handoff does not batch-add reps.

Dynamic joint recalibration is limited to **`recalibration_attempts_per_entry`** switches per safe-mode entry (default `1` in `[low_fps_safe_mode]`). Each time safe mode turns on, that budget is restored; completing a joint switch while safe mode is active consumes it. Further switches wait until safe mode exits or another entry.
