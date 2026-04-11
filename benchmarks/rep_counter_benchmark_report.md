# Rep Counter Replay Benchmark

Command:

`python3 benchmarks/rep_counter_replay_benchmark.py`

Latest run (local development machine):

- `frames=288`
- `p50_ms=5.9214`
- `p95_ms=10.6430`
- `p99_ms=10.9896`
- `mean_ms=5.9672`

Notes:

- This benchmark replays a deterministic landmark sequence through `RepCounterSession.step_landmarks`.
- It is intended for relative regressions/improvements across commits, not absolute production SLAs.
