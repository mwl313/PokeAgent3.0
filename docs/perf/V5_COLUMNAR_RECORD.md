# V5b — columnar collector record path (T1)

Code base `2c54dfb` + this task's changes. New `ColumnarObservationStore`
(one growing array per observation field, SoA) behind
`NativeCollectorConfig.columnar_observation_store` / `run_ddp_ppo.py
--columnar-store`; the collector appends the compact decision-batch row
directly (`RolloutBuffer.record_packed(..., observation_index=...)` →
`add_compact_at`) instead of building a per-row dict of nine one-row arrays.

## 1. Gates — ALL PASS

| gate | result | evidence |
|---|---|---|
| row count identical | PASS | 13,679 == 13,679 (`runs/perf/v5b/columnar_gate.json`) |
| full-row SHA256 identical (old vs columnar, same seed/input) | PASS | equal `row_sha_equal: true` |
| collector stats/model manifest identical | PASS | games/decisions/wins/losses/reward_sum + model SHA equal |
| store round-trip (`get`/`stacked`/`stacked_indices`) | PASS | 3 sampled rows match the batch gather |
| recompute gate (dual runs) | PASS | max abs diff 5.78e-06 / 1.55e-05 vs 1e-3 tol |
| digest parity at fixed step | PASS | model+optimizer digests equal across ranks, 56 steps, 0 skipped |
| tests | PASS | 86 passed (agent + integration) |

Exact gate command:

```
PYTHONPATH=engine/python:. .venv/bin/python scripts/v5b_columnar_gate.py \
  --games 512 --envs 256 --workers 8 --out runs/perf/v5b/columnar_gate.json
```

## 2. A/B (3× dual, 2,048 games total, micro 1024, checkpoint + recompute gate)

```
# baseline (parent commit 2c54dfb, same flags; measured earlier in the same session)
scripts/run_ddp_ppo.py --games 1024 --envs 1024 --workers 16 --minibatch 4096 \
  --checkpoint runs/perf/v5/a1024_run<N>_ckpt.pt --report runs/perf/v5/a1024_run<N>.json
# columnar
scripts/run_ddp_ppo.py ... --columnar-store \
  --checkpoint runs/perf/v5b/col_run<N>_ckpt.pt --report runs/perf/v5b/col_run<N>.json
```

| arm | games/s runs | median | collect median | update median | materialization block |
|---|---|---:|---:|---:|---:|
| baseline | 32.502 / 31.994 / 32.502 | 32.502 | 12.44 s | 42.22 s | 7.60 s |
| columnar | 33.026 / 31.994 / 33.026 | **33.026** | 12.69 s | 41.38 s | 6.98 s |

Median deltas: all-in **+1.61 %**, collect **+2.02 % (regressed)**, update
**−1.97 %**, materialization **−8.2 %**.

## 3. Decision — NOT PROMOTED

All gates pass, but the end-to-end effect (+1.6 %) is inside the declared ±2 %
noise floor and the collection phase *regressed*: the SoA append moves work
from lazy materialization into the record loop (per-field numpy writes) faster
than it removes it. Per the task rule the old path stays the default; the
columnar store remains available behind `--columnar-store` for the follow-up
that keeps the materialization win without the record regression (e.g. batched
block appends instead of per-row writes).

The atlas's "record ≈ 35 % of collection" share did **not** convert into wall
savings on this path; that estimate is now corrected by this measurement.

nvidia-smi (columnar runs): util max 100 %, VRAM 9,750 MiB, power ≤222 W
(instantaneous; caps unchanged 175/150 W), temp ≤61 °C.
