# Phase 0 baseline — real-policy PA3-8M throughput (2026-10-08)

Instrument: `scripts/bench_pa3_end_to_end.py` (CUDA-event timed, explicit
synchronization at the wall boundaries, stage accounting, run manifest with
git SHA / dataset manifest SHA / GPU PCI IDs / driver / power limit / command).

Hardware and stack verified before measuring: GPU0 `05:00.0` 175 W and GPU1
`84:00.0` 150 W, driver 580.178.04, CUDA toolkit 12.8.2 (V12.8.93),
torch 2.14.0+cu126 (`sm_70` present), Python 3.12.3, both GPUs idle. No driver,
power, service or package change.

## Baseline (legacy per-view observation, per-row GPU readback, fp32)

`--games 2048 --envs 1024 --workers 16 --repeats 3 --mode collect --observations perview --precision fp32 --device cuda:0`
(`runs/perf/baseline-collect.json`, git `c55b227` + Phase 0 instrumentation).

| Repeat | natural games | actor games/s | decisions/s | wall s | host RSS GiB | GPU reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,048 | 15.07 | 407.2 | 135.9 | 2.3 | 4.32 |
| 1 | 2,048 | 15.09 | 398.9 | 135.7 | 2.3 | 4.32 |
| 2 | 2,048 | 14.95 | 400.4 | 137.0 | 2.3 | 4.32 |

Median **15.07 natural games/s**, 400 decisions/s, 0 operational errors, 0
aborts. The historical uncapped-plan reference is 16.4 games/s over 10,240
games; the small difference is run length (10 cohorts vs 2) and start-up, not a
regression.

### Where the wall time goes (median of 3, seconds per 2,048 games)

| stage | seconds | share |
|---|---:|---:|
| PA3-8M forward + sampling (excl. table build) | 42.4 | 31% |
| unaccounted Python (typed request/branch construction, loop, list building) | 33.0–35.6 | 25% |
| observation parse + typed adapter (per-view) | 30.0 | 22% |
| rollout buffer record (row objects, observation copy) | 20.1–22.4 | 16% |
| native observation packing | 2.7–3.4 | 2.2% |
| H2D copy | 2.2–2.6 | 1.8% |
| GPU→CPU readback (per-row `.item()`) | 2.2 | 1.7% |
| legal candidate generation | 0.4–0.5 | 0.3% |
| Rust `step_batch` | 0.4 | 0.3% |
| reset | 0.03 | <0.1% |

Reading: with a real policy, the cost is dominated by Python-side data
handling (parse/adapter + buffer record + unaccounted) and by the GPU forward;
the Rust engine itself is ~0.3%. That matches the plan's hypothesis and sets
the phase order: fixed observation batch (Phase 1), packed candidates and typed
object removal (Phase 2), readback/inference (Phase 3), buffer (Phase 4).

## Observation contract audit (Phase 0 item 6)

`scripts/audit_observation_contract.py` → `docs/perf/OBSERVATION_CONTRACT.md`.

* `observe_fixed_batch` (fixed stride + ragged sidecar) and
  `observe_encoded_batch` (per-view blobs) are **field-identical on real engine
  state**, including all five ragged sections — the batch-first path is a pure
  transport change.
* Fixed stride: 41,954 bytes/view.
* The typed adapter forwards the nine fixed fields the embedding consumes. The
  ragged sections (effects, repertoire, types, base moves, move effects) are on
  the wire but **not consumed by the model yet**; they are recorded as a
  completeness blocker, never dropped from the wire.
* Known category-id range 0–1,460 against the placeholder vocabulary 8,192
  (slack 6,731); the final Dex vocabulary export is still required before a real
  training run.
* A first batch-path bug was found and fixed here: a single-request batch is
  "C contiguous" to numpy (the stride of a size-1 dimension is irrelevant) but
  not to torch, so the adapter now tests the stride condition torch enforces.

## Committed-match clock (Phase 0 item 7)

`scripts/run_ppo_smoke.py` used to advance the LR clock with the *requested*
match target while the collector kept cohort overshoot. It now passes the
actual natural completions (`collector.stats.games`), records both numbers and
asserts `committed_matches == natural completions`.

## Readiness (unchanged, independent of performance)

`readiness_check`: 10/16 PASS, FAIL 2, 3, 5, 8, 15, 16 (exit 1). The 100M-match
run remains unauthorized and is not started by any benchmark here.
