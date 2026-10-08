# V2 frontier report (v2.0 plan, P0 + M0 complete, P1/P3/P4 open)

> **v3 accounting note (2026-10-09):** every `games/s` number in this document
> was measured under the *single learner seat* collection contract. The
> `configs/train.yaml` contract (`collect_both_sides_when_current_self_play:
> true`) is now implemented, which doubles the learner rows per match; see
> `V3_NUMERIC_PARITY.md` for the controlled A/B (34.24 → 19.37 games/s with
> 13.42 → 26.87 rows/match, i.e. learner rows/s rises from 460 to 520).
> Games/s is only comparable within one collection contract.

Base `b72f5f4` → branch `optimization/pa3-realpolicy-throughput`. This report
covers the v2.0 plan's correctness phase (P0) and learner breakdown (M0), plus
the measurement-honesty fixes they required. Phase 1 (streaming/columnar), P3
(DDP), P4 (overlap) and P5 (Rust observation) remain open and are listed with
the measured evidence that should drive them.

## What changed

| item | before | after |
|---|---|---|
| PPO epoch/iteration metrics | means of the last microbatch only | row-weighted sums/denominators over every microbatch, one D2H per epoch |
| Gradient accumulation | `sample_weight/len(microbatches)` (biased on padded/uneven actor rows) | exact per-minibatch actor/value denominators (`exact_row_weighted_accumulation`, default on; legacy kept as the A/B flag) |
| Optimizer step accounting | +1 after every `scaler.step()` | +1 only when the scaler actually applied the step, `optimizer_steps_skipped` recorded |
| `grad_norm` | last step only | last step plus `grad_norm_max` over the iteration |
| Checkpoint in the benchmark | `--checkpoint` was unused | real temp→fsync→rename write inside the measured window (0.46–0.74 s, 105 MB) |
| All-in label | one number | `bounded_ppo_games_per_s` and `all_in_committed_games_per_s` (+`report_includes`) |
| Stage share | stage median ÷ another repeat's wall | median of each repeat's own stage/collect ratio |
| Provenance | `git_dirty` flag | `git_diff_hash` for dirty trees |
| Recompute gate | first 1,024 rows | stratified sample over request kind × branch count × actor flag, per-stratum report |
| Per-microbatch device syncs | 2,112 GPU `.sum()` syncs per 2k iteration | CPU padding filter + one minibatch H2D + device-side micro slicing |

## Measured impact

* 2,048-match all-in with checkpoint inside the window: 25.3 committed games/s
  (collect 21.7 s, PPO 58.4 s, checkpoint 0.74 s); PPO health unchanged
  (KL 0.0045, epoch KLs 0.0057/0.0039/0.0037/0.0048, 28 steps, 0 skipped,
  gradient-norm max 10.0, recompute max 1.0e-4 within the declared fp16 gate).
* **P2.3 microbatch selection (new):** with the exact row-weighted accumulation
  the global 4,096 objective and the 28 optimizer steps are identical for any
  micro split. Measured 2,048-match all-in: 128 → 20.46, 256 → 24.66,
  512 → 29.14, **1024 → 31.57 games/s**, with matching policy statistics
  (ratio 1.0004, clip 0.046, entropy 0.9987, value 0.2759). Three-repeat
  confirmation at 1024: **all-in median 32.80 games/s** (min 31.68), PPO
  41.8–42.8 s, checkpoint 0.54–0.57 s inside the window, 28 steps, 0 skipped,
  GPU reserved 9.27 GiB of the 28 GiB budget. Recorded as a launcher option;
  the pinned default stays 256 and the DDP split keeps its own validation.
* **P1.3 streaming minibatch (new, memory win):** `prepare_streaming` /
  `update_streaming` expand one global minibatch at a time from the compact
  store. Speed is neutral (2k: PPO 42.9 → 43.3 s) but the 10,240-match peak RSS
  falls from **18.0 GiB to 7.34 GiB**, and the full 10,240-match all-in is
  **33.37 games/s** (collect 100.2 s, PPO 206.0 s, checkpoint 0.63 s,
  136,259 rows, 136 steps, 0 skipped, KL 0.0051, recompute 6.8e-5, 0 errors).
  Opt-in via `--streaming-minibatch` until the two-rank path is validated.
* 10,240-match scale reference (from the v1.2 commit): collect 96.8 s,
  PPO 290.7 s, 26.43 games/s, 18.0 GiB peak RSS. The v2 correctness changes are
  metric/accumulation corrections, not speed claims; the PPO wall is
  unchanged within noise and the reported KL semantics are now correct.
* Learner breakdown (2k iteration): backward 18.2 s, forward 15.4 s,
  minibatch selection 5.4 s, GAE 1.26 s, observation stack 1.63 s, candidate
  build 0.58 s, H2D 2.5 s, optimizer 1.0 s, unaccounted ~12 s.
* CUDA top kernels: embedding backward scatter 1.45 s, SDPA math 1.25 s,
  pageable H2D 1.00 s, elementwise 0.91 s, FFN GEMM 0.86 s.
* GPU0 during the learner phase: 69–80% SM, 150–175 W, 3.3 GiB VRAM; GPU1 idle.
* Storage: `read_bytes = 0` during training, swap 0 B, 0 major faults, iowait
  0% → **not a bottleneck** (see `V2_IO_NUMA_REPORT.md`).

## Next bottlenecks (priority order, with the evidence)

1. **P1.3 streaming minibatch materialization** — 5.4 s selection + 1.6 s
   observation stack + ~12 s unaccounted and an 18 GiB peak all come from
   materializing the whole iteration before slicing. Select rows from the
   compact store per minibatch.
2. **P1.3 candidate dtype** — `from_rows` builds `[B,4,64,6] int64` (12 KB/row)
   from a `u8` wire; keep the packed bytes and cast once per minibatch (also
   removes part of the 1.0 s pageable H2D).
3. **P2.3 microbatch size** — 2,112 forward calls of 256 rows; with exact
   row-weighted accumulation the global 4,096 objective is preserved for any
   micro split, so 512/1024 are now safe A/B candidates.
4. **P3 dual-GPU DDP all-in** — the actor-only dual number is 162.73 games/s,
   but no DDP learner exists yet; `no_sync` at the global-minibatch boundary
   with weighted gradients is the documented next step.
5. **P4 actor overlap / rolling slots** — the actor pipeline is serial
   (observe → H2D → encode → sample → step) and worker-count-insensitive; the
   gain must come from overlapping independent cohorts.
6. **P5 Rust observation packing** — 41,954 B/view fixed wire is 17% of the
   actor wall; encoder caching and columnar packing are the measured targets.

## Guardrails that still hold

`readiness_check` is 10/16 PASS (FAIL 2, 3, 5, 8, 15, 16) and independent of this
track. The 100M-match run was not started. No rule, model, observation-schema or
PPO-hyperparameter change was made; the corrections preserve the documented
objective and are covered by new regression tests
(`tests/agent/test_update_statistics.py`, 6 tests; 56 agent tests green in
total).
