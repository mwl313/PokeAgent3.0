# PA3-8M real-policy throughput frontier (v1.1 plan execution)

Branch `optimization/pa3-realpolicy-throughput`, base `9cbaa1f` (v1.1 plan
commit `c55b227`, Phase 0-4 commit `6e45ade` and later Phase 5/6 commits).
Hardware: 2x Xeon E5-2673 v4 (40 cores / 80 threads, NUMA0/1), 62 GiB RAM,
2x V100-PCIE-32GB (GPU0 `05:00.0` 175 W, GPU1 `84:00.0` 150 W), driver
580.178.04, CUDA toolkit 12.8.2, torch 2.14.0+cu126, Python 3.12.3. No driver,
power, service or package change; `dsh-web`/`llama-swap` untouched.

Instrument: `scripts/bench_pa3_end_to_end.py` (CUDA-event timed, explicit
synchronization, stage accounting, run manifest with git/dataset SHA, GPU
inventory and the exact command). Every number below is `real_policy_collect`
unless labelled otherwise: PA3-8M sampled on the GPU from the frozen
1,137-team pool with prefix-dependent masks, 0 operational errors in every run.

## Stage-by-stage measured deltas (GPU0, 1,024 envs, 16 workers, 2,048 games/repeat, 3 repeats)

| Stage | games/s | min | ×base | decisions/s | wall s | GPU res. GiB | correctness |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline (per-view obs, per-row sync, fp32, `no_grad`) | 15.07 | 14.95 | 1.00 | 400 | 135.9 | 4.32 | record |
| Phase 1 fixed observation batch | 23.97 | 23.47 | 1.59 | 642 | 87.3 | 4.32 | pass |
| Phase 3.4 + FP16 autocast | 28.10 | 26.84 | 1.86 | 750 | 76.3 | 2.81 | pass |
| Phase 3.3 + `inference_mode` | 31.91 | 31.57 | 2.12 | 846 | 64.9 | 2.80 | pass |
| Phase 2 + packed candidate wire | 50.61 | 47.10 | 3.36 | 1,356 | 43.5 | 2.80 | pass |
| Phase 2 + packed prefix walk | 54.57 | 50.40 | 3.62 | 1,463 | 40.6 | 2.80 | pass |
| Phase 4 + packed rollout rows | **103.44** | 95.59 | **6.86** | 2,729 | 21.4 | 2.80 | pass |
| fp32 control with the same stack | 78.57 | 74.26 | 5.21 | 2,107 | 27.6 | 4.32 | pass |

Stage accounting per 2,048 games (median seconds): baseline parse+adapter 30.0,
buffer record 21.8, model 42.4, unaccounted 34.4, native observation 2.8;
optimized model 5.25 (24.5%), buffer record 4.31 (20.1%), native observation
3.65 (17.0%), unaccounted 2.34 (10.9%), H2D 2.18 (10.2%), parse 1.09 (5.1%),
Rust step 0.34 (1.6%), candidates+table 0.48 (2.2%), readback 0.09.

## Long run and true all-in cost

10,240 natural matches (10,000 target + 240 retained cohort overshoot),
272,551 decisions, 0 errors:

* actor-only collection **100.9 s -> 101.44 games/s** (2,699 decisions/s);
* PPO (GAE + 4 epochs + optimizer + checkpoint path) **389.7 s**;
* **all-in committed throughput 20.87 games/s** over 490.6 s.

PPO health on that iteration: 4 epochs, 136 optimizer steps, 136,259 rows
(128,115 actor rows), approx KL 0.0052 (per epoch 0.00695/0.00444/0.00460/0.00484),
ratio mean 0.977, clip 0.052 (earlier measurement), grad norm 0.759, no NaN/Inf,
no early stop. Sampled vs recomputed log-probability differs by **0.0**.

The learner phase is now 79% of the all-in wall. `RolloutBuffer.to_batch`
materializes the whole iteration (all observations, candidate tables and typed
`ActionRef` objects) before minibatching, which is also what drives the 25.2 GiB
peak host RSS (over the plan's 24 GiB rollout budget). Streaming the minibatch
build is the top next optimization.

## Dual-GPU real-policy actor (Phase 6)

`scripts/run_actor_pair_real.py`, one process per GPU/NUMA node, each with its
own native engine group (1,024 envs / 16 workers), same frozen policy:

| Rank | GPU / NUMA | games | games/s | op errors | RSS |
|---:|---|---:|---:|---:|---:|
| 0 | cuda:0 / NUMA0 | 2,048 | 94.15 | 0 | 2.42 GiB |
| 1 | cuda:1 / NUMA1 | 2,048 | 100.26 | 0 | 2.40 GiB |
| both | wall-aligned | 4,096 | **162.73** | 0 | |

Scaling vs one rank 1.73x (sum of per-rank rates 194.4; the gap is process start
stagger and shared memory bandwidth). GPU1 at 150 W is *not* the slower rank
here. No DDP learner is part of this measurement; the dual-GPU **all-in PPO**
number is still open.

## CPU worker / environment sweep (Phase 5)

Same workload, single rank, 2 repeats:

| config | games/s | decisions/s |
|---|---:|---:|
| workers 8 / 1,024 envs | 104.35 | 2,797 |
| workers 12 / 1,024 envs | 110.92 | 2,973 |
| workers 16 / 1,024 envs | 103.44 | 2,729 |
| workers 20 / 1,024 envs | 106.02 | 2,842 |
| workers 16 / 512 envs | 98.43 | 2,646 |
| workers 16 / 2,048 envs | 101.63 | 2,693 |

Throughput is flat across 8-20 workers and 512-2,048 environments: the pipeline
is no longer limited by Rayon worker parallelism or engine CPU work (Rust
`step_batch` is 0.34 s per 2,048 games, 1.6%). It is limited by the serial
per-round critical path (observe -> H2D -> encode -> sample -> step) plus Python
orchestration, so the next gains must come from overlap/pipelining (Phase 8),
not from more workers.

## Correctness evidence

* `scripts/audit_observation_contract.py`: `observe_fixed_batch` and
  `observe_encoded_batch` are field-identical on real engine state, including
  all five ragged sections (`docs/perf/OBSERVATION_CONTRACT.md`).
* `scripts/audit_candidate_wire.py`: packed wire == legacy tuple oracle for
  every prefix (879 requests / 7,146 candidates / 0 mismatches,
  `docs/perf/CANDIDATE_WIRE.md`).
* Sampled vs recomputed joint log-probability: exactly 0.0 on the optimized
  path (the learner consumes the same stored tables/observations); the declared
  mixed-precision gate is 1e-3 and the FP32 gate stays 1e-4.
* Two real bugs were found by these gates: the single-request stride edge case
  in the observation adapter and a group-index vs round-index row pairing in the
  packed rollout path.
* `readiness_check` is unchanged at 10/16 PASS (FAIL 2, 3, 5, 8, 15, 16) and is
  independent of this performance track. The 100M-match run was not started.

## Adopted / rejected experiments

Adopted: fixed observation batch, packed candidate wire + packed prefix walk,
bulk per-level D2H, `inference_mode`, FP16 autocast for the actor (FP32
probabilities/logprobs), packed rollout rows.

Rejected or not yet adopted: FP16 for the whole learner math (probabilities stay
FP32), CUDA Graph capture (not attempted; dynamic prefix shapes), alternate
inference backend (not attempted), model token/layer reduction (out of scope per
plan section 10.5.6), worker counts away from 16 (no measurable gain).

## Frontier reading (no artificial ceiling)

Actor-only already exceeds the 1,000-games/s-level target's 100x working
frontier at 103 games/s single-GPU and 163 games/s dual-GPU; the honest all-in
number is 20.9 games/s because the PPO learner is the current bottleneck. In
order of measured contribution, the remaining barriers are:

1. **Learner minibatch materialization** (389.7 s of 490.6 s all-in, 25.2 GiB
   peak): stream minibatches from the compact store instead of stacking the
   whole iteration; reuse the packed candidate columns instead of rebuilding
   typed `ActionRef` objects per row.
2. **Per-round serial pipeline** (model 24.5%, buffer 20.1%, observation 17.0%,
   H2D 10.2% at actor scale): Phase 8 double-buffering between independent
   environment cohorts and rolling slot refill.
3. **Rust observation packing** (17%): f16 float block / columnar packing would
   halve the 41,954-byte per-view wire and the H2D copy.
4. **DDP**: the dual-GPU all-in number is unmeasured; NCCL `no_sync` at the
   global-minibatch boundary is the documented next step.

Targets beyond 1,000 games/s remain hypotheses, not claims; nothing here stops
at a number. The measured ceiling on this hardware for the current design is
set by the learner materialization and the serial round pipeline above.
