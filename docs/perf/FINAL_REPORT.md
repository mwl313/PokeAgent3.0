# Performance track final report (v1.1 plan, Phases 0-6)

> **v3 note (2026-10-09):** the single-GPU PPO numbers below were measured under
> the single learner-seat collection contract. The `configs/train.yaml` contract
> (both current-policy sides) is now implemented: the controlled A/B shows
> 34.24 → 19.37 all-in games/s with 13.42 → 26.87 learner rows per match, i.e.
> learner rows/s rises 460 → 520 while games/s is not directly comparable across
> contracts. See `V3_NUMERIC_PARITY.md` and `V3_VRAM_BATCH_SWEEP.md`.

## 1. Scope, source, hardware

* Branch `optimization/pa3-realpolicy-throughput`; commits `c55b227` (plan),
  `6e45ade` (Phases 0-4) and the Phase 5/6 commit that adds this report.
* Dataset `mb-mc-v3-userteam-all-train`, manifest
  `f0509c08544c89a0b0c043218e33795c91af7af2f78e2d6fb83bae69f3babaef`, 1,137
  teams, all train, uniform sampling; Charizard base ability confirmed Blaze by
  the user (2026-10-08).
* Hardware verified read-only: GPU0 `05:00.0` 175 W, GPU1 `84:00.0` 150 W,
  driver 580.178.04, CUDA 12.8.2, torch 2.14.0+cu126 (`sm_70`), 62 GiB RAM.
  No power/driver/service/package change; both GPUs were idle at each run.

## 2. Optimization list and measured deltas

Same script, same seeds, same team set, GPU0, 1,024 envs, 16 workers,
2,048 games x 3 repeats (`runs/perf/*.json`):

| Stage | games/s | ×baseline |
|---|---:|---:|
| Baseline | 15.07 | 1.00 |
| Fixed observation batch (Phase 1) | 23.97 | 1.59 |
| + FP16 autocast (Phase 3.4) | 28.10 | 1.86 |
| + `inference_mode` (Phase 3.3) | 31.91 | 2.12 |
| + packed candidate wire (Phase 2) | 50.61 | 3.36 |
| + packed prefix walk (Phase 2) | 54.57 | 3.62 |
| + packed rollout rows (Phase 4 first cut) | **103.44** | **6.86** |

Per-stage median variation stayed under ~8% (min values in
`docs/perf/ULTIMATE_FRONTIER_REPORT.md`), so the adopted changes exceed run
noise. FP32 control with the same stack: 78.57 games/s.

## 3. Actor-only and all-in throughput

| Metric | Value |
|---|---:|
| Single GPU, real-policy actor-only (2,048 games x3) | 103.44 games/s |
| Single GPU, real-policy actor-only (10,240 games) | 101.44 games/s |
| Single GPU, full PPO all-in (10,240 games, 4 epochs, typed learner build) | 20.87 committed games/s |
| Single GPU, full PPO all-in with the packed from_rows learner build | 26.43 committed games/s |
| Single GPU, full PPO all-in, v2 correctness + 1024-row microbatch (3 repeats) | 32.80 committed games/s |
| Single GPU, full PPO all-in, v2 + microbatch 1024 + streaming, 10,240 matches | **33.37 committed games/s** (peak RSS 7.34 GiB) |
| Dual GPU, real-policy actor-only (2x2,048 games) | 162.73 games/s wall-aligned |
| PPO learner phase alone (10,240-game iteration) | 389.7 s |
| Engine-only reference (Rust, single thread) | 2,585 games/s (not AI) |
| GPU model forward reference | 101.44 games/s of decisions inside the actor |

Resource telemetry: actor RSS 2.4-2.6 GiB per 2,048 games; all-in peak 18.0 GiB
after the packed learner build (25.2 GiB before it, above the 24 GiB rollout
budget - the learner stacks the whole iteration);
GPU reserved 2.80 GiB FP16 vs 4.32 GiB FP32 per card; 0 operational errors,
0 illegal actions, natural matches = wins + losses + draws in every run.

## 4. Correctness tests

* Fixed vs per-view observation parity on real state, all five ragged sections
  (`docs/perf/OBSERVATION_CONTRACT.md`).
* Packed candidate wire vs legacy tuple oracle: 879 requests / 7,146 candidates
  / every prefix / 0 mismatches (`docs/perf/CANDIDATE_WIRE.md`).
* Sampled vs recomputed log-probability 0.0 on the optimized path; FP16 gate
  declared 1e-3, FP32 gate 1e-4.
* PPO: 4 epochs, KL 0.0052, ratio 0.977, grad norm 0.759, no NaN/Inf, no early
  stop, committed matches = actual natural completions (overshoot retained).
* Hidden-information, prefix-mask and legality regressions remain green in the
  Rust/Python suites; `readiness_check` 10/16 PASS (FAIL 2, 3, 5, 8, 15, 16).

## 5. Memory growth and bounded smoke

Actor memory grows ~1.2 GiB per 1,000 matches (compact uint16/float16 rows), so
a 131k-decision iteration fits the 24 GiB rollout budget at collection time.
The all-in peak of 25.2 GiB comes from the learner materializing the full
iteration at once; streaming minibatches is the next change (Phase 4 proper).
A 10,240-match bounded smoke completed with 0 errors and a valid checkpoint
path in the earlier session; this session's 10,240-match run is the all-in
measurement above.

## 6. Remaining bottlenecks and next steps

1. Learner minibatch materialization (75% of all-in, 18.0 GiB peak after the
   packed `from_rows` build): stream per-minibatch materialization from the
   compact store instead of stacking the whole iteration, then reuse the packed
   candidate columns end to end.
2. Serial per-round actor pipeline: Phase 8 double-buffering across independent
   cohorts and rolling slot refill.
3. Rust observation packing (17% of actor): f16 float block / columnar wire.
4. Dual-GPU DDP all-in measurement and `no_sync` accumulation (Phase 6/10).
5. Full-scope engine readiness (criteria 2, 3, 5, 8, 15, 16) remains an
   independent blocker; 100M-match training is still not authorized.

Reproduce: `PYTHONPATH=engine/python:. .venv/bin/python
scripts/bench_pa3_end_to_end.py --games 2048 --envs 1024 --workers 16
--repeats 3 --mode collect --observations fixed --precision fp16
--inference-mode --candidate-wire packed --device cuda:0
--report runs/perf/fastrow-fp16-im.json` (see
`runs/perf/run_manifest.json` for the recorded environment).
