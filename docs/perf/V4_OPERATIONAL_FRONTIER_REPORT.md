# V4 operational frontier report (F1, in progress)

Branch `optimization/pa3-realpolicy-throughput`. This is the operational
summary for the clean-HEAD confirmation panels. Numbers are filled in from the
raw JSON in `runs/perf/v4/` on the day of measurement; nothing is extrapolated.

## 1. Gates completed before this panel

| gate | verdict | evidence |
|---|---|---|
| C0 regularizer gradients | PASS | `V4_CORRECTNESS_FORMULA_AUDIT.md` §1–3 |
| C1 single-GPU oracle | PASS (documented tolerances) | same doc, C1 sections |
| D0/D1/M1 distributed | PASS | `V4_DISTRIBUTED_MATH_AND_PROTOCOL.md` |
| F0 bottleneck atlas | measured | `V4_FULLSTACK_BOTTLENECK_ATLAS.md` |

## 2. Clean-HEAD panels

### 2.1 Single GPU, 2,048 games, 3 repeats

Command:

```
PYTHONPATH=engine/python:. .venv/bin/python scripts/bench_pa3_end_to_end.py \
  --games 2048 --envs 1024 --workers 16 --repeats 3 --mode full \
  --observations fixed --candidate-wire packed --precision fp16 \
  --inference-mode --device cuda:0 --microbatch 1024 --streaming-minibatch \
  --report runs/perf/v4/final_single_2k.json \
  --checkpoint runs/perf/v4/final_single_checkpoint.pt --tag final-clean-single
```

### 2.2 Dual GPU, 1,024 games/rank (2,048 total), 3 repeats per executor

```
scripts/run_ddp_ppo.py --games 1024 --envs 1024 --workers 16 \
  --microbatch 256 --minibatch 4096 [--executor manual] \
  --report runs/perf/v4/final_<executor>_1024r_run<N>.json
```

### 2.3 Results

Code state: HEAD `fbf878e` (`v4 D2: manual executor switch in the two-rank
launcher`). The A0 run manifest records `git_sha =
fbf878e124f35e9168aa0bf60b2662fa6f6b992e`, `git_branch =
optimization/pa3-realpolicy-throughput`, `git_dirty = false`; the dual panels
ran at the same HEAD.
Every run: seed 20261009 (per-rank `seed + rank`), both-seat contract, fixed
observations, packed candidate wire, fp16 autocast + GradScaler, 0 operational
errors, 0 skipped optimizer steps.

| panel | runs (games/s) | median | mean | all-in wall | total games |
|---|---|---:|---:|---:|---:|
| single GPU, 2,048 games | 19.44 / 20.53 / 20.96 | 20.53 | 20.31 | 97.7–105.3 s | 2,048 |
| dual DDP, 1,024 games/rank | 24.67 / 25.92 / 24.67 | 24.67 | 25.09 | 79–83 s | 2,048 |
| dual manual SUM, 1,024/rank | 24.67 / 25.60 / 25.60 | 25.60 | 25.29 | 80–83 s | 2,048 |

All six dual runs: 27,330 + 27,279 rows, 56 optimizer steps, 56 synchronized +
392 no-sync micro steps, model **and** optimizer digests byte-identical across
ranks (`digests_equal: true`), 12.6–13.0 s collection and 58.5–61.5 s update
per rank.

Equal-total comparison (2,048 natural games, same rows and steps):

* median: 20.53 → 24.67 games/s (**+20.2 %**);
* mean: 20.31 → 25.09 games/s (**+23.5 %**).

Executor decision (plan §7.2): DDP and manual are **statistically
indistinguishable** — manual's mean is 0.8 % higher while DDP's median is
3.7 % higher, both inside the ±4 % repeat spread. The plan's rule for an
unclear winner is "prefer the simpler and more stable implementation", so
**DDP stays the default executor** (standard reducer overlap, no 35 MB flat
buffer bookkeeping, tightest fp32 parity: weight Δ = 1 ulp, grad-norm Δ = 0).
The manual executor remains available via `--executor manual` and is
parity-verified.

## 3. Checkpoint / resume status

* Single GPU: the checkpoint write is inside the measured window
  (0.46–0.56 s) in every repeat; C1 proved save → reload → identical next
  update (weight Δ 9.3e-10, moments 1.5e-11).
* Dual GPU: rank 0 writes a checkpoint carrying `global_committed_matches` and
  rank metadata. A full dual resume test (rank-local RNG/rollout cursors) is
  **not yet done** and is listed as remaining work.

## 4. Honesty and rejection record

* No speed claim is made from pre-C0 numbers: the corrected entropy/KL
  gradient changes both values and compute, so the v3 19.365 games/s baseline
  is not comparable and is not used.
* Neutral experiments (epoch-gather, observation fill loop, foreach/fused
  Adam) are recorded in `V4_FULLSTACK_BOTTLENECK_ATLAS.md` §6 and were not
  promoted; the repeat noise floor for these panels is ±2–4 %.
* The learner update is 95.4 % GPU-busy (measured), so CPU-side learner
  micro-optimizations are bounded by the ~5 % idle window; the largest
  remaining envelope is collection/learner overlap (collection is 23 % of
  all-in).
* Remaining tracked work: collection/learner overlap (P3/P4), columnar
  materialization (P5.1), dual checkpoint/resume, per-rank NUMA/NCCL
  telemetry, and the global-permutation minibatch plan (only needed for
  bit-parity with a single-process global shuffle).

## 5. Readiness separation (unchanged)

* Full-engine readiness remains **10/16** (fail 2, 3, 5, 8, 15, 16). None of
  the throughput work above changes the rules engine, the 1,137-team pool,
  hidden-information safety or the legal-move oracle.
* The 100M-game training run was **not** started; it still requires 16/16
  readiness plus explicit user authorization.
* All runs in this document are bounded benchmarks (≤2,048 natural games per
  panel) on the frozen training pool.
