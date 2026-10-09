# miniDC training-pool speedtest (branch `training-pool/minidc-speedtest-ready`)

Development handoff document. It is not a readiness claim, and nothing here
authorizes training. The only acceptable final readiness signal remains
`readiness_check` exiting 0 across all 16 criteria **on the miniDC**.

## Purpose

The frozen training pool is now fully usable by the native engine:
every member's moves, ability and item are implemented, every team completes a
natural battle in the engine-only probe, and there are zero operational aborts.
This branch exists so the miniDC can measure the *real* training-pool speed
(engine-only, PyO3 batch, observation, actor path) and, only if those pass, run
a short PPO smoke test. It does not attempt full M-C engine coverage.

The pool measured below is `mb-mc-v3-userteam-all-train` (1,137 teams): the
previous 1,136-team `mb-mc-v2-all-train` pool plus the single user-approved
Poképaste of 2026-10-08. `docs/TEAM_POOL.md` records the source, its legality
receipt and the one documented Mega-form ability normalization.

## Exact branch scope

* Branch: `training-pool/minidc-speedtest-ready`
* Base: `origin/mac/long-horizon-engine-tail` @ `54d0ff1`
* Commits on this branch:
  * `911ee87` Port Illusion with per-viewer observed identity and reference
    identity fixtures
  * `a45117a` Report engine-only training-pool throughput and repeat rounds in
    the pool probe
  * the documentation commit that adds this file
* Out of scope: full M-C move/ability coverage, dynamic-closure completion,
  throughput claims, and any training run.

## Why this is not full readiness

`training-pool readiness` is a narrower, measurable claim: the frozen pool can
be used for training rollouts without hitting an unsupported mechanic. `full
engine readiness` additionally requires every legal move/ability, the dynamic
call closure and the full-scope corpus gates. This branch satisfies the former
and explicitly does **not** claim the latter.

## Definition of training-pool readiness

1. All 1,136 frozen teams are statically complete (every member move, ability
   and item executable).
2. All 1,136 frozen teams produce natural battle completions under the
   deterministic engine-only probe.
3. Zero operational aborts in the pool probe.
4. Illusion is implemented with per-viewer observed identity, not faked.
5. No silent fallbacks, no operational-abort-as-draw.
6. Player-safe observation, action masks, PyO3 batch path and 2,048-environment
   execution remain valid.

## Latest measured state (Mac mini, this branch)

* Moves: 486/515 executable, 29 blocked
* Abilities: 200/223 executable, 23 blocked
* Items: 166/166 executable, 0 blocked
* Dynamic closure: 33 callers, 11 blocked
* Corpus: 1,148 fixtures / 25,134 decision boundaries, 0 mismatches,
  1,148/1,148 re-verified against the pinned Showdown
* Training pool static: **1,136/1,136**
* Training pool natural: **1,136/1,136** (mean 10.5 turns, 1,136 decisive)
* Operational aborts: **0**
* Illusion: ported (Zoroark / Zoroark-Hisui), witness in
  `engine/tests/illusion.rs` over `engine/data/more_illusion.json`
* `readiness_check`: NOT READY — criteria 1, 4, 6, 7, 9–14 PASS;
  2, 3, 5, 8, 15, 16 FAIL (full-scope coverage and the full-coverage corpus)

## Mac verification commands and outputs

All commands run in this worktree on the Mac mini:

```
bash scripts/cargo.sh test --locked --release --no-fail-fast
    -> 22 test binaries, 0 failures
bash scripts/cargo.sh clippy --locked --all-targets -- -D warnings
    -> clean
bash scripts/cargo.sh clippy --locked --features python --all-targets -- -D warnings
    -> clean
node engine/tests/verify_turn_fixtures.mjs
    -> verified 1148/1148 fixtures / 25134 decision boundaries against the pinned reference
cargo run --release --example coverage_report
    -> moves 486/515 (29 blocked), abilities 200/223 (23 blocked), items 166/166,
       dynamic closure 33 callers / 11 blocked, moves_witnessed 481/486
cargo run --release --example pool_run_report
    -> training pool: 1136 teams
       trajectory completions: 1136 (100.0%)
       static mechanic completeness: 1136
       blocked in this trajectory probe: 0
       mean completed battle length: 10.5 turns
cargo run --release --example readiness_check
    -> TRAINING READINESS: NOT READY (criteria 2,3,5,8,15,16 fail)
```

Engine-only throughput on the Mac mini (development measurement only — not a
final performance claim; the miniDC must re-measure):

```
POOL_REPEAT=20 cargo run --release --example pool_run_report
    -> 22720 games over 20 pool round(s), 325360 decisions,
       ~78,000 decisions/sec, ~5,480 games/sec, wall ~4.1 s
```

## miniDC measurements (2026-10-08, this branch, dataset v3, 1,137 teams)

Host: 2x Xeon E5-2673 v4 (40 cores / 80 threads, 2 NUMA nodes), 62 GiB RAM,
2x V100-PCIE-32GB (175 W / 150 W), driver 580.178.04, system CUDA 12.8.2,
Python 3.12.3, torch 2.14.0+cu126 (`sm_70` present, both GPUs free during the
runs). Rust 1.90.0 from the project-local toolchain. No host package, driver,
service or power setting was changed.

### A. Engine-only training-pool benchmark

`POOL_REPEAT=20 ./target/release/examples/pool_run_report`, one warm-up round
plus three measured runs:

| Run | Games | Decisions | Decisions/s | Games/s | Wall | CPU | Max RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22,740 | 325,880 | 36,965 | 2,579.4 | 8.816 s | 100% of one core | 37.8 MB |
| 2 | 22,740 | 325,880 | 37,039 | 2,584.6 | 8.798 s | 100% | 37.8 MB |
| 3 | 22,740 | 325,880 | 38,536 | 2,689.1 | 8.456 s | 100% | 37.1 MB |

Training-pool state on the miniDC: 1,137 teams, 1,137 trajectory completions
(100.0%), 1,137 statically complete, 0 blocked, mean 10.5 turns, 1,137
decisive, 0 operational aborts.

The miniDC is *slower per engine thread* than the Mac mini reference
(~2,585 vs ~5,480 games/s). The probe is single-threaded, so this is per-core
Broadwell-vs-Apple performance, not a regression. Parallel throughput is what
the training path uses, and it is measured in B/D below.

### B. PyO3 batch benchmark (2x1,024 environments, 16 workers/rank)

`scripts/run_actor_pair.py --envs 1024 --games 4096 --workers 16`, numactl
`--cpunodebind/--membind` per rank, 8,192 natural games total, 0 operational
errors:

| Rank | NUMA | Games | Games/s | Transitions/s | Decisions/s | CPU cores | obs ms/round | step ms/round | mask ms/round | policy ms/round | reset ms/cohort | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 4,096 | 304.1 | 8,564 | 16,314 | 3.04 | 32.81 | 2.99 | 3.52 | 4.74 | 17.4 | 258 MB |
| 1 | 1 | 4,096 | 325.0 | 9,053 | 17,228 | 3.17 | 31.72 | 3.01 | 3.30 | 4.58 | 18.3 | 257 MB |
| combined | | 8,192 | **592.6** | | | | | | | | | |

### C. Observation encoding benchmark

`engine/python/bench_observation.py` (256-env fixture cohort, 10 rounds,
4,641 views, 199.6 MB payload, ~42.0 kB/view):

| Stage | Cost |
|---|---:|
| native `observe_encoded_batch` round trip | 80.40 ms/round |
| numpy decode + stack (per-view blobs) | 36.97 ms/round |
| fixed-batch zero-copy decode | 2.45 ms/round |
| torch tensor conversion (`.copy()` per field) | 212.53 ms/round |

The per-view payload is ~42 kB per decision. The fixed-stride batch payload
decodes about 15x faster than the per-view blobs, and the torch conversion is
~3x the native packing cost. A model-backed policy therefore pays a real
tensor-conversion bill that the random-policy actor path does not.

### D. Actor-path benchmark (single rank, random policy)

`pa3_actor.py --envs 1024 --games 2048 --workers 16 --policy random --pin 0-15`
(2 natural cohorts, 0 operational errors):

| Metric | Value |
|---|---:|
| Games/s | 522.7 |
| Transitions/s | 7,791 |
| Decisions/s | 13,837 |
| CPU cores used | 4.76 of 16 workers |
| obs / mask / step / policy per round | 23.08 / 2.91 / 2.58 / 4.13 ms |
| decode per round | 0.0005 ms (random policy does not parse the payload) |
| reset per cohort | 22.4 ms |
| max RSS | 253 MB |

### Bottleneck reading

1. **Observation encoding and tensor conversion** dominate the training-relevant
   path: ~76% of actor wall time in B is native observation packing, and C shows
   another 2.6x on top of it if the policy converts per-view blobs to torch
   tensors the naive way.
2. **Parallel efficiency, not raw stepping**, limits the engine path: 16 workers
   deliver ~3-4.8 CPU cores of work. Rust transition stepping itself is only
   ~7% of actor wall time.
3. **Two-rank scaling is poor**: one rank reaches 522.7 games/s but two ranks
   reach only 592.6 games/s aggregate, so the second rank adds ~13% for 2x the
   resources. Memory bandwidth/HT contention on the SYS topology is the leading
   explanation and is the next thing to profile.

No model-backed inference benchmark is included here: the PA3-8M actor binding
described in §5 is still being wired, and the random-policy number must not be
reported as end-to-end training throughput.

### E. Bounded PPO smoke (real engine + PA3-8M on GPU 0)

`scripts/run_ppo_smoke.py` collects naturally completed matches from the frozen
training pool with the PA3-8M policy sampled on the GPU, verifies that the
sampled joint log-probability is reproducible by the learner, runs the PPO
update on those rows and checks checkpoint save/resume. It is a correctness
test, never a training run.

Measured on the miniDC with 1,024 environments and 16 workers:

| Run | Matches | Games/s | Rows (actor) | OP errors | Recompute max diff | PPO epochs | approx KL | Grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,024 | 16.1 | 13,639 (12,839) | 0 | 2.0e-5 | 4 | 0.0045 | 1.26 |
| 2 | 10,240 (10,000 target + overshoot) | 16.4 | 132,992 (124,864) | 0 | 1.9e-5 | 4 | 0.0047 | 0.92 |

Run 1 costs: observation 1.6 s, candidate masks 0.2 s, model 19.6 s, Rust
stepping 0.2 s, wall 63.8 s - the GPU forward pass dominates once the policy is
real, and the CPU actor work is single-threaded. GPU peak 3.0 GiB, host peak
1.9 GiB, 0 operational errors, checkpoint saved and reloaded with identical
outputs. This model-backed number is the only throughput in this document that
includes policy inference; it is a smoke-scale measurement, not a training
throughput claim.

Run 2 (the bounded smoke requested for this session) completed 10,240 natural
matches in 625.9 s with 0 operational errors, 266,019 learner decisions, 4.1 M
candidate actions and a clean checkpoint round trip. Cost split: PA3-8M forward
185.0 s (29.6%), native observation packing 13.5 s, legal masks 2.2 s, Rust
stepping 2.0 s, reset 0.15 s; the remaining ~68% is single-threaded Python
orchestration, which is the next optimization target for the training path.
PPO health: 4 epochs, approx KL 0.0047 (per epoch 0.0057/0.0043/0.0046/0.0044),
ratio mean 0.98, clip fraction 0.052, gradient norm 0.92, normalized entropy
0.981, no NaN/Inf, no early stop. The 100M-match run is still not authorized.

## miniDC speedtest commands

Run these on the miniDC in an approved project shell
(`source runtime/minidc.env`), from a checkout of this branch. Use the
project's Rust toolchain instead of the Mac-local `scripts/cargo.sh` wrapper.
Do not upgrade or downgrade driver, CUDA, PyTorch, NCCL or system packages.

### A. Engine-only training-pool benchmark

```
POOL_REPEAT=20 cargo run --release --locked --example pool_run_report
```

Expected: `training pool: 1137 teams`, `trajectory completions: 1137
(100.0%)`, `static mechanic completeness: 1137`, `blocked ... : 0`, and a
throughput line with decisions/sec and games/sec over 22,740 games.

### B. PyO3 batch benchmark (2 × 1,024 = 2,048 environments)

```
bash scripts/build_python.sh            # once, inside the approved venv
PYTHONPATH=engine/python .venv/bin/python scripts/run_actor_pair.py \
    --envs 1024 --games 4096 --workers 16 \
    --teams engine/data/training-teams.json
```

Expected: both ranks cross-check the documented PCI/NUMA/CPU topology, then
report per-rank and aggregate transitions, policy decisions, natural completed
games, observation/legal-action/bridge/step timings, worker scaling and an
operational-error count. Operational errors are never counted as games.

### C. Observation encoding benchmark

```
PYTHONPATH=engine/python .venv/bin/python engine/python/bench_observation.py
```

Expected: packed-payload decode plus numpy/torch tensor conversion cost on a
fixture-cohort batch, reported separately from the Rust-side packing cost.

### D. Actor-path benchmark

```
PYTHONPATH=engine/python .venv/bin/python engine/python/pa3_actor.py \
    --envs 1024 --games 2048 --workers 16 --policy random \
    --teams engine/data/training-teams.json --pin 0-15
```

This exercises observation encode, mask query and `step_batch` with a random
policy placeholder (no model weights needed). For the PA3 path, run the same
actor with the model-backed policy from the trainer side and read the same
accounting block. The actor drains each cohort naturally and resets the group
afterwards, so `--games` is a real multi-cohort target; `--no-reset` keeps the
old single-cohort behaviour. `decode_ms_per_round` stays near zero for the
random policy because it never parses the packed observation payload.

### E. PPO smoke (only after A–D pass)

The PPO trainer/launcher is **not** part of this repository; use the approved
miniDC trainer entry point. Constraints for the smoke run:

* 10,000–100,000 matches maximum; this is a smoke test, never a training run.
* fp16/fp32 as configured; PyTorch 2.14.0+cu126; CUDA 12.8 toolkit only for
  compilation if needed; no bf16/fp8/TF32/FlashAttention2/custom CUDA.
* Check checkpoint save/resume, NaN/Inf, KL, entropy, memory and disk spill.
* Do not stop `dsh-web` or `llama-swap`; the llama-swap unload endpoint is only
  for an explicitly approved run.

## How to interpret the speedtest results

* The engine-only number isolates native simulation cost (no Python, no GPU).
* The PyO3 batch number is the training-relevant engine cost; the gap to A is
  the Python/bridge overhead per round (accounted per decision).
* The observation number tells you whether tensor conversion, not stepping,
  is the bottleneck.
* The actor-path number is the end-to-end CPU+GPU path; if GPU utilization is
  low while CPU workers are saturated, the engine is the limiter, and if the
  reverse, the policy forward is.
* Any operational error in A–D is a blocker; abort counts must be zero before
  PPO smoke is allowed.

## What remains for full readiness

* 29 blocked moves (type-addition trio, Transform/Instruct/Copycat closure,
  Salt Cure/Syrup Bomb/Grav Apple, Magnet Rise, Attract, Uproar, Future Sight,
  …), 23 blocked abilities and the 11 dynamic-closure callers.
* 44 ported abilities still lack a differential witness.
* Full-coverage corpus regeneration, criteria 8/15/16, and the formal
  `readiness_check` exit-0 run on the miniDC.
