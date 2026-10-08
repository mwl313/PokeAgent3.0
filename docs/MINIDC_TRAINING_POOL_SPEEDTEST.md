# miniDC training-pool speedtest (branch `training-pool/minidc-speedtest-ready`)

Development handoff document. It is not a readiness claim, and nothing here
authorizes training. The only acceptable final readiness signal remains
`readiness_check` exiting 0 across all 16 criteria **on the miniDC**.

## Purpose

The frozen 1,136-team training pool is now fully usable by the native engine:
every member's moves, ability and item are implemented, every team completes a
natural battle in the engine-only probe, and there are zero operational aborts.
This branch exists so the miniDC can measure the *real* training-pool speed
(engine-only, PyO3 batch, observation, actor path) and, only if those pass, run
a short PPO smoke test. It does not attempt full M-C engine coverage.

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

## miniDC speedtest commands

Run these on the miniDC in an approved project shell
(`source runtime/minidc.env`), from a checkout of this branch. Use the
project's Rust toolchain instead of the Mac-local `scripts/cargo.sh` wrapper.
Do not upgrade or downgrade driver, CUDA, PyTorch, NCCL or system packages.

### A. Engine-only training-pool benchmark

```
POOL_REPEAT=20 cargo run --release --locked --example pool_run_report
```

Expected: `training pool: 1136 teams`, `trajectory completions: 1136
(100.0%)`, `static mechanic completeness: 1136`, `blocked ... : 0`, and a
throughput line with decisions/sec and games/sec over 22,720 games.

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
accounting block.

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
