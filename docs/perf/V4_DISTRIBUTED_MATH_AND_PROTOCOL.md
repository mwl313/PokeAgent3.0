# V4 distributed math and protocol (D0 / D1 / M1)

Branch `optimization/pa3-realpolicy-throughput`. Base for this step: `cc4a39a`
(C1). This document records the distributed objective contract, the fixed-step
collective protocol, the repaired launcher, and the measured parity of the two
executors against the C1 single-GPU oracle.

## 1. Global objective (D0)

For a minibatch with global valid actor rows `A = Σ_r A_r` and valid value rows
`V = Σ_r V_r`, with `S_actor,r` / `S_value,r` the rank-local row sums of
`(policy − c_ent·H + c_kl·KL)` and `0.5·MSE`:

```
GLOBAL_LOSS = Σ_r S_actor,r / max(A,1) + value_coef · Σ_r S_value,r / max(V,1)
```

* DDP: rank loss = `world_size · (S_actor,r/A + value_coef·S_value,r/V)` so
  DDP's post-backward average reproduces the global gradient.
* Manual all-reduce: rank loss = `S_actor,r/A + value_coef·S_value,r/V` with **no**
  world-size factor; one FP32 flat `all_reduce(SUM)` reproduces the global
  gradient.

Advantages are normalised over the **global** actor rows with a two-pass
(count/sum, then squared deviations) float64 reduction
(`normalize_advantages_global`, `PPOLearner.prepare_streaming_ddp`) matching
`normalize_advantages`'s `unbiased=False` definition.

## 2. Fixed-step collective protocol (D1)

Every rank executes:

1. `batch_count = MAX_r ceil(rows_r / per_rank_minibatch)` minibatches per epoch
   (all ranks see the same count).
2. Exactly `micro_steps = ceil(per_rank_minibatch / microbatch)` micro steps per
   minibatch. A rank with no real rows in a step runs a **graph-connected
   zero** micro (the padding-only value term is reattached with
   `values.sum()*0.0`), so every DDP reducer bucket fires in the same order on
   every rank.
3. `no_sync` for the forward *and* backward of every non-final micro step; the
   final micro step always synchronises (`DDPCommunication.context(synchronize=is_last)`).
4. One all-reduce of the minibatch `(A, V)` before the micros; the loss uses the
   global denominators.
5. After accumulation: `unscale_` → per-rank finite flag → global `MIN` →
   global clip → exactly one shared Adam step, or a joint skip (all ranks) when
   the global flag is false or the minibatch has no valid rows. `scaler.update()`
   runs on every rank, so the scaler state never diverges.
6. Epoch sums (`kl_sum`, `actor_count`, …) are all-reduced before the target-KL
   check, so the early-stop decision and `epochs_run` are identical on all
   ranks.
7. The LR clock receives `global_committed_matches = Σ_r completed natural
   matches` (launcher all-reduce), never a per-rank count.

## 3. Fixed defects (§6.1 of the plan)

| defect | status |
|---|---|
| padding-only minibatch skipped the backward → collective mismatch | fixed: fixed micro-step count, graph-connected zero micros |
| `DDPCommunication` sync counter could skip the last sync | fixed: explicit `synchronize=is_last`; the counter form remains only as a fallback |
| rank-local advantage normalization | fixed: `prepare_streaming_ddp` global two-pass stats |
| rank-local LR clock | fixed: launcher all-reduces completed matches |
| rank-local epoch KL / early stop | fixed: epoch sums all-reduced, one global decision |
| rank-local optimizer/AMP skip | fixed: global finiteness flag + shared step/skip, shared `scaler.update()` |
| rank0-only result JSON, sequential `communicate` | fixed: per-rank JSON + per-rank log files drained concurrently with a watchdog |
| parameter-sum digest only | fixed: full named-tensor SHA256 for model **and** Adam moments |
| checkpoint lacks global clock/rank state | fixed: checkpoint carries `global_committed_matches` and rank metadata |

`find_unused_parameters=True` is still used (the plan forbids promoting
`static_graph`/`False` without measurement); every parameter path is exercised
deterministically by the probe.

## 4. Evidence (measured, this machine)

### 4.1 Tests

```
PYTHONPATH=engine/python:. .venv/bin/python -m pytest tests/agent tests/integration -q
# 86 passed (2 warnings: pre-existing float() on an attached eval tensor and a
# non-writable numpy view; neither affects values)
```

New: `test_global_advantage_norm.py` (4), `test_manual_allreduce_parity.py` (3),
`test_ddp_collective_protocol.py` (2, real gloo ranks),
`test_collect_both_current_sides.py` (3, real engine).

### 4.2 Two-rank gloo protocol on CPU (real process group)

`tests/integration/ddp_protocol_worker.py` runs 2 ranks over real code paths
with complete-trajectory shards, against a single-process reference:

| fixture | rank rows | ‖Δweight‖∞ | ‖Δ moment‖∞ | steps | epoch-KL Δ | sync / no_sync |
|---|---:|---:|---:|---:|---:|---:|
| uneven trajectories | 28 / 2 | 1.5e-08 | 7.0e-10 | 1 / 1 | < 1e-9 | 1 / 1 |
| empty rank (dummies) | 28 / 0 | 1.5e-08 | < 1e-9 | 1 / 1 | < 1e-9 | 1 / 1 |

Both ranks produce the same state SHA and the same collective sequence; the
empty rank's graph-connected dummies keep the reducer in step.

### 4.3 Real two-GPU NCCL D1 smoke

`scripts/run_ddp_ppo.py --games 24 --envs 32 --workers 4 --microbatch 64
--minibatch 256 --epochs 1` (raw: `runs/perf/v4/d1_smoke.json`):

* rank rows 847 / 852 (uneven) → 7 minibatches, 7 synchronized and 7 no-sync
  micro steps on both ranks, 7 optimizer steps each, 0 skipped;
* epoch KL `0.005815615975054125` identical on both ranks, same LR and grad
  norm; model and optimizer digests byte-identical (`digests_equal: true`);
* 0 operational errors, 0.82 GiB peak reserved per rank.

### 4.4 D0 real 4096-row parity vs the single-GPU oracle

`scripts/v4_d0_dual_parity.py` (fixture: 6,650 real both-seat rows; 2,046 +
2,047 rows per rank, one global 4,096-row minibatch per epoch). Raw JSON:
`d0_fp32_1step`, `d0_fp32_3step`, `d0_fp16_3step`, `d0_manual_fp32_3step`.

| panel | ‖Δweight‖∞ | ‖Δ moment‖∞ | steps | LR | epoch-KL Δ | grad-norm rel Δ |
|---|---:|---:|---:|---|---:|---:|
| DDP fp32, 1 step | 1.19e-07 | 6.4e-10 | 1/1 | equal | 0.0 | 0.0 |
| DDP fp32, 3 steps | 1.19e-07 | 8.1e-10 | 3/3 | equal | 9.9e-10 | 0.0 |
| Manual fp32, 3 steps | 1.19e-07 | 1.0e-09 | 3/3 | equal | 2.0e-09 | 8.5e-08 |
| DDP fp16, 1 step | 3.85e-06 | 9.4e-07 | 1/1 | equal | 0.0 | 1.7e-06 |
| DDP fp16, 3 steps | 3.95e-06 | 8.0e-06 | 3/3 | equal | 5.9e-07 | 1.0e-02 |

Interpretation: the protocol introduces **no systematic math difference** —
fp32 gradient norms are bit-equal and the weight delta equals one ulp of the
lr-scale update. In fp16 the step-1 gradient norm still agrees to 1.7e-6 and
epoch-1 KL is identical; the 3-step weight divergence (≈0.27·LR) is Adam's
sign-flip amplification of fp16 noise in coordinates whose true gradient is
below `eps` (the same effect quantified by the C1 oracle, where fp16
cross-shape gradient deltas reach 1.4e-3 on tiny-norm groups).

**Declared D0/D1/M1 tolerances:** fp32 weight/moment Δ ≤ 1e-6, epoch-KL Δ ≤
1e-5, gradient-norm rel Δ ≤ 1e-5, identical step/skip/LR/epochs; fp16 uses
gradient-norm rel Δ ≤ 1e-5 at the first step with the sign-flip behaviour
recorded above.

### 4.5 Protocol boundary (fast first implementation)

The fixed-step protocol runs rank-local minibatch plans; exact single-GPU
parity holds when every rank fits inside one local minibatch (the plan's §5.2
"빠른 1차 구현" contract), which is what the fixtures above use. For
multi-minibatch iterations the *sets* of rows grouped into each step differ
from a global permutation; the protocol is still exact for the rows it
processes (and rank digests match, see 4.3), but bit-comparison against a
single-process global permutation requires the plan's global-permutation +
row-exchange variant. That variant is tracked but not implemented; no speed or
parity claim depends on it.

## 5. Not yet done in this gate

* D2 all-in two-GPU A/B (DDP vs manual, same total natural games) — next.
* Per-rank NCCL/NUMA/PCIe telemetry and the F0 critical-path atlas — next.
* Dual checkpoint/resume continuation (single-GPU checkpoint/resume is at C1).
