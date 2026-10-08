# V3 numeric parity and collection-contract audit (G0)

Base `cba9dfa`; branch `optimization/pa3-realpolicy-throughput`. This closes the
v3.0 plan's G0 gate: real-model gradient parity, streaming-vs-standard
equivalence, the actor collection contract, and the synchronization-count
reconciliation demanded by §3 of the plan.

## 1. Real PA3-8M gradient parity

`tests/agent/test_real_gradient_parity.py` uses the real PA3-8M architecture and
a mock rollout, aligns the behavior policy to the current weights (ratio ≈ 1,
the plan's "identical weights" reference), then compares:

* the **accumulated scaled losses** against the single-pass loss of the same
  rows (this isolates the weighting math from backward round-off), and
* the raw gradients per parameter group (encoder / scorer / value head) with
  gradient clipping disabled so `clip_grad_norm_` cannot distort the result.

Measured on the parity fixture (12 rows, micro splits 1/2/4/6):

* accumulated vs single-pass loss difference ≤ 1e-5 relative (exact within
  FP32), e.g. −0.0099857 both ways on the 12-row reference;
* raw gradient max |Δ| ≤ 1e-4·scale + 1e-7 and global-norm relative difference
  < 1e-4 for every group and every micro split;
* different shuffle orders agree to ~1e-7 (order independence);
* singleton-only / actor-free microbatches, padded final minibatches and the
  value-only rows produce no NaN and match the reference.

**Two false alarms worth recording** (both were test-harness issues, not code):

1. `state_dict()` returns references, so the reference model's optimizer step
   mutated the "frozen" weights the accumulated runs loaded; captured states
   must be cloned.
2. `clip_grad_norm_` scales `parameter.grad` in place; a raw-gradient parity
   comparison must disable clipping (clipping parity is verified separately by
   the streaming test, which compares the final weights after one step).

## 2. Streaming vs whole-iteration equivalence

`tests/agent/test_streaming_equivalence.py` runs
`prepare_batch`+`update` and `prepare_streaming`+`update_streaming` on the same
rows, the same model weights and the same generator seed:

* report parity: policy/value loss, entropy, uniform-KL, KL, ratio, clip
  fraction and gradient norm agree within 1e-5 relative; epoch count,
  optimizer steps, skipped steps and actor rows are identical;
* raw gradient max |Δ| ≤ 1e-5·scale + 1e-8;
* after one optimizer step the weights agree within 1e-6 relative per tensor;
* the actor-free-row variant also passes with no NaN.

The streaming path appends padding while `iter_minibatches` spreads it; the
exact row-weighted accumulation makes the objective invariant to that
placement, which this test confirms empirically (the plan flagged the
placement difference as needing A/B evidence).

## 3. Actor collection contract audit

**Finding (real mismatch, now resolved).** `configs/train.yaml` declares
`collect_both_sides_when_current_self_play: true`, but
`NativeCollector._collect_round` recorded only the assigned learner seat
(`side == _learner_side(roles[env])`). With no history-opponent pool yet, both
seats run the same frozen current policy, so half of the current-policy rows
were being discarded.

**Fix.** `NativeCollectorConfig.collect_both_sides_when_current_self_play`
(default `True`, matching the config) records every current-policy seat, and
the terminal reward is now written to **each recorded side's own trajectory**
(win +1 / loss −1 / draw 0 per side; the match counter still counts each
natural match exactly once). Historical-opponent rows stay excluded
(`collect_historical_opponent_rows: false`).

Controlled A/B (2,048 natural matches, GPU0, 1,024 envs, 16 workers, microbatch
1024, streaming, checkpoint inside the window, 3 repeats):

| collection | all-in games/s (median) | rows | rows/game | optimizer steps | learner rows/s |
|---|---:|---:|---:|---:|---:|
| single learner seat (pre-v3) | 34.24 | 27,490 | 13.42 | 28 | 460 |
| both current sides (contract) | 19.37 | 55,039 | 26.87 | 56 | **520** |

Games/s halves because each match now contributes twice the learner rows; the
training-relevant throughput (**learner rows/s, i.e. current-policy decisions
per second**) is ~13% *higher* because the collection cost is amortised over
twice the data. All v1.1/v2.0 `games/s` figures in the earlier reports were
measured under the single-seat contract; they are labelled as such from here on.

## 4. Synchronization-count reconciliation (§3 of the plan)

The v2 breakdown claimed "2,112 per-microbatch GPU syncs per 2,048-match
iteration". That number was wrong: it mixed the two run scales. The actual
microbatch forward calls are

* 2,048-match iteration: `ceil(27,490/4096) = 7` minibatches per epoch × 4
  epochs = **28** minibatches × 16 micros = **448** per-micro syncs;
* 10,240-match iteration: `ceil(136,259/4096) = 34` × 4 = **136** minibatches ×
  16 micros = **2,176** per-micro syncs.

The number of *optimizer steps* (28 and 136) matches the committed-step counts
in the raw JSON. The removed-sync claim is corrected to 448 / 2,176 in the v3
reporting; the previous figure is not reused as evidence.
