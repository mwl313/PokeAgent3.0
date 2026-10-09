# V4 correctness-formula audit (C0)

Branch `optimization/pa3-realpolicy-throughput`, base
`816707ecf8036ae493140774dc2d3b43e6e1a72e` (the v4.0 plan's pinned SHA).
This document records the C0 gate: the PPO regularizer gradient defect, its
fix, the independent objective oracle, and the evidence that no reported
scalar changed.

## 1. Defect: detached entropy and uniform-KL regularizers

`agent/model/pa3_model.py::PA3Model._branch_stats` computed both PPO
regularizers from detached tensors:

```python
entropy = -(probs.detach() * log_prob.detach()).sum(dim=-1)
mean_log_prob = (log_prob.detach() * mask).sum(dim=-1) / safe_k
```

`agent/ppo/learner.py::_forward_terms` adds
`-entropy_coefficient * entropy + uniform_kl_coefficient * uniform_kl` to the
loss, so both terms contributed **zero gradient** to the encoder/scorer while
still reporting correct scalar values.

Two related hypotheses were checked before fixing:

* The regularizer loss had no `grad_fn` at all (confirmed: pre-fix
  `entropy.requires_grad == False` on a logits tensor with `requires_grad`).
* The full policy/ratio path was **not** disconnected: `_run_branches` writes
  branch results into pre-allocated tensors with in-place slice assignment, and
  PyTorch's `CopySlices` / `CopyBackwards` autograd nodes keep those outputs
  connected to the scorer. The pre-fix learner loss therefore still produced a
  nonzero scorer gradient from the clipped surrogate and the value path; only
  the regularizer contribution was missing. (The `delta_norm` measurement
  below isolates exactly the restored regularizer gradient; the
  clipped-surrogate gradient itself was never in question.)

## 2. Fix

```python
entropy = -(probs * log_prob).sum(dim=-1)
...
mean_log_prob = (log_prob * mask).sum(dim=-1) / safe_k
```

Both regularizers now use the differentiable `probs`/`log_prob` produced by
`masked_log_softmax`. The forward values are unchanged: detaching does not
alter the arithmetic, only the autograd graph. Illegal candidates keep exactly
zero probability; padded/singleton rows keep zero entropy/KL and produce no
gradient for their (masked) logits.

Monitoring stays detached: `_forward_terms` returns `.detach()`ed sums for all
reported metrics, and the actor sampling path (`PA3Model.sample`) stays under
`torch.no_grad()`.

## 3. Evidence

### 3.1 Old-fail / new-pass regression tests

`tests/agent/test_policy_regularizer_gradients.py` (7 tests). On the pre-fix
tree **all 7 fail** (6 with `requires_grad == False`, the learner-level one
with an exactly zero scorer gradient). After the fix all 7 pass:

* `test_entropy_loss_produces_nonzero_logit_gradient`
* `test_uniform_kl_loss_produces_nonzero_logit_gradient`
* `test_entropy_kl_gradient_matches_analytic_and_finite_difference`
  (analytic derivative `-p_i(log p_i + H)/log K` and
  `(p_i - 1/K)/log K`, plus central finite differences)
* `test_singleton_and_padded_branch_zero_gradient`
* `test_all_illegal_mask_is_never_sampled_or_used`
* `test_model_regularizers_reach_scorer_and_keep_scalar_parity`
* `test_ppo_learner_loss_keeps_regularizer_graph` (advantages forced to zero so
  the only possible scorer gradient is the regularizer's)

Commands and results (minidc, `.venv` Python 3.12.3 / torch 2.14.0+cu126):

```
PYTHONPATH=engine/python:. .venv/bin/python -m pytest tests/agent/test_policy_regularizer_gradients.py -q
# pre-fix:  7 failed
# post-fix: 7 passed
PYTHONPATH=engine/python:. .venv/bin/python -m pytest tests/agent -q
# post-fix: 74 passed, 1 warning (tests/agent/test_entropy_kl.py calls float()
#           on an eval tensor that is now graph-attached by design; the value
#           is unchanged and the warning is cosmetic)
```

### 3.2 Scalar parity and gradient change (raw JSON)

`scripts/v4_c0_regularizer_gradient_evidence.py` runs the same real-architecture
minibatch (24 rows, 143,923-parameter test config, seed 20261009) twice: once
with a pre-fix `_branch_stats` monkeypatch and once with the fixed code, then
backpropagates the full learner objective. Raw result:
`runs/perf/v4/c0_regularizer_gradient_evidence.json` (git-ignored; numbers
reproduced below).

| quantity | pre-fix (detached) | fixed | abs diff |
|---|---|---|---|
| full objective loss | -0.009999163448810577 | -0.009999163448810577 | 0.0 |
| policy mean | -4.7087669e-06 | -4.7087669e-06 | 0.0 |
| value mean | 0.45106539 | 0.45106539 | 0.0 |
| entropy mean | 0.99949610 | 0.99949610 | 0.0 |
| uniform-KL mean | 0.00050548691 | 0.00050548691 | 0.0 |
| approx-KL mean | 1.1781509e-10 | 1.1781509e-10 | 0.0 |
| ratio mean | 1.0000031 | 1.0000031 | 0.0 |
| clip fraction | 0.0 | 0.0 | 0.0 |

Parameter-group gradient norms, same weights and rows:

| group | pre-fix | fixed | ‖Δgrad‖ | relative Δ |
|---|---|---|---|---|
| encoder | 0.34459516 | 0.34459069 | 3.0644e-05 | 8.89e-05 |
| scorer | 14.47981453 | 14.47948742 | 1.6270e-03 | 1.12e-04 |
| value_head | 0.0 | 0.0 | 0.0 | — |

The delta is the restored regularizer gradient (exactly zero pre-fix). Because
the initial policy is near-uniform, the per-row entropy gradient is small:
regularizer-only scorer gradients over the same batch are 2.9571 (entropy sum)
and 2.9654 (uniform-KL sum) before the `1/actor_count` normalization and the
0.01/0.001 coefficients. This is a correctness change, **not** a speed change:
baselines measured after this commit are not comparable to pre-fix numbers.

## 4. Full-objective audit status (§3.2 of the plan)

`tests/agent/test_full_ppo_objective_reference.py` adds an independent FP32
reimplementation of the objective that never calls the learner's
loss/aggregation helpers:

* clipped surrogate sign, `min(r·A, clamp(r)·A)`, actor mask and
  `ratio = exp(new − old)`;
* value half-MSE with the configured `value_coefficient` outside
  `loss_unscaled`; actor-free rows collapse the objective exactly onto the
  value loss (shared encoder keeps the critic gradient; scorer gradient is
  exactly zero);
* entropy/KL sign and coefficients re-derived from `_forward` composition;
* `approx_kl = (r−1) − log r` row weighting (guarded by the pre-existing
  `test_update_statistics.py` tests).

The reference matches the learner loss to `atol=1e-6` and every parameter
gradient to `1e-5·scale + 1e-7` (worst relative difference < 1e-4). Adam
multi-step parity (full minibatch vs 4-row microbatches, 3 consecutive
updates) matches model weights (`atol=1e-6`), `exp_avg`, `exp_avg_sq`, `step`,
LR and optimizer-step count. Checkpoint save/resume continues identical
updates.

P0/P0.2/P0.3 items of the v2 audit (statistic aggregation, exact
row-weighted accumulation, GradScaler skip accounting) and the v3 parity items
(real-model accumulated vs single-pass gradients, streaming equivalence) were
re-run green in the same `tests/agent` invocation.

## 5. Collection-contract audit status (§3.3 of the plan)

Covered by the v3 work (`docs/perf/V3_NUMERIC_PARITY.md` §3) plus
`tests/agent/test_rollout_buffer.py::test_current_policy_self_play_collects_both_sides_but_not_history`:
both current-policy seats are recorded once per match, historical-policy rows
are excluded, and the terminal reward is carried to each seat's last request.
The additional v4 regression requested by the plan
(`test_collect_both_current_sides.py`) is tracked in
`V4_COLLECTION_CONTRACT_AUDIT.md`.

## 6. Verification boundary

* This gate was executed on minidc with the pinned torch/CUDA stack; no driver,
  NCCL, power-cap or service change was made.
* Measured-here numbers only (no extrapolation). C1 (single-GPU oracle on real
  PA3-8M rows, FP16 autocast + GradScaler) is the next gate and must pass
  before any distributed or speed result is promoted.
* The pre-fix/post-fix gradient difference is the expected correction; it is
  never reported as a speedup.
