# V2 correctness audit (P0 of the v2.0 plan)

Branch `optimization/pa3-realpolicy-throughput`, base `b72f5f4`. This document
records the P0 defects, their fixes and the regression tests that guard them.
No game rule, observation schema, model architecture or PPO hyperparameter was
changed; the changes are metric-accounting and gradient-accumulation
corrections.

## P0.1 Last-microbatch-only statistic aggregation

**Defect (confirmed in code).** `PPOLearner.update()` accumulated its report
statistics from the `stats` dict left over from the *last* microbatch of each
minibatch, and `epoch_kl` from that same last microbatch. The mean of
per-microbatch means is not the row-weighted mean whenever microbatches differ
in valid actor/value rows (the padded final minibatch, uneven actor masks, or
any partial chunk). The early-stop KL monitor inherited the same bias.

**Fix.**

* `_forward_terms()` returns raw *sums and denominators* (`kl_sum`, `ratio_sum`,
  `clip_sum`, `entropy_sum`, `uniform_kl_sum`, `policy_sum`, `value_sum`,
  `actor_count`, `value_count`) instead of means only; `_forward()` remains as a
  compatible view for the existing tests.
* `update()` accumulates those GPU scalars per microbatch and performs a
  **single D2H read per epoch** (no `.item()` inside the micro loop).
* Epoch and iteration metrics are `sum / denominator` over valid actor rows
  (KL, ratio, clip fraction, entropy, uniform-KL, policy) and valid value rows
  (value loss). The epoch KL used for early stopping is the row-weighted
  average over the whole epoch.

**Regression tests** (`tests/agent/test_update_statistics.py`, 6 tests):
a synthetic batch whose last microbatch has KL = 0 while earlier rows carry
KL ≈ 0.3 reports 0.2046 (row-weighted) instead of 0.0; the actor-mask variant
weights only valid actor rows; ratio/clip fractions are row-weighted; padded
final minibatches match an unpadded reference.

## P0.2 Gradient-accumulation and padding-weighting equivalence

**Finding.** The legacy accumulation used
`scale = micro.sample_weight / len(microbatches)`, which equals the exact
full-minibatch objective only when every microbatch has the same number of
valid actor rows and the minibatch is not padded. On uneven actor masks the
gradient provably differs (the new test measures it).

**Fix.** With `PPOConfig.exact_row_weighted_accumulation = True` (default) each
microbatch contributes its own share of the minibatch's valid actor rows to the
policy/entropy/KL terms and of its valid value rows to the value term:

```
loss = weight * ( actor_loss_micro * (n_actor_micro / n_actor_minibatch)
                + 0.5 * value_mean_micro * (n_valid_micro / n_valid_minibatch) )
```

`weight` keeps the existing cross-rank sample-weighted DDP reduction. Tests
verify that one-microbatch and 1-row-microbatch accumulation give identical
gradients (atol 1e-6), that a padded final minibatch matches the unpadded
reference, and that the legacy path differs on uneven actor rows (the A/B
guard). The legacy path stays available via the flag for exactly that
comparison.

## P0.3 Optimizer-step accounting under GradScaler skips

`optimizer_steps` previously incremented unconditionally after
`scaler.step()`. It now compares `scaler.get_scale()` before and after
`scaler.update()`, counts only steps that actually applied, and records
`optimizer_steps_skipped` in the report. A stub-scaler test forces a skip and
asserts `optimizer_steps == 0`, `optimizer_steps_skipped >= 1`.

## P0.3 Measurement honesty in the benchmark

* `--checkpoint` now performs a **real** crash-safe write (temp file → `fsync`
  → atomic rename) inside the measured window; `checkpoint_write_wall_s` is
  reported separately and the all-in wall includes it. Measured cost on the
  miniDC: 0.46–0.49 s for a 105 MB learner state.
* `report_includes` lists exactly which phases the all-in number covers
  (collect, GAE, prepare, 4-epoch update, scheduler, grad scaler, checkpoint
  write) so it cannot be mistaken for a full production steady-state figure.
* The score table now reports `bounded_ppo_games_per_s` (collect + update) next
  to `all_in_committed_games_per_s` (with checkpoint).
* `stage_medians` reports `median_share_of_collect`, the median of each repeat's
  own stage/collect ratio, instead of pairing a stage median with another
  repeat's wall.
* The run manifest records `git_diff_hash` in addition to `git_dirty`, so a
  dirty-tree run is identifiable exactly rather than only flagged.

## P0.3 Stratified recompute gate

`recompute_check` no longer inspects the first 1,024 rows only. It samples up
to 96 rows per `(request kind, branch count, actor active)` stratum and reports
each stratum's row count and maximum absolute difference, plus the sampled
fraction of the rollout. Measured on a 1,024-match iteration: 546 rows sampled
from 13,775 across 6 strata, max |Δ log p| = 6.6e-5 (fp16 gate 1e-3), with
`kind1_branches2_actor1` at 6.6e-5, `kind2_branches1_actor1` at 3e-6 and the
other four strata exactly 0.

## Not yet done in P0

* A `steady_state_all_in_committed_games_per_s` operational benchmark
  (metrics writer, evaluation window, DDP sync) is still a separate deliverable;
  this document only guarantees the bounded and checkpoint-inclusive numbers
  are labeled correctly.
* Gradient parity across ranks under DDP is part of P3 and not claimed here.
