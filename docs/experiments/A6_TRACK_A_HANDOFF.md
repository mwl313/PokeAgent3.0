# Track A6 handoff

Commit: `e681438`. Historical Self-Play remains implemented in the prior A5
commit. A6 adds `agent.evaluation.frozen_policy.FrozenPPOPolicy`, a versioned
inference-only adapter for Track C.

The adapter contract is `pa3-frozen-policy-v1`, observation schema
`pa3-entity-observation-v1`, and action schema `atomic-action-v1`. It accepts a
player-visible `ObservationBatch` plus native legal candidate lists, returns
legal-prefix log-probability tensors and a value prediction, and exposes a
`PolicyIdentity` containing model digest and local checkpoint SHA-256. It has no
optimizer, scheduler, RNG ownership, battle handle, or hidden-state input; it
cannot receive gradients. Example:

```python
policy = FrozenPPOPolicy.from_checkpoint("boundary.rank0", device="cuda:0")
result = policy.priors_and_value(observation, legal_candidates,
                                 selected_prefix=prefix_indices)
```

The A5 global row bookkeeping is corrected: control produced 875,177 rows
(34.19 global rows/game) and historical produced 748,530 rows (29.24 global
rows/game), summing both DDP ranks rather than reporting rank 0 only.

The known recompute issue is precisely bounded to packed branch-count-2,
actor-active AMP rows, with max difference `1.0507107e-3` versus `1e-3`.
Focused reproduction found no illegal candidates, mask mismatch, or objective
change; the remaining evidence is consistent with a mixed-precision numerical
rounding discrepancy, but it is not promoted as resolved. Production remains
fail-closed; A3–A5 experimental runs used the explicit no-gate flag.

The existing sequential evaluator remains the correctness reference. A6 did not
replace it with a risky batched engine refactor; the measured A5 2,048-game
runtime was 453.8 s (4.51 games/s). Track C may use the adapter immediately for
frozen PPO priors/value calls, but a native batched evaluator is still a future
performance task.
