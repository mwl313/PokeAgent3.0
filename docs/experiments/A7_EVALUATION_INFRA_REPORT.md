# Track A7 evaluation infrastructure receipt

Implemented in this milestone:

- Fixed `FrozenPPOPolicy` initialization by providing its layout and device to
  the legal-candidate conversion helper.
- Added strict selected-prefix validation. Multi-branch scoring now requires an
  explicit legal prefix; the adapter never invents an intermediate argmax path.
- Added `agent.evaluation.policy_archive`: atomic JSON registry, immutable file
  and model digests, provenance/schema fields, incumbent/latest labels, and
  verification.
- Registered six verified local policies in `policy_archive.json`, including
  A5 Control as provisional incumbent and A5 Historical as archive.
- Added `agent.evaluation.league` for evaluation-only frozen policy pair
  receipts. Results are descriptive and never block training or roll back a
  learner.
- Added resumable match-clock scheduling in `agent.evaluation.schedule` with
  50k lightweight and 200k full evaluation boundaries and atomic completion
  receipts.

Tests: the existing focused evaluator/checkpoint/cache suite passes 9 tests.
A real native-engine adapter smoke test produced finite branch scores and value
from the A5 Control checkpoint. Schedule boundary checks return light at
402,256 and full at 552,256 as specified.

Not completed in A7: a genuine multi-environment batched neural evaluator.
The current sequential evaluator remains the correctness reference at the A6
measurement of 2,048 games in 453.8 s (4.51 games/s). No new throughput claim
is made. Implementing the requested batched engine path requires a separate
bounded correctness/performance implementation; it was not substituted with a
Python loop and is therefore explicitly recorded as outstanding.

The recompute discrepancy remains unresolved at `1.0507107e-3` versus `1e-3`.
Production remains fail-closed and no additional PPO training was run.
