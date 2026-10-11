# A7.1 Evaluation League handoff

The external entrypoint is `scripts/run_evaluation_league.py`.  It deliberately
does not run in the PPO optimizer/rollout path.  At a drained experimental
boundary, invoke `schedule` with the rank-0 envelope; it validates that the
envelope has complete rank digests and records a pending job pinned to the
actual checkpoint and committed-match count.  Evaluation can then be invoked
later with `run-pair`, or left pending while GPUs are occupied.

The persistent interfaces for a later A8 execution backend are:

- `build_fixture_manifest` and `FixedPolicyEvaluator.run` define the complete
  seat-swapped fixture and outcome contract.
- `run_pair` takes an engine plus fixtures, verifies both archive entries, and
  atomically stores the identity-keyed receipt.
- `evaluation_identity` defines the cache key: artifact SHA/model digest,
  fixture and team pool hashes, engine/evaluator versions, seed and sampling
  configuration.  A batched evaluator must keep those semantics or use a new
  evaluator version.

The checked-in archive uses portable paths relative to `docs/experiments` and
explicit `verification-only` storage for the protected local A3--A5 artifacts.
Every use verifies SHA-256, model digest, config/provenance, committed matches,
schemas and the paired DDP envelope.  A content-addressed archive copy is
available through `register_checkpoint(..., storage_mode="content-addressed")`.

The 32-game native smoke result is intentionally inconclusive: Control scored
18--14 versus Historical with pair bootstrap CI [0.40625, 0.71875].  It proves
the execution and persistence chain, not superiority.  A7.1 leaves training
state untouched; regression warnings are descriptive and cannot roll back or
block PPO.
