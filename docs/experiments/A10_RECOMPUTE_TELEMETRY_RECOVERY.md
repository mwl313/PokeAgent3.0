# A10 recompute recovery and telemetry

The original rank1 failure at absolute iteration 324 (`1.397e-3` versus
`1e-3`) has no saved rollout. Its exact cause cannot be reconstructed. The
newest verified generation is `iter-000001302528-1791879907652446477`,
iteration 318. Both manifest hashes and actual model/Adam digests agree;
scheduler, GradScaler, frozen 1,220-team manifest, opponent pool, rank metadata
and owned RNG state are preserved. The last completed historical clock was
1,323,008; 20,480 matches of unsaved updates must be replayed, never counted
twice.

A frozen-policy probe of 2,048 games reproduced batch-shape sensitivity:
the 576-row gate differed by at most `2.599e-5`, while isolated-row AMP
recomputation differed by up to `9.875e-3`. For that preview outlier the
original 2,048-row actor group replayed the stored logprob exactly and two
FP32 shapes agreed within `3.815e-6`. This demonstrates numerical shape
sensitivity, but does not prove the old failure was harmless. Single-row
execution is diagnostic and is not the production learner's microbatch.

The opt-in secondary audit records original actor-group membership and hashes
of observations/actions/logprobs, without changing policy sampling. At most
eight primary outliers may be considered. Acceptance requires unchanged legal
inputs/prefixes, actor-group replay and repeat within `1e-4`, FP32 group/isolated
agreement within `1e-4`, and each AMP result within the original `1e-3` of its
FP32 reference. Everything else fails closed. In particular, the larger
isolated diagnostic outliers were rejected by these bounds. The primary
tolerance, PPO objective, optimizer, DDP, LR and runtime profile are unchanged.
Any primary failure writes CPU replay evidence before an update; no policy
logprob is rewritten.

Clean runtime SHA `662d515eca8de5663d1857fefd1fcb521e7df631` passed a bounded
two-V100 restore/update smoke: 4,096 diagnostic matches, both rank gates
passed (`2.742e-5` / `2.599e-5`), matching post-update model/optimizer digests,
zero operational errors or skipped steps, two epochs, 36 optimizer steps/rank.
Both initial model/Adam digests and scheduler/GradScaler/RNG restore checks
passed. The diagnostic used full isolated audits and achieved 52.50 all-in
games/s; it is not a production throughput measurement or campaign progress.
The implementation/regression panel passed 49 tests; the additional registered
source-migration archive test also passed. The subsequent commit adds tests
and this report, with identical runtime code.

Continuation uses a trusted-local receipt bound to the exact source SHA,
checkpoint paths and both artifact hashes. Protected model/learner/engine/
configuration files must remain unchanged. This is not a global SHA bypass.
Recovery checkpoints retain the actual trainer SHA. CPU evaluation snapshots
retain the frozen evaluation execution identity and separately record the
actual `training_source_sha` and migration receipt; old policies and receipts
are not relabeled. Only discovery needs the additive compatibility code;
the battle worker keeps running unchanged.

Real operational receipts, probe JSON, source-migration receipt, launch logs
and production telemetry live under the existing campaign root:
`/home/minidc/Desktop/PokeAgent3.0-A/runs/a10_production_20261013_v2`.
See `operations/recovery_20261013` and
`report_resume_telemetry_20261013.json.rank{0,1}.ppo.jsonl` for the actual
activation outcome and growing completed-iteration series. Diagnostic
smoke metrics are separate under the recovery worktree's
`runs/recovery_20261013/smoke.json*` and do not advance the campaign.

Limitations: normal CUDA updates are numerically bounded rather than bit-exact
across independent runs; the parity gate stratifies 576 rows, not every rollout
row. An unexplained future outlier still stops training and preserves evidence.
Provenance hashing adds CPU work; measure production throughput separately.
