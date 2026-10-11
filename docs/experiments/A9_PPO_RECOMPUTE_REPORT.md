# Track A9 — PPO recompute and evaluation scheduling

Implementation was performed on the A8 clean base `94043b9`.  No PPO
production defaults, battle mechanics, frozen team pool, or Track C files were
changed.  The A8 evaluator and archive remain intact; archive verification
passes for all six registered policies.

## Recompute diagnosis

The reproducible discrepancy was a data-contract problem amplified by mixed
precision, rather than a masking or candidate-ordering error.  The native actor
sampled from FP32 observations, while `ObservationBatch.to_compact_numpy()`
downcast continuous features to FP16 before they entered the rollout buffer.
The recompute path therefore evaluated a different observation.  The effect
was largest in the trained branch-2 actor-active strata.  Additional bounded
diagnostics found no unsafe candidate-prefix, mask, or sample-order mutation.

The fix keeps compact integer fields compact, preserves policy-critical
continuous features as FP32, and runs the behavior-policy encoder/scorer under
FP32 autocast-disabled execution.  The PPO objective and its masks are
unchanged.  The existing 1e-3 recompute gate remains fail-closed; its
tolerance was not increased.

| bounded case | max absolute logprob error | mean | p95 |
| --- | ---: | ---: | ---: |
| FP16 packed random model, before encoder fix | 4.06e-4 | 8.00e-5 | — |
| FP32 packed random model | 1.72e-5 | 1.96e-6 | — |
| FP16 packed with FP32 encoder | 1.18e-5 | 2.19e-6 | 7.13e-6 |
| trained A5 resume, before lossless rollout storage | 2.70e-4 | 7.81e-6 | 3.63e-5 |
| trained A5 resume, after lossless rollout storage | 1.43e-5 / 1.64e-5 (ranks 0/1) | 5.42e-7 / 5.86e-7 | 2.77e-6 / 2.73e-6 |

In the final two-rank resume smoke, rank 0 and rank 1 both passed the gate.
The largest final strata were `kind1_branches2_actor1` (rank 0:
1.43e-5) and `kind2_branches1_actor1` (rank 1: 1.64e-5), well below 1e-3.
The run completed 2,048 games, reached committed-match clock 354,304, and
reported equal model digests with zero operational errors.

This is a bounded correction receipt for the tested A5 checkpoint and current
model path.  It does not claim bit-exact floating-point identity on every GPU
kernel or future engine build; the gate remains mandatory for wider runs.

## Non-aligned evaluation scheduling

When a safe checkpoint first appears after a crossed boundary, the scheduler
assigns it only to the earliest due boundary and records the actual captured
match clock.  Later crossed boundaries remain explicit deferred jobs with
`missing_snapshot=true`.  Repeating the scheduler or restarting the process
does not create a duplicate.  The disposable A5 candidate workflow was
scheduled, processed with a four-game smoke panel, persisted successfully, and
returned no work on the second invocation.

## Validation and scope

The A8 fixed-policy evaluator remains available in sequential and native
batched modes.  A bounded post-change batched pair (four games) completed with
2 wins, 2 losses, 0 draws, 0 operational errors, and 0 illegal actions.
The full 2,048-game A8 throughput panel was not rerun for this numerical-only
change, and GPU peak-memory capture remains outside this receipt.

No A10 training or long RL run was started.  The next milestone is a bounded
fresh PPO iteration using the corrected lossless observation contract, followed
by the approved A10 work only after review.
