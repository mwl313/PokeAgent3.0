# Track A9.2 — One-pass throughput recovery

## Final verdict

**PARTIAL.** The production recompute gate was accelerated safely and the
original A5 throughput path was recovered in an AMP candidate, but that
candidate failed the unchanged `1e-3` correctness gate during sustained
training. The safe A9 Mode A reference remains the promoted configuration.

The starting A5 Control result was 92.57 all-in / 93.94 steady games/s. The
A9.1 Mode A result was 37.23 games/s because it included roughly 9.6 seconds
of unnecessary isolated-row diagnostics and a 24-second FP32 PPO update. After
the gate split, the safe Mode A 25-iteration run measured 53.21 all-in and
53.61 steady games/s. It did not reach 90 games/s.

## Confirmed root causes

The old `recompute_check()` performed the batched production comparison and
then re-ran every sampled row as an isolated one-row forward. The isolated
pass was diagnostic only; it did not participate in the gate decision. The
new production path retains stratification, selected-prefix/candidate/mask
checks, nonfinite rejection, per-stratum reporting, and the same `1e-3`
tolerance, while moving isolated evaluation behind `--recompute-audit`.

On the A9 reference run, gate wall time fell from 9.1–9.7 seconds to
0.16–0.23 seconds per rank/iteration. The final 25-iteration receipt measured
about 0.16–0.17 seconds per gate. Controlled negative behavior remains
fail-closed through the same stored-logprob comparison; no tolerance was
changed.

The remaining safe-path regression is the A9 FP32 numerical execution. Its
PPO update averaged about 23.96 seconds, versus about 9.04 seconds in the
historical A5 path. Collection averaged 10.28 seconds versus A5's 8.35
seconds. Preparation was unchanged at about 2.9 seconds. The final safe
iteration therefore averaged about 38.25 seconds, yielding 53.21 games/s.

The original A5 run had the gate disabled because the then-known discrepancy
was `1.0507107e-3` versus `1e-3`. Its AMP learner path is therefore useful as a
performance reference but was not an A9 correctness-equivalent run.

## Actual campaign results

| Arm | Games | All-in g/s | Steady g/s | Max recompute error | PPO update | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| REF/A, isolated audit | 12,288 | 42.51 | 43.52 | 3.21e-5 | 24.58 s | correctness pass; audit receipt |
| A, safe FP32, 25 iterations | 51,200 | 53.21 | 53.61 | 3.21e-5 | 23.96 s | promoted safe reference |
| B, actual AMP scorer | stopped at iteration 3 | — | — | 1.298e-3 | 9.2–9.6 s | fail-closed |
| C, FP16-rounded input + AMP | stopped at iteration 2 | — | — | 1.671e-3 | 9.5 s | fail-closed |
| D, FP16-rounded input + FP32 scorer | 16,384 before failure | 90.50 | 94.82 | 1.438e-3 at failure | 9.77 s | fail-closed at iteration 14 of 25 attempt |

The D 8-iteration screen was valid through all eight iterations, with 144
optimizer steps per rank, zero operational errors, equal DDP digests, and
per-rank maxima of `9.85e-4` and `4.59e-4`. Its longer 25-iteration run then
failed on rank 1 at iteration 14 with `1.438e-3`; it is therefore not a
promotable sustained result. B and C were also stopped before their requested
durations. The earlier A9.1 B/C/D receipts were invalid as precision
comparisons because the parent launcher did not forward `--precision-mode` to
workers; those receipts are retained but excluded from this table.

The 25-iteration safe A run completed 51,200 games, 441 optimizer steps per
rank, 24 two-epoch updates and one natural one-epoch KL early stop. All 25
recompute gates passed, model and optimizer digests remained equal across
ranks, operational errors were zero, and peak reserved memory was 14.994 GiB.

## Before and after stages

| Stage | A5 historical | A9.1 baseline | A9.2 safe A |
| --- | ---: | ---: | ---: |
| Collection | 8.35 s | 10.62 s | 10.28 s |
| Recompute | disabled | 9.65 s | 0.17 s |
| Preparation | 2.90 s | 2.94 s | 2.9 s |
| PPO update | 9.04 s | 24.34 s | 23.96 s |
| Checkpoint/digest | 0.76 s | ~0 s | measured in receipt |
| Full iteration | ~22.1 s | ~48.2 s | ~38.25 s |

## Checkpoint and compatibility validation

The safe Mode A contract was checkpointed to a new A9.2 prefix and resumed in
a fresh two-rank process. Both make and resume runs passed the gate, preserved
the precision metadata, produced equal rank digests, and completed 2,048
games. The original A5 Control envelopes were not overwritten. D's partial
checkpoint artifacts are retained as negative evidence and are not promoted.

The earlier A9.1 B/C/D receipts are retained as superseded evidence; their
parent launcher did not forward the precision flag, so they are not used for
precision conclusions. The corrected B/C/D receipts are the `*_real` files.

Archive verification passed for all six A8 policies, and the relevant
correctness suite passed 27 tests. No battle mechanics or Track C files were
changed.

## Promotion and A10 readiness

No FP16 candidate is promoted. Mode D demonstrated that the 90+ target is
technically reachable on this hardware only while its long-run parity gate is
allowed to fail; that is not an acceptable training configuration. Mode A is
the safe default at 53.21 all-in games/s. A10 has not started and is not
approved by this milestone. Correctness infrastructure and checkpoint
continuation are ready; production performance and competitive-strength
readiness remain separate open decisions.
