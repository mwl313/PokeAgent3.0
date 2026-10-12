# A9.3 Final Performance Report

Mode D with the D1 stability switch is a viable stable-throughput candidate.
The unchanged sampled/recomputed gate passed all 25 iterations (51,200 games)
and both ranks remained digest-identical.

| arm | games | gate | all-in games/s | steady games/s | result |
|---|---:|---:|---:|---:|---|
| A9.2 Mode D original | 16,384 before failure | 13/14 | — | — | failed at iter 14, max 1.438e-3 |
| A9.3 D1 | 51,200 | 25/25 | 88.41 | 89.45 | pass |

The mean PPO update is approximately 10.4 seconds per rank iteration; the
run executed two PPO epochs and 18 optimizer steps per iteration. No
operational errors were reported. D1 disables V100 FP16 reduced precision
reduction only; it does not change the PPO objective, action masks, minibatch
ordering, or gate tolerance.

A separate one-iteration checkpoint creation and one-iteration restore run
both passed with equal rank digests. The frozen checkpoint loader restores and
exposes `pa3-precision-D-v1` in its policy identity.

The direct causal attribution is bounded: evidence isolates a V100 FP16
reduction stability issue at the contract level, while a complete per-layer
trace and a second independent 25-iteration seed remain future work.
