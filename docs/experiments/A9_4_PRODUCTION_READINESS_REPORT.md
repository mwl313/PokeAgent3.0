# Track A9.4 Production Readiness Report

## A. Numerical execution contract

The selected contract is Mode D with `fp16-reduction-off`: continuous
observation floats are rounded through the canonical FP16-then-FP32 transform,
the encoder uses AMP, scorer and GRU remain FP32, and V100 reduced-precision
FP16 reductions are disabled. The recompute gate remains `1e-3`.

Experimental checkpoints now record and validate the versioned execution
contract, including stability identity and effective backend setting. A D1
checkpoint resumed with ordinary Mode D is rejected before PPO update. Missing
metadata is accepted only through the explicit trusted A5/A9 legacy migration
allowlist. Frozen-policy loading restores the contract, applies D1 backend
configuration, and independently prequantizes each policy's observations.

## B. Independent training validation

| Metric | A9.3 seed 1 | A9.4 seed 2 |
|---|---:|---:|
| Natural games | 51,200 | 51,200 |
| Recompute gates | 25/25 | 25/25 |
| Max rank errors | 2.909e-5 | 3.517e-5 (rank0), 3.219e-5 (rank1) |
| All-in games/sec | 88.41 | 91.41 |
| Steady-state games/sec | 89.45 | 92.67 |
| PPO update | ~10.4 s | ~10.3 s |
| DDP parity | pass | pass |
| Operational errors | 0 | 0 |

The independent arm used seed `20261010`, PPO-2, LR scale 0.5, minibatch 4096,
microbatch 1024, 1024 environments/rank, and 16 workers. All 25 iterations
executed the full two PPO epochs and 18 optimizer steps.

## C. Evaluation performance

A genuine 2,048-game, 1,024-pair seat-swapped panel completed with the batched
Rust evaluator at batch size 128: 21.873 seconds, 93.63 games/sec, 1,042 wins,
1,006 losses, no draws, no operational errors, and no illegal actions. The
95% paired score interval was [0.4907, 0.5269], so this single panel does not
establish competitive superiority. Peak GPU allocation was 226,488,832 bytes.

The historical A8 batch-size measurements remain recorded; a fresh same-manifest
32/64/128/256/512 sweep was not rerun in this campaign.

## D. A10 readiness

The D1 training path is numerically correct and meets the approximate 90
correct all-in games/sec target on the independent seed. Checkpoint/resume and
precision mismatch rejection are verified. Frozen evaluation is precision-safe
and the full panel is operationally clean. A10 may proceed only with a new
experiment-scoped dataset manifest, archive namespace, fixture manifest, and
zero-match schedule state. A10 training was not started.

Remaining limitation: the exhaustive batch-size sweep and scheduler restart
exercise should be completed before committing to a large A10 run.
