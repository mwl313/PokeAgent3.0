# V4 manual all-reduce vs DDP comparison (M1 / D2)

Implementation: `PPOLearner.update_manual_allreduce` with
`flatten_gradients` / `assign_flat_gradients` (fixed parameter order, exact
zeros for missing grads, one FP32 `all_reduce(SUM)` per optimizer step). The
same fixed-step micro protocol as DDP; the rank loss has **no** world-size
factor because SUM replaces DDP's average.

## 1. Parity vs the single-GPU oracle (real 4,096-row fixture, 2 GPUs)

| panel | ‖Δweight‖∞ | ‖Δ moment‖∞ | steps | LR | epoch-KL Δ | grad-norm rel Δ |
|---|---:|---:|---:|---|---:|---:|
| manual fp32, 3 steps | 1.19e-07 | 1.0e-09 | 3/3 | equal | 2.0e-09 | 8.5e-08 |
| DDP fp32, 3 steps | 1.19e-07 | 8.1e-10 | 3/3 | equal | 9.9e-10 | 0.0 |
| DDP fp16, 3 steps | 3.95e-06 | 8.0e-06 | 3/3 | equal | 5.9e-07 | 1.0e-02 |

`tests/agent/test_manual_allreduce_parity.py` (3 tests) additionally proves the
SUM of two uneven-slice local gradients equals the single-process objective on
the union of the same rows, and that the flatten/assign round trip covers
missing grads.

## 2. Equal-total A/B (2,048 natural games, 1,024/rank, 3 repeats each)

| executor | games/s runs | median | mean | update wall | flat all-reduce |
|---|---|---:|---:|---:|---:|
| DDP | 24.67 / 25.92 / 24.67 | 24.67 | 25.09 | 58.5–61.5 s | — |
| manual SUM | 24.67 / 25.60 / 25.60 | 25.60 | 25.29 | 59.1–61.4 s | 10 ms / 35.0 MB per step |

Both executors: 0 operational errors, 0 skipped steps, 56 steps, model and
optimizer digests byte-identical across ranks.

## 3. Decision

The difference (manual +0.8 % mean, DDP +3.7 % median) is inside the ±4 %
repeat spread, so the plan's rule applies: keep the simpler, more stable
implementation. **DDP is the default**; the manual executor remains selectable
(`scripts/run_ddp_ppo.py --executor manual`) and is covered by parity tests.
NCCL is not the bottleneck at this scale: one flat 35 MB all-reduce costs
~10 ms inside a ~1 s update step.
