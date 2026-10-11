# Track A1 validation and next milestone

Base: `8997cb4` with implementation commit `535fb56` plus the bounded
instrumentation changes in this worktree.

## Contiguous cached selection result

The PPO update calls `torch.randperm` once per epoch in `update_streaming`,
`update_ddp`, and `update_manual_allreduce`. Full-size shuffled chunks go
through `RolloutBatch.select`; the inner microbatches use `RolloutBatch.narrow`
already. The new opt-in path can only help if a cached minibatch index tensor is
an ordered contiguous range.

Three same-seed, same-configuration CPU pairs (32 rows, two epochs, four
microbatches per epoch) were run. All eight cached selections per run were
instrumented. The contiguous hit count was **0/24 with the flag on**. Model
digests, policy loss and value loss matched exactly between flag states. The
raw result is `runs/experiments/a/a1_cached_select_benchmark.json`.

The wall timings were noisy (off/on: 1.264/0.674 s, 0.744/0.733 s,
0.719/0.702 s); because the optimized branch never executed, these do not
demonstrate a speedup. The flag remains off by default and is rejected for
promotion. The instrumentation is retained so a future real workload can
report `cached_select_calls`, `cached_contiguous_hits`, and the fraction in the
per-iteration learner profile.

The A1 tests also check values, row masks/order, update parity, gradient/Adam
parity through the existing cache tests, and that the cached source tensor is
unchanged after an update. Relevant suite: **17 passed**.

## Next high-impact optimization proposal

The existing cooled dual-V100 profile attributes the largest learner cost to
backward (~748 ms in the profiled minibatch) and forward (~288 ms), with GEMM,
gather/index, copy/cast, elementwise and reduction kernels dominating the GPU
union. Pure Adam was ~11 ms. The next bounded experiment should therefore
target repeated candidate gather/index and activation materialization in the
cached model hot path, with one candidate-table reuse patch behind an opt-in
flag. It must prove identical candidate masks, logits/logprobs, loss terms,
gradients and optimizer digests on a fixed batch before a GPU timing.

Do not alter action order, branch legality, differentiability, PPO objective,
DDP collectives or production defaults. If parity passes, run three paired
GPU timings; otherwise reject the patch. No large refactor is justified by the
current evidence.
