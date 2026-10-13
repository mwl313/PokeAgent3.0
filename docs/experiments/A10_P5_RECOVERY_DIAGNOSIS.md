# A10 P5 recovery diagnosis

The production run stopped at iteration 186 before the PPO update because rank0's stratified sampled-vs-recomputed logprob gate measured `1.486e-03` against the unchanged Mode D tolerance `1e-03`; rank1 measured `2.384e-05` and passed. The original rollout rows were not persisted, so the exact failing row cannot be reconstructed.

The newest complete two-rank checkpoint is iteration 171 at **700,416 committed matches**. Both rank files were SHA-256 verified against the manifest. The manifest matches source `0b5d930`, A10 dataset hash `1aa64507…fca309bb`, world size 2, PPO-2, and complete boundary metadata.

A separate diagnostic worktree at the exact recorded source SHA restored that checkpoint with the original A10 configuration and ran 15 bounded two-rank iterations (61,440 diagnostic matches), reaching the historical failure clock range through 761,856. All 15 recompute gates passed, all DDP digests matched, and no operational errors occurred. One diagnostic optimizer update was skipped at diagnostic iteration 7; it was not a recompute failure. The historical discrepancy therefore remains **not reproduced** and cannot be claimed fixed by a code change. The evidence is consistent with a rare CUDA/AMP numerical outlier or nondeterministic execution event, but does not prove that cause.

Recovery is safe only by resuming from 700,416 with the exact recorded source SHA and unchanged gate/configuration. The failed-run logs and checkpoint generations remain preserved.
