# A10-P5 final report

A10-P4's continuation mismatch was localized before changing checkpoint state. In normal Mode D, two independent continuous runs had the same initial state and identical first-iteration rollout digests, but differed first in PPO update numerics. The cause is V100 CUDA/PyTorch AMP nondeterminism, not lost collector seeds, optimizer state, scheduler state, or checkpoint publication.

An opt-in deterministic diagnostic mode established the restore contract: uninterrupted and checkpoint/resumed arms matched model digest `a746d0…`, optimizer digest `655115…`, rollout digests, LR clock, 4,096 committed matches, 56 optimizer steps, DDP rank parity, and recompute gates. An independent seed also passed deterministic DDP/recompute checks. Production remains in normal Mode D; its expected baseline variation is recorded as numerically bounded and is not silently converted into bit-exactness.

The A10 profile resolves the frozen 1,220-team simulator data and canonical manifest. Checkpoint scheduling remains 100,000 committed matches with three-generation retention, corruption fallback, latest-generation resume, and fail-closed storage capacity handling. The profile pilot completed 2,048 matches at 47.59 all-in games/s with zero operational errors.

A first detached launch attempt was stopped after inspection found that the parent forwarded the old manifest path to workers despite using the new simulator team file. The profile-to-worker manifest wiring was corrected and covered by regression tests; no progress from that invalid attempt is part of the production run.
