# A9.2 → A10 performance handoff

Track A9.2 does not authorize A10 training.

The safe promoted execution contract is A9 Mode A: FP32 policy-critical
execution, batched recompute gate enabled at `1e-3`, 53.21 all-in games/s on a
25-iteration, 51,200-game run. The isolated audit is offline-only.

Mode D (FP16-rounded inputs, AMP encoder, FP32 scorer/GRU) reached 90.50
games/s for eight iterations, but failed the same gate at iteration 14 of its
longer run (`1.438e-3`). It must not be used for training until its numerical
drift is resolved without changing the tolerance.

Before any future A10 proposal, repeat D diagnosis with saved failing strata,
then require a fresh long-run parity receipt, checkpoint/resume receipt, and
fixed-panel policy comparison. A9.2 supplies performance evidence and a
fail-closed baseline; it does not grant approval for a 100M-match campaign.
