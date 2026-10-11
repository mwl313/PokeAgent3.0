# A2-E fixed-policy evaluation foundation

Commit: `f8d9a00bd6a2d83cffda10023ceb374a5d16bc6d`.

`agent.evaluation.fixed_policy.FixedPolicyEvaluator` is deliberately outside the PPO learner. It drives the existing native engine, submits complete legal joint actions, and never creates an optimizer or increments training match counters. Fixtures contain team IDs, four-word game seeds, pair IDs and both policy seat assignments. The canonical manifest hash makes paired branches use exactly the same panel.

Supported anchors are a seeded random legal policy, a deterministic public-information heuristic (lowest canonical legal action tuple), and a frozen checkpoint policy loaded with strict state matching and `requires_grad=False`. Reports contain wins/losses/draws, operational errors, rejected/illegal actions, per-fixture outcomes, runtime, score and a deterministic bootstrap interval. The interval is descriptive and marked inconclusive when the small panel cannot resolve the practical margin; it is not a significance claim.

Tests cover manifest determinism and seat swapping, random-policy reproducibility and legality, checkpoint immutability, and checkpoint-policy legality. A bounded real Rust-engine panel of two seat-swapped games completed in 0.00388 s with random versus heuristic: 2 wins, 0 losses, 0 draws, 0 operational errors and 0 illegal actions (fixture hash `5550a250441d2db56ed0d20baf82e6c0310ef32f9e65a1cf1ee7b8c9272e1fa9`). A two-game frozen random-initial model versus heuristic panel completed in 0.48243 s with 1 win, 1 loss, 0 draws, and no errors or illegal actions.

The panel is ready as a controlled-baseline foundation, but the sample is too small for a strength conclusion. Before PPO comparisons, add a larger pinned panel (the design target is 256 paired games), save the full JSON receipts, and run the same fixture manifest for every branch and anchor.
