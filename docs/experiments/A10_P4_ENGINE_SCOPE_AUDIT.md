# A10-P4 engine scope audit

The native `coverage_report` was rerun against the simulator-ready 1,220-team file using Showdown commit `14546894d86f9589ac11130c510bbe73b6968665` and `gen9championsvgc2026regmc`. It reports 29 globally blocked moves, 23 globally blocked abilities, and 11 conservative dynamic-closure blockers, but **zero pool blockers** for moves, abilities, and items.

A machine-readable mapping is in `A10_P4_ENGINE_SCOPE_AUDIT.json`. Direct frozen-team sets contain no globally blocked move or ability. The pool does contain bounded dynamic seeds (`Sleep Talk`, `Role Play`, `Skill Swap`, `Entrainment`, and item-transfer moves); their target universes are the moves, abilities, and items actually present on the frozen pool, all of which are executable. The blocked callers Copycat, Gastro Acid, Instruct, Magic Room, Transform, Worry Seed, and blocked ability seeds are absent from the pool. Thus the pool-scoped engine gate is a bounded **PASS**, while the global Regulation M-C gate remains **NOT READY**.

This is not a claim of full-regulation completeness. It is a proof limited to the frozen 1,220-team dataset and the audited dynamic seed universe. Any future dataset or mechanic that expands those universes requires a new scope audit.
