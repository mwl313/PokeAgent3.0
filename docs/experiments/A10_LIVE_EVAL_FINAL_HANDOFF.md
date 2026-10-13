# A10 live evaluation handoff

The live trainer remains uninterrupted. A10 checkpoint validation, immutable model-only snapshot extraction, anchor-0 scheduling, and CPU-only 8-game evaluator smoke are verified. The evaluation timer is active and safely queues milestone jobs; it does not consume training GPUs.

The first real A10 strength result is **not yet available**. The 102,400-match checkpoint is the first exact immutable candidate. The 50k/100k/150k schedule records are delayed references to that true checkpoint clock and must not be interpreted as exact historical models. Full 256-game evaluation remains deferred to a controlled CPU window because the active run occupies both V100s and a single 8-game smoke is not enough to establish training impact.
