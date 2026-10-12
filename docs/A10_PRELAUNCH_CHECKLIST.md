# A10 prelaunch checklist

Before any A10 training, provide the final dataset manifest/hash and team count,
engine compatibility report, initialization choice (random or exact warm start),
experiment identity, evaluation archive and fixture namespace, zero-match
schedule state, checkpoint retention budget, and explicit user authorization.
The pending profile must fail closed until all fields are supplied. Do not infer
model initialization from a zeroed schedule clock.

## A10-P0 incremental pool gate (2026-10-12)

- [x] Capture live M-C/M-B source snapshots and preserve raw source hashes.
- [x] Import only EVs=Yes rows beyond the MC463/MB861 boundary.
- [x] Validate six-member records with the pinned Showdown reference.
- [x] Preserve v3 predecessor and quarantine invalid/ambiguous rows.
- [ ] Pool-scoped Rust handler/closure and action differential checks.
- [ ] Whole-pool bounded natural validation with zero operational errors.
- [ ] User authorization to start A10 PPO training.

Current verdict: **BLOCKED / NOT VERIFIED**. No A10 training started.
