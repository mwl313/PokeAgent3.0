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

## A10-P1 storage and engine gate

- [x] Restore project-local Rust 1.90 toolchain without system-wide changes.
- [x] Add generation-based three-generation checkpoint retention and disk guard.
- [x] Test atomic publication, corruption fallback, missing rank, insufficient space, and protected generation behavior.
- [ ] Verify exact 1,220-team reachable native closure and whole-pool natural completion.
- [ ] Authorize A10 first training run.

Current verdict: **BLOCKED / NOT VERIFIED for engine; storage infrastructure bounded-ready**.

## A10-P2 status

- [x] Rust 1.90 toolchain verified and release pool probe built.
- [x] 1,220-team initialization/natural trajectory probe: 1,220/1,220 complete, zero reported blockers.
- [x] 83 new-team targeted both-seat probe: 332/332 complete, zero operational errors.
- [x] Generation storage integrated behind an opt-in launcher flag.
- [ ] Exhaustive exact-pool closure/action differential validation and fully varied all-team natural campaign.

Current verdict: **BLOCKED for exact-pool engine PASS; storage integration bounded-ready**.

## A10-P3 acceptance (2026-10-13)

- [x] One-time M-B/M-C refresh captured; no new eligible teams; P0 pool frozen at 1,220.
- [x] Varied all-pool native campaign: 19,520/19,520 completed, zero operational errors.
- [x] Rolling checkpoint publication tested at live iteration boundaries; three generations retained with disk guard.
- [x] Bounded 4,096-match two-V100 PPO smoke passed recompute and rank digest gates.
- [ ] Global full move/ability/dynamic closure gate: blocked (29 moves, 23 abilities, 11 callers).
- [ ] Production launch: requires coverage closure and explicit authorization.

## A10-P4 final blocker status (2026-10-13)

- [x] Pool-scoped 1,220-team reachability audit with bounded dynamic closure evidence.
- [x] A10 production profile resolves the frozen dataset and numerical contract.
- [x] 100,000 committed-match checkpoint scheduler and latest-generation resume path implemented.
- [x] Storage-capacity errors fail closed; three-generation retention and corruption fallback tested.
- [ ] Interrupted-versus-uninterrupted final-state parity: **BLOCKED** (model/optimizer digest divergence).
- [ ] A10 production launch: blocked until continuity discrepancy is explained and corrected.

## A10-P5 continuity closure (2026-10-13)

- [x] Independent continuous controls isolate baseline CUDA/AMP update nondeterminism.
- [x] Deterministic diagnostic mode proves bit-exact checkpoint continuation through 4,096 matches.
- [x] Independent seed deterministic DDP/recompute check passes.
- [x] Normal production mode documented as numerically bounded baseline variation.
- [x] A10 profile pilot passes with 1,220-team simulator data.
- [ ] Long-horizon launch process status recorded after final commit.
