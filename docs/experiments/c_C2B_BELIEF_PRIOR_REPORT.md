# C2b CTS opponent belief prior report

**Branch:** `research/c-rust-search-teacher`
**Base:** `6922d9b`
**Commit:** `c3d04e3`
**Status:** `C2B_BELIEF_SAMPLER_ONLY / NO_SEARCH / NO_TEACHER / NO_DISTILLATION`

## Scope and source

C2b adds an offline frozen prior and deterministic sampler only. The source is
the local `engine/data/training-teams.json`:

- source schema: `pa3-team-record-v2`
- format: `gen9championsvgc2026regmc`
- source bytes: 1,144,690
- source SHA-256: `48a050d007bf43e2ba4e0cbea0757bffe1565cfaab9909b7ccb3eb78736c8073`
- entries: 1,137 complete six-member sheets
- weighting: uniform; no reliable frequency metadata was available

`FrozenOpponentPrior::from_training_json` parses the exact bytes, validates
every sheet against both the engine's structure validator and pinned M-C
`legality::validate_team`, records the source hash/schema/format/Dex digest and
generation provenance, then discards every source `Team.id`. Runtime
`BeliefTeam` values contain only the six correlated `TeamSet` records.

Reproduction procedure:

1. Load the pinned `Dex` from `engine/data`.
2. Read the exact frozen `training-teams.json` bytes.
3. Call `FrozenOpponentPrior::from_training_json(&dex, bytes)`.
4. Persist `manifest()` and `to_bytes()` if an offline artifact is needed.

## Filtering contract

`FrozenOpponentPrior::sample(info, seed, count)` accepts only a
`SearchInformationSet`, explicit seed, and bounded sample count (1–4096). It
filters by opponent preview species and explicit fixed source gender. Empty
source gender remains compatible with either observed sex because the engine
randomizes it at reset. Unknown species impose no constraint.

Known opponent move, ability, and item values are deliberately *not* used as
initial-set constraints: the current `Knowledge` schema does not identify
original-set provenance separately from copied/transformed moves, ability
changes, Mega/form changes, or item swaps. Each such observation is recorded in
`unsupported_observations`; no hidden value is invented.

Sampling is without replacement using a private xorshift stream seeded only by
the caller's explicit `u64`. Outputs contain anonymized `BeliefTeam` values and
structured diagnostics: candidate counts, uniform normalized weights, effective
candidate count, rejection reasons, unsupported observations, and explicit
`OutOfSupport` when filtering removes every candidate. There is no fallback to
the live battle state.

## Closed-pool identification bias

Exact six-species preview matching over the 1,137 source sheets produced:

- singleton matches: 817 / 1,137
- singleton fraction: 0.718558488
- mean matching candidates: 2.80123
- zero-match previews: 0

This is only a closed-pool coverage statistic. Because PPO opponents are drawn
from the same pool, it can create a misleading closed-sheet-like advantage. No
source-sheet-disjoint or novel-preview-roster cohort exists in this repository,
so both evaluations are explicitly **not available** and no CTS-strength claim
is made.

## Tests and checks

```text
PATH=/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin:$PATH \
  cargo test --test search_belief --test search_information_set --test search_test_oracle
# 15 passed, 0 failed

PATH=/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin:$PATH \
  nice -n 10 cargo clippy --all-targets -- -D warnings
# passed

/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin/rustfmt \
  --edition 2024 --check engine/src/search/belief.rs \
  engine/tests/search_belief.rs engine/src/search/mod.rs
# passed
```

The tests cover prior hash/schema/count/legality and ID exclusion, deterministic
seeded sampling, unknown-versus-known species, explicit out-of-support,
ambiguous move/ability/item diagnostics, malformed source input, singleton
coverage reporting, and hidden-truth metamorphic equivalence. Existing C0,
C1, C1.5, and C2a tests remain unchanged and were included in the targeted
oracle/information test command.

## No-hidden-state proof

The sampler's public API has no `BattleState`, snapshot, trace, team ID,
pending command, or battle RNG parameter. It reads only the DTO's public
`Knowledge` masks and own-side fields through `SearchInformationSet`. The
metamorphic test changes the actual opponent team ID, EVs, IVs, and nature while
keeping the player's `PlayerView` identical; serialized information, samples,
and diagnostics remain identical for the same explicit seed.

## C2c requirements

Before world construction, C2c must define a separate typed sampled-world
contract: each hypothesis must be derived from an anonymized `BeliefTeam`,
publicly revealed state, and documented lawful uncertainty only; it must reject
unsupported dynamic reconstruction, clear secret caches, carry prior/seed
provenance, and never mutate or expose the live battle snapshot. C2c must add
branch parity and sampled-world visibility tests before any policy evaluation.

The future C3 interface must accept this DTO plus an immutable, provenance-
checked frozen policy checkpoint and preserve the packed action tuple
`(kind, own_slot, move_slot, target_location, switch_destination, resource)`
with prefix-dependent legal expansion. Checkpoint loading, Python integration,
policy inference, search scoring, and distillation are explicitly out of scope.

## Files and commit

Implementation files: `engine/src/search/belief.rs`,
`engine/src/search/mod.rs`, and `engine/tests/search_belief.rs`. Documentation:
this report and the updated `c_SEARCH_API_AUDIT.md`. The verified commit is
`c3d04e3`.
