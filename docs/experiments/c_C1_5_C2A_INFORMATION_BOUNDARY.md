# C1.5 / C2a combat verification and information boundary

**Branch:** `research/c-rust-search-teacher`
**Base:** `986b9f8`
**Commit:** `a0579e1`
**Status:** `INFORMATION_BOUNDARY_ONLY / NOT_A_TEACHER`

## Scope

This milestone extends the known-world oracle only for bounded Rust correctness
tests and adds a typed DTO whose sole source is an audience-filtered
`PlayerView`. It does not add opponent sampling, search scoring, policy calls,
Python integration, distillation, or training.

## Files changed

- `engine/src/search/test_oracle.rs` — prefix candidate access for tests.
- `engine/tests/search_test_oracle.rs` — four real combat-turn tests in
  addition to the three preview/branch tests from C1.
- `engine/src/search/information_set.rs` — versioned `SearchInformationSet`,
  `PublicInformation`, and `OwnInformation` DTOs.
- `engine/src/search/mod.rs` — module registration.
- `engine/tests/search_information_set.rs` — hidden-truth metamorphic and
  schema tests.
- `docs/experiments/c_C1_5_C2A_INFORMATION_BOUNDARY.md` — this report.
- `docs/experiments/c_SEARCH_API_AUDIT.md` — status and verification update.

## C1.5 coverage

The combat helper advances both sides through a legal team preview, captures
that state as the oracle baseline, and then uses complete request-tree leaves
for both sides. Tests cover:

1. two-slot complete simultaneous combat choices, including a non-empty
   prefix-dependent second-slot candidate set and explicit non-default target
   locations;
2. voluntary switch choices on both sides, with post-branch snapshots restored
   through `BattleState::restore` and active roster destinations checked;
3. identical complete combat snapshots/RNG reproducing identical result and
   snapshot bytes;
4. distinct action combinations producing distinct branch snapshots without
   contaminating a later replay;
5. incomplete combat actions rejected while both sides remain at the baseline;
6. Mega resource legality checked whenever the bounded fixture exposes a Mega
   candidate (at most one Mega resource in a complete joint action).

Chance-sensitive resolution is covered at the determinism level: the same
   snapshot and complete choices replay the same engine RNG stream and result.
No dedicated random-damage/status-effect fixture was added because the selected
existing legal team fixture does not guarantee a specific chance-bearing move;
that remains a bounded future fixture gap. No dedicated Mega-form fixture is
claimed when the selected team/request does not expose a Mega candidate.

## C2a DTO field visibility audit

`SearchInformationSet::from_player_view(&PlayerView)` is the only constructor.
It has no `BattleState`, snapshot, team, trace, opponent-command, or RNG
parameter.

| DTO section | Fields | Visibility justification |
| --- | --- | --- |
| `schema_version` | `u16` | Protocol version, not game state. |
| `public` | `turn` | Public decision boundary. |
| `public.knowledge` | `Knowledge` | Audience-filtered public events, public Pokémon facts, public effects, and explicit `Known` masks produced by the engine. Unknown values are not filled. |
| `public.outcome` | `Outcome` | Public terminal/truncated/error/winner status at the current view boundary. |
| `own.side` | `SideId` | Actor identity needed to interpret own/other indexing. |
| `own.state` | `OwnSideView` | Exact own six-Pokémon state, own selected order/active slots, and own Mega use are player-private and permitted. |
| `own.request` | `Request` | The actor's served legal request, including prefix-dependent moves, targets, switches, replacement, and resource availability. It contains no opponent request. |

The DTO uses deterministic `serde_json` bytes, `BTreeMap` ordering inherited
from `Knowledge`, an explicit schema version, schema validation, request
validation, and `deny_unknown_fields` on DTO envelopes. There is no decoder
that reconstructs hidden worlds; unsupported values are unavailable rather than
invented.

## No-leak proof

The metamorphic test creates two legal states with identical P1 visible view
and seed but changes P2's team ID, EV allocations, IVs, and nature. It asserts
the complete `PlayerView` values are equal, the DTOs are equal, serialized bytes
are byte-identical, and the hidden team ID does not occur in the bytes. This
test never passes either raw snapshot into the DTO; snapshots are not accepted
by its type signature.

## Verification

```text
PATH=/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin:$PATH \
  cargo test --test search_test_oracle --test search_information_set
# 9 passed, 0 failed

PATH=/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin:$PATH \
  cargo clippy --all-targets -- -D warnings
# passed

/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin/rustfmt \
  --check engine/src/search/information_set.rs \
  engine/src/search/test_oracle.rs engine/src/search/mod.rs \
  engine/tests/search_test_oracle.rs engine/tests/search_information_set.rs
# passed
```

The expensive full differential corpus was not rerun; no battle mechanics were
changed. Existing C0 full-suite results remain recorded in the audit.

## C2b requirements

Before any search policy work, C2b must define a versioned opponent hidden-world
prior conditioned only on this DTO's public information and documented public
reveals. It must reject inconsistent samples, record missing prior coverage,
keep own private state separate, and add metamorphic tests for hidden truth and
revealed-detail changes. Only after that gate may a bounded Rust sampler be
considered. This branch does not make a teacher-strength or CTS-safety claim.
