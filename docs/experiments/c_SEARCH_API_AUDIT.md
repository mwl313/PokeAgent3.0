# C0 Search API audit

**Source inspected:** `8997cb40c18481f96ba64b72d7480896510a9840` (`research/c-rust-search-teacher`)

## Result

The current engine has enough Rust-local machinery for a small, known-world
branching *test oracle*.  It does **not** have the information-set DTO, frozen
opponent prior, or safe planner boundary required for a CTS/search teacher.
Consequently C1 adds only `engine::search::test_oracle`, which is explicitly
for fully specified test fixtures and cannot be used as a teacher or to emit
training labels.

## Interfaces actually present

| Concern | Existing interface | C1 use / constraint |
| --- | --- | --- |
| Snapshot | `BattleBatch::snapshot(Handle) -> Result<Vec<u8>>` in `engine/src/batch.rs` delegates to `BattleState::snapshot`; snapshot is JSON with schema, rules digest, complete state, pending choices, and RNG. | Save one immutable baseline per oracle and restore it before and after every branch. It is a full-world payload and must never be a planner input. |
| Restore | `BattleBatch::restore(Handle, &[u8])` deserializes and validates through `BattleState::restore`. It preserves the handle generation. | Safe for one owned test slot. Restore failure is surfaced; no error is reclassified as a battle result. |
| Resolution | `StepSpec { handle, choices }` and `BattleBatch::step_batch` preflight all choices. `BattleState::step` stores a side's complete request and calls `commit` only after every requested side has submitted. | Submit P1 and P2 complete requests in one `StepSpec`. This is simultaneous decision submission; move order remains the engine's lawful `commit`/RNG behavior. |
| Legal actions | `Request::candidates(prefix)` and `Request::validate_joint(joint)` in `engine/src/actions.rs`. The tree covers preview, both normal slots, replacements, targets, switches, pass, and Mega resource use. | C1 recursively enumerates complete leaves only and validates each leaf; it never treats padded candidates as legal. |
| Player view | `BattleBatch::observe_batch` returns `PlayerView`; `BattleState::observe` builds owner data plus `Knowledge`. `engine/tests/observation_leakage.rs` already tests observation/mask non-leakage. | Read-only audit evidence only. C1 does not use player views, so it makes no hidden-information safety claim. |
| Python binding | `NativeBatch.snapshot` and `NativeBatch.restore` in `engine/src/python.rs` expose the raw full-state bytes, contrary to the dated pack's expectation that Python focused only on opaque reset/step/observation/candidates. | These bytes are unsafe for a future player-facing planner and must not be exported into any teacher path. C2 needs a restricted DTO instead. |

## State ownership and branching

`BattleBatch` owns `BattleState`s privately and has no clone-slot API. C1's
oracle owns one `BattleBatch`, one valid `Handle`, and its baseline byte vector.
Each evaluation is `restore(baseline) -> step_batch(both choices) -> snapshot
branch -> restore(baseline)`.  This is deliberately allocation-heavy and
single-fixture only; it is a correctness harness, not an optimization or a
candidate production API. Snapshot serialization includes `BattleRng`, so a
restored deterministic fixture replays the same chance stream. The snapshot
schema is `SNAPSHOT_SCHEMA = 20`; no bincode format exists here.

`BattleBatch::step_batch` validates all submitted specs before mutation, while
the two choices in one `StepSpec` are passed to `BattleState::step` in vector
order. The first becomes pending and is not resolved or exposed to the other
side; the second triggers the one simultaneous `commit`. Therefore C1 never
chooses one action after observing a resolved opponent action.

## C1 files and boundaries

- Added: `engine/src/search/mod.rs`, `engine/src/search/test_oracle.rs`, and
  `engine/tests/search_test_oracle.rs`.
- Minimal module registration in `engine/src/lib.rs` is within the ownership
  exception. No battle mechanics, observation/knowledge fields, data pool,
  Python binding, model, PPO, or configuration changes are made.
- C1 fixtures use existing legal training teams only and stay below the
  200-position budget.

## Missing / future-gated work

Before C2, define a typed player-information DTO containing only public
history/current view plus owner private data. Build an `OpponentBeliefSampler`
from a versioned prior conditioned on public reveals, not reset team IDs or
raw snapshots. Add metamorphic hidden-truth tests before exposing search to
Python. Then separately add bounded policy candidates, opponent policy,
chance sampling, score uncertainty, and fallback behavior. No information-set
sampler, policy search, strength claim, distillation, or training is included
in C0/C1.

## No-leak risk register

1. Raw snapshots contain opponents' unrevealed team/set facts and future RNG
   state; treating them as planner inputs would leak secrets.
2. `BattleBatch` is a full true-world simulator. Any future branch construction
   must sample a world from an allowed prior rather than clone the live game.
3. Legal masks are owner request data, but candidate scores can still leak if
   calculated from a live opponent state.
4. Pending submissions and serialized traces are private engine state; a
   player-facing API must not reveal them across sides.

**Status after C2b:** `C2B_BELIEF_SAMPLER_ONLY / NO_SEARCH / NO_TEACHER /
NO_DISTILLATION`. C1/C2a establish branch correctness and the typed
player-information boundary; C2b adds only a deterministic prior/sampler and
does not make a CTS-safe teacher claim.

## Pause checkpoint — 2026-10-11 Asia/Seoul

Track C was paused at the user's request so Track A can use the shared CPU/GPU
resources. No GPU work, training, teacher generation, or benchmark was
started by Track C.

Completed C0/C1 changes:

- Added this audit and the explicit secret-leak risk register.
- Added the test-only `KnownWorldTestOracle`; it restores the captured full
  snapshot before and after every paired P1/P2 branch.
- Added exactly three fixture tests: legal complete preview joint actions for
  both sides (360 ordered leaves each), deterministic snapshot restoration for
  identical simultaneous submissions, and branch isolation plus atomic
  rejection of an incomplete action.
- Preserved the local Poltergeist fix at source commit `8997cb4`; its existing
  regression test passed during the partial suite.

Verification completed:

```text
PATH=/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin:$PATH \
  cargo test --test search_test_oracle
# 3 passed, 0 failed (0.83 s)
```

The first full-suite attempt was deliberately interrupted (exit 130) to
release CPU for Track A. After checking that the 80-core machine had ample
headroom, one reduced-priority repeat completed successfully:

```text
PATH=/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin:$PATH \
  nice -n 10 cargo test
# passed; the differential turns binary passed all 8 tests in 1289.27 s
```

This includes the existing observation-leakage and Poltergeist regressions,
the new oracle tests, snapshot tests, state-batch tests, and every reference
fixture. The required lint also passed:

```text
PATH=/home/minidc/.rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin:$PATH \
  nice -n 10 cargo clippy --all-targets -- -D warnings
# passed
```

`cargo fmt --check` remains unavailable because this local toolchain's cargo
installation reports no `fmt` subcommand; no system component was installed.
The staged diff passed `git diff --cached --check` and was committed as
`0790dcf` (`test(engine): add known-world branch oracle`). Stop for review; do
not begin C2c or modify shared interfaces. See
`c_C1_5_C2A_INFORMATION_BOUNDARY.md` and
`c_C2B_BELIEF_PRIOR_REPORT.md` for the continuation reports.
