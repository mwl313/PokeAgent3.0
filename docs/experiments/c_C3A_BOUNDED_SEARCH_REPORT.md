# C3a: independent bounded search core

**Status:** `C3A_BOUNDED_SEARCH_RESEARCH_ONLY / NO_VALIDATED_TEACHER / NO_LABELS / NO_DISTILLATION`

**Implementation commit:** `e64dc43641a8eabad55538f8a18c46045da8455f`

**Files changed:** `engine/src/search/core.rs`, `engine/src/search/mod.rs`,
`engine/src/search/world.rs`, `engine/tests/search_core.rs`,
`engine/tests/search_world.rs`, and this report.

## Scope and trust boundaries

C3a adds `engine/src/search/core.rs`, a CPU-only coordinator that accepts a
trusted clean-preview `SearchInformationSet`, the frozen anonymized opponent
prior, the pinned Dex, explicit independent seeds, and a bounded configuration.
It does not accept a `BattleState`, snapshot, team identifier, live RNG, PPO
model, or policy checkpoint.

For each sampled anonymized `BeliefTeam` and each retained complete root joint
action, the coordinator constructs a fresh C2c `PreviewWorld`. The hypothetical
opponent action is selected by a coordinator-only baseline that reads only that
hypothetical opponent's own `PlayerView`; its command and view never enter the
root-facing result. Each action/world pair is isolated and submitted as a
simultaneous joint choice through `SyntheticEpisode`.

The public result contains only legal root action sequences, aggregate scores,
coverage counts, truncation/abstention status, and nonsecret budget counters.
Synthetic state, sampled sheets, opponent commands, pending status, snapshots,
and future RNG remain private.

## Scoring

Terminal outcomes use the engine's public reward (`+1` root win, `0` draw,
`-1` root loss). Nonterminal positions use a deliberately simple public
heuristic: mean own observed HP fraction minus mean publicly known opposing HP
fraction, clamped to `[-1, 1]`. Unknown opponent health contributes zero rather
than being inferred. This is not a trained value function, calibrated win
probability, CTS claim, or teacher label.

Scores are arithmetic means over successful sampled-world evaluations. The
implementation never selects a different action per hidden world. A selected
action is returned only for an untruncated complete comparison; the default
preview fixture truncates the 360 legal preview joints to a small configured
prefix and therefore returns `TruncatedCandidates` with no selection.

## Explicit budget accounting

`SearchBudget` separately records candidate generation, retained candidate
evaluations, belief samples, world-construction attempts, engine transitions,
opponent-action samples, chance samples, and total computation units. The hard
limits are 16 worlds, 64 retained root candidates, and 20,000 units; default
tests use 3 worlds, 8 candidates, and 2,000 units. Construction attempts are
charged using the C2c retry diagnostics, and each simultaneous evaluation
charges two engine transitions, one opponent-action sample, and one chance
sample. Exhaustion produces structured `Abstain`/`InsufficientCoverage` rather
than silently discarding work or claiming full search.

Elapsed time and memory are not used as acceptance criteria yet; this remains a
small deterministic CPU fixture, not a benchmark.

## Verification

Added `engine/tests/search_core.rs` covering:

- deterministic repeated search with identical information, prior, config, and
  seeds;
- at least two distinct legal preview actions independently evaluated across
  multiple sampled worlds;
- hidden-truth metamorphic equivalence;
- P2 root perspective;
- structured budget exhaustion;
- rejection/insufficient coverage for an unsupported midbattle root.

The existing C2d tests continue to cover root-only output, simultaneous
episode replay, prefix legality, rollback, and pending-status isolation. The
test-only post-mutation fault hook is now behind `cfg(test)` and is exercised by
a Rust unit test after a resolved two-sided step with consumed RNG.

## Limitations and future gates

This is clean-preview-root research only. It does not reconstruct arbitrary
live midbattle states, model opponent beliefs, search chance branches beyond
the engine transition, or provide an opponent policy beyond the transparent
own-view baseline. Candidate truncation means default results are explicitly
not action-complete. No strength, generalization, CTS safety, or teacher
validity is claimed.

Before C3b or any teacher integration, require: approved information/action
schemas; complete legal-action coverage policy; independent belief and chance
sampling design; stronger public-state scoring validation; deterministic
resource accounting; root-only leakage tests; and a separately reviewed
`TeacherSample` adapter. The future adapter must carry PA3-compatible root
observation bytes, complete legal joint atoms, aggregate search targets,
confidence/provenance diagnostics, and schema versions without changing PPO
rollout storage or objectives. Track A remains entirely independent.
