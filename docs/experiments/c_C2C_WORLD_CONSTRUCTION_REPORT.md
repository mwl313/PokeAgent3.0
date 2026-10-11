# C2c-1 sampled preview-world construction

**Status:** `C2C_PREVIEW_WORLD_ONLY / MIDBATTLE_NOT_VERIFIED / NO_SEARCH / NO_TEACHER / NO_DISTILLATION`

## Scope

This milestone adds a restricted, test-oriented constructor for a clean team
preview. It accepts only a `TrustedSearchInformationSet` created directly from
an audience-filtered `PlayerView`, one anonymized `BeliefTeam`, the pinned Dex,
and an explicit independent seed. The constructor validates both reconstructed
teams, creates a fresh `BattleState::reset`, retries only the engine's
randomized preview-gender draw with a bounded deterministic seed schedule, and
accepts a world only when its player projection exactly equals the supplied
information set. The synthetic state is private to `PreviewWorld`; no snapshot,
raw `BattleState`, opponent IDs, pending commands, live RNG, or hidden fields
are exposed.

The retry bound is 256 independent seeds. Exhaustion returns structured
`ObservationMismatch` diagnostics rather than fabricating unknown values.

`SearchInformationSet::from_bytes` remains an untrusted transport decode. It
cannot be passed to the constructor: the public API requires the private
`TrustedSearchInformationSet` wrapper made from a live `PlayerView`.

## Preview identity audit

The engine's clean-preview path assigns roster member `j` to public entity `j`
for the owner and `j + 6` for the opponent. The same stable roster order is
used by `Request::preview_roster` and the preview knowledge array. Therefore
ordered slot matching is justified for this phase, but it is not treated as a
general battle identity theorem.

Across the frozen 1,137-sheet pool:

| Matching key | Singleton | Zero-match | Mean candidates |
| --- | ---: | ---: | ---: |
| Ordered six-slot species | 817 / 1,137 (0.718558488) | 0 | 2.80123 |
| Order-independent species multiset | 793 / 1,137 (0.697449428) | 0 | 3.12665 |

The order-independent calculation is an audit statistic only. The sampler keeps
ordered matching because that is the pinned engine preview identity contract.

## Tests

Added `engine/tests/search_world.rs` for:

- P1 and P2 role placement, legal fresh preview construction, exact observation
  reproduction, and deterministic repeated construction;
- identical player-safe input plus seed producing identical player projections
  and diagnostics (hidden truth is not an input);
- rejection of a mid-battle/unsupported own view and malformed sampled sheet
  before any seed attempt.

The existing oracle tests still cover complete simultaneous joint actions,
prefix-dependent move/target candidates, voluntary switching, deterministic
restoration, chance/resource branches, branch isolation, and error recovery.

## Mid-battle feasibility gap (C2c-2 design only)

Current `SearchInformationSet` can carry the player's exact current side and
audience-filtered public knowledge, including HP/status/boosts/active slots,
revealed moves, public effects, Mega-used flags, events, and the served legal
request. It cannot safely reconstruct a complete world for arbitrary mid-turn
search: hidden opponent current HP/PP/status durations, private volatiles and
resource state, copied/transformed moves and abilities, item/ability changes,
switch/transform history, pending simultaneous commands, event-order queues,
and the live RNG stream are not player-visible. A future milestone must either
add explicitly justified public projections and bounded history/RNG contracts,
or reject such states. No production mechanics or observation fields were
changed here.

## Verification and limitations

- The C2c constructor is preview-only; no combat-turn sampled world is built.
- No belief sampling changes, policy/search scoring, Python teacher, labels,
  distillation, PPO, or training integration were added.
- Track A's A3 report and fixed-policy implementation were not present in this
  isolated worktree, so no model or GPU was loaded.
- Run targeted Rust tests, clippy, and rustfmt before commit. Do not run the
  expensive differential corpus without review approval.

## C3 readiness contract (future, not implemented)

Before any search-policy work, Track C requires an approved frozen checkpoint
SHA/schema, the PA3 observation encoding, packed joint-action atoms
`(kind, own_slot, move_slot, target_location, switch_destination, resource)`,
prefix-dependent legal joint-action semantics, and fixed CPU fixtures. This
commit intentionally stops before C2c-2/C3.
