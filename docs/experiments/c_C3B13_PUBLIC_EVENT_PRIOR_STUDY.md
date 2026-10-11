# C3b-1.3: public-event conditioning and belief-prior diversity

**Status:** `C3B13_PUBLIC_EVENT_BELIEF_RESEARCH_ONLY / NO_VALIDATED_CTS_TEACHER / NO_LABELS / NO_DISTILLATION`

**Starting point:** `460e74b03ce6543fc0083c0b8c1a219be1ce005c`

**Implementation commits:**

- `6f8d736806da89b668565f52a64464951e0d67f0` — bounded public-event proposal, prior diversity audit, and controlled study
- `4286d1ecf3a56e1e18f43592598f6e92cbbe2a8e` — preserve a tracker unchanged on public-proposal zero support

## Decision

Do **not** start C3b-2. The new proposal leaves the authoritative exact
`SearchInformationSet` comparison intact, but it does not retain support past
the second combat boundary in the bounded independent-oracle study. It also
costs substantially more transition work and turns failures into explicit
proposal-zero-support outcomes rather than demonstrating reliable posterior
coverage.

No suitable independent legal full-team source exists locally for a diversified
research prior. The study therefore does not create a synthetic correlated team
artifact or claim an original-versus-diversified-prior improvement.

## Frozen C3b-1.2 baseline

The pre-existing C3b-1.2 matrix was rerun unchanged. It has 64 configuration
rows, 112 attempted observation boundaries, and 55 collapsed boundaries.
`Supported` below includes `InsufficientCoverage` with a retained root-safe
conditioned observation.

| Actual build / proposal | Rows | Preview | Combat 1 | Combat 2 | Collapse / attempted | Rejected comparisons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| In-prior / FirstLegal | 16 | 16 | 8 | 5 | 11 / 40 | 110 |
| In-prior / SeededSampled | 16 | 0 | 0 | 0 | 16 / 16 | 120 |
| Out-of-prior / FirstLegal | 16 | 16 | 8 | 4 | 12 / 40 | 116 |
| Out-of-prior / SeededSampled | 16 | 0 | 0 | 0 | 16 / 16 | 120 |
| **Total** | **64** | **32** | **16** | **9** | **55 / 112** | **466** |

The baseline mismatch categories remain: event-history window 466, displayed
HP 406, public opponent knowledge 300, public-event-justified action-hypothesis
226, own state/HP 184, and public field effects 64. This is a frozen diagnosis
control, not 64 independent matches: it repeats two actual battle schedules
across sides, build conditions, proposals, and particle counts.

For the C3b-1.3 held-out rows only, a separate legality-checked fixture moves
one permitted allocation point within a complete team and verifies the result
is absent from the frozen prior. It is deliberately separate from the legacy
C3b-1.2 fixture so the latter's frozen result is not altered.

## Public-event semantics audit

The proposal consumes only `SearchInformationSet.public`, copied from the
root player's `PlayerView`: turn, outcome, and audience-filtered `Knowledge`.
Candidate simulation happens in a particle-private episode. Each candidate is
restored before another is considered; the selected candidate then still has to
pass full exact DTO equality in the normal tracker.

| Class | Facts and treatment |
| --- | --- |
| Authoritative exact public constraints | Ordered bounded `SemanticEvent`s: visible move/effect identity and subject, public target, switch, revealed species, Mega/resource activation, status, boosts, item/ability revelations, faints, and field/side effects; public preview/active selection; revealed move repertoire; turn and terminal outcome. The entire public `Knowledge` value is compared exactly. |
| Quantized public constraints | `HealthDisplay` is the public percentage (`numerator/100`) plus its documented 20%/50% boundary colour. It is compared exactly as that public display; this never recovers exact HP. |
| Ambiguous/insufficient | A visible move/event does not reveal an entire simultaneous submitted joint action, unshown target intent, unshown move, ability, item, or a causal hidden choice. The proposal makes no such inference. |
| Private/prohibited | Actual-oracle team IDs and full sets, `BattleState`, snapshots, pending opponent command/submission state, particle private observations, and future or actual RNG. None enters the proposal input, diagnostics, or result DTO. |

`SemanticEvent` is audience-filtered and holds only public entity indices/effect
catalogues and an optional `HealthDisplay`. Its bounded history has capacity 24.
This mode does not introduce a relaxed event likelihood: public equality is
exact, and complete `SearchInformationSet` equality remains the final condition.

## Experimental proposal contract

`OpponentProposal::PublicEventConditionedV1` is opt-in, research-only, and is
not the default tracker policy. At a new boundary it proceeds in this order:

1. Receive the already served root observation from the independent oracle.
2. Enumerate at most eight complete, prefix-valid legal opponent joint actions
   for the particle's own opponent request, using a coordinator-local,
   deterministic seed.
3. Simulate each candidate privately, project only the root `PlayerView`, test
   its public DTO for exact equality, and restore the private episode.
4. Select the deterministic first compatible legal candidate; then run the
   usual simultaneous submission and full exact `SearchInformationSet` filter.

There are no importance weights, so effective sample size is not applicable.
The proposal consumes at most `5 + 2 * 8 = 21` accounting units per particle
per update (versus 5 for existing proposals), with the existing 2,000-unit hard
cap. No hidden battle RNG is read or advanced by proposal seeding.

If every bounded candidate conflicts with authoritative public evidence, the
result is `ProposalZeroSupport`; if the required bound exceeds the caller's
budget, it is `BudgetExhausted`. A total zero-support result preserves the
particle tracker exactly so a later approved proposal policy may retry; it is
not converted to a success or an irreversible observation collapse.

## Prior diversity audit

The original `FrozenOpponentPrior` is unchanged. Its source is
`engine/data/training-teams.json`, whose SHA-256 is
`48a050d007bf43e2ba4e0cbea0757bffe1565cfaab9909b7ccb3eb78736c8073` and
whose prior schema is 1. The corresponding pinned v3 manifest records 1,137
unique legal full teams (1,136 inherited v2 teams plus one approved user team;
source-input SHA-256
`3f3bc36d54e01a553c4026665ad7f7cb207dff7c25e77c13859a355821f12344`).

For the preview fixture, the deterministic root-safe diversity report finds two
compatible candidates, two unique complete-sheet hypotheses, and zero exact
duplicates. It emits only aggregate counts and source provenance, not team IDs
or private sheets. The repository has no disjoint approved legal source,
dev/final split, or held-out pool: the manifest explicitly says held-out team
generalization is unavailable. Roster permutations or independently mixed
private members would violate the required within-team correlations, so no
"diversified" research prior was fabricated.

## Paired controlled comparison

This small CPU-only test uses the same independent actual fixtures for every
proposal condition: P1 and P2 roots, in-prior and legal held-out opponents, two
actual schedules (801 and 811), and two/four particles. Each of the 16 rows
attempts preview plus up to two combat observations under exact conditioning.
`PublicEventConditionedV1` is paired against both existing proposal policies
on precisely those fixtures.

| Proposal | Rows | Boundaries attempted | Preview supported | Combat 1 supported | Combat 2 supported | Collapse | Proposal-zero-support | Public candidate attempts | Compatible candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FirstLegal | 16 | 40 | 16 | 8 | 0 | 16 | 0 | 0 | 0 |
| SeededSampled | 16 | 16 | 0 | 0 | 0 | 16 | 0 | 0 | 0 |
| PublicEventConditionedV1 | 16 | 40 | 16 | 8 | 0 | 0 | 16 | 864 | 60 |

The public-event proposal reaches the same preview/combat-one support as
FirstLegal and no combat-two support. Its 16 terminal failures are explicitly
proposal zero support, not retained belief mass; therefore the apparent
collapse-count reduction is **not** an improvement. It performed 864 bounded
candidate simulations across 40 attempted boundaries (21.6 attempts per
attempted boundary) and found 60 public-compatible candidates before final
exact filtering. No operational transition error or invalid proposal occurred
in this matrix. It does not count an observed opponent move as knowledge of a
complete opponent command; public move/switch/resource coverage is represented
only through the exact event and knowledge projection above.

The 16 configuration rows are correlated repetitions of eight actual
side/build/schedule fixtures, not independent samples. A statistical confidence
interval would misleadingly treat particle counts and paired policy runs as
independent battles. The appropriate uncertainty statement is the exact paired
count table above; it makes no population-level reliability claim.

## Correctness and boundary gates

- Exact replay with the event proposal succeeds on a constructed positive
  control, while full DTO equality remains required.
- A deliberately contradictory public species produces
  `ProposalZeroSupport`, preserves the root observation and particle state, and
  then successfully replays when given the unmodified trusted target.
- A metamorphic test changes a genuinely legal opponent hidden build while the
  root-visible observation is identical; public-event proposal results remain
  identical.
- Existing transaction tests retain rollback for invalid root actions and later
  operational/conversion faults. Candidate trials additionally restore their
  private episode before selection or error propagation.
- Prior sampling, proposal diagnostics, and root results carry no actual
  opponent ID, private set, oracle command, snapshot, or future RNG. The
  root-facing diagnostic is aggregate-only.

## Verification

Passed, CPU-only commands:

```text
cargo test --lib search::particles::tests -- --nocapture
cargo test --test search_core --test search_world --test search_belief \
  --test search_information_set --test search_test_oracle
cargo clippy --all-targets -- -D warnings
rustfmt --edition 2021 --check src/search/particles.rs \
  src/search/world.rs src/search/belief.rs src/search/information_set.rs \
  tests/search_belief.rs
git diff --check
```

The particle unit matrix completed in about 13 seconds locally. The full
differential corpus was not run. No GPU workload, Track A/main change,
production battle-mechanics change, Python integration, teacher-label
generation, distillation, PPO, or training was performed.

## Required before any C3b-2 review

Require a separately approved, disjoint legal team source with a manifest and
held-out split; a larger independent actual-fixture suite including replacement
and resource branches; a public-event likelihood justified field by field if
any comparison is relaxed; calibrated action/chance coverage; and a paired
result that improves retained exact support rather than merely relabeling
failure. Until then, keep the exact tracker as the only credible condition and
prefer a different belief architecture over increasing particle count.

**Final gate:** `C3B13_PUBLIC_EVENT_BELIEF_RESEARCH_ONLY / NO_VALIDATED_CTS_TEACHER / NO_LABELS / NO_DISTILLATION`
