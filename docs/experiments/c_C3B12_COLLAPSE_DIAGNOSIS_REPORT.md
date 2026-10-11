# C3b-1.2: belief-collapse root-cause analysis

**Status:** `C3B12_BELIEF_DIAGNOSTICS_RESEARCH_ONLY / NO_VALIDATED_CTS_TEACHER / NO_LABELS / NO_DISTILLATION`

**Starting commit:** `c320126a627e43bacccaca0afdd9d575cf774a2a`

**Implementation commits:**

- `10d0ab757ff82b394d890d490c5964b2bb172523` — transactional update and root-safe diagnostics
- `0c99bd433411c53c05c29610daeec5a6d11e2980` — benchmark denominators
- `f5bcda908035513e6a9cf104b7beb625cdce13fd` — proposal/chance diversity measurements
- `c6b4bb862a98eaaccf03e486f2562473d9018a51` — aggregate mismatch breakdown

## Decision

Do **not** start C3b-2 yet. Exact conditioning remains the only enabled
research mode. The bounded independent-oracle experiment confirms rapid
multi-turn support loss, but it does not isolate a validated compatibility
relaxation or a sufficiently representative belief/proposal model.

The next useful experiment is a reviewed, public-event likelihood study with
disjoint fixtures and a larger frozen prior. It must demonstrate preservation
of every authoritative player-visible fact before any field-specific matching
mode is enabled. No approximate matching was added here.

## Confirmed defect and fix

`PersistentParticleBelief::update_with_proposal` previously drained its private
particle vector before the update committed. If an unexpected observation DTO
conversion error occurred after one or more earlier particles had been
rejected, it restored only the still-owned subset; earlier rejected or failed
particles were lost.

The update is now strongly transactional:

- every particle is checkpointed before mutation;
- particles remain in their original ordered collection while every transition
  and comparison runs;
- rejected particles are merely marked and are removed only at the final
  successful commit;
- an internal proposal/engine/conversion failure restores every checkpoint,
  including state, RNG, boundary count, hypothesis identity, and order;
- invalid root action, unsupported observation, and budget exhaustion remain
  non-mutating;
- engine operational failures are returned as errors after rollback rather
  than being counted as observational contradictions.

A test-only fault fires on a later particle after an earlier mismatch. It
compares complete private checkpoint bytes and trajectory order before/after,
then verifies deterministic replay against an independently initialized
tracker. This is test instrumentation only and is absent from non-test builds.

## Player-safe mismatch diagnosis

Exact `SearchInformationSet` equality is unchanged. On a failed equality
comparison, root-facing diagnostics now aggregate only categories from the
two root-visible DTOs:

- own state and HP;
- public opponent knowledge;
- displayed health;
- public field/side effects;
- event-history window;
- legal request;
- turn/phase alignment;
- terminal outcome;
- public-event-justified opponent-action hypothesis mismatch; and
- a forward-compatible other-public-knowledge category.

The diagnostic contains counts, never a private hypothetical view, opponent
command, hidden set, snapshot, RNG value, or actual-world truth. It is a
classifier, not a fuzzy comparator. `PlayerView`/`SearchInformationSet` exact
matching remains authoritative.

## Independent-oracle matrix

The deterministic CPU-only matrix covers P1 and P2 roots, in-prior and legal
out-of-prior hidden builds, independent actual battle seeds `801` and `811`,
`FirstLegal` and `SeededSampled` hypothetical opponent proposals, and 2/4/8/16
particle trajectories. It constructs an actual battle independently, then
passes only root actions and trusted root observations to the tracker. Each row
attempts preview plus up to two legal combat observations; a collapsed tracker
does not fabricate further updates.

`supported` below means `Supported` or `InsufficientCoverage` with a retained
root-safe conditioned observation. A repeated sampled sheet counts as a
separate chance trajectory but not as a distinct hidden-team hypothesis.

| Actual build / proposal | Rows | Preview supported | Combat 1 supported | Combat 2 supported | Collapse / attempted boundaries | Rejected comparisons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| In-prior / FirstLegal | 16 | 16 | 8 | 5 | 11 / 40 (27.5%) | 110 |
| In-prior / SeededSampled | 16 | 0 | 0 | 0 | 16 / 16 (100%) | 120 |
| Out-of-prior / FirstLegal | 16 | 16 | 8 | 4 | 12 / 40 (30.0%) | 116 |
| Out-of-prior / SeededSampled | 16 | 0 | 0 | 0 | 16 / 16 (100%) | 120 |
| **Total** | **64** | **32** | **16** | **9** | **55 / 112 (49.1%)** | **466** |

The complete matrix took **11.0 s** in the test build on the local CPU. The
per-update hard budget is 128 units in the matrix (at most 80 for 16 particles)
and the construction hard cap remains 2,000. No GPU, PPO job, full differential
corpus, or training data was used.

## Coverage and mismatch observations

Each update derives its opponent proposal seed from the private trajectory seed
plus the public turn and synthetic boundary count. This coordinator-local RNG
does not read or advance the hypothetical battle chance RNG. A proposal is
generated once, checked through the opponent's own prefix-dependent legal
request, recorded, then submitted; it is not regenerated solely for
diagnostics.

Across the 112 attempted boundaries:

| Aggregate | Boundaries |
| --- | ---: |
| More than one legal opponent proposal | 48 |
| More than one private chance trajectory | 108 |
| More than one player-safe simulated public outcome | 74 |

The aggregate mismatch counts are not mutually exclusive:

| Root-safe category | Count |
| --- | ---: |
| Event-history window | 466 |
| Displayed health | 406 |
| Public opponent knowledge | 300 |
| Public-event-justified opponent action hypothesis | 226 |
| Own state and HP | 184 |
| Public field effects | 64 |

No turn/phase, legal-request, terminal-outcome, or uncategorized public mismatch
occurred in this specific matrix. The event/health/knowledge pattern indicates
that the failures are visible trajectory differences, not merely hidden-state
comparison leakage. It does not prove whether the differences arise from
opponent action, chance, prior support, or all three.

## Root-cause assessment

| Explanation | Evidence | Assessment |
| --- | --- | --- |
| A. Opponent-action proposal coverage | `SeededSampled` creates diverse legal proposals in 48 boundaries but has zero support against actual fixtures whose opponent uses deterministic first-legal choices. `FirstLegal` carries all retained preview support. | Important, but a broader proposal policy is not validated as an improvement. |
| B. Chance/RNG coverage | 108 boundaries used multiple independent chance trajectories and 74 produced multiple public outcomes, yet support still declines sharply. | Plausible contributor; current bounded trajectories do not cover actual chance paths adequately. |
| C. Prior support / hidden diversity | The frozen fixture has only a small compatible-sheet ceiling; repeated trajectories improve RNG variation, not distinct hidden-team coverage. Out-of-prior builds retain slightly less second-combat support (4 vs 5). | Confirmed structural limitation. |
| D. Exact matching too restrictive | The new categories localize visible event/health/knowledge differences, but no representation-only discrepancy was proven. | Unproven. Keep exact equality. |
| E. Engine/state correctness defect | The transactional loss defect was confirmed and fixed. Exact replay remains positive, rollback restores byte-identical snapshots, and no operational errors occurred in the independent matrix. | Fixed for the identified update defect; no remaining engine defect demonstrated by this study. |

## Controls and limitations

Existing exact-replay tests continue to retain support across preview and two
combat boundaries for both root perspectives. Genuine out-of-support remains an
explicit initialization result. The matrix is deliberately small: two actual
seeds, two legal hidden-build conditions, a frozen two-sheet-style fixture, and
no replacement/faint chains, Mega-resource branching, or exhaustive chance
tree. Its results are diagnosis data, not policy quality measurements.

The actual oracle is independent of the particle worlds and does not reveal
actual opponent commands, IDs, private sets, snapshots, hidden stats, or future
RNG. The out-of-prior variation changes only the root's opponent private build;
the tracker receives the same permitted root interface.

## Verification

Passed commands:

```text
cargo test --lib search::particles::tests -- --nocapture
cargo test --test search_core --test search_world --test search_belief \
  --test search_information_set --test search_test_oracle
cargo clippy --all-targets -- -D warnings
rustfmt --edition 2021 --check engine/src/search/particles.rs \
  engine/src/search/world.rs
git diff --check
```

The full differential corpus was intentionally not run. Production battle
mechanics, Track A, `main`, PPO, Python integration, labels, distillation, and
training datasets were not modified.

**Final gate:** `C3B12_BELIEF_DIAGNOSTICS_RESEARCH_ONLY / NO_VALIDATED_CTS_TEACHER / NO_LABELS / NO_DISTILLATION`
