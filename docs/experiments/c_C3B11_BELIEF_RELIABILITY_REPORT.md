# C3b-1.1: persistent belief reliability validation

**Status:** `C3B11_BELIEF_RELIABILITY_RESEARCH_ONLY / NO_VALIDATED_CTS_TEACHER / NO_LABELS / NO_DISTILLATION`

**Starting commit:** `8f70f190975b2afc847c9958b91c4e2e5c9b8c62`

**Implementation commit:** `420a63c6efe6fdf0efcc093a018b8615363b6902`

## Scope and conclusion

This milestone validates the persistent particle tracker as a bounded, test-only
research component. It does not validate a CTS teacher. The result is **B**:
the tracker is operationally safe and reproducible, but not yet reliable enough
for CTS-grade opponent belief tracking or teacher labels.

The dominant evidence is exact-support brittleness. In the independent actual
battle fixture, P1 retained one of two trajectories after the first combat
boundary, while P2 rejected both on the same bounded prior/seed schedule. The
tests now exercise two actual-world seeds, both root perspectives, and both an
in-prior hidden build and a genuinely different out-of-prior hidden build. The
honest statuses remain `Supported`, `InsufficientCoverage`, or
`ParticleCollapse`; no repair or fabricated posterior is applied.

## Verified implementation

`PersistentParticleBelief` keeps complete `SyntheticEpisode` values private,
conditions only on the trusted root `SearchInformationSet`, and retains no
raw world state in `ParticleResult`. The update path now:

- rejects a changed minimum-survival configuration;
- validates every root action against every private episode before mutation;
- checks the full update budget before mutation;
- checkpoints every episode before advancing it;
- restores all processed, surviving, and pending episodes on an unexpected
  observation-conversion error;
- distinguishes `InvalidRootAction`, `BudgetExhausted`, `ParticleCollapse`, and
  `InsufficientCoverage` instead of conflating them;
- records pre/post particle counts, hypothesis diversity, proposal policy,
  proposal counts, and provenance;
- supports a bounded `SeededSampled` hypothetical opponent proposal that uses
  only the sampled world's private opponent `PlayerView` and preserves the
  existing six-field atomic-action/prefix legality contract.

The sampled proposal is a test-only hypothetical transition policy. It is not
an opponent model, not a CTS-safe teacher, and not a source of labels.

## Tests and results

The following targeted commands passed on the local Rust 1.90 toolchain:

```text
cargo test --lib search::particles::tests -- --nocapture       10 passed
cargo test --test search_core --test search_world --test search_belief \
  --test search_information_set --test search_test_oracle       all passed
cargo clippy --all-targets -- -D warnings                       passed
rustfmt --edition 2021 --check engine/src/search/particles.rs \
  engine/src/search/world.rs                                   passed
git diff --check                                                passed
```

The particle tests cover:

1. clean-preview initialization and explicit out-of-support;
2. requested trajectory counts 2, 4, 8, and 16 under the hard CPU budget;
3. two successive combat boundaries for P1 and P2;
4. exact replay and hidden-truth metamorphism;
5. sampled legal opponent proposals and proposal diversity diagnostics;
6. budget exhaustion distinct from collapse;
7. invalid root actions rejected without changing the tracker observation;
8. independent actual-world replay using seeds 991 and 995, both P1/P2 roots,
   and in-prior/out-of-prior opponent hidden builds.

The actual-world oracle supplies only root actions and trusted root observations.
It never supplies opponent commands, opponent team IDs, private sets, hidden
stats, or future RNG. The oracle remains explicitly separate from tracker-owned
synthetic episodes.

## Reliability and coverage limits

The comparison contract remains exact over the current versioned DTO, including
the serialized public event window. No likelihood tolerance, ESS, rejuvenation,
belief repair, or posterior weighting is implemented. The frozen prior has a
small compatible-sheet ceiling, repeated sheets are independent trajectories
but not independent hypotheses, and the first-legal/sampled proposal policies
do not cover the space of real opponent policies. Mega-resource interactions,
multiple chance branches, faint/replacement chains, and broad hidden-build
coverage remain bounded-fixture gaps rather than claims of coverage.

The tracker is forward-only from a clean preview. It does not reconstruct an
actual ongoing PPO battle or invent missing hidden values. A midbattle root
must still be rejected by the existing boundary contract.

## Compatibility audit

The change is additive to the existing search modules. Production battle
mechanics, PPO rollout buffers/objectives, Python integration, Track A, and the
full differential corpus are untouched. Existing callers of `initialize`,
`update`, and `submit_root_with_baseline` retain their behavior; the sampled
proposal path is explicit and opt-in. All root-facing values remain typed DTOs
or structured diagnostics, while private synthetic episodes and opponent
submission details remain coordinator-internal.

## C2b/C3 readiness decision

The implementation is **not ready for a validated CTS teacher**. Before C3
teacher/search work can consume it, the project still needs a reviewed
observation-likelihood contract, a larger legal prior with meaningful hidden
hypothesis diversity, independent multi-seed actual-world fixtures, calibrated
support/collapse thresholds, explicit replacement/faint/chance coverage, and a
reviewed opponent-policy proposal contract. C3b-1.1 therefore stops here; no
labels, search scores, distillation, or PPO changes are produced.

## Future integration boundary

A future `TeacherSample` may contain only a PA3-compatible root observation, a
complete legal joint-action representation using the existing six-field atomic
actions and prefix semantics, search target scores/distributions, confidence and
provenance diagnostics, and explicit schema versions. A teacher must evaluate a
belief distribution compatible with the root observation, not privileged access
to one sampled hidden world. A later distillation learner must consume this
contract without changing the production PPO rollout buffer or PPO objective.

**Final gate:** `C3B11_BELIEF_RELIABILITY_RESEARCH_ONLY / NO_VALIDATED_CTS_TEACHER / NO_LABELS / NO_DISTILLATION`
