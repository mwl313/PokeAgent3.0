# C3a.2 / C3b-0 research report

**Status:** `C3A2_EVALUATOR_HARDENED / C3B0_PARTICLE_RESEARCH_ONLY / NO_VALIDATED_TEACHER / NO_DISTILLATION`

**Starting commit:** `7666d6e004c28fe00888541c442355f27c686ce5`

**Implementation commit:** `acedc33f050a07c8b061c08669bc2964480705da`

## C3a.2 evaluator hardening

The old selector was reproduced against the 360 legal preview joints. Its
short-circuit expression inserted an opening family before confirming that
the reserve family was new. At limit 8, the legacy selector retained 6 unique
opening pairs and 8 reserve combinations; the corrected selector retains 8 and
8. The corrected implementation mutates family sets only after retaining the
candidate. It remains deterministic and fully legal.

Pinned diagnostics agree exactly with retained actions at every requested
limit:

| retained limit | unique opening pairs | unique reserve combinations |
|---:|---:|---:|
| 2 | 2 | 2 |
| 8 | 8 | 8 |
| 16 | 16 | 16 |
| 30 | 30 | 26 |
| 64 | 30 | 30 |

The original C3a.1 evaluator remains callable as
`BoundedSearch::search_c3a1_baseline`. It preserves first-N selection,
candidate-dependent seed derivation, no post-preview rollout, and legacy
unknown-health scoring for apples-to-apples diagnostics only. It never returns
a selected recommendation. The hardened path uses paired world seeds,
stratified selection, bounded combat rollout, and fail-closed scoring.

Nonterminal scoring is explicitly asymmetric: own HP is averaged over all six
own observed Pokémon, while opponent HP is averaged only over publicly known
opponent health entries. The unknown fraction is reported separately; no
hidden value is filled in. Terminal reward is a separate branch. Truncated or
operational-error outcomes are rejected before heuristic scoring.

Diagnostics now include paired per-world score differences, pooled score
dispersion, effective world coverage, ties, and aggregate-vs-per-world ranking
instability. `heuristic_differentiated_actions` counts distinct aggregate
action scores (epsilon `1e-6`), not merely nonzero means. A selected action is
returned only by the hardened evaluator after complete equal-coverage
evaluation; partial or truncated comparisons do not produce a recommendation
or label.

The pinned world-count comparison exercised requested counts 2, 4, and 8.
The current observation-conditioned frozen prior has only two compatible
candidate sheets for this fixture, so sampled support remains `min(request,
2)`. This is reported as a prior-support limitation, not silently treated as
8-world evidence. Runtime for the targeted search-core test binary was about
2.4 seconds after compilation; the particle unit test binary took about
3.1 seconds of test execution. These are bounded fixture measurements, not
performance claims. Budget diagnostics continue to record transitions,
construction attempts, chance samples, and computation units.

## C3b-0 particle feasibility

`engine/src/search/particles.rs` adds a private-world, forward-only prototype.
It samples anonymized complete sheets from the frozen prior using explicit
seeds, constructs fresh clean-preview worlds, submits the known root preview
action, submits one subsequent complete combat action, and conditions only on
the resulting root `SearchInformationSet`. Hypothetical opponent actions are
selected by the existing coordinator hook from each hypothetical opponent's
own `PlayerView`; no root-side result exposes those commands or worlds.

The exact comparison contract currently includes `schema_version`, public
turn, public knowledge, public outcome, own side, own state, and own legal
request. No tolerance model is silently assumed. There are no tolerated fields
yet. Raw `BattleState`, opponent team identifiers/private sets, pending
opponent actions, and live RNG are explicitly excluded. A future likelihood
model must be separately specified before relaxing exact equality.

Statuses are explicit: `OutOfSupport`, `ParticleCollapse`,
`InsufficientCoverage`, or `Supported`. The bounded tests verify deterministic
P1 and P2 replay, a genuine exact-conditioning collapse with a mismatched
target observation, and hidden-truth metamorphic invariance of the trusted
input/result path. No particle is repaired using private state. The prototype
keeps only a root-safe conditioned observation and nonsecret diagnostics.

This proof of concept does not reconstruct arbitrary live midbattle states,
does not infer hidden opponent commands, and does not claim that exact
conditioning retains useful posterior mass beyond the pinned fixture. The
two-particle support ceiling in the current preview prior is an explicit
coverage limitation. A safe next experiment is a reviewed likelihood/tolerance
contract over player-visible fields, with collapse and out-of-support rates
measured before any CTS use.

## Remaining CTS and TeacherSample gates

Before midbattle-root CTS search, require: a validated observation likelihood
model; support and collapse thresholds on disjoint fixtures; independent
chance/opponent-policy review; complete legal-action coverage; deterministic
replay and resource audits; and leakage tests over every root-facing field.
Any future `TeacherSample` must contain a PA3-compatible root observation,
complete legal joint-action atoms, search target scores/distributions,
confidence/provenance diagnostics, and schema versions. It must be generated
from a belief distribution compatible with the root observation, never from
privileged evaluation of one hidden sampled world. No labels, Python
integration, PPO changes, or distillation data are produced here.

Track A and `main` were not modified. No GPU, PPO training, or full
differential corpus was run.
