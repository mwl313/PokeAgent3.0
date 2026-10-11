# C3b-1: persistent CTS belief tracking

**Status:** `C3B1_PERSISTENT_BELIEF_RESEARCH_ONLY / NO_VALIDATED_CTS_TEACHER / NO_LABELS / NO_DISTILLATION`

**Starting commit:** `749c17f6dac19fbb9faf110ec4678c4870a0671c`

**Implementation commit:** `eec005384700adccdc7f400959c2cf60be3fbb31`

## Feasibility audit

The C3b-0 `ParticleBelief::advance_preview_and_one_turn` API returned only a
root-safe `ParticleResult`; it dropped every `SyntheticEpisode` after the
comparison. Consequently, no hypothetical state, RNG stream, pending request,
or hidden sheet could continue to the next boundary. Its exact equality over
the complete `SearchInformationSet` also made survival a strict support test,
not a calibrated likelihood. The prior fixture commonly has only two
observation-compatible sheets, so requesting 4 or 8 particles previously did
not create corresponding hidden-sheet diversity. The old one-shot budget did
not charge prior sampling, construction retries, transitions, opponent
decisions, or comparisons separately, and exhaustion could be reported as
`ParticleCollapse`.

The current implementation retains exact support semantics and does not add a
fuzzy likelihood. This is deliberate: event-window semantics and display
quantization have not yet been validated for safe tolerances.

## Persistent architecture

`PersistentParticleBelief` owns a private vector of `SyntheticEpisode` values.
Each episode contains the full hypothetical battle state but is inaccessible
through the public tracker API. Initialization samples anonymized sheets from
the frozen prior and constructs bounded clean-preview worlds. If the prior
contains fewer unique sheets than requested trajectories, sheets are repeated
as independently seeded trajectories; `distinct_hypotheses` and
`trajectory_count` are reported separately.

Each `update` receives only the root player's submitted legal action and the
next trusted root `PlayerView`. Every private episode selects its hypothetical
opponent action through the existing coordinator hook, which observes only
that hypothetical opponent's own view. The episode is retained only when its
new player-safe DTO equals the trusted target. No live `BattleState`, snapshot,
opponent command, hidden team ID/set, or live RNG enters the tracker.

The flow is:

`trusted preview → private sampled episodes → root action + private baseline →
root observation projection → exact condition → retained private episodes`

Budget exhaustion is checked before mutating a persistent particle set, so a
partial update cannot silently discard unprocessed trajectories or masquerade
as collapse.

## Observation contract

The versioned comparison contract is exact for schema version, public turn,
public knowledge, public outcome, root side, own state, and served legal
request. The current `public.knowledge.events` window is compared exactly as
serialized and is called out separately as an audited event-history field; no
tolerance is applied. Tolerated fields are currently empty. Private battle
state, opponent identifiers/private sets, pending commands, and RNG are
excluded. An impossible side or backwards-turn observation returns
`UnsupportedObservation`; a compatible but rejected exact observation returns
`ParticleCollapse`.

No broad tolerance, posterior repair, hidden-value copying, rejuvenation, or
fabricated weights is implemented. Effective sample size is therefore not
reported: this is conservative equal-weight exact support, not a likelihood
posterior.

## Tests and oracle observations

The test-only suite covers deterministic persistent replay through preview and
two successive combat boundaries for both P1 and P2, independent trajectories
for repeated sheets, exact replay, genuine exact-conditioning collapse,
explicit out-of-support initialization, and budget exhaustion distinct from
collapse. A separate actual-world oracle constructs an independently seeded
battle and supplies only its player observations and root actions to the
tracker; it never supplies opponent commands, team IDs, hidden stats, or RNG.
The oracle test records initialized/advanced/rejected/retained counts and
accepts the honest outcomes `Supported`, `InsufficientCoverage`, or
`ParticleCollapse` for this small exact-support fixture. Existing metamorphic
tests change hidden opponent truth while preserving the player-visible input
and verify identical tracker results.

Exact self-replay is a positive control; an altered target observation is a
negative control. The independent-oracle path is intentionally not claimed as
CTS readiness: exact conditioning may reject most or all particles when the
actual battle's public trajectory is not represented by the bounded prior and
seed schedule. The current frozen prior's two-compatible-sheet ceiling remains
the dominant diversity limitation.

## Accounting and limits

Initialization records prior sampling, world-construction attempts, distinct
hypotheses, trajectory count, Dex/prior provenance, and the explicit seed
schedule. Each update charges five bounded units per trajectory for the two
engine transitions, one opponent decision, one chance stream, and one exact
observation comparison. The hard particle cap is 16 and the hard update budget
is 2,000 units. Statuses are distinct: `Supported`, `InsufficientCoverage`,
`ParticleCollapse`, `OutOfSupport`, `BudgetExhausted`, and
`UnsupportedObservation`.

The targeted particle test binary executes in roughly 3 seconds on the local
CPU fixture after compilation. These measurements are bounded test timings,
not throughput claims.

## Remaining blockers

Genuine midbattle CTS search still requires a validated observation likelihood
model, independent priors with meaningful hidden-sheet diversity, disjoint
independent-oracle fixtures, collapse/support thresholds, broader chance and
opponent-policy coverage, and a reviewed complete legal-action contract.
Exact support alone is not enough to claim useful posterior tracking.

A future `TeacherSample` must remain root-safe: PA3-compatible observation,
complete legal joint actions, search targets, confidence/provenance diagnostics,
and schema versions, generated from a belief distribution compatible with the
observation rather than privileged access to one sampled world. No labels,
teacher integration, PPO changes, or distillation data are generated here.

Track A, `main`, production battle mechanics, datasets, GPUs, and the full
differential corpus were not touched.
