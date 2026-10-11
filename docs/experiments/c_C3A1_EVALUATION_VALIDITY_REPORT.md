# C3a.1: search-evaluation validity

**Status:** `C3A1_EVALUATOR_RESEARCH_ONLY / NO_VALIDATED_TEACHER / NO_LABELS / NO_DISTILLATION`

**Base:** `5ffed3d934ba30905549e95b049f3ed9fbcce8c9`

**Implementation commit:** `ac2266e` (final commit for this milestone)

## Purpose

C3a.1 hardens the independent, CPU-only C3a evaluator. It does not add C3b,
teacher labels, distillation, PPO integration, or Track A dependencies. The
coordinator still accepts only a trusted clean-preview player information set,
an anonymized frozen prior, explicit seeds, and bounded configuration.

## Validity changes

- Every retained root candidate uses the same deterministic seed panel for a
  given sampled-world index. Seeds are derived from `(seed panel, world index)`
  and never from candidate identity. This is a common-random-number comparison;
  action-dependent engine RNG consumption can still legitimately diverge after
  a branch.
- Candidate truncation is deterministic and staged by distinct opening switch
  pairs and reserve combinations before filling the remaining slots. The
  diagnostics report total legal candidates, retained families, and omitted
  families rather than presenting a prefix as exhaustive.
- Each retained candidate resolves preview and, by default, one complete
  simultaneous combat turn. A bounded configuration may request at most two
  combat rollout turns. Every submitted action, transition, opponent baseline
  sample, chance sample, and C2c construction retry is charged to the explicit
  unit budget.
- Nonterminal scores average only publicly known opposing health. Unknown
  health is excluded from the denominator and its fraction is reported. If no
  opposing health is known, the evaluator fails closed for that world instead
  of treating unknown health as zero. Terminal engine reward takes precedence
  over heuristic scoring.
- Results are `Complete` only when retained actions have equal coverage and no
  failed world evaluations. Truncation, budget exhaustion, and unequal
  coverage remain structured non-complete outcomes; incomplete comparisons do
  not return a selected action.

## Verified fixture

The deterministic two-world fixture enumerates 360 legal complete preview
joint actions and retains two. It retains two distinct opening pairs and two
reserve combinations, omitting 28 of each family. Both retained actions are
evaluated in both worlds after preview plus one combat turn, with no failed
evaluations. The observed run produced heuristic means `0.14373311` and
`-0.02392736`, variance `0.0070275087`, zero tied pairs, and two
heuristically differentiated actions. Mean unknown opposing-health fraction
was `2/3`; no hidden value was substituted. The same input, prior, seed panel,
and configuration reproduces byte-for-byte identical results.

The corresponding budget recorded four successful candidate/world evaluations,
16 engine transitions, eight opponent-action samples, eight chance samples,
54 C2c construction attempts, and 448 computation units. These are diagnostic
fixture values, not performance guarantees.

## Coverage and remaining risks

The evaluator is still clean-preview-root only. It does not reconstruct an
actual ongoing PPO battle, search arbitrary chance branches, or claim a
calibrated value, CTS strength, or teacher safety. Candidate selection is
diverse but intentionally incomplete when the legal set exceeds the bound.
The own-view baseline is an execution fixture, not an opponent model. The
common-random panel controls the seed input but cannot make branches with
different RNG consumption identical. Larger priors, mechanics, Mega usage,
and additional chance-sensitive effects require separate bounded fixtures.

## Future C3b gates

Before any teacher work, require reviewed coverage policy, a stronger public
state scoring contract, independent belief/chance validation, resource and
replay audits, and leakage tests over all root-facing fields. A future
`TeacherSample` must carry a PA3-compatible root observation, complete legal
joint-action atoms, aggregate target scores/distributions, confidence and
provenance diagnostics, and schema versions. It must be produced from a belief
distribution compatible with the root observation, never privileged access to
one hidden sampled world. No such samples are generated here.

Track A remains independent. No GPU, PPO checkpoint, rollout-buffer change,
training, or full differential corpus was used.

**Implementation files:** `engine/src/search/core.rs`,
`engine/src/search/world.rs`, `engine/tests/search_core.rs`, and this report.

**Final review gate:** `C3A1_EVALUATOR_RESEARCH_ONLY / NO_VALIDATED_TEACHER / NO_LABELS / NO_DISTILLATION`
