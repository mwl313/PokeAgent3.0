# C2d: safe synthetic multi-boundary episode executor

**Status:** `C2D_SYNTHETIC_EPISODE_ONLY / NO_MIDBATTLE_ROOT / NO_SEARCH / NO_TEACHER / NO_DISTILLATION`

## Scope and boundary

C2d extends the C2c clean-preview world with a privately owned
`SyntheticEpisode`. It holds the hypothetical `BattleState` and pinned `Dex`
internally. The public boundary exposes only the root player's `PlayerView`,
prefix-dependent `Request::candidates`, `StepResult`, and schema/turn/request
diagnostics. No snapshot, raw state, opponent team, opponent action, RNG, or
hidden event state is returned.

`submit_joint` pre-validates every supplied side choice before mutation. It
preserves simultaneous joint-action semantics, allows a legal partial commit to
leave the other side pending, and rolls back the exact serialized state if an
engine operational error occurs after mutation. Invalid submissions are
non-mutating. Root candidates use the observed request, so a root side that has
already submitted receives no candidates while waiting for the other side.

The executor advances synthetic states only. It does not reconstruct an
arbitrary ongoing PPO battle and does not make a sampled world equivalent to a
player's true hidden world.

## Hidden-truth metamorphic coverage

The new test changes the opponent's team identifier, EVs, IVs, and nature while
keeping the same pinned species/seed-visible preview. The two resulting
`TrustedSearchInformationSet` values are equal, and construction with the same
anonymized sample and independent seed produces the same root view. Existing
C2b/C2c tests continue to cover serialized information equality and rejection
of unknown preview species rather than filling them from the sample.

## Verification

The targeted suite covers:

- P1/P2 preview construction and deterministic reproduction;
- simultaneous preview submission and a subsequent combat request boundary;
- prefix-dependent root candidates;
- invalid action non-mutation;
- malformed samples, unsupported midbattle inputs, and unknown opponent masks;
- genuinely different hidden opponent builds with identical root contracts;
- the existing combat-turn oracle for legal joint actions, conditional targets,
  switching/replacement, chance/resource branches, restoration, and isolation.

Run CPU-only targeted Rust tests, Clippy (`-D warnings`), rustfmt, and diff
checks. The full differential corpus and all training workloads remain gated.

## FUTURE_PPO_INTEGRATION_CONTRACT

Future teacher work may define a versioned `TeacherSample` without changing the
production PPO rollout buffer or objective:

1. `schema_version`, engine format/Dex digest, and frozen provenance/checkpoint
   identifiers;
2. PA3-compatible root observation bytes plus the root side and boundary index;
3. complete legal joint actions using the existing six-field atomic action
   `(kind, own_slot, move_slot, target_location, switch_destination, resource)`
   and its prefix-dependent candidate semantics;
4. search target scores or a normalized action distribution over those legal
   joints;
5. confidence, effective-sample count, rejection/coverage diagnostics, and an
   explicit no-label/invalid marker.

The contract must be generated from a belief distribution compatible with the
root observation, never from privileged evaluation of one sampled hidden world.
Before teacher labels or PPO training, gates are required for: approved
information-set and action schema versions; deterministic synthetic episode
reproduction; metamorphic hidden-truth isolation; complete joint-action and
rollback tests; bounded CPU resource accounting; fixed observation encoding;
and an approved Track-A checkpoint/schema integration review. None of those
gates authorizes C3 search, policy inference, Python integration, labels,
distillation, or training in this milestone.

## Remaining limitations

Arbitrary midbattle reconstruction remains rejected. Current C2c information
does not establish hidden opponent HP/PP/status durations, volatiles,
copied/transformed moves or abilities, item changes, pending commands, event
ordering, or live RNG. The executor currently accepts joint choices supplied by
its caller; it does not implement an opponent policy or belief search. C2c-2,
C3 search, teacher labels, and PPO integration remain future review-gated work.

