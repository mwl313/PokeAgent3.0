# Training-readiness gate

Readiness is a conjunction of the criteria below. It is enforced by
`engine/examples/readiness_check.rs`, which prints every criterion with
PASS / FAIL / NOT VERIFIED and exits non-zero unless all of them pass:

```bash
bash scripts/cargo.sh run --locked --release --example readiness_check
```

The gate exists so that partial coverage cannot be reported as readiness. A
lower honest number is preferred over a higher one produced by stubs, generic
fallbacks, or a narrowed denominator.

## Criteria

| # | Criterion | Enforced by |
|---|---|---|
| 1 | Full pinned M-C legal species/form coverage (no unresolved starting candidates, all permitted battle forms) | `scope.json` (`starting_species`, `format_permitted_battle_forms`, `mega_forms`, `unresolved_starting_candidates`) |
| 2 | Full legal move coverage (all 515 allowed moves) | `coverage_report` blocked list + `classify_move` |
| 3 | Full legal ability coverage (all 223 legal abilities) | `Ability::is_ported`, the same predicate `validate_effects` uses at runtime |
| 4 | Full legal item coverage (all 166 allowed items, Mega Stones via the stone mapping) | `Item` handler table + `mega_stones` |
| 5 | Dynamic/reachable effect closure (called, copied, transformed, inherited effects) | machine-generated `engine/data/dynamic-closure.json` (from `scripts/dynamic_closure.mjs`); a caller passes only when it is executable **and** every universe it can reach is fully executable |
| 6 | All 1,136 frozen training teams can execute every member | static member scan of `training-teams.json` |
| 7 | No silent fallback mechanics | `classify_move` requires every callback key and data field to be ported; `fixture_coverage` requires a differential witness or a documented exemption; `readiness_check` recomputes the unwitnessed-move set every run |
| 8 | Zero unsupported-mechanic operational errors on the full-scope validation corpus | follows from 2-4 plus a full-scope corpus run |
| 9 | Player-safe observation tensor complete | `observation.rs::knowledge_field_audit` (compile-time classification of every knowledge field) + `observation_leakage.rs` (hidden state cannot change tensor/mask) + `observation_completeness.rs` (known state must change tensor) |
| 10 | Native player-safe action masks complete | corpus compares request kind, slot presence/replacement, Mega availability, selectable moves with PP, target class, bench set and preview order at every boundary |
| 11 | Deterministic RNG / replay | explicit four-word seed, per-boundary seed comparison, trace replay test |
| 12 | Snapshot / restore | schema-validated snapshots, restore-every-7-steps corpus check, corruption tests |
| 13 | PyO3 batch path | `step_batch` / `observe_encoded_batch` / `request_info_batch` / `candidates_batch`; `test_bridge.py` asserts batched masks equal the per-request walk; `pa3_actor.py` performs a constant number of crossings per round regardless of environment count |
| 14 | 2,048-environment execution validated | two-process 2x1,024 actor with NUMA pinning; serial-equivalence test in the Rust suite |
| 15 | Full-coverage throughput re-measured | only meaningful once 2-4 and 6 pass |
| 16 | Differential mechanics/interaction/full-battle validation green at full coverage | `turns.rs` corpus + `verify_turn_fixtures.mjs` regenerated against the pinned reference |

## Coverage accounting rules

- **Training-pool coverage**, **regulation-wide native coverage**, and
  **dynamic/reachable-effect coverage** are reported separately and must never
  be merged into a single readiness percentage.
- **Dynamic closure** is derived from the pinned reference by
  `scripts/dynamic_closure.mjs`, which scans every in-scope move/ability/item
  handler for call/copy/transform/transfer/suppression tokens and records the
  universe each caller can reach (`moves`, `abilities`, `items`,
  `species_forms`). The closure is deliberately conservative (an
  over-approximation can only make readiness stricter). `export_engine_data.mjs`
  regenerates the file; the manifest pins its digest.
- "Executable" means the classifier accepted the entity *and* the runtime gate
  (`validate_effects` for abilities/items, `classify_move` for moves) will not
  reject it. It does **not** by itself mean "verified": moves additionally
  require a differential fixture (`moves_witnessed`), and abilities require an
  interaction witness (`ability_fixture_report`).
- Training-pool coverage itself has three different meanings, and
  `pool_run_report` reports them as separate numbers:

  | Concept | Meaning | Evidence |
  |---|---|---|
  | Trajectory completions | the team finished at least one natural battle | seeded policy probe |
  | Static mechanic completeness | every member's moves, ability and item are implemented, so any legal trajectory can run | static scan of `training-teams.json` |
  | Fully supported for arbitrary legal play | the whole regulation is implemented and validated, so no trajectory can reach an operational error | readiness criteria 2-6 and 8 |

  A team that completes one battle while its unsupported mechanic never had to
  fire is **not** fully supported; only static mechanic completeness supports
  that claim.
- **Coverage tiers stay distinct**: *implemented* (native behavior exists and
  the runtime gate admits it), *witnessed* (a differential fixture ran it once
  with the reference matching at every boundary), and *interaction-verified*
  (a shared or complex family has representative reference-generated fixtures
  for its interaction branches: priority/order ties, immunity-versus-accuracy
  ordering, request-mask shape, switch and target edge cases). A counter may
  only claim the tier its evidence proves; one encounter is never reported as
  interaction verification.
- **Items: 165/166 is not "all items covered".** The remaining entry is the
  legal held item **Metronome** (consecutive same-move damage bonus), which is
  unimplemented and stays in the denominator; holding it raises an explicit
  operational error. It is not a sentinel or a no-item representation.
- Pool frequency only orders the work. The final scope is the full pinned M-C
  regulation, including effects that can be reached by calling, copying,
  transforming into, or inheriting another effect. The authoritative caller
  list lives in `dynamic-closure.json`; Metronome, Mimic, Assist, Mirror Move
  and Nature Power are outside the pinned M-C scope and are reported only for
  completeness. Sleep Talk is a caller even though its own handler is simple:
  it can execute any move a teammate legally knows.
- Current throughput numbers (for example 417.8 natural games/s wall over
  2x1,024 environments) are **architectural progress metrics**, not the final
  benchmark: roughly half of the environments still terminate on an unported
  mechanic, and the actor counts those as `operational_errors`, never as games.
  The final benchmark must contain only naturally completed legal battles on
  the full training pool with no operational errors.
