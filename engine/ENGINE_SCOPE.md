# Engine implementation contract

The user's 2026-10-06 engine request expands the original inventory-based engine scope: support **every legal Pokémon in the pinned Champions M-C regulation**, including Mega and other reachable forms, every legal set, and interacting and dynamically called effects. The frozen 1,136-team training pool is unchanged. No species or team may be removed to accommodate missing engine mechanics.

The rules authority remains Pokémon Showdown commit `14546894d86f9589ac11130c510bbe73b6968665`, format `gen9championsvgc2026regmc`, closed team sheets, doubles, bring six/pick four. The supplied specification remains unchanged as historical source material. This file records the later user scope instruction.

The implementation must use native Rust for transitions, numeric effect identifiers and typed observations, with PyO3 batching and no reference calls on the training path. Showdown is used for development fixtures and differential verification only. Full support is an implementation and interaction-testing obligation, not merely inclusion in a data table. Unimplemented mechanics must cause explicit operational errors, never silent no-ops, fabricated wins or truncated draws.

Implementation order: pin/export the complete regulation and dependency catalogue; build exact RNG/stat primitives and action/knowledge contracts; implement native event ordering and all reachable mechanics; add differential and information-boundary tests; integrate and measure 2,048 environments across two 16-worker groups. Training and model construction are outside this engine request.

Status: implementation remains incomplete and resumed at the user's request on 2026-10-07. Terrain integration and the subsequent hot-path allocation audit are verified; the full engine goal remains active. Parallel subagent work is authorized; separate fixture and snapshot-validation agents contributed with centralized integration. No full-engine readiness claim yet.

## Present implementation

`engine/data/scope.json` contains 293 starting species/form entries with reference-validated set witnesses, 97 format-permitted battle-only forms (82 Mega), 515 format-allowed moves, and 166 items. `dex.json` retains the full pinned mod catalogue as a conservative dependency superset, including entries outside the regulation. Their presence does not make them legal. There are no unresolved starting-species candidates in this export. Dynamic dependency closure and interactions still require implementation and verification.

Rust currently implements:

- Explicit four-word reference Gen5 RNG seeds, exact draw consumption and forward tie shuffling.
- Champions stats, PP, fixed-point arithmetic, an isolated damage kernel and reference event ordering.
- Prefix-feasible preview, normal and replacement candidates, target rules, shared Mega and bench constraints, Struggle/pass distinctions, and request-visible trapping.
- Numeric persistent knowledge, independent move-reveal history, public HP including Champions boundary colours, and the 88-token role layout padded to 96.
- Private world-state containers and separate player views; hidden opponent allocations, nature, provenance and role mapping are excluded.
- Native parallel reset, observation and `step_batch`, generation-checked handles, whole-submission input validation before mutation, and sparse/reordered result routing. No environment clone, JSON conversion or reference call occurs in batch stepping. Snapshot schema 4 validates integrity, catalogue IDs, roster/active mappings, effect state, pending requests and queued actions.
- A native decision-boundary loop with committed simultaneous choices, preview ordering, action queues, dynamic speed reordering, voluntary/forced switches, faint ordering and natural outcomes.
- Opt-in native traces containing an initial world/knowledge/RNG snapshot, action boundaries and separate audience-filtered semantic-event streams. Replay verifies RNG boundaries and events, including operational failures. Effect IDs carry their catalogue kind so item, move and condition IDs cannot be confused. Public Mega use persists in player knowledge.
- An initial explicit set of damage, Protect, Struggle, stat-change, healing and status moves; Champions paralysis, sleep and freeze rules, burn/poison residuals, and flinching. All other moves and effects still raise operational errors when reached.
- Mega availability from exact Champions base-form/item mappings, one Mega per side, queued evolution, permanent stats/types/ability changes, HP changes preserving damage taken, and persistence through switches/fainting. Only the initial ported abilities can currently execute; this does not yet cover every Mega interaction.
- Initial ordered ability hooks: Intimidate and stat-drop responses/immunities, Speed Boost, Natural Cure/Regenerator, offensive power/STAB modifiers, and defensive damage/attack modifiers. Conditional hooks participate in tie ordering even when they return no modifier.
- Initial held-item handlers: Choice Scarf and move locking, Focus Sash, Life Orb, Rocky Helmet, Leftovers, Expert Belt, Big Root for drain healing, Light Clay for screens, Sitrus/Oran/Lum berries. Their interactions are tested alongside status, spread damage, recoil and Mega abilities; this is not a claim of complete item coverage.
- Native recoil and draining attacks, including spread drain, Rock Head, Reckless and Liquid Ooze. Drain occurs during damage application; recoil uses actual damage dealt and occurs after the first faint-processing boundary. Positive fractional recovery/recoil rounds halves up, independently of the fixed-point modifier rule. The exported Champions move ratios are preserved (including 33/100 recoil). Big Root modifies successful drain healing, while Liquid Ooze cancels healing and damages the user by the unmodified amount, including at full HP. Struggle bypasses Rock Head. DamagingHit handlers now use stable target order without speed-tie RNG.
- Native Tailwind, Reflect and Light Screen with side-target execution, exact residual expiration order, Light Clay duration, fixed-point doubles screen reduction, critical-hit bypass, Brick Break/Psychic Fangs removal and Infiltrator bypass. Tailwind combines with Choice Scarf before final speed rounding and participates in mid-turn queue reordering. Side effects persist through switches and source fainting. The casting side tracks its known duration; opposing exact durations remain unknown so hidden Light Clay is not exposed. Snapshot restore rejects invalid durations and sources for these effects.
- Native rain, sun, sand and snow with move/ability starts, weather-rock duration, same-weather rejection, replacement and expiration. Swift Swim, Chlorophyll, Sand Rush and Slush Rush combine with Tailwind/Choice Scarf before speed rounding. Weather Ball changes type/power; weather modifies Fire/Water damage, sand raises Rock special defense, snow raises Ice defense, sun prevents freezing, and sand damages nonimmune Pokémon before item updates and faint processing. Public weather knowledge hides opposing exact durations; snapshots reject invalid or simultaneous weather. Remaining weather-dependent abilities/moves are still required.
- Native Rain Dish and Ice Body recovery during rain/snow upkeep, and Solar Power Special Attack modification and sun damage. Recovery/damage preserve integer HP truncation and the weather → Update → faint boundary. Solar Power retains its no-op modifier hook outside sun to preserve RNG ties. Eleven new complete legal fixtures require healing/damage, sun boosts, and both active and inactive speed ties with Mega Venusaur Thick Fat. Weather-suppression interactions are covered by the following group; remaining weather abilities stay guarded.
- Native Cloud Nine and Air Lock suppression separates stored timed weather from effective weather. Damage, conditional speed, Weather Ball, weather defenses, freeze prevention and weather-dependent abilities use effective weather; starts, same-weather rejection, duration and expiry retain stored weather. Rain/sun still sort Weather recipients and run Update under suppression; sand/snow skip that guarded call. Zero-HP holders suppress until End, which runs before clearing/fainting and includes the departing holder in WeatherChange sorting. Ability ending is private state, reset on entry/replacement; it was introduced in schema 3 and is validated in current schema 4 snapshots. Earlier schemas are rejected. Nine new legal Cloud Nine battles require all four suppressed expirations, restored effects after switches/fainting, and re-entry. Air Lock shares these reference-equivalent callbacks but has no legal starting holder in the pinned scope; no claim of tested dynamic acquisition or complete Mega Altaria/Pixilate coverage is made.
- Native Sand Force, Sand Veil, Snow Cloak, Overcoat and Hydration. Sand Force uses exact 5325/4096 power modification, including Mega Garchomp; weather evasion uses 3277/4096 before accuracy-stage truncation. Overcoat blocks powder before accuracy and preserves sand immunity. Hydration cures after Grassy healing and before Leftovers/status damage, respecting rain expiry and suppression. Twenty-one new legal complete-battle fixtures require actual boosts, misses, immunities, cures and suppression/resumption. Mud-Slap, Double Team and Sweet Scent use generic damage/boost handlers. Simultaneous forced replacements now skip the intermediate Update exactly as the reference does, preserving tied-speed RNG.
- Native Dry Skin Water absorption before accuracy (including full HP, allied Surf and Protect precedence), defender-owned Fire base-power modifier at priority 17 with exact 5120/4096, and rain healing/sun damage at one eighth maximum HP. Eleven additional legal complete fixtures require actual absorption, healing, Fire power, weather damage, suppressed inactivity and switch-out resumption. The target-equals-source exception is implemented; no legal starting-holder self-target Water hit is claimed as complete-battle coverage. Generic ability acquisition and other unported interacting effects remain guarded.
- Native Water Absorb, Volt Absorb, Earth Eater, Sap Sipper and Motor Drive. TryHit absorption precedes type immunity, natural powder immunity and accuracy, with Protect/Psychic Terrain gates first. Quarter-HP healing and +1 Attack/Speed immunity persist at full HP or +6 stages. Sap Sipper receives Grass powder even on a Grass holder; its registered side-move no-op handlers retain exact speed-tie RNG. Grass side/team moves beyond currently registered legal handlers remain guarded until dependency closure is ported. Discharge and Zap Cannon use generic damage/status metadata. Ten new complete legal cases require actual heals, boosts, immunity, ally hits and tied side-event calls.
- Native Static contact paralysis with an exact 3/10 chance draw, stable DamagingHit target ordering, Rocky Helmet preceding the chance roll, and calls on zero-HP attackers/fainting holders before faint processing. Failed attempts respect existing status, Electric typing and Misty Terrain without revealing the ability; successful status reveals Static before immediate Lum consumption/cure. Fifteen complete legal cases require chance success/failure and these interactions; a separate trace regression verifies the reveal/status/item/cure chronology.
- Native Trick Room with five-turn field duration, toggling off on recast, residual order 27/suborder 1, switching/source-faint persistence and public duration knowledge. Champions action speed uses signed negation under Trick Room, without mainline 13-bit wrapping; direct reference probes cover the arithmetic boundaries. Cached speed refresh now follows queue commitment/insertion and residual boundaries instead of an extra turn-end refresh. Snapshot restore validates room duration and source and permits coexistence with weather.
- Native Electric, Grassy, Misty and Psychic Terrain with move/surge starts, Terrain Extender duration, rejection of same-terrain refresh, replacement and expiration, and coexistence with weather/Trick Room. Grounded Electric/Grass/Psychic power uses exact 5325/4096; Misty halves Dragon power and Grassy halves Earthquake against grounded targets. Electric prevents new sleep, Misty prevents new major status, and Psychic blocks currently ported enemy priority moves against grounded defenders while permitting ally and airborne targets. Grassy heals before Leftovers and before its final duration decrement. Terrain starts/ends preserve TerrainChange sorting/RNG and hide opposing exact durations. Grounding-changing effects, semi-invulnerable interactions, confusion/Yawn and dynamically modified priority remain required alongside their guarded unimplemented handlers. Residual collection includes fainted holders before execution skips them, matching reference tie sorting.
- Inline temporary handler/target/action buffers, exact sorting with heap spill, reusable `candidates_into` and `step_batch_into` buffers, and direct full-joint action validation. Differential RNG/state tests and a sparse/reordered reusable-buffer regression preserve semantics; see `HOT_PATH_AUDIT.md` for measured allocation/throughput limits.
- Versioned native numeric observation encoder with 96 fixed rows, compact category IDs, normalized floats, boolean/known masks and uncapped ragged effects, types, revealed repertoire and typed move-effect metadata. It accepts only PlayerView plus immutable Dex, masks stale unknown payloads and audience-filtered event integers, and excludes outcomes/provenance/RNG. Cached Encoder and caller-owned ObservationBatchBuffers support one native batch call with shared-Dex identity checks, preflight handles, sparse/reordered/dual-view requests and retained high-water storage across empty/smaller batches. All 1,136 teams encode in a 2,048-environment group and match serial output. The numeric schema remains incomplete for legitimately known effect metadata not yet represented in PlayerView and detailed item/ability features absent typed Dex; those fields are explicitly unavailable rather than invented. Python/tensor export remains required.
- Private global effect creation ordering is stored independently of player views and RNG. Ability/item recreation on entry and Mega replacement, accepted status/volatile/side creation, consumed-item clearing and Protect stall restart follow the reference initializer arguments. Schema 4 rejects invalid assigned orders and duplicate retained orders; earlier schemas are rejected. Field weather, terrain and room states remain nonallocating order zero. Redirection sorts local handlers stably by priority, cached speed and ability activation order, without tie RNG. Exact global counters and retained state orders match the pinned reference at every boundary in 32 legal complete battles, including all 20 new Flash Fire/Lightning Rod cases.
- Native Flash Fire absorption changes a shared action-local accuracy sentinel before all spread accuracy draws, stores a persistent sourced volatile, and adds priority-5/suborder-2 Attack/SpA handlers even for conditional no-ops. Alive End removes it; zero-HP faint End does not emit its silent condition End. Mega is emitted before old ability End, then the replacement ability is created. Native Lightning Rod and Storm Drain absorption boost SpA before immunity and accuracy; redirection uses effective move type, generic candidate validity and stable activation order, with tracksTarget/spread bypasses. Twenty additional legal complete cases require actual absorption, caps, Ground/status interactions, selected-holder early exit, tie/reentry and Tailwind reversals, independent Discharge hits, Heat Wave accuracy positive/negative controls, weather/Thick Fat rounding, and Mega acquisition/loss. Storm Drain shares the native callback primitives but has no legal starting or reachable holder in the pinned scope; legal dynamic acquisition remains unverified and required. Shadow Ball and Dark Pulse use the exported damage/secondary declarations.
- A development-only JSONL Showdown oracle adapter with full debug state, pending choices, RNG tracing and snapshots. It is not a training backend.

Validation: 52 Rust integration tests, one Rust unit test and one reference-adapter test pass. Fixtures cover 4,096 raw RNG draws, stats for all collected sets and legal starting species, all catalogue move PP and target classes, 81 speed/event-order cases, 80 target-order handler cases, 3,600 reference recoil-rounding cases, 512 isolated damage cases, HP display boundaries, initialization and recovery, and information-boundary regressions. A 2,048-environment reset/observation test includes all 1,136 training teams. One hundred ninety-two complete synthetic legal reference battles additionally compare every decision boundary: RNG seed, requests, cached action speeds, HP, PP, species, types, abilities, held/consumed items, stats/stages, major status, volatile presence, Mega availability, fainted state and winner. They exercise both sides evolving, post-damage evolution, switch persistence, all six major-status identifiers, secondary effects, flinches, stat changes, healing, retargeting, spread damage, Protect stalling and Struggle recoil, with snapshot restoration during play and complete trace replay. Additional cases cover Intimidate/Defiant/Competitive, Mega ability transitions, stat-drop immunities, recovery on switching, ability-modifier speed ties, berries, Focus Sash, recoil with Rock Head/Reckless, ordinary/contact/spread drain, Liquid Ooze and Big Root interactions. Further fixtures cover simultaneous screens and Tailwind, expiration and repeated casting, extended screen duration, switching, speed ties, Choice Scarf stacking, critical hits, screen breaking and Infiltrator. Fourteen terrain fixtures require all four starts/expirations, ordinary/extended duration, grounded/airborne power and status behavior, Psychic priority protection and exemptions, Grassy healing on its final turn and before Leftovers, replacement and coexistence with weather/Trick Room. Eight additional Trick Room fixtures cover expiration, recasting, simultaneous setters, priority, Tailwind/Choice Scarf/paralysis and switching persistence. Twenty isolated reference action-speed cases cover signed Champions ordering at speed boundaries. Ten initial weather fixtures require all four weather starts and expirations, Weather Ball damage in every weather, sand damage and replacement to occur. Decision-boundary comparisons also check weather state/source/duration and public duration visibility. Decision-boundary comparisons also check side-effect durations and both players' condition knowledge. Drain healing, recoil damage, Liquid Ooze damage, side-effect starts/ends and screen interactions must actually occur in the generated reference corpus. The fixture generator requires specified status and item effects to actually trigger, rather than merely appear in a team. The fixtures are development-only and do not add training teams. Two concurrent native groups of 1,024 environments with 16 workers each complete these fixture battles and match serial snapshots exactly for all 2,048 environments. This test runs the two Rust groups inside one test process; it does not yet verify two Python actor processes or NUMA placement. This is **not** a completed-game throughput measurement with a policy or a full-mechanics parity claim.

## 2026-10-07 continuation: declarative coverage, native masks and the Python boundary

Resumed after the previous agent's usage limit. No architecture was changed and no working implementation was discarded. New verified work:

- **Declarative move executor.** `assets.rs::classify_move` now enables every regulation move whose pinned declaration is exactly expressible by the native executor: no unported callback key, every data field in `HANDLED_MOVE_FIELDS`, and every embedded status/volatile in the ported set. Anything else remains an explicit operational error. `engine/examples/coverage_report.rs` (`bash scripts/cargo.sh run --release --example coverage_report`) prints the exact executable/blocked inventory and the training-pool blockers; it is a cold development instrument, never a battle path.
- **Moves added since the previous report:** the ported `basePowerCallback` family (Acrobatics, Electro Ball, Eruption/Water Spout with exact fractional base power, Flail/Reversal, Grass Knot/Low Kick, Gyro Ball, Hard Press, Heat Crash/Heavy Slam, Hex/Infernal Parade, Last Respects, Power Trip/Stored Power, Rising Voltage), Fake Out and Sucker Punch gating, weather accuracy (Hurricane/Thunder/Blizzard), Grassy Glide priority, Freeze-Dry effectiveness, fixed damage (Seismic Toss, Night Shade, Super Fang, Endeavor), OHKO moves, `willCrit`, `ignoreDefensive`/`ignoreEvasion`, `overrideOffensiveStat`/`overrideDefensiveStat`/`overrideOffensivePokemon` (Body Press, Psyshock, Foul Play), `breaksProtect`, `selfBoost`, `selfdestruct` (`always` and `ifHit`), and multi-hit moves with the reference hit-count sampling, per-hit crit/damage/secondary/DamagingHit phase and per-hit Update events.
- **Items and abilities added:** type-enhancing items, resist berries, Choice Band/Choice Specs, Muscle Band/Wise Glasses, Eviolite, Assault Vest, Light Ball, Leppa Berry, Scope Lens, Leek, White Herb, terrain seeds and related `onUpdate` berries (`item_ports.rs`); Prankster, Rough Skin, Poison Touch, Flame Body and Stamina (`hooks.rs`). Unported items and abilities still raise operational errors.
- **Legal action masks are differentially verified.** The fixture generator records the reference request (kind, per-slot presence/replacement flag, Mega availability, selectable moves with PP, target class, bench destinations and preview roster) and `turns.rs` compares it with the native `Request` at **every** decision boundary of the whole corpus. This closes the largest previous blind spot for player-safe masks; the comparison skips only data the engine intentionally leaves empty for non-actionable slots (a waiting side, or the non-replaced slot of a replacement request) and compares switch destinations as a set because the reference lists them in request-team order while the engine reports stable roster indices.
- **Independent provenance check.** `node engine/tests/verify_turn_fixtures.mjs` replays all 478 fixtures against a freshly booted pinned Showdown and deep-compares every stored boundary: 10,653 decision boundaries verified, 0 mismatches. It is the guard against hand-written or stale expectations.
- **Native submission validator.** `engine/src/legality.rs` validates a submitted six-member team against the pinned format's rules (Obtainable species/moves/abilities, Species Clause, Item Clause = 1, Adjust Level = 50, 31 IVs, 66 Stat Points with a 32-per-stat bound and the uninvested-Serious rejection, no Mythical or Restricted Legendary, Mega-form submissions accepted through their legal base forme). `engine/tests/generate_legality_cases.mjs` records the pinned `TeamValidator`'s own verdict and problem category for 82 cases (61 frozen-pool teams plus crafted mutations of a legal synthetic team), and `engine/tests/legality.rs` asserts the native verdict agrees with the reference on every one. This replaces the previous "static checks are not a complete native validator" gap for the rules this format actually enforces.
- **Corpus:** 478 complete legal battles and 10,653 decision boundaries (`engine/data/turn-fixtures.json`, 254 of them generated per move by `engine/tests/generate_more_move_coverage.mjs`, merged by the exporter). Bugs this corpus caught and that were fixed against the pinned reference: Grass Knot/Low Kick weight thresholds (10x too large), Electro Ball/Gyro Ball using action speed instead of `getStat('spe')`, Memento's `ifHit` faint ordering and status-move case, and fixed `multihit` counts consuming a draw.
- **Python boundary.** `engine/src/python.rs` exposes the batch-oriented `NativeEngine` class (one crossing per batch, GIL released for native work, generation-checked handles, packed observation blobs, candidate/mask queries, snapshot/restore/trace). `scripts/build_python.sh` builds it; `engine/python/pa3_actor.py` is the documented 1,024-environment/16-worker actor with transition, decision, observation, bridge and operational-error accounting; `scripts/run_actor_pair.py` runs the documented two-process, 2x1,024-environment topology with NUMA/affinity placement. Pokémon Showdown is never executed through the binding.
- **Coverage numbers (development instrument, not a readiness claim):** 288/515 regulation moves, 75/223 legal abilities and 84/166 legal items executable; the frozen training pool still has 126 distinct unported moves, 75 abilities and 72 items (plus every unported interaction behind them).

The engine is still **not** training-ready: 227 moves, 148 abilities and 82 items remain explicit operational errors, several interaction families (pivots, guards, redirection moves, confusion, Encore/Taunt/Disable, charge moves, hazards, Transform/called moves) are unported, and the 2,048-environment run above completes only the fraction of battles whose teams avoid unported mechanics.

### Second continuation (same day): item closure, pivots, protect/guard/confusion families, native legality and enforced fixture coverage

- **Held items are closed for the training pool.** `engine/src/items.rs` classifies items from their pinned declaration; a data-driven Mega-Stone path, a native `takeItem`/`setItem` primitive, Knock Off (1.5x only when a removable item is actually taken, no `lastItem` record) and Trick/Switcheroo (Sticky Hold refusal, unremovable restore, empty hand-offs) are implemented and differentially verified. Items: 165/166 executable; the frozen pool now has **0 item blockers**. Two real bugs were found by the new fixtures: a failed Trick/Switcheroo running reference post-move phases (2 extra draws) and the request bench order using live positions instead of the preview pick order.
- **Pivot family.** `uturn`, `voltswitch`, `flipturn`, `partingshot` and `teleport` issue the reference `instaswitch` replacement request (flagged slot only, moves still advertised, no Mega availability, `Wait` on the other side), with Parting Shot's drop-then-flag rule and Teleport's `canTry` gate. `SideState.positions` now mirrors the reference party order (pick order, active first, swap on switch-in). Roughly 770 training-pool slots depend on this family.
- **Protect/guard/confusion families.** Detect, Spiky Shield, Baneful Bunker, King's Shield (shared stall counter and contact punish), Endure (damage clamped at 1 HP after item/berry modification), Wide Guard and Quick Guard (spread/priority blocking with residual order 0 / sub-order 4), and the confusion volatile (exact `random(2,6)` timer, priority-3 BeforeMove ordering, exact self-hit damage, `addVolatile` failure rule for no-effect status moves).
- **Move-flag honesty gate.** Flags such as `cantusetwice` now keep a move an explicit operational error rather than letting it behave as a plain attack.
- **Fixtures are generated from the Rust classifier.** The move corpus generator reads `HANDLED_MOVE_FIELDS`, `PORTED_MOVE_CALLBACK_KEYS`, the handled status/volatile/flag sets and `MoveBehavior::compile` directly, so engine and corpus cannot drift. `engine/tests/fixture_coverage.rs` additionally fails the build when an executable move has neither a differential fixture nor an explicit documented exemption (the exemption list currently names 16 moves with their reasons and is checked for staleness).
- **Force-switch phazing (`roar`, `whirlwind`, `dragontail`, `circlethrow`) is deliberately still unsupported.** The native mechanism is written but the reference consumes two end-of-turn target re-resolution draws the native queue does not model; the fixtures therefore stay explicit operational errors and the exact reproducer, draw trace and remaining steps are recorded in `NEXT_FORCE_SWITCH_REQUIREMENTS.md`.

Verified after this batch (all commands re-run by the primary agent with every worker stopped):

| Check | Result |
|---|---|
| `bash scripts/cargo.sh test --locked --release --no-fail-fast` | 65 tests across 16 binaries, 0 failures |
| `bash scripts/cargo.sh clippy --locked --all-targets -- -D warnings` (also `--features python`) | clean |
| `node engine/tests/verify_turn_fixtures.mjs` | 529 fixtures / 11,653 decision boundaries re-verified against a fresh pinned Showdown, 0 mismatches |
| `engine/data/manifest.json` digests | all six match the exported files |
| `engine/python/test_binding.py` | 64/64 natural battles, packed observations, 1,500+ games/s in-process |
| Coverage | 310/515 moves, 75/223 abilities, 165/166 items; training pool: 109 move / 75 ability / 0 item blockers |

Coverage is still far from the objective. The next highest-leverage work, in order: the 148 remaining legal abilities (Unburden, Hospitality, Flower Veil, Armor Tail, Good as Gold, Unnerve head the training-pool list), the volatile/condition families (Encore, Taunt, Disable, Yawn, Substitute, Leech Seed, Helping Hand, Follow Me/Rage Powder), two-turn and charge moves, hazards, the force-switch RNG accounting, then observation tensor export and the 2,048-environment NUMA actor measurement. No readiness claim is made and no training has started.

## 2026-10-07 items family (primary agent)

The held-item interaction family is ported and differentially verified against
the pinned reference:

- **Data-driven item classification.** `engine/src/items.rs` classifies items
  whose whole pinned declaration is expressed by ported primitives under the
  same rule as moves: a Mega Stone (base-form mapping plus the `onTakeItem`
  refusal) becomes `Item::MegaStone` only when every declared field is inert and
  the sole callback is its own `onTakeItem`. Legal item coverage is now
  `165/166`; only `metronome` remains an explicit operational error. The loader
  still rejects any item whose declaration contains an unported field, so
  `onTakeItem: false`-style items cannot silently slip through.
- **`Pokemon#takeItem` primitive.** `BattleState::take_item_checked` is the
  silent reference take (`Empty`/`Refused`/`Taken`), `take_item` adds the public
  End event for removal moves, `restore_item` is the raw failed-swap assignment,
  and `give_item` is the reference `setItem` give. The reference never records a
  taken item in `lastItem`; only `useItem`/`eatItem` do, and that distinction is
  now enforced by the corpus.
- **Knock Off** (`moves:knockoff.onBasePower` + `onAfterHit`): the 1.5x boost is
  gated on the same TakeItem check as the removal, a Mega Stone on its own base
  form neither boosts nor is removed, and an alive user removes the item of
  every damaged target after the DamagingHit event.
- **Trick / Switcheroo** (`onTryImmunity` Sticky Hold gate + `onHit`): both
  items are taken, the swap only completes when neither take is refused and at
  least one item exists, refused swaps restore both items and let the move fail
  without the reference's post-move phases, and the two hand-offs emit `-item`
  or the silent `-enditem` exactly as the reference does.
- **Choice-lock lifecycle.** The reference `choicelock.onDisableMove` drops the
  lock lazily once the holder no longer has a Choice item (Knock Off, Trick) or
  no longer knows the locked move; `end_turn`/`makeRequest` now mirrors that.
- **Request bench order.** `BattleState::bench` now filters the preview pick
  order (`selected_order`) instead of the live `positions[2..]` arrangement, so
  the engine's own snapshots no longer fail the request/world validator after a
  switch; the reference also lists switch destinations in request-team order.

New development instruments (cold paths, never used by training):
`engine/examples/pool_run_report.rs` (actual first operational blocker per
frozen training team under a deterministic native policy),
`engine/examples/snapshot_probe.rs` (first snapshot-validation failure in the
corpus, with optional item/HP dump), and `debug_fixture`'s `PA3_EVENT_LIMIT`.
Differential coverage for the family comes from
`engine/tests/generate_more_item_interactions.mjs`, which the exporter merges
into `turn-fixtures.json` (10 complete legal battles: Knock Off against
Leftovers/Sitrus/Mega Stone/Choice Scarf/no item, Trick swap/refused-stone/
give-when-empty/both-empty/stone-to-other). With those fixtures in the corpus
the boundary compare was green against the pinned reference, and
`node engine/tests/verify_turn_fixtures.mjs` re-verified every stored boundary
(532 fixtures / 11,682 decision boundaries) from a fresh pinned Showdown.
Remaining red in the shared tree at report time: the in-flight pivot fixtures
(Parting Shot / Flip Turn) owned by the move workstream.

The engine advances complete battles for the currently ported subset. The full native event dispatcher and all remaining move/ability/item/condition handlers, remaining weather/terrain interactions and remaining side effects, called/transformed moves, pivots and other complex event sequences, all Mega interactions, remaining observation metadata and tensor bindings, PyO3 methods, affinity placement and full-regulation completed-game performance measurement remain required. Native subset replay measurements and the code-level hot-path audit are recorded in `HOT_PATH_AUDIT.md`; they do not establish full-mechanics or model throughput. Current static team checks are not yet a complete native replacement for the reference team validator. More targeted edge cases and regulation-wide interaction regressions are required. The dependency catalogue's 2,371 callback entries are source inventory (including nested-condition entries), not implemented native effects. Neither data-table coverage nor passing the primitive tests closes the engine goal.

Next work: harden the supported snapshot state shapes/counters described in `NEXT_SNAPSHOT_CLOSURE_REQUIREMENTS.md`, port normal-move type conversion using the next-family requirements and remaining weather interactions, expand native callback/declarative-effect coverage, strengthen ordered event dispatch and interaction fixtures, complete native team legality and the remaining numeric observation features, then verify regulation-wide battles and hidden-information invariance. Keep the regulation-wide scope even for species absent from the training pool. Wire Python only to the native engine. No model or training run has started.


## 2026-10-07 continuation: `selfSwitch` pivots and the reference party-order model

Same session, after the declarative/mask/Python work below. No architecture was
changed; the additions are additive and differentially verified.

- **Native pivots.** `uturn`, `voltswitch`, `flipturn`, `partingshot` and
  `teleport` now execute. `Move.self_switch` carries the pinned
  `selfSwitch` declaration (`assets::SelfSwitch`), `PokemonState.switch_flag`
  mirrors the reference `switchFlag`, and after the action the post-action block
  turns the flag into the reference's `instaswitch` switch request
  (`RequestKind::Replacement` with only the flagged slot actionable, the
  pivot's move list still advertised, no Mega availability, `Wait` for the
  other side). The chosen switch runs immediately, before the partner's queued
  action, exactly like the reference `instaswitch` order.
- **Parting Shot** implements its own `onHit` drop (the pinned declaration has
  no `boosts` field) and cancels the pivot when the Attack/Sp. Atk drop fails,
  matching the reference's `delete move.selfSwitch`. **Teleport** runs its
  `onTry` gate (`canSwitch`) before any hit step.
- **Reference party order.** `SideState.positions` mirrors `side.pokemon`: the
  four selected members in pick order, with the active slots first and a swap
  on every switch-in. Reserve lists in requests, random drag-in sampling and
  future phazing now follow reference order instead of a fixed roster order.
- **Switch-in phases.** `switch_in` now distinguishes a voluntary switch
  (`BeforeSwitchOut` + Update), a `selfSwitch` pivot (neither: the reference
  already ran `BeforeSwitchOut` when it issued the request and set
  `skipBeforeSwitchOutEventFlag`) and an `isDrag` switch (neither, and the
  `runSwitch` SwitchIn event runs synchronously rather than through the queue).
  Getting the pivot case wrong cost one Update's tie-shuffle RNG; the corpus
  caught it immediately.
- **`forceSwitch` (phazing) stays an explicit operational error.**
  `roar`/`whirlwind`/`dragontail`/`circlethrow` have a complete native
  implementation (flag, `DragOut` step, post-action phazing with a uniform
  reserve sample, drag-mode switch-in), but the reference consumes two RNG
  draws in the end-of-turn target re-resolution that the native queue does not.
  The gate and the exact evidence are recorded in
  `NEXT_FORCE_SWITCH_REQUIREMENTS.md`; the family is not claimed.
- **Fixture generator.** A single-move holder could exhaust its PP and strand
  the battle with no legal choice (`shelter`); probe holders now also receive
  one more implemented move they can legally learn. A stale `fakeout`
  exemption was removed from the coverage guard.

Verified state after these changes: `bash scripts/cargo.sh test --locked
--release` is green (all suites, including the 527-fixture corpus), `clippy
--lib --tests -D warnings` is clean, `node engine/tests/verify_turn_fixtures.mjs`
re-verifies 529 fixtures / 11,653 decision boundaries against a fresh pinned
Showdown with zero mismatches, `.venv/bin/python engine/python/test_binding.py`
still runs 64/64 natural battles through the PyO3 batch binding, and coverage
is 310/515 moves, 75/223 abilities, 165/166 items executable. The engine is
still not training-ready and no readiness claim is made.


### Status snapshot (2026-10-08, second pass, verified)

| Area | Value |
|---|---|
| Regulation scope | 293 starting species, 97 permitted battle forms (82 Mega), 515 allowed moves, 223 legal abilities, 166 legal items, 0 unresolved candidates |
| Moves executable | **347/515** (338 with a differential witness; the remaining 9 have documented exemptions in `engine/tests/fixture_coverage.rs`) |
| Abilities executable | **121/223** (runtime gate `Ability::is_ported`) |
| Items executable | **165/166** - the remaining entry is the legal held item **Metronome**, deliberately unimplemented and reported as an operational error |
| Dynamic/reachable closure | generated by `scripts/dynamic_closure.mjs` (33 callers); Copycat / Transform / Sleep Talk / Instruct / Snore and Trace-like abilities remain blocked |
| Training pool | **627/1136 teams complete at least one natural battle (55.2%)**; 80 move / 38 ability / 0 item distinct blockers |
| Differential corpus | **605 complete legal battles / 13,062 decision boundaries**, zero mismatches, independently re-verified by `node engine/tests/verify_turn_fixtures.mjs` |
| Tests / lint | 20 test binaries green; clippy clean with and without `--features python`; manifest digests verified |
| Readiness | `engine/examples/readiness_check.rs` exits **non-zero** (NOT READY): moves, abilities, the Metronome item, dynamic closure and full-coverage throughput still fail |

`engine/data/known-mismatches.json` is now history, not a work list: all three
tracked entries are `fixed` and merged back into the corpus
(`imprison_shared_pool` earlier; `lock_crossfire` and `move_yawn_3330` in this
pass, below). The ledger keeps the reproduction and the resolution note for
each. The staged scripts `engine/tests/staged_delayed_status.mjs` /
`staged_roost_yawn.mjs` were promoted to the live generators
`generate_more_delayed_status.mjs` / `generate_more_roost_yawn.mjs` (their
fixtures are merged) and are now redundant copies.

### Closed this pass

- **`lock_crossfire`.** The two missing turn-2 `speedSort` calls were attributed
  by stack capture: the size-2 `Battle.runEvent <- runMove` sort is the
  BeforeMove handler list (Taunt 5 + Disable 7, not tied, zero draws, ordering
  already reproduced by the hand-ordered refusal gates), and the size-3
  `Battle.runEvent <- endTurn` sort is the `DisableMove` handler list of the
  holder carrying Taunt + Encore + Disable: three order-less Condition handlers
  at sub-order 2 with the same cached speed, i.e. fully tied and worth the two
  shuffle draws that were the whole divergence. `end_turn` now collects that
  handler set per active Pokémon (holder status/volatiles/ability/item plus
  each live active foe's `onFoeDisableMove`) and speed-sorts it in side/slot
  order before the existing flag pass, which stays authoritative for state.
  Membership and sub-orders are derived from the pinned declarations at Dex
  load (`NativeEffects::disable_move_*`) and fail closed if the declared set
  changes.
- **`move_yawn_3330`.** Yawn's `onTryHit` gate compared the incoming *move* id
  (950) against the yawn *condition* id (165), so the gate never ran. Two
  further comparisons had the same id-namespace mistake (the Poison-Toxic
  invulnerability exemption and the Helping Hand invulnerability short-circuit).
  `NativeEffects::{yawn_move, toxic_move, helping_hand_move}` now carry the
  move ids separately from the same-named conditions. With the gate live, the
  reference's behaviour is reproduced exactly: when the chosen target faints
  before a slower Yawn resolves, both sides retarget with the two `getTarget`
  samples (`getActionSpeed`/`resolveAction` during the mid-turn queue re-sort
  and `runMove` at execution — the native already did both) and the move then
  fails against the statused foe instead of applying a volatile.
  A sweep of every `dex.effects.*` id comparison in the engine (both operand
  orders plus `contains`/`matches!`) found no further cross-namespace
  comparisons: the three above were the whole class.

### Status snapshot (2026-10-07, continuation checkpoint, verified)

| Area | Value |
|---|---|
| Regulation scope | 293 starting species, 97 permitted battle forms (82 Mega), 515 allowed moves, 223 legal abilities, 166 legal items, 0 unresolved candidates |
| Moves executable | **368/515** (358 with a differential witness; the rest documented exemptions in `engine/tests/fixture_coverage.rs`) |
| Abilities executable | **135/223** (runtime gate `Ability::is_ported`) |
| Items executable | **165/166** - the remaining entry is the legal held item **Metronome**, deliberately unimplemented and reported as an operational error |
| Dynamic/reachable closure | generated by `scripts/dynamic_closure.mjs` (33 callers); Copycat / Transform / Sleep Talk / Instruct / Snore and Trace-like abilities remain blocked |
| Training pool | **936/1136 teams complete at least one natural battle (82.4%)**; 848 teams statically complete |
| Differential corpus | **647 complete legal battles / 13,843 decision boundaries**, zero mismatches, independently re-verified by `node engine/tests/verify_turn_fixtures.mjs` |
| Tests / lint | 21 test binaries green; clippy clean with and without `--features python`; manifest digests verified |
| Readiness | `engine/examples/readiness_check.rs` exits **non-zero** (NOT READY): moves, abilities, the Metronome item, dynamic closure, all-teams coverage and full-coverage throughput still fail |

Ported since the previous snapshot (each with reference-generated fixtures and the
full gate set green): Emergency Exit, Toxic Debris + the Toxic Spikes entry-hazard
primitive (which also enabled the whole binding-move family via `partiallytrapped`),
Trace, Fire Mane / Aura Guard / Spicy Spray / Eelevate / Parental Bond,
Glaive Rush, Moody, Moxie, the AfterFaint primitive, Shadow Tag with the request
trapping flags, Terrain Pulse, Clangorous Soul, Perish Song, Population Bomb and
Triple Axel (multiaccuracy), Leech Seed, Soak and Double Shock (in-battle type
changes), Protean / Libero, First Impression, After You, Haze, Psych Up, Pickpocket
and Poltergeist. Perish Song and Poltergeist also carry knowledge-boundary coverage
(`engine/tests/poltergeist.rs`) for effects whose only visible trace is an event,
not a state delta.

## Remaining implementation

## 2026-10-07 continuation: ability batches, support moves and the interaction corpus

Resumed after the previous agent's usage limit. No architecture changed; every
addition is additive, gated by `Ability::is_ported`, and differentially
verified against the pinned reference at every decision boundary.

### Ported this session (39 ability names)

- **Stat / power / damage modifiers:** Guts, Marvel Scale, Fur Coat, Grass Pelt,
  Heatproof, Water Bubble, Punk Rock, Fluffy, Purifying Salt, Sniper, Super
  Luck, Sheer Force, Compound Eyes.
- **Priority, redirection and protection:** Armor Tail, Queenly Majesty, Gale
  Wings, Good as Gold, Soundproof, Bulletproof, Telepathy, Sturdy, Damp.
- **On-hit reactions and status:** Justified, Weak Armor, Gooey, Effect Spore,
  Thermal Exchange, Limber, Immunity, Insomnia, Magma Armor, Cursed Body’s
  requirement note only (not ported).
- **Speed / support / misc:** Unburden, Unnerve, Hospitality, Friend Guard,
  Contrary, Mirror Armor, Unaware, Scrappy, Flower Veil, Quick Feet, Magic
  Guard, Poison Heal.

Highlights of the exact reference semantics implemented: Contrary inverts the
boost table before capping; Mirror Armor reflects each negative stat back to a
living source and skips its own reflected boosts; Flower Veil guards Grass-type
allies from external stat drops and statuses; Unaware zeroes the opposing
offensive/defensive stage in the damage and accuracy stages; Sheer Force
deletes the action's secondaries and self effect before they can draw; Unburden
and Quick Feet chain into the reference `spe` computation, with Quick Feet
suppressing the paralysis halving; Magic Guard refuses every non-move damage
source and Poison Heal converts poison residual damage into a 1/8 heal.

### Support moves

`helpinghand`, `followme` and `ragepowder` are executable. Helping Hand stores
its stacking BasePower multiplier in the volatile (priority 10, condition
sub-order 2); Follow Me and Rage Powder redirect single-target moves through
the existing fast-exit redirect dispatch with priority 1 and the reference
`onFoeRedirectTarget` collection rule (only the attacker's foes, unlike
Lightning Rod/Storm Drain's `onAnyRedirectTarget`). Rage Powder is skipped by a
powder-immune attacker. Damp's `onAnyTryMove` gate runs at the reference
TryMove stage — after PP deduction, before the unsupported-move error — so a
blocked Misty Explosion/Explosion never becomes an operational failure while a
Damp holder is active, and the action still runs its single Update.

### Differentiation status

- `engine/tests/generate_ability_interactions.mjs` + `tests/ability_interactions.rs`
  hold 30 generated interaction fixtures. Every *ported* ability's fixture is
  compared at every decision boundary (RNG seed, HP, status, boosts, stats,
  types, items, volatiles, side/field state and request masks); `KNOWN_MISMATCHES`
  is empty. Fixtures for unported abilities are smoke-walked with the feature
  swapped for a neutral ported ability and automatically become full comparisons
  when the port lands.
- `engine/data/turn-fixtures.json` gained auto-generated fixtures for every
  newly enabled move, including helpinghand/followme/ragepowder.
- `engine/tests/debug_fixture.rs` accepts `--ability-corpus` to replay the
  ability-interaction corpus.

Coverage after this batch (development instrument): 313/515 moves, 118/223
legal abilities, 165/166 items; the frozen training pool runs 109 complete
natural battles with its earliest blockers now led by direclaw, encore,
throatchop, Fairy Aura, Cursed Body, Electro Shot and Solar Beam. The engine is
still not training-ready: 202 moves, 105 abilities and the remaining
interaction families (Encore/Taunt/Disable, hazards, charge/recharge moves,
Transform/called moves, Magic Bounce, Mold Breaker, Emergency Exit, Trace,
Disguise, Illusion, Protean/Libero, Stance Change, Ice Face, Zero to Hero and
the rest) remain explicit operational errors.

## 2026-10-07 continuation: the volatile selection-lock family (Encore, Taunt, Disable, Imprison, Torment, Cursed Body)

The held-item/pivot/guard work left `encore` as the single largest
training-pool blocker (247 of 1,136 teams). The whole selection-lock family is
now native and differentially verified; nothing else was touched.

- **Volatile lifecycle.** `NativeEffects` carries the five pinned condition IDs
  plus `mefirst` and `mentalherb`. `PokemonState.last_move` mirrors the
  reference `Pokemon#lastMove` (set in `moveUsed`, i.e. after BeforeMove and
  after PP deduction, including a failed or missed move and Struggle) and is
  cleared by `clearVolatile` on switch-out and faint. `SNAPSHOT_SCHEMA` is now
  7; restore validates each new volatile's shape (Encore/Disable hold a move
  that is still in the repertoire, Taunt 1..4, Imprison self-sourced and
  untimed, Torment untimed) and rejects malformed payloads.
- **`onStart` semantics.** `BattleState::start_selection_volatile` transcribes
  each reference `onStart`: Encore fails on no last move, `failencore`,
  Z/Max or 0 PP, stores the move and extends its duration by one when no action
  is queued; Taunt extends when the holder is already active and has not queued
  an action; Disable drops one tick when the target has not acted yet or when
  Cursed Body fires mid-move, and fails without a recorded move or with 0 PP;
  Imprison and Torment only record state. `addVolatile`'s no-`onRestart`
  failure and the resulting move failure are preserved.
- **Champions Encore queue change.** When the encored target already queued a
  different move (and holds no Mental Herb), `BattleState::change_action`
  reproduces `BattleQueue#changeAction`/`insertChoice`: cancel the actor's
  queued actions, rebuild the move action with `getActionSpeed`, re-resolve its
  target with `getRandomTarget`, then insert it by `comparePriority` with the
  reference's `random(first, last+1)` tie-break draw. Both draws are visible in
  the corpus RNG comparison.
- **Request masks.** `end_turn`'s disable pass now also applies the reference
  `onDisableMove`/`onFoeDisableMove` handlers: Encore disables every other move
  while the encored move is still known, Taunt every Status move except Me
  First, Disable and Torment their recorded move, and an active opposing
  Imprison hides every move the imprisoning Pokémon knows. Imprison's hidden
  marker keeps the served choice illegal (`disabled: true`); the reference's
  client-side `getMoves` can display it as enabled to the last active slot, but
  the server rejects that choice, so the legal mask excludes it.
- **BeforeMove ordering.** The refusals run in reference priority order:
  mustrecharge 11, sleep/freeze 10, flinch 8, Disable 7, Throat Chop 6, Taunt 5,
  Imprison 4, confusion 3, paralysis 1. Imprison's foe-side gate refuses a move
  that was committed before Imprison landed, which the corpus now exercises.
- **Cursed Body.** `Ability::CursedBody` is ported: an exact 3/10 draw during
  the DamagingHit phase disables the attacker, the roll is skipped while the
  attacker already holds Disable or when the hit was Struggle/Max/future, and
  the ability is revealed.
- **Fixtures.** `engine/tests/generate_more_disable_family.mjs` (7 complete
  legal battles, one per move: Encore after and before the target's action,
  Taunt, Disable, Imprison with the mid-turn refusal, Torment, Cursed Body)
  writes `engine/data/more_disable_family.json`, which the exporter merges.
  Every fixture carries its `coverage.move` witness for `fixture_coverage.rs`.
  `generate_turn_fixtures.mjs` and `verify_turn_fixtures.mjs` now document the
  stored convention: the world move list with the raw disable flag (served
  choice legality), not the client-side display value.

Verified: `bash scripts/cargo.sh test --locked --release --no-fail-fast` (69
tests, 0 failures), `clippy --all-targets -D warnings` clean with and without
`--features python`, `node engine/tests/verify_turn_fixtures.mjs` re-verifies
558 fixtures / 12,160 decision boundaries against a freshly booted pinned
Showdown with 0 mismatches, and the coverage instrument reports 326/515 moves,
119/223 abilities, 165/166 items. The training pool's first blockers are now
led by the charge/recharge and Mega-ability families; 265 teams complete at
least one natural battle. No readiness claim is made and no training has
started.

## Local reproduction

```bash
bash scripts/setup_rust.sh
bash scripts/setup_reference.sh
npm run export:engine
npm run test:engine
bash scripts/cargo.sh clippy --locked --all-targets -- -D warnings
```

Rust 1.90.0 is pinned in `rust-toolchain.toml`; resolved dependencies are in the root `Cargo.lock`. The toolchain lives under ignored `.tools/`, and builds under ignored `target/`. Host Python, torch, CUDA, drivers, GPU power settings and services are unchanged. No GPU work has been performed.

The exporter and Rust ports derive from the pinned Showdown source, whose MIT notice is preserved in `engine/data/SHOWDOWN-LICENSE`. The third-party `jackson-nestelroad/battler` source was inspected in a temporary directory for reuse. Its general Gen9 support does not establish pinned Champions parity, and its interpreted effects/nightly requirements do not fit this project's selected native representation directly; no battler code was incorporated.

### Boost reset fixtures (Haze / Psych Up)

The declarative move corpus gave Haze and Psych Up automatic fixtures, but
neither battle ever held a non-zero boost, so the ported semantics were not
differentially exercised. `engine/tests/generate_more_boost_reset.mjs` adds
four complete legal battles that force the mechanic and are merged into the
corpus by the exporter:

- `boost_reset_haze_clears_positive` - Swords Dance (+2 Attack) and Calm Mind
  boosts on both sides are all cleared on the Haze turn and stay zero.
- `boost_reset_haze_clears_negative` - Icy Wind's -1 Speed on both of the
  Haze user's side's actives is cleared, and the Protect on the Haze turn
  keeps the drop from being re-applied.
- `boost_reset_psychup_copies_positive` - the slower Psych Up user copies the
  target's two Swords Dance stages and never holds an Attack stage before the
  copy.
- `boost_reset_psychup_copies_negative` - Psych Up copies the target's -1
  Speed stage after Icy Wind (the target's ability must not intercept the
  drop; a Clear Body target would silently void the check).

Each fixture records the reference request at every boundary, so the served
legal-action mask is compared as well. Verified after the merge: 641 fixtures
/ 13,687 decision boundaries re-verified by `node engine/tests/verify_turn_fixtures.mjs`
against a freshly booted pinned Showdown with zero mismatches; the full Rust
suite and clippy (with and without `--features python`) are green; the
manifest digest matches the merged corpus.
