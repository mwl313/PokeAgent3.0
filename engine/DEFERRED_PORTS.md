# Deferred ports (development notes)

Notes for mechanics that were prototyped but deliberately not landed, so the
next attempt starts from evidence instead of from scratch. Nothing here is a
readiness claim.

## Raging Bull (Tauros / Paldea forms, pool weight 1) - LANDED 2026-10-08

The second attempt succeeded after closing the `move_magicpowder_29910` ledger
entry. The port is the same sketch as the first attempt: a TryHit-stage screen
shatter (reflect / light screen / aurora veil, ordered after the higher-priority
guards and ability absorptions and before type immunity, accuracy and the decoy
intercept) plus the form-driven `onModifyType` (Combat/Fighting, Blaze/Fire,
Aqua/Water; plain Tauros and a Metronome-item caller keep Normal). Four
reference scenes pin it: screens shattered through a substitute, the shatter
ordering versus the damage line, the Combat form's Fighting type against a pure
Normal foe, and an Aqua-form hit absorbed by Water Absorb that keeps the
screens. No ledger entry is open.

## Mega Sol (pool weight 6) - LANDED 2026-10-08

`abilities:megasol.onWeatherModifyDamage` re-runs Sunny Day's weather damage
modifier for the holder, but the reference implements the outer half in
`Pokemon#effectiveWeather`: while the holder is the active Pokémon and the
weather source effect is that ability, a move or a weather, the effective
weather becomes `sunnyday`. That helper is per-Pokémon, and the pinned data uses
it for the weather damage modifier, Weather Ball, the weather-scaled heals
(Synthesis / Moonlight / Morning Sun), weather accuracy moves and the
weather-keyed abilities, so porting Mega Sol means threading a per-Pokémon
effective weather through every one of those call sites (today the engine has a
single battlefield-level `effective_weather`). Do not land the damage modifier
alone: a Mega Sol holder with any of the other weather interactions would then
silently diverge from the reference. Start by mirroring `effectiveWeather` and
its call sites as one coherent family with a fixture per interaction.

Landed with `mon_weather(dex, actor)` as that per-actor view: the weather damage
modifier derives its rule from the move user (both sides of the relayed sun
handler read the same view) and Solar Beam/Solar Blade skip the weak-weather
halving for a Mega Sol user, which was the first divergent damage input. The
held-out Weather Ball scene is merged and the ledger is empty.

## Symbiosis (pool weight 4) - LANDED 2026-10-08

`abilities:symbiosis.onAllyAfterUseItem` needs the `AfterUseItem` event, which
the engine does not raise today (item consumption is modelled per effect:
berries, Focus Sash, consumable ability/item paths). Port the event at every
consumption site first, then the ally pass-through (skip a switching ally, take
the holder's item through the `TakeItem` refusals, `setItem` on the ally, and
give the item back when either refuses).

Landed as `BattleState::after_use_item`: `consume_item` now runs the reference's
speed-sorted `AfterUseItem` handler set (the user's own Unburden plus every live
ally's Symbiosis), with the hand-off, the receiver-side TakeItem refusal, the
raw `source.item` rollback and the item's Start on the recipient. Four fixtures
(hand-off, two controls, the Unburden handler-set scene) are merged.

## Fling (pool weight 1) - LANDED 2026-10-08

The item table (`items:<id>.fling`) is loaded into `NativeEffects::fling_items`
with the base power and the payload kind; `moves:fling.onPrepareHit` arms the
action from the held item, refuses empty hands and fling-less items (Normal Gem)
before the hit loop, and `conditions:fling.onUpdate` consumes the thrown item on
the next Update, which raises the same `AfterUseItem` set Symbiosis/Unburden
answer.

The payload family is ported too: a thrown Berry is eaten by the target through
its `onEat`, the Mental Herb clears the target's taunt/encore/torment/disable/
heal-block volatiles, the White Herb clears its negative boosts, and a thrown
`fling.status`/`volatileStatus` item applies through the ordinary `secondaries`
phase (the reference rolls `random(100)` for every entry, so the always-on
payload still consumes its roll). Seven fixtures plus the auto-generated
`move_fling_15188` scene are merged; only an item whose `fling.effect` callback
has no native port would still raise `fling payload <item>`.

The earlier hold-out was a modeling error, not a queue bug: a refused
Fling runs **no** `Update` of its own (unlike the Damp `TryMove` abort,
which runs one), so the extra `each_update` in the first draft shifted every
later draw. Mirrored speed ties are fine (see the `tiecheck_*` probe scenes in
`/tmp/pa3_generate_more_tiecheck.mjs`: the queue's Fischer-Yates break matches
the reference for mirrored leads and partners).

## Witness pass (2026-10-08) - 35 ported pool abilities witnessed

`engine/tests/generate_more_witness_tail.mjs` adds one scene per ability that no
corpus exercised before (cross-corpus check over `turn-fixtures.json` coverage
tags plus the interaction corpus): pixilate, emergencyexit, torrent, speedboost,
regenerator, toughclaws, moody, solarpower, swiftswim, liquidvoice, solidrock,
snowcloak, infiltrator, reckless, synchronize, voltabsorb, libero, noguard,
shellarmor, marvelscale, purepower, sandforce, sandveil, sapsipper, strongjaw,
superluck, swarm, filter, hydration, liquidooze, plus, minus, motordrive,
owntempo, quickfeet, slushrush. The boundary-by-boundary comparison is the real
witness; each scene's `verify` only asserts that its precondition happened.

The pass immediately paid off: the Own Tempo scene caught a live divergence
(the native rolled the confusion timer and applied the volatile instead of
refusing it), fixed in `hit_effect` together with the Safeguard gate it shares.
Remaining witness gap after the pass: none.

## Type-addition moves (2026-10-08) - deferred pending an observation decision

Forest's Curse, Trick-or-Treat and Reflect Type stay explicit operational errors
for now. Their battle behaviour is fully specified in the pinned source
(`onHit` calls `addType` / copies `getTypes(true)` plus `addedType`), but the
reference keeps the added type in a *separate* `addedType` field: `pokemon.types`
(what the corpus records and what the player-facing `PublicPokemon.types` holds)
only ever contains the base types, and the added type is announced with a
separate `-start ... typeadd` message.

Porting them faithfully therefore needs one of:
1. an `added_type: Id` field on `PokemonState` with `effective_types` (and the
   ~40 `mon.types.contains` call sites) switched to the effective list while the
   knowledge/tensor keeps the base list, plus a new public feature or event for
   `typeadd` so the observation layer does not silently drop public state; or
2. a fixture-format extension (`types_added`) with the generator and the turn
   test comparing both lists.

Both are observation-contract decisions, not battle-mechanic ones, so they wait
for the same review as Illusion's per-viewer identity. Ingrain and Octolock were
ported in the same family because they only need the existing marker/trap and
grounded models.

## Focus Punch / Beak Blast (2026-10-08) - held out behind an undiagnosed divergence

Both moves are ported in a local WIP patch
(`tmp/focuspunch_beakblast_src.patch`, engine/src only, 13 KB): the
`priorityChargeMove` action is generalised from Chilly Reception to the move's
own condition, Focus Punch gets its `beforeMoveCallback` gate and the flinch
refusal, Beak Blast burns contact attackers through the target's `Hit` handlers
and drops its marker in its own `AfterMove`, and the snapshot validator accepts
both one-turn markers.

They are **not merged** because their two auto-generated coverage scenes
(`move_focuspunch_7606`, `move_beakblast_30327`) expose a draw-count divergence
in turn 1 that the corpus test catches at the first boundary:

* the reference spends 59 draws in turn 1 of `move_focuspunch_7606`, the native
  55 (and 50 vs 49 for `move_beakblast_30327`), with identical draws up to the
  commit sample;
* the shape difference sits in the pre-move phase: the reference's draws 8..16
  are one `getRandomTarget` sample, one `BattleQueue.sort` shuffle and six
  `eachEvent` shuffles before the first move resolves, while the native emits
  five `each_update` draws and one re-sort draw in that window;
* **the divergence reproduces with the whole new family disabled** (forcing the
  priority-charge marker insertion to `continue` keeps the same off-by-N
  counts), so it is pre-existing in the Update/tie structure rather than caused
  by the new code. Chilly Reception scenes pass, so it is not the
  `priorityChargeMove` action itself;
* `tmp/pa3probe/replay_draws.mjs`, `replay_stacks.mjs` (per-draw reference
  attribution) and `PA3_RNG_SITES=1` / `PA3_RNG_DBG=1` (native attribution)
  reproduce the tracing above.

Finer evidence gathered on 2026-10-08 (`tmp/focuspunch_beakblast_src_v2.patch`
adds `PA3_UPDATE_DBG` and `PA3_SORT_DBG` probes plus the same family patch):

* every sort in the reference is an `eachEvent`/`fieldEvent` **of the active
  Pokémon**, keyed by speed with `order`/`subOrder` undefined - i.e. exactly the
  shape `BattleState::each_update` and `field_queue` produce. The reference's
  Update sorts (n=4, 2 shuffles with the 80/80 and 84/84 ties) match the
  native's one for one;
* the divergence is therefore in the **number** of events, not their shapes:
  the native runs 17 `each_update` calls in turn 1 where the reference runs 15
  `Update` events, and the reference spends four shuffles in its residual phase
  (`eachEvent('Weather')` + the field Residual + a trailing Update) that the
  native's `residual`/`weather_upkeep` do not reproduce (its own handler sort
  sees a single handler, `n=1`, and draws nothing);
* `tmp/pa3probe/event_probe.mjs` (wraps `eachEvent`/`fieldEvent`/`speedSort` and
  prints the per-event shuffle count and handler summary) and
  `tmp/pa3probe/replay_draws.mjs` are the tools that produced this; both are
  scratch probes, not repo files.

Next step: make `BattleState::residual` collect one handler **per active
Pokémon** (the reference's `findPokemonEventHandlers(active, 'onResidual',
'duration')`) and run the weather upkeep through an explicit
`eachEvent('Weather')`-shaped sort, then re-run both scenes and check the
per-action draw counts against `replay_draws.mjs`.

## Queue-tie / Focus Punch prototype - preserved, not landed (2026-10-08)

A parallel workstream applied a Focus Punch / Beak Blast + queue-tie prototype
to the root checkout during the 2026-10-08 session. It was removed from the
working tree to keep `mac/long-horizon-engine-tail` green and is preserved in
two places:

* the root stash entry in `git stash list` ->
  `foreign tie-sort/focuspunch WIP preserved 2026-10-08 15:46` (save it to a
  patch before dropping; it also carries the scratch tooling below);
* the worktree `/Users/leah/Projects/pa3-tie` (branch `wip/tie-sort`, based on
  `7c29a2c`), whose index holds the same prototype plus an
  `engine/src/queue.rs` tie-sort change.

Prototype contents (all evidence, no verification): `moves:focuspunch.*` and
`moves:beakblast.*` callback keys, the `priorityChargeMove` generalisation from
Chilly Reception, Focus Punch's `beforeMoveCallback` gate and flinch refusal,
Beak Blast's contact burn and `AfterMove` marker drop, the snapshot validator
entries for both one-turn markers, `debug_fixture --file`, and
`MOVE_FIXTURE_ONLY` / `MOVE_FIXTURE_OUT` hooks in
`generate_more_move_coverage.mjs`.

It is **not** mergeable as-is: with the prototype applied the regenerated
`move_beakblast_30327` scene fails at decision 3 (native P2 mon 0 hp 165 vs
reference 167), which is the same residual/Update event-count divergence
described in the Focus Punch / Beak Blast section above. Land the residual
handler collection fix first, then re-apply this prototype and re-run both
scenes.
