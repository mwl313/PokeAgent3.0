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
