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

## Fling (pool weight 1) - WIP, blocked on a queue tie-break divergence

The pinned plain-item path is implemented and verified locally (the item's
`fling` base power arms the action, the marker volatile's `onUpdate` consumes
the item and runs the same `AfterUseItem` set, empty hands and fling-less items
refuse the move before the hit loop, and a Berry/status/herb payload errors
explicitly). The WIP patch and its generator live outside the repo at
`/Users/leah/Projects/pokeagent3_fling_wip.patch`,
`/Users/leah/Projects/pokeagent3_generate_more_fling.mjs.wip` and
`/Users/leah/Projects/pokeagent3_more_fling.json.wip`.

It is held out because its scenes exposed a **pre-existing queue tie-break
divergence**: with two same-speed actions on opposite sides (a mirrored Snorlax
pair at 50, or the Goodra-Hisui pair at 84 in the auto-generated
`move_fling_15188` scene), the native resolves the tied group in the opposite
order from the reference. Boundary seeds still match (the same draws are
consumed) but the actions are assigned to different Pokémon, so the damage
lands elsewhere and a faint cascades into different Update counts. Reproduce
with the WIP fixture `fling_iron_ball_deals_and_consumes_the_item_8300` (P1
Sneasler + Snorlax vs `foeTeam()`) or the coverage scene `move_fling_15188`
(Abomasnow/Fling and a Goodra-Hisui mirror). Next step: instrument the native
commit-time queue assembly (`BattleQueue.insertChoice`'s per-action tie-break
draw plus `BattleQueue.sort`) against the reference's stack for the tied pair;
the reference logs the insert as `BattleQueue.insertChoice` and the sort as
`BattleQueue.sort`, and the native currently makes only one draw for the two.
