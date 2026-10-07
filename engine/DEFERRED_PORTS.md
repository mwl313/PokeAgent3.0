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

## Mega Sol (pool weight 6) - not started

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

## Symbiosis (pool weight 4) - not started

`abilities:symbiosis.onAllyAfterUseItem` needs the `AfterUseItem` event, which
the engine does not raise today (item consumption is modelled per effect:
berries, Focus Sash, consumable ability/item paths). Port the event at every
consumption site first, then the ally pass-through (skip a switching ally, take
the holder's item through the `TakeItem` refusals, `setItem` on the ally, and
give the item back when either refuses).
