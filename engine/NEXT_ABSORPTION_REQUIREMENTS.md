# Next absorption and redirection mechanics

Read-only implementation scout, 2026-10-07. This is a requirements note, not implemented coverage or training readiness. The full pinned M-C and dynamically called effects objective remains unchanged.

## Authoritative reference

Pinned Showdown commit: `14546894d86f9589ac11130c510bbe73b6968665`.
Read `vendor/pokemon-showdown/data/abilities.ts`: Flash Fire at 1342, Lightning Rod at 2344, Storm Drain at 4637. There are no matching overrides in `data/mods/champions/abilities.ts`.
Target selection is `sim/pokemon.ts:getMoveTargets`; ordering is `sim/battle.ts:priorityEvent`, `runEvent`, and `compareRedirectOrder`; effect creation is `initEffectState`. Whole-spread hit phases are `sim/battle-actions.ts:trySpreadMoveHit` and its `hitStep*` helpers.

Legal holders, from the exported validator-derived `engine/data/scope.json`:

| Ability | Starting species | Permitted battle forms |
| --- | --- | --- |
| Lightning Rod | Manectric, Pikachu, Pincurchin, Raichu, Rhyperior | Mega Sceptile |
| Flash Fire | Arcanine, Hisui Arcanine, Armarouge, Ceruledge, Chandelure, Flareon, Houndoom, Ninetales, Typhlosion | None |
| Storm Drain | None | None |

Absence of a starting holder does not remove Storm Drain from the full dynamic acquisition/called-effect closure requirement. A synthetic illegal starting Gastrodon is not acceptable evidence of a legal M-C complete battle. Shared primitive tests can establish handler behavior while legal acquisition tests remain explicitly pending.

## Lightning Rod and Storm Drain

Both have a `TryHit` callback: when target differs from source and move type matches Electric/Water, boost target SpA by one and return null. At +6, boost fails and an ability immunity message occurs; immunity still applies. Status-category moves also qualify. TryHit precedes type immunity and accuracy, so a Ground-type Rhyperior still gains SpA from Electric attacks. Reuse the exact native boost event processing; do not bypass Contrary/Simple/Defiant interactions when those become supported.

Their `onAnyRedirectTarget` callbacks have default priority zero. Nonmatching type and `pledgecombo` return undefined. Convert target kinds `randomNormal` and `adjacentFoe` to `normal` for candidate validity. Consequently a nominal foe-only Electric/Water move can redirect to the attacker's ally. The actor itself is invalid for normal targeting; source immunity is not sufficient as a redirect exclusion for every target kind. On a valid candidate, disable smart targeting, reveal the redirect ability only if the chosen target changes, and return that holder. Even an already selected valid holder returns immediately and prevents slower holders taking over.

`getMoveTargets` runs redirection after automatic retargeting for a missing/fainted foe, before hit checks, and before retargeting the move log. Spread branches never enter RedirectTarget. Moves with `tracksTarget` bypass it. Existing native `targets()` and `use_move()` have no redirect stage; add one after resolving the effective move/type and initial target, before emitting the final Move target. Weather Ball's effective type must therefore be resolved before redirection.

Ordering is **not** native `speed_sort`: `priorityEvent` requests fast exit, `runEvent` performs a stable `handlers.sort(compareRedirectOrder)`, and returns at the first non-undefined result. Comparator: descending priority, descending cached speed, then ascending ability state effect order when both holders have ability state. No tie shuffle and no new RNG. Keep collection to the active four Pokémon and applicable local effects; no registry scan. Follow Me and Rage Powder have priority 1 and thus precede these priority-0 abilities when later implemented; do not hard-code ability-only redirection as the final architecture.

### Required private ordering state

Current `PokemonState` has no ability state activation order. Generic `EffectState.effect_order` and `Priority.effect_order` exist but do not solve this absence. Add a private ability effect-order value and a battle effect-creation counter (or a documented ordering representation proven equivalent for all supported ordering comparisons). Initial inactive constructor ability states have order zero. On every switch-in, Showdown marks the incoming mon active then recreates ability state, followed by item state. On every setAbility, including permanent Mega changes, End occurs first, then new ability state is created, then Start. Reentry receives a new order even if the ability ID is unchanged. Fainting does not create a replacement ability state. Retain the old private order on inactive/fainted members until the next recreation.

For exact reference effect-order values, `initEffectState` assigns a global counter for nonempty IDs with a target that is active or not a Pokémon. This includes ability/item/volatile/status creations on active Pokémon and side creations. Weather, terrain and room initializers omit a target and therefore retain nonallocating order zero; follow actual initializer arguments. See `NEXT_EFFECT_ORDER_REQUIREMENTS.md` for the audited creation sites and the distinction between unassigned zero and first assigned zero. Do not claim raw counter equality if implementing only relative ability order; relative ability order can determine current redirects but does not certify future mixed-effect ordering. Assigning a new order must not consume RNG. Cached speed changes can override activation-order precedence; Trick Room does not reverse handler speed precedence.

Snapshot/replay must preserve both current orders and counter. Change schema rather than defaulting new fields on old snapshots. Validate counter bounds, legitimate assigned order ranges and uniqueness where semantically applicable, without rejecting inactive historical state. Tests must restore after switch-in, Mega replacement, redirection, and inactive holder reentry. Public own/player views must exclude private effect-order/counter fields. Re-signed corruption tests should cover out-of-range orders/counter and prove private-order changes do not leak into observations.

### Missing move metadata

The current Rust Move lacks `tracksTarget`, `smartTarget`, and pledge-combo state. Export cold numeric/boolean metadata before enabling such moves; preserve action-local mutation rather than changing shared Dex records. Snipe Shot and Stalwart/Propeller Tail are decisive bypass requirements when supported. Currently unsupported moves/abilities remain operational errors, not approximate redirection behavior.

## Flash Fire

TryHit applies to any Fire move with target != source, including Will-O-Wisp. It changes the **shared action-local move accuracy to always-hit**, attempts to add the `flashfire` volatile, and returns null. A repeated activation does not recreate/restart its volatile; it emits immunity. The first activation emits `-start ... ability: Flash Fire` and its volatile source is the attacker, inherited from `battle.event.source` by `addVolatile`. Duration is absent and numeric values empty.

Critical spread consequence: all targets' TryHit events run before all Accuracy checks. A Flash Fire recipient of Heat Wave sets the entire move's accuracy sentinel before remaining targets' accuracy checks, regardless of the recipient's position in spread ordering. Other targets therefore skip accuracy RNG for that action. A protected Flash Fire target never runs its absorption callback. Do not implement this as only a target removal or a per-target accuracy change. Keep immutable Dex data immutable: a local accuracy sentinel is enough for this mutation until generalized active-move state is introduced.

The volatile's ModifyAtk and ModifySpA hooks each have priority 5 and **condition subOrder 2**, rather than ability subOrder 7. When present, they participate even for non-Fire moves and return the no-op modifier in that case. For Fire moves, they multiply the attacking stat by 6144/4096 only if the attacker still has Flash Fire. This is an attacking-stat modifier, not BasePower or final damage. Preserve chain rounding through the existing modifier dispatcher. The condition competes with defensive hooks such as Thick Fat; register the no-op handler and correct condition suborder so the event RNG/order is exact. Weather suppression does not disable Flash Fire.

The ability End calls removeVolatile. That method rejects zero HP. Alive switch-out or Mega replacement removes the volatile and emits the silent condition End; a zero-HP faint End emits no condition End, then native clear_volatile can clear it. General ability End currently returns void and handles weather suppression; adding a fallible Flash Fire End emission requires propagating Result through switch, Mega and faint callers. Do not silently discard event errors. A holder such as Houndoom can activate Flash Fire then Mega into Solar Power; that sequence must remove the old bonus.

Register the existing exported condition ID in Effects. `export_engine_data.mjs` already exports embedded ability conditions. Use normal volatile snapshot storage, but validate Flash Fire's supported shape (no duration, empty values, legitimate source) and lifecycle. Future Baton Pass must honor noCopy. General suppression/acquisition can leave a volatile whose boost becomes inactive until Flash Fire returns; do not equate presence with unconditional power. Current unsupported acquisition/suppression must remain guarded.

## Decisive verification cases

Every complete battle uses TeamValidator-approved full six-member teams and asserts actual activation/redirect/boost/power behavior, not merely presence of a holder. Preserve exact decision states, RNG, traces, snapshots/replay, and batch equivalence.

1. Lightning Rod redirects hostile and allied single-target Electric damage and Thunder Wave; Rhyperior gains SpA despite Ground type immunity. Repeated +6 absorption stays immune and has no accuracy draw.
2. Two Lightning Rod holders at different speeds choose the faster; exact speed ties choose earlier switch activation, including changed order after switch-out/reentry. A selected earlier valid holder still stops dispatch. Change cached speed via supported paralysis/Tailwind to reverse precedence. No redirect tie RNG, including inactive/fainted excluded holders.
3. Spread Discharge damages other recipients and boosts matching holder independently; it is not redirected. Actor's Lightning Rod does not absorb its own move. Non-Electric attacks are unchanged. Protected redirected target follows Protect handling without absorption.
4. Mega Sceptile acquires Lightning Rod and redirects/absorbs in that same turn. Mega Manectric loses Lightning Rod and must stop redirecting. Interactions with Stalwart, Snipe Shot, Follow Me/Rage Powder, pledge-combo and smart targets require separate tests when those effects are ported.
5. Flash Fire absorbs special Fire, physical Fire, and Will-O-Wisp. After activation, compare physical and special Fire power, non-Fire no-op hook behavior, repeat activation, switch removal and reentry.
6. Heat Wave with a Flash Fire target and at least one ordinary target uses no accuracy draw on ordinary targets. Place absorber before and after ordinary recipients and use a seed/accuracy reduction that distinguishes the pre-port miss from the always-hit action. Protected absorber is a negative control retaining accuracy draws.
7. Flash Fire attacking a Thick Fat defender exercises modifier priority/suborder and rounding. A non-Fire special attack while charged is a no-op handler ordering control. Sun/rain and Cloud Nine controls prove weather remains independent.
8. Activated Houndoom Mega evolves into Solar Power, proving alive End removes the volatile before subsequent attacks. Activated holder faints, proving no silent End event at zero HP; snapshot and replay cover both boundaries.
9. Restore with active Flash Fire state, repeated activation, switched-out order state, and simultaneous replacement of redirect holders; 2x1024 environments match serial outputs. Player views expose public activation information without private sorting state.

This note deliberately does not certify dynamic closure, all 82 Mega forms, or full engine readiness. Those remain required beyond this family.
