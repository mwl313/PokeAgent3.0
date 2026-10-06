# Private effect ordering implementation proposal

Read-only audit, 2026-10-07. This note proposes changes; it does not certify implemented redirection or full engine readiness. Pinned reference commit: `14546894d86f9589ac11130c510bbe73b6968665`.

## Exact counter rule and an important correction

`vendor/pokemon-showdown/sim/battle.ts:3320` assigns `effectOrder = battle.effectOrder++` ONLY when the ID is nonempty AND `obj.target` exists AND that target is either an active Pokémon or a non-Pokémon. Explicit supplied order overrides allocation. Otherwise order is zero. Counter starts at zero (`battle.ts:261`), so **zero is also a legitimate first assigned order**, not an unassigned sentinel.

Correction to the broader language in NEXT_ABSORPTION_REQUIREMENTS: weather, terrain and room creation do not automatically allocate an order merely because they are field effects. Current reference weather (`field.ts:70`), terrain (`field.ts:141`) and pseudo-weather (`field.ts:200`) initializers OMIT target and thus keep order zero without incrementing the counter. Side conditions and slot conditions explicitly provide target Side (`side.ts:426`, `side.ts:478`) and allocate. Follow the initializer arguments, not effect category.

Ability-only relative activation counters suffice to order current Lightning Rod/Storm Drain holders but cannot establish raw global counter equality. For full reference equality, account for every allocating creation below. Counter holes caused by removed/replaced states are legitimate; never renumber retained states.

## Proposed private representation

Add battle `next_effect_order: u64` and Pokémon `ability_effect_order: Option<u64>`, `item_effect_order: Option<u64>`. Option distinguishes inactive never-assigned states from the first assigned order zero; numeric reference projection maps None to zero. Extend generic EffectState to distinguish assigned versus nonallocating order, either Option<u64> or a private assignment flag alongside order. Centralize allocation in a checked increment helper; overflow is an explicit engine error, never wrapping or silent fallback. No allocation consumes RNG. Keep these fields out of explicit OwnPokemonView/PlayerView/Knowledge projections and future numeric observation encoding.

A narrower ability-relative representation is acceptable only if explicitly labeled relative ordering, with proofs/tests for current comparisons. Do not compare it numerically to privileged reference global orders or reuse it for mixed handler ordering.

## Current creation and removal sites

| Native site | Reference behavior and required action |
| --- | --- |
| state.rs Pokémon/reset construction | Constructor ability/item states target inactive Pokémon (`pokemon.ts:423`,426): unassigned zero; species state has no target (`332`), empty status no ID (`380`), all nonallocating. Initialize counter zero. |
| battle.rs switch_in, after incoming becomes active | Allocate incoming ability, then nonempty item, in that exact order before SwitchIn callbacks. Reference `battle-actions.ts:136–143`. Reentry reallocates even same ability/item. Empty item remains unassigned zero. Preserve inactive outgoing historical ability order until recreation. |
| battle.rs run_mega | Species/stat/cache updates happen first. End old ability while its old state/order still exists; replace ability state and allocate new order; then Start. Reference Champions `scripts.ts:112`, `pokemon.ts:1923–1943`, `pokemon.ts:1418`. Even unchanged nonempty ability is recreated through setAbility. Item state is not recreated by Mega itself. |
| battle.rs ChoiceScarf choice_lock entry | First successful addVolatile allocates on active actor. Repeated existing lock is not recreated. Reference `items.ts` Choice item ModifyMove uses addVolatile; `pokemon.ts:1982–1999` handles existing/restart before allocation. Avoid allocating eagerly inside or_insert argument construction. |
| battle.rs Protect + stall creation | New Protect allocates first; stall first creation allocates second. Existing stall restarts and retains order, duration refreshes and counter increases (`conditions.ts:437–456`). Current native unconditional stall replacement must change to preserve its order when introducing bookkeeping. Failed protect consumes no new effect order; stall deletion leaves a hole. |
| battle.rs hit_effect status commit | Nonempty accepted status on active recipient allocates after SetStatus acceptance, before status Start/AfterSetStatus (`pokemon.ts:1729–1746`). Static status uses the same site. Failed existing/type/Misty status allocates nothing. |
| battle.rs hit_effect flinch entry | New accepted active volatile allocates; repeated existing flinch does not (`pokemon.ts:1982–1999`). The common hit_effect early HP0 return matches the reference addVolatile rejection; failed/immune attempts allocate nothing. |
| battle/hooks.rs start_side_condition | Tailwind/Reflect/LightScreen allocate on new Side-target state (`side.ts:426`). Same-condition failed restart does not allocate. End/removal consumes no counter. |
| battle/weather.rs start_weather | Nonallocating zero: reference initializer lacks target (`field.ts:70`). Replacement removes old state, new state order remains zero. |
| battle/terrain.rs start_terrain | Nonallocating zero: lacks target (`field.ts:141`). Same terrain fails without allocation. |
| battle/room.rs toggle_trick_room | Nonallocating zero: pseudoWeather initializer lacks target (`field.ts:200`). Toggle-off does not allocate. |
| battle.rs cure_status | Empty status initializer has empty ID, therefore clears to unassigned zero without increment (`pokemon.ts:1732`). Preserve semantics for Natural Cure, Lum and Hydration. |
| battle/hooks.rs item consumption | Berries clear item state with no allocation (`pokemon.ts:1804–1806`); useItem similarly clears. Clear private item assignment when item becomes empty, preserve previous_item provenance separately. |
| battle.rs clear_volatile/faint/switch-out | Clears volatile states; does not recreate ability/item states. Reference clearVolatile retains state/order and only deletes started flags (`pokemon.ts:1556`); faint End precedes clearVolatile, fainted=true and inactive (`battle.ts:2560–2570`). Existing ending flag remains private and lifecycle-valid. |

Future Flash Fire addVolatile joins the common accepted-new-volatile allocator, repeated activation retains order, alive End removes without allocating. Future hazards/slot conditions, item replacement, ability acquisition, Transform/copied volatiles must use their actual reference initializer arguments; copied volatile initialization with active new target allocates (`pokemon.ts:1253`). No generic blanket field increment.

## Snapshot invariants and required tests

Bump schema 3 when adding mandatory private fields. Reject old snapshots rather than guessing activation history. Assigned order must be strictly below next_effect_order; unassigned states must be explicitly represented. Accept first assigned zero. Require uniqueness across currently retained assigned state objects for the exact-global representation, including inactive historical ability/item/status states; do not require contiguous orders or monotonic roster order. Never reject inactive/fainted historical nonzero orders merely because holder is no longer active. Field states known to have nonallocating construction must retain unassigned zero. Active nonempty ability/item states after completed entry need assigned orders; absent consumed items need unassigned item order. Keep existing source/duration/ID validation.

Check creation order in preview completion, simultaneous replacement, reentry, Mega into/out of redirect ability, accepted and failed statuses, repeated Choice lock/flinch, Protect stall restart, item consumption, side conditions, and nonincrementing weather/terrain/room toggles. Compare privileged global counter/orders against reference at every boundary when claiming exact equality. Restore/replay must retain redirect precedence after reentry and Mega; re-signed corruption tests cover out-of-range/duplicate assigned orders and counter below live historical maximum. Private-order changes consistent with validation must leave both player views and numeric observations unchanged.

Redirection comparator uses priority, cached speed, then ability activation order with stable sorting, without tie RNG (`battle.ts:416–423`,794–798). Global order must not be added indiscriminately to all current priority comparators: reference resolvePriority only attaches effectOrder for SwitchIn/RedirectTarget (`battle.ts:997–1003`). Keep ordinary residual ties shuffled exactly as before.
