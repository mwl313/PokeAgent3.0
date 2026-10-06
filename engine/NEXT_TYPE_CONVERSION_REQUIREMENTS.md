# Type conversion requirements

Read-only scout, 2026-10-07. Pinned reference commit `14546894d86f9589ac11130c510bbe73b6968665`, Champions M-C. This is source-derived implementation guidance; passing source inspection is not differential certification or training readiness. The full legal and dynamically called effects objective remains unchanged.

## Pinned definitions and actual legal holders

Primary definitions are `vendor/pokemon-showdown/data/abilities.ts` at Aerilate57, Galvanize1598, Normalize2987, Pixilate3297, Refrigerate3813. `data/mods/champions/abilities.ts` contains no override for these five. The resolved `Dex.forFormat(scope.format)` callbacks all chainModify `[4915,4096]`, an approximately 20% increase. Do not import the 30% multiplier from older generations or use a floating 1.2 multiplier.

| Ability | Legal starting holders | Permitted reachable forms |
| --- | --- | --- |
| Pixilate | Sylveon | Mega Altaria, Mega Gardevoir |
| Aerilate | None | Mega Pinsir, Mega Salamence |
| Refrigerate | Aurorus | Mega Glalie |
| Galvanize | None | None |
| Normalize | None | None |

These lists were obtained from validator-derived `engine/data/scope.json`; forms were inspected through the resolved pinned format Dex. Illegal Alolan Golem/Delcatty starting teams are not legal complete-battle evidence. Dynamic closure analysis must determine whether Galvanize/Normalize can actually be acquired by a legal battle; absence of direct holders alone does not certify closure. Their common primitive may be implemented/tested without falsely claiming legal complete battles. Keep unsupported acquisition paths as operational errors until implemented or genuinely proven unreachable.

Important adjacent full-scope effects: Dragonize on Mega Feraligatr is the same seven-exclusion, priority-1, marker/BP23/4915 mechanism with Dragon destination. Liquid Voice on legal Primarina is priority-1 sound-to-Water conversion without the converter power bonus. Fire Mane on Mega Pyroar is a priority5 attacking-stat multiplier, not type conversion. Eelevate on Mega Eelektross is airborneness/after-faint boosting, not conversion. Do not misclassify these newly legal Champions abilities by their names.

## Exact callback logic

Pixilate/Aerilate/Refrigerate/Galvanize have onModifyTypePriority **-1**, ability suborder7. They convert only when the current effective move type is Normal, and the move is not in the seven-element exclusion list unless activeMove.isMax. A damaging Z move and Tera Blast used while terastallized are excluded too. Status Normal moves do qualify; this is not a damage-only condition.

Destinations are Fairy/Flying/Ice/Electric respectively. Upon qualification they set `move.typeChangerBoosted = this.effect`, an ability effect identity. No activation/reveal log is emitted.

Normalize has onModifyTypePriority **+1**, ability suborder7. It does not require a particular incoming type: all eligible moves become Normal, and even a move already Normal acquires its power-boost marker. It uses the same Max/Z/Tera exceptions and adds Hidden Power and Struggle to the exclusion list.

All five register an onBasePowerPriority **23**, ability suborder7 callback. Its conditional modifier is 4915 if the action marker equals the current handler's ability effect identity, otherwise a no-op 4096 in the native modifier representation. Preserve the handler even when the marker is absent/mismatched. Do not decide the bonus by final move type or by comparing original and final types. Later type changes can leave the original marker intact; Normalize can mark an unchanged Normal type.

Apply the bonus through existing exact BasePower modifier chaining and stats::modify. It is not a stat multiplier, STAB multiplier, or final damage multiplier. Per-target BasePower dispatch retains its normal ordering/RNG semantics.

## Exclusions and move-owned preparation

Compile exclusion IDs/flags cold, never use strings or registry scans in the battle path:

- Common seven: Judgment, Multi-Attack, Natural Gift, Revelation Dance, Techno Blast, Terrain Pulse, Weather Ball.
- Normalize additionally excludes Hidden Power and Struggle.

Do not collapse these to one nine-element list for every ability; that does not mirror source. Hidden Power has its own ModifyType before the ability phase and normally supplies a non-Normal type. Struggle has its own ModifyMove setting type `???` before the ability phase, so the four Normal converters do not convert it even though it is not explicitly in their exclusion list. Current native Struggle is special-cased for immunity/STAB; its conversion bypass must faithfully represent that effective typeless preparation rather than treating canonical Normal as eligible.

Weather Ball is excluded even with no weather, even when weather is suppressed and it remains Normal, and even when some weather would produce a destination matching a converter. Its own ModifyType derives effective weather and its own ModifyMove doubles base power under effective weather. Terrain Pulse similarly resolves its own type/power from grounding and terrain and remains excluded even when ungrounded/no terrain. Neither gets the converter bonus.

Current permitted move inventory includes Electrify, Terrain Pulse, Weather Ball and Struggle. It does not directly permit Ion Deluge, Hidden Power, Tera Blast, Judgment, Multi-Attack, Natural Gift, Revelation Dance or Techno Blast. The latter can still require called-effect closure checks. Max/Z/Tera are not legal mechanics of this format; do not claim their behavior certified merely by hard-coding false. Unsupported variants and unported move-owned callbacks must remain explicit errors.

## Event order and action API

`sim/battle-actions.ts:useMoveInner` establishes the sequence:

1. Active move construction and initial target resolution.
2. Single-event move-owned ModifyType.
3. Single-event move-owned ModifyMove; retarget if it changes target kind.
4. Ordered runEvent ModifyType for actor effects.
5. Ordered runEvent ModifyMove for actor effects; retarget again if needed.
6. Move logging, target list/redirection and hit preparation.
7. Complete-spread TryHit/type/accuracy stages, then per-target BasePower/stat/damage.

The five abilities have no separate ModifyMove callback. They must still run in that proper phase, after Weather Ball/Struggle preparation and before Lightning Rod/Storm Drain redirection and every absorption/immunity check. Normalized Fire must no longer trigger Flash Fire; Galvanized Normal must become eligible for Electric redirection/absorption. Converted spread moves still stay spread and are not redirected. The callback modifies type/marker, not move ID, target kind, category, priority, sound/contact flags or PP/choice locking.

Suggested native API:

- Cold Move exclusion bits: one common-converter bit, one Normalize bit. Numeric destination IDs and helper classification are resolved while loading Dex.
- Action-local `type_changer_boosted: Option<Id>` initialized None every use_move, including repeated/called moves. Store the actual ability ID, not a generic boolean.
- `modify_action_type(dex, actor, base_move, active_move) -> Result<()>` operating only on local active effects. Its ordered callback descriptors use priority, cached holder speed, effect-kind suborder and existing exact speed_sort when needed.
- Extend MoveContext with action-effective type and the boost marker; BasePower23 reads the marker. Canonical metadata remains borrowed immutable.

An interim implementation can keep the current temporary Move clone plus a separate marker, but the clone owns Vec metadata and is an allocation WATCH. A compact `ActiveMove<'a>` borrowing base metadata with overridden primitive type/power/accuracy/target/smart-target fields avoids cloning those vectors. Do not deref-coerce it to &Move in helpers that then read the canonical type: that silently loses conversion. Either helpers accept ActiveMove/context or they receive explicit effective type.

Every current type consumer must use the effective action type: redirect.rs, absorb_try_hit, type-chart immunity/effectiveness, STAB, weather damage, terrain_power_modifier, Blaze/Torrent/Overgrow/Swarm, Thick Fat, Dry Skin, Flash Fire, and freeze-related DamagingHit thawing. Move flags such as defrost/thaws_target retain their own canonical semantics and must not be inferred solely from converted type. The `before_move` actor-status checks occur before these type modifications, as currently arranged.

## Other handlers and no-op RNG

Do not turn the type phase into a registry scan. Collect the actor's applicable ability, volatiles, and relevant field effects only. A conditional no-op remains a registered hook before its predicate is evaluated. Initially one supported actor converter alone cannot form a tie, but future legal field/volatile effects require the real ordering.

Electrify's volatile ModifyType priority is **-2**, condition suborder2. It changes every non-Struggle move to Electric after a converter (-1), retaining the original converter marker and bonus. Ion Deluge's field ModifyType is -2, field suborder5, and changes only moves still Normal at that point; an already converted Fairy/Flying/Ice move is unaffected. Neither resets typeChangerBoosted. Liquid Voice also has -1 ability priority but no bonus marker. Suppression and ability acquisition must match the reference handlers; unsupported Neutralizing Gas/Gastro Acid/Skill Swap/Role Play/Transform etc cannot be silently approximated.

Normal converter flags are not breakable defensive immunity flags. Do not extend target Mold Breaker handling into suppressing an actor's own converter. Global ability suppression still needs the true ignoringAbility semantics when implemented.

## Mega, snapshots and hidden knowledge

Mega abilities are selected before the queued move executes. The first Hyper Voice/Quick Attack after Mega must immediately use the new converter and new species STAB. Do not cache the pre-Mega ability or a bonus into move slots. Conversion remains per action; switch-out and repeated use do not preserve a marker in the Pokémon state. A fully completed action normally needs no new snapshot field; if future mechanics suspend/resume inside an action, its effective type/marker/modified flags must then be persisted in the resumable action state.

The callbacks do not log their ability. Native calculation may use the opponent's true ability internally but actor-facing knowledge cannot automatically reveal it. In particular, Sylveon may have Cute Charm and Aurorus may have Snow Warning: a private ability cannot be embedded merely because conversion was used in damage computation. An explicitly documented inference rule from legitimate public evidence is separate. Public Mega forms may uniquely identify their ability through the existing permanent-form disclosure contract. Keep original canonical move IDs and static Dex move type in persistent repertoire; no hidden action marker is an observation feature.

## Decisive legal fixtures and guards

The independent fixture scout validated full six-member teams and actual turn1 reference interactions for seven cases: Sylveon/Aurorus Hyper Voice90 becomes108 Fairy/Ice; Mega Altaria/Gardevoir Hyper Voice90 becomes108 Fairy; Mega Salamence Hyper Voice90 becomes108 Flying; Mega Pinsir Quick Attack40 becomes48 Flying; Mega Glalie Body Slam85 becomes102 Ice. Glalie cannot learn Hyper Voice, Pinsir cannot learn Double-Edge, and Gardevoir cannot learn Weather Ball in this pinned format. Do not invent those shortcuts.

Required coverage:

- Each legal starting/reachable converter executes both a converting action and a non-Normal/no-marker control. Compare actual effective type, marker identity, exact BasePower/RNG and full battle state, not merely species presence.
- Special spread Hyper Voice and physical priority Quick Attack distinguish type, STAB, spread reduction, Psychic Terrain and exact 4915 rounding. Body Slam requires its 30% paralysis callback and full secondary ordering; do not drop it to enable Mega Glalie.
- Base-form to Mega transition proves converter acquisition in that same turn. A Mega Gardevoir base Synchronize and Mega Pinsir base Hyper Cutter require complete handlers or a different legal, fully supported base ability; do not register them inertly.
- Legal Weather Ball holders among this family are Sylveon, Aurorus, Altaria and Glalie. Test no weather, effective weather and Cloud Nine suppression; power may be50/100 but no converter marker/bonus.
- Converted type chooses type immunity/STAB/terrain/absorption correctly. Electrify requires its full legal volatile/duration/action-order mechanics and preserves a converter marker when overriding final type. Galvanize/Normalize interactions need primitive/called-acquisition evidence without illegal starting teams.
- Move-owned exclusions with no legal holder learnset require appropriate primitive/called-path tests or documented unreachable closure; do not pretend direct complete-battle coverage.
- Hidden knowledge remains unchanged when only an unrevealed ability is privately changed between otherwise indistinguishable safe views. No synthetic Ability event from converter callbacks. Numeric observations still expose only the safe view.
- Snapshot/replay through Mega, converted damage/status/spread moves and subsequent nonconverting moves; serial versus native batch equivalence; full reference regression corpus stays intact.

Final completion still requires all legal Mega/reachable forms, called-effect closure, remaining mechanics and observation/bridge/performance gates. This family is concrete progress toward that scope, not a smaller replacement objective.
