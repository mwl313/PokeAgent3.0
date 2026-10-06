# Flash Fire native integration audit

2026-10-07. Requirements and read-only integration review against pinned Showdown `14546894d86f9589ac11130c510bbe73b6968665`. This note does not certify the in-progress patches or full engine readiness. See `NEXT_ABSORPTION_REQUIREMENTS.md` for the wider redirection family and retained dynamic closure requirement.

## Sources and current patch boundaries

- `vendor/pokemon-showdown/data/abilities.ts:flashfire`: TryHit, ability End, and embedded condition stat modifiers.
- `sim/battle-actions.ts:trySpreadMoveHit`, `hitStepTryHitEvent`, `hitStepAccuracy`: whole-spread phase ordering.
- `sim/pokemon.ts:addVolatile`, `removeVolatile`, `setAbility`: source inheritance, zero-HP rejection, repeated activation, lifecycle.
- `data/mods/champions/scripts.ts:formeChange`: permanent form/Mega logging precedes setAbility/old ability End. There is no Flash Fire override in Champions abilities.
- `sim/battle.ts:resolvePriority`, `speedSort`, `initEffectState`: condition suborder and global effect creation order.

Live native code now contains FlashFire registration/condition ID, action-local accuracy, a condition stat hook, fallible ability End, and schema-4 effect order state. These are implementation changes, not passing differential evidence. The parent owns their integration and validation.

## Caller and API contract

`BattleState::use_move` must resolve dynamic type (including Weather Ball) before TryHit. Retain immutable shared Dex Move metadata. Initialize `let mut action_accuracy = m.accuracy.map(u16::from)` once before processing all TryHit recipients. Pass `&mut action_accuracy` to `absorb_try_hit(dex, target, actor, m.move_type, ...)`.

`absorb_try_hit` returns `Result<bool>`; true represents null absorption and removes that recipient from later type immunity, accuracy, damage and hit effects. Flash Fire applies only when target != source and effective move type is Fire. Category does not restrict it: Will-O-Wisp is absorbed too. Before any attempt to add the volatile, set shared action accuracy to None (the always-hit sentinel). Then:

1. If target HP is positive and `flashfire` is absent, allocate one effect order, insert a condition EffectState with no duration/values and original attacker source, reveal Flash Fire, emit one EffectStart.
2. If already charged, or addVolatile cannot create state because target HP is zero, reveal the immunity ability and absorb without creating or restarting the volatile. Preserve original source/order on repeated activation.

Zero-HP handling matters even if uncommon at currently exposed decision boundaries: pinned addVolatile rejects it, while the TryHit callback still returns null and changes accuracy.

Do not early-return from the entire move merely because one target absorbs. Existing complete-spread target filtering must continue until every recipient has had its TryHit opportunity, then every surviving recipient uses the final action accuracy sentinel. Heat Wave can therefore become always-hit for all remaining recipients, including recipients occurring before the absorber in spread order. A protected absorber never runs this callback, so it does not change accuracy. Keep reference ordering of TryHit before type/natural powder immunity; retain no additional speed tie RNG in this stable target phase.

## Stat modifier contract and RNG

Presence of the `flashfire` volatile registers a ModifyAtk or ModifySpA condition hook for the selected attacking stat event. The registration is unconditional on move type and current ability while the volatile exists. Its callback modifier is 6144 only for a Fire move while the attacker currently has an unsuppressed Flash Fire ability; otherwise 4096. This conditional no-op still participates in event ordering.

Priority is 5 (native scaled 50000), speed is the holder's cached speed, and **sub_order is 2**, because this is a Condition. The existing ability `add` closure uses sub_order 7; do not reuse it unchanged. The current separate hooks.push with sub_order2 implements the correct representation. Fold into existing damage::chain_modifiers and apply once through stats::modify at the attacking-stat boundary. This is not a BasePower or final-damage modifier. Critical stat-stage selection occurs before ModifyAtk/SpA, as in the existing native damage path.

A subtle regression test is same-speed charged Flash Fire versus Thick Fat on a special attack: both priority5 but suborders2 and7 differ, so they are **not a full tie** and consume no tie-shuffle RNG. Cached speed precedes suborder; faster Thick Fat may run before the condition. Do not force condition-before-ability regardless of speed. Physical Thick Fat is priority6 and precedes priority5. Non-Fire attacks while charged must still retain the no-op condition hook. Future equal-priority/equal-speed/equal-suborder condition handlers can form real ties; the ordinary exact native speed_sort remains necessary.

Future Gastro Acid/Neutralizing Gas/Mold Breaker and acquisition interactions must use the real hasAbility/suppression predicate rather than equating enum ID with an unsuppressed active ability. Those mechanics remain unsupported until ported and tested; the current enum comparison cannot certify them. Weather suppression does not suppress Flash Fire.

## Lifecycle and public event order

`ability_end` is now Result<()> and all three existing callers propagate `?`: voluntary switch-out, permanent Mega ability replacement, and faint processing. Preserve existing weather suppression End behavior and ordering.

For Flash Fire End, only an alive holder with the volatile removes it and emits EffectEnd. Reference removeVolatile returns false at HP0. Therefore native Faint emission, then ability End, then clear_volatile must produce no extra Flash Fire EffectEnd on a zero-HP faint. Alive switch End removes it before clear_volatile/incoming Switch. Future Transform/Skill Swap/Role Play and same-ability replacement must also invoke End rather than leaving a charged bonus behind.

Champions permanent formeChange sets species/stats, logs the Mega change, then setAbility invokes old End, then replaces/starts the new ability. The original native run_mega emitted End before Mega. This was silent for weather End but became observable when Flash Fire added EffectEnd. The live parent patch now emits Mega after stats/form update and **before** old End, then assigns the new ability/order. Preserve that fix. Activated Houndoom -> Mega Houndoom must show Mega before Flash Fire EffectEnd and subsequent Solar Power behavior, with no leftover Flash Fire boost.

## Explicit public source semantics

Native activation must reveal the ability and condition independently, because numeric ability and condition catalogues are distinct. First absorption yields Ability plus EffectStart; repeated immunity reveals Ability without restarting the condition. Condition End removes public effect presence without erasing the publicly identified ability.

Knowledge::apply previously recorded only EffectStart.present. The source is legitimately observed in the triggering move, so the now-documented semantic convention is: **EffectStart.target = Some(public source entity)**. Knowledge::apply sets EffectKnowledge.source from that explicit event target. None adds no source inference; it preserves existing legitimately known source until End, and a fresh post-End effect starts unknown. Other event targets must not fill effect sources. Producers must not populate the source from unobserved hidden provenance.

The knowledge patch and new `tests/effect_sources.rs` cover entity0 known versus unknown, absence of invented duration/stacks, End/restart clearing, ordinary targets, and invalid source atomicity. The creation-site patch now passes Some(source). SemanticEvent audience mapping in emit converts world Entity into the viewer's public index, so source remains stable in each player's preview indexing. This does not expose private source team IDs or sets.

Flash Fire's no-finite-duration state should not invent a numeric zero duration or stack. Known masks remain unavailable for those quantities. Existing observation encoding preserves explicitly known sources in ragged effect entries.

## Snapshot requirements

Use the existing generic volatile state with:

- condition ID `dex.effects.flash_fire`;
- duration None;
- empty numeric values;
- original attacking source Some(valid side, roster);
- one assigned creation order, preserved on repeat activation.

Source may later be on the bench or fainted; do not require it to remain active/alive. At current supported decision boundaries the holder is active and not fainted; reject a charged bench/fainted holder after lifecycle cleanup. A queued zero-HP holder awaiting faint processing may exist in resumable state and should not be confused with a fully fainted holder. Suppressed/acquired ability states can retain an inactive condition in future supported mechanics, so do not permanently restrict condition presence to the FlashFire enum without an explicit scope distinction.

The parent now uses schema4, private ability/item creation orders, effect_order_assigned, and a checked global creation counter. Validate each assigned order against the counter and preserve the distinction between first assigned order0 and unassigned zero. Cross-effect raw order parity requires all qualifying effect creations to share this counter; merely assigning a Flash Fire local ordinal is insufficient. Old schema3 snapshots are rejected explicitly. None of these private counter fields belong in actor observations.

Re-signed corrupt snapshot tests should reject finite duration, numeric payload, missing/invalid source and invalid assigned order. Restore through initial activation, repeat activation, switch removal, faint and Mega removal; continue RNG/trace identically after restore.

## Redirection metadata status

The previous scout identified missing tracksTarget/smartTarget/pledgecombo metadata. The live Move struct and cold loader now include tracks_target, smart_target and pledge_combo, and an in-progress redirect.rs exists. Do not call those gaps resolved until the module is wired and legal reference interactions pass. Flash Fire itself requires only effective move type plus action accuracy; it must not accidentally piggyback a partial redirection implementation.

Smart-target mutation and pledge-combo state belong to the action, never shared Dex mutation. Stalwart/Propeller Tail, Snipe Shot, Follow Me/Rage Powder and ability activation-order redirection require their own full interaction coverage. The new static flags should also eventually be represented in the versioned observation move metadata; their newly available typed data is not currently emitted by observation.rs.

## Required decisive checks

- Legal first/repeat absorption for physical Fire, special Fire and Will-O-Wisp, proving no PP/accuracy/damage draw on the absorbed target and no volatile restart.
- Heat Wave normal target before and after an absorber: both hit without accuracy RNG; protected absorber is the negative control.
- Exact special same-speed Thick Fat suborder case and physical priority6 case, including non-Fire no-op control and HP/stat values that exercise integer rounding.
- Initial/source metadata, repeated attacks from a different source preserve original source, HP0 callback predicate primitive, alive switch End versus faint no-End.
- Houndoom's actual Mega transition: Mega event before End, cleared condition, no residual bonus, Solar Power exact mechanics.
- Snapshot/replay and all current complete reference battles, batch equality and actor encoding; no reduction of legal Pokémon/team scope or readiness claims.
