//! Native, typed effect hooks. Even a conditional hook returning no modifier
//! participates in reference event ordering and tie RNG consumption.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum BoostCause {
    Move { secondary: bool },
    Ability(Ability),
    /// Held-item sourced boosts (terrain seeds).
    Item,
}

#[derive(Clone, Copy)]
pub(super) enum ModifierEvent {
    BasePower,
    Attack,
    SpecialAttack,
    Defense,
    SpecialDefense,
    Damage,
}

/// Result of a reference `Pokemon#takeItem` call. `Refused` is the
/// `onTakeItem` false branch, `Empty` the no-item branch; item-swap moves
/// distinguish the two while Knock Off-style removers do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TakeOutcome {
    Taken(Id),
    Refused,
    Empty,
}

/// Shared event-handler list: reference priority, cached speed and modifier.
pub(super) type HookList = SmallVec<[(Priority, u32); 8]>;

#[derive(Clone, Copy)]
pub(super) struct MoveContext<'a> {
    pub actor: Entity,
    pub target: Entity,
    pub move_data: &'a ActiveMove<'a>,
    pub effectiveness: i8,
    pub critical: bool,
}

impl BattleState {
    /// Reference `Battle#suppressingAbility(target)`: the move currently being
    /// resolved ignores the target's ability because its user carries Mold
    /// Breaker (the only in-scope carrier; Teravolt / Turboblaze are outside the
    /// pinned M-C catalogue) or because the move itself declares
    /// `ignoreAbility`. The user can never suppress its own ability, an Ability
    /// Shield protects the holder, and only `flags.breakable` abilities can be
    /// ignored at all.
    pub(super) fn suppressing_ability(
        &self,
        dex: &Dex,
        source: Entity,
        target: Entity,
        m: &ActiveMove<'_>,
    ) -> bool {
        if source == target {
            return false;
        }
        let ability = dex.effects.abilities[self.mon(source).ability as usize];
        let ignores = m.ignore_ability || ability == Ability::Moldbreaker;
        ignores
            && self.mon(target).item != dex.effects.ability_shield
            && dex.effects.breakable_abilities[self.mon(target).ability as usize]
    }

    /// `TryHitSide` handlers for the side a foe-targeted move reaches:
    /// Sap Sipper boosts on a Grass move without blocking it, and Magic Bounce
    /// reflects a reflectable hazard back at the original user and refuses the
    /// original. The reference sorts this event by speed with tie shuffles, so
    /// the handler list (including conditional no-ops) drives RNG parity.
    pub(super) fn ally_try_hit_side(
        &mut self,
        dex: &Dex,
        actor: Entity,
        target: Entity,
        m: &ActiveMove<'_>,
    ) -> Result<bool> {
        // `SapSipper` and `MagicBounce` are mutually exclusive per Pokémon;
        // the kind is tracked so one sorted list reproduces the reference
        // handler order.
        let mut handlers: SmallVec<[(Entity, u8, Priority); 4]> = self
            .active_entities(false)
            .into_iter()
            .filter_map(|e| {
                if e.side != target.side || self.mon(e).hp == 0 {
                    return None;
                }
                let kind = match dex.effects.abilities[self.mon(e).ability as usize] {
                    Ability::SapSipper => 0u8,
                    Ability::Magicbounce if !self.suppressing_ability(dex, actor, e, m) => 1,
                    _ => return None,
                };
                Some((
                    e,
                    kind,
                    Priority {
                        speed: self.mon(e).cached_speed,
                        sub_order: 7,
                        ..Default::default()
                    },
                ))
            })
            .collect();
        speed_sort(&mut handlers, &mut self.rng, |(_, _, priority)| *priority);
        for (holder, kind, _) in handlers {
            if kind == 0 {
                if actor != holder
                    && target.side == actor.side
                    && m.move_type == dex.effects.grass
                    && self.mon(holder).boosts[0] < 6
                {
                    // This side hook has no immunity message when its boost is
                    // capped, so a failed boost must not reveal a hidden
                    // ability.
                    self.reveal_ability(holder)?;
                    self.boost(
                        dex,
                        holder,
                        actor,
                        [1, 0, 0, 0, 0, 0, 0],
                        BoostCause::Ability(Ability::SapSipper),
                    )?;
                }
                continue;
            }
            // `abilities:magicbounce.onAllyTryHitSide`: the first (fastest)
            // holder reflects the hazard with a nested `useMove` back at the
            // original user and refuses the original move (a null return ends
            // the single-target handler loop).
            if m.reflectable
                && !m.has_bounced
                && actor != holder
                && target.side != actor.side
                && let Some(loc) = self.location_of(holder, actor)
            {
                self.reveal_ability(holder)?;
                self.use_move_inner(
                    dex,
                    holder,
                    crate::actions::NO_SLOT,
                    m.id,
                    loc,
                    crate::battle::MoveUse {
                        called: true,
                        bounced: true,
                        priority: m.priority,
                        explicit_target: true,
                        caller_slot: crate::actions::NO_SLOT,
                        source_effect: 0,
                    },
                )?;
                return Ok(true);
            }
        }
        Ok(false)
    }
    /// Non-redirection TryHit absorbers. Return true for reference null, even
    /// when healing fails at full HP or a boost is already capped.
    pub(super) fn absorb_try_hit(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Entity,
        m: &ActiveMove<'_>,
        action_accuracy: &mut Option<u16>,
    ) -> Result<bool> {
        if target == source {
            return Ok(false);
        }
        let ability = if self.suppressing_ability(dex, source, target, m) {
            Ability::Unimplemented
        } else {
            dex.effects.abilities[self.mon(target).ability as usize]
        };
        // `abilities:magicbounce.onTryHit` (priority 1): a reflectable move
        // from another source is used straight back at that source and
        // refused here. `hitStepTryHitEvent` orders handlers left-to-right
        // rather than by speed, so this event consumes no tie RNG.
        if ability == Ability::Magicbounce && m.reflectable && !m.has_bounced {
            return self.bounce_move(dex, target, source, m).map(|()| true);
        }
        // `abilities:goodasgold.onTryHit`: any status move from another source
        // is refused outright. The public immunity message reveals the ability.
        if ability == Ability::Goodasgold && m.category == Category::Status {
            self.reveal_ability(target)?;
            return Ok(true);
        }
        // `abilities:soundproof.onTryHit`: sound-flagged moves are refused.
        if ability == Ability::Soundproof && m.sound {
            self.reveal_ability(target)?;
            return Ok(true);
        }
        // `abilities:bulletproof.onTryHit`: bullet-flagged moves are refused.
        if ability == Ability::Bulletproof && m.bullet {
            self.reveal_ability(target)?;
            return Ok(true);
        }
        // `abilities:telepathy.onTryHit`: an ally's damaging move is refused.
        if ability == Ability::Telepathy
            && target.side == source.side
            && m.category != Category::Status
        {
            self.reveal_ability(target)?;
            return Ok(true);
        }
        // `abilities:sturdy.onTryHit`: one-hit KO moves are refused.
        if ability == Ability::Sturdy && m.ohko.is_some() {
            self.reveal_ability(target)?;
            return Ok(true);
        }
        let move_type = m.move_type;
        if ability == Ability::FlashFire && move_type == dex.effects.fire {
            *action_accuracy = None;
            if self.mon(target).hp > 0
                && !self
                    .mon(target)
                    .volatiles
                    .contains_key(&dex.effects.flash_fire)
            {
                let order = self.allocate_effect_order()?;
                self.mon_mut(target).volatiles.insert(
                    dex.effects.flash_fire,
                    EffectState {
                        id: dex.effects.flash_fire,
                        effect_order: order,
                        effect_order_assigned: true,
                        source: Some((
                            if source.side == 0 {
                                SideId::P1
                            } else {
                                SideId::P2
                            },
                            source.roster,
                        )),
                        ..Default::default()
                    },
                );
                self.reveal_ability(target)?;
                self.emit(
                    EventKind::EffectStart,
                    target,
                    Some(source),
                    EffectRef::Condition(dex.effects.flash_fire),
                    0,
                    false,
                )?;
            } else {
                self.reveal_ability(target)?;
            }
            return Ok(true);
        }
        let matches_type = match ability {
            Ability::DrySkin | Ability::WaterAbsorb | Ability::StormDrain => {
                move_type == dex.effects.water
            }
            Ability::VoltAbsorb | Ability::MotorDrive | Ability::LightningRod => {
                move_type == dex.effects.electric
            }
            Ability::EarthEater => move_type == dex.effects.ground,
            Ability::SapSipper => move_type == dex.effects.grass,
            _ => false,
        };
        if !matches_type {
            return Ok(false);
        }
        if matches!(
            ability,
            Ability::SapSipper | Ability::MotorDrive | Ability::LightningRod | Ability::StormDrain
        ) {
            self.reveal_ability(target)?;
            let mut changes = [0; 7];
            changes[match ability {
                Ability::SapSipper => 0,
                Ability::MotorDrive => 4,
                _ => 2,
            }] = 1;
            self.boost(dex, target, source, changes, BoostCause::Ability(ability))?;
        } else {
            self.absorption_heal(dex, target)?;
        }
        Ok(true)
    }

    /// `abilities:magicbounce`: use the incoming reflectable move back at its
    /// source through the nested `BattleActions#useMove` path. The reflected
    /// action carries `hasBounced` so a second holder cannot reflect it again,
    /// and inherits the outer action's stored effective priority exactly as
    /// `useMoveInner` copies `battle.activeMove.priority`.
    fn bounce_move(
        &mut self,
        dex: &Dex,
        bouncer: Entity,
        source: Entity,
        m: &ActiveMove<'_>,
    ) -> Result<()> {
        let Some(loc) = self.location_of(bouncer, source) else {
            return Ok(());
        };
        self.reveal_ability(bouncer)?;
        self.use_move_inner(
            dex,
            bouncer,
            crate::actions::NO_SLOT,
            m.id,
            loc,
            crate::battle::MoveUse {
                called: true,
                bounced: true,
                priority: m.priority,
                caller_slot: crate::actions::NO_SLOT,
                explicit_target: true,
                source_effect: 0,
            },
        )
    }

    /// Location of `target` from `actor`'s perspective: negative for the
    /// actor's own side, positive for the foe side, 1-based by active slot.
    fn location_of(&self, actor: Entity, target: Entity) -> Option<i8> {
        let slot = self.mon(target).active_slot?;
        Some(if target.side == actor.side {
            -(slot as i8 + 1)
        } else {
            slot as i8 + 1
        })
    }

    /// Common quarter-HP absorption heal. Full HP still blocks
    /// the move and reveals the ability through the reference immunity message.
    pub(super) fn absorption_heal(&mut self, dex: &Dex, target: Entity) -> Result<()> {
        self.reveal_ability(target)?;
        let p = self.mon(target);
        if p.hp > 0
            && !p.fainted
            && p.active_slot.is_some()
            && p.hp < p.stats[0]
            && !self.heal_blocked(dex, target)
        {
            let amount = (p.stats[0] / 4).max(1).min(p.stats[0] - p.hp);
            self.mon_mut(target).hp += amount;
            self.emit(
                EventKind::Heal,
                target,
                None,
                EffectRef::Ability(self.mon(target).ability),
                i32::from(amount),
                true,
            )?;
        }
        Ok(())
    }
    /// ModifyAccuracy precedes accuracy/evasion stages. Numeric accuracy only;
    /// always-hit moves retain their sentinel and never acquire an RNG draw.
    pub(super) fn modify_accuracy(
        &self,
        dex: &Dex,
        actor: Entity,
        target: Entity,
        accuracy: Option<u16>,
        minimize_bypass: bool,
    ) -> Option<u16> {
        let accuracy = accuracy?;
        // `moves:glaiverush.condition.onAccuracy`: while the drawback volatile
        // is up, moves used against the holder never miss.
        if self.mon(target).volatiles.contains_key(&dex.effects.glaive_rush) {
            return None;
        }
        // `moves:minimize.condition.onAccuracy`: a `flags.minimize` move used
        // against a minimized target returns true from the Accuracy event, so
        // the roll is skipped entirely.
        if minimize_bypass && self.mon(target).volatiles.contains_key(&dex.effects.minimize) {
            return None;
        }
        // No Guard (`onAnyAccuracyPriority: 0`): while an unsuppressed holder is
        // active, moves used by or against it never miss. The reference returns
        // `true` from the handler, which bypasses the accuracy roll entirely.
        for e in self.active_entities(false) {
            if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Noguard
                && (e == actor || e == target)
            {
                return None;
            }
        }
        let attacker = dex.effects.abilities[self.mon(actor).ability as usize];
        let defender = dex.effects.abilities[self.mon(target).ability as usize];
        let mut modifier = 4096;
        // Sand Veil / Snow Cloak are defender-owned ModifyAccuracy handlers;
        // Compound Eyes is source-owned. Every `chainModify` contribution
        // accumulates into one modifier that is truncated once, so fold order
        // does not change the result. Conditional no-op handlers participate
        // without consuming a tie draw.
        if matches!(defender, Ability::SandVeil | Ability::SnowCloak) {
            let weather = self.effective_weather(dex);
            let active = (defender == Ability::SandVeil && weather == dex.effects.sand)
                || (defender == Ability::SnowCloak && weather == dex.effects.snow);
            if active {
                modifier = damage::chain_modifiers(modifier, 3277);
            }
        }
        if attacker == Ability::Compoundeyes {
            modifier = damage::chain_modifiers(modifier, 5325);
        }
        // `abilities:tangledfeet.onModifyAccuracy` (priority -1): a confused
        // holder halves the accuracy of moves aimed at it.
        if defender == Ability::Tangledfeet
            && self.mon(target).volatiles.contains_key(&dex.effects.confusion)
        {
            modifier = damage::chain_modifiers(modifier, 2048);
        }
        if modifier == 4096 {
            return Some(accuracy);
        }
        Some(stats::modify(u32::from(accuracy), modifier) as u16)
    }

    pub(super) fn start_side_condition(
        &mut self,
        dex: &Dex,
        actor: Entity,
        id: Id,
    ) -> Result<bool> {
        let side = actor.side as usize;
        if self.sides[side].conditions.contains_key(&id) {
            return Ok(false);
        }
        let duration = if id == dex.effects.tailwind {
            4
        } else if id == dex.effects.reflect
            || id == dex.effects.light_screen
            || id == dex.effects.aurora_veil
        {
            if dex.effects.items[self.mon(actor).item as usize] == Item::LightClay {
                8
            } else {
                5
            }
        } else {
            return Err(EngineError::Unsupported(format!("side condition {id}")));
        };
        let order = self.allocate_effect_order()?;
        self.sides[side].conditions.insert(
            id,
            EffectState {
                id,
                effect_order: order,
                effect_order_assigned: true,
                duration: Some(duration),
                source: Some((
                    if side == 0 { SideId::P1 } else { SideId::P2 },
                    actor.roster,
                )),
                ..Default::default()
            },
        );
        self.emit(
            EventKind::SideEffectStart,
            actor,
            None,
            EffectRef::Condition(id),
            i32::from(duration),
            false,
        )?;
        Ok(true)
    }
    /// `moves:healblock.condition.onTryHeal` returns false for every recovery
    /// whose source is not a Z-Move, so every ported `Battle#heal` call site
    /// checks the holder's volatile before restoring HP.
    pub(super) fn heal_blocked(&self, dex: &Dex, e: Entity) -> bool {
        self.mon(e).volatiles.contains_key(&dex.effects.heal_block)
    }

    /// `Battle#heal`'s ported `TryHeal` pipeline. The pinned scope declares
    /// three handlers: Big Root (`onTryHealPriority: 1`, whose `chainModify`
    /// scales the final amount), Liquid Ooze (priority 0, held by the drained
    /// Pokémon: it damages the healer by the *unmodified* relay amount and
    /// returns 0, which stops the event) and Heal Block (priority 0, held by
    /// the healer: it returns false and stops the event before any later
    /// handler). The two priority-0 handlers order exactly like the reference
    /// `speedSort` — faster cached speed first, then the lower effect order —
    /// so Heal Block suppresses the heal (and the Ooze damage) only when its
    /// handler sorts ahead. Ripen is the one other declared `onTryHeal`
    /// handler and stays an explicit operational error through its unported
    /// ability.
    pub(super) fn drain_heal(
        &mut self,
        dex: &Dex,
        actor: Entity,
        target: Entity,
        amount: u32,
        effect: EffectRef,
    ) -> Result<u32> {
        if amount == 0 {
            return Ok(0);
        }
        // A fainted drained slot keeps the reference behaviour the ported
        // corpus already pins (the Ooze handler does not fire).
        let ooze = (dex.effects.abilities[self.mon(target).ability as usize]
            == Ability::LiquidOoze
            && !self.mon(target).fainted)
            .then_some(target);
        let blocked = self.heal_blocked(dex, actor);
        if blocked || ooze.is_some() {
            // Reference `comparePriority`: higher cached speed first, then the
            // lower `effectOrder` for exact ties.
            let heal_block_first = match ooze {
                None => true,
                Some(ooze_holder) => {
                    let heal_speed = self.mon(actor).cached_speed;
                    let ooze_speed = self.mon(ooze_holder).cached_speed;
                    let heal_order = self
                        .mon(actor)
                        .volatiles
                        .get(&dex.effects.heal_block)
                        .map_or(0, |state| state.effect_order);
                    let ooze_order = self.mon(ooze_holder).ability_effect_order.unwrap_or(0);
                    if !blocked {
                        false
                    } else {
                        heal_speed > ooze_speed
                            || heal_speed == ooze_speed && heal_order < ooze_order
                    }
                }
            };
            if heal_block_first {
                return Ok(0);
            }
        }
        if let Some(ooze_holder) = ooze
            && !self.mon(ooze_holder).fainted
        {
            if self.mon(actor).hp > 0 {
                self.reveal_ability(ooze_holder)?;
                self.indirect_damage(
                    dex,
                    actor,
                    ooze_holder,
                    amount,
                    EffectRef::Ability(self.mon(ooze_holder).ability),
                )?;
            }
            return Ok(0);
        }
        let amount = if dex.effects.items[self.mon(actor).item as usize] == Item::BigRoot {
            stats::modify(amount, 5324)
        } else {
            amount
        };
        let p = self.mon(actor);
        if p.hp == 0 || p.hp == p.stats[0] || amount == 0 {
            return Ok(0);
        }
        let actual = amount.min(u32::from(p.stats[0] - p.hp)) as u16;
        self.mon_mut(actor).hp += actual;
        self.emit(
            EventKind::Heal,
            actor,
            Some(target),
            effect,
            i32::from(actual),
            true,
        )?;
        Ok(u32::from(actual))
    }

    pub(super) fn indirect_damage(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Entity,
        amount: u32,
        effect: EffectRef,
    ) -> Result<()> {
        // `abilities:magicguard.onDamage` refuses every source that is not a
        // move. An ability-sourced refusal names the source's ability.
        if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Magicguard
            && !matches!(effect, EffectRef::Move(_))
        {
            if let EffectRef::Ability(id) = effect
                && id as usize != self.mon(target).ability as usize
            {
                self.reveal_ability(source)?;
            }
            return Ok(());
        }
        if self.mon(target).hp == 0 || amount == 0 {
            return Ok(());
        }
        let actual = amount.min(u32::from(self.mon(target).hp)) as u16;
        self.mon_mut(target).hp -= actual;
        if actual != 0 {
            let hp = self.mon(target).hp;
            self.mon_mut(target).hurt_this_turn = hp;
        }
        if self.mon(target).hp == 0 {
            self.faint_queue.push(crate::state::FaintData {
                target,
                source: Some(source),
                from_move: matches!(effect, EffectRef::Move(_)),
            });
        }
        self.emit(
            EventKind::Damage,
            target,
            Some(source),
            effect,
            -i32::from(actual),
            true,
        )
    }

    pub(super) fn consume_item(&mut self, dex: &Dex, e: Entity) -> Result<Id> {
        let item = self.mon(e).item;
        self.mon_mut(e).item = 0;
        self.mon_mut(e).item_effect_order = None;
        self.mon_mut(e).previous_item = item;
        self.emit(EventKind::EndItem, e, None, EffectRef::Item(item), 0, false)?;
        self.activate_unburden(dex, e)?;
        Ok(item)
    }

    /// `abilities:unburden.onAfterUseItem` / `onTakeItem`: losing the held item
    /// grants the sourced `unburden` volatile. The reference condition has no
    /// Start/End callback, so neither transition emits a public message.
    pub(super) fn activate_unburden(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if dex.effects.abilities[self.mon(e).ability as usize] != Ability::Unburden {
            return Ok(());
        }
        if self.mon(e).volatiles.contains_key(&dex.effects.unburden) {
            return Ok(());
        }
        let order = self.allocate_effect_order()?;
        self.mon_mut(e).volatiles.insert(
            dex.effects.unburden,
            EffectState {
                id: dex.effects.unburden,
                effect_order: order,
                effect_order_assigned: true,
                ..Default::default()
            },
        );
        Ok(())
    }

    /// Reference `Pokemon#takeItem` without any message: runs the holder's
    /// `onTakeItem` refusal (Mega Stones on their own base form) and clears the
    /// slot. The reference logs the removal in the calling move, so this
    /// primitive stays silent; `take_item` adds the public End event for
    /// removal moves such as Knock Off.
    pub(super) fn take_item_checked(
        &mut self,
        dex: &Dex,
        target: Entity,
    ) -> Result<TakeOutcome> {
        let item = self.mon(target).item;
        if item == 0 {
            return Ok(TakeOutcome::Empty);
        }
        if dex.item_take_refused(item, self.mon(target).base_species) {
            return Ok(TakeOutcome::Refused);
        }
        self.mon_mut(target).item = 0;
        self.mon_mut(target).item_effect_order = None;
        self.activate_unburden(dex, target)?;
        Ok(TakeOutcome::Taken(item))
    }

    /// Silent take plus the public item End event, used by removal moves.
    /// The reference never records a taken item in `lastItem`; only
    /// `useItem`/`eatItem` set that provenance.
    pub(super) fn take_item(&mut self, dex: &Dex, target: Entity, source: Entity) -> Result<Id> {
        let TakeOutcome::Taken(item) = self.take_item_checked(dex, target)? else {
            return Ok(0);
        };
        // Reference `Pokemon.takeItem` clears the slot without recording
        // `lastItem`; only `useItem`/`eatItem` set that provenance.
        self.emit(
            EventKind::EndItem,
            target,
            Some(source),
            EffectRef::Item(item),
            0,
            false,
        )?;
        Ok(item)
    }

    /// Reference failed-swap restore: the raw `pokemon.item = id` assignment
    /// used when Trick/Switcheroo cannot complete. It re-registers the item
    /// without a public `-item` message, without re-running the item's Start
    /// event, and leaves `lastItem` untouched.
    pub(super) fn restore_item(&mut self, e: Entity, item: Id) -> Result<()> {
        if item == 0 {
            return Ok(());
        }
        let order = self.allocate_effect_order()?;
        self.mon_mut(e).item = item;
        self.mon_mut(e).item_effect_order = Some(order);
        Ok(())
    }

    pub(super) fn item_heal(
        &mut self,
        dex: &Dex,
        e: Entity,
        amount: u16,
        consume: bool,
    ) -> Result<()> {
        let p = self.mon(e);
        if p.hp == 0 || p.hp == p.stats[0] {
            return Ok(());
        }
        let amount = amount.max(1).min(p.stats[0] - p.hp);
        let item = if consume {
            self.consume_item(dex, e)?
        } else {
            let item = self.mon(e).item;
            self.emit(EventKind::Item, e, None, EffectRef::Item(item), 0, false)?;
            item
        };
        // The reference consumes the item before `this.heal` runs, so a Heal
        // Block holder still eats the berry and simply recovers nothing.
        if self.heal_blocked(dex, e) {
            return Ok(());
        }
        self.mon_mut(e).hp += amount;
        self.emit(
            EventKind::Heal,
            e,
            None,
            EffectRef::Item(item),
            i32::from(amount),
            true,
        )
    }

    pub(super) fn item_update(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        let p = self.mon(e);
        if p.hp == 0 {
            return Ok(());
        }
        match dex.effects.items[p.item as usize] {
            Item::SitrusBerry if u32::from(p.hp) * 2 <= u32::from(p.stats[0]) => {
                if !self.unnerve_blocks_eat(dex, e) {
                    self.item_heal(dex, e, p.stats[0] / 4, true)?
                }
            }
            Item::OranBerry if u32::from(p.hp) * 2 <= u32::from(p.stats[0]) => {
                if !self.unnerve_blocks_eat(dex, e) {
                    self.item_heal(dex, e, 10, true)?
                }
            }
            Item::LumBerry if p.status != 0 => {
                if !self.unnerve_blocks_eat(dex, e) {
                    self.consume_item(dex, e)?;
                    self.cure_status(e)?;
                }
            }
            _ => (),
        }
        super::item_ports::update(self, dex, e)
    }

    /// `abilities:unnerve.onFoeTryEatItem`: any active opposing Unnerve holder
    /// refuses the eater's item consumption. Only berry paths run through the
    /// reference `eatItem`; seeds, herbs and Focus Sash use `useItem` and are
    /// not blocked.
    pub(super) fn unnerve_blocks_eat(&self, dex: &Dex, eater: Entity) -> bool {
        self.active_entities(false).into_iter().any(|other| {
            other.side != eater.side
                && dex.effects.abilities[self.mon(other).ability as usize] == Ability::Unnerve
        })
    }

    /// Reference `getImmunity('powder', pokemon)`: Grass types and Overcoat
    /// holders are powder-immune. Safety Goggles remains an unported item and
    /// therefore an explicit operational error rather than an approximation.
    pub(super) fn powder_immune(&self, dex: &Dex, e: Entity) -> bool {
        self.mon(e).types.contains(&dex.effects.grass)
            || dex.effects.abilities[self.mon(e).ability as usize] == Ability::Overcoat
    }

    /// Ported `onSetStatus` refusals. The caller decides whether the public
    /// immunity message is emitted (the reference only prints it when the
    /// source effect carries a `status` field).
    pub(super) fn status_immune_ability(
        &self,
        dex: &Dex,
        target: Entity,
        status: Id,
    ) -> Option<Ability> {
        let ability = dex.effects.abilities[self.mon(target).ability as usize];
        let refused = match ability {
            Ability::Limber => status == dex.effects.paralysis,
            Ability::Immunity => status == dex.effects.poison || status == dex.effects.toxic,
            Ability::Insomnia => status == dex.effects.sleep,
            Ability::Waterbubble | Ability::Thermalexchange => status == dex.effects.burn,
            Ability::Purifyingsalt => true,
            _ => false,
        };
        refused.then_some(ability)
    }

    /// Ported `onUpdate` cures: a status the holder is immune to is removed at
    /// the next Update boundary even when it arrived from another effect.
    pub(super) fn update_cured_status(&self, dex: &Dex, e: Entity) -> bool {
        let ability = dex.effects.abilities[self.mon(e).ability as usize];
        let status = self.mon(e).status;
        match ability {
            Ability::Limber => status == dex.effects.paralysis,
            Ability::Immunity => status == dex.effects.poison || status == dex.effects.toxic,
            Ability::Insomnia => status == dex.effects.sleep,
            Ability::Magmaarmor => status == dex.effects.freeze,
            Ability::Waterbubble | Ability::Thermalexchange => status == dex.effects.burn,
            _ => false,
        }
    }

    /// `abilities:superluck.onModifyCritRatio` adds one stage before the
    /// gen9 clamp to 4.
    pub(super) fn crit_ratio(&self, dex: &Dex, actor: Entity, base: u8) -> u8 {
        let bonus = u8::from(
            dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Superluck,
        );
        (base + bonus).min(4)
    }

    /// Reference `BattleQueue#willMove(pokemon)`: the Pokémon still has an
    /// unexecuted queued move action. Analytic boosts while every other active
    /// Pokémon has already acted.
    fn moves_last(&self, actor: Entity) -> bool {
        !self.active_entities(false).into_iter().any(|e| {
            e != actor
                && self
                    .queue
                    .iter()
                    .any(|q| q.kind == QueuedKind::Move && q.actor == Some(e))
        })
    }

    pub(super) fn damage_item(&mut self, dex: &Dex, target: Entity, damage: u16) -> Result<u16> {
        let p = self.mon(target);
        if dex.effects.items[p.item as usize] == Item::FocusSash
            && p.hp > 0
            && p.hp == p.stats[0]
            && damage >= p.hp
        {
            let damage = p.hp - 1;
            self.consume_item(dex, target)?;
            return Ok(damage);
        }
        Ok(damage)
    }

    /// `abilities:sturdy.onDamage` (default priority 0, before Focus Sash's
    /// -40): a full-HP holder survives a hit that would otherwise KO it.
    pub(super) fn sturdy_clamp(
        &mut self,
        dex: &Dex,
        target: Entity,
        damage: u16,
        suppressing: bool,
    ) -> Result<u16> {
        let (hp, max_hp) = {
            let p = self.mon(target);
            (p.hp, p.stats[0])
        };
        if !suppressing
            && dex.effects.abilities[self.mon(target).ability as usize] == Ability::Sturdy
            && hp > 0
            && hp == max_hp
            && damage >= hp
        {
            self.reveal_ability(target)?;
            return Ok(hp - 1);
        }
        Ok(damage)
    }

    pub(super) fn item_damage(
        &mut self,
        dex: &Dex,
        target: Entity,
        holder: Entity,
        amount: u16,
    ) -> Result<()> {
        // Item-sourced damage is refused by Magic Guard.
        if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Magicguard {
            return Ok(());
        }
        if self.mon(target).hp == 0 {
            return Ok(());
        }
        let actual = amount.max(1).min(self.mon(target).hp);
        let item = self.mon(holder).item;
        self.emit(
            EventKind::Item,
            holder,
            None,
            EffectRef::Item(item),
            0,
            false,
        )?;
        self.mon_mut(target).hp -= actual;
        let hp = self.mon(target).hp;
        self.mon_mut(target).hurt_this_turn = hp;
        if self.mon(target).hp == 0 {
            self.faint_queue.push(crate::state::FaintData {
                target,
                source: Some(holder),
                from_move: false,
            });
        }
        self.emit(
            EventKind::Damage,
            target,
            Some(holder),
            EffectRef::Item(item),
            -i32::from(actual),
            true,
        )
    }

    pub(super) fn damaging_hit(
        &mut self,
        dex: &Dex,
        actor: Entity,
        targets: &[Entity],
        m: &ActiveMove<'_>,
    ) -> Result<()> {
        if m.category == Category::Status {
            return Ok(());
        }
        let mut handlers = SmallVec::<[(Entity, u8, Priority, usize); 8]>::new();
        for (index, &target) in targets.iter().enumerate() {
            if self.mon(target).status == dex.effects.freeze {
                handlers.push((
                    target,
                    0,
                    Priority {
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Static {
                handlers.push((
                    target,
                    2,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // Contact-triggered abilities. `RoughSkin` carries reference
            // `onDamagingHitOrder: 1`, which sorts before Rocky Helmet's order
            // 2; the others use the default order and keep insertion order
            // within a target. All of them run under `compare_left_to_right`,
            // so only order, priority and target index participate.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::RoughSkin {
                handlers.push((
                    target,
                    5,
                    Priority {
                        order: 1,
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::FlameBody {
                handlers.push((
                    target,
                    3,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // `abilities:poisonpoint.onDamagingHit`: the same default-order
            // contact roll as Static/Flame Body, but the attacker is poisoned.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Poisonpoint {
                handlers.push((
                    target,
                    15,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Stamina {
                handlers.push((
                    target,
                    6,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // Additional default-priority `onDamagingHit` abilities. A Pokémon
            // has a single ability, so these never compete inside one target
            // bucket; the reference comparator still orders them by target
            // index across a spread.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Justified {
                handlers.push((
                    target,
                    7,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Weakarmor {
                handlers.push((
                    target,
                    8,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Gooey {
                handlers.push((
                    target,
                    9,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Effectspore {
                handlers.push((
                    target,
                    10,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Thermalexchange
            {
                handlers.push((
                    target,
                    11,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // Cursed Body uses the default `onDamagingHit` order/sub-order.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Cursedbody {
                handlers.push((
                    target,
                    12,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // Toxic Debris is another default `onDamagingHit` ability.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Toxicdebris {
                handlers.push((
                    target,
                    13,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // Spicy Spray burns the attacker on every damaging hit.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Spicyspray {
                handlers.push((
                    target,
                    14,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // `abilities:seedsower.onDamagingHit`: any damaging hit scatters
            // Grassy Terrain with the holder as its source.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Seedsower {
                handlers.push((
                    target,
                    16,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // `abilities:mummy.onDamagingHit`: a contact hit overwrites the
            // attacker's ability unless it cannot be suppressed.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Mummy {
                handlers.push((
                    target,
                    17,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // `abilities:wanderingspirit.onDamagingHit`: a contact hit swaps
            // both abilities through the shared `Battle#skillSwap` helper.
            if dex.effects.abilities[self.mon(target).ability as usize]
                == Ability::Wanderingspirit
            {
                handlers.push((
                    target,
                    18,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            if dex.effects.items[self.mon(target).item as usize] == Item::RockyHelmet {
                handlers.push((
                    target,
                    1,
                    Priority {
                        order: 2,
                        sub_order: 8,
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
            // `onSourceDamagingHit` handlers are appended after that target's
            // own handlers by the reference and inherit its index, so Poison
            // Touch runs once per damaged target after everything else there.
            if dex.effects.abilities[self.mon(actor).ability as usize] == Ability::PoisonTouch {
                handlers.push((
                    target,
                    4,
                    Priority {
                        sub_order: 7,
                        speed: self.mon(actor).cached_speed,
                        ..Default::default()
                    },
                    index,
                ));
            }
        }
        handlers.sort_by(|a, b| a.2.compare_left_to_right(a.3, &b.2, b.3));
        for (target, kind, _, _) in handlers {
            if kind == 1 {
                if m.contact {
                    self.item_damage(dex, actor, target, self.mon(actor).stats[0] / 6)?;
                }
            } else if kind == 2 {
                // DamagingHit is stable target order, never a speed-tie shuffle.
                // Contact always draws even if status will fail or either HP is
                // zero while the holder is still awaiting faint processing.
                if m.contact && self.rng.chance(3, 10) {
                    let effect = crate::effects::HitEffect {
                        status: dex.effects.paralysis,
                        ..Default::default()
                    };
                    self.hit_effect_with_ability(
                        dex,
                        actor,
                        target,
                        &effect,
                        HitContext {
                            ability_source: Some(target),
                            ..Default::default()
                        },
                    )?;
                }
            } else if kind == 3 {
                // Flame Body: exact 3/10 burn roll on contact, with the ability
                // holder as the status source and the reveal before any heal.
                if m.contact && self.rng.chance(3, 10) {
                    let effect = crate::effects::HitEffect {
                        status: dex.effects.burn,
                        ..Default::default()
                    };
                    self.hit_effect_with_ability(
                        dex,
                        actor,
                        target,
                        &effect,
                        HitContext {
                            ability_source: Some(target),
                            ..Default::default()
                        },
                    )?;
                }
            } else if kind == 4 {
                // Poison Touch is the attacker's ability; the damaged target
                // receives the status with the attacker as its source.
                if m.contact && self.rng.chance(3, 10) {
                    let effect = crate::effects::HitEffect {
                        status: dex.effects.poison,
                        ..Default::default()
                    };
                    self.hit_effect_with_ability(
                        dex,
                        target,
                        actor,
                        &effect,
                        HitContext {
                            ability_source: Some(actor),
                            ..Default::default()
                        },
                    )?;
                }
            } else if kind == 5 {
                // Rough Skin: 1/8 of the attacker's maximum HP, attributed to
                // the holder's ability and applied before faint processing.
                if m.contact {
                    self.reveal_ability(target)?;
                    let amount = u32::from(self.mon(actor).stats[0]) / 8;
                    self.indirect_damage(
                        dex,
                        actor,
                        target,
                        amount,
                        EffectRef::Ability(self.mon(target).ability),
                    )?;
                }
            } else if kind == 6 {
                self.boost(
                    dex,
                    target,
                    actor,
                    [0, 1, 0, 0, 0, 0, 0],
                    BoostCause::Ability(Ability::Stamina),
                )?;
            } else if kind == 7 {
                // Justified: a Dark-type hit raises Attack by one. The boost
                // defaults to the ability holder as target and the attacker as
                // source, exactly like `this.boost({atk: 1})`.
                if m.move_type == dex.effects.dark {
                    self.boost(
                        dex,
                        target,
                        actor,
                        [1, 0, 0, 0, 0, 0, 0],
                        BoostCause::Ability(Ability::Justified),
                    )?;
                }
            } else if kind == 8 {
                // Weak Armor: physical hits drop Defense by one and raise
                // Speed by two, applied as a self-boost (`target, target`).
                if m.category == Category::Physical {
                    self.boost(
                        dex,
                        target,
                        target,
                        [0, -1, 0, 0, 2, 0, 0],
                        BoostCause::Ability(Ability::Weakarmor),
                    )?;
                }
            } else if kind == 9 {
                // Gooey: contact drops the attacker's Speed by one. The ability
                // is revealed before the boost is attempted.
                if m.contact {
                    self.reveal_ability(target)?;
                    self.boost(
                        dex,
                        actor,
                        target,
                        [0, 0, 0, 0, -1, 0, 0],
                        BoostCause::Ability(Ability::Gooey),
                    )?;
                }
            } else if kind == 10 {
                // Effect Spore: an exact 0..99 draw selects the sleep (<11),
                // paralysis (<21) or poison (<30) bracket. Powder-immune
                // attackers neither roll nor receive a status.
                if m.contact && !self.powder_immune(dex, actor) {
                    let roll = self.rng.below(100);
                    let status = if roll < 11 {
                        Some(dex.effects.sleep)
                    } else if roll < 21 {
                        Some(dex.effects.paralysis)
                    } else if roll < 30 {
                        Some(dex.effects.poison)
                    } else {
                        None
                    };
                    if let Some(status) = status {
                        let effect = crate::effects::HitEffect {
                            status,
                            ..Default::default()
                        };
                        self.hit_effect_with_ability(
                            dex,
                            actor,
                            target,
                            &effect,
                            HitContext {
                                ability_source: Some(target),
                                ..Default::default()
                            },
                        )?;
                    }
                }
            } else if kind == 11 {
                // Thermal Exchange: Fire-type hits raise Attack by one.
                if m.move_type == dex.effects.fire {
                    self.boost(
                        dex,
                        target,
                        actor,
                        [1, 0, 0, 0, 0, 0, 0],
                        BoostCause::Ability(Ability::Thermalexchange),
                    )?;
                }
            } else if kind == 12 {
                // Cursed Body: a 3/10 roll disables the attacker's last move.
                // The roll is skipped entirely when the attacker is already
                // disabled or the hit was Struggle, Max or a future move.
                if !self.mon(actor).volatiles.contains_key(&dex.effects.disable)
                    && !m.is_max
                    && !m.future_move
                    && m.id != dex.effects.struggle
                    && self.rng.chance(3, 10)
                {
                    self.reveal_ability(target)?;
                    let suppressing = self.suppressing_ability(dex, actor, target, m);
                    self.start_selection_volatile(
                        dex,
                        actor,
                        Some(target),
                        dex.effects.disable,
                        // The disable lands while the attacker's move is active.
                        true,
                        suppressing,
                    )?;
                }
            } else if kind == 13 {
                // `abilities:toxicdebris.onDamagingHit`: a physical hit
                // scatters Toxic Spikes onto the attacker's side — the
                // attacker's foe side when an ally dealt the friendly fire.
                if m.category == Category::Physical {
                    let side = if actor.side == target.side {
                        1 - actor.side
                    } else {
                        actor.side
                    };
                    let layers = self.sides[side as usize]
                        .conditions
                        .get(&dex.effects.toxic_spikes)
                        .and_then(|state| state.values.first())
                        .copied()
                        .unwrap_or(0);
                    if layers < 2 {
                        self.reveal_ability(target)?;
                        self.add_side_hazard(
                            dex,
                            side as usize,
                            target,
                            dex.effects.toxic_spikes,
                        )?;
                    }
                }
            } else if kind == 14 {
                // `abilities:spicyspray.onDamagingHit`: the attacker is burned
                // outright by any damaging hit, with the holder as its source.
                let effect = crate::effects::HitEffect {
                    status: dex.effects.burn,
                    ..Default::default()
                };
                self.hit_effect_with_ability(
                    dex,
                    actor,
                    target,
                    &effect,
                    HitContext {
                        ability_source: Some(target),
                        ..Default::default()
                    },
                )?;
            } else if kind == 15 {
                // `abilities:poisonpoint.onDamagingHit`: an exact 3/10 poison
                // roll on contact, with the holder as the status source. The
                // roll is consumed even when the attacker cannot be poisoned.
                if m.contact && self.rng.chance(3, 10) {
                    let effect = crate::effects::HitEffect {
                        status: dex.effects.poison,
                        ..Default::default()
                    };
                    self.hit_effect_with_ability(
                        dex,
                        actor,
                        target,
                        &effect,
                        HitContext {
                            ability_source: Some(target),
                            ..Default::default()
                        },
                    )?;
                }
            } else if kind == 16 {
                // `abilities:seedsower.onDamagingHit`: scatter Grassy Terrain
                // with the holder as its source (also revealing the ability).
                self.start_terrain(dex, target, dex.effects.grassy_terrain, true)?;
            } else if kind == 17 {
                // `abilities:mummy.onDamagingHit`: a contact hit replaces the
                // attacker's ability with Mummy unless that ability cannot be
                // suppressed (or is already Mummy).
                let attacker_ability = self.mon(actor).ability;
                if m.contact
                    && !dex.effects.no_suppress_abilities[attacker_ability as usize]
                    && dex.effects.abilities[attacker_ability as usize] != Ability::Mummy
                {
                    let mummy = dex.id("abilities", "mummy")?;
                    self.set_ability(dex, actor, mummy)?;
                }
            } else if kind == 18 {
                // `abilities:wanderingspirit.onDamagingHit`: a contact hit
                // exchanges both abilities through the shared Skill Swap
                // primitive (fail gates and End/Start ordering included).
                if m.contact {
                    self.skill_swap(dex, actor, target)?;
                }
            } else if m.move_type == dex.effects.fire {
                self.cure_status(target)?;
            }
        }
        Ok(())
    }

    pub(super) fn reveal_ability(&mut self, e: Entity) -> Result<()> {
        self.emit(
            EventKind::Ability,
            e,
            None,
            EffectRef::Ability(self.mon(e).ability),
            0,
            false,
        )
    }



    pub(super) fn ability_switch_in(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if self.mon(e).hp > 0
            && matches!(
                dex.effects.abilities[self.mon(e).ability as usize],
                Ability::CloudNine | Ability::AirLock
            )
        {
            self.reveal_ability(e)?;
        }
        // `abilities:zerotohero.onSwitchIn`: the Hero forme announces itself
        // once per transformation, on the first switch-in after the switch-out
        // forme change armed it.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Zerotohero
            && !self.mon(e).hero_message_displayed
            && dex.species[self.mon(e).species as usize].base_species
                == dex.id("species", "Palafin")?
            && self.mon(e).species == dex.id("species", "Palafin-Hero")?
        {
            self.mon_mut(e).hero_message_displayed = true;
            self.reveal_ability(e)?;
        }
        self.ability_start(dex, e)
    }

    pub(super) fn ability_start(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if self.mon(e).hp == 0 {
            return Ok(());
        }
        // `abilities:moldbreaker.onStart`: the ability announces itself when it
        // starts (switch-in, or a copied/altered ability), which is public
        // knowledge for both players.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Moldbreaker {
            self.reveal_ability(e)?;
        }
        // `abilities:pressure.onStart`: Pressure is announced on entry.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Pressure {
            self.reveal_ability(e)?;
        }
        // `abilities:supremeoverlord.onStart`: the boost is frozen at the
        // entry-time fainted count (clamped to five) and announced only when
        // the count is nonzero.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Supremeoverlord {
            let fallen = self.sides[e.side as usize]
                .pokemon
                .iter()
                .filter(|p| p.fainted)
                .count()
                .min(5) as u8;
            self.mon_mut(e).supreme_overlord_fallen = fallen;
            if fallen > 0 {
                self.reveal_ability(e)?;
            }
        }
        // `abilities:curiousmedicine.onStart`: every adjacent ally's stat
        // boosts are cleared. The reference emits one message per ally, so an
        // ally-less holder stays hidden.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Curiousmedicine {
            let allies: SmallVec<[Entity; 1]> = self.sides[e.side as usize]
                .active
                .iter()
                .flatten()
                .map(|&roster| Entity {
                    side: e.side,
                    roster,
                })
                .filter(|ally| *ally != e)
                .collect();
            if !allies.is_empty() {
                self.reveal_ability(e)?;
                for ally in allies {
                    for stat in 0..7usize {
                        let old = self.mon(ally).boosts[stat];
                        if old != 0 {
                            self.mon_mut(ally).boosts[stat] = 0;
                            self.emit(
                                EventKind::Boost,
                                ally,
                                None,
                                EffectRef::Stat(stat as Id),
                                -i32::from(old),
                                false,
                            )?;
                        }
                    }
                }
            }
        }
        // `abilities:frisk.onStart`: every active foe's held item is announced.
        // The ability only becomes public knowledge through the first item
        // message, so an item-less opposing side stays hidden.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Frisk {
            let foes: SmallVec<[Entity; 2]> = self
                .active_entities(false)
                .into_iter()
                .filter(|foe| foe.side != e.side && self.mon(*foe).item != 0)
                .collect();
            for (index, foe) in foes.into_iter().enumerate() {
                if index == 0 {
                    self.reveal_ability(e)?;
                }
                self.emit(
                    EventKind::Item,
                    foe,
                    None,
                    EffectRef::Item(self.mon(foe).item),
                    0,
                    false,
                )?;
            }
        }
        // `abilities:trace.onStart`: arm the one-shot seek and immediately run
        // the same `Update` callback. The pinned regulation has no `noability`
        // or Ability Shield, so only the `notrace` filter can refuse a copy.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Trace {
            self.trace_update(dex, e)?;
        }
        if matches!(
            dex.effects.abilities[self.mon(e).ability as usize],
            Ability::CloudNine | Ability::AirLock
        ) {
            self.mon_mut(e).ability_ending = false;
            self.field_change_order();
        }
        let weather = match dex.effects.abilities[self.mon(e).ability as usize] {
            Ability::Drizzle => dex.effects.rain,
            Ability::Drought => dex.effects.sun,
            Ability::SandStream => dex.effects.sand,
            Ability::SnowWarning => dex.effects.snow,
            _ => 0,
        };
        let terrain = match dex.effects.abilities[self.mon(e).ability as usize] {
            Ability::ElectricSurge => dex.effects.electric_terrain,
            Ability::GrassySurge => dex.effects.grassy_terrain,
            Ability::MistySurge => dex.effects.misty_terrain,
            Ability::PsychicSurge => dex.effects.psychic_terrain,
            _ => 0,
        };
        if terrain != 0 {
            self.start_terrain(dex, e, terrain, true)?;
        }
        if weather != 0 {
            self.start_weather(dex, e, weather, true)?;
        }
        // `abilities:screencleaner.onStart`: remove Reflect, Light Screen and
        // Aurora Veil from the holder's side and then from each opposing side,
        // announcing the ability once before the first removal. The reference
        // iterates the condition ids outermost, so the removals interleave
        // per condition rather than per side.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Screencleaner {
            let mut activated = false;
            for id in [
                dex.effects.reflect,
                dex.effects.light_screen,
                dex.effects.aurora_veil,
            ] {
                for side in [e.side as usize, 1 - e.side as usize] {
                    if self.sides[side].conditions.remove(&id).is_some() {
                        if !activated {
                            activated = true;
                            self.reveal_ability(e)?;
                        }
                        self.emit(
                            EventKind::SideEffectEnd,
                            Entity {
                                side: side as u8,
                                roster: 0,
                            },
                            None,
                            EffectRef::Condition(id),
                            0,
                            false,
                        )?;
                    }
                }
            }
        }
        // `abilities:hospitality.onStart`: heal each adjacent ally by
        // `baseMaxhp / 4`. In doubles the only adjacent ally is the partner;
        // `Battle#heal` skips fainted, inactive and full-HP allies.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Hospitality {
            let partners: SmallVec<[Entity; 1]> = self
                .active_entities(false)
                .into_iter()
                .filter(|p| p.side == e.side && *p != e)
                .collect();
            for ally in partners {
                let p = self.mon(ally);
                // `abilities:hospitality.onStart` heals through `this.heal`, so
                // a Heal Blocked ally recovers nothing (and the ability stays
                // unrevealed, since only the heal message would name it).
                if p.hp == 0 || p.hp >= p.stats[0] || self.heal_blocked(dex, ally) {
                    continue;
                }
                let amount = (p.stats[0] / 4).max(1).min(p.stats[0] - p.hp);
                self.reveal_ability(e)?;
                self.mon_mut(ally).hp += amount;
                self.emit(
                    EventKind::Heal,
                    ally,
                    Some(e),
                    EffectRef::Ability(self.mon(e).ability),
                    i32::from(amount),
                    true,
                )?;
            }
        }
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Intimidate {
            let foes: SmallVec<[Entity; 4]> = self
                .active_entities(false)
                .into_iter()
                .filter(|p| p.side != e.side && self.mon(*p).hp > 0)
                .collect();
            if !foes.is_empty() {
                self.reveal_ability(e)?;
            }
            for target in foes {
                self.boost(
                    dex,
                    target,
                    e,
                    [-1, 0, 0, 0, 0, 0, 0],
                    BoostCause::Ability(Ability::Intimidate),
                )?;
            }
        }
        Ok(())
    }

    /// Reference `abilities:trace.onUpdate`: sample one adjacent live foe whose
    /// ability is not flagged `notrace` and copy it through `setAbility`.
    fn trace_update(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        let foes: SmallVec<[Entity; 2]> = self
            .active_entities(false)
            .into_iter()
            .filter(|p| {
                p.side != e.side
                    && self.mon(*p).hp > 0
                    && !dex.effects.no_trace_abilities[self.mon(*p).ability as usize]
            })
            .collect();
        if foes.is_empty() {
            return Ok(());
        }
        // In doubles every foe is adjacent; `battle.sample` always draws.
        let index = self.rng.below(foes.len() as u32) as usize;
        let target = foes[index];
        let ability = self.mon(target).ability;
        self.set_ability(dex, e, ability)?;
        Ok(())
    }

    /// `abilities:zerotohero.onSwitchOut`: the reference permanent
    /// `formeChange` updates the stored species, recalculates the stored
    /// stats (the two formes share the same HP base, so `updateMaxHp` keeps
    /// the current HP) and resets the ability state through
    /// `setAbility(same, isFromFormeChange = true)`.
    fn zero_to_hero_forme(&mut self, dex: &Dex, e: Entity, form: Id) -> Result<()> {
        let species = &dex.species[form as usize];
        let mon = self.mon_mut(e);
        let new_stats = crate::stats::champions_stats(
            species.base_stats,
            mon.points,
            dex.natures[mon.nature as usize],
            species.max_hp,
        );
        let damage_taken = mon.stats[0].saturating_sub(mon.hp);
        if mon.hp > 0 {
            mon.hp = new_stats[0].saturating_sub(damage_taken).max(1);
        }
        mon.stats = new_stats;
        mon.cached_speed = i32::from(new_stats[5]);
        mon.species = form;
        mon.base_species = form;
        mon.types = species.types.clone();
        // `heroMessageDisplayed = false` arms the next switch-in message.
        mon.hero_message_displayed = false;
        self.emit(EventKind::Forme, e, None, EffectRef::Species(form), 0, true)?;
        let order = self.allocate_effect_order()?;
        self.mon_mut(e).ability_effect_order = Some(order);
        Ok(())
    }

    /// Reference `Pokemon#setAbility`: end the outgoing ability, reset the
    /// holder's ability state (a fresh effect order), reveal the incoming
    /// ability and run its `Start` callbacks. The reference's `[of]` source
    /// attribution is a message detail the native event model does not carry.
    pub(super) fn set_ability(&mut self, dex: &Dex, e: Entity, ability: Id) -> Result<bool> {
        if self.mon(e).ability == ability {
            return Ok(false);
        }
        self.ability_end(dex, e)?;
        let order = self.allocate_effect_order()?;
        let mon = self.mon_mut(e);
        mon.ability = ability;
        mon.ability_ending = false;
        mon.ability_effect_order = Some(order);
        self.reveal_ability(e)?;
        self.ability_start(dex, e)?;
        Ok(true)
    }

    pub(super) fn ability_switch_out(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        match dex.effects.abilities[self.mon(e).ability as usize] {
            // `abilities:zerotohero.onSwitchOut`: a Palafin leaving the field
            // becomes Palafin-Hero permanently (the switch-in message is armed
            // for the next entry).
            Ability::Zerotohero => {
                let palafin = dex.id("species", "Palafin")?;
                if dex.species[self.mon(e).base_species as usize].base_species == palafin {
                    let hero = dex.id("species", "Palafin-Hero")?;
                    if self.mon(e).species != hero {
                        self.zero_to_hero_forme(dex, e, hero)?;
                    }
                }
            }
            Ability::Regenerator => {
                let p = self.mon(e);
                let amount = (p.stats[0] / 3).min(p.stats[0] - p.hp);
                // `abilities:regenerator.onSwitchOut` calls the raw
                // `pokemon.heal`, which bypasses the TryHeal event: Heal Block
                // does not stop this recovery.
                if amount > 0 {
                    self.mon_mut(e).hp += amount;
                    self.reveal_ability(e)?;
                    self.emit(
                        EventKind::Heal,
                        e,
                        None,
                        EffectRef::Ability(self.mon(e).ability),
                        i32::from(amount),
                        true,
                    )?;
                }
            }
            Ability::NaturalCure if self.mon(e).status != 0 => {
                self.reveal_ability(e)?;
                self.cure_status(e)?;
            }
            _ => (),
        }
        Ok(())
    }

    pub(super) fn boost(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Entity,
        mut changes: [i8; 7],
        cause: BoostCause,
    ) -> Result<bool> {
        if self.mon(target).hp == 0
            || self.mon(target).active_slot.is_none()
            || !self.sides[(1 - target.side) as usize]
                .pokemon
                .iter()
                .any(|p| p.selected && !p.fainted)
        {
            return Ok(false);
        }
        let ability = dex.effects.abilities[self.mon(target).ability as usize];
        // `abilities:contrary.onChangeBoost` runs before the boost table is
        // capped; every incoming entry is inverted.
        if ability == Ability::Contrary {
            for change in changes.iter_mut() {
                *change = change.saturating_neg();
            }
        }
        // The reference caps the entire incoming boost table before TryBoost.
        for (i, change) in changes.iter_mut().enumerate() {
            let old = self.mon(target).boosts[i];
            *change = (old + *change).clamp(-6, 6) - old;
        }
        let intimidate = matches!(cause, BoostCause::Ability(Ability::Intimidate));
        let mut blocked = false;
        for (stat, change) in changes.iter_mut().enumerate() {
            if (source != target
                && *change < 0
                && (ability == Ability::ClearBody
                    || ability == Ability::HyperCutter && stat == 0
                    // `abilities:bigpecks.onTryBoost`: only Defense drops are
                    // refused; a secondary-sourced drop is refused silently.
                    || ability == Ability::Bigpecks && stat == 1))
                || (intimidate
                    && stat == 0
                    && *change != 0
                    && matches!(
                        ability,
                        Ability::InnerFocus
                            | Ability::OwnTempo
                            | Ability::Oblivious
                            | Ability::Scrappy
                    ))
            {
                *change = 0;
                blocked = true;
            }
        }
        // `abilities:mirrorarmor.onTryBoost`: negative boosts from another
        // Pokémon are removed from the incoming table and applied back to the
        // source. The reference skips its own reflected boosts, and stats
        // already at -6 have a zero clamped delta so they never arrive here.
        if ability == Ability::Mirrorarmor
            && source != target
            && !matches!(cause, BoostCause::Ability(Ability::Mirrorarmor))
        {
            let mut mirrored = false;
            for (stat, change) in changes.iter_mut().enumerate() {
                if *change < 0 {
                    let delta = *change;
                    *change = 0;
                    mirrored = true;
                    if self.mon(source).hp > 0 {
                        self.reveal_ability(target)?;
                        let mut reflected = [0i8; 7];
                        reflected[stat] = delta;
                        self.boost(
                            dex,
                            source,
                            target,
                            reflected,
                            BoostCause::Ability(Ability::Mirrorarmor),
                        )?;
                    }
                }
            }
            let _ = mirrored;
        }
        // `abilities:flowerveil.onAllyTryBoost`: an adjacent Flower Veil holder
        // refuses every negative boost aimed at a Grass-type ally (or itself)
        // from another Pokémon.
        if source != target && self.mon(target).types.contains(&dex.effects.grass) {
            let holders: SmallVec<[Entity; 2]> = self
                .active_entities(false)
                .into_iter()
                .filter(|h| {
                    h.side == target.side
                        && dex.effects.abilities[self.mon(*h).ability as usize]
                            == Ability::Flowerveil
                })
                .collect();
            if !holders.is_empty() {
                let mut removed = false;
                for change in changes.iter_mut() {
                    if *change < 0 {
                        *change = 0;
                        removed = true;
                    }
                }
                if removed && !matches!(cause, BoostCause::Move { secondary: true }) {
                    for holder in holders {
                        self.reveal_ability(holder)?;
                    }
                }
            }
        }
        if blocked
            && !matches!(
                cause,
                BoostCause::Move { secondary: true } | BoostCause::Item
            )
        {
            self.reveal_ability(target)?;
        }
        let mut changed = false;
        let mut raised = false;
        let mut lowered = false;
        for (stat, change) in changes.into_iter().enumerate() {
            let old = self.mon(target).boosts[stat];
            let new = (old + change).clamp(-6, 6);
            if new == old {
                continue;
            }
            if new > old {
                raised = true;
            } else {
                lowered = true;
            }
            self.mon_mut(target).boosts[stat] = new;
            self.emit(
                EventKind::Boost,
                target,
                Some(source),
                EffectRef::Stat(stat as Id),
                i32::from(new - old),
                false,
            )?;
            changed = true;
            // AfterEachBoost triggers once per lowered stat, including multiple
            // stats in one move. Ally/self reductions do not trigger these abilities.
            if new < old
                && source.side != target.side
                && matches!(ability, Ability::Defiant | Ability::Competitive)
            {
                self.reveal_ability(target)?;
                let mut response = [0; 7];
                response[if ability == Ability::Defiant { 0 } else { 2 }] = 2;
                self.boost(dex, target, target, response, BoostCause::Ability(ability))?;
            }
        }
        // Reference `boost`: the turn flags read the *applied* deltas, so a
        // fully capped boost leaves them untouched.
        if raised {
            self.mon_mut(target).stats_raised_this_turn = true;
        }
        if lowered {
            self.mon_mut(target).stats_lowered_this_turn = true;
        }
        Ok(changed)
    }

    fn modifiers(
        &mut self,
        dex: &Dex,
        event: ModifierEvent,
        context: MoveContext<'_>,
        value: u32,
    ) -> Result<u32> {
        Ok(self.modifiers_with_participation(dex, event, context, value)?.0)
    }

    /// Returns the folded modifier together with whether any handler
    /// participated. The reference truncates a fractional relay value
    /// (`modify(value, 4096)`) exactly when at least one handler ran, which
    /// matters for `move.basePower * hp / maxhp` style callbacks.
    pub(super) fn modifiers_with_participation(
        &mut self,
        dex: &Dex,
        event: ModifierEvent,
        context: MoveContext<'_>,
        value: u32,
    ) -> Result<(u32, bool)> {
        let MoveContext {
            actor,
            target,
            move_data: m,
            effectiveness,
            critical,
        } = context;
        let a = self.mon(actor);
        let d = self.mon(target);
        let attacking = dex.effects.abilities[a.ability as usize];
        // `Battle#suppressingAbility`: a Mold Breaker move ignores the
        // defender's ability for every modifier event it triggers.
        let defending = if self.suppressing_ability(dex, actor, target, m) {
            Ability::Unimplemented
        } else {
            dex.effects.abilities[d.ability as usize]
        };
        let mut hooks = HookList::new();
        let mut add = |e: Entity, priority: i32, modifier: u32| {
            hooks.push((
                Priority {
                    priority: priority * 10000,
                    speed: self.mon(e).cached_speed,
                    sub_order: 7,
                    ..Default::default()
                },
                modifier,
            ));
        };
        match event {
            ModifierEvent::BasePower => {
                match attacking {
                Ability::SandForce => add(
                    actor,
                    21,
                    if self.effective_weather(dex) == dex.effects.sand
                        && [dex.effects.rock, dex.effects.ground, dex.effects.steel]
                            .contains(&m.move_type)
                    {
                        5325
                    } else {
                        4096
                    },
                ),
                Ability::Pixilate
                | Ability::Aerilate
                | Ability::Refrigerate
                | Ability::Galvanize
                | Ability::Normalize
                | Ability::Dragonize => add(
                    actor,
                    23,
                    if m.type_changer_boosted == Some(attacking) {
                        4915
                    } else {
                        4096
                    },
                ),
                Ability::Technician => add(actor, 30, if value <= 60 { 6144 } else { 4096 }),
                Ability::ToughClaws => add(actor, 21, if m.contact { 5325 } else { 4096 }),
                Ability::IronFist => add(actor, 23, if m.punch { 4915 } else { 4096 }),
                Ability::MegaLauncher => add(actor, 19, if m.pulse { 6144 } else { 4096 }),
                Ability::Sharpness => add(actor, 19, if m.slicing { 6144 } else { 4096 }),
                Ability::StrongJaw => add(actor, 19, if m.bite { 6144 } else { 4096 }),
                // `abilities:analytic.onBasePower` (priority 21): 1.3x while no
                // other active Pokémon still has an unexecuted move action.
                Ability::Analytic => add(
                    actor,
                    21,
                    if self.moves_last(actor) { 5325 } else { 4096 },
                ),
                // `abilities:supremeoverlord.onBasePower` (priority 21): the
                // entry-time fainted count raises power by 10% per member.
                Ability::Supremeoverlord => add(
                    actor,
                    21,
                    match self.mon(actor).supreme_overlord_fallen {
                        1 => 4506,
                        2 => 4915,
                        3 => 5325,
                        4 => 5734,
                        5 => 6144,
                        _ => 4096,
                    },
                ),
                Ability::Reckless => add(actor, 23, if m.recoil.is_some() { 4915 } else { 4096 }),
                Ability::Sheerforce => add(
                    actor,
                    21,
                    if m.sheer_force || m.sheer_force_boosted {
                        5325
                    } else {
                        4096
                    },
                ),
                // `abilities:rivalry.onBasePower` (priority 24): same gender
                // multiplies by 1.25, opposite by 0.75; a genderless partner
                // on either side leaves the power unchanged. The engine stores
                // the reference `''` (genderless) as 0, `M` as 1 and `F` as 2.
                Ability::Rivalry => {
                    let attacker = self.mon(actor).gender;
                    let defender = self.mon(target).gender;
                    add(
                        actor,
                        24,
                        if attacker != 0 && defender != 0 {
                            if attacker == defender { 5120 } else { 3072 }
                        } else {
                            4096
                        },
                    )
                }
                Ability::Punkrock => add(actor, 7, if m.sound { 5325 } else { 4096 }),
                _ => (),
                }
                // `moves:knockoff.onBasePower`: the 1.5x boost only applies
                // when the target's item passes the TakeItem check, so a Mega
                // Stone on its own base form neither boosts nor is removed.
                if m.hooks & crate::effects::hook::KNOCK_OFF != 0
                    && d.item != 0
                    && !dex.item_take_refused(d.item, d.base_species)
                {
                    add(actor, 0, 6144);
                }
                // `moves:lashout.onBasePower`: doubles while the user's stats
                // were lowered this turn.
                if m.hooks & crate::effects::hook::LASH_OUT != 0 && a.stats_lowered_this_turn {
                    add(actor, 0, 8192);
                }
                // `moves:barbbarrage.onBasePower`: doubles against a poisoned
                // target (regular or badly poisoned).
                if m.hooks & crate::effects::hook::BARB_BARRAGE != 0
                    && (d.status == dex.effects.poison || d.status == dex.effects.toxic)
                {
                    add(actor, 0, 8192);
                }
            }
            ModifierEvent::Attack | ModifierEvent::SpecialAttack => {
                if matches!(
                    attacking,
                    Ability::Blaze | Ability::Torrent | Ability::Overgrow | Ability::Swarm
                ) {
                    let same_type = match attacking {
                        Ability::Blaze => m.move_type == dex.effects.fire,
                        Ability::Torrent => m.move_type == dex.effects.water,
                        Ability::Overgrow => m.move_type == dex.effects.grass,
                        Ability::Swarm => m.move_type == dex.effects.bug,
                        _ => unreachable!(),
                    };
                    add(
                        actor,
                        5,
                        if u32::from(a.hp) * 3 <= u32::from(a.stats[0]) && same_type {
                            6144
                        } else {
                            4096
                        },
                    );
                }
                if matches!(event, ModifierEvent::Attack) && attacking == Ability::HugePower {
                    add(actor, 5, 8192);
                }
                // `abilities:firemane.onModifyAtk|onModifySpA` (priority 5):
                // Fire-type moves are boosted 1.5x unconditionally.
                if attacking == Ability::Firemane {
                    add(
                        actor,
                        5,
                        if m.move_type == dex.effects.fire {
                            6144
                        } else {
                            4096
                        },
                    );
                }
                if matches!(event, ModifierEvent::SpecialAttack) && attacking == Ability::SolarPower
                {
                    // The handler exists outside sun too. Keep the no-op entry
                    // for exact priority ties/RNG against defensive handlers.
                    add(
                        actor,
                        5,
                        if self.effective_weather(dex) == dex.effects.sun {
                            6144
                        } else {
                            4096
                        },
                    );
                }
                // `abilities:guts.onModifyAtk` (priority 5) boosts the holder's
                // Attack while it has any major status.
                if matches!(event, ModifierEvent::Attack) && attacking == Ability::Guts {
                    add(
                        actor,
                        5,
                        if a.status != 0 { 6144 } else { 4096 },
                    );
                }
                // `abilities:plus|minus.onModifySpA` (priority 5): 1.5x while
                // another active ally on the field holds Plus or Minus. The
                // reference's `allies()` excludes the holder itself.
                if matches!(event, ModifierEvent::SpecialAttack)
                    && matches!(attacking, Ability::Plus | Ability::Minus)
                {
                    let ally = self.active_entities(true).into_iter().any(|e| {
                        e != actor
                            && e.side == actor.side
                            && matches!(
                                dex.effects.abilities[self.mon(e).ability as usize],
                                Ability::Plus | Ability::Minus
                            )
                    });
                    add(actor, 5, if ally { 6144 } else { 4096 });
                }
                // `abilities:waterbubble.onModifyAtk/SpA` doubles the holder's
                // Water attacks (default priority 0).
                if attacking == Ability::Waterbubble {
                    add(
                        actor,
                        0,
                        if m.move_type == dex.effects.water {
                            8192
                        } else {
                            4096
                        },
                    );
                }
                // Defender-owned `onSourceModifyAtk/SpA` weakeners. Heatproof
                // and Purifying Salt carry priorities 6/5; Water Bubble uses
                // 5/5.
                match defending {
                    Ability::Heatproof => add(
                        target,
                        if matches!(event, ModifierEvent::Attack) {
                            6
                        } else {
                            5
                        },
                        if m.move_type == dex.effects.fire {
                            2048
                        } else {
                            4096
                        },
                    ),
                    Ability::Purifyingsalt => add(
                        target,
                        if matches!(event, ModifierEvent::Attack) {
                            6
                        } else {
                            5
                        },
                        if m.move_type == dex.effects.ghost {
                            2048
                        } else {
                            4096
                        },
                    ),
                    Ability::Waterbubble => add(
                        target,
                        5,
                        if m.move_type == dex.effects.fire {
                            2048
                        } else {
                            4096
                        },
                    ),
                    _ => (),
                }
                if defending == Ability::ThickFat {
                    add(
                        target,
                        if matches!(event, ModifierEvent::Attack) {
                            6
                        } else {
                            5
                        },
                        if m.move_type == dex.effects.ice || m.move_type == dex.effects.fire {
                            2048
                        } else {
                            4096
                        },
                    );
                }
            }
            ModifierEvent::Damage => {
                match defending {
                    Ability::Filter => {
                        add(target, 0, if effectiveness > 0 { 3072 } else { 4096 })
                    }
                    Ability::Multiscale => {
                        add(target, 0, if d.hp == d.stats[0] { 2048 } else { 4096 })
                    }
                    // `abilities:fluffy.onSourceModifyDamage`: Fire doubles,
                    // contact halves, applied in that order on one ratio.
                    Ability::Fluffy => {
                        let modifier = match (m.move_type == dex.effects.fire, m.contact) {
                            (true, true) => 4096,
                            (true, false) => 8192,
                            (false, true) => 2048,
                            (false, false) => 4096,
                        };
                        add(target, 0, modifier)
                    }
                    // `abilities:punkrock.onSourceModifyDamage`.
                    Ability::Punkrock => add(target, 0, if m.sound { 2048 } else { 4096 }),
                    // `abilities:auraguard.onSourceModifyDamage`: contact moves
                    // deal half damage to the holder.
                    Ability::Auraguard => add(target, 0, if m.contact { 2048 } else { 4096 }),
                    _ => (),
                }
                // `moves:glaiverush.condition.onSourceModifyDamage`: the holder
                // of the drawback volatile takes doubled damage.
                if self
                    .mon(target)
                    .volatiles
                    .contains_key(&dex.effects.glaive_rush)
                {
                    add(target, 0, 8192);
                }
                // `moves:minimize.condition.onSourceModifyDamage`: a move
                // carrying `flags.minimize` deals doubled damage to the
                // minimized holder.
                if m.minimize
                    && self
                        .mon(target)
                        .volatiles
                        .contains_key(&dex.effects.minimize)
                {
                    add(target, 0, 8192);
                }
                // `abilities:sniper.onModifyDamage` is attacker-owned: the
                // holder's own critical hits deal 1.5x.
                if attacking == Ability::Sniper {
                    add(actor, 0, if critical { 6144 } else { 4096 });
                }
                // `items:metronome.condition.onModifyDamage`: the attacker's
                // consecutive-use counter scales every damaging move it uses
                // (4096, 4915, 5734, 6553, 7372, 8192 for stacks 0..5+, and
                // the handler also participates at zero).
                if let Some(state) = self.mon(actor).volatiles.get(&dex.effects.metronome) {
                    const METRONOME_MODS: [u32; 6] = [4096, 4915, 5734, 6553, 7372, 8192];
                    let index = state.values[0].clamp(0, 5) as usize;
                    add(actor, 0, METRONOME_MODS[index]);
                }
            }
            // `abilities:furcoat|marvelscale|grasspelt.onModifyDef` all carry
            // priority 6 and chain with the holder's item modifiers.
            ModifierEvent::Defense => {
                if defending == Ability::Furcoat {
                    add(target, 6, 8192);
                }
                if defending == Ability::Marvelscale {
                    add(target, 6, if d.status != 0 { 6144 } else { 4096 });
                }
                if defending == Ability::Grasspelt {
                    add(
                        target,
                        6,
                        if self.terrain_id(dex) == dex.effects.grassy_terrain {
                            6144
                        } else {
                            4096
                        },
                    );
                }
            }
            ModifierEvent::SpecialDefense => {}
        }
        if matches!(event, ModifierEvent::BasePower) && defending == Ability::DrySkin {
            // SourceBasePower is a defender-owned hook. Retain the no-op
            // outside Fire for exact handler priority/speed ordering.
            add(
                target,
                17,
                if m.move_type == dex.effects.fire {
                    5120
                } else {
                    4096
                },
            );
        }
        // `helpinghand` condition (priority 10, condition sub-order 2): the
        // stored multiplier chains into the holder's BasePower.
        if matches!(event, ModifierEvent::BasePower)
            && let Some(state) = a.volatiles.get(&dex.effects.helping_hand)
        {
            hooks.push((
                Priority {
                    priority: 10 * 10000,
                    speed: self.mon(actor).cached_speed,
                    sub_order: 2,
                    ..Default::default()
                },
                state.values.first().copied().unwrap_or(6144) as u32,
            ));
        }
        let terrain = self.terrain_id(dex);
        if matches!(event, ModifierEvent::BasePower) && terrain != 0 {
            hooks.push((
                Priority {
                    priority: 60000,
                    sub_order: 5,
                    ..Default::default()
                },
                self.terrain_power_modifier(dex, context),
            ));
        }
        // Global aura abilities (`onAnyBasePowerPriority: 20`). Exactly one
        // holder applies the boost — the reference marks the first handler to
        // run as `move.auraBooster` — while every other holder keeps a no-op
        // entry so the handler set (and therefore tie ordering) matches.
        // `abilities:steelyspirit.onAllyBasePower` (priority 22): every holder
        // on the attacker's side (its own included) adds 1.5x to a Steel move;
        // non-Steel moves keep the no-op entries for exact ordering.
        if matches!(event, ModifierEvent::BasePower) {
            for holder in self.active_entities(false) {
                if holder.side == actor.side
                    && dex.effects.abilities[self.mon(holder).ability as usize]
                        == Ability::Steelyspirit
                {
                    hooks.push((
                        Priority {
                            priority: 22 * 10000,
                            speed: self.mon(holder).cached_speed,
                            sub_order: 7,
                            ..Default::default()
                        },
                        if m.move_type == dex.effects.steel {
                            6144
                        } else {
                            4096
                        },
                    ));
                }
            }
        }
        if matches!(event, ModifierEvent::BasePower) {
            let fairy = dex.effects.fairy;
            let mut aura: SmallVec<[(Entity, i32); 4]> = SmallVec::new();
            for e in self.active_entities(false) {
                // The pinned `fairyaura` handlers do not consult
                // `suppressingAbility`; only the onStart message does.
                if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Fairyaura {
                    aura.push((e, self.mon(e).cached_speed));
                }
            }
            if !aura.is_empty() {
                let mut best = 0usize;
                for (index, (_, speed)) in aura.iter().enumerate() {
                    if *speed > aura[best].1 {
                        best = index;
                    }
                }
                let relevant = m.category != Category::Status && m.move_type == fairy && target != actor;
                for (index, (holder, speed)) in aura.iter().enumerate() {
                    hooks.push((
                        Priority {
                            priority: 20 * 10000,
                            speed: *speed,
                            sub_order: 7,
                            ..Default::default()
                        },
                        if relevant && index == best { 5448 } else { 4096 },
                    ));
                    let _ = holder;
                }
            }
        }
        // `moves:expandingforce.onBasePower` (priority 0): 1.5x for a grounded
        // user in Psychic Terrain. It chains after the terrain's own 1.3x
        // boost, mirroring the reference's handler priority order.
        if matches!(event, ModifierEvent::BasePower)
            && m.hooks & crate::effects::hook::EXPANDING_FORCE != 0
            && terrain == dex.effects.psychic_terrain
            && self.grounded(dex, actor)
        {
            hooks.push((
                Priority {
                    speed: self.mon(actor).cached_speed,
                    ..Default::default()
                },
                6144,
            ));
        }
        // Two-turn charge recipes that modify this damage: Solar Beam/Solar
        // Blade halve their BasePower in the weak weathers (the move's own
        // priority-0 `onBasePower`), and a charged target's condition doubles
        // Bounce's BasePower from Gust/Twister (`onSourceBasePower`).
        if matches!(event, ModifierEvent::BasePower) {
            let weather = self.effective_weather(dex);
            if let Some(spec) = m.charge.as_ref()
                && spec.half_in_weak_weather
                && matches!(
                    weather,
                    w if w == dex.effects.rain || w == dex.effects.sand || w == dex.effects.snow
                )
            {
                hooks.push((
                    Priority {
                        speed: self.mon(actor).cached_speed,
                        ..Default::default()
                    },
                    2048,
                ));
            }
            if let Some(spec) = self.charging_spec(dex, target)
                && spec.power_double.contains(&m.id)
            {
                hooks.push((
                    Priority {
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    8192,
                ));
            }
        }
        if matches!(event, ModifierEvent::Damage) {
            if let Some(spec) = self.charging_spec(dex, target)
                && spec.damage_double.contains(&m.id)
            {
                hooks.push((
                    Priority {
                        speed: self.mon(target).cached_speed,
                        ..Default::default()
                    },
                    8192,
                ));
            }
            // onAny screen hooks exist on both sides, even when their predicate
            // returns no modifier. Side handlers have no Pokémon speed.
            for (side, state) in self.sides.iter().enumerate() {
                for &id in state.conditions.keys() {
                    let matches_category = (id == dex.effects.reflect
                        && m.category == Category::Physical)
                        || (id == dex.effects.light_screen && m.category == Category::Special);
                    if id == dex.effects.reflect || id == dex.effects.light_screen {
                        hooks.push((
                            Priority {
                                sub_order: 4,
                                ..Default::default()
                            },
                            if target != actor
                                && target.side as usize == side
                                && matches_category
                                && !critical
                                && attacking != Ability::Infiltrator
                            {
                                2732
                            } else {
                                4096
                            },
                        ));
                    } else if id == dex.effects.aurora_veil {
                        // The veil halves both categories but never stacks with
                        // the matching screen (that screen's own handler wins).
                        let screen_takes_over = (m.category == Category::Physical
                            && state.conditions.contains_key(&dex.effects.reflect))
                            || (m.category == Category::Special
                                && state
                                    .conditions
                                    .contains_key(&dex.effects.light_screen));
                        hooks.push((
                            Priority {
                                sub_order: 4,
                                ..Default::default()
                            },
                            if target != actor
                                && target.side as usize == side
                                && !screen_takes_over
                                && !critical
                                && attacking != Ability::Infiltrator
                            {
                                2732
                            } else {
                                4096
                            },
                        ));
                    }
                }
            }
        }
        if matches!(event, ModifierEvent::Damage) {
            let item = dex.effects.items[self.mon(actor).item as usize];
            if matches!(item, Item::LifeOrb | Item::ExpertBelt) {
                hooks.push((
                    Priority {
                        speed: self.mon(actor).cached_speed,
                        sub_order: 8,
                        ..Default::default()
                    },
                    if item == Item::LifeOrb {
                        5324
                    } else if effectiveness > 0 {
                        4915
                    } else {
                        4096
                    },
                ));
            }
        }
        // `abilities:friendguard.onAnyModifyDamage`: every active holder of the
        // ability registers a handler for any damage event (including the
        // target's own, which returns no modifier); the conditional no-op still
        // participates in speed-tie ordering.
        if matches!(event, ModifierEvent::Damage) {
            for holder in self.active_entities(false) {
                if dex.effects.abilities[self.mon(holder).ability as usize]
                    != Ability::Friendguard
                {
                    continue;
                }
                let modifier = if holder != target && holder.side == target.side {
                    3072
                } else {
                    4096
                };
                hooks.push((
                    Priority {
                        speed: self.mon(holder).cached_speed,
                        sub_order: 7,
                        ..Default::default()
                    },
                    modifier,
                ));
            }
        }
        if matches!(event, ModifierEvent::Attack | ModifierEvent::SpecialAttack)
            && self
                .mon(actor)
                .volatiles
                .contains_key(&dex.effects.flash_fire)
        {
            hooks.push((
                Priority {
                    priority: 5 * 10000,
                    speed: self.mon(actor).cached_speed,
                    sub_order: 2,
                    ..Default::default()
                },
                if attacking == Ability::FlashFire && m.move_type == dex.effects.fire {
                    6144
                } else {
                    4096
                },
            ));
        }
        super::item_ports::collect_hooks(self, dex, event, context, &mut hooks)?;
        let participated = !hooks.is_empty();
        speed_sort(&mut hooks, &mut self.rng, |h| h.0);
        let folded = hooks
            .into_iter()
            .fold(4096, |combined, (_, modifier)| {
                damage::chain_modifiers(combined, modifier)
            });
        Ok((folded, participated))
    }

    pub(super) fn modify_value(
        &mut self,
        dex: &Dex,
        event: ModifierEvent,
        context: MoveContext<'_>,
        value: u32,
    ) -> Result<u32> {
        let modifier = self.modifiers(dex, event, context, value)?;
        Ok(stats::modify(value, modifier))
    }

    pub(super) fn damage_modifier(
        &mut self,
        dex: &Dex,
        context: MoveContext<'_>,
    ) -> Result<u32> {
        self.modifiers(dex, ModifierEvent::Damage, context, 0)
    }
}

impl Ability {
    /// Native port gate. An ability is only executable when every reference
    /// callback it declares has a native port; anything else stays an explicit
    /// operational error rather than a silent no-op. Flipping a variant here is
    /// the *only* way to enable it, so the classifier cannot drift from the
    /// implementation.
    pub fn is_ported(self) -> bool {
        !matches!(
            self,
            Ability::Unimplemented
            | Ability::Aftermath
            | Ability::Angerpoint
            | Ability::Anticipation
            | Ability::Battlebond
            | Ability::Berserk
            | Ability::Cheekpouch
            | Ability::Corrosion
            | Ability::Cudchew
            | Ability::Cutecharm
            | Ability::Earlybird
            | Ability::Electromorphosis
            | Ability::Embodyaspectcornerstone
            | Ability::Embodyaspecthearthflame
            | Ability::Embodyaspectteal
            | Ability::Embodyaspectwellspring
            | Ability::Forecast
            | Ability::Forewarn
            | Ability::Gluttony
            | Ability::Guarddog
            | Ability::Gulpmissile
            | Ability::Harvest
            | Ability::Heavymetal
            | Ability::Hungerswitch
            | Ability::Hustle
            | Ability::Iceface
            | Ability::Illuminate
            | Ability::Illusion
            | Ability::Imposter
            | Ability::Innardsout
            | Ability::Keeneye
            | Ability::Klutz
            | Ability::Lightmetal
            | Ability::Longreach
            | Ability::Megasol
            | Ability::Merciless
            | Ability::Mimicry
            | Ability::Opportunist
            | Ability::Pickup
            | Ability::Piercingdrill
            | Ability::Quickdraw
            | Ability::Rattled
            | Ability::Receiver
            | Ability::Ripen
            | Ability::Runaway
            | Ability::Sandspit
            | Ability::Shedskin
            | Ability::Shielddust
            | Ability::Shieldsdown
            | Ability::Skilllink
            | Ability::Stakeout
            | Ability::Stall
            | Ability::Steadfast
            | Ability::Stench
            | Ability::Stickyhold
            | Ability::Suctioncups
            | Ability::Supersweetsyrup
            | Ability::Sweetveil
            | Ability::Symbiosis
            | Ability::Unseenfist
            | Ability::Vitalspirit
            | Ability::Whitesmoke
        )
    }
}
