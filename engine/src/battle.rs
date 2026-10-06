//! Native decision-boundary loop. Missing effect handlers are explicit errors;
//! engine/data/scope.json remains the required scope, independent of this port's progress.
use crate::{
    EngineError, Result,
    actions::{
        ActionKind, AtomicAction, MoveChoice, NO_SLOT, Request, RequestKind, Resource, SlotRequest,
    },
    assets::{Category, Dex, Id, Target},
    damage::{self, DamageInput},
    effects::{Ability, Entity, Item, MoveBehavior, QueuedAction, QueuedKind},
    knowledge::{EffectRef, EventKind, HealthDisplay, SemanticEvent, public_health},
    queue::{Priority, speed_sort},
    state::{
        BattleState, EffectState, EndReason, NativeTrace, Outcome, PokemonState, SideId,
        TraceEntry, TraceEvent,
    },
    stats,
};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use std::cmp::Ordering;
mod hooks;
mod bp_callbacks;
mod item_ports;
mod redirect;
mod room;
mod terrain;
mod type_conversion;
#[cfg(test)]
mod type_conversion_tests;
use type_conversion::ActiveMove;
mod weather;
use hooks::{BoostCause, ModifierEvent, MoveContext};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepResult {
    pub accepted: bool,
    pub outcome: Outcome,
    pub request_kinds: [RequestKind; 2],
}

impl BattleState {
    fn mon(&self, e: Entity) -> &PokemonState {
        &self.sides[e.side as usize].pokemon[e.roster as usize]
    }
    fn mon_mut(&mut self, e: Entity) -> &mut PokemonState {
        &mut self.sides[e.side as usize].pokemon[e.roster as usize]
    }

    pub fn enable_trace(&mut self) -> Result<()> {
        if self.trace.is_none() {
            self.trace = Some(NativeTrace {
                oracle_commit: crate::ORACLE_COMMIT.into(),
                format: crate::FORMAT.into(),
                initial_state: self.snapshot()?,
                actions: vec![],
                events: Default::default(),
            });
        }
        Ok(())
    }
    pub fn export_trace(&self) -> Option<&NativeTrace> {
        self.trace.as_ref()
    }

    pub fn replay_trace(dex: &Dex, trace: &NativeTrace) -> Result<Self> {
        if trace.oracle_commit != crate::ORACLE_COMMIT || trace.format != crate::FORMAT {
            return Err(EngineError::AssetMismatch("trace reference/format".into()));
        }
        let mut state = Self::restore(dex, &trace.initial_state)?;
        if state.trace.is_some() {
            return Err(EngineError::InvalidInput(
                "nested trace initial state".into(),
            ));
        }
        state.enable_trace()?;
        for action in &trace.actions {
            if state.turn != action.turn || state.rng.seed() != action.rng_before {
                return Err(EngineError::AssetMismatch(
                    "trace decision/RNG boundary".into(),
                ));
            }
            state.step(dex, action.side, &action.actions)?;
        }
        if state.trace.as_ref().unwrap().events != trace.events {
            return Err(EngineError::AssetMismatch(
                "trace semantic event stream".into(),
            ));
        }
        Ok(state)
    }

    /// Commit one complete player request. The other player's view is unchanged
    /// until every requested player has committed. Invalid input is non-mutating.
    pub fn step(
        &mut self,
        dex: &Dex,
        side: SideId,
        actions: &[AtomicAction],
    ) -> Result<StepResult> {
        self.validate_choice(side, actions)?;
        let i = side.index();
        if let Some(trace) = &mut self.trace {
            trace.actions.push(TraceEntry {
                side,
                actions: actions.to_vec(),
                rng_before: self.rng.seed(),
                turn: self.turn,
            });
        }
        self.pending[i] = Some(actions.to_vec());
        let ready =
            (0..2).all(|s| self.requests[s].kind == RequestKind::Wait || self.pending[s].is_some());
        if ready && let Err(error) = self.commit(dex) {
            self.outcome.operational_error = Some(error.to_string());
            self.pending = [None, None];
            for r in &mut self.requests {
                r.kind = RequestKind::Finished;
            }
            // This is an operational failure, not a completed battle or draw.
        }
        Ok(StepResult {
            accepted: true,
            outcome: self.outcome.clone(),
            request_kinds: std::array::from_fn(|s| {
                if self.pending[s].is_some() {
                    RequestKind::Wait
                } else {
                    self.requests[s].kind
                }
            }),
        })
    }

    pub(crate) fn validate_choice(&self, side: SideId, actions: &[AtomicAction]) -> Result<()> {
        let i = side.index();
        if self.outcome.terminated
            || self.outcome.truncated
            || self.outcome.operational_error.is_some()
            || self.pending[i].is_some()
            || matches!(
                self.requests[i].kind,
                RequestKind::Wait | RequestKind::Finished
            )
        {
            return Err(EngineError::InvalidInput(
                "side has no unsubmitted request".into(),
            ));
        }
        self.requests[i].validate_joint(actions)
    }

    fn commit(&mut self, dex: &Dex) -> Result<()> {
        self.update_speed(dex);
        if self.requests[0].kind == RequestKind::Preview {
            let mut picks = SmallVec::<[Priority; 8]>::new();
            for side in 0..2 {
                let actions = self.pending[side].take().unwrap();
                let selected = std::array::from_fn(|i| actions[i].switch_destination);
                self.sides[side].selected_order = Some(selected);
                // The reference party array is the preview pick order; its
                // first two entries become the leads.
                self.sides[side].positions = selected;
                for (index, roster) in selected.into_iter().enumerate() {
                    let e = Entity {
                        side: side as u8,
                        roster,
                    };
                    self.mon_mut(e).selected = true;
                    picks.push(Priority {
                        order: 1,
                        priority: -(index as i32) * 10000,
                        speed: self.speed(dex, e),
                        ..Default::default()
                    });
                }
            }
            speed_sort(&mut picks, &mut self.rng, |p| *p);
            for side in 0..2 {
                for slot in 0..2 {
                    let roster = self.sides[side].selected_order.unwrap()[slot];
                    self.switch_in(
                        dex,
                        Entity {
                            side: side as u8,
                            roster,
                        },
                        slot as u8,
                    )?;
                }
            }
            self.mid_turn = true;
        } else {
            let old_queue = std::mem::take(&mut self.queue);
            for side in 0..2 {
                let replacement = self.requests[side].kind == RequestKind::Replacement;
                let Some(actions) = self.pending[side].take() else {
                    continue;
                };
                for action in actions {
                    if action.kind == ActionKind::Pass {
                        continue;
                    }
                    let roster = self.sides[side].active[action.own_slot as usize]
                        .ok_or_else(|| EngineError::InvalidInput("missing active entity".into()))?;
                    let actor = Entity {
                        side: side as u8,
                        roster,
                    };
                    let mut queued = QueuedAction {
                        kind: QueuedKind::Move,
                        actor: Some(actor),
                        move_slot: action.move_slot,
                        move_id: 0,
                        target_location: action.target_location,
                        destination: action.switch_destination,
                        priority: Priority {
                            order: 200,
                            speed: self.speed(dex, actor),
                            ..Default::default()
                        },
                    };
                    if action.kind == ActionKind::Switch {
                        queued.kind = QueuedKind::Switch;
                        queued.priority.order = if replacement { 3 } else { 103 };
                    } else {
                        queued.move_id = if action.move_slot == NO_SLOT {
                            dex.effects.struggle
                        } else {
                            self.mon(actor).moves[action.move_slot as usize].id
                        };
                        let m = &dex.moves[queued.move_id as usize];
                        if dex.effects.moves[m.id as usize] == MoveBehavior::Unimplemented {
                            return Err(EngineError::Unsupported(format!(
                                "move {}",
                                dex.names["moves"][m.id as usize]
                            )));
                        }
                        queued.priority.priority =
                            i32::from(self.effective_priority(dex, actor, m.id)) * 10000;
                        if queued.target_location == 0 {
                            queued.target_location = self.random_target_location(actor, m.target);
                        }
                        // getActionSpeed resolves a target even for a constant
                        // priority. That resolution can consume a reference RNG draw.
                        self.resolve_target_location(actor, m.target, queued.target_location);
                        if action.resource == Resource::Mega {
                            self.queue.push(QueuedAction {
                                kind: QueuedKind::Mega,
                                actor: Some(actor),
                                move_slot: NO_SLOT,
                                move_id: 0,
                                target_location: 0,
                                destination: NO_SLOT,
                                priority: Priority {
                                    order: 104,
                                    speed: self.speed(dex, actor),
                                    ..Default::default()
                                },
                            });
                        }
                    }
                    self.queue.push(queued);
                }
            }
            speed_sort(&mut self.queue, &mut self.rng, |q| q.priority);
            self.queue.extend(old_queue);
            if !self.mid_turn {
                self.insert_action(Self::field_action(QueuedKind::BeforeTurn, 4));
                self.queue
                    .push(Self::field_action(QueuedKind::Residual, 300));
                self.mid_turn = true;
            }
        }
        for r in &mut self.requests {
            r.kind = RequestKind::Wait;
        }
        self.advance(dex)
    }

    fn field_action(kind: QueuedKind, order: u32) -> QueuedAction {
        QueuedAction {
            kind,
            actor: None,
            move_slot: NO_SLOT,
            move_id: 0,
            target_location: 0,
            destination: NO_SLOT,
            priority: Priority {
                order,
                speed: 1,
                ..Default::default()
            },
        }
    }

    fn active_entities(&self, include_fainted: bool) -> SmallVec<[Entity; 4]> {
        let mut result = SmallVec::new();
        for side in 0..2 {
            for roster in self.sides[side].active.iter().flatten() {
                let e = Entity {
                    side: side as u8,
                    roster: *roster,
                };
                if include_fainted || !self.mon(e).fainted {
                    result.push(e);
                }
            }
        }
        result
    }

    fn speed(&self, dex: &Dex, e: Entity) -> i32 {
        let mut speed = stats::apply_stage(u32::from(self.mon(e).stats[5]), self.mon(e).boosts[4]);
        // Faint processing sets reference isActive=false, preventing both own
        // handlers and bubbling to side conditions during getActionSpeed.
        if self.mon(e).fainted {
            return self.action_speed(dex, speed);
        }
        let mut modifier = 4096;
        let weather = self.effective_weather(dex);
        let ability = dex.effects.abilities[self.mon(e).ability as usize];
        if matches!((ability, weather), (Ability::SwiftSwim, w) if w == dex.effects.rain)
            || matches!((ability, weather), (Ability::Chlorophyll, w) if w == dex.effects.sun)
            || matches!((ability, weather), (Ability::SandRush, w) if w == dex.effects.sand)
            || matches!((ability, weather), (Ability::SlushRush, w) if w == dex.effects.snow)
        {
            modifier = damage::chain_modifiers(modifier, 8192);
        }
        // `abilities:unburden` condition `onModifySpe`: doubles Speed while the
        // volatile is present and the holder has no item.
        if ability == Ability::Unburden
            && self.mon(e).item == 0
            && self.mon(e).volatiles.contains_key(&dex.effects.unburden)
        {
            modifier = damage::chain_modifiers(modifier, 8192);
        }
        // `abilities:quickfeet.onModifySpe`: 1.5x while statused.
        if ability == Ability::Quickfeet && self.mon(e).status != 0 {
            modifier = damage::chain_modifiers(modifier, 6144);
        }
        if dex.effects.items[self.mon(e).item as usize] == Item::ChoiceScarf {
            modifier = damage::chain_modifiers(modifier, 6144);
        }
        // Iron Ball is the other `onModifySpe` item; both are item handlers and
        // consequently chain after any ability handler on the same holder.
        if dex.effects.items[self.mon(e).item as usize] == Item::IronBall {
            modifier = damage::chain_modifiers(modifier, 2048);
        }
        if self.sides[e.side as usize]
            .conditions
            .contains_key(&dex.effects.tailwind)
        {
            modifier = damage::chain_modifiers(modifier, 8192);
        }
        speed = stats::modify(speed, modifier);
        // `par.onModifySpe` halves the stat unless the holder has Quick Feet.
        if self.mon(e).status == dex.effects.paralysis && ability != Ability::Quickfeet {
            speed /= 2;
        }
        self.action_speed(dex, speed.min(10000))
    }

    fn update_speed(&mut self, dex: &Dex) {
        for e in self.active_entities(false) {
            self.mon_mut(e).cached_speed = self.speed(dex, e);
        }
    }

    /// Reference `getActionSpeed`: base move priority plus ported
    /// `onModifyPriority` callbacks (Grassy Glide).
    fn effective_priority(&self, dex: &Dex, actor: Entity, move_id: Id) -> i8 {
        let mut priority = dex.moves[move_id as usize].priority;
        if dex.effects.move_hooks[move_id as usize]
            & crate::effects::hook::PRIORITY_GRASSY_GLIDE
            != 0
            && self.terrain_id(dex) == dex.effects.grassy_terrain
            && self.grounded(dex, actor)
        {
            priority += 1;
        }
        // Prankster raises the priority of status moves by one. The reference
        // resolves this through the actor's own `onModifyPriority` handler.
        if dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Prankster
            && dex.moves[move_id as usize].category == Category::Status
        {
            priority += 1;
        }
        // `abilities:galewings.onModifyPriority`: Flying moves gain +1 priority
        // while the holder is exactly at full HP.
        if dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Galewings
            && dex.moves[move_id as usize].move_type == dex.effects.flying
            && self.mon(actor).hp == self.mon(actor).stats[0]
        {
            priority += 1;
        }
        priority
    }

    /// Reference Sucker Punch `onTry`: the move only runs when the chosen
    /// target has a queued move action and that move is not a status move.
    fn sucker_punch_target_attacks(&self, dex: &Dex, target: Option<Entity>) -> bool {
        let Some(target) = target else {
            return false;
        };
        if self.mon(target).fainted || self.mon(target).hp == 0 {
            return false;
        }
        let Some(action) = self
            .queue
            .iter()
            .find(|q| q.kind == QueuedKind::Move && q.actor == Some(target))
        else {
            return false;
        };
        dex.moves[action.move_id as usize].category != Category::Status
    }

    fn each_update(&mut self, dex: &Dex) -> Result<()> {
        let mut active: SmallVec<[(Entity, Priority); 4]> = self
            .active_entities(false)
            .into_iter()
            .map(|e| {
                (
                    e,
                    Priority {
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                )
            })
            .collect();
        if std::env::var("PA3_RNG_DBG").is_ok() {
            eprintln!(
                "RNG update-list {:?} speeds {:?}",
                active.iter().map(|(e, _)| (e.side, e.roster)).collect::<Vec<_>>(),
                active.iter().map(|(_, p)| p.speed).collect::<Vec<_>>()
            );
        }
        speed_sort(&mut active, &mut self.rng, |x| x.1);
        for (e, _) in active {
            // `eachEvent('Update')` collects only the holder's own handlers, in
            // status -> volatile -> ability -> item order. The ported
            // status-immunity `onUpdate` cures therefore run before any item
            // Update handler.
            if self.update_cured_status(dex, e) {
                self.reveal_ability(e)?;
                self.cure_status(e)?;
            }
            self.item_update(dex, e)?;
        }
        Ok(())
    }

    fn insert_action(&mut self, action: QueuedAction) {
        let first = self
            .queue
            .iter()
            .position(|q| action.priority.compare(&q.priority) != Ordering::Greater);
        if let Some(first) = first {
            let last = self
                .queue
                .iter()
                .position(|q| action.priority.compare(&q.priority) == Ordering::Less)
                .unwrap_or(self.queue.len());
            let at = if first == last {
                first
            } else {
                self.rng.range(first as u32, last as u32 + 1) as usize
            };
            self.queue.insert(at, action);
        } else {
            self.queue.push(action);
        }
    }

    fn validate_effects(&self, dex: &Dex, e: Entity) -> Result<()> {
        let mon = self.mon(e);
        if !dex.effects.abilities[mon.ability as usize].is_ported() {
            return Err(EngineError::Unsupported(format!(
                "ability {}",
                dex.names["abilities"][mon.ability as usize]
            )));
        }
        if dex.effects.items[mon.item as usize] == Item::Unimplemented
            && dex.effects.mega_stones[mon.item as usize].is_empty()
        {
            return Err(EngineError::Unsupported(format!(
                "item {}",
                dex.names["items"][mon.item as usize]
            )));
        }
        Ok(())
    }

    fn switch_in(&mut self, dex: &Dex, incoming: Entity, slot: u8) -> Result<()> {
        self.switch_in_inner(dex, incoming, slot, false)
    }

    /// Reference `switchIn`. `drag` mirrors the reference's `isDrag` argument:
    /// a dragged-in Pokémon runs its SwitchIn event immediately instead of
    /// queueing a `runSwitch` action.
    fn switch_in_inner(
        &mut self,
        dex: &Dex,
        incoming: Entity,
        slot: u8,
        drag: bool,
    ) -> Result<()> {
        self.validate_effects(dex, incoming)?;
        if let Some(roster) = self.sides[incoming.side as usize].active[slot as usize] {
            let old = Entity {
                side: incoming.side,
                roster,
            };
            // A `selfSwitch` pivot already ran `BeforeSwitchOut` when the
            // switch request was issued, which flags the reference to skip the
            // pre-switch Update inside `switchIn` as well.
            let pivot_switch = self.mon(old).switch_flag.is_some();
            if self.mon(old).hp > 0 {
                // Reference `switchIn` runs BeforeSwitchOut plus a full Update
                // for a voluntary switch only.
                if !drag && !pivot_switch {
                    self.each_update(dex)?;
                }
                self.ability_switch_out(dex, old)?;
                self.ability_end(dex, old)?;
            }
            self.clear_volatile(dex, old);
            // Reference `clearVolatile` also drops the pending switch flags of
            // the Pokémon leaving the field.
            self.mon_mut(old).switch_flag = None;
            self.mon_mut(old).force_switch_flag = false;
            self.mon_mut(old).active_slot = None;
            // Reference `switchIn` swaps the outgoing Pokémon into the incoming
            // Pokémon's reserve slot, so the party array order is preserved.
            let positions = &mut self.sides[incoming.side as usize].positions;
            if let Some(src) = positions.iter().position(|r| *r == incoming.roster) {
                positions.swap(src, slot as usize);
            }
        }
        self.sides[incoming.side as usize].active[slot as usize] = Some(incoming.roster);
        self.mon_mut(incoming).active_slot = Some(slot);
        self.mon_mut(incoming).switch_flag = None;
        self.mon_mut(incoming).force_switch_flag = false;
        self.mon_mut(incoming).ability_ending = false;
        let ability_order = if self.mon(incoming).ability != 0 {
            Some(self.allocate_effect_order()?)
        } else {
            None
        };
        let item_order = if self.mon(incoming).item != 0 {
            Some(self.allocate_effect_order()?)
        } else {
            None
        };
        self.mon_mut(incoming).ability_effect_order = ability_order;
        self.mon_mut(incoming).item_effect_order = item_order;
        self.mon_mut(incoming).active_turns = 0;
        self.mon_mut(incoming).active_move_actions = 0;
        if self.mon(incoming).status == dex.effects.toxic {
            self.mon_mut(incoming).status_state.values[0] = 0;
        }
        self.mon_mut(incoming)
            .moves
            .iter_mut()
            .for_each(|m| m.used = false);
        self.emit(
            EventKind::Switch,
            incoming,
            None,
            EffectRef::None,
            i32::from(slot),
            true,
        )?;
        // Reference insertChoice refreshes the entering Pokémon's cached speed.
        self.mon_mut(incoming).cached_speed = self.speed(dex, incoming);
        let priority = Priority {
            order: 101,
            speed: self.mon(incoming).cached_speed,
            ..Default::default()
        };
        if drag {
            // `isDrag` runs the switch-in events synchronously in the reference.
            self.run_switch_batch(dex, smallvec![incoming])?;
        } else {
            self.insert_action(QueuedAction {
                kind: QueuedKind::RunSwitch,
                actor: Some(incoming),
                move_slot: NO_SLOT,
                move_id: 0,
                target_location: 0,
                destination: NO_SLOT,
                priority,
            });
        }
        Ok(())
    }

    /// Shared `runSwitch` batch: SwitchIn ties are resolved once over all
    /// active Pokémon, then every entrant's handlers use that fixed order.
    fn run_switch_batch(&mut self, dex: &Dex, incoming: SmallVec<[Entity; 4]>) -> Result<()> {
        let mut active: SmallVec<[(Entity, Priority); 4]> = self
            .active_entities(true)
            .into_iter()
            .map(|e| {
                (
                    e,
                    Priority {
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                )
            })
            .collect();
        speed_sort(&mut active, &mut self.rng, |p| p.1);
        for (e, _) in active {
            if incoming.contains(&e) {
                self.ability_switch_in(dex, e)?;
            }
        }
        // Held-item Start follows ability Start for each entrant, then the
        // any-switch-in White Herb check.
        for e in incoming.iter().copied() {
            item_ports::start(self, dex, e)?;
        }
        item_ports::white_herb_event(self, dex)?;
        Ok(())
    }

    fn mega_form(&self, dex: &Dex, e: Entity) -> Option<Id> {
        if self.sides[e.side as usize].mega_used {
            return None;
        }
        let mon = self.mon(e);
        dex.effects.mega_stones[mon.item as usize]
            .iter()
            .find_map(|(base, mega)| (*base == mon.base_species).then_some(*mega))
    }

    fn run_mega(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        let Some(form) = self.mega_form(dex, e) else {
            return Ok(());
        };
        let species = &dex.species[form as usize];
        let ability = species.abilities[0];
        if !dex.effects.abilities[ability as usize].is_ported() {
            return Err(EngineError::Unsupported(format!(
                "Mega ability {}",
                dex.names["abilities"][ability as usize]
            )));
        }
        let mon = self.mon_mut(e);
        let new_stats = stats::champions_stats(
            species.base_stats,
            mon.points,
            dex.natures[mon.nature as usize],
            species.max_hp,
        );
        // Champions permanent forme changes preserve damage taken and can change
        // max HP through this shared forme-change path. Mega formes persist through fainting.
        let damage_taken = mon.stats[0] - mon.hp;
        if mon.hp > 0 {
            mon.hp = new_stats[0].saturating_sub(damage_taken).max(1);
        }
        mon.stats = new_stats;
        mon.cached_speed = i32::from(new_stats[5]);
        mon.species = form;
        mon.base_species = form;
        mon.types = species.types.clone();
        self.emit(EventKind::Mega, e, None, EffectRef::Species(form), 0, true)?;
        self.ability_end(dex, e)?;
        let ability_order = Some(self.allocate_effect_order()?);
        let mon = self.mon_mut(e);
        mon.ability = ability;
        mon.ability_effect_order = ability_order;
        mon.base_ability = ability;
        mon.ability_ending = false;
        self.sides[e.side as usize].mega_used = true;
        // The permanent forme and its single ability are public information.
        self.emit(
            EventKind::Ability,
            e,
            None,
            EffectRef::Ability(ability),
            0,
            false,
        )?;
        self.emit(
            EventKind::Item,
            e,
            None,
            EffectRef::Item(self.mon(e).item),
            0,
            false,
        )?;
        for viewer in 0..2 {
            let index = e.roster as usize + if e.side as usize == viewer { 0 } else { 6 };
            self.knowledge[viewer].pokemon[index].types = species.types.clone();
        }
        self.ability_start(dex, e)?;
        Ok(())
    }

    fn clear_volatile(&mut self, dex: &Dex, e: Entity) {
        let mon = self.mon_mut(e);
        mon.boosts = [0; 7];
        mon.volatiles.clear();
        mon.transformed = false;
        mon.ability = mon.base_ability;
        mon.species = mon.base_species;
        mon.types = dex.species[mon.species as usize].types.clone();
        mon.stats = stats::champions_stats(
            dex.species[mon.species as usize].base_stats,
            mon.points,
            dex.natures[mon.nature as usize],
            dex.species[mon.species as usize].max_hp,
        );
        mon.cached_speed = i32::from(mon.stats[5]);
        // Untransformed move PP is shared with the base set; reset only temporary moves.
        // Reference `moveSlots = baseMoveSlots.slice()` copies array slots that
        // alias the base objects, so the disabled flag (and PP) survives a
        // switch and is only recomputed for active Pokémon at turn start.
        let disabled: SmallVec<[bool; 4]> = mon.moves.iter().map(|m| m.disabled).collect();
        mon.moves = mon.base_moves.clone();
        for (slot, flag) in mon.moves.iter_mut().zip(disabled) {
            slot.disabled = flag;
        }
    }

    fn advance(&mut self, dex: &Dex) -> Result<()> {
        while !self.queue.is_empty() {
            let action = self.queue.remove(0);
            match action.kind {
                QueuedKind::BeforeTurn => self.each_update(dex)?,
                QueuedKind::RunSwitch => {
                    let mut incoming: SmallVec<[Entity; 4]> = smallvec![action.actor.unwrap()];
                    while self
                        .queue
                        .first()
                        .is_some_and(|q| q.kind == QueuedKind::RunSwitch)
                    {
                        incoming.push(self.queue.remove(0).actor.unwrap());
                    }
                    self.run_switch_batch(dex, incoming)?;
                }
                QueuedKind::Switch => {
                    let actor = action.actor.unwrap();
                    let slot = self.mon(actor).active_slot.unwrap();
                    self.switch_in(
                        dex,
                        Entity {
                            side: actor.side,
                            roster: action.destination,
                        },
                        slot,
                    )?;
                }
                QueuedKind::Move => {
                    let actor = action.actor.unwrap();
                    if self.mon(actor).fainted || self.mon(actor).active_slot.is_none() {
                        continue;
                    }
                    self.use_move(
                        dex,
                        actor,
                        action.move_slot,
                        action.move_id,
                        action.target_location,
                    )?;
                    item_ports::white_herb_event(self, dex)?;
                }
                QueuedKind::Residual => self.residual(dex)?,
                QueuedKind::Mega => self.run_mega(dex, action.actor.unwrap())?,
            }
            // Reference phazing runs directly after the action and before the
            // faint check: a dragged-in Pokémon's SwitchIn events resolve
            // synchronously.
            self.resolve_forced_switches(dex)?;
            if std::env::var("PA3_RNG_DBG").is_ok() {
                eprintln!(
                    "RNG after action {:?} actor {:?} seed {:?} draws {}",
                    action.kind,
                    action.actor,
                    self.rng.seed(),
                    self.rng.draws
                );
            }
            self.process_faints(dex, true)?;
            if self.outcome.terminated {
                return Ok(());
            }
            // Reference returns before Update while another forced replacement
            // is pending. Both entrants arrive before SwitchIn/Update tie draws.
            if self
                .queue
                .first()
                .is_some_and(|next| next.kind == QueuedKind::Switch && next.priority.order == 3)
            {
                continue;
            }
            self.each_update(dex)?;
            if self.make_pivot_requests(dex) {
                return Ok(());
            }
            if self.queue.is_empty() && self.make_replacement_requests() {
                return Ok(());
            }
            if self
                .queue
                .first()
                .is_some_and(|q| q.kind == QueuedKind::Move)
            {
                self.update_speed(dex);
                for i in 0..self.queue.len() {
                    if let Some(e) = self.queue[i].actor {
                        self.queue[i].priority.speed = self.speed(dex, e);
                        if self.queue[i].kind == QueuedKind::Move {
                            // Reference re-resolves priority through
                            // getActionSpeed for every queued action here.
                            let move_id = self.queue[i].move_id;
                            self.queue[i].priority.priority =
                                i32::from(self.effective_priority(dex, e, move_id)) * 10000;
                        }
                    }
                    if self.queue[i].kind == QueuedKind::Move {
                        let q = &self.queue[i];
                        self.resolve_target_location(
                            q.actor.unwrap(),
                            dex.moves[q.move_id as usize].target,
                            q.target_location,
                        );
                    }
                }
                speed_sort(&mut self.queue, &mut self.rng, |q| q.priority);
            }
        }
        self.end_turn(dex)
    }

    fn random_target_location(&mut self, actor: Entity, target: Target) -> i8 {
        let slot = self.mon(actor).active_slot.unwrap();
        if matches!(
            target,
            Target::SelfOnly
                | Target::All
                | Target::AllySide
                | Target::AllyTeam
                | Target::AdjacentAllyOrSelf
        ) {
            return -(slot as i8 + 1);
        }
        let side = if target == Target::AdjacentAlly {
            actor.side
        } else {
            1 - actor.side
        };
        let options: SmallVec<[i8; 2]> = self.sides[side as usize]
            .active
            .iter()
            .enumerate()
            .filter_map(|(slot, roster)| {
                let e = Entity {
                    side,
                    roster: (*roster)?,
                };
                (self.mon(e).hp > 0 && e != actor).then_some(if side == actor.side {
                    -(slot as i8 + 1)
                } else {
                    slot as i8 + 1
                })
            })
            .collect();
        if options.is_empty() {
            if side == actor.side { 0 } else { 1 }
        } else {
            options[self.rng.below(options.len() as u32) as usize]
        }
    }

    fn at_location(&self, actor: Entity, loc: i8) -> Option<Entity> {
        if loc == 0 {
            return None;
        }
        let side = if loc < 0 { actor.side } else { 1 - actor.side };
        self.sides[side as usize].active[loc.unsigned_abs() as usize - 1]
            .map(|roster| Entity { side, roster })
    }

    fn resolve_target_location(&mut self, actor: Entity, target: Target, loc: i8) -> i8 {
        let slot = self.mon(actor).active_slot.unwrap_or(0);
        if target != Target::RandomNormal
            && target.valid_location(slot, loc)
            && let Some(e) = self.at_location(actor, loc)
        {
            if !self.mon(e).fainted {
                return loc;
            }
            if e.side == actor.side {
                return if target == Target::AdjacentAllyOrSelf {
                    -(slot as i8 + 1)
                } else {
                    loc
                };
            }
        }
        self.random_target_location(actor, target)
    }

    fn targets(&mut self, actor: Entity, target: Target, loc: i8) -> SmallVec<[Entity; 4]> {
        if target == Target::SelfOnly {
            return smallvec![actor];
        }
        if target == Target::Allies {
            // Reference `alliesAndSelf`: every live slot on the acting side in
            // slot order, including the user.
            return self.sides[actor.side as usize]
                .active
                .iter()
                .flatten()
                .map(|roster| Entity {
                    side: actor.side,
                    roster: *roster,
                })
                .filter(|e| self.mon(*e).hp > 0)
                .collect();
        }
        if matches!(target, Target::AllAdjacent | Target::AllAdjacentFoes) {
            // Reference spread ordering: adjacent allies first, then foes.
            let mut all = self.active_entities(false);
            all.sort_by_key(|e| if e.side == actor.side { 0 } else { 1 });
            return all
                .into_iter()
                .filter(|e| *e != actor && (target == Target::AllAdjacent || e.side != actor.side))
                .collect();
        }
        self.at_location(actor, loc)
            .filter(|e| self.mon(*e).hp > 0)
            .into_iter()
            .collect()
    }

    fn use_move(&mut self, dex: &Dex, actor: Entity, slot: u8, move_id: Id, loc: i8) -> Result<()> {
        let m = &dex.moves[move_id as usize];
        let behavior = dex.effects.moves[move_id as usize];
        self.validate_effects(dex, actor)?;
        // `abilities:damp.onAnyTryMove` cancels the explosion family at the
        // reference TryMove stage, which runs after PP deduction but before the
        // move message and every hit step. The gate must precede the native
        // unsupported-move error so a blocked Misty Explosion never becomes an
        // operational failure while a Damp holder is active.
        if dex.effects.damp_moves.contains(&move_id)
            && let Some(holder) = self.active_entities(false).into_iter().find(|h| {
                dex.effects.abilities[self.mon(*h).ability as usize] == Ability::Damp
            })
        {
            if slot != NO_SLOT {
                let mon = self.mon_mut(actor);
                let pp = mon.moves[slot as usize].pp;
                if pp > 0 {
                    mon.moves[slot as usize].pp = pp - 1;
                    mon.moves[slot as usize].used = true;
                    mon.base_moves[slot as usize].pp = pp - 1;
                }
            }
            self.emit(
                EventKind::Move,
                actor,
                None,
                EffectRef::Move(move_id),
                0,
                false,
            )?;
            self.reveal_ability(holder)?;
            // The reference TryMove abort still runs the single Update that
            // precedes the action's own queue re-sort.
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::Unimplemented {
            return Err(EngineError::Unsupported(format!("move ID {move_id}")));
        }
        // Reference `runMove` counts the attempted action before BeforeMove, so
        // a flinched, sleeping or fully paralysed attempt still counts.
        let attempts = self.mon(actor).active_move_actions;
        self.mon_mut(actor).active_move_actions = attempts.saturating_add(1);
        let loc = self.resolve_target_location(actor, m.target, loc);
        if !self.before_move(dex, actor, m)? {
            return Ok(());
        }
        if slot != NO_SLOT {
            let mon = self.mon_mut(actor);
            let pp = mon.moves[slot as usize].pp;
            if pp == 0 {
                return Ok(());
            }
            mon.moves[slot as usize].pp = pp - 1;
            mon.moves[slot as usize].used = true;
            mon.base_moves[slot as usize].pp = pp - 1;
        }
        if m.defrost && self.mon(actor).status == dex.effects.freeze {
            self.cure_status(actor)?;
        }
        if matches!(
            dex.effects.items[self.mon(actor).item as usize],
            Item::ChoiceScarf | Item::ChoiceBand | Item::ChoiceSpecs
        )
            && behavior != MoveBehavior::Struggle
            && !self
                .mon(actor)
                .volatiles
                .contains_key(&dex.effects.choice_lock)
        {
            let order = self.allocate_effect_order()?;
            self.mon_mut(actor).volatiles.insert(
                dex.effects.choice_lock,
                EffectState {
                    id: dex.effects.choice_lock,
                    values: vec![i64::from(move_id)],
                    effect_order: order,
                    effect_order_assigned: true,
                    ..Default::default()
                },
            );
        }
        let weather = self.effective_weather(dex);
        let action = self.active_move(dex, actor, m, behavior);
        // A move-owned ModifyMove that changes the target class makes the
        // reference re-resolve a random target after both dispatches
        // (`singleEvent` then `runEvent`): two samples that spread moves then
        // ignore. Expanding Force's Psychic Terrain conversion is the only
        // ported case.
        if action.target != dex.moves[move_id as usize].target
            && dex.effects.move_hooks[move_id as usize] & crate::effects::hook::EXPANDING_FORCE != 0
        {
            self.random_target_location(actor, action.target);
            self.random_target_location(actor, action.target);
        }
        let m = &action;
        let selected = if m.target == Target::SelfOnly {
            Some(actor)
        } else {
            self.at_location(actor, loc)
        };
        let redirected = self.redirect_target(dex, actor, m, selected)?;
        let mut targets = self.targets(actor, m.target, loc);
        if !matches!(
            m.target,
            Target::All
                | Target::FoeSide
                | Target::AllySide
                | Target::AllyTeam
                | Target::AllAdjacent
                | Target::AllAdjacentFoes
                | Target::Allies
        ) && let Some(target) = redirected
            && !self.mon(target).fainted
            && self.mon(target).hp > 0
        {
            targets.clear();
            targets.push(target);
        }
        self.emit(
            EventKind::Move,
            actor,
            targets.first().copied(),
            EffectRef::Move(move_id),
            0,
            false,
        )?;
        // `abilities:armortail|queenlymajesty.onFoeTryMove` (the reference
        // TryMove event) runs after the public move message and before the
        // move-owned `Try` gates. A positive-priority move aimed at the
        // holder's side is cancelled; `foeSide` and `all` target classes are
        // exempt. The `cant` message names the ability publicly.
        if !matches!(
            m.target,
            Target::FoeSide
                | Target::All
                | Target::AllySide
                | Target::AllyTeam
                | Target::Allies
                | Target::SelfOnly
        ) {
            let priority = self.effective_priority(dex, actor, move_id);
            if priority > 0 {
                let blocked = self.active_entities(false).into_iter().find(|holder| {
                    holder.side != actor.side
                        && matches!(
                            dex.effects.abilities[self.mon(*holder).ability as usize],
                            Ability::Armortail | Ability::Queenlymajesty
                        )
                        && targets.iter().any(|t| t.side == holder.side)
                });
                if let Some(holder) = blocked {
                    self.reveal_ability(holder)?;
                    return Ok(());
                }
            }
        }
        // Move-owned `Try` gates run after the public move message and before
        // redirection-independent hit steps. A failed try consumes no RNG and
        // ends the move without damage or secondary effects.
        let hooks = dex.effects.move_hooks[move_id as usize];
        if hooks & crate::effects::hook::FAKE_OUT_FIRST_TURN != 0
            && self.mon(actor).active_move_actions > 1
        {
            return Ok(());
        }
        if hooks & crate::effects::hook::SUCKER_PUNCH != 0 {
            let target = redirected.or(selected);
            if !self.sucker_punch_target_attacks(dex, target) {
                return Ok(());
            }
        }
        // `moves:teleport.onTry`: Teleport fails before any hit step when the
        // user has no switchable reserve. The plain `selfSwitch` moves instead
        // nullify their own result after a failed pivot attempt.
        if hooks & crate::effects::hook::TELEPORT != 0 && !self.can_switch(actor.side as usize) {
            return Ok(());
        }
        if behavior == MoveBehavior::Terrain {
            self.start_terrain(dex, actor, m.terrain, false)?;
            return Ok(());
        }
        if behavior == MoveBehavior::TrickRoom {
            self.toggle_trick_room(dex, actor)?;
            return Ok(());
        }
        if behavior == MoveBehavior::Weather {
            self.start_weather(dex, actor, m.weather, false)?;
            return Ok(());
        }
        if behavior == MoveBehavior::SideCondition {
            // Side-target moves use tryMoveHit, bypassing the Pokémon hit loop
            // and its two Update events. The queue runs the post-action Update.
            // `moves:auroraveil.onTry`: the screen only starts in snow.
            if hooks & crate::effects::hook::AURORA_VEIL != 0
                && self.effective_weather(dex) != dex.effects.snow
            {
                return Ok(());
            }
            self.ally_try_hit_side(dex, actor, actor, m.move_type)?;
            self.start_side_condition(dex, actor, m.side_condition)?;
            return Ok(());
        }
        if matches!(
            behavior,
            MoveBehavior::HelpingHand | MoveBehavior::FollowMe | MoveBehavior::RagePowder
        ) {
            // These three moves set a duration-one volatile and run the two
            // move-loop Update events without any accuracy or damage step.
            let (victim, volatile) = match behavior {
                MoveBehavior::HelpingHand => {
                    // `moves:helpinghand.onTryHit`: the ally must have a
                    // queued action this turn unless it just switched in.
                    let Some(ally) = self.at_location(actor, loc) else {
                        return Ok(());
                    };
                    if ally == actor
                        || self.mon(ally).active_turns > 0
                            && !self
                                .queue
                                .iter()
                                .any(|q| q.actor == Some(ally))
                    {
                        return Ok(());
                    }
                    (ally, dex.effects.helping_hand)
                }
                MoveBehavior::FollowMe => (actor, dex.effects.follow_me),
                _ => (actor, dex.effects.rage_powder),
            };
            let existing = self
                .mon(victim)
                .volatiles
                .get(&volatile)
                .and_then(|state| state.values.first().copied());
            let order = match self.mon(victim).volatiles.get(&volatile) {
                Some(state) => state.effect_order,
                None => self.allocate_effect_order()?,
            };
            // `helpinghand` restarts multiply its stored BasePower multiplier;
            // Follow Me / Rage Powder simply refresh their one-turn duration.
            let multiplier = if volatile == dex.effects.helping_hand {
                damage::chain_modifiers(existing.unwrap_or(4096) as u32, 6144)
            } else {
                4096
            };
            self.mon_mut(victim).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    effect_order: order,
                    effect_order_assigned: true,
                    duration: Some(1),
                    source: Some((
                        if actor.side == 0 {
                            SideId::P1
                        } else {
                            SideId::P2
                        },
                        actor.roster,
                    )),
                    values: vec![i64::from(multiplier)],
                },
            );
            self.emit(
                EventKind::EffectStart,
                victim,
                Some(actor),
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::Guard {
            // Wide Guard / Quick Guard: `onTry` requires a remaining action,
            // then `sideCondition` and `onHitSide` (which adds `stall`) run
            // without a StallMove roll.
            let acts_left = self
                .queue
                .iter()
                .any(|q| matches!(q.kind, QueuedKind::Move | QueuedKind::Switch));
            if !acts_left {
                return Ok(());
            }
            let condition = m.side_condition;
            if condition == 0 {
                return Err(EngineError::Unsupported(format!(
                    "guard move {move_id} has no side condition"
                )));
            }
            self.start_guard(dex, actor, condition)?;
            let counter = self
                .mon(actor)
                .volatiles
                .get(&dex.effects.stall)
                .and_then(|s| s.values.first())
                .copied();
            self.add_stall(dex, actor, counter)?;
            return Ok(());
        }
        if matches!(behavior, MoveBehavior::Protect | MoveBehavior::Endure) {
            let acts_left = self
                .queue
                .iter()
                .any(|q| matches!(q.kind, QueuedKind::Move | QueuedKind::Switch));
            if !acts_left {
                return Ok(());
            }
            let counter = self
                .mon(actor)
                .volatiles
                .get(&dex.effects.stall)
                .and_then(|s| s.values.first())
                .copied();
            if let Some(counter) = counter
                && !self.rng.chance(1, counter as u32)
            {
                self.mon_mut(actor).volatiles.remove(&dex.effects.stall);
                return Ok(());
            }
            let volatile = if behavior == MoveBehavior::Endure {
                dex.effects.endure
            } else {
                m.hit.volatile
            };
            if volatile == 0 {
                return Err(EngineError::Unsupported(format!(
                    "stalling move {move_id} has no protection volatile"
                )));
            }
            let protect_order = self.allocate_effect_order()?;
            self.mon_mut(actor).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    effect_order: protect_order,
                    effect_order_assigned: true,
                    duration: Some(1),
                    ..Default::default()
                },
            );
            self.add_stall(dex, actor, counter)?;
            self.emit(
                EventKind::EffectStart,
                actor,
                None,
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            // Status self-target effects still run the two move-loop Update events.
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if m.multihit.is_some() {
            return self.use_multihit_move(dex, actor, move_id, m, targets);
        }
        let spread = targets.len() > 1;
        // `selfdestruct: 'always'` faints the user before any target
        // resolution, accuracy check or absorption callback.
        if m.self_destruct == crate::assets::SelfDestructMode::Always {
            self.faint_now(actor);
        }
        // TryHit callbacks share the action accuracy sentinel across every
        // recipient. Resolve that sentinel before any spread accuracy draws.
        let mut action_accuracy = m.accuracy.map(u16::from);
        let effective_priority = self.effective_priority(dex, actor, move_id);
        let mut hit = SmallVec::<[(Entity, i8); 4]>::new();
        // `TryHitSide` runs before the hit steps. Soundproof's own
        // `onAllyTryHitSide` announces the holder when a sound move reaches a
        // spread target on the holder's side; it never blocks the move.
        if targets.len() > 1 && m.sound {
            for target in &targets {
                for ally in self.active_entities(false) {
                    if ally.side == target.side
                        && dex.effects.abilities[self.mon(ally).ability as usize]
                            == Ability::Soundproof
                    {
                        self.reveal_ability(ally)?;
                    }
                }
            }
        }
        for target in targets {
            self.validate_effects(dex, target)?;
            // `hitStepTryHitEvent` runs whole-spread with handlers ordered by
            // priority: the priority-4 side guards and the priority-3
            // protection volatiles both precede every ability TryHit.
            if m.protect && !m.breaks_protect {
                if self.guard_blocks(dex, target, m, effective_priority) {
                    continue;
                }
                if let Some(volatile) = self.blocking_protection(dex, target) {
                    self.protect_punish(dex, target, actor, m, move_id, volatile)?;
                    continue;
                }
            }
            let ability = dex.effects.abilities[self.mon(target).ability as usize];
            if self.terrain_id(dex) == dex.effects.psychic_terrain
                && effective_priority > 0
                && target.side != actor.side
                && self.grounded(dex, target)
            {
                continue;
            }
            if m.powder
                && target != actor
                && ability == Ability::Overcoat
                && !self.mon(target).types.contains(&dex.effects.grass)
            {
                self.reveal_ability(target)?;
                continue;
            }
            // TryHit precedes type and natural powder immunity. Sap Sipper
            // therefore boosts even when a Grass holder receives Sleep Powder.
            if self.absorb_try_hit(dex, target, actor, m, &mut action_accuracy)? {
                continue;
            }
            // Natural Prankster immunity (gen 7+): a status move boosted by the
            // attacker's Prankster cannot affect a Dark-type foe. It is checked
            // after TryHit absorption and before type immunity and accuracy.
            if m.category == Category::Status
                && target.side != actor.side
                && dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Prankster
                && self.mon(target).types.contains(&dex.effects.dark)
            {
                self.emit(
                    EventKind::Ability,
                    actor,
                    None,
                    EffectRef::Ability(self.mon(actor).ability),
                    0,
                    false,
                )?;
                continue;
            }
            // Scrappy's ModifyMove marks Normal and Fighting actions as
            // immunity-bypassing for this action only.
            let scrappy_bypass = m.scrappy
                && (m.move_type == dex.effects.normal || m.move_type == dex.effects.fighting);
            let effectiveness =
                if behavior == MoveBehavior::Struggle || m.ignore_immunity || scrappy_bypass {
                Some(0)
            } else if m.move_type == dex.effects.ground && ability == Ability::Levitate {
                self.emit(
                    EventKind::Ability,
                    target,
                    None,
                    EffectRef::Ability(self.mon(target).ability),
                    0,
                    false,
                )?;
                None
            } else {
                self.mon(target)
                    .types
                    .iter()
                    .try_fold(0i8, |total, &kind| {
                        let mut value = dex.type_chart[m.move_type as usize][kind as usize];
                        // Freeze-Dry's own onEffectiveness replaces the Water
                        // resistance with a super-effective value.
                        if hooks & crate::effects::hook::FREEZE_DRY != 0
                            && kind == dex.effects.water
                        {
                            value = 1;
                        }
                        if value == -127 {
                            None
                        } else {
                            Some(total + value)
                        }
                    })
                    .map(|total| total.clamp(-6, 6))
            };
            if m.powder && target != actor && self.mon(target).types.contains(&dex.effects.grass) {
                continue;
            }
            // `hitStepTryImmunity` precedes accuracy: Sticky Hold refuses
            // Trick/Switcheroo and the target is not affected at all.
            if behavior == MoveBehavior::Trick
                && self.mon(target).ability == dex.effects.sticky_hold
            {
                continue;
            }
            if let Some(effectiveness) = effectiveness {
                hit.push((target, effectiveness));
            }
        }
        // Accuracy checks for the complete spread precede all damage draws.
        let ohko = m.ohko;
        let ignore_evasion = m.ignore_evasion;
        let ice_type = dex.effects.ice;
        let toxic_never_misses = m.hit.status == dex.effects.toxic
            && self.mon(actor).types.contains(&dex.effects.poison_type);
        let actor_accuracy_boost = self.mon(actor).boosts[5];
        let actor_unaware =
            dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Unaware;
        let actor_level = self.mon(actor).level;
        let actor_is_ice = self.mon(actor).types.contains(&ice_type);
        hit.retain(|(target, _)| {
            let target_level = self.mon(*target).level;
            let defender_unaware =
                dex.effects.abilities[self.mon(*target).ability as usize] == Ability::Unaware;
            let target_evasion = if ignore_evasion || actor_unaware {
                0
            } else {
                self.mon(*target).boosts[6]
            };
            let target_types = self.mon(*target).types.clone();
            if let Some(ohko_type) = ohko {
                // OHKO accuracy bypasses every accuracy modifier.
                let mut accuracy =
                    if ohko_type != 0 && ohko_type == ice_type && !actor_is_ice {
                        20
                    } else {
                        30
                    };
                if actor_level >= target_level
                    && (ohko_type == 0 || !target_types.contains(&ohko_type))
                {
                    accuracy += u32::from(actor_level - target_level);
                } else {
                    return false;
                }
                return self.rng.below(100) < accuracy;
            }
            let Some(accuracy) = self.modify_accuracy(dex, actor, *target, action_accuracy) else {
                return true;
            };
            let attacker_accuracy = if defender_unaware {
                0
            } else {
                actor_accuracy_boost
            };
            let boost = (attacker_accuracy - target_evasion).clamp(-6, 6);
            let accuracy = if boost > 0 {
                u32::from(accuracy) * (3 + boost as u32) / 3
            } else {
                u32::from(accuracy) * 3 / (3 + (-boost) as u32)
            };
            if toxic_never_misses {
                return true;
            }
            self.rng.below(100) < accuracy
        });
        if hit.is_empty() {
            return Ok(());
        }
        // Reference hitStepBreakProtect runs after the accuracy step and
        // before any damage: a protecting target loses its protection (and the
        // reused stall counter) and the move continues normally.
        if m.breaks_protect {
            for &(target, _) in &hit {
                if self
                    .mon_mut(target)
                    .volatiles
                    .remove(&dex.effects.protect)
                    .is_some()
                {
                    self.mon_mut(target).volatiles.remove(&dex.effects.stall);
                    self.emit(
                        EventKind::EffectEnd,
                        target,
                        None,
                        EffectRef::Condition(dex.effects.protect),
                        0,
                        false,
                    )?;
                }
            }
        }
        if behavior == MoveBehavior::ScreenBreak {
            for &(target, _) in &hit {
                for id in [
                    dex.effects.reflect,
                    dex.effects.light_screen,
                    dex.effects.aurora_veil,
                ] {
                    if self.sides[target.side as usize]
                        .conditions
                        .remove(&id)
                        .is_some()
                    {
                        self.emit(
                            EventKind::SideEffectEnd,
                            target,
                            None,
                            EffectRef::Condition(id),
                            0,
                            false,
                        )?;
                    }
                }
            }
        }
        let hit_targets: SmallVec<[Entity; 4]> = hit.iter().map(|(t, _)| *t).collect();
        let mut damages = SmallVec::<[(Entity, u16); 4]>::new();
        for (target, effectiveness) in hit {
            if m.category == Category::Status {
                continue;
            }
            // OHKO and fixed-damage moves resolve before the damage kernel and
            // therefore consume no critical-hit or damage randomizer draws.
            if let Some(amount) = self.fixed_damage_amount(dex, m, actor, target) {
                // Final Gambit's callback faints the user while resolving the
                // damage amount, before the damage is applied.
                if m.fixed_damage == Some(crate::assets::FixedDamage::UserHp) {
                    self.faint_now(actor);
                }
                damages.push((target, amount.min(u32::from(u16::MAX)) as u16));
                continue;
            }
            // `willCrit` skips the crit draw entirely; armor still cancels it.
            let crit_ratio = self.crit_ratio(
            dex,
            actor,
            m.crit_ratio + item_ports::crit_ratio_bonus(self, dex, actor),
        );
            let rolled_crit = if m.will_crit {
                true
            } else {
                damage::critical_hit(crit_ratio, None, &mut self.rng)
            };
            let critical = rolled_crit
                && dex.effects.abilities[self.mon(target).ability as usize] != Ability::Armor;
            let physical = m.category == Category::Physical;
            // `overrideOffensiveStat`/`overrideDefensiveStat` replace the
            // category default; `overrideOffensivePokemon: 'target'` reads the
            // defender's Attack stat and boosts.
            let ai = m
                .override_offensive_stat
                .map_or(if physical { 1 } else { 3 }, usize::from);
            let di = m
                .override_defensive_stat
                .map_or(if physical { 2 } else { 4 }, usize::from);
            let attacker = if m.override_offensive_target {
                target
            } else {
                actor
            };
            // `abilities:unaware.onAnyModifyBoost`: an Unaware defender zeroes
            // the attacker's offensive stage; an Unaware attacker zeroes the
            // defender's defensive stage (and evasion).
            let defender_unaware =
                dex.effects.abilities[self.mon(target).ability as usize] == Ability::Unaware;
            let attacker_unaware =
                dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Unaware;
            let ab = if defender_unaware {
                0
            } else {
                self.mon(attacker).boosts[ai - 1]
            };
            let db = if m.ignore_defensive || attacker_unaware {
                0
            } else {
                self.mon(target).boosts[di - 1]
            };
            let mut attack = stats::apply_stage(
                u32::from(self.mon(attacker).stats[ai]),
                if critical { ab.max(0) } else { ab },
            );
            let mut defense = stats::apply_stage(
                u32::from(self.mon(target).stats[di]),
                if critical { db.min(0) } else { db },
            );
            if (weather == dex.effects.sand
                && di == 4
                && self.mon(target).types.contains(&dex.effects.rock))
                || (weather == dex.effects.snow
                    && di == 2
                    && self.mon(target).types.contains(&dex.effects.ice))
            {
                defense = stats::modify(defense, 6144);
            }
            let context = MoveContext {
                actor,
                target,
                move_data: m,
                effectiveness,
                critical,
            };
            // A callback base power of exactly zero means the reference returns
            // `undefined`: no damage is dealt and the damage stages are skipped.
            let base_power = self.base_power(dex, m.bp_callback, u32::from(m.power), actor, target);
            if base_power == 0 {
                continue;
            }
            let power_den = self.base_power_den(m.bp_callback, actor);
            // A fractional callback power is truncated by the first
            // participating BasePower handler (reference `modify(value, mod)`),
            // and otherwise flows into the damage formula as an exact rational.
            let (power, power_den) = if power_den > 1 {
                let (modifier, participated) = self.modifiers_with_participation(
                    dex,
                    ModifierEvent::BasePower,
                    context,
                    base_power,
                )?;
                if participated {
                    let scaled = (u64::from(base_power) * u64::from(modifier))
                        / (u64::from(power_den) * 4096);
                    (scaled as u32, 1)
                } else {
                    (base_power, power_den)
                }
            } else {
                (
                    self.modify_value(dex, ModifierEvent::BasePower, context, base_power)?,
                    1,
                )
            };
            // The reference re-derives the modifier event from the move's
            // category, so Body Press still runs ModifyAtk handlers.
            attack = self.modify_value(
                dex,
                if physical {
                    ModifierEvent::Attack
                } else {
                    ModifierEvent::SpecialAttack
                },
                context,
                attack,
            )?;
            defense = self.modify_value(
                dex,
                if physical {
                    ModifierEvent::Defense
                } else {
                    ModifierEvent::SpecialDefense
                },
                context,
                defense,
            )?;
            let a = self.mon(actor);
            let ability = dex.effects.abilities[a.ability as usize];
            let damage = damage::calculate_before_final(
                DamageInput {
                    level: a.level,
                    power,
                    power_den,
                    attack,
                    defense,
                    spread,
                    parental_bond_second_hit: false,
                    weather_modifier: self.weather_damage_modifier(dex, m.move_type),
                    critical,
                    stab_modifier: if behavior != MoveBehavior::Struggle
                        && a.types.contains(&m.move_type)
                    {
                        if ability == Ability::Adaptability {
                            8192
                        } else {
                            6144
                        }
                    } else {
                        4096
                    },
                    effectiveness,
                    // `modifyDamage` skips the burn halving for a Guts holder.
                    burn: a.status == dex.effects.burn
                        && m.category == Category::Physical
                        && ability != Ability::Guts,
                    final_modifier: 4096,
                    bypass_protect: false,
                },
                &mut self.rng,
            )?;
            let final_modifier = self.damage_modifier(dex, context)?;
            let damage = damage::finish_damage(damage, final_modifier, false);
            damages.push((target, damage));
        }
        let mut total_damage = 0u32;
        let mut hit_any = false;
        for (target, damage) in damages {
            // `endure` clamps after item/berry damage modification and before
            // the damage is applied (reference `onDamage` priority -10).
            let damage = self.sturdy_clamp(dex, target, damage)?;
            let damage = self.damage_item(dex, target, damage)?;
            let damage = self.endure_clamp(dex, target, damage);
            let actual = damage.min(self.mon(target).hp);
            total_damage += u32::from(actual);
            hit_any = true;
            self.mon_mut(target).hp -= actual;
            if self.mon(target).hp == 0 {
                self.faint_queue.push(target);
            }
            self.emit(
                EventKind::Damage,
                target,
                Some(actor),
                EffectRef::Move(move_id),
                -i32::from(actual),
                true,
            )?;
            if actual != 0
                && let Some(fraction) = m.drain
            {
                self.drain_heal(
                    dex,
                    actor,
                    target,
                    stats::round_fraction(u32::from(actual), fraction),
                )?;
            }
        }
        let mut did_anything = m.category != Category::Status;
        for &target in &hit_targets {
            if behavior == MoveBehavior::Trick {
                // Trick/Switcheroo decide success themselves: the empty generic
                // payload must not mark a refused swap as "did anything", or
                // the failed move would run the reference's post-move phases.
                did_anything |= self.trick_swap(dex, actor, target)?;
            } else if m.force_switch {
                // Reference `runMoveEffects`: a force-switch move's only
                // contribution to `didAnything` is
                // `!!this.battle.canSwitch(target.side)`. The empty payload
                // must not count as a successful effect, or a phazing move
                // against a side with no reserve would run the success-path
                // move-loop Updates and desynchronize the RNG (roar against
                // an empty bench).
                did_anything |= self.can_switch(target.side as usize);
            } else {
                did_anything |= self.hit_effect(dex, target, actor, &m.hit, false)?;
            }
            // `moves:partingshot.onHit` applies the Attack/Sp. Atk drop itself
            // (the pinned declaration has no `boosts` field) and deletes its
            // own `selfSwitch` when nothing changed.
            if hooks & crate::effects::hook::PARTING_SHOT != 0 {
                did_anything |= self.boost(
                    dex,
                    target,
                    actor,
                    // Index order is [atk, def, spa, spd, spe, accuracy,
                    // evasion]; Parting Shot drops Attack and Sp. Atk.
                    [-1, 0, -1, 0, 0, 0, 0],
                    BoostCause::Move { secondary: false },
                )?;
            }
        }
        // Reference `spreadMoveHit` sets `source.switchFlag` once the move has
        // resolved against at least one target, `didAnything` is truthy (or a
        // numeric damage result, including zero), the user is still alive and a
        // reserve exists. Parting Shot deletes its own `selfSwitch` when the
        // Attack/Sp. Atk drop fails.
        let pivot = m.self_switch == crate::assets::SelfSwitch::Switch
            && self.mon(actor).hp > 0
            && !hit_targets.is_empty()
            && self.can_switch(actor.side as usize)
            && (hooks & crate::effects::hook::PARTING_SHOT == 0 || did_anything);
        if pivot {
            self.mon_mut(actor).switch_flag = Some(move_id);
        }
        // `selfdestruct: 'ifHit'` faints the user inside the reference's
        // per-target effect phase, after the target's boosts/status resolve and
        // whenever the move connected with at least one target (even a status
        // move such as Memento).
        let connected = hit_any || !hit_targets.is_empty();
        if connected && m.self_destruct == crate::assets::SelfDestructMode::IfHit {
            self.faint_now(actor);
        }
        let self_destruct = m.self_destruct != crate::assets::SelfDestructMode::None;
        if !did_anything && !self_destruct {
            return Ok(());
        }
        // Sheer Force deletes the action's self effect and every secondary, so
        // those draws and effects never happen for the marked action.
        if let Some(effect) = m.self_effect.as_ref().filter(|_| !m.sheer_force) {
            // Reference `selfDrops`: the roll only happens for a boosting self
            // effect that is not a secondary. A pure volatile self effect such
            // as `mustrecharge` runs `moveHit` directly and draws nothing.
            if effect.boosts.iter().any(|b| *b != 0) {
                self.rng.below(100);
            }
            self.hit_effect(dex, actor, actor, effect, false)?;
        }
        for &target in &hit_targets {
            for secondary in m.secondaries.iter().filter(|_| !m.sheer_force) {
                if self.rng.below(100) < u32::from(secondary.chance) {
                    self.hit_effect(dex, target, actor, &secondary.target, true)?;
                    if hooks & crate::effects::hook::DIRE_CLAW != 0 {
                        self.dire_claw_secondary(dex, actor, target)?;
                    }
                    if hooks & crate::effects::hook::THROAT_CHOP != 0 {
                        self.throat_chop_secondary(dex, actor, target)?;
                    }
                    if let Some(effect) = &secondary.own {
                        self.hit_effect(dex, actor, actor, effect, true)?;
                    }
                }
            }
        }
        // Reference `forceSwitch`: the phazing step runs after self drops and
        // secondaries, marks every surviving in-range target, and lets the
        // post-action block drag a random reserve in. `DragOut` handlers that
        // refuse the drag (Suction Cups, Guard Dog, Ingrain) keep the target in
        // place; those effects stay operational errors until ported.
        if m.force_switch {
            for &target in &hit_targets {
                if self.mon(target).hp == 0
                    || self.mon(actor).hp == 0
                    || !self.can_switch(target.side as usize)
                {
                    continue;
                }
                self.mon_mut(target).force_switch_flag = true;
            }
        }
        // `selfBoost` is applied as a self-targeted hit after the full hit
        // sequence and only when the move connected with at least one target.
        // It draws no RNG: the reference only rolls for `move.self` drops.
        if let Some(effect) = &m.self_boost {
            self.hit_effect(dex, actor, actor, effect, false)?;
        }
        self.damaging_hit(dex, actor, &hit_targets, m)?;
        // `moves:knockoff.onAfterHit`: after the DamagingHit event, an alive
        // user removes the item of every target the move damaged.
        if m.hooks & crate::effects::hook::KNOCK_OFF != 0 && self.mon(actor).hp > 0 {
            for &target in &hit_targets {
                self.take_item(dex, target, actor)?;
            }
        }
        self.each_update(dex)?;
        self.process_faints(dex, self.mon(actor).hp == 0)?;
        self.each_update(dex)?;
        if total_damage > 0 {
            if behavior == MoveBehavior::Struggle {
                let recoil =
                    stats::round_fraction(u32::from(self.mon(actor).stats[0]), [1, 4]).max(1);
                self.indirect_damage(dex, actor, actor, recoil, EffectRef::Move(move_id))?;
            } else if let Some(fraction) = m.recoil
                && dex.effects.abilities[self.mon(actor).ability as usize] != Ability::RockHead
            {
                let recoil = stats::round_fraction(total_damage, fraction).max(1);
                self.indirect_damage(
                    dex,
                    actor,
                    actor,
                    recoil,
                    EffectRef::Condition(dex.effects.recoil),
                )?;
            }
        }
        if m.thaws_target {
            for target in hit_targets {
                if self.mon(target).status == dex.effects.freeze {
                    self.cure_status(target)?;
                }
            }
        }
        if m.category != Category::Status
            && dex.effects.items[self.mon(actor).item as usize] == Item::LifeOrb
        {
            self.item_damage(dex, actor, actor, self.mon(actor).stats[0] / 10)?;
        }
        self.process_faints(dex, true)?;
        self.check_win(None);
        Ok(())
    }

    /// Reference `hitStepMoveHitLoop` for multi-hit moves. Every hit repeats the
    /// full per-target phase (TryHit, immunity, accuracy, crit, damage, primary
    /// and secondary effects, DamagingHit) and consumes an `Update` event; a
    /// blocked, immune or missed hit ends the remaining hits exactly as
    /// `moveDamage.some(val => val !== false)` does in the reference.
    fn use_multihit_move(
        &mut self,
        dex: &Dex,
        actor: Entity,
        move_id: Id,
        m: &ActiveMove<'_>,
        targets: SmallVec<[Entity; 4]>,
    ) -> Result<()> {
        let mut action_accuracy = m.accuracy.map(u16::from);
        let spread = targets.len() > 1;
        let effective_priority = self.effective_priority(dex, actor, move_id);
        let mut blocked = SmallVec::<[(Entity, Id); 4]>::new();
        let mut kept = SmallVec::<[Entity; 4]>::new();
        for e in targets {
            self.validate_effects(dex, e)?;
            if m.protect && !m.breaks_protect {
                if self.guard_blocks(dex, e, m, effective_priority) {
                    continue;
                }
                if let Some(volatile) = self.blocking_protection(dex, e) {
                    blocked.push((e, volatile));
                    continue;
                }
            }
            kept.push(e);
        }
        for (target, volatile) in blocked {
            self.protect_punish(dex, target, actor, m, move_id, volatile)?;
        }
        let targets = kept;
        if targets.is_empty() {
            return Ok(());
        }
        // `hitStepTryHitEvent`, type immunity and accuracy run once for the
        // whole action, before `hitStepMoveHitLoop` samples the hit count.
        let mut connected = SmallVec::<[(Entity, i8); 4]>::new();
        for &target in &targets {
            self.validate_effects(dex, target)?;
            let ability = dex.effects.abilities[self.mon(target).ability as usize];
            if self.terrain_id(dex) == dex.effects.psychic_terrain
                && m.priority > 0
                && target.side != actor.side
                && self.grounded(dex, target)
            {
                return Ok(());
            }
            if m.powder
                && target != actor
                && ability == Ability::Overcoat
                && !self.mon(target).types.contains(&dex.effects.grass)
            {
                self.reveal_ability(target)?;
                return Ok(());
            }
            // TryHit (absorption) precedes type immunity and accuracy.
            if self.absorb_try_hit(dex, target, actor, m, &mut action_accuracy)? {
                return Ok(());
            }
            let effectiveness = if m.ignore_immunity {
                Some(0)
            } else if m.move_type == dex.effects.ground && ability == Ability::Levitate {
                self.emit(
                    EventKind::Ability,
                    target,
                    None,
                    EffectRef::Ability(self.mon(target).ability),
                    0,
                    false,
                )?;
                None
            } else {
                self.mon(target)
                    .types
                    .iter()
                    .try_fold(0i8, |total, &kind| {
                        let value = dex.type_chart[m.move_type as usize][kind as usize];
                        if value == -127 {
                            None
                        } else {
                            Some(total + value)
                        }
                    })
                    .map(|total| total.clamp(-6, 6))
            };
            let Some(effectiveness) = effectiveness else {
                return Ok(());
            };
            if !self.roll_move_accuracy(dex, actor, target, m, action_accuracy) {
                return Ok(());
            }
            connected.push((target, effectiveness));
        }
        if connected.is_empty() {
            return Ok(());
        }
        let hit_count = self.multihit_count(m);
        let mut total_damage = 0u32;
        for hit in 1..=hit_count {
            if hit > 1
                && (self.mon(actor).hp == 0
                    || connected.iter().all(|(t, _)| self.mon(*t).hp == 0))
            {
                break;
            }
            for &(target, effectiveness) in &connected {
                if self.mon(target).hp == 0 {
                    continue;
                }
                let damage = self.resolve_hit_damage(dex, actor, target, m, effectiveness, spread)?;
                let damage = self.sturdy_clamp(dex, target, damage)?;
                let damage = self.damage_item(dex, target, damage)?;
                let damage = self.endure_clamp(dex, target, damage);
                let actual = damage.min(self.mon(target).hp);
                total_damage += u32::from(actual);
                self.mon_mut(target).hp -= actual;
                if self.mon(target).hp == 0 {
                    self.faint_queue.push(target);
                }
                self.emit(
                    EventKind::Damage,
                    target,
                    Some(actor),
                    EffectRef::Move(move_id),
                    -i32::from(actual),
                    true,
                )?;
                if actual != 0
                    && let Some(fraction) = m.drain
                {
                    self.drain_heal(
                        dex,
                        actor,
                        target,
                        stats::round_fraction(u32::from(actual), fraction),
                    )?;
                }
                self.hit_effect(dex, target, actor, &m.hit, false)?;
                if hit == 1
                    && let Some(effect) = m.self_effect.as_ref().filter(|_| !m.sheer_force)
                {
                    self.rng.below(100);
                    self.hit_effect(dex, actor, actor, effect, false)?;
                }
                for secondary in m.secondaries.iter().filter(|_| !m.sheer_force) {
                    if self.rng.below(100) < u32::from(secondary.chance) {
                        self.hit_effect(dex, target, actor, &secondary.target, true)?;
                        if m.hooks & crate::effects::hook::DIRE_CLAW != 0 {
                            self.dire_claw_secondary(dex, actor, target)?;
                        }
                        if m.hooks & crate::effects::hook::THROAT_CHOP != 0 {
                            self.throat_chop_secondary(dex, actor, target)?;
                        }
                        if let Some(effect) = &secondary.own {
                            self.hit_effect(dex, actor, actor, effect, true)?;
                        }
                    }
                }
                self.damaging_hit(dex, actor, std::slice::from_ref(&target), m)?;
            }
            self.each_update(dex)?;
        }
        self.process_faints(dex, self.mon(actor).hp == 0)?;
        if self.outcome.terminated {
            return Ok(());
        }
        // `selfBoost` applies once after the whole hit sequence (reference
        // `useMoveInner`) and only when the move connected with a target.
        if total_damage > 0
            && let Some(effect) = &m.self_boost
        {
            self.hit_effect(dex, actor, actor, effect, false)?;
        }
        if total_damage > 0
            && let Some(fraction) = m.recoil
            && dex.effects.abilities[self.mon(actor).ability as usize] != Ability::RockHead
        {
            let recoil = stats::round_fraction(total_damage, fraction).max(1);
            self.indirect_damage(
                dex,
                actor,
                actor,
                recoil,
                EffectRef::Condition(dex.effects.recoil),
            )?;
        }
        self.each_update(dex)?;
        if m.thaws_target {
            for target in targets {
                if self.mon(target).status == dex.effects.freeze {
                    self.cure_status(target)?;
                }
            }
        }
        if dex.effects.items[self.mon(actor).item as usize] == Item::LifeOrb {
            self.item_damage(dex, actor, actor, self.mon(actor).stats[0] / 10)?;
        }
        self.process_faints(dex, true)?;
        self.check_win(None);
        Ok(())
    }

    /// Reference `battle.sample([...])`/`battle.random(a, b)` hit-count draws.
    fn multihit_count(&mut self, m: &crate::assets::Move) -> u32 {
        match m.multihit {
            None | Some([0, 0]) => 1,
            Some([2, 5]) => {
                const TABLE: [u32; 20] = [
                    2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                ];
                TABLE[self.rng.below(TABLE.len() as u32) as usize]
            }
            // A fixed numeric `multihit` consumes no draw in the reference.
            Some([low, high]) if low == high => u32::from(low),
            Some([low, high]) => {
                u32::from(low) + self.rng.below(u32::from(high) - u32::from(low) + 1)
            }
        }
    }

    /// One accuracy roll for a single target, sharing the action-local
    /// accuracy sentinel that TryHit handlers may force to always-hit.
    fn roll_move_accuracy(
        &mut self,
        dex: &Dex,
        actor: Entity,
        target: Entity,
        m: &ActiveMove<'_>,
        action_accuracy: Option<u16>,
    ) -> bool {
        let Some(accuracy) = self.modify_accuracy(dex, actor, target, action_accuracy) else {
            return true;
        };
        let attacker_unaware =
            dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Unaware;
        let defender_unaware =
            dex.effects.abilities[self.mon(target).ability as usize] == Ability::Unaware;
        let evasion = if m.ignore_evasion || attacker_unaware {
            0
        } else {
            self.mon(target).boosts[6]
        };
        let attacker_accuracy = if defender_unaware {
            0
        } else {
            self.mon(actor).boosts[5]
        };
        let boost = (attacker_accuracy - evasion).clamp(-6, 6);
        let accuracy = if boost > 0 {
            u32::from(accuracy) * (3 + boost as u32) / 3
        } else {
            u32::from(accuracy) * 3 / (3 + (-boost) as u32)
        };
        if m.hit.status == dex.effects.toxic
            && self.mon(actor).types.contains(&dex.effects.poison_type)
        {
            return true;
        }
        self.rng.below(100) < accuracy
    }

    /// Damage for one resolved hit, mirroring the single-target damage path.
    /// Reference `moves:trick.onHit` / `switcheroo.onHit`: both items are taken
    /// and swapped only when neither take is refused and at least one item
    /// exists; otherwise both items are restored and the move fails.
    fn trick_swap(&mut self, dex: &Dex, actor: Entity, target: Entity) -> Result<bool> {
        use crate::battle::hooks::TakeOutcome;
        let yours = self.take_item_checked(dex, target)?;
        let mine = self.take_item_checked(dex, actor)?;
        if matches!(yours, TakeOutcome::Refused)
            || matches!(mine, TakeOutcome::Refused)
            || (yours == TakeOutcome::Empty && mine == TakeOutcome::Empty)
        {
            if let TakeOutcome::Taken(item) = yours {
                self.restore_item(target, item)?;
            }
            if let TakeOutcome::Taken(item) = mine {
                self.restore_item(actor, item)?;
            }
            return Ok(false);
        }
        // The reference logs each hand-off: `-item` for a new holder and a
        // silent `-enditem` for the emptied side. `lastItem` is untouched.
        match mine {
            TakeOutcome::Taken(item) => self.give_item(dex, target, actor, item)?,
            TakeOutcome::Empty => {
                if let TakeOutcome::Taken(item) = yours {
                    self.emit(
                        EventKind::EndItem,
                        target,
                        Some(actor),
                        EffectRef::Item(item),
                        0,
                        false,
                    )?;
                }
            }
            TakeOutcome::Refused => unreachable!("refused swaps restore before logging"),
        }
        match yours {
            TakeOutcome::Taken(item) => self.give_item(dex, actor, target, item)?,
            TakeOutcome::Empty => {
                if let TakeOutcome::Taken(item) = mine {
                    self.emit(
                        EventKind::EndItem,
                        actor,
                        Some(target),
                        EffectRef::Item(item),
                        0,
                        false,
                    )?;
                }
            }
            TakeOutcome::Refused => unreachable!("refused swaps restore before logging"),
        }
        Ok(true)
    }

    /// Reference `Pokemon#setItem` when an item is handed over: register the
    /// item, announce it publicly and run its Start event. Only `useItem` and
    /// friends record `lastItem`, so the give leaves it untouched.
    fn give_item(&mut self, dex: &Dex, recipient: Entity, source: Entity, item: Id) -> Result<()> {
        let order = self.allocate_effect_order()?;
        self.mon_mut(recipient).item = item;
        self.mon_mut(recipient).item_effect_order = Some(order);
        self.emit(
            EventKind::Item,
            recipient,
            Some(source),
            EffectRef::Item(item),
            0,
            false,
        )?;
        item_ports::start(self, dex, recipient)
    }

    /// Damage for one resolved hit, mirroring the single-target damage path.
    fn resolve_hit_damage(
        &mut self,
        dex: &Dex,
        actor: Entity,
        target: Entity,
        m: &ActiveMove<'_>,
        effectiveness: i8,
        spread: bool,
    ) -> Result<u16> {
        let crit_ratio = self.crit_ratio(
            dex,
            actor,
            m.crit_ratio + item_ports::crit_ratio_bonus(self, dex, actor),
        );
        let rolled_crit = if m.will_crit {
            true
        } else {
            damage::critical_hit(crit_ratio, None, &mut self.rng)
        };
        let critical = rolled_crit
            && dex.effects.abilities[self.mon(target).ability as usize] != Ability::Armor;
        let physical = m.category == Category::Physical;
        let ai = m
            .override_offensive_stat
            .map_or(if physical { 1 } else { 3 }, usize::from);
        let di = m
            .override_defensive_stat
            .map_or(if physical { 2 } else { 4 }, usize::from);
        let attacker = if m.override_offensive_target {
            target
        } else {
            actor
        };
        // `abilities:unaware.onAnyModifyBoost` (see the multihit path).
        let defender_unaware =
            dex.effects.abilities[self.mon(target).ability as usize] == Ability::Unaware;
        let attacker_unaware =
            dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Unaware;
        let ab = if defender_unaware {
            0
        } else {
            self.mon(attacker).boosts[ai - 1]
        };
        let db = if m.ignore_defensive || attacker_unaware {
            0
        } else {
            self.mon(target).boosts[di - 1]
        };
        let mut attack = stats::apply_stage(
            u32::from(self.mon(attacker).stats[ai]),
            if critical { ab.max(0) } else { ab },
        );
        let mut defense = stats::apply_stage(
            u32::from(self.mon(target).stats[di]),
            if critical { db.min(0) } else { db },
        );
        let weather = self.effective_weather(dex);
        if (weather == dex.effects.sand
            && di == 4
            && self.mon(target).types.contains(&dex.effects.rock))
            || (weather == dex.effects.snow
                && di == 2
                && self.mon(target).types.contains(&dex.effects.ice))
        {
            defense = stats::modify(defense, 6144);
        }
        let context = MoveContext {
            actor,
            target,
            move_data: m,
            effectiveness,
            critical,
        };
        let base_power = self.base_power(dex, m.bp_callback, u32::from(m.power), actor, target);
        if base_power == 0 {
            return Ok(0);
        }
        let power = self.modify_value(dex, ModifierEvent::BasePower, context, base_power)?;
        attack = self.modify_value(
            dex,
            if physical {
                ModifierEvent::Attack
            } else {
                ModifierEvent::SpecialAttack
            },
            context,
            attack,
        )?;
        defense = self.modify_value(
            dex,
            if physical {
                ModifierEvent::Defense
            } else {
                ModifierEvent::SpecialDefense
            },
            context,
            defense,
        )?;
        let a = self.mon(actor);
        let ability = dex.effects.abilities[a.ability as usize];
        let damage = damage::calculate_before_final(
            DamageInput {
                level: a.level,
                power,
                power_den: 1,
                attack,
                defense,
                spread,
                parental_bond_second_hit: false,
                weather_modifier: self.weather_damage_modifier(dex, m.move_type),
                critical,
                stab_modifier: if a.types.contains(&m.move_type) {
                    if ability == Ability::Adaptability {
                        8192
                    } else {
                        6144
                    }
                } else {
                    4096
                },
                effectiveness,
                // `modifyDamage` skips the burn halving for a Guts holder.
                burn: a.status == dex.effects.burn
                    && m.category == Category::Physical
                    && ability != Ability::Guts,
                final_modifier: 4096,
                bypass_protect: false,
            },
            &mut self.rng,
        )?;
        let final_modifier = self.damage_modifier(dex, context)?;
        Ok(damage::finish_damage(damage, final_modifier, false))
    }

    fn before_move(&mut self, dex: &Dex, e: Entity, m: &crate::assets::Move) -> Result<bool> {
        // Reference BeforeMove ordering by handler priority:
        // mustrecharge (11) > sleep/freeze (10) > flinch (8) > confusion (3) >
        // paralysis (1). A cancel before a later handler also suppresses that
        // handler's RNG draw.
        if self
            .mon(e)
            .volatiles
            .contains_key(&dex.effects.must_recharge)
        {
            self.mon_mut(e).volatiles.remove(&dex.effects.must_recharge);
            self.emit(
                EventKind::EffectEnd,
                e,
                None,
                EffectRef::Condition(dex.effects.must_recharge),
                0,
                false,
            )?;
            return Ok(false);
        }
        let status = self.mon(e).status;
        if status == dex.effects.sleep || (status == dex.effects.freeze && !m.defrost) {
            self.mon_mut(e).status_state.values[0] -= 1;
            let expired = self.mon(e).status_state.values[0] <= 0;
            if expired || (status == dex.effects.freeze && self.rng.chance(1, 4)) {
                self.cure_status(e)?;
            } else {
                return Ok(false);
            }
        }
        if self.mon(e).volatiles.contains_key(&dex.effects.flinch) {
            return Ok(false);
        }
        // `moves:throatchop.condition.onBeforeMove` (priority 6, between
        // flinch and confusion): a sound move is refused outright. The
        // condition's `onModifyMove` guard is the same rule one phase later
        // and is unreachable once BeforeMove has already refused the move.
        if m.sound
            && self
                .mon(e)
                .volatiles
                .contains_key(&dex.effects.throat_chop)
        {
            return Ok(false);
        }
        if self.mon(e).volatiles.contains_key(&dex.effects.confusion) {
            let expired = {
                let state = self.mon_mut(e).volatiles.get_mut(&dex.effects.confusion);
                let Some(state) = state else { unreachable!() };
                let time = state.values.first_mut().unwrap_or_else(|| unreachable!());
                *time -= 1;
                *time == 0
            };
            if expired {
                self.mon_mut(e).volatiles.remove(&dex.effects.confusion);
                self.emit(
                    EventKind::EffectEnd,
                    e,
                    None,
                    EffectRef::Condition(dex.effects.confusion),
                    0,
                    false,
                )?;
            } else {
                // Reference `randomChance(33, 100)`: a hit on 33% of turns.
                if self.rng.chance(33, 100) {
                    self.confusion_self_hit(dex, e)?;
                    return Ok(false);
                }
            }
        }
        if status == dex.effects.paralysis && self.rng.chance(1, 8) {
            return Ok(false);
        }
        Ok(true)
    }

    /// Reference `getConfusionDamage(pokemon, 40)`: a typeless physical
    /// self-hit that uses the holder's own boosted stats, no STAB, no
    /// effectiveness and no ability or item stat modifiers.
    fn confusion_self_hit(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        let mon = self.mon(e);
        let attack = stats::apply_stage(u32::from(mon.stats[1]), mon.boosts[0]);
        let defense = stats::apply_stage(u32::from(mon.stats[2]), mon.boosts[1]);
        let raw = 22u32 * 40 * attack / defense / 50 + 2;
        let truncated = raw & 0xFFFF;
        let rolled = stats::random_damage(truncated, &mut self.rng);
        let amount = self.endure_clamp(dex, e, rolled.max(1) as u16) as u32;
        let actual = amount.min(u32::from(self.mon(e).hp));
        if actual == 0 {
            return Ok(());
        }
        self.mon_mut(e).hp -= actual as u16;
        if self.mon(e).hp == 0 {
            self.faint_queue.push(e);
        }
        self.emit(
            EventKind::Damage,
            e,
            Some(e),
            EffectRef::Condition(dex.effects.confusion),
            -(actual as i32),
            true,
        )?;
        Ok(())
    }

    /// Reference `stall.onRestart`/`onStart`: a fresh cast starts the counter at
    /// three, a restart multiplies the retained counter by three up to 729.
    /// The retained effect order is preserved; only a new state allocates one.
    fn add_stall(&mut self, dex: &Dex, actor: Entity, counter: Option<i64>) -> Result<()> {
        let order = if let Some(state) = self.mon(actor).volatiles.get(&dex.effects.stall) {
            state.effect_order
        } else {
            self.allocate_effect_order()?
        };
        self.mon_mut(actor).volatiles.insert(
            dex.effects.stall,
            EffectState {
                id: dex.effects.stall,
                effect_order: order,
                effect_order_assigned: true,
                duration: Some(2),
                values: vec![(counter.unwrap_or(1).saturating_mul(3)).min(729)],
                ..Default::default()
            },
        );
        Ok(())
    }

    /// Duration-one side protection. `Side.addSideCondition` fails when the
    /// condition already exists and has no `onSideRestart`, so no new effect
    /// order is allocated on a repeated cast.
    fn start_guard(&mut self, dex: &Dex, actor: Entity, condition: Id) -> Result<bool> {
        let _ = dex;
        let side = actor.side as usize;
        if self.sides[side].conditions.contains_key(&condition) {
            return Ok(false);
        }
        let order = self.allocate_effect_order()?;
        self.sides[side].conditions.insert(
            condition,
            EffectState {
                id: condition,
                effect_order: order,
                effect_order_assigned: true,
                duration: Some(1),
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
            EffectRef::Condition(condition),
            1,
            false,
        )?;
        Ok(true)
    }

    /// The protection volatile currently shielding this entity, if any.
    fn blocking_protection(&self, dex: &Dex, e: Entity) -> Option<Id> {
        let volatiles = &self.mon(e).volatiles;
        dex.effects
            .protection_volatiles()
            .into_iter()
            .find(|id| volatiles.contains_key(id))
    }

    /// Reference priority-4 side-condition guards (Wide Guard, Quick Guard).
    fn guard_blocks(
        &self,
        dex: &Dex,
        target: Entity,
        m: &ActiveMove<'_>,
        effective_priority: i8,
    ) -> bool {
        let conditions = &self.sides[target.side as usize].conditions;
        if conditions.contains_key(&dex.effects.wide_guard)
            && matches!(m.target, Target::AllAdjacent | Target::AllAdjacentFoes)
        {
            return true;
        }
        conditions.contains_key(&dex.effects.quick_guard) && effective_priority > 0
    }

    /// Contact punishment of the volatile that actually blocked the hit.
    fn protect_punish(
        &mut self,
        dex: &Dex,
        target: Entity,
        actor: Entity,
        m: &ActiveMove<'_>,
        move_id: Id,
        volatile: Id,
    ) -> Result<()> {
        if self.mon(actor).hp == 0 || !m.contact {
            return Ok(());
        }
        match dex.effects.protect_punish(volatile) {
            crate::effects::ProtectPunish::None => Ok(()),
            crate::effects::ProtectPunish::DamageEighthMaxHp => {
                let amount = u32::from(self.mon(actor).stats[0]) / 8;
                self.indirect_damage(dex, actor, target, amount, EffectRef::Move(move_id))
            }
            crate::effects::ProtectPunish::Poison => {
                let effect = crate::effects::HitEffect {
                    status: dex.effects.poison,
                    ..Default::default()
                };
                self.hit_effect(dex, actor, target, &effect, false)?;
                Ok(())
            }
            crate::effects::ProtectPunish::AttackDown => {
                self.boost(
                    dex,
                    actor,
                    target,
                    [0, -1, 0, 0, 0, 0, 0],
                    BoostCause::Move { secondary: false },
                )?;
                Ok(())
            }
        }
    }

    /// Reference `endure.condition.onDamage` (priority -10): a Move-sourced hit
    /// that would reach zero HP leaves the holder at exactly one HP.
    fn endure_clamp(&mut self, dex: &Dex, target: Entity, damage: u16) -> u16 {
        let hp = self.mon(target).hp;
        if hp > 0 && damage >= hp && self.mon(target).volatiles.contains_key(&dex.effects.endure)
        {
            return hp - 1;
        }
        damage
    }

    fn cure_status(&mut self, e: Entity) -> Result<()> {
        if self.mon(e).hp == 0 || self.mon(e).status == 0 {
            return Ok(());
        }
        let status = self.mon(e).status;
        self.mon_mut(e).status = 0;
        self.mon_mut(e).status_state = Default::default();
        self.emit(
            EventKind::CureStatus,
            e,
            None,
            EffectRef::Condition(status),
            0,
            false,
        )
    }

    /// Reference `battle.faint` at the point of effect execution: the Pokémon
    /// drops to zero HP immediately and is processed at the next faint
    /// boundary. Re-fainting a zero-HP Pokémon is a no-op.
    fn faint_now(&mut self, e: Entity) {
        if self.mon(e).hp == 0 {
            return;
        }
        self.mon_mut(e).hp = 0;
        self.faint_queue.push(e);
    }

    /// Fixed-damage and OHKO amounts resolved before the damage kernel. OHKO
    /// damage is the target's maximum HP; Fixed Damage follows the exact
    /// reference callback formulas.
    fn fixed_damage_amount(
        &self,
        dex: &Dex,
        m: &crate::assets::Move,
        actor: Entity,
        target: Entity,
    ) -> Option<u32> {
        use crate::assets::FixedDamage;
        if m.ohko.is_some() {
            return Some(u32::from(self.mon(target).stats[0]));
        }
        match m.fixed_damage? {
            FixedDamage::Level => Some(u32::from(self.mon(actor).level)),
            FixedDamage::Flat(amount) => Some(u32::from(amount)),
            FixedDamage::HalfTargetHp => {
                Some((u32::from(self.mon(target).hp) / 2).max(1))
            }
            FixedDamage::Endeavor => Some(
                u32::from(self.mon(target).hp).saturating_sub(u32::from(self.mon(actor).hp)),
            ),
            FixedDamage::UserHp => {
                let _ = dex;
                Some(u32::from(self.mon(actor).hp))
            }
        }
    }

    fn hit_effect(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Entity,
        effect: &crate::effects::HitEffect,
        secondary: bool,
    ) -> Result<bool> {
        self.hit_effect_with_ability(dex, target, source, effect, secondary, None)
    }

    /// `moves:direclaw.secondary.onHit` (Champions 30% secondary): sample one
    /// of poison/paralysis/sleep, then `target.trySetStatus(status, source)`.
    /// The sample is consumed even when the target is fainted or immune, so it
    /// runs before the status application exactly like the reference.
    fn dire_claw_secondary(&mut self, dex: &Dex, source: Entity, target: Entity) -> Result<()> {
        let statuses = [dex.effects.poison, dex.effects.paralysis, dex.effects.sleep];
        let status = statuses[self.rng.below(statuses.len() as u32) as usize];
        let effect = crate::effects::HitEffect {
            status,
            ..Default::default()
        };
        self.hit_effect(dex, target, source, &effect, true)?;
        Ok(())
    }

    /// `moves:throatchop.secondary.onHit`: `target.addVolatile('throatchop')`.
    /// The embedded condition has duration 2 and no `onRestart`, so a repeated
    /// hit leaves the existing timer untouched and emits no second start.
    fn throat_chop_secondary(
        &mut self,
        dex: &Dex,
        source: Entity,
        target: Entity,
    ) -> Result<bool> {
        if self.mon(target).hp == 0
            || self
                .mon(target)
                .volatiles
                .contains_key(&dex.effects.throat_chop)
        {
            return Ok(false);
        }
        let order = self.allocate_effect_order()?;
        self.mon_mut(target).volatiles.insert(
            dex.effects.throat_chop,
            EffectState {
                id: dex.effects.throat_chop,
                duration: Some(2),
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
        self.emit(
            EventKind::EffectStart,
            target,
            Some(source),
            EffectRef::Condition(dex.effects.throat_chop),
            0,
            false,
        )?;
        Ok(true)
    }

    fn hit_effect_with_ability(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Entity,
        effect: &crate::effects::HitEffect,
        secondary: bool,
        ability_source: Option<Entity>,
    ) -> Result<bool> {
        if self.mon(target).hp == 0 {
            return Ok(false);
        }
        let mut changed = false;
        let has_boost = effect.boosts.iter().any(|b| *b != 0);
        if has_boost {
            changed = self.boost(
                dex,
                target,
                source,
                effect.boosts,
                BoostCause::Move { secondary },
            )?;
        }
        if let Some([n, d]) = effect.heal {
            let p = self.mon(target);
            let amount =
                ((u32::from(p.stats[0]) * u32::from(n) + u32::from(d) / 2) / u32::from(d)) as u16;
            let amount = amount.min(p.stats[0] - p.hp);
            if amount == 0 {
                return Ok(false);
            }
            self.mon_mut(target).hp += amount;
            self.emit(
                EventKind::Heal,
                target,
                Some(source),
                EffectRef::None,
                i32::from(amount),
                true,
            )?;
            changed = true;
        }
        if effect.status != 0 {
            if self.mon(target).status != 0 {
                return Ok(false);
            }
            let status = effect.status;
            let p = self.mon(target);
            let fx = &dex.effects;
            let immune = (status == fx.burn && p.types.contains(&fx.fire))
                || (status == fx.paralysis && p.types.contains(&fx.electric))
                || (status == fx.freeze
                    && (p.types.contains(&fx.ice)
                        || self.effective_weather(dex) == fx.sun
                        // Magma Armor reports immunity through `onImmunity`.
                        || dex.effects.abilities[p.ability as usize]
                            == Ability::Magmaarmor))
                || ([fx.poison, fx.toxic].contains(&status)
                    && (p.types.contains(&fx.poison_type) || p.types.contains(&fx.steel)));
            let terrain = self.terrain_id(dex);
            let terrain_blocks = self.grounded(dex, target)
                && (terrain == fx.misty_terrain
                    || (terrain == fx.electric_terrain && status == fx.sleep));
            if immune || terrain_blocks {
                return Ok(false);
            }
            // `onSetStatus` refusals for the ported status-immunity abilities.
            // The public immunity message only appears when the source effect
            // carries a `status` field, i.e. not for ability-sourced statuses.
            if self.status_immune_ability(dex, target, status).is_some() {
                if ability_source.is_none() {
                    self.reveal_ability(target)?;
                }
                return Ok(false);
            }
            // `abilities:flowerveil.onAllySetStatus`: a Grass-type ally (or the
            // holder itself) refuses statuses from another Pokémon. The public
            // block message is skipped for secondary and ability sources.
            if target != source && self.mon(target).types.contains(&fx.grass) {
                let holder = self.active_entities(false).into_iter().find(|h| {
                    h.side == target.side
                        && dex.effects.abilities[self.mon(*h).ability as usize]
                            == Ability::Flowerveil
                });
                if let Some(holder) = holder {
                    if !secondary && ability_source.is_none() {
                        self.reveal_ability(holder)?;
                    }
                    return Ok(false);
                }
            }
            if ![
                fx.burn,
                fx.paralysis,
                fx.sleep,
                fx.freeze,
                fx.poison,
                fx.toxic,
            ]
            .contains(&status)
            {
                return Err(EngineError::Unsupported(format!("status {status}")));
            }
            let status_order = self.allocate_effect_order()?;
            let values = if status == fx.sleep {
                vec![if self.rng.below(3) == 0 { 2 } else { 3 }]
            } else if status == fx.freeze {
                vec![3]
            } else if status == fx.toxic {
                vec![0]
            } else {
                vec![]
            };
            self.mon_mut(target).status = status;
            self.mon_mut(target).status_state = EffectState {
                id: status,
                effect_order: status_order,
                effect_order_assigned: true,
                source: Some((
                    if source.side == 0 {
                        SideId::P1
                    } else {
                        SideId::P2
                    },
                    source.roster,
                )),
                values,
                ..Default::default()
            };
            // The status Start message identifies an ability source before
            // AfterSetStatus items such as Lum Berry can cure the status.
            if let Some(holder) = ability_source {
                self.reveal_ability(holder)?;
            }
            self.emit(
                EventKind::Status,
                target,
                Some(source),
                EffectRef::Condition(status),
                0,
                false,
            )?;
            // AfterSetStatus's ability callback precedes the same holder's
            // item callback (suborder 7 before 8), without a speed-tie draw.
            // Synchronize activates even when reflection will fail; successful
            // reflection follows the same status/terrain/immunity/Lum pipeline.
            if dex.effects.abilities[self.mon(target).ability as usize] == Ability::Synchronize
                && source != target
                && status != fx.sleep
                && status != fx.freeze
            {
                self.reveal_ability(target)?;
                let reflected = crate::effects::HitEffect {
                    status,
                    ..Default::default()
                };
                self.hit_effect(dex, source, target, &reflected, false)?;
            }
            if dex.effects.items[self.mon(target).item as usize] == Item::LumBerry {
                self.item_update(dex, target)?;
            }
            changed = true;
        }
        if effect.volatile != 0 {
            let volatile = effect.volatile;
            if volatile == dex.effects.flinch {
                if dex.effects.abilities[self.mon(target).ability as usize] == Ability::InnerFocus {
                    return Ok(false);
                }
                // Existing flinch is not restarted, but adding it still succeeds.
                if !self.mon(target).volatiles.contains_key(&volatile) {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        volatile,
                        EffectState {
                            id: volatile,
                            duration: Some(1),
                            effect_order: order,
                            effect_order_assigned: true,
                            ..Default::default()
                        },
                    );
                }
                changed = true;
            } else if volatile == dex.effects.must_recharge {
                // `mustrecharge` has no `onRestart`, so a repeated add fails and
                // no new state is created. Its duration is decremented by the
                // residual phase and the skip is consumed by BeforeMove.
                if !self.mon(target).volatiles.contains_key(&volatile) {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        volatile,
                        EffectState {
                            id: volatile,
                            duration: Some(2),
                            effect_order: order,
                            effect_order_assigned: true,
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        None,
                        EffectRef::Condition(volatile),
                        0,
                        false,
                    )?;
                }
                changed = true;
            } else if volatile == dex.effects.confusion {
                // `confusion` has no `onRestart`: a repeated add fails without
                // rolling a new timer, and the whole effect reports failure so
                // the move's per-hit Update events are skipped exactly as the
                // reference does when every target effect returns false.
                if !self.mon(target).volatiles.contains_key(&volatile) {
                    // Reference `onStart`: `effectState.time = this.random(2, 6)`.
                    let time = i64::from(self.rng.range(2, 6));
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        volatile,
                        EffectState {
                            id: volatile,
                            effect_order: order,
                            effect_order_assigned: true,
                            values: vec![time],
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        None,
                        EffectRef::Condition(volatile),
                        0,
                        false,
                    )?;
                    changed = true;
                }
                // A failed re-add contributes nothing; an earlier successful
                // boost or status in the same effect still counts.
            } else {
                return Err(EngineError::Unsupported(format!("volatile {volatile}")));
            }
        }
        Ok(changed
            || (!has_boost && effect.heal.is_none() && effect.status == 0 && effect.volatile == 0))
    }

    fn emit(
        &mut self,
        kind: EventKind,
        subject: Entity,
        target: Option<Entity>,
        effect: EffectRef,
        value: i32,
        with_health: bool,
    ) -> Result<()> {
        let hp = self.mon(subject).hp;
        let max_hp = self.mon(subject).stats[0];
        for viewer in 0..2 {
            let entity = |e: Entity| e.roster + if e.side as usize == viewer { 0 } else { 6 };
            let health = with_health.then(|| {
                if viewer == subject.side as usize {
                    HealthDisplay {
                        numerator: hp,
                        denominator: max_hp,
                        boundary_color: 0,
                    }
                } else {
                    public_health(hp, max_hp)
                }
            });
            // Actual damage integers are private to the owner. Opponents get
            // their permitted HP display; no hidden exact delta in event tokens.
            let public_value = if matches!(
                kind,
                EventKind::Damage
                    | EventKind::Heal
                    | EventKind::SideEffectStart
                    | EventKind::FieldEffectStart
            ) && viewer != subject.side as usize
            {
                0
            } else {
                value
            };
            let event = SemanticEvent {
                kind,
                subject: entity(subject),
                target: target.map(entity),
                effect: effect.id(),
                effect_kind: effect.kind(),
                value: public_value,
                health,
            };
            self.knowledge[viewer].apply(event)?;
            if let Some(trace) = &mut self.trace {
                trace.events[viewer].push(TraceEvent {
                    turn: self.turn,
                    event,
                });
            }
        }
        Ok(())
    }

    fn process_faints(&mut self, dex: &Dex, check_win: bool) -> Result<()> {
        if self.outcome.terminated {
            return Ok(());
        }
        let mut last = None;
        for e in std::mem::take(&mut self.faint_queue) {
            if self.mon(e).fainted {
                continue;
            }
            self.emit(EventKind::Faint, e, None, EffectRef::None, 0, true)?;
            self.ability_end(dex, e)?;
            self.clear_volatile(dex, e);
            self.mon_mut(e).fainted = true;
            last = Some(e);
        }
        if check_win && last.is_some() {
            self.check_win(last);
        }
        Ok(())
    }

    fn check_win(&mut self, last_faint: Option<Entity>) {
        if self.outcome.terminated {
            return;
        }
        let left: [usize; 2] = std::array::from_fn(|s| {
            self.sides[s]
                .pokemon
                .iter()
                .filter(|p| p.selected && !p.fainted)
                .count()
        });
        if left.contains(&0) {
            // Gen 5+ awards a simultaneous last KO to the last fainted side.
            // The queue order is the order damage was applied, not roster order.
            let winner = if left == [0, 0] {
                last_faint.map(|e| if e.side == 0 { SideId::P1 } else { SideId::P2 })
            } else {
                Some(if left[0] > 0 { SideId::P1 } else { SideId::P2 })
            };
            self.outcome = Outcome {
                terminated: true,
                reason: Some(EndReason::LastPokemon),
                winner,
                ..Default::default()
            };
            for request in &mut self.requests {
                request.kind = RequestKind::Finished;
            }
        }
    }

    fn residual(&mut self, dex: &Dex) -> Result<()> {
        let dbg = std::env::var("PA3_RNG_DBG").is_ok();
        if dbg {
            eprintln!("RNG residual start draws {}", self.rng.draws);
        }
        self.update_speed(dex);
        let mut handlers = SmallVec::<[(Entity, Id, u8, Priority); 16]>::new();
        // Reference sorts handlers from occupied slots, including fainted
        // holders, then skips their effects during execution.
        for e in self.active_entities(true) {
            if self.terrain_id(dex) == dex.effects.grassy_terrain {
                handlers.push((
                    e,
                    dex.effects.grassy_terrain,
                    8,
                    Priority {
                        order: 5,
                        sub_order: 2,
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                ));
            }
            let status = self.mon(e).status;
            let order = if status == dex.effects.burn {
                10
            } else if [dex.effects.poison, dex.effects.toxic].contains(&status) {
                9
            } else {
                0
            };
            if order != 0 {
                handlers.push((
                    e,
                    status,
                    1u8,
                    Priority {
                        order,
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                ));
            }
            if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Hydration {
                // Collect the handler even when status/weather predicates fail:
                // its position in residual ties is still reference-visible.
                handlers.push((
                    e,
                    self.mon(e).ability,
                    9,
                    Priority {
                        order: 5,
                        sub_order: 3,
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                ));
            }
            if dex.effects.abilities[self.mon(e).ability as usize] == Ability::SpeedBoost {
                handlers.push((
                    e,
                    self.mon(e).ability,
                    2,
                    Priority {
                        order: 28,
                        sub_order: 2,
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                ));
            }
            if dex.effects.items[self.mon(e).item as usize] == Item::Leftovers {
                handlers.push((
                    e,
                    self.mon(e).item,
                    3,
                    Priority {
                        order: 5,
                        sub_order: 4,
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                ));
            }
            // Reference residual order/suborder for the remaining ported items.
            let item = dex.effects.items[self.mon(e).item as usize];
            let item_order = match item {
                Item::BlackSludge => Some((5, 4, 3u8)),
                Item::StickyBarb => Some((28, 3, 11u8)),
                Item::WhiteHerb => Some((29, 8, 12u8)),
                _ => None,
            };
            if let Some((order, sub_order, kind)) = item_order {
                handlers.push((
                    e,
                    self.mon(e).item,
                    kind,
                    Priority {
                        order,
                        sub_order,
                        speed: self.mon(e).cached_speed,
                        ..Default::default()
                    },
                ));
            }
            for (&id, state) in &self.mon(e).volatiles {
                if state.duration.is_some() {
                    // `moves:throatchop.condition` declares `onResidualOrder: 22`
                    // for its expiry tick; other timed volatiles stay unordered.
                    let (order, sub_order) = if id == dex.effects.throat_chop {
                        (22, 0)
                    } else {
                        (0, 0)
                    };
                    handlers.push((
                        e,
                        id,
                        0u8,
                        Priority {
                            order,
                            sub_order,
                            speed: self.mon(e).cached_speed,
                            ..Default::default()
                        },
                    ));
                }
            }
        }
        for side in 0..2 {
            for (&id, state) in &self.sides[side].conditions {
                // Showdown resolves `onSideResidualOrder`; the Reflect / Light
                // Screen / Tailwind family uses 26 with their own sub-orders,
                // while Wide Guard / Quick Guard have no order and fall into
                // the unordered bucket (order 0 sorts last) as side
                // conditions (sub-order 4).
                let (order, sub_order) = if id == dex.effects.reflect {
                    (26, 1)
                } else if id == dex.effects.light_screen {
                    (26, 2)
                } else if id == dex.effects.tailwind {
                    (26, 5)
                } else if id == dex.effects.aurora_veil {
                    (26, 10)
                } else if id == dex.effects.wide_guard || id == dex.effects.quick_guard {
                    (0, 4)
                } else {
                    return Err(EngineError::Unsupported(format!("side condition {id}")));
                };
                if state.duration.is_some() {
                    handlers.push((
                        Entity {
                            side: side as u8,
                            roster: 0,
                        },
                        id,
                        4,
                        Priority {
                            order,
                            sub_order,
                            ..Default::default()
                        },
                    ));
                }
            }
        }
        let weather = self.weather_id(dex);
        if weather != 0 {
            handlers.push((
                Entity { side: 0, roster: 0 },
                weather,
                5,
                Priority {
                    order: 1,
                    sub_order: 5,
                    ..Default::default()
                },
            ));
        }
        if self.field.contains_key(&dex.effects.trick_room) {
            handlers.push((
                Entity { side: 0, roster: 0 },
                dex.effects.trick_room,
                6,
                Priority {
                    order: 27,
                    sub_order: 1,
                    ..Default::default()
                },
            ));
        }
        let terrain = self.terrain_id(dex);
        if terrain != 0 {
            handlers.push((
                Entity { side: 0, roster: 0 },
                terrain,
                7,
                Priority {
                    order: 27,
                    sub_order: 7,
                    ..Default::default()
                },
            ));
        }
        speed_sort(&mut handlers, &mut self.rng, |x| x.3);
        if dbg {
            eprintln!(
                "RNG residual after sort n={} draws {}",
                handlers.len(),
                self.rng.draws
            );
        }
        for (e, id, status, _) in handlers {
            if status == 7 {
                self.terrain_upkeep(id)?;
                continue;
            }
            if matches!(status, 11 | 12) {
                item_ports::residual_item(self, dex, e, id)?;
                continue;
            }
            if status == 8 {
                self.grassy_heal(dex, e)?;
                continue;
            }
            if status == 6 {
                self.trick_room_upkeep(dex)?;
                continue;
            }
            if status == 5 {
                self.weather_upkeep(dex, id)?;
                if self.outcome.terminated {
                    return Ok(());
                }
                continue;
            }
            if status == 4 {
                let state = self.sides[e.side as usize].conditions.get_mut(&id).unwrap();
                let duration = state.duration.as_mut().unwrap();
                *duration = duration.saturating_sub(1);
                // The casting side knows its own held-item duration. Tick
                // knowledge without revealing the opponent's hidden Light Clay.
                for (viewer, knowledge) in self.knowledge.iter_mut().enumerate() {
                    let relative = usize::from(viewer != e.side as usize);
                    if let Some(effect) = knowledge.sides[relative].get_mut(&id)
                        && effect.duration.known
                    {
                        effect.duration.value = effect.duration.value.saturating_sub(1);
                    }
                }
                if *duration == 0 {
                    let state = self.sides[e.side as usize].conditions.remove(&id).unwrap();
                    let source = state
                        .source
                        .map(|(side, roster)| Entity {
                            side: side.index() as u8,
                            roster,
                        })
                        .unwrap_or(e);
                    self.emit(
                        EventKind::SideEffectEnd,
                        source,
                        None,
                        EffectRef::Condition(id),
                        0,
                        false,
                    )?;
                }
                continue;
            }
            if self.mon(e).fainted {
                continue;
            }
            if status == 9 {
                if self.mon(e).ability == id
                    && self.mon(e).status != 0
                    && self.effective_weather(dex) == dex.effects.rain
                {
                    self.reveal_ability(e)?;
                    self.cure_status(e)?;
                }
            } else if status == 3 {
                if self.mon(e).item == id {
                    self.item_heal(dex, e, self.mon(e).stats[0] / 16, false)?;
                }
            } else if status == 2 {
                if self.mon(e).ability != id {
                    continue;
                }
                if self.mon(e).active_turns > 0 {
                    self.reveal_ability(e)?;
                    self.boost(
                        dex,
                        e,
                        e,
                        [0, 0, 0, 0, 1, 0, 0],
                        BoostCause::Ability(Ability::SpeedBoost),
                    )?;
                }
            } else if status == 1 {
                if self.mon(e).status != id {
                    continue;
                }
                let ability = dex.effects.abilities[self.mon(e).ability as usize];
                // `abilities:magicguard.onDamage` refuses the whole source,
                // after the toxic stage has already advanced.
                let stage = if id == dex.effects.toxic {
                    let state = &mut self.mon_mut(e).status_state;
                    state.values[0] = (state.values[0] + 1).min(15);
                    state.values[0] as u16
                } else {
                    1
                };
                if ability == Ability::Magicguard {
                    continue;
                }
                // `abilities:poisonheal.onDamage`: poison residual heals
                // `baseMaxhp / 8` instead of damaging.
                if ability == Ability::Poisonheal
                    && (id == dex.effects.poison || id == dex.effects.toxic)
                {
                    let p = self.mon(e);
                    if p.hp > 0 && p.hp < p.stats[0] {
                        let amount = (p.stats[0] / 8).max(1).min(p.stats[0] - p.hp);
                        self.reveal_ability(e)?;
                        self.mon_mut(e).hp += amount;
                        self.emit(
                            EventKind::Heal,
                            e,
                            None,
                            EffectRef::Ability(self.mon(e).ability),
                            i32::from(amount),
                            true,
                        )?;
                    }
                    continue;
                }
                let denominator = if id == dex.effects.poison { 8 } else { 16 };
                let mut damage = (self.mon(e).stats[0] / denominator).max(1) * stage;
                // `abilities:heatproof.onDamage` halves burn residual damage.
                if ability == Ability::Heatproof && id == dex.effects.burn {
                    damage = (damage / 2).max(1);
                }
                let actual = damage.min(self.mon(e).hp);
                self.mon_mut(e).hp -= actual;
                if self.mon(e).hp == 0 {
                    self.faint_queue.push(e);
                }
                self.emit(
                    EventKind::Damage,
                    e,
                    None,
                    EffectRef::Condition(id),
                    -i32::from(actual),
                    true,
                )?;
                self.process_faints(dex, true)?;
                if self.outcome.terminated {
                    return Ok(());
                }
            } else {
                let expired = if let Some(state) = self.mon_mut(e).volatiles.get_mut(&id) {
                    let duration = state.duration.as_mut().unwrap();
                    *duration = duration.saturating_sub(1);
                    *duration == 0
                } else {
                    false
                };
                if expired {
                    self.mon_mut(e).volatiles.remove(&id);
                    if id == dex.effects.protect || id == dex.effects.throat_chop {
                        self.emit(
                            EventKind::EffectEnd,
                            e,
                            None,
                            EffectRef::Condition(id),
                            0,
                            false,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn bench(&self, side: usize) -> SmallVec<[u8; 4]> {
        // Reference requests list switch destinations in request-team order
        // (the preview pick order), which never changes when Pokémon switch;
        // `positions[2..]` is the live arrangement and would reorder reserves.
        let Some(order) = self.sides[side].selected_order else {
            return SmallVec::new();
        };
        order
            .iter()
            .copied()
            .filter(|r| {
                let p = &self.sides[side].pokemon[*r as usize];
                !p.fainted && p.active_slot.is_none()
            })
            .collect()
    }

    /// Reference `possibleSwitches`: possible switch-ins in live party order
    /// (`side.pokemon` positions after the active slots). The phazing drag
    /// samples this array, which is deliberately not the request-team order
    /// used by `bench`.
    fn party_reserves(&self, side: usize) -> SmallVec<[u8; 4]> {
        let state = &self.sides[side];
        state.positions[2..]
            .iter()
            .copied()
            .filter(|r| !state.pokemon[*r as usize].fainted)
            .collect()
    }

    /// Reference `canSwitch`: a selected, unfainted reserve exists.
    fn can_switch(&self, side: usize) -> bool {
        !self.bench(side).is_empty()
    }

    /// Reference post-action `switchFlag` handling. `selfSwitch` pivots flag
    /// their user while the move resolves; the next action boundary turns that
    /// flag into an `instaswitch` request unless the side has nothing to send
    /// in, in which case the reference clears the flag and play continues.
    /// Returns true when a switch request was issued and the remaining queue
    /// must wait for the choice.
    /// Reference phazing: every active Pokémon marked by `forceSwitch` drags a
    /// random reserve in, and the flag clears whether or not the drag resolves.
    fn resolve_forced_switches(&mut self, dex: &Dex) -> Result<()> {
        for side in 0..2 {
            for slot in 0..2 {
                let Some(roster) = self.sides[side].active[slot] else {
                    continue;
                };
                let target = Entity {
                    side: side as u8,
                    roster,
                };
                if !self.mon(target).force_switch_flag {
                    continue;
                }
                self.mon_mut(target).force_switch_flag = false;
                if self.mon(target).hp == 0 || !self.can_switch(side) {
                    continue;
                }
                // `getRandomSwitchable`: one uniform draw over the reserve
                // array, in reference party order.
                let bench = self.party_reserves(side);
                if bench.is_empty() {
                    continue;
                }
                let index = self.rng.below(bench.len() as u32) as usize;
                let incoming = Entity {
                    side: side as u8,
                    roster: bench[index],
                };
                self.switch_in_inner(dex, incoming, slot as u8, true)?;
            }
        }
        Ok(())
    }

    fn make_pivot_requests(&mut self, dex: &Dex) -> bool {
        if self.outcome.terminated {
            return false;
        }
        let mut needed = [false; 2];
        for (side, needed) in needed.iter_mut().enumerate() {
            let flagged = self.sides[side]
                .active
                .iter()
                .flatten()
                .any(|r| self.sides[side].pokemon[*r as usize].switch_flag.is_some());
            if !flagged {
                continue;
            }
            if self.can_switch(side) {
                *needed = true;
            } else {
                for r in self.sides[side].active.iter().flatten() {
                    self.sides[side].pokemon[*r as usize].switch_flag = None;
                }
            }
        }
        if !needed.iter().any(|b| *b) {
            return false;
        }
        for (side, needed) in needed.into_iter().enumerate() {
            let slots = std::array::from_fn(|slot| {
                let Some(roster) = self.sides[side].active[slot] else {
                    return SlotRequest::default();
                };
                let p = &self.sides[side].pokemon[roster as usize];
                SlotRequest {
                    present: !p.fainted,
                    // A `selfSwitch` pivot is alive and keeps its move list;
                    // only `forceSwitch` marks the slot as actionable.
                    requires_replacement: needed && p.switch_flag.is_some(),
                    // The reference's switch request carries no `active` entry,
                    // so it advertises no Mega availability for the swapper.
                    can_mega: false,
                    moves: p
                        .moves
                        .iter()
                        .enumerate()
                        .map(|(slot, mv)| MoveChoice {
                            id: mv.id,
                            slot: slot as u8,
                            target: dex.moves[mv.id as usize].target,
                            disabled: mv.disabled,
                            pp: mv.pp,
                        })
                        .collect(),
                    ..Default::default()
                }
            });
            self.requests[side] = Request {
                kind: if needed {
                    RequestKind::Replacement
                } else {
                    RequestKind::Wait
                },
                slots,
                bench: self.bench(side).into_vec(),
                preview_roster: vec![],
            };
        }
        true
    }

    fn make_replacement_requests(&mut self) -> bool {
        // checkFainted replaces the status with the faint marker only after the
        // queue drains. A terminal KO can retain the previous status in the world.
        for e in self.active_entities(true) {
            if self.mon(e).fainted {
                self.mon_mut(e).status = 0;
            }
        }
        let needed: [bool; 2] = std::array::from_fn(|side| {
            !self.bench(side).is_empty()
                && self.sides[side]
                    .active
                    .iter()
                    .flatten()
                    .any(|r| self.sides[side].pokemon[*r as usize].fainted)
        });
        if !needed.iter().any(|b| *b) {
            return false;
        }
        for (side, needed) in needed.into_iter().enumerate() {
            let slots = std::array::from_fn(|slot| SlotRequest {
                requires_replacement: self.sides[side].active[slot]
                    .is_some_and(|r| self.sides[side].pokemon[r as usize].fainted),
                ..Default::default()
            });
            self.requests[side] = Request {
                kind: if needed {
                    RequestKind::Replacement
                } else {
                    RequestKind::Wait
                },
                slots,
                bench: self.bench(side).into_vec(),
                preview_roster: vec![],
            };
        }
        true
    }

    fn end_turn(&mut self, dex: &Dex) -> Result<()> {
        let dbg = std::env::var("PA3_RNG_DBG").is_ok();
        if dbg {
            eprintln!("RNG end_turn start draws {}", self.rng.draws);
        }
        for e in self.active_entities(false) {
            self.mon_mut(e).active_turns += 1;
        }
        self.mid_turn = false;
        self.turn += 1;
        if self.turn > 1000 {
            self.outcome = Outcome {
                terminated: true,
                reason: Some(EndReason::RuleTurnLimit),
                ..Default::default()
            };
            for r in &mut self.requests {
                r.kind = RequestKind::Finished;
            }
            return Ok(());
        }
        for side in 0..2 {
            for mon in &mut self.sides[side].pokemon {
                // Reference `makeRequest` only resets and re-applies disabled
                // move flags for Pokémon currently on the field; a benched
                // Pokémon keeps its frozen flags until it is active again.
                if mon.active_slot.is_none() {
                    continue;
                }
                // Reference `choicelock.onDisableMove`: the lock is dropped
                // lazily when the holder no longer has a Choice item (Knock
                // Off, Trick, …) or no longer knows the locked move.
                let choice_item = matches!(
                    dex.effects.items[mon.item as usize],
                    Item::ChoiceScarf | Item::ChoiceBand | Item::ChoiceSpecs
                );
                let mut locked = mon
                    .volatiles
                    .get(&dex.effects.choice_lock)
                    .and_then(|v| v.values.first())
                    .copied();
                if locked.is_some_and(|id| {
                    !choice_item || !mon.moves.iter().any(|mv| i64::from(mv.id) == id)
                }) {
                    mon.volatiles.remove(&dex.effects.choice_lock);
                    locked = None;
                }
                // Champions `fakeout.onDisableMove`: only an active Pokémon
                // that has already run a move this stint loses Fake Out.
                let fake_out_disabled = mon.active_move_actions != 0;
                // `moves:throatchop.condition.onDisableMove`: every sound move
                // is disabled in the request while the volatile is active.
                let throat_chop = mon
                    .volatiles
                    .contains_key(&dex.effects.throat_chop);
                for mv in &mut mon.moves {
                    mv.disabled = locked.is_some_and(|id| id != i64::from(mv.id))
                        || (fake_out_disabled && mv.id == dex.effects.fake_out)
                        || (throat_chop && dex.moves[mv.id as usize].sound);
                }
            }
            let slots = std::array::from_fn(|slot| {
                let Some(roster) = self.sides[side].active[slot] else {
                    return SlotRequest::default();
                };
                let p = &self.sides[side].pokemon[roster as usize];
                SlotRequest {
                    present: !p.fainted,
                    can_mega: !p.fainted
                        && self
                            .mega_form(
                                dex,
                                Entity {
                                    side: side as u8,
                                    roster,
                                },
                            )
                            .is_some(),
                    moves: p
                        .moves
                        .iter()
                        .enumerate()
                        .map(|(slot, mv)| MoveChoice {
                            id: mv.id,
                            slot: slot as u8,
                            target: dex.moves[mv.id as usize].target,
                            disabled: mv.disabled,
                            pp: mv.pp,
                        })
                        .collect(),
                    ..Default::default()
                }
            });
            self.requests[side] = Request {
                kind: RequestKind::Normal,
                slots,
                bench: self.bench(side).into_vec(),
                preview_roster: vec![],
            };
        }
        Ok(())
    }
}
