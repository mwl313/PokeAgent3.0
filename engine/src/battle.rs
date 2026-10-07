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
        BattleState, EffectState, EndReason, FaintData, MoveResult, NativeTrace, Outcome,
        PokemonState, SideId, TraceEntry, TraceEvent,
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

/// One hit of a hit phase: the spread flag and the 1-based hit number
/// (`multiaccuracy` and `basePowerCallback` formulas read the hit index).
#[derive(Clone, Copy)]
pub(crate) struct HitPhase {
    pub spread: bool,
    pub hit: u32,
    /// `abilities:parentalbond`: the second hit is quarter power.
    pub parental_bond_second_hit: bool,
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
                        // Reference `getActionSpeed` reads `getTarget` for the
                        // queued action. The Recharge pseudo-move has no target
                        // class, so that read samples a random foe instead of
                        // resolving the chosen location.
                        let recharge_lock = self
                            .mon(actor)
                            .volatiles
                            .contains_key(&dex.effects.must_recharge);
                        if recharge_lock {
                            self.sample_random_foe(actor);
                        } else {
                            if queued.target_location == 0 {
                                queued.target_location =
                                    self.random_target_location(actor, m.target);
                            }
                            // getActionSpeed resolves a target even for a
                            // constant priority. That resolution can consume a
                            // reference RNG draw.
                            self.resolve_target_location(actor, m.target, queued.target_location);
                        }
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
    /// Reference `Pokemon#getTypes()`: the stored type list after volatile
    /// `onType` handlers run. Roost's Flying removal is the only ported
    /// volatile type handler; an emptied list falls back to Normal.
    pub(super) fn effective_types(&self, dex: &Dex, e: Entity) -> SmallVec<[Id; 4]> {
        let mon = self.mon(e);
        let mut types: SmallVec<[Id; 4]> = mon.types.iter().copied().collect();
        if mon.volatiles.contains_key(&dex.effects.roost) {
            types.retain(|t| *t != dex.effects.flying);
            if types.is_empty() {
                types.push(dex.effects.normal);
            }
        }
        types
    }

    /// Reference `Pokemon#setType`: replace the stored type list outright.
    /// The pinned regulation reaches this only through Soak and Double Shock,
    /// whose results are always representable; a typeless result (a pure
    /// Electric Double Shock user) stays an explicit operational error.
    /// `knownType`/`apparentType` bookkeeping has no native counterpart.
    pub(super) fn set_type(&mut self, dex: &Dex, e: Entity, types: &[Id]) -> Result<bool> {
        let _ = dex;
        if types.is_empty() {
            return Err(EngineError::Unsupported("typeless result".into()));
        }
        // Type id 0 is the `'???'` placeholder Double Shock leaves behind.
        if self.mon(e).types.as_slice() == types {
            return Ok(false);
        }
        self.mon_mut(e).types = types.to_vec();
        Ok(true)
    }

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
            let pivot_switch =
                self.mon(old).switch_flag.is_some() || self.mon(old).plain_switch_flag;
            if self.mon(old).hp > 0 {
                // Reference `switchIn` runs BeforeSwitchOut plus a full Update
                // for a voluntary switch only.
                if !drag && !pivot_switch {
                    self.each_update(dex)?;
                }
                // Reference `switchIn`: a Pokémon leaving the field (pivot,
                // emergency exit, drag) cannot use its queued move any more.
                self.queue.retain(|q| q.actor != Some(old));
                self.ability_switch_out(dex, old)?;
                self.ability_end(dex, old)?;
            }
            self.clear_volatile(dex, old);
            // Reference `clearVolatile` also drops the pending switch flags of
            // the Pokémon leaving the field.
            self.mon_mut(old).switch_flag = None;
            self.mon_mut(old).plain_switch_flag = false;
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
        self.mon_mut(incoming).plain_switch_flag = false;
        self.mon_mut(incoming).force_switch_flag = false;
        self.mon_mut(incoming).ability_ending = false;
        self.mon_mut(incoming).protean_used = false;
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
                // Reference `findSideEventHandlers(side, 'onSwitchIn')`: entry
                // hazards are side conditions of the entrant's own side and
                // sort at sub-order 4, before its ability (7) and item (8).
                self.hazard_switch_in(dex, e)?;
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

    /// Reference `Side#addSideCondition` for the entry-hazard family: a fresh
    /// layer starts at one; an existing condition restarts and adds a layer up
    /// to the reference cap (Toxic Spikes caps at 2), returning false when the
    /// cap is already reached. Hazards carry no duration, so they never join
    /// the timed residual sweep.
    fn add_side_hazard(
        &mut self,
        dex: &Dex,
        side: usize,
        source: Entity,
        id: Id,
    ) -> Result<bool> {
        debug_assert_eq!(id, dex.effects.toxic_spikes);
        // The public event names the *side* that now carries the hazard, not
        // the Toxic Debris holder (which stays the recorded source).
        let subject = self.sides[side]
            .active
            .iter()
            .flatten()
            .next()
            .map(|roster| Entity {
                side: side as u8,
                roster: *roster,
            })
            .unwrap_or(Entity {
                side: side as u8,
                roster: 0,
            });
        if let Some(state) = self.sides[side].conditions.get_mut(&id) {
            let layers = state.values.first().copied().unwrap_or(1);
            if layers >= 2 {
                return Ok(false);
            }
            state.values[0] = layers + 1;
            let layers = state.values[0];
            state.duration = Some(layers as u16);
            self.emit(
                EventKind::SideEffectStart,
                subject,
                None,
                EffectRef::Condition(id),
                layers as i32,
                false,
            )?;
            return Ok(true);
        }
        let order = self.allocate_effect_order()?;
        self.sides[side].conditions.insert(
            id,
            EffectState {
                id,
                // The layer count is stored in the duration slot so the
                // entry-hazard fixture contract (id, layers) matches every
                // other side condition; `residual` skips this id entirely.
                duration: Some(1),
                values: vec![1],
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
            },
        );
        self.emit(
            EventKind::SideEffectStart,
            subject,
            None,
            EffectRef::Condition(id),
            1,
            false,
        )?;
        Ok(true)
    }

    /// Reference `moves:toxicspikes.condition.onSwitchIn`: a grounded entrant
    /// absorbs the hazard when it is a Poison type, ignores it as a Steel type,
    /// and is otherwise poisoned (one layer) or badly poisoned (two layers) by
    /// the opposing side's first active Pokémon.
    fn hazard_switch_in(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        let side = e.side as usize;
        let mut hazards: SmallVec<[(u32, Id); 2]> = self.sides[side]
            .conditions
            .iter()
            .filter(|(id, _)| **id == dex.effects.toxic_spikes)
            .map(|(id, state)| (state.effect_order, *id))
            .collect();
        hazards.sort_by_key(|(order, _)| *order);
        for (_, id) in hazards {
            if !self.grounded(dex, e) {
                continue;
            }
            if self.mon(e).types.contains(&dex.effects.poison_type) {
                self.sides[side].conditions.remove(&id);
                self.emit(
                    EventKind::SideEffectEnd,
                    e,
                    None,
                    EffectRef::Condition(id),
                    0,
                    false,
                )?;
                continue;
            }
            if self.mon(e).types.contains(&dex.effects.steel) {
                continue;
            }
            let layers = self.sides[side].conditions[&id]
                .values
                .first()
                .copied()
                .unwrap_or(1);
            let status = if layers >= 2 {
                dex.effects.toxic
            } else {
                dex.effects.poison
            };
            let foe = 1 - side;
            let Some(source) = self.sides[foe]
                .active
                .iter()
                .flatten()
                .map(|roster| Entity {
                    side: foe as u8,
                    roster: *roster,
                })
                .next()
            else {
                // The reference passes `side.foe.active[0]`, which can only be
                // empty once the battle is over; nothing to apply.
                continue;
            };
            let effect = crate::effects::HitEffect {
                status,
                ..Default::default()
            };
            self.hit_effect(dex, e, source, &effect, false)?;
        }
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
        // Reference `clearVolatile` also clears the recorded last move; Encore,
        // Disable and Torment read it after the Pokémon re-enters.
        mon.last_move = 0;
        // Reference `clearVolatile`: the hit counter and both move-result
        // slots are per-stint state (Rage Fist, Stomping Tantrum).
        mon.times_attacked = 0;
        mon.move_this_turn_result = crate::state::MoveResult::Undefined;
        mon.move_last_turn_result = crate::state::MoveResult::Undefined;
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
                        // An action can only be re-resolved while its actor is
                        // still on the field; a mid-turn switch cancels it.
                        if self.mon(e).active_slot.is_none() {
                            continue;
                        }
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

    /// Reference `Side#randomFoe`: samples one live foe with `battle.sample`,
    /// which consumes a draw even when the foe list is empty.
    fn sample_random_foe(&mut self, actor: Entity) -> Option<Entity> {
        let foes: SmallVec<[Entity; 2]> = self.sides[(1 - actor.side) as usize]
            .active
            .iter()
            .flatten()
            .map(|roster| Entity {
                side: 1 - actor.side,
                roster: *roster,
            })
            .filter(|e| self.mon(*e).hp > 0)
            .collect();
        let index = self.rng.below(foes.len() as u32) as usize;
        foes.get(index).copied()
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

    /// Reference `onTryMove` of the two-turn charge family. Returns `true`
    /// when the move may proceed this turn, `false` after a charge turn.
    ///
    /// `runEvent('ChargeMove')` has exactly one handler in the pinned game
    /// (Power Herb), and Power Herb is not a legal regulation item, so the
    /// event has no reachable handler. `runEvent('PrepareHit')` is likewise
    /// only served by Libero / Protean / Parental Bond, all of which stay
    /// explicit operational errors, so a skipping port cannot go silent.
    fn charge_try_move(
        &mut self,
        dex: &Dex,
        actor: Entity,
        m: &crate::assets::Move,
        move_id: Id,
        loc: i8,
    ) -> Result<bool> {
        let Some(spec) = m.charge.as_ref() else {
            return Ok(true);
        };
        // `if (attacker.removeVolatile(move.id)) return;` — the second turn.
        if self.mon(actor).volatiles.contains_key(&move_id) {
            self.mon_mut(actor).volatiles.remove(&move_id);
            return Ok(true);
        }
        // `this.add('-prepare', attacker, move.name)`: the charged move is
        // public from this point, and `twoturnmove.onStart` records the
        // player's chosen location for the release turn.
        if spec.prepare_boost.iter().any(|b| *b != 0) {
            self.boost(
                dex,
                actor,
                actor,
                spec.prepare_boost,
                BoostCause::Move { secondary: false },
            )?;
        }
        if spec.instant_weather.contains(&self.effective_weather(dex)) {
            return Ok(true);
        }
        let target_location = i64::from(loc);
        let order = self.allocate_effect_order()?;
        self.mon_mut(actor).volatiles.insert(
            dex.effects.two_turn_move,
            EffectState {
                id: dex.effects.two_turn_move,
                duration: Some(2),
                values: vec![i64::from(move_id), target_location],
                effect_order: order,
                effect_order_assigned: true,
                ..Default::default()
            },
        );
        self.emit(
            EventKind::EffectStart,
            actor,
            None,
            EffectRef::Condition(dex.effects.two_turn_move),
            0,
            false,
        )?;
        // `attacker.addVolatile(effect.id)`: the move's own marker volatile,
        // which carries the declared condition (semi-invulnerability) and its
        // duration when the move declares one.
        let order = self.allocate_effect_order()?;
        self.mon_mut(actor).volatiles.insert(
            move_id,
            EffectState {
                id: move_id,
                duration: spec.volatile_duration,
                values: vec![target_location],
                effect_order: order,
                effect_order_assigned: true,
                ..Default::default()
            },
        );
        Ok(false)
    }

    /// The ported charge recipe of the move `e` is currently charging, if any:
    /// the marker volatile's id is the charging move's own id.
    pub(super) fn charging_spec<'a>(
        &self,
        dex: &'a Dex,
        e: Entity,
    ) -> Option<&'a crate::effects::ChargeSpec> {
        self.mon(e).volatiles.keys().find_map(|id| {
            let index = *id as usize;
            (index < dex.moves.len())
                .then(|| dex.moves[index].charge.as_ref())
                .flatten()
        })
    }

    fn use_move(&mut self, dex: &Dex, actor: Entity, slot: u8, move_id: Id, loc: i8) -> Result<()> {
        // Reference `moveUsed(move, targetLoc)` records the player's chosen
        // location before `getTarget` resolves it; the two-turn charge
        // condition stores that value for the release turn.
        let chosen_location = loc;
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
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            // The reference TryMove abort still runs the single Update that
            // precedes the action's own queue re-sort.
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::Unimplemented {
            return Err(EngineError::Unsupported(format!("move ID {move_id}")));
        }
        // Reference `useMove` clears the attempt result before running; the
        // outcome is recorded at the point the move resolves. Stomping Tantrum
        // reads the rolled-over value next turn.
        self.mon_mut(actor).move_this_turn_result = MoveResult::Undefined;
        // Reference `runMove` counts the attempted action before BeforeMove, so
        // a flinched, sleeping or fully paralysed attempt still counts.
        let attempts = self.mon(actor).active_move_actions;
        self.mon_mut(actor).active_move_actions = attempts.saturating_add(1);
        // Reference `runMove` reads `getTarget` before `BeforeMove` runs. The
        // Recharge pseudo-move has no target class, so that read falls through
        // to `getRandomTarget` and samples a random foe (one draw) instead of
        // resolving the stored location.
        let recharge_lock = self
            .mon(actor)
            .volatiles
            .contains_key(&dex.effects.must_recharge);
        let loc = if recharge_lock {
            self.sample_random_foe(actor);
            loc
        } else {
            self.resolve_target_location(actor, m.target, loc)
        };
        if let Some(result) = self.before_move(dex, actor, m)? {
            self.mon_mut(actor).move_this_turn_result = result;
            return Ok(());
        }
        // Reference `useMoveInner` skips PP deduction while the Pokémon is
        // locked (`getLockedMove()`), i.e. on the release turn of a charge and
        // on the forced Recharge turn.
        let locked = self
            .mon(actor)
            .volatiles
            .contains_key(&dex.effects.two_turn_move)
            || self
                .mon(actor)
                .volatiles
                .contains_key(&dex.effects.must_recharge);
        if slot != NO_SLOT && !locked {
            let mon = self.mon_mut(actor);
            let pp = mon.moves[slot as usize].pp;
            if pp == 0 {
                return Ok(());
            }
            mon.moves[slot as usize].pp = pp - 1;
            mon.moves[slot as usize].used = true;
            mon.base_moves[slot as usize].pp = pp - 1;
        }
        // Reference `Pokemon#moveUsed` records the move before any hit steps,
        // so a missed, failed or status-refused move still becomes `lastMove`
        // for Encore, Disable, Torment and Cursed Body.
        self.mon_mut(actor).last_move = move_id;
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
        // Reference `useMoveInner` runs the move's own `onTryMove` before any
        // other TryMove handler. The two-turn charge family spends this turn
        // preparing (returning `null`) unless its recipe completes early.
        if !self.charge_try_move(dex, actor, m, move_id, chosen_location)? {
            return Ok(());
        }
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
                    self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                    return Ok(());
                }
            }
        }
        // Move-owned `Try` gates run after the public move message and before
        // redirection-independent hit steps. A failed try consumes no RNG and
        // ends the move without damage or secondary effects.
        let hooks = dex.effects.move_hooks[move_id as usize];
        if (hooks & crate::effects::hook::FAKE_OUT_FIRST_TURN != 0
            || hooks & crate::effects::hook::FIRST_IMPRESSION != 0)
            && self.mon(actor).active_move_actions > 1
        {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        if hooks & crate::effects::hook::DOUBLE_SHOCK != 0
            && !self.effective_types(dex, actor).contains(&dex.effects.electric)
        {
            // `moves:doubleshock.onTryMove`: the user must still be Electric.
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        if hooks & crate::effects::hook::SUCKER_PUNCH != 0 {
            let target = redirected.or(selected);
            if !self.sucker_punch_target_attacks(dex, target) {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
        }
        // `moves:teleport.onTry`: Teleport fails before any hit step when the
        // user has no switchable reserve. The plain `selfSwitch` moves instead
        // nullify their own result after a failed pivot attempt.
        // `moves:clangoroussoul.onTry`: the user must stay above a third of
        // its maximum HP (integer arithmetic) and above one HP in total.
        if hooks & crate::effects::hook::CLANGOROUS_SOUL != 0 {
            let max_hp = u32::from(self.mon(actor).stats[0]);
            if max_hp == 1 || u32::from(self.mon(actor).hp) <= max_hp * 33 / 100 {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
        }
        if hooks & crate::effects::hook::TELEPORT != 0 && !self.can_switch(actor.side as usize) {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        if behavior == MoveBehavior::Terrain {
            self.start_terrain(dex, actor, m.terrain, false)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::TrickRoom {
            self.toggle_trick_room(dex, actor)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::Weather {
            self.start_weather(dex, actor, m.weather, false)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::PerishSong {
            // Reference `moves:perishsong.onHitField`: every active Pokémon is
            // checked in field order. A miss, a protection block or a TryHit
            // refusal still counts as a successful move; only a target that
            // already carries the countdown contributes nothing, and the move
            // fails outright when no target contributes.
            let mut result = false;
            let mut message = false;
            let priority = self.effective_priority(dex, actor, move_id);
            for target in self.active_entities(false) {
                if self.mon(target).hp == 0 {
                    continue;
                }
                let protected = m.protect
                    && !m.breaks_protect
                    && (self.guard_blocks(dex, target, m, priority)
                        || self.blocking_protection(dex, target).is_some());
                if protected {
                    result = true;
                    continue;
                }
                if let Some(spec) = self.charging_spec(dex, target)
                    && spec.semi_invulnerable
                    && !spec.invuln_exceptions.contains(&move_id)
                {
                    result = true;
                    continue;
                }
                let mut accuracy = m.accuracy.map(u16::from);
                if self.absorb_try_hit(dex, target, actor, m, &mut accuracy)? {
                    result = true;
                    continue;
                }
                if !self.mon(target).volatiles.contains_key(&dex.effects.perish_song) {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        dex.effects.perish_song,
                        EffectState {
                            id: dex.effects.perish_song,
                            duration: Some(4),
                            source: Some((
                                if actor.side == 0 {
                                    SideId::P1
                                } else {
                                    SideId::P2
                                },
                                actor.roster,
                            )),
                            effect_order: order,
                            effect_order_assigned: true,
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        Some(actor),
                        EffectRef::Condition(dex.effects.perish_song),
                        0,
                        false,
                    )?;
                    result = true;
                    message = true;
                }
            }
            let _ = (message, priority);
            self.mon_mut(actor).move_this_turn_result = if result {
                MoveResult::Success
            } else {
                MoveResult::Failed
            };
            // A status move never enters the hit loop, so the reference runs no
            // Update and no AfterMoveSecondary phase here; the action tail's
            // single Update is the only one.
            return Ok(());
        }
        if behavior == MoveBehavior::Haze {
            // Reference `moves:haze.onHitField`: a public clear-all message and
            // `clearBoosts()` on every active. The native emits one Boost event
            // per stat that actually changes so observers stay exact.
            for target in self.active_entities(false) {
                for stat in 0..7usize {
                    let old = self.mon(target).boosts[stat];
                    if old == 0 {
                        continue;
                    }
                    self.mon_mut(target).boosts[stat] = 0;
                    self.emit(
                        EventKind::Boost,
                        target,
                        None,
                        EffectRef::Stat(stat as Id),
                        -i32::from(old),
                        false,
                    )?;
                }
            }
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::SideCondition {
            // Side-target moves use tryMoveHit, bypassing the Pokémon hit loop
            // and its two Update events. The queue runs the post-action Update.
            // `moves:auroraveil.onTry`: the screen only starts in snow.
            if hooks & crate::effects::hook::AURORA_VEIL != 0
                && self.effective_weather(dex) != dex.effects.snow
            {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            self.ally_try_hit_side(dex, actor, actor, m.move_type)?;
            self.start_side_condition(dex, actor, m.side_condition)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
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
                        self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                        return Ok(());
                    };
                    if ally == actor
                        || self.mon(ally).active_turns > 0
                            && !self
                                .queue
                                .iter()
                                .any(|q| q.actor == Some(ally))
                    {
                        self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
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
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
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
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
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
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if matches!(behavior, MoveBehavior::Protect | MoveBehavior::Endure) {
            let acts_left = self
                .queue
                .iter()
                .any(|q| matches!(q.kind, QueuedKind::Move | QueuedKind::Switch));
            if !acts_left {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
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
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
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
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        let spread = targets.len() > 1;
        // PrepareHit abilities run once per action, before both the multi-hit
        // dispatch and the single-hit steps.
        let preparer = dex.effects.abilities[self.mon(actor).ability as usize];
        // `abilities:protean|libero.onPrepareHit`: once per switch-in the user
        // becomes the action's (post-ModifyType) type before the hit steps,
        // even when the action later misses.
        if matches!(preparer, Ability::Protean | Ability::Libero)
            && !self.mon(actor).protean_used
            && !m.future_move
            && m.move_type != 0
        {
            let kinds = self.effective_types(dex, actor);
            if kinds.as_slice() != [m.move_type] {
                self.set_type(dex, actor, &[m.move_type])?;
                self.mon_mut(actor).protean_used = true;
                self.reveal_ability(actor)?;
            }
        }
        // `abilities:parentalbond.onPrepareHit`: a single-target, non-status,
        // non-charge, non-future, non-multi-hit damaging move gains a second
        // hit at a quarter power.
        let parental_bond = preparer == Ability::Parentalbond
            && m.category != Category::Status
            && m.multihit.is_none()
            && !m.no_parental_bond
            && m.charge.is_none()
            && !m.future_move
            && !spread
            && !m.is_z
            && !m.is_max;
        if m.multihit.is_some() || parental_bond {
            return self.use_multihit_move(dex, actor, move_id, m, targets, parental_bond);
        }
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
        // Reference `spreadMoveHit` target bookkeeping: a protection block is
        // the `NOT_FAIL` case (recorded as `null`), while a type immunity or a
        // missed accuracy roll is a real failure (`false`).
        let mut blocked_by_protection = false;
        let mut failed_otherwise = false;
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
            // `hitStepInvulnerabilityEvent` (step 0 of the hit pipeline): a
            // semi-invulnerable target is missed unless the incoming move is
            // on the recipe's exception list. The reference exempts Helping
            // Hand and a Poison-type attacker's Toxic.
            if let Some(spec) = self.charging_spec(dex, target)
                && spec.semi_invulnerable
                && !spec.invuln_exceptions.contains(&move_id)
                && move_id != dex.effects.helping_hand_move
                && !(move_id == dex.effects.toxic_move
                    && self.mon(actor).types.contains(&dex.effects.poison_type))
            {
                failed_otherwise = true;
                continue;
            }
            // `moves:yawn.onTryHit`: the target must be status-free and able to
            // fall asleep, or the move fails against it before any hit step.
            if move_id == dex.effects.yawn_move
                && (self.mon(target).status != 0
                    || self
                        .status_immune_ability(dex, target, dex.effects.sleep)
                        .is_some()
                    || self.terrain_id(dex) == dex.effects.electric_terrain
                        && self.grounded(dex, target))
            {
                failed_otherwise = true;
                continue;
            }
            // `hitStepTryHitEvent` runs whole-spread with handlers ordered by
            // priority: the priority-4 side guards and the priority-3
            // protection volatiles both precede every ability TryHit.
            if m.protect && !m.breaks_protect {
                if self.guard_blocks(dex, target, m, effective_priority) {
                    blocked_by_protection = true;
                    continue;
                }
                if let Some(volatile) = self.blocking_protection(dex, target) {
                    self.protect_punish(dex, target, actor, m, move_id, volatile)?;
                    blocked_by_protection = true;
                    continue;
                }
            }
            let ability = dex.effects.abilities[self.mon(target).ability as usize];
            if self.terrain_id(dex) == dex.effects.psychic_terrain
                && effective_priority > 0
                && target.side != actor.side
                && self.grounded(dex, target)
            {
                failed_otherwise = true;
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
                self.effective_types(dex, target)
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
            if m.powder
                && target != actor
                && self.effective_types(dex, target).contains(&dex.effects.grass)
            {
                failed_otherwise = true;
                continue;
            }
            // `hitStepTryImmunity` precedes accuracy: Sticky Hold refuses
            // Trick/Switcheroo and the target is not affected at all.
            if behavior == MoveBehavior::Trick
                && self.mon(target).ability == dex.effects.sticky_hold
            {
                failed_otherwise = true;
                continue;
            }
            // `moves:leechseed.onTryImmunity`: Grass-type targets refuse the
            // seed before any accuracy roll.
            if m.hit.volatile == dex.effects.leech_seed
                && self.mon(target).types.contains(&dex.effects.grass)
            {
                failed_otherwise = true;
                continue;
            }
            if let Some(effectiveness) = effectiveness {
                hit.push((target, effectiveness));
            } else {
                failed_otherwise = true;
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
        let mut missed_accuracy = false;
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
            if self.rng.below(100) < accuracy {
                true
            } else {
                missed_accuracy = true;
                false
            }
        });
        // `moves:disable.onTryHit` is a *move-owned* callback: the reference
        // runs it inside `spreadMoveHit` after the accuracy step, so the roll
        // happens even when the move then does nothing. A target with no
        // recorded last move (or a Struggle / Z / Max last move) is refused.
        if hooks & crate::effects::hook::DISABLE_TARGET_GATE != 0 {
            let before = hit.len();
            hit.retain(|(target, _)| {
                let last = self.mon(*target).last_move;
                last != 0
                    && last != dex.effects.struggle
                    && !dex.moves[last as usize].is_z
                    && !dex.moves[last as usize].is_max
            });
            failed_otherwise |= hit.len() != before;
        }
        failed_otherwise |= missed_accuracy;
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
            let base_power = self.base_power(dex, m.bp_callback, u32::from(m.power), actor, target, 1);
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
                        && self.effective_types(dex, actor).contains(&m.move_type)
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
        // Reference `hitStepMoveHitLoop` passes `hurtThisTurn + curDamage`,
        // i.e. each damaged target's HP before this move's damage, into the
        // Emergency Exit check at the end of the action.
        let mut hit_before: SmallVec<[(Entity, u16); 4]> = SmallVec::new();
        for (target, damage) in damages {
            // `endure` clamps after item/berry damage modification and before
            // the damage is applied (reference `onDamage` priority -10).
            let damage = self.sturdy_clamp(dex, target, damage)?;
            let damage = self.damage_item(dex, target, damage)?;
            let damage = self.endure_clamp(dex, target, damage);
            let hp_before = self.mon(target).hp;
            hit_before.push((target, hp_before));
            let actual = damage.min(hp_before);
            total_damage += u32::from(actual);
            hit_any = true;
            self.mon_mut(target).hp -= actual;
            // Reference `hitStepMoveHitLoop`: a landed hit increments the
            // target's `timesAttacked`, even when it dealt zero damage.
            if target != actor {
                let count = self.mon(target).times_attacked;
                self.mon_mut(target).times_attacked = count.saturating_add(1);
            }
            if self.mon(target).hp == 0 {
                self.faint_queue.push(FaintData {
                    target,
                    source: Some(actor),
                    from_move: true,
                });
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
                // `moves:afteryou.onHit`: the ally's queued move action jumps
                // to the front of the queue; a target with no queued move
                // makes the move fail.
                if hooks & crate::effects::hook::AFTER_YOU != 0 {
                    if let Some(index) = self.queue.iter().position(|q| {
                        q.kind == QueuedKind::Move && q.actor == Some(target)
                    }) {
                        let mut action = self.queue.remove(index);
                        action.priority.order = 3;
                        self.queue.insert(0, action);
                        did_anything = true;
                    }
                    continue;
                }
                // `moves:psychup.onHit`: the user copies every boost stage of
                // the target (the crit-stage volatiles it also copies cannot
                // exist natively, so only the stages matter).
                if hooks & crate::effects::hook::PSYCH_UP != 0 {
                    for stat in 0..7usize {
                        let want = self.mon(target).boosts[stat];
                        let old = self.mon(actor).boosts[stat];
                        if want == old {
                            continue;
                        }
                        self.mon_mut(actor).boosts[stat] = want;
                        self.emit(
                            EventKind::Boost,
                            actor,
                            Some(target),
                            EffectRef::Stat(stat as Id),
                            i32::from(want - old),
                            false,
                        )?;
                    }
                    did_anything = true;
                    continue;
                }
                // `moves:soak.onHit`: pure-Water targets refuse; anything else
                // is overwritten with pure Water.
                if hooks & crate::effects::hook::SOAK != 0
                    && self.effective_types(dex, target).as_slice() != [dex.effects.water]
                {
                    did_anything |= self.set_type(dex, target, &[dex.effects.water])?;
                } else if hooks & crate::effects::hook::SOAK == 0 {
                    did_anything |= self.hit_effect(dex, target, actor, &m.hit, false)?;
                }
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
        // `moves:doubleshock.self.onHit`: a landed Double Shock strips every
        // Electric type from the user (Pawmot keeps Fighting). A typeless
        // result is unreachable in the pinned regulation and fails closed.
        if hooks & crate::effects::hook::DOUBLE_SHOCK != 0 && did_anything {
            // The reference maps every Electric slot to the `'???'` placeholder
            // instead of dropping it; the native spells that placeholder as
            // type id 0 (the empty catalogue row), which is neutral on both
            // sides of the type chart exactly like `'???'`.
            let mapped: SmallVec<[Id; 4]> = self
                .effective_types(dex, actor)
                .into_iter()
                .map(|kind| if kind == dex.effects.electric { 0 } else { kind })
                .collect();
            self.set_type(dex, actor, &mapped)?;
        }
        // `moves:clangoroussoul.onHit`: once the five-stat self boost applied,
        // the user pays a third of its maximum HP as direct damage (no Damage
        // event, so Magic Guard cannot refuse it).
        if hooks & crate::effects::hook::CLANGOROUS_SOUL != 0 && did_anything {
            let amount = (u32::from(self.mon(actor).stats[0]) * 33 / 100).max(1);
            let actual = amount.min(u32::from(self.mon(actor).hp));
            if actual > 0 {
                self.mon_mut(actor).hp -= actual as u16;
                if self.mon(actor).hp == 0 {
                    self.faint_queue.push(FaintData {
                        target: actor,
                        source: Some(actor),
                        from_move: true,
                    });
                }
                self.emit(
                    EventKind::Damage,
                    actor,
                    Some(actor),
                    EffectRef::Move(move_id),
                    -(actual as i32),
                    true,
                )?;
            }
        }
        // Reference `trySpreadMoveHit` result: `true` when at least one target
        // survived every hit step (a status move also needs its effect to have
        // applied), `null` when protection was the only refusal, `false`
        // otherwise. Stomping Tantrum's callback reads the rolled value.
        let landed = !hit_targets.is_empty() && (m.category != Category::Status || did_anything);
        self.mon_mut(actor).move_this_turn_result = if landed {
            MoveResult::Success
        } else if blocked_by_protection && !failed_otherwise {
            MoveResult::Skipped
        } else {
            MoveResult::Failed
        };
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
        // Reference `spreadMoveHit` removes non-connecting targets before
        // `selfDrops` runs, so a missed, blocked or immune move never applies
        // its `self` payload (Overheat keeps its Sp. Atk, Hyper Beam does not
        // set mustrecharge).
        if let Some(effect) = m.self_effect.as_ref().filter(|_| !m.sheer_force && landed) {
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
        // Reference `spreadMoveHit` (Champions): the user's own Emergency Exit
        // check runs right after the DamagingHit event, with the HP it had
        // before that event (Rough Skin-style recoil can drop it under half).
        let user_hp_before_damaging_hit = self.mon(actor).hp;
        self.damaging_hit(dex, actor, &hit_targets, m)?;
        if !hit_targets.is_empty() {
            self.emergency_exit_check(dex, actor, user_hp_before_damaging_hit)?;
        }
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
        // `abilities:pickpocket.onAfterMoveSecondary`: a contact hit against a
        // holder with no item and no pending switch steals the attacker's item
        // (`source.switchFlag === true` is strict, so a pivot's move-id flag
        // does not block the steal).
        if m.contact {
            for &target in &hit_targets {
                if target == actor
                    || dex.effects.abilities[self.mon(target).ability as usize]
                        != Ability::Pickpocket
                    || self.mon(target).item != 0
                    || self.mon(target).switch_flag.is_some()
                    || self.mon(target).plain_switch_flag
                    || self.mon(target).force_switch_flag
                    || self.mon(actor).plain_switch_flag
                {
                    continue;
                }
                let crate::battle::hooks::TakeOutcome::Taken(item) =
                    self.take_item_checked(dex, actor)?
                else {
                    continue;
                };
                let order = self.allocate_effect_order()?;
                self.mon_mut(target).item = item;
                self.mon_mut(target).item_effect_order = Some(order);
                self.reveal_ability(target)?;
                self.emit(
                    EventKind::Item,
                    target,
                    Some(actor),
                    EffectRef::Item(item),
                    0,
                    false,
                )?;
            }
        }
        // Reference `hitStepMoveHitLoop` tail: every damaged target that is
        // still alive checks Emergency Exit against its pre-move HP.
        for &(target, hp_before) in &hit_before {
            self.emergency_exit_check(dex, target, hp_before)?;
        }
        if total_damage > 0 {
            if behavior == MoveBehavior::Struggle {
                let recoil =
                    stats::round_fraction(u32::from(self.mon(actor).stats[0]), [1, 4]).max(1);
                let hp_before = self.mon(actor).hp;
                self.indirect_damage(dex, actor, actor, recoil, EffectRef::Move(move_id))?;
                self.emergency_exit_check(dex, actor, hp_before)?;
            } else if let Some(fraction) = m.recoil
                && dex.effects.abilities[self.mon(actor).ability as usize] != Ability::RockHead
            {
                let recoil = stats::round_fraction(total_damage, fraction).max(1);
                let hp_before = self.mon(actor).hp;
                self.indirect_damage(
                    dex,
                    actor,
                    actor,
                    recoil,
                    EffectRef::Condition(dex.effects.recoil),
                )?;
                self.emergency_exit_check(dex, actor, hp_before)?;
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
            let hp_before = self.mon(actor).hp;
            self.item_damage(dex, actor, actor, self.mon(actor).stats[0] / 10)?;
            self.emergency_exit_check(dex, actor, hp_before)?;
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
        parental_bond: bool,
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
        // Parental Bond overrides the move's own hit count with exactly two.
        let hit_count = if parental_bond {
            2
        } else {
            self.multihit_count(m)
        };
        let mut total_damage = 0u32;
        // Reference `hitStepMoveHitLoop` reads `hurtThisTurn + move.totalDamage`
        // for the end-of-action Emergency Exit checks, i.e. each damaged
        // target's HP before this move started resolving.
        let mut hit_before: SmallVec<[(Entity, u16); 4]> = SmallVec::new();
        // `multiaccuracy`: hits after the first roll accuracy again and the
        // first miss ends the remaining hits (Population Bomb, Triple Axel).
        let multi_accuracy = m.hooks & crate::effects::hook::MULTI_ACCURACY != 0;
        for hit in 1..=hit_count {
            if hit > 1
                && (self.mon(actor).hp == 0
                    || connected.iter().all(|(t, _)| self.mon(*t).hp == 0))
            {
                break;
            }
            let mut missed = false;
            for &(target, effectiveness) in &connected {
                if self.mon(target).hp == 0 {
                    continue;
                }
                if hit > 1
                    && multi_accuracy
                    && !self.roll_move_accuracy(dex, actor, target, m, m.accuracy.map(u16::from))
                {
                    missed = true;
                    break;
                }
                if hit == 1 {
                    hit_before.push((target, self.mon(target).hp));
                }
                let damage = self.resolve_hit_damage(
                    dex,
                    actor,
                    target,
                    m,
                    effectiveness,
                    HitPhase {
                        spread,
                        hit,
                        parental_bond_second_hit: parental_bond && hit == 2,
                    },
                )?;
                let damage = self.sturdy_clamp(dex, target, damage)?;
                let damage = self.damage_item(dex, target, damage)?;
                let damage = self.endure_clamp(dex, target, damage);
                let actual = damage.min(self.mon(target).hp);
                total_damage += u32::from(actual);
                self.mon_mut(target).hp -= actual;
                if target != actor {
                    let count = self.mon(target).times_attacked;
                    self.mon_mut(target).times_attacked = count.saturating_add(1);
                }
                if self.mon(target).hp == 0 {
                    self.faint_queue.push(FaintData {
                        target,
                        source: Some(actor),
                        from_move: true,
                    });
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
                    // Reference `selfDrops` rolls only for a boosting self
                    // drop; a pure volatile self effect (mustrecharge) runs
                    // `moveHit` directly and draws nothing.
                    if effect.boosts.iter().any(|b| *b != 0) {
                        self.rng.below(100);
                    }
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
                let user_hp_before_damaging_hit = self.mon(actor).hp;
                self.damaging_hit(dex, actor, std::slice::from_ref(&target), m)?;
                self.emergency_exit_check(dex, actor, user_hp_before_damaging_hit)?;
            }
            if missed {
                break;
            }
            self.each_update(dex)?;
        }
        self.process_faints(dex, self.mon(actor).hp == 0)?;
        if self.outcome.terminated {
            return Ok(());
        }
        for &(target, hp_before) in &hit_before {
            self.emergency_exit_check(dex, target, hp_before)?;
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
            let hp_before = self.mon(actor).hp;
            self.indirect_damage(
                dex,
                actor,
                actor,
                recoil,
                EffectRef::Condition(dex.effects.recoil),
            )?;
            self.emergency_exit_check(dex, actor, hp_before)?;
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
            let hp_before = self.mon(actor).hp;
            self.item_damage(dex, actor, actor, self.mon(actor).stats[0] / 10)?;
            self.emergency_exit_check(dex, actor, hp_before)?;
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
        phase: HitPhase,
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
        let base_power =
            self.base_power(dex, m.bp_callback, u32::from(m.power), actor, target, phase.hit);
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
                spread: phase.spread,
                parental_bond_second_hit: phase.parental_bond_second_hit,
                weather_modifier: self.weather_damage_modifier(dex, m.move_type),
                critical,
                stab_modifier: if self.effective_types(dex, actor).contains(&m.move_type) {
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

    /// Reference `BattleQueue#changeAction` + `insertChoice`: replace the
    /// Pokémon's queued action with the encored move, re-resolving its target
    /// and re-inserting it in priority order. Both the target resolution and
    /// the insertion tie-break can consume reference RNG draws.
    fn change_action(&mut self, dex: &Dex, actor: Entity, move_id: Id) -> Result<()> {
        self.queue.retain(|q| q.actor != Some(actor));
        let Some(slot) = self.mon(actor).moves.iter().position(|mv| mv.id == move_id) else {
            return Ok(());
        };
        self.update_speed(dex);
        let m = &dex.moves[move_id as usize];
        let mut action = QueuedAction {
            kind: QueuedKind::Move,
            actor: Some(actor),
            move_slot: slot as u8,
            move_id,
            target_location: 0,
            destination: NO_SLOT,
            priority: Priority {
                order: 200,
                priority: i32::from(self.effective_priority(dex, actor, move_id)) * 10000,
                speed: self.speed(dex, actor),
                ..Default::default()
            },
        };
        // `resolveAction`: an action without a chosen location samples a random
        // valid target; `getActionSpeed` then resolves it again exactly as the
        // commit-time queue builder does.
        action.target_location = self.random_target_location(actor, m.target);
        self.resolve_target_location(actor, m.target, action.target_location);
        let mut first = None;
        let mut last = None;
        for (index, current) in self.queue.iter().enumerate() {
            let compared = action.priority.compare(&current.priority);
            if compared != Ordering::Greater && first.is_none() {
                first = Some(index);
            }
            if compared == Ordering::Less {
                last = Some(index);
                break;
            }
        }
        match first {
            None => self.queue.push(action),
            Some(first) => {
                let last = last.unwrap_or(self.queue.len());
                let index = if first == last {
                    first
                } else {
                    self.rng.range(first as u32, last as u32 + 1) as usize
                };
                self.queue.insert(index, action);
            }
        }
        Ok(())
    }

    fn before_move(
        &mut self,
        dex: &Dex,
        e: Entity,
        m: &crate::assets::Move,
    ) -> Result<Option<MoveResult>> {
        // `moves:glaiverush.condition.onBeforeMovePriority: 100`: the drawback
        // volatile is removed before every other BeforeMove handler.
        if self.mon(e).volatiles.contains_key(&dex.effects.glaive_rush) {
            self.mon_mut(e).volatiles.remove(&dex.effects.glaive_rush);
            self.emit(
                EventKind::EffectEnd,
                e,
                None,
                EffectRef::Condition(dex.effects.glaive_rush),
                0,
                false,
            )?;
        }
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
            // `mustrecharge.onBeforeMove` returns null: the reference records
            // the skipped attempt without marking it as a failure.
            return Ok(Some(MoveResult::Skipped));
        }
        let status = self.mon(e).status;
        if status == dex.effects.sleep || (status == dex.effects.freeze && !m.defrost) {
            self.mon_mut(e).status_state.values[0] -= 1;
            let expired = self.mon(e).status_state.values[0] <= 0;
            if expired || (status == dex.effects.freeze && self.rng.chance(1, 4)) {
                self.cure_status(e)?;
            } else {
                return Ok(Some(MoveResult::Failed));
            }
        }
        if self.mon(e).volatiles.contains_key(&dex.effects.flinch) {
            return Ok(Some(MoveResult::Failed));
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
            return Ok(Some(MoveResult::Failed));
        }
        // `moves:disable.condition.onBeforeMove` (priority 7).
        if let Some(state) = self.mon(e).volatiles.get(&dex.effects.disable)
            && state.values.first() == Some(&i64::from(m.id))
        {
            return Ok(Some(MoveResult::Failed));
        }
        // `moves:taunt.condition.onBeforeMove` (priority 5): Status moves are
        // refused outright, with Me First exempt.
        if self.mon(e).volatiles.contains_key(&dex.effects.taunt)
            && m.category == Category::Status
            && m.id != dex.effects.me_first
        {
            return Ok(Some(MoveResult::Failed));
        }
        // `moves:imprison.condition.onFoeBeforeMove` (priority 4): a foe's
        // Imprison refuses any non-Struggle move the imprisoning Pokémon knows.
        if m.id != dex.effects.struggle
            && self.active_entities(false).into_iter().any(|foe| {
                foe.side != e.side
                    && self.mon(foe).volatiles.contains_key(&dex.effects.imprison)
                    && self
                        .mon(foe)
                        .moves
                        .iter()
                        .any(|mv| mv.id == m.id)
            })
        {
            return Ok(Some(MoveResult::Failed));
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
                    return Ok(Some(MoveResult::Failed));
                }
            }
        }
        if status == dex.effects.paralysis && self.rng.chance(1, 8) {
            return Ok(Some(MoveResult::Failed));
        }
        Ok(None)
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
            self.faint_queue.push(FaintData {
                target: e,
                source: Some(e),
                from_move: false,
            });
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
        self.faint_queue.push(FaintData {
            target: e,
            source: Some(e),
            from_move: true,
        });
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
            } else if volatile == dex.effects.encore
                || volatile == dex.effects.taunt
                || volatile == dex.effects.disable
                || volatile == dex.effects.imprison
                || volatile == dex.effects.torment
                || volatile == dex.effects.yawn
                || volatile == dex.effects.roost
                || volatile == dex.effects.glaive_rush
                || volatile == dex.effects.partially_trapped
                || volatile == dex.effects.leech_seed
            {
                changed |= self.start_selection_volatile(
                    dex,
                    target,
                    Some(source),
                    volatile,
                    false,
                )?;
            } else {
                return Err(EngineError::Unsupported(format!("volatile {volatile}")));
            }
        }
        Ok(changed
            || (!has_boost && effect.heal.is_none() && effect.status == 0 && effect.volatile == 0))
    }

    /// Reference `onStart` for the volatile selection-lock family (Encore,
    /// Taunt, Disable, Imprison, Torment). Returns the `addVolatile` result:
    /// `false` when the reference refuses the state and the move reports
    /// failure, `true` when the volatile was created.
    ///
    /// `mid_move` is the reference's
    /// `pokemon === this.activePokemon && this.activeMove && !isExternal`
    /// branch, which Cursed Body triggers while the attacker's move is active.
    fn start_selection_volatile(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Option<Entity>,
        volatile: Id,
        mid_move: bool,
    ) -> Result<bool> {
        // `addVolatile` fails when the volatile already exists and declares no
        // `onRestart`; none of this family restarts.
        if self.mon(target).volatiles.contains_key(&volatile) {
            return Ok(false);
        }
        let source_slot = source.map(|e| {
            (
                if e.side == 0 { SideId::P1 } else { SideId::P2 },
                e.roster,
            )
        });
        let will_move = self
            .queue
            .iter()
            .any(|q| q.kind == QueuedKind::Move && q.actor == Some(target));
        if volatile == dex.effects.encore {
            let last = self.mon(target).last_move;
            if last == 0 {
                return Ok(false);
            }
            let Some(slot) = self.mon(target).moves.iter().position(|mv| mv.id == last) else {
                return Ok(false);
            };
            if self.mon(target).moves[slot].pp == 0
                || dex.moves[last as usize].fail_encore
                || dex.moves[last as usize].is_z
                || dex.moves[last as usize].is_max
            {
                return Ok(false);
            }
            let mut duration = 3u16;
            let queued = self
                .queue
                .iter()
                .find(|q| q.kind == QueuedKind::Move && q.actor == Some(target))
                .map(|q| q.move_id);
            if queued.is_none() {
                duration += 1;
            } else if queued != Some(last) && self.mon(target).item != dex.effects.mental_herb {
                // Champions Encore replaces the target's queued action with the
                // encored move, which re-resolves its target and re-inserts the
                // action in speed order (both can consume RNG).
                self.change_action(dex, target, last)?;
            }
            let order = self.allocate_effect_order()?;
            self.mon_mut(target).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    duration: Some(duration),
                    source: source_slot,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: vec![i64::from(last)],
                },
            );
            self.emit(
                EventKind::EffectStart,
                target,
                source,
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            return Ok(true);
        }
        if volatile == dex.effects.taunt {
            // `onStart`: an already-active Pokémon that has not queued an action
            // this turn keeps the volatile one turn longer.
            let mut duration = 3u16;
            if self.mon(target).active_turns > 0 && !will_move {
                duration += 1;
            }
            let order = self.allocate_effect_order()?;
            self.mon_mut(target).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    duration: Some(duration),
                    source: source_slot,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: vec![],
                },
            );
            self.emit(
                EventKind::EffectStart,
                target,
                source,
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            return Ok(true);
        }
        if volatile == dex.effects.disable {
            // The duration drops one tick when the target has not acted yet
            // this turn, or when Cursed Body fires during the attacker's move.
            let mut duration = 5u16;
            if will_move || mid_move {
                duration -= 1;
            }
            let last = self.mon(target).last_move;
            if last == 0 {
                return Ok(false);
            }
            let Some(slot) = self.mon(target).moves.iter().position(|mv| mv.id == last) else {
                return Ok(false);
            };
            if self.mon(target).moves[slot].pp == 0 {
                return Ok(false);
            }
            let order = self.allocate_effect_order()?;
            self.mon_mut(target).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    duration: Some(duration),
                    source: source_slot,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: vec![i64::from(last)],
                },
            );
            self.emit(
                EventKind::EffectStart,
                target,
                source,
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            return Ok(true);
        }
        if volatile == dex.effects.leech_seed {
            // `moves:leechseed.condition`: no duration; the residual drains the
            // holder into the recorded slot. Grass-type targets are refused by
            // the move's `onTryImmunity` before this point.
            if self.mon(target).types.contains(&dex.effects.grass) {
                return Ok(false);
            }
            let order = self.allocate_effect_order()?;
            self.mon_mut(target).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    source: source_slot,
                    effect_order: order,
                    effect_order_assigned: true,
                    ..Default::default()
                },
            );
            self.emit(
                EventKind::EffectStart,
                target,
                source,
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            return Ok(true);
        }
        if volatile == dex.effects.partially_trapped {
            // `partiallytrapped.durationCallback`: a 5-or-6 turn bind (Grip Claw
            // is not a legal item). `boundDivisor` is 8 without Binding Band.
            let duration = self.rng.range(5, 7) as u16;
            let order = self.allocate_effect_order()?;
            self.mon_mut(target).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    duration: Some(duration),
                    source: source_slot,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: vec![8],
                },
            );
            self.emit(
                EventKind::EffectStart,
                target,
                source,
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            return Ok(true);
        }
        // `moves:yawn.condition` (2 turns, residual order 23) and
        // `moves:roost.condition` (1 turn, residual order 25) are the only
        // ported volatiles that carry a numeric duration without their own
        // rest-of-family handling.
        if volatile == dex.effects.yawn || volatile == dex.effects.roost {
            let duration = if volatile == dex.effects.yawn { 2 } else { 1 };
            let order = self.allocate_effect_order()?;
            self.mon_mut(target).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    duration: Some(duration),
                    source: source_slot,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: vec![],
                },
            );
            self.emit(
                EventKind::EffectStart,
                target,
                source,
                EffectRef::Condition(volatile),
                0,
                false,
            )?;
            return Ok(true);
        }
        // Imprison and Torment have no duration; their `onStart` only records
        // the state (and the source, which Imprison's foe-side handlers read).
        let order = self.allocate_effect_order()?;
        self.mon_mut(target).volatiles.insert(
            volatile,
            EffectState {
                id: volatile,
                source: source_slot,
                effect_order: order,
                effect_order_assigned: true,
                ..Default::default()
            },
        );
        self.emit(
            EventKind::EffectStart,
            target,
            source,
            EffectRef::Condition(volatile),
            0,
            false,
        )?;
        Ok(true)
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
        let mut last: Option<FaintData> = None;
        let length = self.faint_queue.len();
        for data in std::mem::take(&mut self.faint_queue) {
            let e = data.target;
            if self.mon(e).fainted {
                continue;
            }
            self.emit(EventKind::Faint, e, None, EffectRef::None, 0, true)?;
            self.ability_end(dex, e)?;
            self.clear_volatile(dex, e);
            self.mon_mut(e).fainted = true;
            last = Some(data);
        }
        if check_win && last.is_some() && self.check_win(last.map(|data| data.target)) {
            return Ok(());
        }
        // Reference `faintMessages` tail: one AfterFaint event per batch, with
        // the last faint's source and the queue length as the relay value.
        if let Some(data) = last {
            self.after_faint(dex, data.source, data.from_move, length)?;
        }
        Ok(())
    }

    /// Reference `runEvent('AfterFaint', target, source, effect, length)`:
    /// the faint's source checks its own `onSourceAfterFaint` handler. Only
    /// move-caused faints run it, and only while the source is still on the
    /// field; the handler list holds at most one ability, so no RNG is drawn.
    fn after_faint(
        &mut self,
        dex: &Dex,
        source: Option<Entity>,
        from_move: bool,
        length: usize,
    ) -> Result<()> {
        let Some(source) = source else {
            return Ok(());
        };
        if !from_move
            || self.mon(source).hp == 0
            || self.mon(source).active_slot.is_none()
            || length == 0
        {
            return Ok(());
        }
        let ability = dex.effects.abilities[self.mon(source).ability as usize];
        let mut changes = [0i8; 7];
        match ability {
            // `abilities:eelevate.onSourceAfterFaint`: boost the source's best
            // stat by the number of fainted Pokémon (first stat wins ties).
            Ability::Eelevate => {
                // `stats` is [hp, atk, def, spa, spd, spe] while `boosts` is
                // [atk, def, spa, spd, spe], so the chosen stat maps down one.
                let mut best = 1usize;
                for index in 2..=5 {
                    if self.mon(source).stats[index] > self.mon(source).stats[best] {
                        best = index;
                    }
                }
                changes[best - 1] = length.min(6) as i8;
            }
            // `abilities:moxie.onSourceAfterFaint`: Attack rises by the count.
            Ability::Moxie => changes[0] = length.min(6) as i8,
            _ => return Ok(()),
        }
        self.boost(
            dex,
            source,
            source,
            changes,
            BoostCause::Ability(ability),
        )?;
        Ok(())
    }

    fn check_win(&mut self, last_faint: Option<Entity>) -> bool {
        if self.outcome.terminated {
            return false;
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
            return true;
        }
        false
    }

    fn residual(&mut self, dex: &Dex) -> Result<()> {
        let dbg = std::env::var("PA3_RNG_DBG").is_ok();
        if dbg {
            eprintln!("RNG residual start draws {}", self.rng.draws);
        }
        // Reference `runAction` captures each active Pokémon's HP before the
        // residual `fieldEvent` and checks Emergency Exit against that value
        // once the phase finishes.
        let residual_before: SmallVec<[(Entity, u16); 4]> = self
            .active_entities(false)
            .into_iter()
            .map(|e| (e, self.mon(e).hp))
            .collect();
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
            // `abilities:moody.onResidual` shares Speed Boost's order/sub-order.
            if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Moody {
                handlers.push((
                    e,
                    self.mon(e).ability,
                    13,
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
                // Timed volatiles tick in this sweep; Leech Seed is the one
                // duration-less volatile with its own residual handler.
                if state.duration.is_some() || id == dex.effects.leech_seed {
                    // Reference `onResidualOrder`: Taunt 15, Encore 16, Disable
                    // 17, Throat Chop 22; other timed volatiles stay unordered.
                    let (order, sub_order) = if id == dex.effects.taunt {
                        (15, 0)
                    } else if id == dex.effects.encore {
                        (16, 0)
                    } else if id == dex.effects.disable {
                        (17, 0)
                    } else if id == dex.effects.throat_chop {
                        (22, 0)
                    } else if id == dex.effects.yawn {
                        (23, 0)
                    } else if id == dex.effects.roost {
                        (25, 0)
                    } else if id == dex.effects.partially_trapped {
                        (13, 0)
                    } else if id == dex.effects.perish_song {
                        (24, 0)
                    } else if id == dex.effects.leech_seed {
                        (8, 0)
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
                } else if id == dex.effects.toxic_spikes {
                    // Entry hazards carry no residual handler at all: their
                    // `duration` field stores the layer count for the fixture
                    // contract, so they must never join the timed sweep.
                    continue;
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
            if status == 0 && id == dex.effects.leech_seed {
                // `moves:leechseed.condition.onResidual` (order 8): drain an
                // eighth of the holder's maximum HP into the seeding slot.
                let Some(state) = self.mon(e).volatiles.get(&id) else {
                    continue;
                };
                let Some((side, roster)) = state.source else {
                    continue;
                };
                let source = Entity {
                    side: side.index() as u8,
                    roster,
                };
                if self.mon(source).fainted || self.mon(source).hp == 0 {
                    continue;
                }
                let amount = (u32::from(self.mon(e).stats[0]) / 8).max(1);
                let actual = amount.min(u32::from(self.mon(e).hp)) as u16;
                if actual == 0 {
                    continue;
                }
                self.mon_mut(e).hp -= actual;
                if self.mon(e).hp == 0 {
                    self.faint_queue.push(FaintData {
                        target: e,
                        source: Some(source),
                        from_move: false,
                    });
                }
                self.emit(
                    EventKind::Damage,
                    e,
                    Some(source),
                    EffectRef::Condition(id),
                    -i32::from(actual),
                    true,
                )?;
                // `this.heal(damage, target, pokemon)`: a plain heal that fails
                // silently at full HP.
                let room = self.mon(source).stats[0] - self.mon(source).hp;
                let healed = actual.min(room);
                if healed > 0 {
                    self.mon_mut(source).hp += healed;
                    self.emit(
                        EventKind::Heal,
                        source,
                        Some(e),
                        EffectRef::Condition(id),
                        i32::from(healed),
                        true,
                    )?;
                }
                continue;
            }
            if status == 0 && id == dex.effects.partially_trapped {
                // `moves:partiallytrapped.condition.onResidual` (order 13): the
                // bind ends silently when its source left the field or has not
                // acted yet, otherwise it deals `baseMaxhp / boundDivisor`.
                // An earlier handler in this sweep may have removed the
                // volatile already (the target fainted or switched out).
                let Some(state) = self.mon(e).volatiles.get(&id) else {
                    continue;
                };
                let source = state.source.map(|(side, roster)| Entity {
                    side: side.index() as u8,
                    roster,
                });
                let keep = source.is_some_and(|source| {
                    self.mon(source).active_slot.is_some()
                        && self.mon(source).hp > 0
                        && self.mon(source).active_turns > 0
                });
                if !keep {
                    self.mon_mut(e).volatiles.remove(&id);
                    self.emit(
                        EventKind::EffectEnd,
                        e,
                        None,
                        EffectRef::Condition(id),
                        0,
                        false,
                    )?;
                    continue;
                }
                let divisor = self.mon(e).volatiles[&id]
                    .values
                    .first()
                    .copied()
                    .filter(|value| *value > 0)
                    .unwrap_or(8) as u32;
                let amount = (u32::from(self.mon(e).stats[0]) / divisor).max(1);
                if let Some(source) = source {
                    self.indirect_damage(
                        dex,
                        e,
                        source,
                        amount,
                        EffectRef::Condition(id),
                    )?;
                }
                continue;
            }
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
            } else if status == 13 {
                // `abilities:moody.onResidual`: sample one stat below +6 to
                // raise by two, then a different stat above -6 to drop by one.
                // Both samples draw even when their pool holds one entry.
                if self.mon(e).ability != id || self.mon(e).hp == 0 {
                    continue;
                }
                let boosts = self.mon(e).boosts;
                let raisable: SmallVec<[u8; 5]> = (0..5u8)
                    .filter(|index| boosts[*index as usize] < 6)
                    .collect();
                let mut changes = [0i8; 7];
                let raised = if raisable.is_empty() {
                    None
                } else {
                    Some(raisable[self.rng.below(raisable.len() as u32) as usize])
                };
                if let Some(raised) = raised {
                    changes[raised as usize] = 2;
                }
                let lowerable: SmallVec<[u8; 5]> = (0..5u8)
                    .filter(|index| {
                        boosts[*index as usize] > -6 && Some(*index) != raised
                    })
                    .collect();
                if !lowerable.is_empty() {
                    let lowered = lowerable[self.rng.below(lowerable.len() as u32) as usize];
                    changes[lowered as usize] = -1;
                }
                self.reveal_ability(e)?;
                self.boost(dex, e, e, changes, BoostCause::Ability(Ability::Moody))?;
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
                    let source = self.mon(e).status_state.source.map(|(side, roster)| Entity {
                        side: side.index() as u8,
                        roster,
                    });
                    self.faint_queue.push(FaintData {
                        target: e,
                        source,
                        from_move: false,
                    });
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
                // `moves:encore.condition.onResidual` (order 16): the volatile
                // ends early when the encored move is gone or out of PP.
                if id == dex.effects.encore {
                    let move_id = self
                        .mon(e)
                        .volatiles
                        .get(&id)
                        .and_then(|state| state.values.first())
                        .copied();
                    let lost = move_id.is_none_or(|move_id| {
                        !self
                            .mon(e)
                            .moves
                            .iter()
                            .any(|mv| i64::from(mv.id) == move_id && mv.pp > 0)
                    });
                    if lost {
                        self.mon_mut(e).volatiles.remove(&id);
                        self.emit(
                            EventKind::EffectEnd,
                            e,
                            None,
                            EffectRef::Condition(id),
                            0,
                            false,
                        )?;
                        continue;
                    }
                }
                let expired = if let Some(state) = self.mon_mut(e).volatiles.get_mut(&id) {
                    let duration = state.duration.as_mut().unwrap();
                    *duration = duration.saturating_sub(1);
                    *duration == 0
                } else {
                    false
                };
                if expired {
                    let yawn_source = self.mon(e).volatiles.get(&id).and_then(|state| {
                        state.source.map(|(side, roster)| Entity {
                            side: side.index() as u8,
                            roster,
                        })
                    });
                    self.mon_mut(e).volatiles.remove(&id);
                    if id == dex.effects.yawn {
                        // `moves:yawn.condition.onEnd`: the target falls asleep
                        // from the recorded source once the counter runs out.
                        let effect = crate::effects::HitEffect {
                            status: dex.effects.sleep,
                            ..Default::default()
                        };
                        if let Some(source) = yawn_source {
                            self.hit_effect(dex, e, source, &effect, false)?;
                        }
                    }
                    if id == dex.effects.perish_song {
                        // `moves:perishsong.condition.onEnd`: the counter
                        // reaching zero faints the holder.
                        self.faint_now(e);
                    }
                    if id == dex.effects.protect
                        || id == dex.effects.throat_chop
                        || id == dex.effects.taunt
                        || id == dex.effects.encore
                        || id == dex.effects.disable
                        || id == dex.effects.torment
                        || id == dex.effects.yawn
                        || id == dex.effects.roost
                    {
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
        // Reference `runAction` end-of-turn switch checks: each active Pokémon
        // checks Emergency Exit against its pre-residual HP.
        for &(target, hp_before) in &residual_before {
            self.emergency_exit_check(dex, target, hp_before)?;
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

    /// Reference request-time `TrapPokemon` / `MaybeTrapPokemon` pass for one
    /// active Pokémon. Returns the holder's trap state as `Some(hidden)` when a
    /// trapping effect holds it (Shadow Tag/Arena Trap/Magnet Pull mark hidden
    /// traps, a live `partiallytrapped` source a real one) and whether any
    /// foe's trapping ability merely *might* hold it. Ghost types are immune
    /// through the `trapped` pseudo-type, and a Shadow Tag holder ignores
    /// another Shadow Tag. In doubles every active is adjacent, so the
    /// reference's adjacency filter never excludes a foe.
    pub(crate) fn trap_flags(&self, dex: &Dex, e: Entity) -> (Option<bool>, bool) {
        // The pinned `trapped` pseudo-type only marks Ghost types immune.
        let immune = self.mon(e).types.contains(&dex.effects.ghost);
        let mut trapped: Option<bool> = None;
        let mut maybe = false;
        if immune {
            return (None, false);
        }
        // `moves:partiallytrapped.condition.onTrapPokemon`: the volatile's
        // source must still be on the field.
        if self
            .mon(e)
            .volatiles
            .contains_key(&dex.effects.partially_trapped)
        {
            let source_active = self.mon(e).volatiles[&dex.effects.partially_trapped]
                .source
                .is_some_and(|(side, roster)| {
                    let source = Entity {
                        side: side.index() as u8,
                        roster,
                    };
                    self.mon(source).hp > 0 && self.mon(source).active_slot.is_some()
                });
            if source_active {
                trapped = Some(false);
            }
        }
        let ability = dex.effects.abilities[self.mon(e).ability as usize];
        for roster in self.sides[(1 - e.side) as usize].active.iter().flatten() {
            let foe = Entity {
                side: 1 - e.side,
                roster: *roster,
            };
            if self.mon(foe).hp == 0 {
                continue;
            }
            match dex.effects.abilities[self.mon(foe).ability as usize] {
                Ability::Shadowtag if ability != Ability::Shadowtag => {
                    trapped = Some(true);
                    maybe = true;
                }
                Ability::Arenatrap if self.grounded(dex, e) => {
                    trapped = Some(true);
                    maybe = true;
                }
                Ability::Magnetpull if self.mon(e).types.contains(&dex.effects.steel) => {
                    trapped = Some(true);
                    maybe = true;
                }
                _ => (),
            }
        }
        (trapped, maybe)
    }

    /// Reference `abilities:emergencyexit.onEmergencyExit` (the pinned
    /// Champions override): the holder marks itself to leave the field when a
    /// single damage event drops it from above half HP to half HP or below,
    /// provided the side still has a reserve and no switch is already pending.
    /// `original_hp` is the relay value the reference passes — the holder's HP
    /// before the damage event being resolved. The handler list holds at most
    /// one handler in the pinned data (the ability itself), so the reference's
    /// speed sort consumes no RNG; only the flag and the reveal are visible.
    fn emergency_exit_check(
        &mut self,
        dex: &Dex,
        target: Entity,
        original_hp: u16,
    ) -> Result<()> {
        if dex.effects.abilities[self.mon(target).ability as usize] != Ability::Emergencyexit {
            return Ok(());
        }
        let half = self.mon(target).stats[0] / 2;
        if self.mon(target).hp == 0
            || self.mon(target).hp > half
            || original_hp <= half
            || self.mon(target).force_switch_flag
            || self.mon(target).switch_flag.is_some()
            || self.mon(target).plain_switch_flag
            || !self.can_switch(target.side as usize)
        {
            return Ok(());
        }
        self.mon_mut(target).plain_switch_flag = true;
        self.reveal_ability(target)?;
        Ok(())
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
                .any(|r| {
                    let p = &self.sides[side].pokemon[*r as usize];
                    p.switch_flag.is_some() || p.plain_switch_flag
                });
            if !flagged {
                continue;
            }
            if self.can_switch(side) {
                *needed = true;
            } else {
                for r in self.sides[side].active.iter().flatten() {
                    self.sides[side].pokemon[*r as usize].switch_flag = None;
                    self.sides[side].pokemon[*r as usize].plain_switch_flag = false;
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
                    requires_replacement: needed
                        && (p.switch_flag.is_some() || p.plain_switch_flag),
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
                            hidden: mv.hidden,
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

    /// Reference `runEvent('DisableMove', pokemon)` handler set for one active
    /// Pokémon: the holder's status, volatiles (declaration order), ability,
    /// item, species and slot conditions, plus each live active foe's
    /// `onFoeDisableMove`. Only the entries the pinned data declares exist
    /// (`dex.effects.disable_move_*`, validated at load); the priority fields
    /// are exactly `resolvePriority`'s order/priority/speed/subOrder, so the
    /// caller's `speed_sort` consumes the reference tie shuffles.
    fn disable_move_handlers(&self, dex: &Dex, e: Entity) -> SmallVec<[Priority; 4]> {
        let mon = &self.sides[e.side as usize].pokemon[e.roster as usize];
        let mut handlers: SmallVec<[Priority; 4]> = SmallVec::new();
        let mut push = |sub_order: i32, speed: i32| {
            handlers.push(Priority {
                sub_order,
                speed,
                ..Default::default()
            });
        };
        if let Some(&sub_order) = dex.effects.disable_move_conditions.get(&mon.status) {
            push(sub_order, mon.cached_speed);
        }
        for id in mon.volatiles.keys() {
            if let Some(&sub_order) = dex.effects.disable_move_conditions.get(id) {
                push(sub_order, mon.cached_speed);
            }
        }
        if let Some(&sub_order) = dex.effects.disable_move_abilities.get(&mon.ability) {
            push(sub_order, mon.cached_speed);
        }
        if let Some(&sub_order) = dex.effects.disable_move_items.get(&mon.item) {
            push(sub_order, mon.cached_speed);
        }
        // Reference `findEventHandlers` collects the prefixed foe handlers
        // after the target's own; `foes()` keeps only live actives.
        for slot in self.sides[(1 - e.side) as usize].active.iter().flatten() {
            let foe = &self.sides[(1 - e.side) as usize].pokemon[*slot as usize];
            if foe.hp == 0 {
                continue;
            }
            for id in foe.volatiles.keys() {
                if let Some(&sub_order) = dex.effects.foe_disable_move_conditions.get(id) {
                    push(sub_order, foe.cached_speed);
                }
            }
        }
        handlers
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
        // Reference `endTurn` runs `runEvent('DisableMove', pokemon)` for every
        // active Pokémon before the flags below are applied; the handler list
        // is speed-sorted, so a fully tied set (Taunt + Encore + Disable on one
        // holder) consumes a shuffle draw. The flag pass below stays
        // authoritative for state; this pass reproduces the collection and the
        // sort's visible draws in side/slot order.
        for side in 0..2 {
            for slot in 0..2 {
                let Some(roster) = self.sides[side].active[slot] else {
                    continue;
                };
                let mut handlers = self.disable_move_handlers(
                    dex,
                    Entity {
                        side: side as u8,
                        roster,
                    },
                );
                speed_sort(&mut handlers, &mut self.rng, |p| *p);
            }
        }
        // Reference `onFoeDisableMove`: an active foe's Imprison hides every
        // move the imprisoning Pokémon also knows from this side's requests.
        // Collected before the mutable per-Pokémon pass.
        let imprisoned: [SmallVec<[Id; 8]>; 2] = std::array::from_fn(|side| {
            let foe = 1 - side;
            let mut moves: SmallVec<[Id; 8]> = SmallVec::new();
            for roster in self.sides[foe].active.iter().flatten() {
                let p = &self.sides[foe].pokemon[*roster as usize];
                if !p.fainted && p.volatiles.contains_key(&dex.effects.imprison) {
                    moves.extend(p.moves.iter().map(|mv| mv.id));
                }
            }
            moves
        });
        for (side, imprisoned_moves) in imprisoned.iter().enumerate() {
            for mon in &mut self.sides[side].pokemon {
                // Reference `makeRequest` only resets and re-applies disabled
                // move flags for Pokémon currently on the field; a benched
                // Pokémon keeps its frozen flags until it is active again.
                if mon.active_slot.is_none() {
                    continue;
                }
                // Reference turn-loop rollover: the previous attempt's result
                // becomes `moveLastTurnResult` for the new decision boundary.
                mon.move_last_turn_result = mon.move_this_turn_result;
                mon.move_this_turn_result = crate::state::MoveResult::Undefined;
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
                // `moves:encore.condition.onDisableMove`: every move except the
                // encored one is disabled while the holder still has it.
                let encore = mon
                    .volatiles
                    .get(&dex.effects.encore)
                    .and_then(|state| state.values.first())
                    .copied()
                    .filter(|id| mon.moves.iter().any(|mv| i64::from(mv.id) == *id));
                // `moves:disable.condition.onDisableMove` / `onEnd`.
                let disabled_move = mon
                    .volatiles
                    .get(&dex.effects.disable)
                    .and_then(|state| state.values.first())
                    .copied();
                // `moves:taunt.condition.onDisableMove`: Status moves only,
                // with Me First exempt.
                let taunted = mon.volatiles.contains_key(&dex.effects.taunt);
                // `moves:torment.condition.onDisableMove`: the last used move.
                let tormented = mon.volatiles.contains_key(&dex.effects.torment);
                let last_move = mon.last_move;
                for mv in &mut mon.moves {
                    let mut disabled = locked.is_some_and(|id| id != i64::from(mv.id))
                        || (fake_out_disabled
                            && (mv.id == dex.effects.fake_out
                                || mv.id == dex.effects.first_impression))
                        || (throat_chop && dex.moves[mv.id as usize].sound);
                    if let Some(id) = encore {
                        disabled |= i64::from(mv.id) != id;
                    }
                    if taunted
                        && dex.moves[mv.id as usize].category == Category::Status
                        && mv.id != dex.effects.me_first
                    {
                        disabled = true;
                    }
                    if disabled_move.is_some_and(|id| id == i64::from(mv.id)) {
                        disabled = true;
                    }
                    if tormented && last_move != 0 && mv.id == last_move {
                        disabled = true;
                    }
                    let hidden = imprisoned_moves.contains(&mv.id);
                    if hidden {
                        // `onFoeDisableMove` marks the move `'hidden'`. The
                        // served request only turns that into `disabled` for
                        // the side's last active Pokemon (`getMoves(lockedMove,
                        // restrictData = isLastActive())`), so the world flag and
                        // the served flag are tracked separately.
                        disabled = true;
                    }
                    mv.disabled = disabled;
                    mv.hidden = hidden;
                }
            }
            let slots = std::array::from_fn(|slot| {
                let Some(roster) = self.sides[side].active[slot] else {
                    return SlotRequest::default();
                };
                let p = &self.sides[side].pokemon[roster as usize];
                let last_active = (slot + 1..2).all(|later| {
                    self.sides[side].active[later]
                        .is_none_or(|r| self.sides[side].pokemon[r as usize].fainted)
                });
                // Reference `getLockedMove()`: `mustrecharge.onLockMove`
                // returns the Recharge pseudo-move, and `twoturnmove.onLockMove`
                // the charging move with the location recorded on its start.
                // A locked slot refuses switches and offers no Mega.
                let (locked_move, locked_recharge, locked_target_location) = p.locked_state(dex);
                let locked = locked_move.is_some() || locked_recharge;
                // Reference request data: `TrapPokemon`/`MaybeTrapPokemon` run
                // for every active Pokémon; only the side's last active slot
                // exposes `maybeTrapped`, and a hidden trap is served as
                // `trapped` for every other slot.
                let (trap_state, trap_maybe) = self.trap_flags(
                    dex,
                    Entity {
                        side: side as u8,
                        roster,
                    },
                );
                let can_switch_in = self.can_switch(side);
                let (slot_trapped, slot_maybe) = if locked {
                    // `getMoveRequestData` marks any locked Pokémon trapped.
                    (true, false)
                } else if last_active {
                    (
                        can_switch_in && trap_state == Some(false),
                        can_switch_in && trap_state != Some(false) && trap_maybe,
                    )
                } else {
                    (can_switch_in && trap_state.is_some(), false)
                };
                let moves = if locked_recharge {
                    Vec::new()
                } else if let Some(id) = locked_move {
                    p.moves
                        .iter()
                        .enumerate()
                        .filter(|(_, mv)| mv.id == id)
                        .map(|(slot, mv)| MoveChoice {
                            id: mv.id,
                            slot: slot as u8,
                            target: dex.moves[mv.id as usize].target,
                            disabled: false,
                            hidden: false,
                            pp: mv.pp,
                        })
                        .collect()
                } else {
                    p.moves
                        .iter()
                        .enumerate()
                        .map(|(slot, mv)| MoveChoice {
                            id: mv.id,
                            slot: slot as u8,
                            target: dex.moves[mv.id as usize].target,
                            disabled: mv.disabled,
                            hidden: mv.hidden,
                            pp: mv.pp,
                        })
                        .collect()
                };
                SlotRequest {
                    present: !p.fainted,
                    can_mega: !locked
                        && !p.fainted
                        && self
                            .mega_form(
                                dex,
                                Entity {
                                    side: side as u8,
                                    roster,
                                },
                            )
                            .is_some(),
                    moves,
                    trapped: slot_trapped,
                    maybe_trapped: slot_maybe,
                    locked_move,
                    locked_recharge,
                    locked_target_location,
                    last_active,
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
