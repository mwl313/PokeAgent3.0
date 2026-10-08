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

/// Caller-side flags for `hit_effect_with_ability`: whether the effect is a
/// secondary roll, the entity that owns the ability being applied (ability
/// sourced statuses reveal it), and whether the active move ignores the
/// target's breakable ability (Mold Breaker).
#[derive(Clone, Copy, Default)]
pub(super) struct HitContext {
    secondary: bool,
    ability_source: Option<Entity>,
    suppressing: bool,
}

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

/// How `use_move_inner` was entered. A chosen action runs `runMove`'s outer
/// phases; a nested `BattleActions#useMove` (Sleep Talk, Magic Bounce) skips
/// `BeforeMove`, PP deduction, `moveUsed` bookkeeping and the action counter,
/// and the Magic Bounce reflection additionally inherits the outer action's
/// priority and carries `hasBounced`.
#[derive(Default)]
pub(super) struct MoveUse<'a> {
    pub called: bool,
    pub bounced: bool,
    pub priority: Option<i8>,
    /// `move.sourceEffect`: the id of the effect that queued or called this
    /// action (only the Round chain sets it today).
    pub source_effect: Id,
    /// Move slot that pays a nested move's Pressure `DeductPP` cost: the
    /// caller's slot for a move called by another move, `NO_SLOT` for a
    /// reflection whose source effect is an ability rather than a move.
    pub caller_slot: u8,
    /// `BattleActions#useMove` was given an explicit `target`: the reference
    /// skips its `getRandomTarget` fallback entirely (one fewer draw for a
    /// spread class, whose target list is rebuilt from the move anyway).
    pub explicit_target: bool,
    /// Whether the attempt passed the PP gate and therefore fires the
    /// reference's `AfterMove` events (a `BeforeMove` refusal returns before
    /// them, so it must not tick the rampage lock).
    pub ran: Option<&'a mut bool>,
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
                        source_effect: 0,
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
                        let target = self.queued_target(dex, actor, m.id);
                        // Reference `resolveAction`: a move declaring
                        // `beforeTurnCallback` unshifts an order-5
                        // `beforeTurnMove` sub-action *before* the move action
                        // resolves its own target, and that sub-action samples
                        // its own random target when none was chosen.
                        let before_turn_move = matches!(
                            dex.effects.moves[m.id as usize],
                            MoveBehavior::Counter | MoveBehavior::MirrorCoat
                        );
                        let mut before_turn_location = action.target_location;
                        if before_turn_move && before_turn_location == 0 {
                            before_turn_location = self.random_target_location(actor, target);
                        }
                        if recharge_lock {
                            self.sample_random_foe(actor);
                        } else {
                            if queued.target_location == 0 {
                                queued.target_location =
                                    self.random_target_location(actor, target);
                            }
                            // getActionSpeed resolves a target even for a
                            // constant priority. That resolution can consume a
                            // reference RNG draw.
                            self.resolve_target_location(actor, target, queued.target_location);
                        }
                        if action.resource == Resource::Mega {
                            self.queue.push(QueuedAction {
                                kind: QueuedKind::Mega,
                                actor: Some(actor),
                                move_slot: NO_SLOT,
                                move_id: 0,
                                source_effect: 0,
                                target_location: 0,
                                destination: NO_SLOT,
                                priority: Priority {
                                    order: 104,
                                    speed: self.speed(dex, actor),
                                    ..Default::default()
                                },
                            });
                        }
                        // Reference `resolveAction`: a move declaring
                        // `priorityChargeCallback` queues a
                        // `priorityChargeMove` action (order 107) that runs the
                        // callback before any move of the turn.
                        if m.priority_charge {
                            self.queue.push(QueuedAction {
                                kind: QueuedKind::PriorityCharge,
                                actor: Some(actor),
                                move_slot: NO_SLOT,
                                move_id: queued.move_id,
                                source_effect: 0,
                                target_location: 0,
                                destination: NO_SLOT,
                                priority: Priority {
                                    order: 107,
                                    speed: self.speed(dex, actor),
                                    ..Default::default()
                                },
                            });
                        }
                        // Reference `BattleQueue#resolveAction`: a move
                        // declaring `beforeTurnCallback` queues a
                        // `beforeTurnMove` action (order 5) that runs its
                        // callback before every move of the turn.
                        if before_turn_move {
                            self.queue.push(QueuedAction {
                                kind: QueuedKind::BeforeTurnMove,
                                actor: Some(actor),
                                move_slot: NO_SLOT,
                                move_id: queued.move_id,
                                source_effect: 0,
                                target_location: before_turn_location,
                                destination: NO_SLOT,
                                priority: Priority {
                                    order: 5,
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
            source_effect: 0,
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
        // `abilities:surgesurfer.onModifySpe`: doubles Speed while the field
        // terrain is Electric Terrain. The reference reads
        // `this.field.isTerrain('electricterrain')`, so grounding is not
        // required and a suppressed/absent terrain simply returns undefined.
        if ability == Ability::Surgesurfer
            && self.terrain_id(dex) == dex.effects.electric_terrain
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

    /// `Pokemon#getMoves` substitutes the *served* target class before the
    /// request advertises a move, and `Side#chooseMove` validates the submitted
    /// location against that served value rather than the raw dex row:
    ///   - `case 'curse': if (!this.hasType('Ghost')) target = 'self';`
    ///   - `case 'pollenpuff': if (this.volatiles['healblock']) target =
    ///     'adjacentFoe';`
    ///   - `case 'terastarstorm'`: Terapagos-Stellar advertises
    ///     `allAdjacentFoes` (that forme is outside the pinned regulation).
    ///
    /// The action mask has to serve exactly what the reference validates, so
    /// the non-Ghost Curse choice is location-less and a Heal-Blocked Pollen
    /// Puff cannot address its ally.
    pub(crate) fn served_target(&self, dex: &Dex, e: Entity, move_id: Id) -> Target {
        let hooks = dex.effects.move_hooks[move_id as usize];
        if hooks & crate::effects::hook::CURSE != 0
            && !self.effective_types(dex, e).contains(&dex.effects.ghost)
        {
            Target::SelfOnly
        } else if hooks & crate::effects::hook::POLLEN_PUFF != 0
            && self.mon(e).volatiles.contains_key(&dex.effects.heal_block)
        {
            Target::AdjacentFoe
        } else {
            dex.moves[move_id as usize].target
        }
    }

    /// `battle-queue.ts#insertChoice` clones the queued move and applies the
    /// champions mod's Curse rewrite (`!hasType('Ghost')` -> `target = 'self'`)
    /// before every later target read, so the queue-time target of a
    /// location-less Curse choice is never the raw dex row: the first
    /// `getRandomTarget`, each `getActionSpeed`'s `getTarget`, the `runMove`
    /// read and the Encore `changeAction` re-insert all resolve a non-Ghost
    /// Curse as self and never sample a foe.
    fn queued_target(&self, dex: &Dex, actor: Entity, move_id: Id) -> Target {
        if dex.effects.move_hooks[move_id as usize] & crate::effects::hook::CURSE != 0
            && !self
                .effective_types(dex, actor)
                .contains(&dex.effects.ghost)
        {
            Target::SelfOnly
        } else {
            dex.moves[move_id as usize].target
        }
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
            // Volatile `Update` handlers (sub-order 2) run before the ability
            // (7) and item (8) groups; the Fling marker is the ported case.
            self.fling_update(dex, e)?;
            // Ability Update handlers run before item Update handlers.
            self.disguise_update(dex, e)?;
            self.item_update(dex, e)?;
        }
        Ok(())
    }

    /// `abilities:disguise.onUpdate`: a pending bust changes the holder to its
    /// busted forme and then pays an eighth of its maximum HP as damage whose
    /// effect is the new species. The forme change is cosmetic and permanent -
    /// Mimikyu-Busted keeps the same stats, types and ability - so only the
    /// species identity and the public events change.
    fn disguise_update(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if !self.mon(e).disguise_busted {
            return Ok(());
        }
        self.mon_mut(e).disguise_busted = false;
        let busted = if self.mon(e).species == dex.effects.mimikyu_totem {
            dex.effects.mimikyu_busted_totem
        } else {
            dex.effects.mimikyu_busted
        };
        let types = dex.species[busted as usize].types.clone();
        {
            let mon = self.mon_mut(e);
            mon.species = busted;
            mon.base_species = busted;
            mon.types = types.clone();
        }
        self.emit(
            EventKind::Forme,
            e,
            None,
            EffectRef::Species(busted),
            0,
            true,
        )?;
        for viewer in 0..2 {
            let index = e.roster as usize + if e.side as usize == viewer { 0 } else { 6 };
            self.knowledge[viewer].pokemon[index].types = types.clone();
        }
        let amount = u32::from(self.mon(e).stats[0]) / 8;
        self.indirect_damage(dex, e, e, amount, EffectRef::Species(busted))?;
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
                // Reference `switchIn` -> `copyVolatileFrom`: a Baton Pass
                // replacement adopts the outgoing Pokémon's boosts and every
                // non-`noCopy` volatile, a Shed Tail replacement only the
                // decoy; both run before the outgoing set is cleared.
                match self.mon(old).switch_flag {
                    Some(id) if id == dex.effects.baton_pass_move => {
                        self.copy_volatiles(dex, old, incoming, crate::assets::SelfSwitch::CopyVolatile)?;
                    }
                    Some(id) if id == dex.effects.shed_tail_move => {
                        self.copy_volatiles(dex, old, incoming, crate::assets::SelfSwitch::ShedTail)?;
                    }
                    _ => {}
                }
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
        // Reference `Pokemon#switchIn`: `newlySwitched = true`, cleared at the
        // next turn rollover.
        self.mon_mut(incoming).newly_switched = true;
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
                source_effect: 0,
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
                // `moves:healingwish.condition.onSwitchIn`: a slot condition is
                // a side-condition handler with the entering Pokémon as its
                // target, so it sorts at sub-order 3, before the entry hazards
                // (sub-order 4) and the entrant's ability (7). `onSwap` fully
                // heals and cures the entrant, then consumes the marker.
                let slot = self.mon(e).active_slot.unwrap_or(0);
                let healing_wish = self.sides[e.side as usize].slot_conditions
                    [slot as usize]
                    .contains_key(&dex.effects.healing_wish)
                    && !self.mon(e).fainted
                    && (self.mon(e).hp < self.mon(e).stats[0] || self.mon(e).status != 0);
                if healing_wish {
                    self.sides[e.side as usize].slot_conditions[slot as usize]
                        .remove(&dex.effects.healing_wish);
                    let max = u32::from(self.mon(e).stats[0]);
                    let healed = max.saturating_sub(u32::from(self.mon(e).hp));
                    self.mon_mut(e).hp = max as u16;
                    if healed > 0 {
                        self.emit(
                            EventKind::Heal,
                            e,
                            None,
                            EffectRef::Condition(dex.effects.healing_wish),
                            healed as i32,
                            true,
                        )?;
                    }
                    self.cure_status(e)?;
                }
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
        // `onSideRestart` layer caps: Spikes three, Toxic Spikes two, Stealth
        // Rock and Sticky Web a single layer.
        let cap = if id == dex.effects.spikes {
            3
        } else if id == dex.effects.toxic_spikes {
            2
        } else {
            1
        };
        // The public event names the *side* that now carries the hazard, not
        // the source (which stays the recorded source).
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
            if layers >= cap {
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
                // Spikes and Toxic Spikes store their layer count in the
                // duration slot so the entry-hazard fixture contract
                // (id, layers) matches every other side condition; Stealth
                // Rock and Sticky Web carry no layers (the reference reports
                // zero). `residual` skips every hazard id entirely.
                duration: if cap > 1 { Some(1) } else { None },
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
            if cap > 1 { 1 } else { 0 },
            false,
        )?;
        Ok(true)
    }

    /// Reference `moves:toxicspikes.condition.onSwitchIn`: a grounded entrant
    /// absorbs the hazard when it is a Poison type, ignores it as a Steel type,
    /// and is otherwise poisoned (one layer) or badly poisoned (two layers) by
    /// the opposing side's first active Pokémon.
    ///
    /// All four entry hazards share the side-condition `SwitchIn` event. The
    /// reference collects them in `sideConditions` insertion order (the native
    /// BTreeMap is id-ordered, so `effect_order` reconstructs it) and runs the
    /// ordinary `speedSort` over the equally-keyed handlers, which shuffles the
    /// whole tie group and consumes the Fisher-Yates draws. Each hazard then
    /// applies its own switch-in rule.
    fn hazard_switch_in(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        let side = e.side as usize;
        let mut hazards: SmallVec<[(Id, Priority); 4]> = self.sides[side]
            .conditions
            .iter()
            .filter(|(id, _)| {
                [
                    dex.effects.spikes,
                    dex.effects.stealth_rock,
                    dex.effects.toxic_spikes,
                    dex.effects.sticky_web,
                ]
                .contains(id)
            })
            .map(|(id, state)| {
                (
                    *id,
                    Priority {
                        sub_order: 4,
                        effect_order: state.effect_order,
                        ..Default::default()
                    },
                )
            })
            .collect();
        hazards.sort_by_key(|(_, priority)| priority.effect_order);
        speed_sort(&mut hazards, &mut self.rng, |x| x.1);
        for (id, _) in hazards {
            if id == dex.effects.spikes {
                // Spikes: grounded entrants only, `damageAmounts[layers] *
                // maxhp / 24` floored, minimum one.
                if !self.grounded(dex, e) {
                    continue;
                }
                let layers = self.sides[side].conditions[&id]
                    .values
                    .first()
                    .copied()
                    .unwrap_or(1)
                    .clamp(1, 3) as usize;
                let table = [0u32, 3, 4, 6][layers];
                let amount = (table * u32::from(self.mon(e).stats[0]) / 24).max(1) as u16;
                self.entry_hazard_damage(e, id, amount)?;
                continue;
            }
            if id == dex.effects.stealth_rock {
                // Stealth Rock: `maxhp * 2^typeMod / 8` floored, minimum one,
                // against the Rock effectiveness of the entrant's types.
                let mut type_mod = 0i32;
                let mut immune = false;
                for &kind in &self.mon(e).types {
                    let value = i32::from(dex.type_chart[dex.effects.rock as usize][kind as usize]);
                    if value == -127 {
                        immune = true;
                        break;
                    }
                    type_mod += value;
                }
                if immune {
                    continue;
                }
                let type_mod = type_mod.clamp(-6, 6);
                let max_hp = u32::from(self.mon(e).stats[0]);
                let amount = if type_mod >= 0 {
                    (max_hp * (1u32 << type_mod) / 8).max(1)
                } else {
                    (max_hp / (8 * (1u32 << -type_mod))).max(1)
                };
                self.entry_hazard_damage(e, id, amount as u16)?;
                continue;
            }
            if id == dex.effects.sticky_web {
                // Sticky Web: a grounded entrant loses one Speed stage; the
                // source is the opposing side's first active.
                if !self.grounded(dex, e) {
                    continue;
                }
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
                    continue;
                };
                // The reference's public `-activate ... move: Sticky Web`
                // line is a message detail the native event model does not
                // carry; the hazard's public presence is already tracked by
                // its side-effect start event.
                self.boost(
                    dex,
                    e,
                    source,
                    [0, 0, 0, 0, -1, 0, 0],
                    BoostCause::Move { secondary: false },
                )?;
                continue;
            }
            // Toxic Spikes.
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

    /// Entry-hazard damage: `Battle#damage` with the hazard condition as the
    /// effect and no source Pokémon. A zero-HP entrant joins the faint queue
    /// and is processed at the next faint boundary.
    fn entry_hazard_damage(
        &mut self,
        e: Entity,
        id: Id,
        amount: u16,
    ) -> Result<()> {
        let actual = amount.min(self.mon(e).hp);
        if actual == 0 {
            return Ok(());
        }
        self.mon_mut(e).hp -= actual;
        if self.mon(e).hp == 0 {
            self.faint_queue.push(FaintData {
                target: e,
                source: None,
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
        )
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

    /// `abilities:stancechange.onModifyMove`: the Aegislash base chain
    /// switches to Blade for a damaging move (Struggle included) and back to
    /// Shield for King's Shield; any other status move leaves the forme
    /// alone. The change is non-permanent: `base_species` keeps the submitted
    /// forme so switch-out reverts it, and only the stored stats are
    /// recomputed (both formes share the same base HP, so HP is preserved).
    /// `moves:fling.onPrepareHit`: the thrown item sets the action's base power
    /// and arms the marker volatile that consumes it after the hit loop.
    /// `item.fling` data decides the payload; the pinned plain-item path is
    /// ported, while a Berry/status/herb payload stays an explicit operational
    /// error until its on-hit port lands.
    fn fling_prepare(
        &mut self,
        dex: &Dex,
        actor: Entity,
        action: &mut ActiveMove<'_>,
    ) -> Result<bool> {
        let item = self.mon(actor).item;
        if item == 0 {
            return Ok(false);
        }
        let Some(spec) = dex.effects.fling_items[item as usize] else {
            return Ok(false);
        };
        // `singleEvent('TakeItem', item, state, source, source, move, item)`:
        // an item that refuses removal (a Mega Stone on its own base form) is
        // not thrown.
        if dex.item_take_refused(item, self.mon(actor).base_species) {
            return Ok(false);
        }
        if spec.kind == crate::effects::FlingKind::Unsupported {
            return Err(EngineError::Unsupported(format!(
                "fling payload {}",
                dex.names["items"][item as usize]
            )));
        }
        action.power = spec.base_power;
        action.fling = Some((item, spec.kind));
        if !self.mon(actor).volatiles.contains_key(&dex.effects.fling) {
            let order = self.allocate_effect_order()?;
            self.mon_mut(actor).volatiles.insert(
                dex.effects.fling,
                crate::state::EffectState {
                    id: dex.effects.fling,
                    effect_order: order,
                    effect_order_assigned: true,
                    ..Default::default()
                },
            );
        }
        Ok(true)
    }

    /// `moves:fling.condition.onUpdate`: the marker consumes the thrown item on
    /// the next `Update`, which also runs the item's `AfterUseItem` set (so
    /// Symbiosis and Unburden answer a throw exactly like any other use).
    fn fling_update(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if self
            .mon_mut(e)
            .volatiles
            .remove(&dex.effects.fling)
            .is_none()
        {
            return Ok(());
        }
        self.consume_item(dex, e)?;
        Ok(())
    }

    /// Reference `Pokemon#setSpecies` for a temporary forme change: the new
    /// species' types and base stats replace the old ones, `speed` follows the
    /// new base speed, and the public change reaches both viewers' knowledge.
    /// `setSpecies` only initializes the max-HP/current-HP pair for a fresh
    /// Pokémon, so an established forme keeps its HP values.
    pub(super) fn forme_change(&mut self, dex: &Dex, e: Entity, species_id: Id) -> Result<()> {
        let species = &dex.species[species_id as usize];
        let mon = self.mon(e);
        let mut new_stats = stats::champions_stats(
            species.base_stats,
            mon.points,
            dex.natures[mon.nature as usize],
            species.max_hp,
        );
        new_stats[0] = mon.stats[0];
        let types = species.types.clone();
        {
            let mon = self.mon_mut(e);
            mon.species = species_id;
            mon.types = types.clone();
            mon.stats = new_stats;
            mon.cached_speed = i32::from(new_stats[5]);
        }
        self.emit(
            EventKind::Forme,
            e,
            None,
            EffectRef::Species(species_id),
            0,
            false,
        )?;
        // Both players see the new forme's typing immediately.
        for viewer in 0..2 {
            let index = e.roster as usize + if e.side as usize == viewer { 0 } else { 6 };
            self.knowledge[viewer].pokemon[index].types = types.clone();
        }
        Ok(())
    }

    fn stance_change(
        &mut self,
        dex: &Dex,
        actor: Entity,
        move_id: Id,
        category: Category,
    ) -> Result<()> {
        if dex.effects.abilities[self.mon(actor).ability as usize] != Ability::Stancechange {
            return Ok(());
        }
        let mon = self.mon(actor);
        if mon.transformed || dex.species[mon.species as usize].base_species != dex.effects.aegislash
        {
            return Ok(());
        }
        // `if (move.category === 'Status' && move.id !== 'kingsshield') return;`
        if category == Category::Status && move_id != dex.effects.kings_shield_move {
            return Ok(());
        }
        let target = if move_id == dex.effects.kings_shield_move {
            dex.effects.aegislash
        } else {
            dex.effects.aegislash_blade
        };
        if mon.species == target {
            return Ok(());
        }
        let species = &dex.species[target as usize];
        let new_stats = stats::champions_stats(
            species.base_stats,
            mon.points,
            dex.natures[mon.nature as usize],
            species.max_hp,
        );
        let types = species.types.clone();
        {
            let mon = self.mon_mut(actor);
            mon.species = target;
            mon.types = types.clone();
            mon.stats = new_stats;
            mon.cached_speed = i32::from(new_stats[5]);
            if mon.hp > mon.stats[0] {
                mon.hp = mon.stats[0];
            }
        }
        self.emit(
            EventKind::Forme,
            actor,
            None,
            EffectRef::Species(target),
            0,
            false,
        )?;
        // Both players see the new forme's typing immediately.
        for viewer in 0..2 {
            let index = actor.roster as usize + if actor.side as usize == viewer { 0 } else { 6 };
            self.knowledge[viewer].pokemon[index].types = types.clone();
        }
        Ok(())
    }

    /// Reference `Pokemon#copyVolatileFrom`: the incoming Pokémon clears its
    /// own volatile state and then receives a shallow copy of the outgoing
    /// set. `copyvolatile` (Baton Pass) also adopts the boost stages and every
    /// volatile whose condition does not declare `noCopy`; `shedtail` moves
    /// only the decoy and no boosts. A condition with an `onCopy` callback is
    /// unreachable from a ported effect today and stays an explicit
    /// operational error instead of a silent no-op.
    fn copy_volatiles(
        &mut self,
        dex: &Dex,
        from: Entity,
        to: Entity,
        cause: crate::assets::SelfSwitch,
    ) -> Result<()> {
        let shed_tail = cause == crate::assets::SelfSwitch::ShedTail;
        self.clear_volatile(dex, to);
        if !shed_tail {
            self.mon_mut(to).boosts = self.mon(from).boosts;
        }
        let copied: Vec<(Id, EffectState)> = self
            .mon(from)
            .volatiles
            .iter()
            .map(|(id, state)| (*id, state.clone()))
            .collect();
        for (id, state) in copied {
            if shed_tail && id != dex.effects.substitute {
                continue;
            }
            if dex.effects.no_copy_conditions.contains(&id) {
                continue;
            }
            if id == dex.effects.power_trick || id == dex.effects.power_shift {
                // `moves:powertrick|powershift.condition.onCopy`: the incoming
                // Pokémon swaps its own stored Attack and Defense as it adopts
                // the marker.
                let atk = self.mon(to).stats[1];
                let def = self.mon(to).stats[2];
                self.mon_mut(to).stats[1] = def;
                self.mon_mut(to).stats[2] = atk;
            } else if dex.effects.copy_callback_conditions.contains(&id) {
                return Err(EngineError::Unsupported(format!(
                    "copied volatile {id} declares onCopy"
                )));
            }
            self.mon_mut(to).volatiles.insert(id, state);
        }
        Ok(())
    }

    /// `Pokemon#removeLinkedVolatiles`: the `trapped`/`trapper` pair stores the
    /// other side as its source, so clearing either clears the partner silently
    /// (neither condition declares an End callback).
    fn clear_linked_trap(&mut self, dex: &Dex, e: Entity) {
        let slot = (
            if e.side == 0 { SideId::P1 } else { SideId::P2 },
            e.roster,
        );
        if let Some(link) = self
            .mon(e)
            .volatiles
            .get(&dex.effects.trapped)
            .and_then(|state| state.source)
        {
            let holder = Entity {
                side: link.0.index() as u8,
                roster: link.1,
            };
            if self
                .mon(holder)
                .volatiles
                .get(&dex.effects.trapper)
                .is_some_and(|state| state.source == Some(slot))
            {
                self.mon_mut(holder).volatiles.remove(&dex.effects.trapper);
            }
        }
        if let Some(link) = self
            .mon(e)
            .volatiles
            .get(&dex.effects.trapper)
            .and_then(|state| state.source)
        {
            let holder = Entity {
                side: link.0.index() as u8,
                roster: link.1,
            };
            if self
                .mon(holder)
                .volatiles
                .get(&dex.effects.trapped)
                .is_some_and(|state| state.source == Some(slot))
            {
                self.mon_mut(holder).volatiles.remove(&dex.effects.trapped);
            }
        }
    }

    fn clear_volatile(&mut self, dex: &Dex, e: Entity) {
        self.clear_linked_trap(dex, e);
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
        mon.disguise_busted = false;
        // Reference `Pokemon#clearVolatile`: the per-turn damage and stat
        // flags do not survive a switch-out.
        mon.hurt_this_turn = 0;
        mon.stats_raised_this_turn = false;
        mon.stats_lowered_this_turn = false;
        mon.attacked_by.clear();
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
                    let incoming = Entity {
                        side: actor.side,
                        roster: action.destination,
                    };
                    // A switch action submitted under the `revivalblessing`
                    // slot condition revives the chosen fainted member instead
                    // of swapping the user out.
                    if self.sides[actor.side as usize].slot_conditions[slot as usize]
                        .contains_key(&dex.effects.revival_blessing)
                    {
                        self.apply_revival_blessing(dex, actor, incoming, slot)?;
                    } else {
                        self.switch_in(dex, incoming, slot)?;
                    }
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
                        action.source_effect,
                    )?;
                    item_ports::white_herb_event(self, dex)?;
                }
                QueuedKind::PriorityCharge => {
                    // `moves:chillyreception.priorityChargeCallback`: the
                    // queued action adds the move's one-turn volatile before
                    // any move of the turn. The reference start is silent for
                    // this condition, and its residual duration tick removes
                    // it at the end of the turn.
                    let actor = action.actor.unwrap();
                    if self
                        .mon(actor)
                        .volatiles
                        .contains_key(&dex.effects.chilly_reception)
                    {
                        continue;
                    }
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(actor).volatiles.insert(
                        dex.effects.chilly_reception,
                        EffectState {
                            id: dex.effects.chilly_reception,
                            duration: Some(1),
                            effect_order: order,
                            effect_order_assigned: true,
                            ..Default::default()
                        },
                    );
                }
                QueuedKind::Residual => self.residual(dex)?,
                QueuedKind::BeforeTurnMove => {
                    // Reference `runAction` case `beforeTurnMove`: the queued
                    // move's `beforeTurnCallback` runs before any move of the
                    // turn. The action resolves its stored target first
                    // (`getTarget`) and is skipped when no target exists; the
                    // Counter / Mirror Coat callback then adds its one-turn
                    // retaliation volatile (whose `onStart` clears slot and
                    // damage).
                    let actor = action.actor.unwrap();
                    if !self.mon(actor).fainted && self.mon(actor).active_slot.is_some() {
                        let target = dex.moves[action.move_id as usize].target;
                        let resolved_loc =
                            self.resolve_target_location(actor, target, action.target_location);
                        if self.at_location(actor, resolved_loc).is_some() {
                            let volatile = if dex.effects.moves[action.move_id as usize]
                                == MoveBehavior::MirrorCoat
                            {
                                dex.effects.mirrorcoat
                            } else {
                                dex.effects.counter
                            };
                            if !self.mon(actor).volatiles.contains_key(&volatile) {
                                let order = self.allocate_effect_order()?;
                                self.mon_mut(actor).volatiles.insert(
                                    volatile,
                                    EffectState {
                                        id: volatile,
                                        duration: Some(1),
                                        effect_order: order,
                                        effect_order_assigned: true,
                                        source: Some((
                                            if actor.side == 0 {
                                                SideId::P1
                                            } else {
                                                SideId::P2
                                            },
                                            actor.roster,
                                        )),
                                        ..Default::default()
                                    },
                                );
                            }
                        }
                    }
                }
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
                        let target = self.queued_target(dex, q.actor.unwrap(), q.move_id);
                        self.resolve_target_location(
                            q.actor.unwrap(),
                            target,
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
        // Reference `Battle#getTarget` validates the stored location with
        // `validTargetLoc`, where a `scripted` move resolves exactly like a
        // `normal` one (adjacent slot, never the user's own). The served
        // request still never offers it a chosen location, because
        // `Target#chooses_target` stays false for `scripted`.
        let valid = if target == Target::Scripted {
            loc != 0 && loc != -(slot as i8 + 1) && (-2..=2).contains(&loc)
        } else {
            target.valid_location(slot, loc)
        };
        if target != Target::RandomNormal
            && valid
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
        if spec.instant_weather.contains(&self.mon_weather(dex, actor)) {
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

    fn use_move(
        &mut self,
        dex: &Dex,
        actor: Entity,
        slot: u8,
        move_id: Id,
        loc: i8,
        source_effect: Id,
    ) -> Result<()> {
        // `runMove` fires the AfterMove events after `useMove` returns; a
        // BeforeMove refusal (sleep, flinch, full paralysis, confusion
        // self-hit, frozen) returns before them, so the rampage lock must not
        // tick on a turn the Pokémon never moved.
        let mut ran = false;
        let result = self.use_move_inner(
            dex,
            actor,
            slot,
            move_id,
            loc,
            MoveUse {
                caller_slot: slot,
                source_effect,
                ran: Some(&mut ran),
                ..Default::default()
            },
        );
        // `conditions:charge.onAfterMove|onMoveAborted`: an Electric attempt
        // consumes the volatile whether it resolved or was aborted.
        self.charge_after_move(dex, actor, move_id)?;
        if ran {
            self.locked_move_after_move(dex, actor)?;
        }
        result
    }

    /// `BattleActions#useMove`: a move invoked by another move (Sleep Talk
    /// today, the rest of the caller family later). The called move resolves its
    /// target through `Battle#getRandomTarget` (one RNG sample when the target
    /// class needs one) and never pays PP, so it enters the ordinary move
    /// pipeline with `slot = NO_SLOT` and `calls_move` set.
    fn use_called_move(
        &mut self,
        dex: &Dex,
        actor: Entity,
        move_id: Id,
        caller_slot: u8,
    ) -> Result<()> {
        let target = dex.moves[move_id as usize].target;
        let loc = self.random_target_location(actor, target);
        let mut ran = false;
        let result = self.use_move_inner(
            dex,
            actor,
            NO_SLOT,
            move_id,
            loc,
            MoveUse {
                called: true,
                caller_slot,
                ran: Some(&mut ran),
                ..Default::default()
            },
        );
        // A called Electric move consumes the Charge volatile as well.
        self.charge_after_move(dex, actor, move_id)?;
        result
    }

    fn use_move_inner(
        &mut self,
        dex: &Dex,
        actor: Entity,
        slot: u8,
        move_id: Id,
        loc: i8,
        mut call: MoveUse,
    ) -> Result<()> {
        let called = call.called;
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
        // a flinched, sleeping or fully paralysed attempt still counts. A move
        // invoked through `BattleActions#useMove` (Sleep Talk, Magic Bounce)
        // runs no `runMove`, so the counter stays with the outer action.
        if !called {
            let attempts = self.mon(actor).active_move_actions;
            self.mon_mut(actor).active_move_actions = attempts.saturating_add(1);
        }
        // Reference `runMove` reads `getTarget` before `BeforeMove` runs. The
        // Recharge pseudo-move has no target class, so that read falls through
        // to `getRandomTarget` and samples a random foe (one draw) instead of
        // resolving the stored location.
        let recharge_lock = self
            .mon(actor)
            .volatiles
            .contains_key(&dex.effects.must_recharge);
        let loc = if call.explicit_target {
            loc
        } else if recharge_lock {
            self.sample_random_foe(actor);
            loc
        } else {
            // A queued action reads the insertChoice-hooked target; a nested
            // `useMove` (Sleep Talk, Magic Bounce) resolves the called move's
            // own class instead.
            let target = if called {
                m.target
            } else {
                self.queued_target(dex, actor, m.id)
            };
            self.resolve_target_location(actor, target, loc)
        };
        // The reference runs the BeforeMove event once per *action*, before
        // `useMove`; a move called by another move (Sleep Talk) must not run it
        // again, or the sleep counter would tick twice in one turn.
        if !called {
            if let Some(result) = self.before_move(dex, actor, m)? {
                // Reference `runEvent('MoveAborted')`: a cancelled attempt
                // drops Destiny Bond (`onMoveAborted` has no move guard).
                self.drop_destiny_bond(dex, actor)?;
                self.mon_mut(actor).move_this_turn_result = result;
                return Ok(());
            }
            // `moves:destinybond.condition.onBeforeMove` (priority -1) removes
            // the bond before any attack that is not Destiny Bond itself.
            if m.id != dex.effects.destiny_bond_move {
                self.drop_destiny_bond(dex, actor)?;
            }
        }
        // Reference `useMoveInner` skips PP deduction while the Pokémon is
        // locked (`getLockedMove()`), i.e. on the release turn of a charge, on
        // the forced Recharge turn and on every continuation turn of a rampage.
        let locked = self
            .mon(actor)
            .volatiles
            .contains_key(&dex.effects.two_turn_move)
            || self
                .mon(actor)
                .volatiles
                .contains_key(&dex.effects.must_recharge)
            || self
                .mon(actor)
                .volatiles
                .contains_key(&dex.effects.locked_move);
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
        // Past the PP gate the reference has committed the attempt: `moveUsed`
        // records the move and its chosen location, and `runMove` fires the
        // AfterMove events even when the move whiffs or is refused by its own
        // gates. A nested `useMove` reports it too but its callers ignore it.
        if let Some(ran) = call.ran.as_deref_mut() {
            *ran = true;
        }
        // Reference `Pokemon#moveUsed` records the move before any hit steps,
        // so a missed, failed or status-refused move still becomes `lastMove`
        // for Encore, Disable, Torment and Cursed Body. Only `runMove` calls
        // `moveUsed`; a nested `useMove` leaves `lastMove` alone.
        if !called {
            let mon = self.mon_mut(actor);
            mon.last_move = move_id;
            mon.last_move_target_location = chosen_location;
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
        let mut action = self.active_move(dex, actor, m, behavior);
        action.move_uid = self.allocate_move_uid()?;
        action.calls_move = called;
        action.has_bounced = call.bounced;
        // `moves:round.basePowerCallback`: a Round action that another Round
        // pulled to the front carries `move.sourceEffect === 'round'` and
        // doubles its base power.
        if behavior == MoveBehavior::Round && call.source_effect == dex.effects.round {
            action.power = action.power.saturating_mul(2);
        }
        // Reference `useMoveInner`: a nested caller inherits the outer
        // action's stored priority (`battle.queue` writes the ModifyPriority
        // result onto the active move). A chosen action resolves its own.
        let effective_priority = call
            .priority
            .unwrap_or_else(|| self.effective_priority(dex, actor, move_id));
        action.priority = Some(effective_priority);
        // `abilities:stancechange.onModifyMove` (priority 1): a damaging move
        // or King's Shield switches the Aegislash forme before the action
        // resolves, so the move itself is used with the new forme's stats.
        self.stance_change(dex, actor, move_id, m.category)?;
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
        // `moves:fling.onPrepareHit`: the thrown item sets the action's base
        // power (and its payload) before any target resolves. A refusal leaves
        // the action without targets, so it reports the reference's failure.
        // `moves:lastresort.onTry`: the move fails until every other move slot
        // has been used at least once.
        if behavior == MoveBehavior::LastResort {
            let knows = self.mon(actor).moves.len() >= 2;
            let ready = self
                .mon(actor)
                .moves
                .iter()
                .all(|mv| mv.used || mv.id == move_id);
            if !knows || !ready {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
        }
        // `moves:counter|mirrorcoat.onTry`: the retaliation fails without its
        // one-turn volatile or before a qualifying hit was recorded in it
        // (`slot === null`).
        if matches!(behavior, MoveBehavior::Counter | MoveBehavior::MirrorCoat) {
            let volatile = if behavior == MoveBehavior::MirrorCoat {
                dex.effects.mirrorcoat
            } else {
                dex.effects.counter
            };
            let recorded = self
                .mon(actor)
                .volatiles
                .get(&volatile)
                .is_some_and(|state| state.values.len() == 3);
            if !recorded {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
        }
        // `moves:metalburst|comeuppance.onTry`: the retaliation fails unless a
        // non-ally damaged the user this turn (`getLastDamagedBy(true)` with
        // its `thisTurn` flag).
        if matches!(
            behavior,
            MoveBehavior::MetalBurst | MoveBehavior::Comeuppance
        ) && self.last_damaged_by(actor).is_none()
        {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        let fling_ready = behavior != MoveBehavior::Fling
            || self.fling_prepare(dex, actor, &mut action)?;
        if !fling_ready {
            // The refused throw never enters the hit loop and runs no `Update`
            // of its own: the reference logs `-fail` and moves straight to the
            // next queued action's queue re-sort.
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
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
        // `moves:counter|mirrorcoat.condition.onRedirectTarget` (priority -1)
        // and `moves:metalburst|comeuppance.onModifyTarget` both aim the
        // retaliation at the attacker recorded this turn, but only when no
        // higher-priority redirector (Follow Me, Rage Powder, a redirection
        // ability) already claimed the move. The recorded value is an absolute
        // slot: `getAtSlot` reads whoever occupies it now, so a pivot that
        // refilled the slot is followed, and a fainted occupant leaves Counter
        // and Mirror Coat without a target (the reference's `-fail`) while
        // Metal Burst and Comeuppance re-sample a live foe.
        if matches!(
            behavior,
            MoveBehavior::Counter
                | MoveBehavior::MirrorCoat
                | MoveBehavior::MetalBurst
                | MoveBehavior::Comeuppance
        ) && redirected == selected
        {
            let counter_move = matches!(behavior, MoveBehavior::Counter | MoveBehavior::MirrorCoat);
            let recorded = if counter_move {
                let volatile = if behavior == MoveBehavior::MirrorCoat {
                    dex.effects.mirrorcoat
                } else {
                    dex.effects.counter
                };
                self.mon(actor).volatiles.get(&volatile).and_then(|state| {
                    if state.values.len() == 3 {
                        Some((
                            if state.values[1] == 0 {
                                SideId::P1
                            } else {
                                SideId::P2
                            },
                            state.values[2] as u8,
                        ))
                    } else {
                        None
                    }
                })
            } else {
                self.last_damaged_by(actor).map(|(side, slot, _)| (side, slot))
            };
            if let Some((side, slot)) = recorded {
                match self
                    .entity_at_slot(side, slot)
                    .filter(|e| !self.mon(*e).fainted && self.mon(*e).hp > 0)
                {
                    Some(target) => {
                        targets.clear();
                        targets.push(target);
                    }
                    None if counter_move => targets.clear(),
                    None => match self.sample_random_foe(actor) {
                        Some(target) => {
                            targets.clear();
                            targets.push(target);
                        }
                        None => targets.clear(),
                    },
                }
            }
        }
        // `moves:curse.onModifyMove` (Ghost branch): a Ghost user whose chosen
        // target is an ally - or that has no target at all - re-samples a
        // random foe, exactly like the reference's `getRandomTarget` on a
        // `randomNormal` class.
        if m.hooks & crate::effects::hook::CURSE != 0
            && self
                .effective_types(dex, actor)
                .contains(&dex.effects.ghost)
            && targets
                .first()
                .is_none_or(|target| target.side == actor.side)
        {
            match self.sample_random_foe(actor) {
                Some(target) => {
                    targets.clear();
                    targets.push(target);
                }
                None => targets.clear(),
            }
        }
        // `abilities:pressure.onDeductPP`: the reference resolves the move's
        // apparent targets (after redirection) and charges one extra PP per
        // opposing Pressure holder among them (`pressureTargets`; `foeSide`
        // moves resolve none, `mustpressure` moves use every foe). The base PP
        // deduction in `runMove` already happened; a move invoked by another
        // move charges the caller's slot (`callerMoveForPressure`), while a
        // Magic Bounce reflection has an ability source effect and pays
        // nothing.
        let pp_slot = if call.called { call.caller_slot } else { slot };
        if pp_slot != NO_SLOT && !locked {
            let pressure_targets: SmallVec<[Entity; 4]> = if m.must_pressure {
                self.active_entities(false)
                    .into_iter()
                    .filter(|foe| foe.side != actor.side)
                    .collect()
            } else if m.target == Target::FoeSide {
                SmallVec::new()
            } else {
                targets.clone()
            };
            let extra = pressure_targets
                .into_iter()
                .filter(|target| {
                    target.side != actor.side
                        && self.mon(*target).hp > 0
                        && dex.effects.abilities[self.mon(*target).ability as usize]
                            == Ability::Pressure
                })
                .count() as u8;
            if extra > 0 {
                let mon = self.mon_mut(actor);
                let pp = mon.moves[pp_slot as usize].pp;
                mon.moves[pp_slot as usize].pp = pp.saturating_sub(extra);
                mon.base_moves[pp_slot as usize].pp = pp.saturating_sub(extra);
            }
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
            let priority = effective_priority;
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
        // `moves:pollenpuff.onTryHit`: a Pollen Puff aimed at an ally drops to
        // zero power and gains `move.infiltrates` for the action, so the heal
        // passes through the ally's decoy. `onTryMove` then refuses that use
        // while the *user* is under Heal Block.
        let pollen_ally = hooks & crate::effects::hook::POLLEN_PUFF != 0
            && targets
                .first()
                .is_some_and(|target| target.side == actor.side);
        if pollen_ally && self.heal_blocked(dex, actor) {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
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
        // `moves:upperhand.onTry`: the move fails outright unless the target
        // still has a queued move action whose declaration priority is
        // positive and whose category is not Status. The reference reads
        // `action.move.priority` (the move data, not the modified action
        // priority), so Prankster/Gale Wings boosts do not qualify a move.
        if behavior == MoveBehavior::UpperHand {
            let target = redirected.or(selected);
            let qualifies = target.is_some_and(|target| {
                !self.mon(target).fainted
                    && self.mon(target).hp > 0
                    && self
                        .queue
                        .iter()
                        .find(|q| q.kind == QueuedKind::Move && q.actor == Some(target))
                        .is_some_and(|q| {
                            let data = &dex.moves[q.move_id as usize];
                            data.priority > 0 && data.category != Category::Status
                        })
            });
            if !qualifies {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
        }
        // `moves:round.onTry`: another queued Round move action (of any
        // Pokémon on either side) jumps to the front of the queue, marked with
        // this effect as its source, so it resolves immediately after this
        // Round and doubles its own base power.
        if behavior == MoveBehavior::Round
            && let Some(index) = self
                .queue
                .iter()
                .position(|q| q.kind == QueuedKind::Move && q.move_id == dex.effects.round)
        {
            let mut action = self.queue.remove(index);
            action.priority.order = 3;
            action.source_effect = dex.effects.round;
            self.queue.insert(0, action);
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
        // `moves:poltergeist.onTry`: the move fails outright when the target
        // holds no item, before protection, immunity and accuracy. The gate
        // reads the raw item field, so Klutz does not hide the item from it.
        if hooks & crate::effects::hook::POLTERGEIST != 0 {
            // The reference `onTry` reads the player's selected target, before
            // redirection rewrites it.
            if selected.is_none_or(|target| self.mon(target).item == 0) {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
        }
        // `moves:steelroller.onTry`: the move fails outright while no terrain
        // is active, before protection, immunity and accuracy.
        if hooks & crate::effects::hook::STEEL_ROLLER != 0 && self.terrain_id(dex) == 0 {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        // `moves:spitup.onTry`: the move needs the user's stockpile volatile.
        if hooks & crate::effects::hook::SPIT_UP != 0 && self.stockpile_layers(dex, actor) == 0 {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        // `moves:snore.onTry`: the move fails outright unless the user is
        // asleep (Comatose has no in-scope holder and stays an explicit
        // operational error).
        if hooks & crate::effects::hook::SNORE != 0
            && self.mon(actor).status != dex.effects.sleep
        {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        // `moves:noretreat.onTry`: the move fails while its own marker
        // volatile is up (the `trapped` deletion branch needs the unported
        // Mean Look volatile and cannot be reached in the pinned regulation).
        if hooks & crate::effects::hook::NO_RETREAT != 0
            && self
                .mon(actor)
                .volatiles
                .contains_key(&dex.effects.no_retreat)
        {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        // `moves:burnup.onTryMove`: the user must still be Fire-type; the
        // fail message names the move and the type is stripped on a landed hit.
        if hooks & crate::effects::hook::BURN_UP != 0
            && !self.effective_types(dex, actor).contains(&dex.effects.fire)
        {
            self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
            return Ok(());
        }
        // `items:metronome.condition.onTryMove` (priority -2, the last TryMove
        // handler): a lost item removes the counter volatile here; otherwise
        // the consecutive-use counter advances only when the previous turn
        // used the same move successfully. A two-turn release counts its
        // charge turn as one step, exactly like the reference branch.
        if self
            .mon(actor)
            .volatiles
            .contains_key(&dex.effects.metronome)
        {
            if dex.effects.items[self.mon(actor).item as usize] != Item::Metronome {
                self.mon_mut(actor).volatiles.remove(&dex.effects.metronome);
            } else {
                let charged = self
                    .mon(actor)
                    .volatiles
                    .contains_key(&dex.effects.two_turn_move);
                let (num, last) = {
                    let state = &self.mon(actor).volatiles[&dex.effects.metronome];
                    (state.values[0], state.values[1])
                };
                let same = last == i64::from(move_id);
                let previous_succeeded =
                    self.mon(actor).move_last_turn_result == MoveResult::Success;
                let next = if same && previous_succeeded {
                    num + 1
                } else if charged {
                    if same { num + 1 } else { 1 }
                } else {
                    0
                };
                let state = self
                    .mon_mut(actor)
                    .volatiles
                    .get_mut(&dex.effects.metronome)
                    .expect("metronome volatile present");
                state.values[0] = next;
                state.values[1] = i64::from(move_id);
            }
        }
        // `abilities:protean|libero.onPrepareHit` runs after the move's own
        // `onTry` gates but before the hit steps. The early-returning behavior
        // branches below run their own Try gates, so they invoke the ability
        // right after those gates instead.
        let late_prepare_hit = matches!(
            behavior,
            MoveBehavior::SleepTalk
                | MoveBehavior::Rest
                | MoveBehavior::Stockpile
                | MoveBehavior::Swallow
                | MoveBehavior::Guard
                | MoveBehavior::Protect
                | MoveBehavior::Endure
                | MoveBehavior::AllySwitch
        ) || (behavior == MoveBehavior::SideCondition
            && hooks & crate::effects::hook::AURORA_VEIL != 0);
        if !late_prepare_hit {
            self.prepare_hit_abilities(dex, actor, m)?;
        }
        if behavior == MoveBehavior::Terrain {
            self.start_terrain(dex, actor, m.terrain, false)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::SleepTalk {
            // `moves:sleeptalk.onTry`: only a sleeping user (or Comatose, which
            // no in-scope ability provides) may use the move.
            if self.mon(actor).status != dex.effects.sleep {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            self.prepare_hit_abilities(dex, actor, m)?;
            // `moves:sleeptalk.onHit`: collect the user's own eligible moves in
            // slot order, then `this.sample` one of them (one RNG draw) and use
            // it through `actions.useMove`.
            let mut candidates: SmallVec<[Id; 4]> = SmallVec::new();
            for slot in self.mon(actor).moves.iter() {
                let id = slot.id;
                if id == 0 || dex.moves[id as usize].no_sleep_talk {
                    continue;
                }
                // `charge` moves are excluded; Z/Max forms do not exist in the
                // pinned regulation.
                if dex.moves[id as usize].charge.is_some() {
                    continue;
                }
                candidates.push(id);
            }
            if candidates.is_empty() {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            let pick = candidates[self.rng.below(candidates.len() as u32) as usize];
            self.use_called_move(dex, actor, pick, slot)?;
            // Reference `hitStepMoveHitLoop` for the Sleep Talk action itself:
            // `moves:sleeptalk.onHit` ignores the called move's result and
            // returns undefined, which `runMoveEffects` turns into "did
            // something". The outer hit loop therefore always runs both
            // `eachEvent('Update')` handler-set sorts once a move was sampled
            // and used - even when the called move itself failed before its
            // own hit loop (e.g. a called Protect with no remaining action).
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::DestinyBond {
            // `moves:destinybond.onPrepareHit`: attempting the move while the
            // bond is already up removes it and fails. The hit loop's
            // all-false `moveDamage` break returns before either
            // `eachEvent('Update')` sort, so the failed path draws nothing.
            if self.mon(actor).volatiles.contains_key(&dex.effects.destiny_bond) {
                self.drop_destiny_bond(dex, actor)?;
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            let order = self.allocate_effect_order()?;
            self.mon_mut(actor).volatiles.insert(
                dex.effects.destiny_bond,
                EffectState {
                    id: dex.effects.destiny_bond,
                    effect_order: order,
                    effect_order_assigned: true,
                    source: Some((
                        if actor.side == 0 {
                            SideId::P1
                        } else {
                            SideId::P2
                        },
                        actor.roster,
                    )),
                    ..Default::default()
                },
            );
            self.emit(
                EventKind::EffectStart,
                actor,
                None,
                EffectRef::Condition(dex.effects.destiny_bond),
                0,
                false,
            )?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            // `hitStepMoveHitLoop` runs one `eachEvent('Update')` inside the
            // loop and one at the end, like the other self-target status
            // moves (Rest, Revival Blessing).
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::Rest {
            // `moves:rest.onTry` gates, in declaration order: an already
            // sleeping (or Comatose) user fails outright, a full-HP user
            // fails with the heal fail message, and the insomnia family
            // fails with its ability named. Comatose has no in-scope holder
            // and stays an explicit operational error.
            let p = self.mon(actor);
            if p.status == dex.effects.sleep {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            if p.hp == p.stats[0] {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            if dex.effects.abilities[p.ability as usize] == Ability::Insomnia {
                // The fail message names the ability, so it is revealed.
                self.reveal_ability(actor)?;
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            self.prepare_hit_abilities(dex, actor, m)?;
            // `moves:rest.onHit` calls `target.setStatus('slp', source, move)`
            // directly, so an existing major status (burn, paralysis, ...) is
            // replaced rather than refusing the move. The SetStatus pipeline
            // still refuses a grounded user under Misty Terrain (any status)
            // or Electric Terrain (sleep), and a Leaf Guard holder in sun.
            let terrain = self.terrain_id(dex);
            let terrain_blocks = self.grounded(dex, actor)
                && (terrain == dex.effects.misty_terrain
                    || terrain == dex.effects.electric_terrain);
            let leaf_guard = dex.effects.abilities[self.mon(actor).ability as usize]
                == Ability::Leafguard
                && self.effective_weather(dex) == dex.effects.sun;
            if terrain_blocks
                || leaf_guard
                || self
                    .status_immune_ability(dex, actor, dex.effects.sleep)
                    .is_some()
            {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            // `slp.onStart` samples its own 2..4 start time (one draw) before
            // Rest overwrites both counters with three.
            let _start_time = self.rng.below(3);
            let order = self.allocate_effect_order()?;
            let mon = self.mon_mut(actor);
            mon.status = dex.effects.sleep;
            mon.status_state = EffectState {
                id: dex.effects.sleep,
                effect_order: order,
                effect_order_assigned: true,
                source: Some((
                    if actor.side == 0 { SideId::P1 } else { SideId::P2 },
                    actor.roster,
                )),
                values: vec![3],
                ..Default::default()
            };
            self.emit(
                EventKind::Status,
                actor,
                Some(actor),
                EffectRef::Condition(dex.effects.sleep),
                0,
                false,
            )?;
            // AfterSetStatus: a Lum Berry cures the fresh sleep before the
            // heal half of the move runs.
            if dex.effects.items[self.mon(actor).item as usize] == Item::LumBerry {
                self.item_update(dex, actor)?;
            }
            // `this.heal(target.maxhp)`: top the user up. The reference heal
            // is silent for the `rest` effect id, and Heal Block has already
            // refused the move at BeforeMove, so no TryHeal gate applies.
            let missing = self.mon(actor).stats[0] - self.mon(actor).hp;
            if missing > 0 {
                self.mon_mut(actor).hp += missing;
                self.emit(
                    EventKind::Heal,
                    actor,
                    Some(actor),
                    EffectRef::None,
                    i32::from(missing),
                    true,
                )?;
            }
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            // Reference `hitStepMoveHitLoop` runs `eachEvent('Update')` once
            // after the hit step and once more at the end of the loop.
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::WeatherHeal {
            // `moves:synthesis|moonlight|morningsun.onHit`: half the user's
            // maximum HP, two thirds in sun and a quarter in any other
            // weather. A full-HP user fails with the heal fail message.
            let max_hp = u32::from(self.mon(actor).stats[0]);
            let weather = self.mon_weather(dex, actor);
            let factor = if weather == dex.effects.sun {
                0.667
            } else if weather == dex.effects.rain
                || weather == dex.effects.sand
                || weather == dex.effects.snow
            {
                0.25
            } else {
                0.5
            };
            let amount = ((f64::from(max_hp) * factor).floor() as u32)
                .min(max_hp - u32::from(self.mon(actor).hp));
            if amount == 0 {
                // `onHit` returns NOT_FAIL after the fail message, which keeps
                // the hit loop alive: both Update sorts still run.
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                self.each_update(dex)?;
                self.each_update(dex)?;
                return Ok(());
            }
            self.prepare_hit_abilities(dex, actor, m)?;
            self.mon_mut(actor).hp += amount as u16;
            self.emit(
                EventKind::Heal,
                actor,
                Some(actor),
                EffectRef::Move(move_id),
                amount as i32,
                false,
            )?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::BellyDrum {
            // `moves:bellydrum.onHit`: fails at half HP or less, at a capped
            // Attack stage and for a one-HP maximum, otherwise pays half the
            // user's maximum HP through the direct-damage pipeline and sets
            // Attack to +6 (the reference boosts by 12 stages, which clamps).
            if self.mon(actor).boosts[0] >= 6
                || u32::from(self.mon(actor).hp) * 2 <= u32::from(self.mon(actor).stats[0])
                || self.mon(actor).stats[0] == 1
            {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            self.prepare_hit_abilities(dex, actor, m)?;
            let cost = self.mon(actor).stats[0] / 2;
            self.indirect_damage(dex, actor, actor, u32::from(cost), EffectRef::Move(move_id))?;
            let mut atk = [0i8; 7];
            atk[0] = 12;
            self.boost(dex, actor, actor, atk, BoostCause::Move { secondary: false })?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::RevivalBlessing {
            // `moves:revivalblessing.onTryHit` (`PrepareHit` already ran for
            // the ability hooks): the move fails outright unless the side has a
            // fainted party member.
            if !self.sides[actor.side as usize]
                .pokemon
                .iter()
                .any(|p| p.selected && p.fainted)
            {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            // The move's `slotCondition` lands on the user's slot and its
            // `selfSwitch` marks the user, which turns the post-action request
            // into the revive choice. The user itself stays on the field when
            // the choice is committed.
            let slot = self
                .mon(actor)
                .active_slot
                .expect("revival blessing user is active");
            let order = self.allocate_effect_order()?;
            self.sides[actor.side as usize].slot_conditions[slot as usize].insert(
                dex.effects.revival_blessing,
                EffectState {
                    id: dex.effects.revival_blessing,
                    duration: Some(1),
                    effect_order: order,
                    effect_order_assigned: true,
                    source: Some((
                        if actor.side == 0 {
                            SideId::P1
                        } else {
                            SideId::P2
                        },
                        actor.roster,
                    )),
                    ..Default::default()
                },
            );
            self.mon_mut(actor).switch_flag = Some(move_id);
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::TrickRoom {
            self.toggle_trick_room(dex, actor)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::Gravity {
            // Reference `useMoveInner`: an `all`-target move goes through
            // `tryMoveHit`, so the pseudo-weather is added by `runMoveEffects`
            // without the move-loop Update pair. A second cast while Gravity is
            // up fails without touching state.
            let started = self.start_gravity(dex, actor)?;
            self.mon_mut(actor).move_this_turn_result = if started {
                MoveResult::Success
            } else {
                MoveResult::Failed
            };
            return Ok(());
        }
        if behavior == MoveBehavior::Weather {
            self.start_weather(dex, actor, m.weather, false)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::ChillyReception {
            // Reference `useMoveInner`: an `all`-target move runs `tryMoveHit`,
            // i.e. the `TryHitField` event (no handlers in the pinned data)
            // followed by `moveHit` against the first resolved target, which
            // applies the `weather` field and the inline `selfSwitch` gate.
            // This path never runs the move-loop Update pair - only the
            // post-action Update the queue loop already performs.
            self.start_weather(dex, actor, m.weather, false)?;
            if self.can_switch(actor.side as usize)
                && !self.mon(actor).volatiles.contains_key(&dex.effects.commanded)
            {
                self.mon_mut(actor).switch_flag = Some(move_id);
            }
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
            let priority = m.priority.unwrap_or_else(|| self.effective_priority(dex, actor, move_id));
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
        if behavior == MoveBehavior::CourtChange {
            // `moves:courtchange.onHitField` (a field move, so it bypasses the
            // hit loop like Haze): every listed side condition moves across to
            // the other side, keeping its state object. With neither side
            // holding one the move fails. Both sides' conditions are public, so
            // the knowledge entries move with them.
            let mine = actor.side as usize;
            let foe = 1 - mine;
            let swappable = [
                dex.effects.spikes,
                dex.effects.toxic_spikes,
                dex.effects.stealth_rock,
                dex.effects.sticky_web,
                dex.effects.reflect,
                dex.effects.light_screen,
                dex.effects.aurora_veil,
                dex.effects.safeguard,
                dex.effects.tailwind,
            ];
            let mut swapped = false;
            for id in swappable {
                let own = self.sides[mine].conditions.remove(&id);
                let theirs = self.sides[foe].conditions.remove(&id);
                if own.is_none() && theirs.is_none() {
                    continue;
                }
                swapped = true;
                if let Some(state) = own {
                    self.sides[foe].conditions.insert(id, state);
                }
                if let Some(state) = theirs {
                    self.sides[mine].conditions.insert(id, state);
                }
                for viewer in 0..2 {
                    let rel_mine = usize::from(mine != viewer);
                    let rel_foe = usize::from(foe != viewer);
                    let moved = self.knowledge[viewer].sides[rel_mine].remove(&id);
                    let other = self.knowledge[viewer].sides[rel_foe].remove(&id);
                    if let Some(entry) = moved {
                        self.knowledge[viewer].sides[rel_foe].insert(id, entry);
                    }
                    if let Some(entry) = other {
                        self.knowledge[viewer].sides[rel_mine].insert(id, entry);
                    }
                }
            }
            self.mon_mut(actor).move_this_turn_result = if swapped {
                MoveResult::Success
            } else {
                MoveResult::Failed
            };
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
            self.prepare_hit_abilities(dex, actor, m)?;
            self.ally_try_hit_side(dex, actor, actor, m)?;
            self.start_side_condition(dex, actor, m.side_condition)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            return Ok(());
        }
        if behavior == MoveBehavior::FoeHazard {
            // `tryMoveHit` for a `foeSide` move: the `TryHitSide` event's
            // target is the opposing side's first active Pokémon. Magic Bounce
            // may reflect the hazard back at the user; otherwise the side
            // condition is added (or a capped restart fails the move). Like
            // the other side-target moves no hit loop or Update runs.
            self.prepare_hit_abilities(dex, actor, m)?;
            let foe = (1 - actor.side) as usize;
            let target = self.sides[foe]
                .active
                .iter()
                .flatten()
                .next()
                .map(|roster| Entity {
                    side: foe as u8,
                    roster: *roster,
                });
            let reflected = match target {
                Some(target) => self.ally_try_hit_side(dex, actor, target, m)?,
                None => false,
            };
            let added = if reflected {
                false
            } else {
                self.add_side_hazard(dex, foe, actor, m.side_condition)?
            };
            self.mon_mut(actor).move_this_turn_result = if added {
                MoveResult::Success
            } else {
                MoveResult::Failed
            };
            return Ok(());
        }
        if behavior == MoveBehavior::Stockpile {
            // `moves:stockpile.onTry`: a fourth layer fails before any hit
            // step. Otherwise the volatile starts (layers 1) or restarts
            // (layers + 1), announces itself and raises Defense and Special
            // Defense one stage each, recording every raise that actually
            // changed a stage so `onEnd` can reverse exactly those.
            if self.stockpile_layers(dex, actor) >= 3 {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            self.prepare_hit_abilities(dex, actor, m)?;
            self.start_stockpile(dex, actor)?;
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::Swallow {
            // `moves:swallow.onTry|onHit`: requires a stored stockpile, heals
            // a quarter, half or all of the user's maximum HP and always
            // consumes the volatile. A refused heal (full HP or Heal Block)
            // still consumes it and leaves `moveThisTurnResult` null.
            if self.stockpile_layers(dex, actor) == 0 {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            self.prepare_hit_abilities(dex, actor, m)?;
            let healed = self.swallow_heal(dex, actor)?;
            self.mon_mut(actor).move_this_turn_result = if healed {
                MoveResult::Success
            } else {
                MoveResult::Undefined
            };
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::AllySwitch {
            // `moves:allyswitch.onPrepareHit`: add (or restart) the
            // `allyswitch` volatile before any hit step. A fresh volatile
            // stores `counter = 3`; a restart runs the reference escalating
            // success roll `randomChance(1, counter)` and removes the volatile
            // when the roll fails. A failed prepare aborts the move before the
            // hit loop, so neither Update event runs.
            let volatile = dex.effects.ally_switch;
            let existing = self.mon(actor).volatiles.get(&volatile).cloned();
            let prepared = match existing {
                Some(mut state) => {
                    let counter = u32::try_from(state.values.first().copied().unwrap_or(1))
                        .unwrap_or(1)
                        .max(1);
                    if self.rng.chance(1, counter) {
                        // `counterMax` is 729; the stored counter only grows
                        // while it is still below the cap.
                        if counter < 729 {
                            state.values[0] = i64::from(counter * 3);
                        }
                        state.duration = Some(2);
                        self.mon_mut(actor).volatiles.insert(volatile, state);
                        true
                    } else {
                        self.mon_mut(actor).volatiles.remove(&volatile);
                        false
                    }
                }
                None => {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(actor).volatiles.insert(
                        volatile,
                        EffectState {
                            id: volatile,
                            duration: Some(2),
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
                            values: vec![3],
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        actor,
                        Some(actor),
                        EffectRef::Condition(volatile),
                        0,
                        false,
                    )?;
                    true
                }
            };
            if !prepared {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            // The reference's move-owned PrepareHit precedes the ability
            // PrepareHit, so Protean/Libero only run once the volatile stuck.
            self.prepare_hit_abilities(dex, actor, m)?;
            // `moves:allyswitch.onHit`: doubles only. The user swaps slots
            // with its partner when that slot holds a living Pokémon;
            // otherwise the move reports `NOT_FAIL` (the hit loop still runs
            // both Update events, but `moveThisTurnResult` stays null).
            let slot = self.mon(actor).active_slot.unwrap_or(0);
            let other_slot = 1 - slot;
            let other = self.sides[actor.side as usize].active[other_slot as usize];
            let swap = other.filter(|roster| {
                !self.mon(Entity {
                    side: actor.side,
                    roster: *roster,
                })
                .fainted
            });
            if let Some(other_roster) = swap {
                let side = &mut self.sides[actor.side as usize];
                side.active[slot as usize] = Some(other_roster);
                side.active[other_slot as usize] = Some(actor.roster);
                side.positions
                    .swap(slot as usize, other_slot as usize);
                self.mon_mut(actor).active_slot = Some(other_slot);
                self.mon_mut(Entity {
                    side: actor.side,
                    roster: other_roster,
                })
                .active_slot = Some(slot);
                self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            } else {
                // `onHit` returns `NOT_FAIL`: the move does not count as a
                // failure, so Stomping Tantrum keeps reading a null result.
                self.mon_mut(actor).move_this_turn_result = MoveResult::Undefined;
            }
            // The single-target hit loop runs one Update per hit plus the
            // trailing Update, exactly like the other self-target volatiles.
            self.each_update(dex)?;
            self.each_update(dex)?;
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
            self.prepare_hit_abilities(dex, actor, m)?;
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
            self.prepare_hit_abilities(dex, actor, m)?;
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
        if behavior == MoveBehavior::Substitute {
            // `moves:substitute`: a self-targeting status move. `onTryHit`
            // refuses an existing decoy and a user that cannot pay (hp at or
            // below maxHP/4, or the Shedinja clause maxHP === 1). The
            // condition's `onStart` then stores floor(maxHP/4) as the decoy's
            // HP and ends any `partiallytrapped` volatile; `onHit` pays the
            // same floor(maxHP/4) as direct damage.
            let max_hp = u32::from(self.mon(actor).stats[0]);
            let hp = u32::from(self.mon(actor).hp);
            if self
                .mon(actor)
                .volatiles
                .contains_key(&dex.effects.substitute)
                || max_hp == 1
                || hp <= max_hp / 4
            {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            let sub_hp = max_hp / 4;
            let order = self.allocate_effect_order()?;
            self.mon_mut(actor).volatiles.insert(
                dex.effects.substitute,
                EffectState {
                    id: dex.effects.substitute,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: vec![i64::from(sub_hp)],
                    ..Default::default()
                },
            );
            self.emit(
                EventKind::EffectStart,
                actor,
                None,
                EffectRef::Condition(dex.effects.substitute),
                0,
                false,
            )?;
            if self
                .mon_mut(actor)
                .volatiles
                .remove(&dex.effects.partially_trapped)
                .is_some()
            {
                self.emit(
                    EventKind::EffectEnd,
                    actor,
                    None,
                    EffectRef::Condition(dex.effects.partially_trapped),
                    0,
                    false,
                )?;
            }
            // `onHit` direct damage: no Damage-event clamps (Sturdy, Focus
            // Sash and Endure cannot refuse the cost); the TryHit gate already
            // proved hp > maxHP/4, so this cannot faint the user.
            let cost = (max_hp / 4).min(hp);
            if cost > 0 {
                self.mon_mut(actor).hp -= cost as u16;
                self.emit(
                    EventKind::Damage,
                    actor,
                    Some(actor),
                    EffectRef::Move(move_id),
                    -(cost as i32),
                    true,
                )?;
            }
            // A self-target status move that did anything runs the two
            // move-loop Update events, exactly like the Protect family.
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        if behavior == MoveBehavior::ShedTail {
            // `moves:shedtail.onTryHit`: the move fails before any hit step
            // when the user cannot switch, is commanded, already carries a
            // decoy, or cannot pay half its maximum HP (the check is `<=
            // ceil(maxHP/2)`, so exactly half is refused).
            let max_hp = u32::from(self.mon(actor).stats[0]);
            let hp = u32::from(self.mon(actor).hp);
            if !self.can_switch(actor.side as usize)
                || self.mon(actor).volatiles.contains_key(&dex.effects.commanded)
                || self.mon(actor).volatiles.contains_key(&dex.effects.substitute)
                || hp <= max_hp.div_ceil(2)
            {
                self.mon_mut(actor).move_this_turn_result = MoveResult::Failed;
                return Ok(());
            }
            // `volatileStatus: 'substitute'` runs first: the shared condition's
            // `onStart` stores floor(maxHP/4) as the decoy's HP and ends any
            // `partiallytrapped` volatile.
            let sub_hp = max_hp / 4;
            let order = self.allocate_effect_order()?;
            self.mon_mut(actor).volatiles.insert(
                dex.effects.substitute,
                EffectState {
                    id: dex.effects.substitute,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: vec![i64::from(sub_hp)],
                    ..Default::default()
                },
            );
            self.emit(
                EventKind::EffectStart,
                actor,
                None,
                EffectRef::Condition(dex.effects.substitute),
                0,
                false,
            )?;
            if self
                .mon_mut(actor)
                .volatiles
                .remove(&dex.effects.partially_trapped)
                .is_some()
            {
                self.emit(
                    EventKind::EffectEnd,
                    actor,
                    None,
                    EffectRef::Condition(dex.effects.partially_trapped),
                    0,
                    false,
                )?;
            }
            // `moves:shedtail.onHit`: `directDamage(ceil(maxHP/2))`, a raw
            // subtraction with no Damage-event clamps. The gate proved
            // `hp > ceil(maxHP/2)`, so this cannot faint the user.
            let cost = max_hp.div_ceil(2).min(hp);
            if cost > 0 {
                self.mon_mut(actor).hp -= cost as u16;
                self.emit(
                    EventKind::Damage,
                    actor,
                    Some(actor),
                    EffectRef::Move(move_id),
                    -(cost as i32),
                    true,
                )?;
            }
            // `selfSwitch: 'shedtail'`: the pivot flag; the replacement copy
            // carries only the decoy across.
            self.mon_mut(actor).switch_flag = Some(move_id);
            self.mon_mut(actor).move_this_turn_result = MoveResult::Success;
            self.each_update(dex)?;
            self.each_update(dex)?;
            return Ok(());
        }
        let spread = targets.len() > 1;
        // PrepareHit abilities run once per action, before both the multi-hit
        // dispatch and the single-hit steps. (Protean/Libero already ran at
        // the shared PrepareHit point above.)
        let preparer = dex.effects.abilities[self.mon(actor).ability as usize];
        // `abilities:parentalbond.onPrepareHit`: a single-target, non-status,
        // non-charge, non-future, non-multi-hit damaging move gains a second
        // hit at a quarter power.
        let parental_bond = preparer == Ability::Parentalbond
            && m.category != Category::Status
            && m.multihit.is_none()
            && m.allies.is_empty()
            && !m.no_parental_bond
            && m.charge.is_none()
            && !m.future_move
            && !spread
            && !m.is_z
            && !m.is_max;
        if m.multihit.is_some() || !m.allies.is_empty() || parental_bond {
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
        let mut hit = SmallVec::<[(Entity, i8); 4]>::new();
        // Reference `spreadMoveHit` target bookkeeping: a protection block is
        // the `NOT_FAIL` case (recorded as `null`), while a type immunity or a
        // missed accuracy roll is a real failure (`false`).
        let mut blocked_by_protection = false;
        // `moves:healingwish.onTryHit` returns the reference's `NOT_FAIL`: the
        // attempt is recorded as skipped rather than failed.
        let mut refused_not_fail = false;
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
            if m.protect && !m.breaks_protect && !self.ability_bypasses_protect(dex, actor, m) {
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
            // `moves:ragingbull.onTryHit`: the move shatters the target side's
            // screens as part of the TryHit step, i.e. after the priority-3
            // protection guards and the priority-1 ability absorptions (which
            // break the reference event before the move's own handler runs)
            // and before type immunity, accuracy and the decoy intercept.
            if m.hooks & crate::effects::hook::RAGING_BULL != 0 {
                for id in [
                    dex.effects.reflect,
                    dex.effects.light_screen,
                    dex.effects.aurora_veil,
                ] {
                    if self.sides[target.side as usize].conditions.remove(&id).is_some() {
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
            } else if m.move_type == dex.effects.ground
                && ability == Ability::Levitate
                && !self.suppressing_ability(dex, actor, target, m)
                && !self.grounded(dex, target)
            {
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
                // `Pokemon#runImmunity`: for a Ground move the immunity is
                // decided by `isGrounded`, so a grounded target's Flying type
                // no longer blocks the move (Gravity, Smack Down, Ingrain).
                let grounded = self.grounded(dex, target);
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
                        if value == -127
                            && !(m.move_type == dex.effects.ground
                                && kind == dex.effects.flying
                                && grounded)
                        {
                            None
                        } else {
                            Some(total + if value == -127 { 0 } else { value })
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
            // `moves:octolock.onTryImmunity`: `getImmunity('trapped', ...)`
            // refuses the marker against a Ghost-type target.
            if behavior == MoveBehavior::Octolock
                && self.mon(target).types.contains(&dex.effects.ghost)
            {
                failed_otherwise = true;
                continue;
            }
            // `moves:endeavor.onTryImmunity`: the move is refused outright
            // unless the user's HP is strictly below the target's.
            if m.fixed_damage == Some(crate::assets::FixedDamage::Endeavor)
                && self.mon(actor).hp >= self.mon(target).hp
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
            let Some(accuracy) =
                self.modify_accuracy(dex, actor, *target, action_accuracy, m.minimize)
            else {
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
        // `moves:entrainment.onTryHit`: a different, replaceable target ability
        // and a user ability without `noentrain`.
        if hooks & crate::effects::hook::ENTRAINMENT != 0 {
            let source_ability = self.mon(actor).ability;
            let before = hit.len();
            hit.retain(|(target, _)| {
                let target_ability = self.mon(*target).ability;
                *target != actor
                    && target_ability != source_ability
                    && !dex.effects.no_suppress_abilities[target_ability as usize]
                    && target_ability != dex.effects.truant_ability
                    && !dex.effects.no_entrain_abilities[source_ability as usize]
            });
            failed_otherwise |= hit.len() != before;
        }
        // `moves:roleplay.onTryHit`: the target's ability must differ from the
        // user's, must not carry `failroleplay`, and the user's must be
        // replaceable.
        if hooks & crate::effects::hook::ROLE_PLAY != 0 {
            let source_ability = self.mon(actor).ability;
            let before = hit.len();
            hit.retain(|(target, _)| {
                let target_ability = self.mon(*target).ability;
                target_ability != source_ability
                    && !dex.effects.fail_role_play_abilities[target_ability as usize]
                    && !dex.effects.no_suppress_abilities[source_ability as usize]
            });
            failed_otherwise |= hit.len() != before;
        }
        // `moves:simplebeam.onTryHit`: the target's ability must be
        // replaceable and neither Simple nor Truant.
        if hooks & crate::effects::hook::SIMPLE_BEAM != 0 {
            let before = hit.len();
            hit.retain(|(target, _)| {
                let target_ability = self.mon(*target).ability;
                !dex.effects.no_suppress_abilities[target_ability as usize]
                    && target_ability != dex.effects.simple_ability
                    && target_ability != dex.effects.truant_ability
            });
            failed_otherwise |= hit.len() != before;
        }
        failed_otherwise |= missed_accuracy;
        if hit.is_empty() {
            // Reference `useMoveInner` runs the move-owned `onMoveFail` before
            // the action ends. Steel Beam pays its half-maximum-HP recoil here
            // too, so a miss or a Protect block still damages the user.
            if m.mind_blown_recoil || m.has_crash_damage {
                // Steel Beam rounds its half-maximum recoil; the crash-damage
                // family hands `baseMaxhp / 2` to the truncating damage path.
                let max = u32::from(self.mon(actor).stats[0]);
                let recoil = if m.has_crash_damage {
                    max / 2
                } else {
                    stats::round_fraction(max, [1, 2])
                };
                let hp_before = self.mon(actor).hp;
                self.indirect_damage(dex, actor, actor, recoil, EffectRef::Move(move_id))?;
                self.emergency_exit_check(dex, actor, hp_before)?;
            }
            // `moves:spitup.onAfterMove`: `runMove` always fires AfterMove
            // once `useMove` was reached, so a missed or blocked Spit Up still
            // consumes the user's stockpile.
            if m.hooks & crate::effects::hook::SPIT_UP != 0 {
                self.remove_stockpile(dex, actor)?;
            }
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
        // Targets whose primary hit the decoy consumed. Every per-target
        // effect phase below runs without them, exactly like the reference's
        // `runMoveEffects`/`selfDrops`/`forceSwitch` null-target handling.
        let mut sub_absorbed = SmallVec::<[Entity; 4]>::new();
        let mut damages = SmallVec::<[(Entity, u16); 4]>::new();
        // Reference `spreadMoveHit` runs `tryPrimaryHitEvent` before
        // `getSpreadDamage`: every decoy handler resolves its damage in a
        // pre-pass over the whole target list, so a decoy-absorbed target's
        // crit/randomizer draws are spent before the other targets'. Iterating
        // those targets first reproduces the draw order without changing any
        // per-target computation.
        let decoys: SmallVec<[Entity; 4]> = hit
            .iter()
            .filter(|(target, _)| {
                self.decoy_absorbs(dex, actor, *target, m, m.infiltrates || pollen_ally)
            })
            .map(|(target, _)| *target)
            .collect();
        let ordered: SmallVec<[(Entity, i8); 4]> = hit
            .iter()
            .copied()
            .filter(|(target, _)| decoys.contains(target))
            .chain(
                hit.iter()
                    .copied()
                    .filter(|(target, _)| !decoys.contains(target)),
            )
            .collect();
        for (target, effectiveness) in ordered {
            // `moves:pollenpuff.onTryHit` set `basePower = 0` for the ally
            // case: `getDamage` returns `undefined` before any crit or
            // randomizer draw, so the heal phase below is the only effect.
            if pollen_ally && target.side == actor.side {
                continue;
            }
            if m.category == Category::Status {
                continue;
            }
            // `moves:poltergeist.onTryHit`: the move-owned handler runs at the
            // start of the damage step, after protection, type immunity and
            // accuracy have passed, and publicly reveals the held item.
            if hooks & crate::effects::hook::POLTERGEIST != 0 {
                self.emit(
                    EventKind::Item,
                    target,
                    Some(actor),
                    EffectRef::Item(self.mon(target).item),
                    0,
                    false,
                )?;
            }
            // OHKO and fixed-damage moves resolve before the damage kernel and
            // therefore consume no critical-hit or damage randomizer draws.
            if let Some(amount) = self.fixed_damage_amount(dex, m, actor, target) {
                // Final Gambit's callback faints the user while resolving the
                // damage amount, before the damage is applied.
                if m.fixed_damage == Some(crate::assets::FixedDamage::UserHp) {
                    self.faint_now(actor);
                }
                let amount = amount.min(u32::from(u16::MAX)) as u16;
                if self.intercept_substitute(dex, actor, target, m, amount, m.infiltrates)? {
                    sub_absorbed.push(target);
                } else {
                    damages.push((target, amount));
                }
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
            let base_power = self.base_power(dex, m, actor, target, 1);
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
                    weather_modifier: self.weather_damage_modifier(dex, actor, target, m.move_type),
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
            // `abilities:piercingdrill|unseenfist` quarter the damage of a hit
            // whose protection was bypassed.
            let bypassed =
                self.protection_bypassed(dex, actor, target, m, effective_priority);
            let damage = damage::finish_damage(damage, final_modifier, bypassed);
            // The decoy call in the reference still resolves the full damage
            // (identical RNG draws) and then eats it instead of the target.
            if self.intercept_substitute(dex, actor, target, m, damage, m.infiltrates || pollen_ally)? {
                sub_absorbed.push(target);
            } else {
                damages.push((target, damage));
            }
        }
        let effect_targets: SmallVec<[Entity; 4]> = hit_targets
            .iter()
            .copied()
            .filter(|target| !sub_absorbed.contains(target))
            .collect();
        let mut total_damage = 0u32;
        let mut hit_any = false;
        // Reference `hitStepMoveHitLoop` passes `hurtThisTurn + curDamage`,
        // i.e. each damaged target's HP before this move's damage, into the
        // Emergency Exit check at the end of the action.
        let mut hit_before: SmallVec<[(Entity, u16); 4]> = SmallVec::new();
        // Per-target damage for the `DamagingHit` event, parallel to
        // `effect_targets` (the reference passes `damagedDamage`).
        let mut hit_damages: SmallVec<[u16; 4]> = SmallVec::new();
        for (target, damage) in damages {
            // `abilities:disguise.onDamage` (onDamagePriority 1, so it runs
            // before Sturdy, Focus Sash and Endure): the first damaging move
            // against an undisguised Mimikyu is absorbed - the hit still lands
            // and counts for `timesAttacked`, but no HP is lost - and the
            // holder is marked for the forme change at the next Update.
            if target != actor
                && dex.effects.abilities[self.mon(target).ability as usize]
                    == Ability::Disguise
                && !self.suppressing_ability(dex, actor, target, m)
                && (self.mon(target).species == dex.effects.mimikyu
                    || self.mon(target).species == dex.effects.mimikyu_totem)
            {
                self.reveal_ability(target)?;
                self.mon_mut(target).disguise_busted = true;
                hit_any = true;
                let count = self.mon(target).times_attacked;
                self.mon_mut(target).times_attacked = count.saturating_add(1);
                // `abilities:disguise.onDamage` sets the damage to 0 rather
                // than false, so the hit still records a `gotAttacked` entry.
                self.record_attacked_by(target, actor, m.move_uid, 0);
                hit_damages.push(0);
                hit_before.push((target, self.mon(target).hp));
                self.emit(
                    EventKind::Damage,
                    target,
                    Some(actor),
                    EffectRef::Move(move_id),
                    0,
                    true,
                )?;
                continue;
            }
            // `endure` clamps after item/berry damage modification and before
            // the damage is applied (reference `onDamage` priority -10).
            let damage = self.sturdy_clamp(
                dex,
                target,
                damage,
                self.suppressing_ability(dex, actor, target, m),
            )?;
            let damage = self.damage_item(dex, target, damage)?;
            let damage = self.endure_clamp(dex, target, damage);
            let hp_before = self.mon(target).hp;
            hit_before.push((target, hp_before));
            let actual = damage.min(hp_before);
            hit_damages.push(actual);
            total_damage += u32::from(actual);
            hit_any = true;
            self.mon_mut(target).hp -= actual;
            if actual != 0 {
                let hp = self.mon(target).hp;
                self.mon_mut(target).hurt_this_turn = hp;
            }
            // Reference `hitStepMoveHitLoop`: a landed hit increments the
            // target's `timesAttacked`, even when it dealt zero damage.
            if target != actor {
                let count = self.mon(target).times_attacked;
                self.mon_mut(target).times_attacked = count.saturating_add(1);
                // Reference `gotAttacked` records one entry per move with the
                // move's accumulated damage (merged across this move's hits).
                self.record_attacked_by(target, actor, m.move_uid, actual);
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
                    EffectRef::Condition(dex.effects.drain),
                )?;
            }
        }
        // A status move the decoy ate still counts as having done something
        // (`HIT_SUBSTITUTE` is truthy in `runMoveEffects`), so the action is a
        // success rather than a failure.
        let mut did_anything =
            m.category != Category::Status || !sub_absorbed.is_empty();
        for &target in &effect_targets {
            // Status moves also run the reference's TryPrimaryHit decoy stage
            // before `runMoveEffects`: a decoy absorbs the whole move (zero
            // damage) and the action still counts as a success.
            if m.category == Category::Status
                && self.intercept_substitute(dex, actor, target, m, 0, m.infiltrates)?
            {
                sub_absorbed.push(target);
                did_anything = true;
                continue;
            }
            // `moves:fling.onPrepareHit`'s per-item `move.onHit`: the thrown
            // Berry is eaten by the target through its `onEat`, and the two
            // herb callbacks clear the target's volatiles or negative boosts.
            // All three apply per target right after its damage, before the
            // action's `secondaries` phase.
            if let Some((item, kind)) = m.fling {
                match kind {
                    crate::effects::FlingKind::Berry => {
                        self.eat_berry(dex, target, item)?;
                    }
                    crate::effects::FlingKind::MentalHerb => {
                        let mut conditions = vec![
                            dex.effects.taunt,
                            dex.effects.encore,
                            dex.effects.torment,
                            dex.effects.disable,
                            dex.effects.heal_block,
                        ];
                        // The Attract move itself stays unported, so its
                        // volatile is looked up opportunistically.
                        if let Ok(attract) = dex.id("conditions", "attract") {
                            conditions.push(attract);
                        }
                        if conditions
                            .iter()
                            .any(|id| self.mon(target).volatiles.contains_key(id))
                        {
                            for id in conditions {
                                if self.mon_mut(target).volatiles.remove(&id).is_some() {
                                    self.emit(
                                        EventKind::EffectEnd,
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
                    crate::effects::FlingKind::WhiteHerb => {
                        let boosts = self.mon(target).boosts;
                        if boosts.iter().any(|b| *b < 0) {
                            let mon = self.mon_mut(target);
                            for boost in mon.boosts.iter_mut() {
                                if *boost < 0 {
                                    *boost = 0;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            if behavior == MoveBehavior::Trick {
                // Trick/Switcheroo decide success themselves: the empty generic
                // payload must not mark a refused swap as "did anything", or
                // the failed move would run the reference's post-move phases.
                did_anything |= self.trick_swap(dex, actor, target)?;
            } else if behavior == MoveBehavior::SkillSwap {
                // `moves:skillswap.onHit`: the whole move succeeds or fails on
                // the exchange result, so the empty generic payload must not
                // mark a refused swap as "did anything".
                did_anything |= self.skill_swap(dex, actor, target)?;
            } else if behavior == MoveBehavior::StatSwap {
                // `moves:powerswap|guardswap.onHit`: `setBoost` writes the two
                // stages directly (no TryBoost hooks, so Defiant stays quiet)
                // and both sides' new stages are public. Power Swap moves
                // Attack/Sp. Atk (boost indexes 0 and 2), Guard Swap
                // Defense/Sp. Def (1 and 3).
                let stats: [usize; 2] = if m.id == dex.effects.power_swap_move {
                    [0, 2]
                } else {
                    [1, 3]
                };
                let actor_old = self.mon(actor).boosts;
                let target_old = self.mon(target).boosts;
                for stat in stats {
                    let actor_new = target_old[stat];
                    let target_new = actor_old[stat];
                    self.mon_mut(actor).boosts[stat] = actor_new;
                    self.mon_mut(target).boosts[stat] = target_new;
                    let actor_delta = actor_new - actor_old[stat];
                    let target_delta = target_new - target_old[stat];
                    if actor_delta != 0 {
                        self.emit(
                            EventKind::Boost,
                            actor,
                            Some(target),
                            EffectRef::Stat(stat as Id),
                            i32::from(actor_delta),
                            false,
                        )?;
                    }
                    if target_delta != 0 {
                        self.emit(
                            EventKind::Boost,
                            target,
                            Some(actor),
                            EffectRef::Stat(stat as Id),
                            i32::from(target_delta),
                            false,
                        )?;
                    }
                }
                did_anything = true;
            } else if matches!(behavior, MoveBehavior::PowerTrick | MoveBehavior::PowerShift) {
                // `moves:powertrick|powershift.condition`: a self volatile that
                // swaps the stored Attack and Defense. `onRestart` removes the
                // marker (running `onEnd`, which swaps them back), so re-using
                // the move cancels it; a switch-out recomputes the base stats.
                let volatile = if behavior == MoveBehavior::PowerShift {
                    dex.effects.power_shift
                } else {
                    dex.effects.power_trick
                };
                if self.mon(target).volatiles.contains_key(&volatile) {
                    self.mon_mut(target).volatiles.remove(&volatile);
                    self.emit(
                        EventKind::EffectEnd,
                        target,
                        None,
                        EffectRef::Condition(volatile),
                        0,
                        false,
                    )?;
                } else {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        volatile,
                        EffectState {
                            id: volatile,
                            effect_order: order,
                            effect_order_assigned: true,
                            source: Some((
                                if target.side == 0 {
                                    SideId::P1
                                } else {
                                    SideId::P2
                                },
                                target.roster,
                            )),
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
                let atk = self.mon(target).stats[1];
                let def = self.mon(target).stats[2];
                self.mon_mut(target).stats[1] = def;
                self.mon_mut(target).stats[2] = atk;
                did_anything = true;
            } else if matches!(behavior, MoveBehavior::PowerSplit | MoveBehavior::GuardSplit) {
                // `moves:powersplit|guardsplit.onHit`: both stored stats become
                // their floored average. The change is not a volatile, so a
                // switch-out recomputes the base stats like the reference's
                // `setSpecies`.
                let (first, second) = if behavior == MoveBehavior::PowerSplit {
                    (1usize, 3usize)
                } else {
                    (2usize, 4usize)
                };
                for index in [first, second] {
                    let averaged = (u32::from(self.mon(actor).stats[index])
                        + u32::from(self.mon(target).stats[index]))
                        / 2;
                    let averaged = averaged.min(u32::from(u16::MAX)) as u16;
                    self.mon_mut(actor).stats[index] = averaged;
                    self.mon_mut(target).stats[index] = averaged;
                }
                did_anything = true;
            } else if behavior == MoveBehavior::SpeedSwap {
                // `moves:speedswap.onHit`: the two stored Speed stats swap. The
                // cached `speed` stays stale until the next `updateSpeed`,
                // exactly like the reference's direct `storedStats.spe` write.
                let actor_spe = self.mon(actor).stats[5];
                let target_spe = self.mon(target).stats[5];
                self.mon_mut(actor).stats[5] = target_spe;
                self.mon_mut(target).stats[5] = actor_spe;
                did_anything = true;
            } else if behavior == MoveBehavior::TopsyTurvy {
                // `moves:topsyturvy.onHit`: every nonzero stage flips sign and
                // the whole move reports failure when nothing changed.
                let boosts = self.mon(target).boosts;
                let mut inverted = false;
                for (stat, old) in boosts.iter().copied().enumerate() {
                    if old == 0 {
                        continue;
                    }
                    self.mon_mut(target).boosts[stat] = -old;
                    inverted = true;
                    self.emit(
                        EventKind::Boost,
                        target,
                        Some(actor),
                        EffectRef::Stat(stat as Id),
                        -2 * i32::from(old),
                        false,
                    )?;
                }
                did_anything |= inverted;
            } else if behavior == MoveBehavior::ClearSmog {
                // `moves:clearsmog.onHit`: the damaging hit lands first and the
                // target's stages reset afterwards.
                let boosts = self.mon(target).boosts;
                for (stat, old) in boosts.iter().copied().enumerate() {
                    if old == 0 {
                        continue;
                    }
                    self.mon_mut(target).boosts[stat] = 0;
                    self.emit(
                        EventKind::Boost,
                        target,
                        Some(actor),
                        EffectRef::Stat(stat as Id),
                        -i32::from(old),
                        false,
                    )?;
                }
                did_anything = true;
            } else if behavior == MoveBehavior::TrapTarget {
                // `moves:block|meanlook.onHit`: the `trapped` volatile is added
                // with the user as its source and the add's result is the
                // move's success. A Ghost still receives the marker (and its
                // public activation) but stays immune to the actual trap.
                did_anything |= self.start_selection_volatile(
                    dex,
                    target,
                    Some(actor),
                    dex.effects.trapped,
                    false,
                    false,
                )?;
            } else if behavior == MoveBehavior::JawLock {
                // `moves:jawlock.onHit`: both sides are pinned (each sourced by
                // the other); the landed damage already marks the move.
                self.start_selection_volatile(
                    dex,
                    target,
                    Some(actor),
                    dex.effects.trapped,
                    false,
                    false,
                )?;
                self.start_selection_volatile(
                    dex,
                    actor,
                    Some(target),
                    dex.effects.trapped,
                    false,
                    false,
                )?;
                did_anything = true;
            } else if behavior == MoveBehavior::AquaRing {
                // `moves:aquaring`: the self volatile announces itself and
                // heals a sixteenth of the maximum HP every residual. A
                // repeated use fails (the condition declares no `onRestart`).
                if !self
                    .mon(target)
                    .volatiles
                    .contains_key(&dex.effects.aqua_ring)
                {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        dex.effects.aqua_ring,
                        EffectState {
                            id: dex.effects.aqua_ring,
                            source: Some((
                                if target.side == 0 {
                                    SideId::P1
                                } else {
                                    SideId::P2
                                },
                                target.roster,
                            )),
                            effect_order: order,
                            effect_order_assigned: true,
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        None,
                        EffectRef::Condition(dex.effects.aqua_ring),
                        0,
                        false,
                    )?;
                    did_anything = true;
                }
            } else if behavior == MoveBehavior::Wish {
                // `moves:wish.condition.onStart`: the wisher's slot remembers
                // half of the user's maximum HP and the turn it started. The
                // slot condition resolves on the next turn's residual (order
                // 4) and heals whoever occupies the slot then.
                let slot = self.mon(target).active_slot.unwrap_or(0);
                let side = actor.side as usize;
                // `addSlotCondition` refuses while the marker exists (the
                // condition declares no `onRestart`), which fails the move.
                if !self.sides[side].slot_conditions[slot as usize].contains_key(&dex.effects.wish)
                {
                    let half = i64::from(self.mon(actor).stats[0]) / 2;
                    let order = self.allocate_effect_order()?;
                    self.sides[side].slot_conditions[slot as usize].insert(
                        dex.effects.wish,
                        EffectState {
                            id: dex.effects.wish,
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
                            values: vec![half, i64::from(self.turn % 256)],
                            ..Default::default()
                        },
                    );
                    did_anything = true;
                }
            } else if behavior == MoveBehavior::HealingWish {
                // `moves:healingwish.onTryHit`: without a reserve to switch in,
                // the move is refused with the reference's `NOT_FAIL` and the
                // user stays alive (the `ifHit` self-destruct never runs).
                // Otherwise the wisher's slot is marked for the replacement,
                // which is fully healed and cured as it enters.
                if !self.can_switch(actor.side as usize) {
                    refused_not_fail = true;
                    continue;
                }
                let slot = self.mon(target).active_slot.unwrap_or(0);
                let side = actor.side as usize;
                if !self.sides[side].slot_conditions[slot as usize]
                    .contains_key(&dex.effects.healing_wish)
                {
                    let order = self.allocate_effect_order()?;
                    self.sides[side].slot_conditions[slot as usize].insert(
                        dex.effects.healing_wish,
                        EffectState {
                            id: dex.effects.healing_wish,
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
                    did_anything = true;
                }
            } else if behavior == MoveBehavior::HealBell {
                // `moves:healbell.onHit`: every party member of the target's
                // side is cured, except Soundproof and Good as Gold holders
                // whose ability is not suppressed (they announce immunity and
                // keep their status). The move fails when nobody was cured.
                let side = target.side as usize;
                let mut cured = false;
                for roster in 0..self.sides[side].pokemon.len() as u8 {
                    let ally = Entity {
                        side: target.side,
                        roster,
                    };
                    if ally != actor && !self.suppressing_ability(dex, actor, ally, m) {
                        let ability = dex.effects.abilities[self.mon(ally).ability as usize];
                        if ability == Ability::Soundproof || ability == Ability::Goodasgold {
                            self.reveal_ability(ally)?;
                            continue;
                        }
                    }
                    if self.mon(ally).hp > 0 && self.mon(ally).status != 0 {
                        self.cure_status(ally)?;
                        cured = true;
                    }
                }
                did_anything |= cured;
            } else if behavior == MoveBehavior::MagneticFlux {
                // `moves:magneticflux.onHitSide`: every Plus/Minus holder on
                // the user's side (the user included, `side.allies()`) gains
                // Defense and Special Defense. The move fails when no holder
                // could be raised.
                let side = actor.side;
                let mut boosted = false;
                let holders: Vec<Entity> = self.sides[side as usize]
                    .active
                    .iter()
                    .flatten()
                    .map(|roster| Entity {
                        side,
                        roster: *roster,
                    })
                    .filter(|e| {
                        matches!(
                            dex.effects.abilities[self.mon(*e).ability as usize],
                            Ability::Plus | Ability::Minus
                        )
                    })
                    .collect();
                for holder in holders {
                    boosted |= self.boost(
                        dex,
                        holder,
                        actor,
                        [0, 1, 0, 1, 0, 0, 0],
                        BoostCause::Move { secondary: false },
                    )?;
                }
                did_anything |= boosted;
            } else if behavior == MoveBehavior::Ingrain {
                // `moves:ingrain.condition.onStart`: a self marker that grounds
                // the holder, pins it in place and heals it at residual order
                // 7. A repeated use fails (the condition declares no
                // `onRestart`).
                if !self.mon(target).volatiles.contains_key(&dex.effects.ingrain) {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        dex.effects.ingrain,
                        EffectState {
                            id: dex.effects.ingrain,
                            effect_order: order,
                            effect_order_assigned: true,
                            source: Some((
                                if target.side == 0 {
                                    SideId::P1
                                } else {
                                    SideId::P2
                                },
                                target.roster,
                            )),
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        None,
                        EffectRef::Condition(dex.effects.ingrain),
                        0,
                        false,
                    )?;
                    did_anything = true;
                }
            } else if behavior == MoveBehavior::Octolock {
                // `moves:octolock.condition.onStart`: the target is pinned by
                // the user and loses a stage of Defense and Special Defense at
                // residual order 14 for as long as the user stays active.
                if !self.mon(target).volatiles.contains_key(&dex.effects.octolock) {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        dex.effects.octolock,
                        EffectState {
                            id: dex.effects.octolock,
                            effect_order: order,
                            effect_order_assigned: true,
                            source: Some((
                                if actor.side == 0 {
                                    SideId::P1
                                } else {
                                    SideId::P2
                                },
                                actor.roster,
                            )),
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        Some(actor),
                        EffectRef::Condition(dex.effects.octolock),
                        0,
                        false,
                    )?;
                    did_anything = true;
                }
            } else if behavior == MoveBehavior::TidyUp {
                // `moves:tidyup.onHit`: every Substitute on the field is
                // removed, the entry hazards are cleared on the user's side and
                // on any foe side that holds them, and Attack and Speed rise by
                // one as a self-boost. The move reports failure when neither
                // happened (`!!this.boost(...) || success`).
                let mut success = false;
                for e in self.active_entities(true) {
                    if self
                        .mon_mut(e)
                        .volatiles
                        .remove(&dex.effects.substitute)
                        .is_some()
                    {
                        success = true;
                        self.emit(
                            EventKind::EffectEnd,
                            e,
                            None,
                            EffectRef::Condition(dex.effects.substitute),
                            0,
                            false,
                        )?;
                    }
                }
                for side in [target.side as usize, (1 - target.side) as usize] {
                    for id in [
                        dex.effects.spikes,
                        dex.effects.toxic_spikes,
                        dex.effects.stealth_rock,
                        dex.effects.sticky_web,
                    ] {
                        if self.sides[side].conditions.remove(&id).is_some() {
                            success = true;
                            self.emit(
                                EventKind::SideEffectEnd,
                                target,
                                Some(target),
                                EffectRef::Condition(id),
                                0,
                                false,
                            )?;
                        }
                    }
                }
                let boosted = self.boost(
                    dex,
                    target,
                    target,
                    [1, 0, 0, 0, 1, 0, 0],
                    BoostCause::Move { secondary: false },
                )?;
                did_anything |= success || boosted;
            } else if behavior == MoveBehavior::Spite {
                // `moves:spite.onHit`: the target's last move loses four PP
                // (`deductPP` marks the slot used and clamps at zero); the move
                // fails when nothing could be deducted.
                let last = self.mon(target).last_move;
                let deducted = self
                    .mon(target)
                    .moves
                    .iter()
                    .position(|mv| last != 0 && mv.id == last)
                    .map(|slot| {
                        // `moveSlots` aliases `baseMoveSlots`, so a PP change
                        // must land on both (a faint or switch-out rebuilds the
                        // slots from the base copy).
                        let mon = self.mon_mut(target);
                        let before = mon.moves[slot].pp;
                        let after = before.saturating_sub(4);
                        mon.moves[slot].pp = after;
                        mon.moves[slot].used = true;
                        mon.base_moves[slot].pp = after;
                        i32::from(before - after)
                    })
                    .unwrap_or(0);
                did_anything |= deducted > 0;
            } else if behavior == MoveBehavior::Defog {
                // `moves:defog.onHit`: the evasion drop (skipped behind a
                // decoy unless the user infiltrates), then the target side's
                // screens and hazards, then the user side's hazards, then
                // `field.clearTerrain()`; only the drop and the hazard
                // removals count as success.
                did_anything |= self.defog(dex, actor, target)?;
            } else if behavior == MoveBehavior::CorrosiveGas {
                // `moves:corrosivegas.onHit`: destroy the target's held item.
                // The reference's status result is a numeric zero, so the move
                // still counts as having connected (both move-loop Updates
                // run) even when the target holds nothing; the per-target
                // `-fail` message is a log detail the native model omits.
                if let crate::battle::hooks::TakeOutcome::Taken(item) =
                    self.take_item_checked(dex, target)?
                {
                    self.emit(
                        EventKind::EndItem,
                        target,
                        Some(actor),
                        EffectRef::Item(item),
                        0,
                        false,
                    )?;
                }
                did_anything = true;
            } else if behavior == MoveBehavior::Quash {
                // `moves:quash.onHit`: doubles only, and only while the
                // target still has a queued move action. The reference
                // rewrites that action's order to 201 in place; the queue
                // re-sort that follows this action (Gen 8+) then leaves it
                // after every other move but before the residual phase.
                let doubles = self.sides[0].active.len() > 1;
                if doubles
                    && let Some(index) = self
                        .queue
                        .iter()
                        .position(|q| q.kind == QueuedKind::Move && q.actor == Some(target))
                {
                    self.queue[index].priority.order = 201;
                    did_anything = true;
                }
            } else if behavior == MoveBehavior::Recycle {
                // `moves:recycle.onHit`: with empty hands, restore the last
                // consumed item (clearing `lastItem` first, then `setItem`).
                if self.mon(actor).item == 0 && self.mon(actor).previous_item != 0 {
                    let item = self.mon(actor).previous_item;
                    self.mon_mut(actor).previous_item = 0;
                    self.give_item(dex, actor, actor, item)?;
                    did_anything = true;
                }
            } else if behavior == MoveBehavior::HealPulse {
                // `moves:healpulse.onHit`: `this.heal(Math.ceil(baseMaxhp/2))`
                // (Mega Launcher's 0.75 variant has no in-scope holder). A
                // full-HP target is refused with `NOT_FAIL`, which still
                // counts as a connected hit; the `heal` flag already refused
                // the move under Heal Block at BeforeMove.
                did_anything = true;
                let amount = u32::from(self.mon(target).stats[0]).div_ceil(2);
                self.heal_for_move(dex, target, amount)?;
            } else if behavior == MoveBehavior::PainSplit {
                // `moves:painsplit.onHit`: both HP values are set to
                // `Math.floor((targetHP + userHP) / 2) || 1`. `sethp` bypasses
                // every heal gate, and the handler always counts as a
                // successful hit.
                let target_hp = u32::from(self.mon(target).hp);
                let actor_hp = u32::from(self.mon(actor).hp);
                let average = ((target_hp + actor_hp) / 2).max(1);
                self.set_hp_by(dex, target, actor, move_id, average)?;
                self.set_hp_by(dex, actor, actor, move_id, average)?;
                did_anything = true;
            } else if behavior == MoveBehavior::ItemSteal {
                // `moves:bugbite.onHit` / `moves:pluck.onHit`: while the user
                // is alive, a Berry held by the target is taken through the
                // reference `TakeItem` refusal pipeline and immediately eaten
                // by the user. Any other item refuses the whole branch.
                let item = self.mon(target).item;
                if self.mon(actor).hp > 0
                    && item != 0
                    && dex.effects.berry_items[item as usize]
                    && let crate::battle::hooks::TakeOutcome::Taken(item) =
                        self.take_item_checked(dex, target)?
                {
                    self.emit(
                        EventKind::EndItem,
                        target,
                        Some(actor),
                        EffectRef::Item(item),
                        0,
                        false,
                    )?;
                    self.eat_berry(dex, actor, item)?;
                }
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
                // `moves:steelroller.onHit`: a landed hit clears the active
                // terrain (after damage, before the self/secondary phase).
                // A decoy hit instead clears through `intercept_substitute`'s
                // `onAfterSubDamage` dispatch.
                if hooks & crate::effects::hook::STEEL_ROLLER != 0 {
                    self.clear_terrain(dex, actor)?;
                    continue;
                }
                // `moves:strengthsap.onHit`: a target already at -6 Attack
                // fails the move outright. Otherwise the heal amount is the
                // target's stage-boosted Attack (`getStat('atk', false, true)`
                // skips ModifyStat ability modifiers), the Attack drop is
                // applied first, and the move succeeds when either the drop
                // changed a stage or the heal actually restored HP.
                if hooks & crate::effects::hook::STRENGTH_SAP != 0 {
                    let (atk_stage, atk_stat) = {
                        let t = self.mon(target);
                        (t.boosts[0], t.stats[1])
                    };
                    if atk_stage > -6 {
                        let amount = stats::apply_stage(u32::from(atk_stat), atk_stage);
                        let mut drop = [0i8; 7];
                        drop[0] = -1;
                        let changed = self.boost(
                            dex,
                            target,
                            actor,
                            drop,
                            BoostCause::Move { secondary: false },
                        )?;
                        let healed = self.drain_heal(
                            dex,
                            actor,
                            target,
                            amount,
                            EffectRef::Move(move_id),
                        )?;
                        did_anything |= changed || healed > 0;
                    }
                    continue;
                }
                // `moves:soak.onHit`: pure-Water targets refuse; anything else
                // is overwritten with pure Water.
                // `moves:curse.onTryHit|onHit`: a Ghost user curses the target
                // (refused while that volatile is already up) and pays half its
                // own maximum HP; any other user replaces the whole payload
                // with a self boost and never touches the curse volatile.
                if hooks & crate::effects::hook::CURSE != 0 {
                    let ghost = self
                        .effective_types(dex, actor)
                        .contains(&dex.effects.ghost);
                    if !ghost {
                        // Champions `moves:curse.onHit`: the direct `this.boost`
                        // call never runs `selfDrops`, so the self boost spends
                        // no draw, and its result *is* the move's success - a
                        // fully boosted user fails the move.
                        did_anything = self.boost(
                            dex,
                            actor,
                            actor,
                            [1, 1, 0, 0, -1, 0, 0],
                            BoostCause::Move { secondary: false },
                        )?;
                        continue;
                    }
                    if self.mon(target).volatiles.contains_key(&dex.effects.curse) {
                        did_anything = false;
                        continue;
                    }
                    // `moves:curse.onHit`: `directDamage(source.maxhp / 2)` on
                    // the user, with the move as the effect.
                    let max_hp = u32::from(self.mon(actor).stats[0]);
                    let cost = (max_hp / 2).max(1).min(u32::from(self.mon(actor).hp));
                    if cost > 0 {
                        self.mon_mut(actor).hp -= cost as u16;
                        self.emit(
                            EventKind::Damage,
                            actor,
                            Some(actor),
                            EffectRef::Move(move_id),
                            -(cost as i32),
                            true,
                        )?;
                        if self.mon(actor).hp == 0 {
                            self.faint_queue.push(crate::state::FaintData {
                                target: actor,
                                source: Some(actor),
                                from_move: true,
                            });
                        }
                    }
                    // `onHit`: `delete target.volatiles['curse']` then
                    // `target.addVolatile('curse')`, i.e. the drain volatile is
                    // (re)started with the curser as its source.
                    self.mon_mut(target).volatiles.remove(&dex.effects.curse);
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        dex.effects.curse,
                        EffectState {
                            id: dex.effects.curse,
                            effect_order: order,
                            effect_order_assigned: true,
                            source: Some((
                                if actor.side == 0 {
                                    SideId::P1
                                } else {
                                    SideId::P2
                                },
                                actor.roster,
                            )),
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        Some(actor),
                        EffectRef::Condition(dex.effects.curse),
                        0,
                        false,
                    )?;
                    did_anything = true;
                    continue;
                }
                if hooks & crate::effects::hook::SOAK != 0
                    && self.effective_types(dex, target).as_slice() != [dex.effects.water]
                {
                    did_anything |= self.set_type(dex, target, &[dex.effects.water])?;
                } else if hooks & crate::effects::hook::SOAK == 0 {
                    // `moves:growth.onModifyMove` may have replaced the declared
                    // boosts for this action (the sun branch).
                    let override_hit;
                    let payload = match m.boost_override {
                        Some(boosts) => {
                            override_hit = crate::effects::HitEffect { boosts, ..m.hit };
                            &override_hit
                        }
                        None => &m.hit,
                    };
                    did_anything |=
                        self.hit_effect_from_move(dex, target, actor, payload, false, m)?;
                }
                // `moves:magicpowder.onHit`: a pure-Psychic target refuses the
                // move outright (`onHit` returns false), so the generic empty
                // payload's "connected" result must be discarded or the
                // reference's per-hit loop would run its Update pair and spend
                // two draws the failed move never reaches. The refusal test is
                // `getTypes().join() === 'Psychic'`, i.e. the effective type
                // list, not the stored one.
                if hooks & crate::effects::hook::MAGIC_POWDER != 0 {
                    did_anything =
                        self.effective_types(dex, target).as_slice() != [dex.effects.psychic]
                            && self.set_type(dex, target, &[dex.effects.psychic])?;
                }
                // `moves:batonpass.onHit` (and the same inline `selfSwitch`
                // gate in the reference's `spreadMoveHit`): the move fails when
                // the user's side cannot switch or the user is commanded. That
                // refusal must discard the generic empty payload's "connected"
                // result, or the failed move would still run the hit-loop
                // Update pair and raise the pivot flag.
                if m.self_switch == crate::assets::SelfSwitch::CopyVolatile {
                    did_anything = self.can_switch(actor.side as usize)
                        && !self
                            .mon(actor)
                            .volatiles
                            .contains_key(&dex.effects.commanded);
                }
                // `moves:pollenpuff.onHit`: an ally-targeted use heals half of
                // the target's maximum HP instead of damaging it. A refused
                // heal (full HP, Heal Block) is `NOT_FAIL`, so it must discard
                // the generic payload's success just like the other refusals.
                if hooks & crate::effects::hook::POLLEN_PUFF != 0 && target.side == actor.side {
                    let amount = u32::from(self.mon(target).stats[0]) / 2;
                    did_anything = self.heal_for_move(dex, target, amount)? > 0;
                }
                // `setAbility` payloads of the ability-transfer moves.
                if hooks & crate::effects::hook::ENTRAINMENT != 0 {
                    let ability = self.mon(actor).ability;
                    did_anything |= self.set_ability(dex, target, ability)?;
                } else if hooks & crate::effects::hook::ROLE_PLAY != 0 {
                    let ability = self.mon(target).ability;
                    did_anything |= self.set_ability(dex, actor, ability)?;
                } else if hooks & crate::effects::hook::SIMPLE_BEAM != 0 {
                    did_anything |=
                        self.set_ability(dex, target, dex.effects.simple_ability)?;
                }
                // `moves:burningjealousy.onHit`: each target whose stats were
                // raised this turn is burned (silently refused when the
                // status cannot land, as the move declares no `status` field).
                if hooks & crate::effects::hook::BURNING_JEALOUSY != 0
                    && self.mon(target).stats_raised_this_turn
                {
                    let burn = crate::effects::HitEffect {
                        status: dex.effects.burn,
                        ..Default::default()
                    };
                    let _ = self.hit_effect_from_move(dex, target, actor, &burn, true, m)?;
                }
                // `moves:acupressure.onHit`: one draw samples a stat below +6,
                // which then rises two stages; a fully capped target fails.
                if hooks & crate::effects::hook::ACUPRESSURE != 0 {
                    let candidates: smallvec::SmallVec<[usize; 7]> = (0..7)
                        .filter(|slot| self.mon(target).boosts[*slot] < 6)
                        .collect();
                    if candidates.is_empty() {
                        did_anything = false;
                        continue;
                    }
                    let pick =
                        candidates[self.rng.below(candidates.len() as u32) as usize];
                    let mut boosts = [0i8; 7];
                    boosts[pick] = 2;
                    did_anything |= self.boost(
                        dex,
                        target,
                        actor,
                        boosts,
                        BoostCause::Move { secondary: false },
                    )?;
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
        // `moves:burnup.self.onHit`: the same placeholder strip for Fire.
        if hooks & crate::effects::hook::BURN_UP != 0 && did_anything {
            let mapped: SmallVec<[Id; 4]> = self
                .effective_types(dex, actor)
                .into_iter()
                .map(|kind| if kind == dex.effects.fire { 0 } else { kind })
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
        } else if (blocked_by_protection || refused_not_fail) && !failed_otherwise {
            MoveResult::Skipped
        } else {
            MoveResult::Failed
        };
        // Reference `spreadMoveHit` sets `source.switchFlag` once the move has
        // resolved against at least one target, `didAnything` is truthy (or a
        // numeric damage result, including zero), the user is still alive and a
        // reserve exists. Parting Shot deletes its own `selfSwitch` when the
        // Attack/Sp. Atk drop fails.
        // `Baton Pass` carries the same flag through its `copyvolatile` payload;
        // its `canSwitch`/commanded gate is folded into `did_anything` above.
        let pivot = matches!(
            m.self_switch,
            crate::assets::SelfSwitch::Switch | crate::assets::SelfSwitch::CopyVolatile
        ) && self.mon(actor).hp > 0
            && !hit_targets.is_empty()
            && self.can_switch(actor.side as usize)
            && (hooks & crate::effects::hook::PARTING_SHOT == 0 || did_anything);
        if pivot {
            self.mon_mut(actor).switch_flag = Some(move_id);
        }
        // `selfdestruct: 'ifHit'` faints the user inside the reference's
        // per-target effect phase, after the target's boosts/status resolve:
        // `damage[i] !== false`, i.e. the effect actually applied. A status
        // move whose payload was refused (Memento at the caps, Healing Wish
        // without a reserve) therefore leaves the user alive.
        let connected = hit_any || !hit_targets.is_empty();
        let self_hit = if m.category == Category::Status {
            did_anything
        } else {
            connected
        };
        if self_hit && m.self_destruct == crate::assets::SelfDestructMode::IfHit {
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
            if effect.volatile == dex.effects.locked_move {
                // `conditions:lockedmove`: the rampage lock records the move
                // and rolls its true duration on a fresh start; a re-add runs
                // `onRestart`, which refreshes only the declared duration.
                self.start_locked_move(dex, actor, move_id)?;
            } else {
                self.hit_effect(dex, actor, actor, effect, false)?;
            }
        }
        for &target in &hit_targets {
            // The reference's `secondaries` skips only targets marked `false`;
            // a decoy-absorbed target is `null` and still consumes each roll
            // while its own payload is dropped and a secondary `self` applies.
            let absorbed = sub_absorbed.contains(&target);
            for secondary in m.secondaries.iter().filter(|_| !m.sheer_force) {
                if self.rng.below(100) < u32::from(secondary.chance) {
                    if !absorbed {
                        self.hit_effect_from_move(dex, target, actor, &secondary.target, true, m)?;
                    }
                    if hooks & crate::effects::hook::DIRE_CLAW != 0 && !absorbed {
                        self.dire_claw_secondary(dex, actor, target)?;
                    }
                    if behavior == MoveBehavior::SpiritShackle
                        && !absorbed
                        && self.mon(actor).active_slot.is_some()
                    {
                        // `moves:spiritshackle.secondary.onHit`: the 100%
                        // secondary pins the target (its roll is the loop's).
                        self.start_selection_volatile(
                            dex,
                            target,
                            Some(actor),
                            dex.effects.trapped,
                            false,
                            false,
                        )?;
                    }
                    if hooks & crate::effects::hook::THROAT_CHOP != 0 && !absorbed {
                        self.throat_chop_secondary(dex, actor, target)?;
                    }
                    if hooks & crate::effects::hook::ALLURING_VOICE != 0
                        && !absorbed
                        && self.mon(target).stats_raised_this_turn
                    {
                        self.alluring_voice_secondary(dex, actor, target)?;
                    }
                    if hooks & crate::effects::hook::TRI_ATTACK != 0 && !absorbed {
                        self.tri_attack_secondary(dex, actor, target)?;
                    }
                    if hooks & crate::effects::hook::EERIE_SPELL != 0 && !absorbed {
                        self.eerie_spell_secondary(dex, target)?;
                    }
                    if let Some(effect) = &secondary.own {
                        self.hit_effect(dex, actor, actor, effect, true)?;
                    }
                }
            }
            // `secondaries()` rolls `random(100)` for every entry and the
            // thrown item's status/volatile entry declares no chance, so the
            // roll is consumed and the effect always applies.
            if let Some((_, kind)) = m.fling {
                let effect = match kind {
                    crate::effects::FlingKind::Status(status) => Some(crate::effects::HitEffect {
                        status,
                        ..Default::default()
                    }),
                    crate::effects::FlingKind::Volatile(volatile) => {
                        Some(crate::effects::HitEffect {
                            volatile,
                            ..Default::default()
                        })
                    }
                    _ => None,
                };
                if let Some(effect) = effect {
                    self.rng.below(100);
                    if !absorbed {
                        self.hit_effect_from_move(dex, target, actor, &effect, true, m)?;
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
            for &target in &effect_targets {
                if self.mon(target).hp == 0
                    || self.mon(actor).hp == 0
                    || !self.can_switch(target.side as usize)
                {
                    continue;
                }
                // `moves:ingrain.condition.onDragOut` returns null, which
                // refuses the drag without the Status-move `-fail` branch.
                if self.mon(target).volatiles.contains_key(&dex.effects.ingrain) {
                    continue;
                }
                self.mon_mut(target).force_switch_flag = true;
            }
        }
        // `useMoveInner`: `if (move.selfBoost && moveResult) this.moveHit(...)`.
        // A self-targeted hit is voided once the move's own damage ended the
        // battle (the reference's `spreadMoveHit` refuses a hit against a side
        // that has already lost), so skip the boost when every foe is down.
        // It draws no RNG: the reference only rolls for `move.self` drops.
        if let Some(effect) = &m.self_boost {
            let battle_over = (0..2usize)
                .filter(|side| *side != actor.side as usize)
                .all(|side| {
                    self.sides[side]
                        .pokemon
                        .iter()
                        .all(|mon| !mon.selected || mon.hp == 0)
                });
            if !battle_over {
                self.hit_effect(dex, actor, actor, effect, false)?;
            }
        }
        // Reference `spreadMoveHit` (Champions): the user's own Emergency Exit
        // check runs right after the DamagingHit event, with the HP it had
        // before that event (Rough Skin-style recoil can drop it under half).
        let user_hp_before_damaging_hit = self.mon(actor).hp;
        self.damaging_hit(dex, actor, &effect_targets, &hit_damages, m)?;
        // `moves:ceaselessedge.onAfterHit` / `moves:stoneaxe.onAfterHit`: an
        // alive user scatters its hazard for every damaged target unless Sheer
        // Force consumed the action's secondary (`!move.hasSheerForce`).
        if !m.sheer_force && self.mon(actor).hp > 0 {
            let hazard = if move_id == dex.effects.ceaseless_edge {
                Some(dex.effects.spikes)
            } else if move_id == dex.effects.stone_axe {
                Some(dex.effects.stealth_rock)
            } else {
                None
            };
            if let Some(id) = hazard {
                for _ in &effect_targets {
                    self.add_side_hazard(dex, 1 - actor.side as usize, actor, id)?;
                }
            }
        }
        // `moves:icespinner.onAfterHit`: a landed hit clears the active terrain.
        if hooks & crate::effects::hook::ICE_SPINNER != 0 && self.mon(actor).hp > 0 {
            self.clear_terrain(dex, actor)?;
        }
        // `moves:mortalspin|rapidspin.onAfterHit`: the user sheds Leech Seed,
        // its own entry hazards and partial trapping unless Sheer Force
        // suppressed the action's effects.
        if hooks & (crate::effects::hook::MORTAL_SPIN | crate::effects::hook::RAPID_SPIN) != 0
            && !m.sheer_force
            && self.mon(actor).hp > 0
        {
            self.mortal_spin_shed(dex, actor)?;
        }
        if !hit_targets.is_empty() {
            self.emergency_exit_check(dex, actor, user_hp_before_damaging_hit)?;
        }
        // `moves:knockoff.onAfterHit`: after the DamagingHit event, an alive
        // user removes the item of every target the move damaged.
        if m.hooks & crate::effects::hook::KNOCK_OFF != 0 && self.mon(actor).hp > 0 {
            for &target in &effect_targets {
                self.take_item(dex, target, actor)?;
            }
        }
        // `moves:thief|covet.onAfterHit`: an empty-handed, alive user takes the
        // first damaged target's item. The reference re-checks the user's item
        // per target, so only one steal can land; a refused give returns the
        // item to its original holder.
        if (m.id == dex.effects.thief_move || m.id == dex.effects.covet_move)
            && self.mon(actor).hp > 0
        {
            for &target in &effect_targets {
                if target == actor || self.mon(actor).item != 0 {
                    continue;
                }
                let crate::battle::hooks::TakeOutcome::Taken(item) =
                    self.take_item_checked(dex, target)?
                else {
                    continue;
                };
                if self.mon(actor).hp == 0 || self.mon(actor).active_slot.is_none() {
                    self.mon_mut(target).item = item;
                    continue;
                }
                // Thief's reference log carries a silent `-enditem`; both
                // moves still make the removal public state.
                self.emit(
                    EventKind::EndItem,
                    target,
                    Some(actor),
                    EffectRef::Item(item),
                    0,
                    false,
                )?;
                self.give_item(dex, actor, target, item)?;
            }
        }
        // Reference `useMoveInner`: the `all` / `foeSide` / `allySide` /
        // `allyTeam` classes run `tryMoveHit`, which never enters
        // `hitStepMoveHitLoop`, so neither of its two `Update` events fires —
        // only the action tail's own Update does.
        let hit_loop_updates = !matches!(
            m.target,
            Target::All | Target::FoeSide | Target::AllySide | Target::AllyTeam
        );
        if hit_loop_updates {
            self.each_update(dex)?;
        }
        self.process_faints(dex, self.mon(actor).hp == 0)?;
        if hit_loop_updates {
            self.each_update(dex)?;
        }
        // `moves:fellstinger.onAfterMoveSecondarySelf`: Attack +3 when the move
        // KO'd its target (the reference's `!target || target.fainted ||
        // target.hp <= 0`).
        if behavior == MoveBehavior::FellStinger
            && hit_targets
                .iter()
                .any(|t| self.mon(*t).fainted || self.mon(*t).hp == 0)
        {
            self.boost(
                dex,
                actor,
                actor,
                [3, 0, 0, 0, 0, 0, 0],
                BoostCause::Move { secondary: false },
            )?;
        }
        // `abilities:magician.onAfterMoveSecondarySelf`: a damaging move steals
        // the first item found among the hit targets in speed order, but only
        // while the user's own hands are empty. The reference speed-sorts the
        // hit-target list, so ties consume their shuffle draws here.
        if dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Magician
            && !self.mon(actor).plain_switch_flag
            && !hit_targets.is_empty()
            && self.mon(actor).item == 0
        {
            let mut ordered: SmallVec<[(Entity, Priority); 4]> = hit_targets
                .iter()
                .map(|&e| {
                    (
                        e,
                        Priority {
                            speed: self.mon(e).cached_speed,
                            ..Default::default()
                        },
                    )
                })
                .collect();
            speed_sort(&mut ordered, &mut self.rng, |x| x.1);
            for (target, _) in ordered {
                if target == actor {
                    continue;
                }
                let crate::battle::hooks::TakeOutcome::Taken(item) =
                    self.take_item_checked(dex, target)?
                else {
                    continue;
                };
                // `Pokemon#setItem` refuses only a fainted or inactive
                // recipient; a refused give returns the item to its holder.
                if self.mon(actor).hp == 0 || self.mon(actor).active_slot.is_none() {
                    self.mon_mut(target).item = item;
                    continue;
                }
                self.reveal_ability(actor)?;
                self.give_item(dex, actor, target, item)?;
                return Ok(());
            }
        }
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
            } else if m.mind_blown_recoil {
                // `applyRecoilDamage` with `mindBlownRecoil`: the user pays
                // round(maxHP / 2) with the move itself as the effect, so the
                // recoil is move damage for Magic Guard and ignores Rock Head.
                let recoil =
                    stats::round_fraction(u32::from(self.mon(actor).stats[0]), [1, 2]);
                let hp_before = self.mon(actor).hp;
                self.indirect_damage(dex, actor, actor, recoil, EffectRef::Move(move_id))?;
                self.emergency_exit_check(dex, actor, hp_before)?;
            }
        }
        if m.thaws_target {
            for &target in &effect_targets {
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
        // `moves:spitup.onAfterMove`: the reference fires the move's AfterMove
        // event after the whole `useMove` sequence, which is where Spit Up
        // consumes the user's stockpile.
        if m.hooks & crate::effects::hook::SPIT_UP != 0 {
            self.remove_stockpile(dex, actor)?;
        }
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
        // `getSmartTargets` (Dragon Darts): a smart-target action resolves
        // against the chosen target and that target's first live adjacent
        // ally, in that order. A missing or fainted ally - or a fainted chosen
        // target - falls back to a single target and clears the flag for the
        // action.
        let mut targets = targets;
        let mut smart = false;
        if m.smart_target && targets.len() == 1 {
            let chosen = targets[0];
            let ally = self.sides[chosen.side as usize]
                .active
                .iter()
                .flatten()
                .map(|roster| Entity {
                    side: chosen.side,
                    roster: *roster,
                })
                .find(|e| *e != chosen);
            match ally {
                Some(ally) if ally != actor && self.mon(ally).hp > 0 => {
                    if self.mon(chosen).hp > 0 {
                        targets.push(ally);
                        smart = true;
                    } else {
                        targets = smallvec::smallvec![ally];
                    }
                }
                _ => {}
            }
        }
        // Reference `spreadMoveHit`: a smart-target hit is not a spread hit.
        let spread = targets.len() > 1 && !m.smart_target;
        let effective_priority = m
            .priority
            .unwrap_or_else(|| self.effective_priority(dex, actor, move_id));
        let mut blocked = SmallVec::<[(Entity, Id); 4]>::new();
        // Reference `trySpreadMoveHit` clears `move.smartTarget` once any hit
        // step refuses a target, which turns the remaining hits into a normal
        // all-targets hit.
        let mut at_least_one_failure = false;
        let mut kept = SmallVec::<[Entity; 4]>::new();
        for e in targets {
            self.validate_effects(dex, e)?;
            if m.protect && !m.breaks_protect && !self.ability_bypasses_protect(dex, actor, m) {
                if self.guard_blocks(dex, e, m, effective_priority) {
                    at_least_one_failure = true;
                    continue;
                }
                if let Some(volatile) = self.blocking_protection(dex, e) {
                    blocked.push((e, volatile));
                    at_least_one_failure = true;
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
                && effective_priority > 0
                && target.side != actor.side
                && self.grounded(dex, target)
            {
                at_least_one_failure = true;
                continue;
            }
            if m.powder
                && target != actor
                && ability == Ability::Overcoat
                && !self.mon(target).types.contains(&dex.effects.grass)
            {
                self.reveal_ability(target)?;
                at_least_one_failure = true;
                continue;
            }
            // TryHit (absorption) precedes type immunity and accuracy.
            if self.absorb_try_hit(dex, target, actor, m, &mut action_accuracy)? {
                at_least_one_failure = true;
                continue;
            }
            let effectiveness = if m.ignore_immunity {
                Some(0)
            } else if m.move_type == dex.effects.ground
                && ability == Ability::Levitate
                && !self.suppressing_ability(dex, actor, target, m)
                && !self.grounded(dex, target)
            {
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
                // `Pokemon#runImmunity`: for a Ground move the immunity is
                // decided by `isGrounded`, so a grounded target's Flying type
                // no longer blocks the move (Gravity, Smack Down, Ingrain).
                let grounded = self.grounded(dex, target);
                self.mon(target)
                    .types
                    .iter()
                    .try_fold(0i8, |total, &kind| {
                        let value = dex.type_chart[m.move_type as usize][kind as usize];
                        if value == -127
                            && !(m.move_type == dex.effects.ground
                                && kind == dex.effects.flying
                                && grounded)
                        {
                            None
                        } else {
                            Some(total + if value == -127 { 0 } else { value })
                        }
                    })
                    .map(|total| total.clamp(-6, 6))
            };
            let Some(effectiveness) = effectiveness else {
                at_least_one_failure = true;
                continue;
            };
            if !self.roll_move_accuracy(dex, actor, target, m, action_accuracy) {
                at_least_one_failure = true;
                continue;
            }
            connected.push((target, effectiveness));
        }
        if connected.is_empty() {
            return Ok(());
        }
        // Reference `trySpreadMoveHit`: any refused target clears the
        // smart-target flag, so the remaining hits resolve like a normal
        // multi-hit move against every surviving target.
        if smart && at_least_one_failure {
            smart = false;
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
        // Per-target decoy bookkeeping of the most recent hit, mirroring the
        // reference's `targetsCopy` nulling that survives into the
        // `AfterMoveSecondary` phase.
        let mut sub_absorbed: SmallVec<[Entity; 4]> = SmallVec::new();
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
            // Reference `hitStepMoveHitLoop`: a smart-target action resolves
            // each hit against one entry of the list (`targets[hit - 1]`); a
            // step-filtered index simply spends the hit with no target.
            let single: SmallVec<[(Entity, i8); 1]> = if smart {
                connected
                    .get(hit as usize - 1)
                    .copied()
                    .into_iter()
                    .collect()
            } else {
                SmallVec::new()
            };
            let hit_targets: &[(Entity, i8)] = if smart { &single } else { &connected };
            for &(target, effectiveness) in hit_targets {
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
                // `abilities:disguise.onDamage` inside the multi-hit loop: the
                // first hit against an undisguised Mimikyu is absorbed, and the
                // reference runs the Update *between* hits, so the remaining
                // hits of the same move land on the busted forme (which keeps
                // the same stats). The hit still counts for `timesAttacked` and
                // the reported hit count.
                if target != actor
                    && dex.effects.abilities[self.mon(target).ability as usize]
                        == Ability::Disguise
                    && !self.suppressing_ability(dex, actor, target, m)
                    && (self.mon(target).species == dex.effects.mimikyu
                        || self.mon(target).species == dex.effects.mimikyu_totem)
                {
                    self.reveal_ability(target)?;
                    self.mon_mut(target).disguise_busted = true;
                    let count = self.mon(target).times_attacked;
                    self.mon_mut(target).times_attacked = count.saturating_add(1);
                    self.record_attacked_by(target, actor, m.move_uid, 0);
                    self.emit(
                        EventKind::Damage,
                        target,
                        Some(actor),
                        EffectRef::Move(move_id),
                        0,
                        true,
                    )?;
                    self.disguise_update(dex, target)?;
                    continue;
                }
                sub_absorbed.retain(|t| *t != target);
                if self.intercept_substitute(dex, actor, target, m, damage, m.infiltrates)? {
                    sub_absorbed.push(target);
                    // `selfDrops` still runs against the nulled target, and the
                    // reference's `secondaries` still rolls each chance while
                    // dropping the target payload.
                    if hit == 1
                        && let Some(effect) = m.self_effect.as_ref().filter(|_| !m.sheer_force)
                    {
                        if effect.boosts.iter().any(|b| *b != 0) {
                            self.rng.below(100);
                        }
                        self.hit_effect(dex, actor, actor, effect, false)?;
                    }
                    for secondary in m.secondaries.iter().filter(|_| !m.sheer_force) {
                        if self.rng.below(100) < u32::from(secondary.chance)
                            && let Some(effect) = &secondary.own
                        {
                            self.hit_effect(dex, actor, actor, effect, true)?;
                        }
                    }
                    continue;
                }
                let damage = self.sturdy_clamp(
                    dex,
                    target,
                    damage,
                    self.suppressing_ability(dex, actor, target, m),
                )?;
                let damage = self.damage_item(dex, target, damage)?;
                let damage = self.endure_clamp(dex, target, damage);
                let actual = damage.min(self.mon(target).hp);
                total_damage += u32::from(actual);
                self.mon_mut(target).hp -= actual;
                if actual != 0 {
                    let hp = self.mon(target).hp;
                    self.mon_mut(target).hurt_this_turn = hp;
                }
                if target != actor {
                    let count = self.mon(target).times_attacked;
                    self.mon_mut(target).times_attacked = count.saturating_add(1);
                    self.record_attacked_by(target, actor, m.move_uid, actual);
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
                    EffectRef::Condition(dex.effects.drain),
                )?;
                }
                self.hit_effect_from_move(dex, target, actor, &m.hit, false, m)?;
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
                        self.hit_effect_from_move(dex, target, actor, &secondary.target, true, m)?;
                        if m.hooks & crate::effects::hook::DIRE_CLAW != 0 {
                            self.dire_claw_secondary(dex, actor, target)?;
                        }
                        if m.hooks & crate::effects::hook::ALLURING_VOICE != 0
                            && self.mon(target).stats_raised_this_turn
                        {
                            self.alluring_voice_secondary(dex, actor, target)?;
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
                self.damaging_hit(dex, actor, std::slice::from_ref(&target), &[actual], m)?;
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
        } else if total_damage > 0 && m.mind_blown_recoil {
            // `mindBlownRecoil`: round(maxHP / 2) as move damage, so Magic
            // Guard keeps it and Rock Head cannot refuse it.
            let recoil = stats::round_fraction(u32::from(self.mon(actor).stats[0]), [1, 2]);
            let hp_before = self.mon(actor).hp;
            self.indirect_damage(dex, actor, actor, recoil, EffectRef::Move(move_id))?;
            self.emergency_exit_check(dex, actor, hp_before)?;
        }
        self.each_update(dex)?;
        if m.thaws_target {
            for target in targets {
                if sub_absorbed.contains(&target) {
                    continue;
                }
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
    fn multihit_count(&mut self, m: &ActiveMove<'_>) -> u32 {
        // `moves:beatup.onModifyMove`: a plain numeric `multihit` set by the
        // callback, so the count is exact and consumes no draw.
        if !m.allies.is_empty() {
            return m.allies.len() as u32;
        }
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
        let Some(accuracy) =
            self.modify_accuracy(dex, actor, target, action_accuracy, m.minimize)
        else {
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
    /// `moves:defog.onHit`: drop the target's evasion by one (skipped behind a
    /// decoy unless the user carries Infiltrator), remove the target side's
    /// screens and then both sides' entry hazards in reference order, clear
    /// the terrain, and report whether anything counted as a success.
    fn defog(&mut self, dex: &Dex, actor: Entity, target: Entity) -> Result<bool> {
        let mut success = false;
        let decoy = self.mon(target).volatiles.contains_key(&dex.effects.substitute);
        let infiltrates =
            dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Infiltrator;
        if !decoy || infiltrates {
            success = self.boost(
                dex,
                target,
                actor,
                [0, 0, 0, 0, 0, 0, -1],
                BoostCause::Move { secondary: false },
            )?;
        }
        let hazards = [
            dex.effects.spikes,
            dex.effects.toxic_spikes,
            dex.effects.stealth_rock,
            dex.effects.sticky_web,
        ];
        for id in [
            dex.effects.reflect,
            dex.effects.light_screen,
            dex.effects.aurora_veil,
        ] {
            if self.sides[target.side as usize].conditions.remove(&id).is_some() {
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
        for side in [target.side as usize, actor.side as usize] {
            for id in hazards {
                if self.sides[side].conditions.remove(&id).is_some() {
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
                    self.emit(
                        EventKind::SideEffectEnd,
                        subject,
                        Some(actor),
                        EffectRef::Condition(id),
                        0,
                        false,
                    )?;
                    success = true;
                }
            }
        }
        self.clear_terrain(dex, actor)?;
        Ok(success)
    }

    /// Reference `Battle#skillSwap`: fail-gate both sides, announce, run the
    /// outgoing abilities' End callbacks (source then target), exchange the
    /// ability ids with fresh effect orders, then run the incoming abilities'
    /// Start callbacks in the reference order (the source's old ability starts
    /// on the target first, the target's old ability on the source second).
    fn skill_swap(&mut self, dex: &Dex, actor: Entity, target: Entity) -> Result<bool> {
        if self.mon(actor).fainted || self.mon(target).fainted {
            return Ok(false);
        }
        let source_ability = self.mon(actor).ability;
        let target_ability = self.mon(target).ability;
        if dex.effects.no_skill_swap_abilities[source_ability as usize]
            || dex.effects.no_skill_swap_abilities[target_ability as usize]
        {
            return Ok(false);
        }
        // The reference logs the exchanged ability names for a foe use; both
        // become public knowledge either way.
        self.reveal_ability(actor)?;
        self.reveal_ability(target)?;
        self.ability_end(dex, actor)?;
        self.ability_end(dex, target)?;
        let order = self.allocate_effect_order()?;
        self.mon_mut(actor).ability = target_ability;
        self.mon_mut(actor).ability_effect_order = Some(order);
        let order = self.allocate_effect_order()?;
        self.mon_mut(target).ability = source_ability;
        self.mon_mut(target).ability_effect_order = Some(order);
        self.ability_start(dex, target)?;
        self.ability_start(dex, actor)?;
        Ok(true)
    }

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
        let base_power = self.base_power(dex, m, actor, target, phase.hit);
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
                weather_modifier: self.weather_damage_modifier(dex, actor, target, m.move_type),
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
        // `abilities:piercingdrill|unseenfist` quarter the damage of a hit
        // whose protection was bypassed.
        let priority = m
            .priority
            .unwrap_or_else(|| self.effective_priority(dex, actor, m.id));
        let bypassed = self.protection_bypassed(dex, actor, target, m, priority);
        Ok(damage::finish_damage(damage, final_modifier, bypassed))
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
        let mut action = QueuedAction {
            kind: QueuedKind::Move,
            actor: Some(actor),
            move_slot: slot as u8,
            move_id,
            source_effect: 0,
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
        let target = self.queued_target(dex, actor, move_id);
        action.target_location = self.random_target_location(actor, target);
        self.resolve_target_location(actor, target, action.target_location);
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

    /// `moves:destinybond.condition`: drop the bond volatile, emitting the
    /// matching public volatile-end event when it was present.
    fn drop_destiny_bond(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if self
            .mon_mut(e)
            .volatiles
            .remove(&dex.effects.destiny_bond)
            .is_some()
        {
            self.emit(
                EventKind::EffectEnd,
                e,
                None,
                EffectRef::Condition(dex.effects.destiny_bond),
                0,
                false,
            )?;
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
            } else if status == dex.effects.sleep && m.sleep_usable {
                // `slp.onBeforeMove` still ticks the counter and prints the
                // "cant" message, but a `sleepUsable` move (Sleep Talk, Snore)
                // is not refused by the condition and continues to run.
            } else {
                return Ok(Some(MoveResult::Failed));
            }
        }
        if self.mon(e).volatiles.contains_key(&dex.effects.flinch) {
            return Ok(Some(MoveResult::Failed));
        }
        // `moves:throatchop.condition.onBeforeMove` and
        // `moves:healblock.condition.onBeforeMove` (both priority 6, between
        // flinch and confusion): a sound move under Throat Chop, or a
        // `heal`-flag move under Heal Block, is refused outright. Both
        // callbacks return false, so when a move carries both flags and both
        // volatiles are present the reference's creation-order tiebreak only
        // changes the refusal message; the engine reports the same failure.
        // Each condition's `onModifyMove` guard is the same rule one phase
        // later and is unreachable once BeforeMove has already refused.
        if (m.sound
            && self
                .mon(e)
                .volatiles
                .contains_key(&dex.effects.throat_chop))
            || (m.heal
                && self
                    .mon(e)
                    .volatiles
                    .contains_key(&dex.effects.heal_block))
        {
            return Ok(Some(MoveResult::Failed));
        }
        // `moves:gravity.condition.onBeforeMove` (priority 6, the same band as
        // Throat Chop and Heal Block): a `flags.gravity` move is refused while
        // the pseudo-weather is up. The condition's `onModifyMove` guard is the
        // same rule one phase later and is unreachable once this refuses.
        if m.gravity && self.field.contains_key(&dex.effects.gravity) {
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
        let hp = self.mon(e).hp;
        self.mon_mut(e).hurt_this_turn = hp;
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

    /// `abilities:piercingdrill|unseenfist.onHitProtect`: a contact move from
    /// the holder cancels the target's protection. The reference sets the
    /// hit's `bypassProtect` marker at the same time, which `modifyDamage`
    /// turns into a quarter-damage modifier.
    fn ability_bypasses_protect(
        &self,
        dex: &Dex,
        actor: Entity,
        m: &crate::assets::Move,
    ) -> bool {
        m.contact
            && matches!(
                dex.effects.abilities[self.mon(actor).ability as usize],
                Ability::Piercingdrill | Ability::Unseenfist
            )
    }

    /// Whether this action's protection gate was actually bypassed for the
    /// target. The reference only marks `bypassProtect` when a blocking
    /// condition (a protection volatile or a Guard side condition) consulted
    /// the `HitProtect` event, so an unprotected target is not quartered.
    fn protection_bypassed(
        &self,
        dex: &Dex,
        actor: Entity,
        target: Entity,
        m: &ActiveMove<'_>,
        effective_priority: i8,
    ) -> bool {
        self.ability_bypasses_protect(dex, actor, m)
            && (self.blocking_protection(dex, target).is_some()
                || self.guard_blocks(dex, target, m, effective_priority))
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
            FixedDamage::CounterStored => {
                // `moves:counter|mirrorcoat.damageCallback`: the volatile's
                // recorded `2 * damage`, or 1 when that value is zero.
                let volatile = if m.id == dex.effects.counter_move {
                    dex.effects.counter
                } else {
                    dex.effects.mirrorcoat
                };
                let stored = self
                    .mon(actor)
                    .volatiles
                    .get(&volatile)
                    .and_then(|state| state.values.first())
                    .copied()
                    .unwrap_or(0);
                Some(u32::try_from(stored).unwrap_or(0).max(1))
            }
            FixedDamage::LastDamagedBy => {
                // `moves:metalburst|comeuppance.damageCallback`:
                // `floor(1.5 * damage) || 1` against the recorded attacker.
                let damage = self
                    .last_damaged_by(actor)
                    .map(|(_, _, damage)| u32::from(damage))
                    .unwrap_or(0);
                Some((damage * 3 / 2).max(1))
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
        self.hit_effect_with_ability(
            dex,
            target,
            source,
            effect,
            HitContext {
                secondary,
                ..Default::default()
            },
        )
    }

    /// Move-driven `hit_effect`: a Mold Breaker user's move also suppresses the
    /// target's `onSetStatus` refusal (Limber, Insomnia, Immunity, Purifying
    /// Salt, ...), exactly like every other handler owned by a breakable
    /// ability.
    fn hit_effect_from_move(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Entity,
        effect: &crate::effects::HitEffect,
        secondary: bool,
        m: &ActiveMove<'_>,
    ) -> Result<bool> {
        let suppressing = self.suppressing_ability(dex, source, target, m);
        self.hit_effect_with_ability(
            dex,
            target,
            source,
            effect,
            HitContext {
                secondary,
                suppressing,
                ..Default::default()
            },
        )
    }

    /// `abilities:protean|libero.onPrepareHit`: once per switch-in the user
    /// becomes the action's (post-ModifyType) type before the hit steps, even
    /// when the action later misses.
    fn prepare_hit_abilities(
        &mut self,
        dex: &Dex,
        actor: Entity,
        m: &ActiveMove<'_>,
    ) -> Result<()> {
        let preparer = dex.effects.abilities[self.mon(actor).ability as usize];
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
        Ok(())
    }

    /// `moves:stockpile.condition` payload readers. The volatile stores
    /// `[layers, def, spd]`, where `def`/`spd` count the successful stage
    /// changes (stored negative) that `onEnd` reverses.
    fn stockpile_layers(&self, dex: &Dex, e: Entity) -> u32 {
        self.mon(e)
            .volatiles
            .get(&dex.effects.stockpile)
            .and_then(|state| state.values.first())
            .copied()
            .unwrap_or(0)
            .clamp(0, 3) as u32
    }

    /// `moves:stockpile.condition.onStart|onRestart`: create or advance the
    /// layered volatile, announce it and apply the Defense/Special Defense
    /// raise. Returns `false` only for the unreachable restart-at-three case.
    fn start_stockpile(&mut self, dex: &Dex, actor: Entity) -> Result<bool> {
        let volatile = dex.effects.stockpile;
        let existing = self.mon(actor).volatiles.get(&volatile).cloned();
        let (layers, mut def, mut spd, order) = match existing {
            Some(state) => {
                let layers = state.values.first().copied().unwrap_or(1).max(1);
                if layers >= 3 {
                    return Ok(false);
                }
                (
                    layers + 1,
                    state.values.get(1).copied().unwrap_or(0),
                    state.values.get(2).copied().unwrap_or(0),
                    state.effect_order,
                )
            }
            None => (1, 0, 0, self.allocate_effect_order()?),
        };
        // The reference `onStart`/`onRestart` announce the new layer before
        // running the boost.
        self.mon_mut(actor).volatiles.insert(
            volatile,
            EffectState {
                id: volatile,
                duration: None,
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
                values: vec![layers, def, spd],
            },
        );
        self.emit(
            EventKind::EffectStart,
            actor,
            Some(actor),
            EffectRef::Condition(volatile),
            layers as i32,
            false,
        )?;
        let before = [self.mon(actor).boosts[1], self.mon(actor).boosts[3]];
        self.boost(
            dex,
            actor,
            actor,
            [0, 1, 0, 1, 0, 0, 0],
            BoostCause::Move { secondary: false },
        )?;
        let after = [self.mon(actor).boosts[1], self.mon(actor).boosts[3]];
        if after[0] != before[0] {
            def -= 1;
        }
        if after[1] != before[1] {
            spd -= 1;
        }
        if let Some(state) = self.mon_mut(actor).volatiles.get_mut(&volatile) {
            state.values = vec![layers, def, spd];
        }
        Ok(true)
    }

    /// `moves:stockpile.condition.onEnd`: reverse the recorded stage raises and
    /// remove the volatile. Returns whether the volatile existed.
    fn remove_stockpile(&mut self, dex: &Dex, e: Entity) -> Result<bool> {
        let volatile = dex.effects.stockpile;
        let Some(state) = self.mon_mut(e).volatiles.remove(&volatile) else {
            return Ok(false);
        };
        let def = state.values.get(1).copied().unwrap_or(0).clamp(-3, 0) as i8;
        let spd = state.values.get(2).copied().unwrap_or(0).clamp(-3, 0) as i8;
        if def != 0 || spd != 0 {
            self.boost(
                dex,
                e,
                e,
                [0, def, 0, spd, 0, 0, 0],
                BoostCause::Move { secondary: false },
            )?;
        }
        self.emit(
            EventKind::EffectEnd,
            e,
            None,
            EffectRef::Condition(volatile),
            0,
            false,
        )?;
        Ok(true)
    }

    /// `moves:swallow.onHit`: `this.heal(this.modify(maxhp, healAmount[layers-1]))`
    /// followed by the stockpile removal. Returns whether any HP was restored;
    /// a refused heal (full HP or Heal Block) still consumes the volatile.
    fn swallow_heal(&mut self, dex: &Dex, actor: Entity) -> Result<bool> {
        let layers = self.stockpile_layers(dex, actor).clamp(1, 3);
        let [numerator, denominator] = [[1u32, 4u32], [1, 2], [1, 1]][(layers - 1) as usize];
        let amount = Self::modify_fraction(u32::from(self.mon(actor).stats[0]), numerator, denominator);
        let healed = self.heal_for_move(dex, actor, amount)?;
        self.remove_stockpile(dex, actor)?;
        Ok(healed > 0)
    }

    /// Reference `Battle#modify`: `trunc((trunc(value * trunc(numerator *
    /// 4096 / denominator)) + 2047) / 4096)`. Swallow is the only ported
    /// caller that depends on this exact truncation (an odd maximum HP heals
    /// `floor(maxhp / 2)` at two layers).
    fn modify_fraction(value: u32, numerator: u32, denominator: u32) -> u32 {
        let modifier = numerator * 4096 / denominator;
        (value * modifier + 2047) / 4096
    }

    /// Reference `Pokemon#sethp`: write an exact HP value, bypassing every
    /// heal gate (used by Pain Split for both participants). Emits the public
    /// HP change as a heal or damage event so both observations and the
    /// knowledge layer track it.
    fn set_hp_by(
        &mut self,
        _dex: &Dex,
        e: Entity,
        source: Entity,
        move_id: Id,
        value: u32,
    ) -> Result<()> {
        let max_hp = u32::from(self.mon(e).stats[0]);
        let value = value.min(max_hp);
        let current = u32::from(self.mon(e).hp);
        if value == current {
            return Ok(());
        }
        self.mon_mut(e).hp = value as u16;
        let delta = value as i32 - current as i32;
        self.emit(
            if delta > 0 { EventKind::Heal } else { EventKind::Damage },
            e,
            Some(source),
            EffectRef::Move(move_id),
            delta,
            true,
        )
    }

    /// Reference `Battle#heal` for a move-driven self heal: the TryHeal gate
    /// (Heal Block), the dead/inactive/full-HP refusals and the capped
    /// restoration. Returns the HP actually restored.
    fn heal_for_move(&mut self, dex: &Dex, target: Entity, amount: u32) -> Result<u32> {
        if amount == 0
            || self.mon(target).hp == 0
            || self.mon(target).active_slot.is_none()
            || self.heal_blocked(dex, target)
        {
            return Ok(0);
        }
        let p = self.mon(target);
        let missing = u32::from(p.stats[0] - p.hp);
        if missing == 0 {
            return Ok(0);
        }
        let healed = amount.min(missing);
        self.mon_mut(target).hp += healed as u16;
        self.emit(
            EventKind::Heal,
            target,
            Some(target),
            EffectRef::None,
            healed as i32,
            true,
        )?;
        Ok(healed)
    }

    /// Predicate for both the pre-pass ordering and the interception below:
    /// a decoy consumes a primary hit whose source differs from its holder
    /// unless the action carries `flags.bypasssub`.
    fn decoy_absorbs(
        &self,
        dex: &Dex,
        source: Entity,
        target: Entity,
        m: &crate::assets::Move,
        infiltrates: bool,
    ) -> bool {
        target != source
            && !infiltrates
            && !m.bypass_sub
            && self
                .mon(target)
                .volatiles
                .contains_key(&dex.effects.substitute)
    }

    /// `moves:substitute.condition.onTryPrimaryHit`: a primary hit whose source
    /// differs from the target and whose action does not carry
    /// `flags.bypasssub` is consumed by the target's decoy. `damage` is the
    /// damage the hit resolved to (`0` for status moves, which the decoy also
    /// absorbs). Returns `true` when the decoy took the hit, in which case the
    /// caller must skip the target's own damage, effects and post-hit phases;
    /// recoil and drain are driven by the damage the decoy ate.
    fn intercept_substitute(
        &mut self,
        dex: &Dex,
        source: Entity,
        target: Entity,
        m: &crate::assets::Move,
        damage: u16,
        infiltrates: bool,
    ) -> Result<bool> {
        if !self.decoy_absorbs(dex, source, target, m, infiltrates) {
            return Ok(false);
        }
        let sub_hp = self
            .mon(target)
            .volatiles
            .get(&dex.effects.substitute)
            .and_then(|effect| effect.values.first())
            .copied()
            .unwrap_or(0);
        let dealt = i64::from(damage).min(sub_hp.max(0));
        if sub_hp <= i64::from(damage) {
            // The decoy breaks; `removeVolatile` emits the public end. The
            // reference adds a bare `-ohko` message for OHKO moves here, which
            // carries no state.
            self.mon_mut(target)
                .volatiles
                .remove(&dex.effects.substitute);
            self.emit(
                EventKind::EffectEnd,
                target,
                None,
                EffectRef::Condition(dex.effects.substitute),
                0,
                false,
            )?;
        } else if let Some(effect) = self
            .mon_mut(target)
            .volatiles
            .get_mut(&dex.effects.substitute)
        {
            // `-activate ... [damage]` is a message-only announcement; no
            // knowledge field changes while the decoy survives.
            effect.values[0] -= dealt;
        }
        if dealt > 0
            && let Some(fraction) = m.recoil
            && dex.effects.abilities[self.mon(source).ability as usize] != Ability::RockHead
        {
            let recoil = stats::round_fraction(dealt as u32, fraction).max(1);
            self.indirect_damage(
                dex,
                source,
                source,
                recoil,
                EffectRef::Condition(dex.effects.recoil),
            )?;
        }
        if let Some([numerator, denominator]) = m.drain {
            // The substitute path uses `Math.ceil`, unlike the `Math.round`
            // that `battle.damage` applies to an ordinary drain.
            let amount = (dealt as u64 * u64::from(numerator))
                .div_ceil(u64::from(denominator));
            self.drain_heal(
                dex,
                source,
                target,
                amount as u32,
                EffectRef::Condition(dex.effects.drain),
            )?;
        }
        // The substitute condition's own `onTryPrimaryHit` ends by firing the
        // `AfterSubDamage` event (`moves:steelroller.onAfterSubDamage`), so a
        // decoy hit still clears the terrain even though the move's `onHit`
        // never runs for an absorbed hit.
        if m.hooks & crate::effects::hook::STEEL_ROLLER != 0 {
            self.clear_terrain(dex, source)?;
        }
        // `moves:icespinner.onAfterSubDamage`: a decoy hit still clears the
        // terrain while the user is alive.
        if m.hooks & crate::effects::hook::ICE_SPINNER != 0 && self.mon(source).hp > 0 {
            self.clear_terrain(dex, source)?;
        }
        // `moves:ceaselessedge.onAfterSubDamage` / `stoneaxe.onAfterSubDamage`:
        // a decoy hit still scatters the hazard while the user is alive and
        // Sheer Force did not consume the secondary. The asset-level `Move`
        // cannot see the action marker, so recompute the `onModifyMove` rule.
        let sheer_force = !m.secondaries.is_empty()
            && dex.effects.abilities[self.mon(source).ability as usize] == Ability::Sheerforce;
        // `moves:mortalspin|rapidspin.onAfterSubDamage`: the same shed as
        // `onAfterHit`.
        if m.hooks & (crate::effects::hook::MORTAL_SPIN | crate::effects::hook::RAPID_SPIN) != 0
            && !sheer_force
            && self.mon(source).hp > 0
        {
            self.mortal_spin_shed(dex, source)?;
        }
        if !sheer_force && self.mon(source).hp > 0 {
            let hazard = if m.id == dex.effects.ceaseless_edge {
                Some(dex.effects.spikes)
            } else if m.id == dex.effects.stone_axe {
                Some(dex.effects.stealth_rock)
            } else {
                None
            };
            if let Some(id) = hazard {
                self.add_side_hazard(dex, 1 - source.side as usize, source, id)?;
            }
        }
        Ok(true)
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

    /// `moves:mortalspin|rapidspin.onAfterHit|onAfterSubDamage`: drop the
    /// user's Leech Seed, partial trapping and its own entry hazards.
    fn mortal_spin_shed(&mut self, dex: &Dex, user: Entity) -> Result<()> {
        self.mon_mut(user).volatiles.remove(&dex.effects.leech_seed);
        self.mon_mut(user).volatiles.remove(&dex.effects.partially_trapped);
        for id in [
            dex.effects.spikes,
            dex.effects.toxic_spikes,
            dex.effects.stealth_rock,
            dex.effects.sticky_web,
        ] {
            if self.sides[user.side as usize].conditions.remove(&id).is_some() {
                self.emit(
                    EventKind::SideEffectEnd,
                    user,
                    Some(user),
                    EffectRef::Condition(id),
                    0,
                    false,
                )?;
            }
        }
        Ok(())
    }

    /// `moves:eeriespell.secondary.onHit`: deduct up to three PP from the
    /// target's last move, failing silently when it has none left.
    fn eerie_spell_secondary(&mut self, _dex: &Dex, target: Entity) -> Result<()> {
        if self.mon(target).hp == 0 {
            return Ok(());
        }
        let last = self.mon(target).last_move;
        if last == 0 {
            return Ok(());
        }
        let deducted = {
            // PP lives in both the live moveset and the `base_moves` copy the
            // observation serves; every deduction site updates the two
            // together.
            let mon = self.mon_mut(target);
            let amount = mon
                .moves
                .iter()
                .find(|slot| slot.id == last && slot.pp > 0)
                .map(|slot| slot.pp.min(3))
                .unwrap_or(0);
            if amount > 0 {
                for slot in mon.moves.iter_mut().chain(mon.base_moves.iter_mut()) {
                    if slot.id == last {
                        slot.pp -= slot.pp.min(amount);
                    }
                }
            }
            amount
        };
        // The reference's `-activate` message carries the drained amount but
        // changes no state beyond the PP deduction above.
        let _ = deducted;
        Ok(())
    }

    /// `moves:triattack.secondary.onHit`: `this.sample(['brn', 'par', 'frz'])`
    /// picks one status, applied through `trySetStatus`.
    fn tri_attack_secondary(&mut self, dex: &Dex, source: Entity, target: Entity) -> Result<()> {
        let statuses = [dex.effects.burn, dex.effects.paralysis, dex.effects.freeze];
        let status = statuses[self.rng.below(statuses.len() as u32) as usize];
        let effect = crate::effects::HitEffect {
            status,
            ..Default::default()
        };
        self.hit_effect(dex, target, source, &effect, true)?;
        Ok(())
    }

    /// `moves:alluringvoice.secondary.onHit`: a target whose stats were raised
    /// this turn gains the confusion volatile (with the reference
    /// `random(2, 6)` timer); an already-confused target is left untouched.
    fn alluring_voice_secondary(&mut self, dex: &Dex, source: Entity, target: Entity) -> Result<()> {
        let effect = crate::effects::HitEffect {
            volatile: dex.effects.confusion,
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
        context: HitContext,
    ) -> Result<bool> {
        let HitContext {
            secondary,
            ability_source,
            suppressing,
        } = context;
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
            // `moves:*heal*.onHit` recovers through `this.heal`, which Heal
            // Block refuses; the effect then reports that nothing happened.
            if amount == 0 || self.heal_blocked(dex, target) {
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
            // `conditions:safeguard.onSetStatus`: a status applied by another
            // Pokémon is refused while the target's side holds Safeguard.
            // (Yawn cannot reach this stage: its volatile is refused when the
            // side is protected.)
            if !suppressing
                && source != target
                && self.sides[target.side as usize]
                    .conditions
                    .contains_key(&dex.effects.safeguard)
            {
                if !secondary && ability_source.is_none() {
                    self.reveal_ability(target)?;
                }
                return Ok(false);
            }
            // `onSetStatus` refusals for the ported status-immunity abilities.
            // The public immunity message only appears when the source effect
            // carries a `status` field, i.e. not for ability-sourced statuses.
            // `abilities:leafguard.onSetStatus`: while sun is effective the
            // holder refuses every new major status. The public immunity
            // message is only shown for a move that declares a primary status,
            // so a secondary roll is refused silently.
            if !suppressing
                && dex.effects.abilities[self.mon(target).ability as usize] == Ability::Leafguard
                && self.effective_weather(dex) == dex.effects.sun
            {
                if !secondary && ability_source.is_none() {
                    self.reveal_ability(target)?;
                }
                return Ok(false);
            }
            if !suppressing
                && self.status_immune_ability(dex, target, status).is_some()
            {
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
                // `mods/champions` overrides `slp.onStart`: the start time is
                // `this.sample([2, 3, 3])`, so the single draw maps 0 to two
                // turns and both other outcomes to three.
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
                //
                // `abilities:owntempo.onTryAddVolatile` refuses the confusion
                // outright (no message, no timer roll), and
                // `conditions:safeguard.onTryAddVolatile` refuses one that a
                // different Pokémon tried to inflict while the target's side
                // holds Safeguard. Both run before the add and neither reveals
                // an ability.
                let own_tempo =
                    dex.effects.abilities[self.mon(target).ability as usize] == Ability::OwnTempo;
                let safeguarded = target != source
                    && self.sides[target.side as usize]
                        .conditions
                        .contains_key(&dex.effects.safeguard);
                if !own_tempo && !safeguarded && !self.mon(target).volatiles.contains_key(&volatile) {
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
            } else if volatile == dex.effects.yawn
                && !suppressing
                && dex.effects.abilities[self.mon(target).ability as usize] == Ability::Leafguard
                && self.effective_weather(dex) == dex.effects.sun
            {
                // `abilities:leafguard.onTryAddVolatile`: the attempted Yawn is
                // refused and the immunity message reveals the ability.
                self.reveal_ability(target)?;
            } else if (volatile == dex.effects.yawn || volatile == dex.effects.confusion)
                && target != source
                && self.sides[target.side as usize]
                    .conditions
                    .contains_key(&dex.effects.safeguard)
            {
                // `conditions:safeguard.onTryAddVolatile`: Yawn and confusion
                // from another Pokémon are refused while the target's side
                // holds Safeguard.
                changed |= false;
            } else if volatile == dex.effects.encore
                || volatile == dex.effects.taunt
                || volatile == dex.effects.disable
                || volatile == dex.effects.imprison
                || volatile == dex.effects.torment
                || volatile == dex.effects.yawn
                || volatile == dex.effects.roost
                || volatile == dex.effects.glaive_rush
                || volatile == dex.effects.minimize
                || volatile == dex.effects.partially_trapped
                || volatile == dex.effects.leech_seed
                || volatile == dex.effects.heal_block
                || volatile == dex.effects.trapped
            {
                changed |= self.start_selection_volatile(
                    dex,
                    target,
                    Some(source),
                    volatile,
                    false,
                    suppressing,
                )?;
            } else if volatile == dex.effects.smack_down {
                // `moves:smackdown.condition.onStart`: only an airborne target
                // (Flying type or Levitate) gains the marker. An Iron Ball or
                // Ingrain holder and an active Gravity field are already
                // grounded and refuse it; the fly/bounce, Magnet Rise and
                // Telekinesis cancels are unreachable in this regulation and
                // stay explicit operational errors upstream.
                let airborne = self.effective_types(dex, target).contains(&dex.effects.flying)
                    || dex.effects.abilities[self.mon(target).ability as usize]
                        == Ability::Levitate;
                let already_grounded =
                    dex.effects.items[self.mon(target).item as usize] == Item::IronBall;
                if airborne && !already_grounded {
                    if !self.mon(target).volatiles.contains_key(&volatile) {
                        let order = self.allocate_effect_order()?;
                        self.mon_mut(target).volatiles.insert(
                            volatile,
                            EffectState {
                                id: volatile,
                                effect_order: order,
                                effect_order_assigned: true,
                                source: Some((
                                    if source.side == 0 { SideId::P1 } else { SideId::P2 },
                                    source.roster,
                                )),
                                ..Default::default()
                            },
                        );
                        self.emit(
                            EventKind::EffectStart,
                            target,
                            Some(source),
                            EffectRef::Condition(volatile),
                            0,
                            false,
                        )?;
                        changed = true;
                    } else {
                        changed |= false;
                    }
                } else {
                    changed |= false;
                }
            } else if volatile == dex.effects.no_retreat {
                // `moves:noretreat.condition`: a bare marker (no duration, no
                // payload) that also pins the holder in place.
                if !self.mon(target).volatiles.contains_key(&volatile) {
                    let order = self.allocate_effect_order()?;
                    self.mon_mut(target).volatiles.insert(
                        volatile,
                        EffectState {
                            id: volatile,
                            effect_order: order,
                            effect_order_assigned: true,
                            source: Some((
                                if source.side == 0 { SideId::P1 } else { SideId::P2 },
                                source.roster,
                            )),
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        Some(source),
                        EffectRef::Condition(volatile),
                        0,
                        false,
                    )?;
                    changed = true;
                }
            } else if volatile == dex.effects.charge {
                changed |= self.add_charge_volatile(dex, target, Some(source))?;
            } else if volatile == dex.effects.focus_energy || volatile == dex.effects.dragon_cheer {
                // Focus Energy / Dragon Cheer: no duration and no restart. The
                // `onStart` gate refuses the new volatile while the other crit
                // volatile is up, and an existing volatile makes the re-add
                // fail (the move reports failure).
                let other = if volatile == dex.effects.focus_energy {
                    dex.effects.dragon_cheer
                } else {
                    dex.effects.focus_energy
                };
                if self.mon(target).volatiles.contains_key(&volatile)
                    || self.mon(target).volatiles.contains_key(&other)
                {
                    changed |= false;
                } else {
                    let order = self.allocate_effect_order()?;
                    // `dragoncheer.condition.onStart` records whether the
                    // target was Dragon-type when the volatile started.
                    let values = if volatile == dex.effects.dragon_cheer {
                        vec![i64::from(self.mon(target).types.contains(&dex.effects.dragon))]
                    } else {
                        Vec::new()
                    };
                    self.mon_mut(target).volatiles.insert(
                        volatile,
                        EffectState {
                            id: volatile,
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
                            values,
                            ..Default::default()
                        },
                    );
                    self.emit(
                        EventKind::EffectStart,
                        target,
                        Some(source),
                        EffectRef::Condition(volatile),
                        0,
                        false,
                    )?;
                    changed = true;
                }
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
        suppressing: bool,
    ) -> Result<bool> {
        // `abilities:aromaveil.onAllyTryAddVolatile`: a holder on the
        // recipient's side refuses Taunt, Encore, Disable, Torment and Heal
        // Block from any source. The public block message (and reveal) only
        // appears for a move-sourced effect; a Cursed Body disable is refused
        // silently. Mold Breaker suppresses the breakable gate.
        if !suppressing
            && [
                dex.effects.encore,
                dex.effects.taunt,
                dex.effects.disable,
                dex.effects.torment,
                dex.effects.heal_block,
            ]
            .contains(&volatile)
        {
            let holder = self.active_entities(true).into_iter().find(|h| {
                h.side == target.side
                    && dex.effects.abilities[self.mon(*h).ability as usize] == Ability::Aromaveil
            });
            if let Some(holder) = holder {
                if !mid_move {
                    self.reveal_ability(holder)?;
                }
                return Ok(false);
            }
        }
        // `addVolatile` fails when the volatile already exists and declares no
        // `onRestart`; the rest of this family does not restart, and
        // `moves:minimize.condition.onRestart` returns null, which is refused
        // the same way (the caller has already applied the move's boosts).
        //
        // `Pokemon#addVolatile` also runs `runStatusImmunity`: the `trapped`
        // pseudo-type is refused outright by Ghost types, which makes Block,
        // Mean Look, Jaw Lock and Spirit Shackle fail against them.
        if volatile == dex.effects.trapped
            && self.mon(target).types.contains(&dex.effects.ghost)
        {
            return Ok(false);
        }
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
            // Reference `Pokemon#addVolatile`: the condition records the
            // seeder's *slot* (`source.getSlot()`); the residual resolves the
            // slot's current occupant with `Battle#getAtSlot`, so a seed keeps
            // draining into whoever stands in that slot after a switch or a
            // faint-and-replace.
            let seeder_slot = source
                .and_then(|seeder| self.mon(seeder).active_slot)
                .map(|slot| vec![i64::from(slot)])
                .unwrap_or_default();
            self.mon_mut(target).volatiles.insert(
                volatile,
                EffectState {
                    id: volatile,
                    source: source_slot,
                    effect_order: order,
                    effect_order_assigned: true,
                    values: seeder_slot,
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
        // `moves:yawn.condition` (2 turns, residual order 23),
        // `moves:roost.condition` (1 turn, residual order 25) and
        // `moves:healblock.condition` (2 turns from Psychic Noise, residual
        // order 20) are the ported volatiles that carry a numeric duration
        // without their own rest-of-family handling. Heal Block's
        // `durationCallback` returns 5 for the past-generation Heal Block move
        // and 7 under Persistent, but neither is legal in the pinned format,
        // so only Psychic Noise's 2 is reachable here.
        if volatile == dex.effects.yawn
            || volatile == dex.effects.roost
            || volatile == dex.effects.heal_block
        {
            let duration = match volatile {
                id if id == dex.effects.yawn => 2,
                id if id == dex.effects.heal_block => 2,
                _ => 1,
            };
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
        // `addVolatile(..., linkedStatus)`: a `trapped` marker links a silent
        // `trapper` partner onto its source, each side storing the other as its
        // source so that clearing either clears the pair.
        if volatile == dex.effects.trapped
            && let Some(source_entity) = source
            && source_entity != target
        {
            let slot = (
                if target.side == 0 {
                    SideId::P1
                } else {
                    SideId::P2
                },
                target.roster,
            );
            if !self.mon(source_entity).volatiles.contains_key(&dex.effects.trapper) {
                let order = self.allocate_effect_order()?;
                self.mon_mut(source_entity).volatiles.insert(
                    dex.effects.trapper,
                    EffectState {
                        id: dex.effects.trapper,
                        source: Some(slot),
                        effect_order: order,
                        effect_order_assigned: true,
                        ..Default::default()
                    },
                );
            }
        }
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
        // Drain in FIFO order including entries pushed while processing (the
        // reference `faintMessages` loops until its queue is empty, so a
        // Destiny Bond counter-faint is processed in the same pass).
        while !self.faint_queue.is_empty() {
            let data = self.faint_queue.remove(0);
            let e = data.target;
            if self.mon(e).fainted {
                continue;
            }
            self.emit(EventKind::Faint, e, None, EffectRef::None, 0, true)?;
            // `moves:destinybond.condition.onFaint`: a faint caused by a foe's
            // move drags the source down with it. The counter-faint is queued
            // with no source/effect (reference `source.faint()`), so it
            // cannot chain. Future moves are not ported, so `!futuremove`
            // holds for every `from_move` faint today.
            if data.from_move
                && let Some(source) = data.source
                && source.side != e.side
                && self.mon(e).volatiles.contains_key(&dex.effects.destiny_bond)
                && !self.mon(source).fainted
                && self.mon(source).hp > 0
            {
                // Reference `source.faint()` zeroes HP as it queues the
                // counter-faint (the Faint event itself comes later, during
                // faint processing).
                self.mon_mut(source).hp = 0;
                self.faint_queue.push(FaintData {
                    target: source,
                    source: None,
                    from_move: false,
                });
            }
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
            // `abilities:healer.onResidual` shares Hydration's order/sub-order;
            // the roll only happens for a statused adjacent ally.
            if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Healer {
                handlers.push((
                    e,
                    self.mon(e).ability,
                    14,
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
                if state.duration.is_some()
                    || id == dex.effects.leech_seed
                    || id == dex.effects.curse
                    || id == dex.effects.aqua_ring
                    // Duration-less volatiles with their own residual handler.
                    || id == dex.effects.ingrain
                    || id == dex.effects.octolock
                {
                    // A charge marker's volatile id is the *move* id, which
                    // shares the numeric space with condition ids: it declares
                    // no residual order of its own, so identify it first or a
                    // move id that happens to equal a condition id would borrow
                    // that condition's order (Bounce's move id equals the
                    // magnetrise condition id).
                    let is_charge_marker = (id as usize) < dex.moves.len()
                        && dex.moves[id as usize].charge.is_some();
                    // Reference `onResidualOrder`: Taunt 15, Encore 16, Disable
                    // 17, Throat Chop 22; other timed volatiles stay unordered.
                    let (order, sub_order) = if is_charge_marker {
                        (0, 0)
                    } else if id == dex.effects.taunt {
                        (15, 0)
                    } else if id == dex.effects.encore {
                        (16, 0)
                    } else if id == dex.effects.disable {
                        (17, 0)
                    } else if id == dex.effects.heal_block {
                        (20, 0)
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
                    } else if id == dex.effects.curse {
                        // `moves:curse.condition.onResidualOrder: 12`.
                        (12, 0)
                    } else if id == dex.effects.aqua_ring {
                        // `moves:aquaring.condition.onResidualOrder: 6`.
                        (6, 0)
                    } else if id == dex.effects.ingrain {
                        // `moves:ingrain.condition.onResidualOrder: 7`.
                        (7, 0)
                    } else if id == dex.effects.octolock {
                        // `moves:octolock.condition.onResidualOrder: 14`.
                        (14, 0)
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
            // Slot conditions resolve as side handlers whose target is the
            // slot's occupant (what `findSideEventHandlers(side, 'onResidual',
            // getKey, active)` passes), so they sort with that Pokémon's speed
            // at sub-order 3. Wish declares `onResidualOrder: 4`.
            for slot in 0..2usize {
                for &id in self.sides[side].slot_conditions[slot].keys() {
                    // Wish is the only ported slot condition with an
                    // `onResidual`; Healing Wish resolves on switch-in only.
                    if id != dex.effects.wish {
                        continue;
                    }
                    let Some(roster) = self.sides[side].active[slot] else {
                        continue;
                    };
                    let occupant = Entity {
                        side: side as u8,
                        roster,
                    };
                    handlers.push((
                        occupant,
                        id,
                        16 + slot as u8,
                        Priority {
                            order: 4,
                            sub_order: 3,
                            speed: self.mon(occupant).cached_speed,
                            ..Default::default()
                        },
                    ));
                }
            }
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
                } else if id == dex.effects.safeguard {
                    // `conditions:safeguard.onSideResidualOrder: 26`,
                    // `SubOrder: 3` (between Light Screen and Tailwind).
                    (26, 3)
                } else if [
                    dex.effects.spikes,
                    dex.effects.stealth_rock,
                    dex.effects.toxic_spikes,
                    dex.effects.sticky_web,
                ]
                .contains(&id)
                {
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
        if self.field.contains_key(&dex.effects.gravity) {
            handlers.push((
                Entity { side: 0, roster: 0 },
                dex.effects.gravity,
                15,
                Priority {
                    order: 27,
                    sub_order: 2,
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
            if (status == 16 || status == 17) && id == dex.effects.wish {
                // `moves:wish.condition.onResidual` (order 4): the marker
                // resolves on the first residual after the turn it started, so
                // the wisher's own residual leaves it in place. Removing it
                // runs `onEnd`, which heals the current occupant of that slot
                // for the stored half-maximum amount.
                let slot = usize::from(status - 16);
                let side = e.side as usize;
                let Some(state) = self.sides[side].slot_conditions[slot].get(&id) else {
                    continue;
                };
                let started = state.values.get(1).copied().unwrap_or(0);
                if i64::from(self.turn % 256) <= started {
                    continue;
                }
                let Some(state) = self.sides[side].slot_conditions[slot].remove(&id) else {
                    continue;
                };
                let amount = state.values.first().copied().unwrap_or(0).max(0) as u32;
                let Some(roster) = self.sides[side].active[slot] else {
                    continue;
                };
                let target = Entity {
                    side: e.side,
                    roster,
                };
                if self.mon(target).fainted || self.mon(target).hp == 0 {
                    continue;
                }
                let max = u32::from(self.mon(target).stats[0]);
                let healed = amount.min(max.saturating_sub(u32::from(self.mon(target).hp)));
                if healed == 0 {
                    continue;
                }
                self.mon_mut(target).hp += healed as u16;
                self.emit(
                    EventKind::Heal,
                    target,
                    None,
                    EffectRef::Condition(dex.effects.wish),
                    healed as i32,
                    true,
                )?;
                continue;
            }
            if status == 0 && id == dex.effects.curse {
                // `moves:curse.condition.onResidual`: the cursed holder loses a
                // quarter of its maximum HP to the curser. `this.damage` runs
                // the Damage event, so Magic Guard refuses it.
                let source = self.mon(e).volatiles.get(&id).and_then(|state| {
                    state.source.map(|(side, roster)| Entity {
                        side: side.index() as u8,
                        roster,
                    })
                });
                if let Some(source) = source
                    && self.mon(e).hp > 0
                {
                    let amount = (u32::from(self.mon(e).stats[0]) / 4).max(1);
                    self.indirect_damage(dex, e, source, amount, EffectRef::Condition(id))?;
                    self.process_faints(dex, true)?;
                    if self.outcome.terminated {
                        return Ok(());
                    }
                }
                continue;
            }
            if status == 0 && id == dex.effects.ingrain {
                // `moves:ingrain.condition.onResidual` (order 7): recover a
                // sixteenth of the maximum HP; Heal Block refuses the recovery
                // while the marker stays in place.
                if self.mon(e).hp > 0 && !self.heal_blocked(dex, e) {
                    let max = u32::from(self.mon(e).stats[0]);
                    let amount = (max / 16).max(1).min(max - u32::from(self.mon(e).hp));
                    if amount > 0 {
                        self.mon_mut(e).hp += amount as u16;
                        self.emit(
                            EventKind::Heal,
                            e,
                            None,
                            EffectRef::Condition(id),
                            amount as i32,
                            true,
                        )?;
                    }
                }
                continue;
            }
            if status == 0 && id == dex.effects.octolock {
                // `moves:octolock.condition.onResidual` (order 14): the marker
                // ends silently once its source left the field, fainted or has
                // not acted yet; otherwise the holder loses a stage of Defense
                // and Special Defense (the boost is attributed to the source,
                // so Clear Body and friends can refuse it).
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
                let source = source.unwrap();
                self.boost(
                    dex,
                    e,
                    source,
                    [0, -1, 0, -1, 0, 0, 0],
                    BoostCause::Move { secondary: false },
                )?;
                continue;
            }
            if status == 0 && id == dex.effects.aqua_ring {
                // `moves:aquaring.condition.onResidual` (order 6): recover a
                // sixteenth of the maximum HP; Heal Block refuses the recovery
                // while the volatile stays in place.
                if self.mon(e).hp > 0 && !self.heal_blocked(dex, e) {
                    let max = u32::from(self.mon(e).stats[0]);
                    let amount = (max / 16).max(1).min(max - u32::from(self.mon(e).hp));
                    if amount > 0 {
                        self.mon_mut(e).hp += amount as u16;
                        self.emit(
                            EventKind::Heal,
                            e,
                            None,
                            EffectRef::Condition(id),
                            amount as i32,
                            true,
                        )?;
                    }
                }
                continue;
            }
            if status == 0 && id == dex.effects.leech_seed {
                // `moves:leechseed.condition.onResidual` (order 8): drain an
                // eighth of the holder's maximum HP into the *current occupant*
                // of the seeding slot (`Battle#getAtSlot(sourceSlot)`); an empty
                // or fainted slot leeches nothing and keeps the seed.
                let Some(state) = self.mon(e).volatiles.get(&id) else {
                    continue;
                };
                let Some((side, _roster)) = state.source else {
                    continue;
                };
                let Some(slot) = state.values.first().copied() else {
                    continue;
                };
                let Some(roster) = self.sides[side.index()].active[slot as usize] else {
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
                let hp = self.mon(e).hp;
                self.mon_mut(e).hurt_this_turn = hp;
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
                // `this.heal(damage, target, pokemon)`: 'leechseed' is on both
                // TryHeal lists, so the seeded slot's Liquid Ooze and the
                // seeder's Big Root interact exactly like a drain heal.
                self.drain_heal(dex, source, e, u32::from(actual), EffectRef::Condition(id))?;
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
                self.terrain_upkeep(dex, id)?;
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
            if status == 15 {
                self.gravity_upkeep(dex)?;
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
            } else if status == 14 {
                // `abilities:healer.onResidual`: every statused adjacent ally
                // is cured on the Champions even 1/2 roll (the base game uses
                // 3/10; `data/mods/champions/abilities.ts` overrides it),
                // revealing the ability once per successful cure.
                if self.mon(e).ability != id {
                    continue;
                }
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
                for ally in allies {
                    if self.mon(ally).status != 0 && self.rng.chance(1, 2) {
                        self.reveal_ability(e)?;
                        self.cure_status(ally)?;
                    }
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
                    // The heal runs through `this.heal`, so Heal Block refuses
                    // it; the residual damage stays refused either way and the
                    // ability is only revealed by the heal message.
                    if p.hp > 0 && p.hp < p.stats[0] && !self.heal_blocked(dex, e) {
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
                if actual != 0 {
                    let hp = self.mon(e).hp;
                    self.mon_mut(e).hurt_this_turn = hp;
                }
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
                    let locked_true_duration = if id == dex.effects.locked_move {
                        self.mon(e)
                            .volatiles
                            .get(&id)
                            .and_then(|state| state.values.get(1))
                            .copied()
                    } else {
                        None
                    };
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
                    if id == dex.effects.locked_move {
                        // `conditions:lockedmove.onEnd` (reached on the expiry
                        // residual, which skips `onResidual`): the lock ends
                        // and the user is confused once the rolled duration has
                        // run out. A sleeping holder is not "calmed" here —
                        // only a non-expiry residual bypasses the fatigue.
                        if locked_true_duration.unwrap_or(0) <= 1 {
                            let fatigue = crate::effects::HitEffect {
                                volatile: dex.effects.confusion,
                                ..Default::default()
                            };
                            self.hit_effect(dex, e, e, &fatigue, false)?;
                        }
                    }
                    if id == dex.effects.protect
                        || id == dex.effects.throat_chop
                        || id == dex.effects.heal_block
                        || id == dex.effects.taunt
                        || id == dex.effects.encore
                        || id == dex.effects.disable
                        || id == dex.effects.torment
                        || id == dex.effects.yawn
                        || id == dex.effects.roost
                        || id == dex.effects.locked_move
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
                } else if id == dex.effects.locked_move {
                    // `conditions:lockedmove.onResidual`: a sleeping holder
                    // drops the lock without the end-of-rampage confusion (the
                    // reference deletes the volatile, so no End event runs);
                    // otherwise the rolled true duration loses a turn.
                    if self.mon(e).status == dex.effects.sleep {
                        self.mon_mut(e).volatiles.remove(&id);
                    } else if let Some(true_duration) = self
                        .mon_mut(e)
                        .volatiles
                        .get_mut(&id)
                        .and_then(|state| state.values.get_mut(1))
                    {
                        *true_duration = true_duration.saturating_sub(1);
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

    /// Reference `BattleQueue#willMove`: the queue still holds a move action
    /// for this Pokémon (`null` for a fainted one). The action being resolved
    /// has already been shifted off the queue, exactly like the reference.
    pub(crate) fn queued_to_move(&self, e: Entity) -> bool {
        if self.mon(e).fainted {
            return false;
        }
        self.queue
            .iter()
            .any(|q| q.kind == crate::effects::QueuedKind::Move && q.actor == Some(e))
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

    /// Reference `getSwitchRequestData().reviving` destinations: every fainted
    /// selected party member in request-team order, the only legal choice for a
    /// slot that holds the `revivalblessing` condition.
    fn revival_targets(&self, side: usize) -> Vec<u8> {
        let Some(order) = self.sides[side].selected_order else {
            return Vec::new();
        };
        order
            .iter()
            .copied()
            .filter(|r| self.sides[side].pokemon[*r as usize].fainted)
            .collect()
    }

    /// The active slot holding the `revivalblessing` slot condition, if any.
    fn revival_slot(&self, dex: &Dex, side: usize) -> Option<u8> {
        (0..2u8).find(|slot| {
            self.sides[side].slot_conditions[*slot as usize]
                .contains_key(&dex.effects.revival_blessing)
        })
    }

    /// `conditions:charge`: the Electric-doubling volatile. `onStart` and
    /// `onRestart` both announce it (naming the granting ability when
    /// Electromorphosis added it), and the marker carries no duration or
    /// payload beyond its source.
    fn add_charge_volatile(
        &mut self,
        dex: &Dex,
        target: Entity,
        source: Option<Entity>,
    ) -> Result<bool> {
        if !self.mon(target).volatiles.contains_key(&dex.effects.charge) {
            let order = self.allocate_effect_order()?;
            self.mon_mut(target).volatiles.insert(
                dex.effects.charge,
                EffectState {
                    id: dex.effects.charge,
                    effect_order: order,
                    effect_order_assigned: true,
                    source: source.map(|s| {
                        (
                            if s.side == 0 { SideId::P1 } else { SideId::P2 },
                            s.roster,
                        )
                    }),
                    ..Default::default()
                },
            );
        }
        self.emit(
            EventKind::EffectStart,
            target,
            source,
            EffectRef::Condition(dex.effects.charge),
            0,
            false,
        )?;
        Ok(true)
    }

    /// Reference `Pokemon#gotAttacked`: one `attackedBy` entry per move
    /// instance per target, carrying the damage of that move's *last* hit (the
    /// reference replaces `moveDamage` on every hit of its hit loop and records
    /// it once afterwards, so a multi-hit move keeps only the final hit) and
    /// the attacker. Metal Burst and Comeuppance read the last non-ally entry.
    fn record_attacked_by(&mut self, target: Entity, source: Entity, uid: u32, damage: u16) {
        let Some(slot) = self.mon(source).active_slot else {
            return;
        };
        let list = &mut self.mon_mut(target).attacked_by;
        if let Some(last) = list.last_mut()
            && last.move_uid == uid
            && last.source_side.index() == usize::from(source.side)
            && last.source_slot == slot
        {
            last.damage = damage;
            return;
        }
        list.push(crate::state::AttackedBy {
            move_uid: uid,
            source_side: if source.side == 0 {
                SideId::P1
            } else {
                SideId::P2
            },
            source_slot: slot,
            source_roster: source.roster,
            damage,
        });
    }

    /// The last `attackedBy` entry whose attacker is not an ally of `holder`,
    /// mirroring `getLastDamagedBy(true)`. A cleared bucket behaves exactly
    /// like the reference's stale `thisTurn: false` entries because every
    /// reader gates on the current turn.
    fn last_damaged_by(&self, holder: Entity) -> Option<(SideId, u8, u16)> {
        self.mon(holder)
            .attacked_by
            .iter()
            .rev()
            .find(|entry| entry.source_side.index() != usize::from(holder.side))
            .map(|entry| (entry.source_side, entry.source_slot, entry.damage))
    }

    /// Reference `Battle#getAtSlot`: the Pokémon currently occupying the
    /// recorded attacker's absolute slot, or `None` when that slot is empty.
    fn entity_at_slot(&self, side: SideId, slot: u8) -> Option<Entity> {
        self.sides[side.index()].active[usize::from(slot)].map(|roster| Entity {
            side: side.index() as u8,
            roster,
        })
    }

    /// `moves:charge.condition.onAfterMove|onMoveAborted`: any Electric-type
    /// move other than Charge itself consumes the volatile (silently ending).
    fn charge_after_move(&mut self, dex: &Dex, actor: Entity, move_id: Id) -> Result<()> {
        if move_id == 0
            || move_id == dex.effects.charge_move
            || !self.mon(actor).volatiles.contains_key(&dex.effects.charge)
        {
            return Ok(());
        }
        let declared = dex.moves[move_id as usize].move_type;
        let (kind, _) = self.converted_move_type(dex, actor, &dex.moves[move_id as usize], declared);
        if kind != dex.effects.electric {
            return Ok(());
        }
        self.mon_mut(actor).volatiles.remove(&dex.effects.charge);
        self.emit(
            EventKind::EffectEnd,
            actor,
            None,
            EffectRef::Condition(dex.effects.charge),
            0,
            false,
        )?;
        Ok(())
    }

    /// `conditions:lockedmove.onStart|onRestart`: a fresh start rolls
    /// `random(2, 4)` for the true duration and records the locked move; a
    /// re-add runs `onRestart`, which only refreshes the declared two-turn
    /// duration while the rolled duration still has turns left.
    fn start_locked_move(&mut self, dex: &Dex, actor: Entity, move_id: Id) -> Result<()> {
        if let Some(state) = self
            .mon_mut(actor)
            .volatiles
            .get_mut(&dex.effects.locked_move)
        {
            if state.values.get(1).copied().unwrap_or(0) >= 2 {
                state.duration = Some(2);
            }
            return Ok(());
        }
        let true_duration = i64::from(self.rng.range(2, 4));
        let order = self.allocate_effect_order()?;
        self.mon_mut(actor).volatiles.insert(
            dex.effects.locked_move,
            EffectState {
                id: dex.effects.locked_move,
                effect_order: order,
                effect_order_assigned: true,
                duration: Some(2),
                source: Some((
                    if actor.side == 0 {
                        SideId::P1
                    } else {
                        SideId::P2
                    },
                    actor.roster,
                )),
                values: vec![i64::from(move_id), true_duration],
            },
        );
        Ok(())
    }

    /// `conditions:lockedmove.onAfterMove`: the rampage lock ends after the
    /// move on the turn its declared duration has one tick left, and the user
    /// is confused once the rolled true duration has run out.
    fn locked_move_after_move(&mut self, dex: &Dex, actor: Entity) -> Result<()> {
        let Some(state) = self.mon(actor).volatiles.get(&dex.effects.locked_move) else {
            return Ok(());
        };
        if state.duration != Some(1) {
            return Ok(());
        }
        let true_duration = state.values.get(1).copied().unwrap_or(0);
        self.mon_mut(actor).volatiles.remove(&dex.effects.locked_move);
        self.emit(
            EventKind::EffectEnd,
            actor,
            None,
            EffectRef::Condition(dex.effects.locked_move),
            0,
            false,
        )?;
        if true_duration <= 1 {
            // `onEnd`: `target.addVolatile('confusion')` with the lock as its
            // source effect (`[fatigue]`); the timer roll matches any other
            // confusion start, and Own Tempo still refuses it.
            let fatigue = crate::effects::HitEffect {
                volatile: dex.effects.confusion,
                ..Default::default()
            };
            self.hit_effect(dex, actor, actor, &fatigue, false)?;
        }
        Ok(())
    }

    /// Reference `Battle#runAction` case `'revivalblessing'`: the chosen
    /// fainted party member returns at half its maximum HP with its status and
    /// faint flags cleared, the side counts one more living member and the
    /// slot condition is consumed. A revived member still occupying an active
    /// slot re-enters the field immediately (`instaswitch`), which runs the
    /// full switch-in pipeline (hazards, ability End/Start, Update).
    fn apply_revival_blessing(
        &mut self,
        dex: &Dex,
        actor: Entity,
        revived: Entity,
        slot: u8,
    ) -> Result<()> {
        let move_id = self.mon(actor).switch_flag.unwrap_or(0);
        self.sides[actor.side as usize].slot_conditions[slot as usize]
            .remove(&dex.effects.revival_blessing);
        // `Side#chooseSwitch` clears the user's switch flag when the revival
        // choice is accepted: the user does not leave the field.
        self.mon_mut(actor).switch_flag = None;
        self.mon_mut(actor).plain_switch_flag = false;
        let active_slot = self.mon(revived).active_slot;
        let healed = (self.mon(revived).stats[0] / 2).max(1);
        {
            let mon = self.mon_mut(revived);
            mon.fainted = false;
            mon.status = 0;
            mon.status_state = EffectState::default();
            mon.hp = healed;
        }
        self.emit(
            EventKind::Heal,
            revived,
            Some(actor),
            EffectRef::Move(move_id),
            i32::from(healed),
            false,
        )?;
        if let Some(active_slot) = active_slot {
            self.switch_in(dex, revived, active_slot)?;
        }
        Ok(())
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
        // `moves:noretreat.condition.onTrapPokemon`: the marker pins its own
        // holder in place (`tryTrap` always succeeds for the holder).
        if self.mon(e).volatiles.contains_key(&dex.effects.no_retreat) {
            trapped = Some(false);
        }
        // `moves:ingrain.condition.onTrapPokemon`: the marker pins its own
        // holder (`tryTrap` always succeeds for the holder).
        if self.mon(e).volatiles.contains_key(&dex.effects.ingrain) {
            trapped = Some(false);
        }
        // `moves:octolock.condition.onTrapPokemon`: the holder stays pinned
        // while the recorded source is still active.
        if let Some(state) = self.mon(e).volatiles.get(&dex.effects.octolock) {
            let source_active = state.source.is_some_and(|(side, roster)| {
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
        // `moves:trapped.condition.onTrapPokemon`: the marker pins its holder
        // while the trapper recorded as its source is still active.
        if let Some(state) = self.mon(e).volatiles.get(&dex.effects.trapped) {
            let source_active = state.source.is_some_and(|(side, roster)| {
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
                })
                || self.revival_slot(dex, side).is_some();
            if !flagged {
                continue;
            }
            // A revival choice needs no live reserve: its destinations are the
            // side's fainted members.
            if self.can_switch(side) || self.revival_slot(dex, side).is_some() {
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
            let reviving = self.revival_slot(dex, side);
            let slots = std::array::from_fn(|slot| {
                let Some(roster) = self.sides[side].active[slot] else {
                    return SlotRequest::default();
                };
                let p = &self.sides[side].pokemon[roster as usize];
                let slot_reviving = reviving == Some(slot as u8);
                SlotRequest {
                    present: !p.fainted,
                    // A `selfSwitch` pivot is alive and keeps its move list;
                    // only `forceSwitch` marks the slot as actionable.
                    requires_replacement: needed
                        && (p.switch_flag.is_some() || p.plain_switch_flag || slot_reviving),
                    reviving: slot_reviving,
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
                            target: self.served_target(
                                dex,
                                Entity {
                                    side: side as u8,
                                    roster,
                                },
                                mv.id,
                            ),
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
                revive_targets: if reviving.is_some() {
                    self.revival_targets(side)
                } else {
                    vec![]
                },
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
                revive_targets: vec![],
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
        // `moves:gravity.condition.onDisableMove`: a field condition, so the
        // reference collects it through the field handlers (`findEventHandlers`
        // line for `findFieldEventHandlers`) rather than from the holder.
        if self.field.contains_key(&dex.effects.gravity)
            && let Some(&sub_order) = dex.effects.disable_move_conditions.get(&dex.effects.gravity)
        {
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
                // Reference `nextTurn`: the per-turn damage and stat-history
                // flags reset for every active Pokémon.
                mon.hurt_this_turn = 0;
                mon.stats_raised_this_turn = false;
                mon.stats_lowered_this_turn = false;
                mon.newly_switched = false;
                // Reference turn-loop rollover: older `attackedBy` entries lose
                // `thisTurn` (or drop when their attacker left the field);
                // Metal Burst / Comeuppance only ever read the current turn, so
                // the bucket is cleared outright.
                mon.attacked_by.clear();
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
                // `moves:healblock.condition.onDisableMove`: every `heal`-flag
                // move is disabled in the request while the volatile is active.
                let heal_block = mon
                    .volatiles
                    .contains_key(&dex.effects.heal_block);
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
                // `moves:gravity.condition.onDisableMove`: every `flags.gravity`
                // move is disabled in the request while Gravity is up.
                let gravity_up = self.field.contains_key(&dex.effects.gravity);
                // `moves:torment.condition.onDisableMove`: the last used move.
                let tormented = mon.volatiles.contains_key(&dex.effects.torment);
                let last_move = mon.last_move;
                for mv in &mut mon.moves {
                    let mut disabled = locked.is_some_and(|id| id != i64::from(mv.id))
                        || (fake_out_disabled
                            && (mv.id == dex.effects.fake_out
                                || mv.id == dex.effects.first_impression))
                        || (throat_chop && dex.moves[mv.id as usize].sound)
                        || (heal_block && dex.moves[mv.id as usize].heal)
                        || (gravity_up && dex.moves[mv.id as usize].gravity);
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
                    // `flags.cantusetwice`: the reference disables the move
                    // while it is still the holder's last used move, so a
                    // consecutive Gigaton Hammer is refused by the request.
                    if dex.moves[mv.id as usize].cant_use_twice && last_move == mv.id {
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
                            target: self.served_target(
                                dex,
                                Entity {
                                    side: side as u8,
                                    roster,
                                },
                                mv.id,
                            ),
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
                revive_targets: vec![],
                preview_roster: vec![],
            };
        }
        Ok(())
    }
}
