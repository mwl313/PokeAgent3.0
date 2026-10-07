//! Champions Trick Room negates modified action speed without mainline wrapping.
use super::*;

impl BattleState {
    pub(super) fn action_speed(&self, dex: &Dex, speed: u32) -> i32 {
        stats::action_speed(speed, self.field.contains_key(&dex.effects.trick_room))
    }

    fn end_trick_room(&mut self, dex: &Dex) -> Result<()> {
        let state = self.field.remove(&dex.effects.trick_room).unwrap();
        let (side, roster) = state.source.unwrap();
        self.emit(
            EventKind::FieldEffectEnd,
            Entity {
                side: side.index() as u8,
                roster,
            },
            None,
            EffectRef::Condition(dex.effects.trick_room),
            0,
            false,
        )
    }

    pub(super) fn toggle_trick_room(&mut self, dex: &Dex, source: Entity) -> Result<()> {
        if self.field.contains_key(&dex.effects.trick_room) {
            return self.end_trick_room(dex);
        }
        self.field.insert(
            dex.effects.trick_room,
            EffectState {
                id: dex.effects.trick_room,
                duration: Some(5),
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
        // Room duration has no legal hidden-item modifier. Both players know it.
        self.emit_public_field_start(source, dex.effects.trick_room, 5)
    }

    /// `moves:gravity.condition`: a five-turn pseudo-weather. `Persistent` is
    /// not legal in the pinned regulation, so the duration is always five. The
    /// start removes every aerial lock the pinned regulation can create (the
    /// Fly / Bounce charge markers and their `twoturnmove`), cancels those
    /// queued actions, and announce the Grounding. `magnetrise`, `telekinesis`
    /// and `skydrop` volatiles cannot exist while their moves stay unported.
    pub(super) fn start_gravity(&mut self, dex: &Dex, source: Entity) -> Result<bool> {
        if self.field.contains_key(&dex.effects.gravity) {
            // References `Field#addPseudoWeather`: no `onRestart` handler, so a
            // second cast while Gravity is up fails without touching state.
            return Ok(false);
        }
        self.field.insert(
            dex.effects.gravity,
            EffectState {
                id: dex.effects.gravity,
                duration: Some(5),
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
        // Unlike Trick Room, Gravity's duration is private to the caster: the
        // shared `emit` zeroes `FieldEffectStart` values for the other viewer,
        // so only the owner learns the remaining turns.
        self.emit(
            EventKind::FieldEffectStart,
            source,
            None,
            EffectRef::Condition(dex.effects.gravity),
            5,
            false,
        )?;
        for e in self.active_entities(false) {
            // The two-turn marker volatile is keyed by the charging move's own
            // id, so the reference's `removeVolatile('fly'|'bounce')` targets
            // those move ids directly.
            let charging = [dex.effects.fly_move, dex.effects.bounce_move]
                .into_iter()
                .find(|id| self.mon(e).volatiles.contains_key(id));
            if let Some(id) = charging {
                self.mon_mut(e).volatiles.remove(&id);
                // `queue.cancelMove(pokemon)`: the charge's queued action is
                // dropped before it resolves.
                self.queue.retain(|q| q.actor != Some(e));
                self.mon_mut(e).volatiles.remove(&dex.effects.two_turn_move);
                self.emit(
                    EventKind::Ability,
                    e,
                    None,
                    EffectRef::Condition(dex.effects.gravity),
                    0,
                    false,
                )?;
            }
        }
        Ok(true)
    }

    /// `moves:gravity.condition.onFieldResidual` (order 27, sub-order 2).
    pub(super) fn gravity_upkeep(&mut self, dex: &Dex) -> Result<()> {
        let state = self.field.get_mut(&dex.effects.gravity).unwrap();
        let duration = state.duration.as_mut().unwrap();
        *duration -= 1;
        let expired = *duration == 0;
        for knowledge in &mut self.knowledge {
            if let Some(effect) = knowledge.field.get_mut(&dex.effects.gravity)
                && effect.duration.known
            {
                effect.duration.value -= 1;
            }
        }
        if expired {
            let state = self.field.remove(&dex.effects.gravity).unwrap();
            let (side, roster) = state.source.unwrap();
            self.emit(
                EventKind::FieldEffectEnd,
                Entity {
                    side: side.index() as u8,
                    roster,
                },
                None,
                EffectRef::Condition(dex.effects.gravity),
                0,
                false,
            )?;
        }
        Ok(())
    }

    pub(super) fn trick_room_upkeep(&mut self, dex: &Dex) -> Result<()> {
        let state = self.field.get_mut(&dex.effects.trick_room).unwrap();
        let duration = state.duration.as_mut().unwrap();
        *duration -= 1;
        let expired = *duration == 0;
        for knowledge in &mut self.knowledge {
            if let Some(effect) = knowledge.field.get_mut(&dex.effects.trick_room) {
                effect.duration.value -= 1;
            }
        }
        if expired {
            self.end_trick_room(dex)?;
        }
        Ok(())
    }

    fn emit_public_field_start(&mut self, source: Entity, id: Id, duration: i32) -> Result<()> {
        for viewer in 0..2 {
            let event = SemanticEvent {
                kind: EventKind::FieldEffectStart,
                subject: source.roster + if source.side as usize == viewer { 0 } else { 6 },
                target: None,
                effect: id,
                effect_kind: crate::knowledge::EffectKind::Condition,
                value: duration,
                health: None,
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
}
