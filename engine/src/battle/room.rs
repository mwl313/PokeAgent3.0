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
