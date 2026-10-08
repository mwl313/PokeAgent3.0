//! Terrain field lifecycle and grounded effects. Unsupported grounding-changing
//! items/abilities/volatiles stay guarded until their complete handlers exist.
use super::*;

impl BattleState {
    pub(super) fn terrain_id(&self, dex: &Dex) -> Id {
        [
            dex.effects.electric_terrain,
            dex.effects.grassy_terrain,
            dex.effects.misty_terrain,
            dex.effects.psychic_terrain,
        ]
        .into_iter()
        .find(|id| self.field.contains_key(id))
        .unwrap_or(0)
    }

    pub(super) fn grounded(&self, dex: &Dex, e: Entity) -> bool {
        // `conditions:smackdown`: the marker grounds its holder regardless of
        // type or ability.
        if self.mon(e).volatiles.contains_key(&dex.effects.smack_down) {
            return true;
        }
        // `conditions:gravity`: while the pseudo-weather is up every active
        // Pokémon is grounded (`BattlePokemon#isGrounded`).
        if self.field.contains_key(&dex.effects.gravity) {
            return true;
        }
        // `moves:ingrain.condition`: the marker grounds its holder regardless
        // of type or ability.
        if self.mon(e).volatiles.contains_key(&dex.effects.ingrain) {
            return true;
        }
        !self.effective_types(dex, e).contains(&dex.effects.flying)
            && dex.effects.abilities[self.mon(e).ability as usize] != Ability::Levitate
    }

    pub(super) fn start_terrain(
        &mut self,
        dex: &Dex,
        source: Entity,
        id: Id,
        ability: bool,
    ) -> Result<bool> {
        let old = self.terrain_id(dex);
        if old == id {
            return Ok(false);
        }
        if ![
            dex.effects.electric_terrain,
            dex.effects.grassy_terrain,
            dex.effects.misty_terrain,
            dex.effects.psychic_terrain,
        ]
        .contains(&id)
        {
            return Err(EngineError::Unsupported(format!("terrain {id}")));
        }
        let duration = if dex.effects.items[self.mon(source).item as usize] == Item::TerrainExtender
        {
            8
        } else {
            5
        };
        if old != 0 {
            self.field.remove(&old);
            // Semantic state replacement, not a reference onFieldEnd callback.
            self.emit(
                EventKind::FieldEffectEnd,
                source,
                None,
                EffectRef::Condition(old),
                0,
                false,
            )?;
        }
        self.field.insert(
            id,
            EffectState {
                id,
                duration: Some(duration),
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
        if ability {
            self.reveal_ability(source)?;
        }
        self.emit(
            EventKind::FieldEffectStart,
            source,
            None,
            EffectRef::Condition(id),
            i32::from(duration),
            false,
        )?;
        self.field_change_order();
        // Reference `setTerrain` runs the global TerrainChange event after the
        // field state is committed.
        super::item_ports::terrain_change_event(self, dex)?;
        Ok(true)
    }

    pub(super) fn terrain_upkeep(&mut self, dex: &Dex, id: Id) -> Result<()> {
        let state = self.field.get_mut(&id).unwrap();
        let duration = state.duration.as_mut().unwrap();
        *duration -= 1;
        let expired = *duration == 0;
        for knowledge in &mut self.knowledge {
            if let Some(effect) = knowledge.field.get_mut(&id)
                && effect.duration.known
            {
                effect.duration.value -= 1;
            }
        }
        if expired {
            let state = self.field.remove(&id).unwrap();
            let (side, roster) = state.source.unwrap();
            self.emit(
                EventKind::FieldEffectEnd,
                Entity {
                    side: side.index() as u8,
                    roster,
                },
                None,
                EffectRef::Condition(id),
                0,
                false,
            )?;
            self.field_change_order();
            // Reference `field.clearTerrain` (the residual handler's `end`):
            // the terrain is dropped and then the global TerrainChange event
            // runs, so terrain-bound handlers (Mimicry, seeds) re-evaluate.
            super::item_ports::terrain_change_event(self, dex)?;
        }
        Ok(())
    }

    /// `abilities:mimicry.onTerrainChange`: the holder adopts the active
    /// terrain's type and reverts to its base typing when no terrain is up.
    /// Both reference call sites are public (the `-start ... typechange [from]
    /// ability: Mimicry` message and the `-activate ability: Mimicry` revert),
    /// so a change reveals the ability to both players.
    pub(super) fn mimicry_terrain_change(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if self.mon(e).hp == 0
            || dex.effects.abilities[self.mon(e).ability as usize] != Ability::Mimicry
        {
            return Ok(());
        }
        let terrain = self.terrain_id(dex);
        let types: SmallVec<[Id; 2]> = if terrain == dex.effects.electric_terrain {
            smallvec![dex.effects.electric]
        } else if terrain == dex.effects.grassy_terrain {
            smallvec![dex.effects.grass]
        } else if terrain == dex.effects.misty_terrain {
            smallvec![dex.effects.fairy]
        } else if terrain == dex.effects.psychic_terrain {
            smallvec![dex.effects.psychic]
        } else {
            dex.species[self.mon(e).base_species as usize]
                .types
                .clone()
                .into()
        };
        if self.mon(e).types.as_slice() == types.as_slice() {
            return Ok(());
        }
        self.set_type(dex, e, &types)?;
        // Both players see the new typing immediately, exactly like a forme
        // change.
        for viewer in 0..2 {
            let index = e.roster as usize + if e.side as usize == viewer { 0 } else { 6 };
            self.knowledge[viewer].pokemon[index].types = types.to_vec();
        }
        self.reveal_ability(e)?;
        Ok(())
    }

    /// Reference `Field#clearTerrain`: run the active terrain's FieldEnd, drop
    /// the field state and then run the global TerrainChange event. Steel
    /// Roller is the only legal caller in the pinned regulation.
    pub(super) fn clear_terrain(&mut self, dex: &Dex, source: Entity) -> Result<bool> {
        let Some(id) = [
            dex.effects.electric_terrain,
            dex.effects.grassy_terrain,
            dex.effects.misty_terrain,
            dex.effects.psychic_terrain,
        ]
        .into_iter()
        .find(|id| self.field.contains_key(id))
        else {
            return Ok(false);
        };
        self.field.remove(&id);
        self.emit(
            EventKind::FieldEffectEnd,
            source,
            None,
            EffectRef::Condition(id),
            0,
            false,
        )?;
        self.field_change_order();
        super::item_ports::terrain_change_event(self, dex)?;
        Ok(true)
    }

    pub(super) fn grassy_heal(&mut self, dex: &Dex, target: Entity) -> Result<()> {
        let p = self.mon(target);
        // `moves:grassyterrain.condition.onResidual` heals through `this.heal`.
        if p.fainted
            || p.hp == 0
            || p.hp == p.stats[0]
            || !self.grounded(dex, target)
            || self.heal_blocked(dex, target)
        {
            return Ok(());
        }
        let amount = (p.stats[0] / 16).max(1).min(p.stats[0] - p.hp);
        self.mon_mut(target).hp += amount;
        self.emit(
            EventKind::Heal,
            target,
            Some(target),
            EffectRef::Condition(dex.effects.grassy_terrain),
            i32::from(amount),
            true,
        )
    }

    pub(super) fn terrain_power_modifier(&self, dex: &Dex, context: MoveContext<'_>) -> u32 {
        let terrain = self.terrain_id(dex);
        let m = context.move_data;
        if (terrain == dex.effects.grassy_terrain && dex.effects.quake_moves.contains(&m.id)
            || terrain == dex.effects.misty_terrain && m.move_type == dex.effects.dragon)
            && self.grounded(dex, context.target)
        {
            return 2048;
        }
        if ((terrain == dex.effects.electric_terrain && m.move_type == dex.effects.electric)
            || (terrain == dex.effects.grassy_terrain && m.move_type == dex.effects.grass)
            || (terrain == dex.effects.psychic_terrain && m.move_type == dex.effects.psychic))
            && self.grounded(dex, context.actor)
        {
            5325
        } else {
            4096
        }
    }
}
