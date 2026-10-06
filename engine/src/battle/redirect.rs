//! Fast-exit RedirectTarget dispatch. Unlike ordinary events, exact speed ties
//! use private ability activation order and never shuffle or consume RNG.
use super::*;

#[derive(Clone, Copy)]
struct RedirectHandler {
    holder: Entity,
    priority: i32,
    speed: i32,
    /// Private reference `effectOrder`: ability activation order for abilities,
    /// volatile creation order for Follow Me / Rage Powder.
    effect_order: u32,
    /// Ability redirections reveal the holder when the chosen target changes;
    /// the move-sourced volatiles do not.
    reveal: bool,
    /// Rage Powder is ignored by a powder-immune attacker.
    powder: bool,
    /// `onAnyRedirectTarget` (Lightning Rod / Storm Drain) collects from every
    /// active Pokémon; `onFoeRedirectTarget` (Follow Me / Rage Powder) only
    /// from the attacker's foes.
    foe_only: bool,
}

impl BattleState {
    pub(super) fn redirect_target(
        &mut self,
        dex: &Dex,
        actor: Entity,
        m: &ActiveMove<'_>,
        selected: Option<Entity>,
    ) -> Result<Option<Entity>> {
        // These branches do not dispatch RedirectTarget in getMoveTargets.
        if m.tracks_target
            || matches!(
                m.target,
                Target::All
                    | Target::FoeSide
                    | Target::AllySide
                    | Target::AllyTeam
                    | Target::AllAdjacent
                    | Target::AllAdjacentFoes
                    | Target::Allies
            )
        {
            return Ok(selected);
        }
        if m.smart_target {
            return Err(EngineError::Unsupported("smart move targeting".into()));
        }
        if m.pledge_combo {
            return Ok(selected);
        }
        let target_kind = match m.target {
            Target::RandomNormal | Target::AdjacentFoe | Target::Scripted => Target::Normal,
            kind => kind,
        };
        let mut handlers: SmallVec<[RedirectHandler; 4]> = SmallVec::new();
        for holder in self.active_entities(false) {
            let mon = self.mon(holder);
            let matches_type = match dex.effects.abilities[mon.ability as usize] {
                Ability::LightningRod => m.move_type == dex.effects.electric,
                Ability::StormDrain => m.move_type == dex.effects.water,
                _ => false,
            };
            if matches_type {
                handlers.push(RedirectHandler {
                    holder,
                    priority: 0,
                    speed: mon.cached_speed,
                    effect_order: mon.ability_effect_order.unwrap_or(0),
                    reveal: true,
                    powder: false,
                    foe_only: false,
                });
            }
            // `followme` / `ragepowder` conditions both carry
            // `onFoeRedirectTargetPriority: 1` and redirect any single-target
            // move that could legally target the holder.
            if let Some(state) = mon.volatiles.get(&dex.effects.follow_me) {
                handlers.push(RedirectHandler {
                    holder,
                    priority: 1,
                    speed: mon.cached_speed,
                    effect_order: state.effect_order,
                    reveal: false,
                    powder: false,
                    foe_only: true,
                });
            }
            if let Some(state) = mon.volatiles.get(&dex.effects.rage_powder) {
                handlers.push(RedirectHandler {
                    holder,
                    priority: 1,
                    speed: mon.cached_speed,
                    effect_order: state.effect_order,
                    reveal: false,
                    powder: true,
                    foe_only: true,
                });
            }
        }
        handlers.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| b.speed.cmp(&a.speed))
                .then_with(|| a.effect_order.cmp(&b.effect_order))
        });
        for handler in handlers {
            if handler.foe_only && handler.holder.side == actor.side {
                continue;
            }
            if handler.powder && self.powder_immune(dex, actor) {
                continue;
            }
            let holder = handler.holder;
            let slot = self
                .mon(holder)
                .active_slot
                .expect("active redirect holder");
            let loc = if holder.side == actor.side {
                -(slot as i8 + 1)
            } else {
                slot as i8 + 1
            };
            if target_kind.valid_location(self.mon(actor).active_slot.unwrap_or(0), loc) {
                // An already selected holder still returns immediately: a
                // slower holder cannot replace it, and no reveal is generated.
                if handler.reveal && selected != Some(holder) {
                    self.reveal_ability(holder)?;
                }
                return Ok(Some(holder));
            }
        }
        Ok(selected)
    }
}
