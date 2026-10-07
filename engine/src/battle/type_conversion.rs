//! Scalar action-local overrides borrow cold metadata; conversion never clones
//! the Move's vectors or mutates the shared Dex.
use super::*;
use std::ops::Deref;

pub(super) struct ActiveMove<'a> {
    data: &'a crate::assets::Move,
    pub move_type: Id,
    /// Action-local target class. `Target::AllAdjacentFoes` for a grounded
    /// Expanding Force in Psychic Terrain.
    pub target: Target,
    pub power: u16,
    /// Action-local accuracy; `None` means the move always hits.
    pub accuracy: Option<u8>,
    pub type_changer_boosted: Option<Ability>,
    /// `abilities:sheerforce.onModifyMove` deleted this action's secondaries and
    /// self effect and marked it for the priority-21 BasePower boost.
    pub sheer_force: bool,
    /// `abilities:scrappy.onModifyMove`: Normal and Fighting moves ignore the
    /// Ghost type immunity for this action.
    pub scrappy: bool,
}
impl Deref for ActiveMove<'_> {
    type Target = crate::assets::Move;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl BattleState {
    pub(super) fn active_move<'a>(
        &self,
        dex: &Dex,
        actor: Entity,
        data: &'a crate::assets::Move,
        behavior: MoveBehavior,
    ) -> ActiveMove<'a> {
        let mut action = ActiveMove {
            data,
            move_type: data.move_type,
            target: data.target,
            power: data.power,
            accuracy: data.accuracy,
            type_changer_boosted: None,
            sheer_force: false,
            scrappy: false,
        };
        // The move's own callbacks precede the actor's ModifyType event.
        if behavior == MoveBehavior::Struggle {
            action.move_type = 0;
        }
        // `moves:terrainpulse.onModifyType|onModifyMove`: a grounded user's
        // Terrain Pulse takes the active terrain's type and doubles its power.
        if behavior == MoveBehavior::TerrainPulse {
            let terrain = self.terrain_id(dex);
            if terrain != 0 && self.grounded(dex, actor) {
                action.power = data.power * 2;
                action.move_type = if terrain == dex.effects.electric_terrain {
                    dex.effects.electric
                } else if terrain == dex.effects.grassy_terrain {
                    dex.effects.grass
                } else if terrain == dex.effects.misty_terrain {
                    dex.effects.fairy
                } else {
                    dex.effects.psychic
                };
            }
        }
        let weather = self.effective_weather(dex);
        if behavior == MoveBehavior::WeatherBall && weather != 0 {
            action.power = 100;
            action.move_type = if weather == dex.effects.rain {
                dex.effects.water
            } else if weather == dex.effects.sun {
                dex.effects.fire
            } else if weather == dex.effects.sand {
                dex.effects.rock
            } else {
                dex.effects.ice
            };
        }
        // Ported move-owned `onModifyMove` callbacks use the target's effective
        // weather, which is the field's effective weather in every supported
        // state (no Cloud Nine/utility-umbrella override differs per target).
        let hooks = dex.effects.move_hooks[data.id as usize];
        if hooks & crate::effects::hook::ACCURACY_SNOW != 0 && weather == dex.effects.snow {
            action.accuracy = None;
        }
        if hooks & crate::effects::hook::ACCURACY_RAIN_SUN != 0 {
            if weather == dex.effects.rain {
                action.accuracy = None;
            } else if weather == dex.effects.sun {
                action.accuracy = Some(50);
            }
        }
        if hooks & crate::effects::hook::EXPANDING_FORCE != 0
            && self.terrain_id(dex) == dex.effects.psychic_terrain
            && self.grounded(dex, actor)
        {
            // `moves:expandingforce.onModifyMove`: a grounded user in Psychic
            // Terrain makes the move hit both adjacent foes.
            action.target = Target::AllAdjacentFoes;
        }
        let ability = dex.effects.abilities[self.mon(actor).ability as usize];
        // `abilities:sheerforce.onModifyMove`: a move with declared secondaries
        // loses its secondaries and self effect for this action and gains the
        // Sheer Force BasePower marker.
        if ability == Ability::Sheerforce && !data.secondaries.is_empty() {
            action.sheer_force = true;
        }
        // `abilities:scrappy.onModifyMove`: the action gains a Normal/Fighting
        // immunity bypass, without changing its type or category.
        if ability == Ability::Scrappy {
            action.scrappy = true;
        }
        if ability == Ability::LiquidVoice {
            // Dynamax is not a legal Champions state and remains unsupported.
            if data.sound {
                action.move_type = dex.effects.water;
            }
            return action;
        }
        let destination = match ability {
            Ability::Pixilate => dex.effects.fairy,
            Ability::Aerilate => dex.effects.flying,
            Ability::Refrigerate => dex.effects.ice,
            Ability::Galvanize => dex.effects.electric,
            Ability::Dragonize => dex.effects.dragon,
            Ability::Normalize => dex.effects.normal,
            _ => return action,
        };
        let excluded = if ability == Ability::Normalize {
            data.normalize_excluded
        } else {
            data.conversion_excluded
        };
        if (ability == Ability::Normalize || action.move_type == dex.effects.normal)
            && (!excluded || data.is_max)
            && !(data.is_z && data.category != Category::Status)
        {
            action.move_type = destination;
            action.type_changer_boosted = Some(ability);
        }
        // These callbacks do not emit an ability reveal or draw RNG. Other
        // ordered ModifyType participants remain explicitly unsupported.
        action
    }
}
