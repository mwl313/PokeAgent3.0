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
    /// `moves:beatup.onModifyMove`: the party members the action's hits consume
    /// in order, as roster indices in reference `side.pokemon` order. The user
    /// is always included; every other member only while alive and
    /// status-free.
    pub allies: SmallVec<[u8; 4]>,
    /// Reference `move.callsMove`: this action was invoked by another move
    /// (`BattleActions#useMove`) rather than chosen by the player. Handlers
    /// such as the Metronome item's counter read it to tell a called move from
    /// a chosen one.
    pub calls_move: bool,
    /// Reference `move.hasBounced`: this action was already reflected by Magic
    /// Bounce, so a second holder must not bounce it again.
    pub has_bounced: bool,
    /// The action's effective priority (`battle.queue` writes the
    /// `ModifyPriority` result back onto the active move). A nested `useMove`
    /// inherits the outer action's priority; `None` means "not yet resolved".
    pub priority: Option<i8>,
    /// Action-local copy of `move.tracksTarget`. `abilities:stalwart` sets it
    /// for every non-scripted move its holder uses, so `getMoveTargets` skips
    /// redirection.
    pub tracks_target: bool,
    /// `abilities:infiltrator.onModifyMove`: every move the holder uses
    /// ignores the target's decoy (and its side's screens).
    pub infiltrates: bool,
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
            allies: SmallVec::new(),
            calls_move: false,
            has_bounced: false,
            priority: None,
            tracks_target: data.tracks_target,
            infiltrates: false,
        };
        // `abilities:stalwart.onModifyMove` (priority 1): the holder's moves
        // ignore redirection. Stalwart has no `breakable` flag, so Mold Breaker
        // never suppresses it.
        if dex.effects.abilities[self.mon(actor).ability as usize] == Ability::Stalwart
            && data.target != Target::Scripted
        {
            action.tracks_target = true;
        }
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
        // `moves:beatup.onModifyMove`: capture `pokemon.side.pokemon.filter(
        // ally => ally === pokemon || (!ally.fainted && !ally.status))`. The
        // reference filters the live party array, so a member that has already
        // fainted or carries a major status is skipped for this action.
        if hooks & crate::effects::hook::BEAT_UP != 0 {
            for &roster in &self.sides[actor.side as usize].positions {
                let member = &self.sides[actor.side as usize].pokemon[roster as usize];
                if roster == actor.roster || (!member.fainted && member.status == 0) {
                    action.allies.push(roster);
                }
            }
        }
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
        // `moves:ragingbull.onModifyType`: the three Paldea forms take their
        // own primary type; every other species (including plain Tauros and a
        // Metronome-item caller) keeps the declared Normal type.
        if hooks & crate::effects::hook::RAGING_BULL != 0 {
            let species = self.mon(actor).species;
            action.move_type = if species == dex.effects.tauros_paldea_combat {
                dex.effects.fighting
            } else if species == dex.effects.tauros_paldea_blaze {
                dex.effects.fire
            } else if species == dex.effects.tauros_paldea_aqua {
                dex.effects.water
            } else {
                action.move_type
            };
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
        // `abilities:infiltrator.onModifyMove`: the action ignores the target's
        // decoy (`moves:substitute.condition.onTryPrimaryHit` returns early on
        // `move.infiltrates`).
        if ability == Ability::Infiltrator {
            action.infiltrates = true;
        }
        let (move_type, type_changer_boosted) =
            self.converted_move_type(dex, actor, data, action.move_type);
        action.move_type = move_type;
        action.type_changer_boosted = type_changer_boosted;
        // These callbacks do not emit an ability reveal or draw RNG. Other
        // ordered ModifyType participants remain explicitly unsupported.
        action
    }

    /// The action's effective type after the ported `ModifyType` abilities:
    /// Pixilate/Aerilate/Refrigerate/Galvanize/Dragonize convert Normal moves,
    /// Normalize converts everything outside its exclusion list, and Liquid
    /// Voice converts sound moves to Water. Returns the resulting type and the
    /// ability that converted it (the BasePower boost key).
    pub(super) fn converted_move_type(
        &self,
        dex: &Dex,
        actor: Entity,
        data: &crate::assets::Move,
        current_type: Id,
    ) -> (Id, Option<Ability>) {
        let ability = dex.effects.abilities[self.mon(actor).ability as usize];
        if ability == Ability::LiquidVoice {
            // Dynamax is not a legal Champions state and remains unsupported.
            if data.sound {
                return (dex.effects.water, None);
            }
            return (current_type, None);
        }
        let destination = match ability {
            Ability::Pixilate => dex.effects.fairy,
            Ability::Aerilate => dex.effects.flying,
            Ability::Refrigerate => dex.effects.ice,
            Ability::Galvanize => dex.effects.electric,
            Ability::Dragonize => dex.effects.dragon,
            Ability::Normalize => dex.effects.normal,
            _ => return (current_type, None),
        };
        let excluded = if ability == Ability::Normalize {
            data.normalize_excluded
        } else {
            data.conversion_excluded
        };
        if (ability == Ability::Normalize || current_type == dex.effects.normal)
            && (!excluded || data.is_max)
            && !(data.is_z && data.category != Category::Status)
        {
            (destination, Some(ability))
        } else {
            (current_type, None)
        }
    }
}
