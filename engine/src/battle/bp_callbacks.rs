//! Native ports of reference `basePowerCallback` formulas. Each helper is a
//! direct transcription of the pinned callback source and reads only local
//! battle state; no registry scan or reference call occurs here.
use super::*;
use crate::effects::BasePowerKind;

impl BattleState {
    /// Reference `pokemon.getStat('spe')`: the stored speed stat with stat
    /// stages applied and no ability, item, weather or side-condition
    /// modifiers. Callback formulas that read raw stats use this.
    fn raw_speed(&self, _dex: &Dex, e: Entity) -> i32 {
        stats::apply_stage(u32::from(self.mon(e).stats[5]), self.mon(e).boosts[4]) as i32
    }

    /// Returns the callback-derived base power, or the move's declared power
    /// when it has no ported callback.
    pub(super) fn base_power(
        &self,
        dex: &Dex,
        m: &ActiveMove<'_>,
        actor: Entity,
        target: Entity,
        hit: u32,
    ) -> u32 {
        let kind = m.bp_callback;
        let declared = u32::from(m.power);
        let Some(kind) = kind else {
            return declared;
        };
        match kind {
            // `moves:tripleaxel.basePowerCallback`: `20 * move.hit`.
            BasePowerKind::TripleAxel => declared * hit.max(1),
            BasePowerKind::Acrobatics => {
                if self.mon(actor).item == 0 {
                    declared * 2
                } else {
                    declared
                }
            }
            BasePowerKind::ElectroBall => {
                // `getStat('spe')`: stored stat with stat stages only, before
                // ability/item/weather speed modifiers.
                let actor_speed = self.raw_speed(dex, actor);
                let target_speed = self.raw_speed(dex, target);
                let ratio = if target_speed <= 0 {
                    0
                } else {
                    (actor_speed / target_speed).max(0) as usize
                };
                [40, 60, 80, 120, 150][ratio.min(4)]
            }
            BasePowerKind::Eruption => {
                let mon = self.mon(actor);
                if mon.hp == 0 {
                    return 0;
                }
                // Reference keeps this fractional (`move.basePower * hp / maxhp`)
                // and only truncates at the damage formula or the first
                // participating BasePower handler. Return the numerator here;
                // `base_power_den` supplies the denominator.
                declared.saturating_mul(u32::from(mon.hp))
            }
            BasePowerKind::Flail => {
                let mon = self.mon(actor);
                if mon.hp == 0 {
                    return 0;
                }
                let ratio = ((u32::from(mon.hp) * 48) / u32::from(mon.stats[0])).max(1);
                match ratio {
                    0..=1 => 200,
                    2..=4 => 150,
                    5..=9 => 100,
                    10..=16 => 80,
                    17..=32 => 40,
                    _ => 20,
                }
            }
            BasePowerKind::GrassKnot | BasePowerKind::LowKick => {
                // Reference compares `target.getWeight()` (hectograms) against
                // 2000/1000/500/250/100, not the 10x-scaled grams.
                let weight = dex.species[self.mon(target).species as usize].weight_hg;
                match weight {
                    w if w >= 2_000 => 120,
                    w if w >= 1_000 => 100,
                    w if w >= 500 => 80,
                    w if w >= 250 => 60,
                    w if w >= 100 => 40,
                    _ => 20,
                }
            }
            BasePowerKind::GyroBall => {
                let actor_speed = self.raw_speed(dex, actor);
                let target_speed = self.raw_speed(dex, target);
                if actor_speed <= 0 {
                    return 1;
                }
                // JS float division then `Math.floor(...) + 1`, capped at 150.
                let power = ((25 * target_speed) / actor_speed) + 1;
                power.clamp(1, 150) as u32
            }
            BasePowerKind::HardPress => {
                let mon = self.mon(target);
                if mon.stats[0] == 0 {
                    return 1;
                }
                let hp_fraction = (u32::from(mon.hp) * 4096) / u32::from(mon.stats[0]);
                // Math.floor(Math.floor((100 * (100 * floor(hp*4096/maxhp)) + 2048 - 1) / 4096) / 100) || 1
                let scaled = 100 * hp_fraction;
                let inner = (100 * scaled + 2048 - 1) / 4096;
                (inner / 100).max(1)
            }
            BasePowerKind::HeatCrash => {
                let actor_weight = dex.species[self.mon(actor).species as usize].weight_hg;
                let target_weight = dex.species[self.mon(target).species as usize].weight_hg;
                if actor_weight >= target_weight * 5 {
                    120
                } else if actor_weight >= target_weight * 4 {
                    100
                } else if actor_weight >= target_weight * 3 {
                    80
                } else if actor_weight >= target_weight * 2 {
                    60
                } else {
                    40
                }
            }
            BasePowerKind::Hex | BasePowerKind::InfernalParade => {
                if self.mon(target).status != 0 {
                    declared * 2
                } else {
                    declared
                }
            }
            BasePowerKind::LastRespects => {
                let fainted = self.sides[actor.side as usize]
                    .pokemon
                    .iter()
                    .filter(|p| p.fainted || p.hp == 0)
                    .count() as u32;
                50 + 50 * fainted
            }
            BasePowerKind::RageFist => {
                // `moves:ragefist.basePowerCallback`: 50 + 50 * timesAttacked
                // with a hard cap of 350. The counter is the holder's own
                // `timesAttacked`, incremented once per landed hit taken.
                (50 + 50 * u32::from(self.mon(actor).times_attacked)).min(350)
            }
            BasePowerKind::StompingTantrum => {
                // `moves:stompingtantrum.basePowerCallback`: doubles only when
                // the user's previous move failed (reference `false`, not the
                // `null` of a skipped recharge / charge turn).
                if self.mon(actor).move_last_turn_result == crate::state::MoveResult::Failed {
                    declared * 2
                } else {
                    declared
                }
            }
            BasePowerKind::TemperFlare => {
                // `moves:temperflare.basePowerCallback`: identical test to
                // Stomping Tantrum's previous-move-failure read.
                if self.mon(actor).move_last_turn_result == crate::state::MoveResult::Failed {
                    declared * 2
                } else {
                    declared
                }
            }
            BasePowerKind::Assurance => {
                // `moves:assurance.basePowerCallback`: `target.hurtThisTurn`
                // is the target's post-damage HP from any earlier damage this
                // turn; it is falsy when no damage landed or the target
                // fainted (HP 0) before this check.
                if self.mon(target).hurt_this_turn {
                    declared * 2
                } else {
                    declared
                }
            }
            BasePowerKind::PowerTrip => {
                let boosts: u32 = self.mon(actor).boosts.iter().map(|b| (*b).max(0) as u32).sum();
                declared + 20 * boosts
            }
            BasePowerKind::RisingVoltage => {
                if self.terrain_id(dex) == dex.effects.electric_terrain && self.grounded(dex, target)
                {
                    declared * 2
                } else {
                    declared
                }
            }
            // `moves:beatup.basePowerCallback`: `5 + floor(baseAtk / 10)` of
            // the ally the current hit consumes. `move.allies` is captured by
            // `onModifyMove`; the callback shifts one entry per hit, starting
            // with the first (hit 1).
            BasePowerKind::BeatUp => {
                let Some(&roster) = m.allies.get(hit.saturating_sub(1) as usize) else {
                    return 0;
                };
                let member = Entity {
                    side: actor.side,
                    roster,
                };
                let base_atk =
                    u32::from(dex.species[self.mon(member).base_species as usize].base_stats[1]);
                5 + base_atk / 10
            }
        }
    }

    /// Denominator of a callback-derived base power. Only `move.basePower *
    /// hp / maxhp` style callbacks are fractional; everything else is integral.
    pub(super) fn base_power_den(&self, kind: Option<BasePowerKind>, actor: Entity) -> u32 {
        match kind {
            Some(BasePowerKind::Eruption) => u32::from(self.mon(actor).stats[0]).max(1),
            _ => 1,
        }
    }
}
