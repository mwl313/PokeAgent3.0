//! Integer damage kernel. Inputs are the results of the native effect event
//! chain; this kernel does not replace move/ability/item handlers.
use crate::{
    EngineError, Result,
    rng::BattleRng,
    stats::{modify, random_damage},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DamageInput {
    pub level: u16,
    pub power: u32,
    pub attack: u32,
    pub defense: u32,
    pub spread: bool,
    pub parental_bond_second_hit: bool,
    pub weather_modifier: u32,
    pub critical: bool,
    pub stab_modifier: u32,
    pub effectiveness: i8,
    pub burn: bool,
    pub final_modifier: u32,
    pub bypass_protect: bool,
    /// Denominator for a fractional callback base power (`move.basePower *
    /// hp / maxhp`). `1` means the power is integral.
    #[serde(default = "one")]
    pub power_den: u32,
}

const fn one() -> u32 {
    1
}

/// Critical-hit sampling occurs before the BasePower event and damage kernel.
/// Even a guaranteed 1/1 crit from the stage table consumes a reference draw.
pub fn critical_hit(stage: u8, guaranteed: Option<bool>, rng: &mut BattleRng) -> bool {
    if let Some(value) = guaranteed {
        return value;
    }
    let denominators = [0, 24, 8, 2, 1];
    let denominator = denominators[stage.min(4) as usize];
    denominator != 0 && rng.chance(1, denominator)
}

/// Combine event modifiers with reference 12-bit fixed-point rounding.
pub fn chain_modifiers(a: u32, b: u32) -> u32 {
    ((a.wrapping_mul(b).wrapping_add(2048) as i32) >> 12) as u32
}

pub fn calculate(input: DamageInput, rng: &mut BattleRng) -> Result<u16> {
    let damage = calculate_before_final(input, rng)?;
    Ok(finish_damage(
        damage,
        input.final_modifier,
        input.bypass_protect,
    ))
}

/// The ModifyDamage event runs after the damage roll, STAB and burn. Keeping
/// this boundary explicit preserves hook ordering when that event consumes RNG.
pub fn calculate_before_final(input: DamageInput, rng: &mut BattleRng) -> Result<u32> {
    if input.defense == 0 {
        return Err(EngineError::InvalidInput("zero defense".into()));
    }
    if !(-6..=6).contains(&input.effectiveness) {
        return Err(EngineError::InvalidInput(
            "damage kernel requires resolved immunity and clamped effectiveness".into(),
        ));
    }
    let stage = 2 * u32::from(input.level) / 5 + 2;
    // Fractional callback powers keep the reference's `tr(tr(base * power) *
    // attack)` grouping: the division happens after the attack multiplication.
    let mut damage = if input.power_den > 1 {
        ((u64::from(stage) * u64::from(input.power) * u64::from(input.attack))
            / u64::from(input.power_den)) as u32
    } else {
        stage.wrapping_mul(input.power).wrapping_mul(input.attack)
    };
    damage = damage / input.defense / 50;
    damage += 2;
    if input.spread {
        damage = modify(damage, 3072);
    } else if input.parental_bond_second_hit {
        damage = modify(damage, 1024);
    }
    damage = modify(damage, input.weather_modifier);
    if input.critical {
        damage = (u64::from(damage) * 3 / 2) as u32;
    }
    damage = random_damage(damage, rng);
    damage = modify(damage, input.stab_modifier);
    if input.effectiveness > 0 {
        damage *= 1 << input.effectiveness;
    } else {
        damage >>= -input.effectiveness;
    }
    if input.burn {
        damage = modify(damage, 2048);
    }
    Ok(damage)
}

pub fn finish_damage(mut damage: u32, final_modifier: u32, bypass_protect: bool) -> u16 {
    damage = modify(damage, final_modifier);
    if bypass_protect {
        damage = modify(damage, 1024);
    }
    // The minimum check precedes the final 16-bit overflow, as in the reference.
    if damage == 0 { 1 } else { damage as u16 }
}
