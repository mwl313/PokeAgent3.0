//! Integer Champions calculations, not legacy EV calculations.
use serde::{Deserialize, Serialize};

pub const STAT_NAMES: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nature {
    pub plus: Option<usize>,
    pub minus: Option<usize>,
}

pub fn champions_stats(
    base: [u16; 6],
    points: [u8; 6],
    nature: Nature,
    max_hp: Option<u16>,
) -> [u16; 6] {
    std::array::from_fn(|i| {
        let stat = u32::from(base[i]) + u32::from(points[i]) + if i == 0 { 75 } else { 20 };
        if i == 0 {
            return max_hp.unwrap_or(stat as u16);
        }
        // Reference truncates the multiplied nature value to 16 bits first.
        if nature.plus == Some(i) {
            ((stat * 110) as u16) / 100
        } else if nature.minus == Some(i) {
            ((stat * 90) as u16) / 100
        } else {
            stat as u16
        }
    })
}

pub fn champions_pp(base_pp: u8, no_boost: bool) -> u8 {
    let capped = base_pp.min(20);
    if no_boost {
        capped
    } else {
        (capped / 5 + 1) * 4
    }
}

/// Showdown's fixed-point modifier rounding: exact half rounds down.
pub fn modify(value: u32, modifier: u32) -> u32 {
    ((u64::from(value.wrapping_mul(modifier)) + 2047) >> 12) as u32
}

pub fn apply_stage(stat: u32, stage: i8) -> u32 {
    let stage = stage.clamp(-6, 6);
    if stage >= 0 {
        stat * (2 + stage as u32) / 2
    } else {
        stat * 2 / (2 + (-stage) as u32)
    }
}

/// Reference damage randomizer: one uniform draw in [0,16), mapped to 100..85.
pub fn random_damage(damage: u32, rng: &mut crate::rng::BattleRng) -> u32 {
    damage.wrapping_mul(100 - rng.below(16)) / 100
}

/// Positive Math.round of a rational amount (drain/recoil), not the
/// fixed-point modifier rule: exact halves round UP at this boundary.
pub fn round_fraction(value: u32, [numerator, denominator]: [u16; 2]) -> u32 {
    let product = u64::from(value) * u64::from(numerator);
    ((2 * product + u64::from(denominator)) / (2 * u64::from(denominator))) as u32
}

/// Champions action speed after modifiers and the stat cap. The Champions
/// override removes mainline Trick Room underflow and uses signed negation.
pub fn action_speed(modified_speed: u32, trick_room: bool) -> i32 {
    if trick_room {
        -(modified_speed as i32)
    } else {
        modified_speed as i32
    }
}
