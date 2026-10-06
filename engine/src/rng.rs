//! Showdown's explicitly seeded Gen5 RNG, ported from sim/prng.ts (MIT).
//! The reference also supports Sodium seeds; this API deliberately requires four
//! 16-bit seed words, so both engines select the same reference RNG algorithm.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BattleRng {
    state: u64,
    pub draws: u64,
}

impl BattleRng {
    pub fn new(seed: [u16; 4]) -> Self {
        Self {
            state: seed.into_iter().fold(0, |a, b| (a << 16) | u64::from(b)),
            draws: 0,
        }
    }

    pub fn seed(&self) -> [u16; 4] {
        [48, 32, 16, 0].map(|shift| (self.state >> shift) as u16)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(0x5D588B656C078965)
            .wrapping_add(0x269EC3);
        self.draws = self.draws.wrapping_add(1);
        (self.state >> 32) as u32
    }

    /// A draw is consumed even for an interval of size one, as in Showdown.
    pub fn below(&mut self, upper: u32) -> u32 {
        ((u64::from(self.next_u32()) * u64::from(upper)) >> 32) as u32
    }

    pub fn range(&mut self, lower: u32, upper: u32) -> u32 {
        assert!(upper >= lower);
        lower + self.below(upper - lower)
    }

    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        assert!(denominator > 0 && numerator <= denominator);
        self.below(denominator) < numerator
    }

    /// Forward Fisher-Yates; the iteration direction affects every later draw.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for start in 0..values.len().saturating_sub(1) {
            let next = self.range(start as u32, values.len() as u32) as usize;
            values.swap(start, next);
        }
    }
}
