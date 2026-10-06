//! Exact reference selection-sort/tie-shuffle order. Ordinary unstable sorting
//! would perturb the RNG stream and therefore every later random effect.
use crate::rng::BattleRng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Priority {
    pub order: u32,
    /// Scaled by 10,000 to represent reference fractional priorities exactly.
    pub priority: i32,
    pub speed: i32,
    pub sub_order: i32,
    pub effect_order: u32,
}

impl Priority {
    pub fn compare(&self, other: &Self) -> Ordering {
        let order = |x| if x == 0 { 1u64 << 32 } else { u64::from(x) };
        order(self.order)
            .cmp(&order(other.order))
            .then_with(|| other.priority.cmp(&self.priority))
            .then_with(|| other.speed.cmp(&self.speed))
            .then_with(|| self.sub_order.cmp(&other.sub_order))
            .then_with(|| self.effect_order.cmp(&other.effect_order))
    }

    /// TryHit/DamagingHit use stable target order, not speed ties or RNG.
    pub fn compare_left_to_right(
        &self,
        index: usize,
        other: &Self,
        other_index: usize,
    ) -> Ordering {
        let order = |x| if x == 0 { 1u64 << 32 } else { u64::from(x) };
        order(self.order)
            .cmp(&order(other.order))
            .then_with(|| other.priority.cmp(&self.priority))
            .then_with(|| index.cmp(&other_index))
    }
}

pub fn speed_sort<T>(values: &mut [T], rng: &mut BattleRng, priority: impl Fn(&T) -> Priority) {
    let mut sorted = 0;
    // Events can outnumber the four active Pokémon; do not cap handler count.
    let mut next: SmallVec<[usize; 32]> = SmallVec::new();
    while sorted + 1 < values.len() {
        next.clear();
        next.push(sorted);
        for i in sorted + 1..values.len() {
            match priority(&values[next[0]]).compare(&priority(&values[i])) {
                Ordering::Less => (),
                Ordering::Greater => {
                    next.clear();
                    next.push(i);
                }
                Ordering::Equal => next.push(i),
            }
        }
        for (i, &index) in next.iter().enumerate() {
            values.swap(sorted + i, index);
        }
        rng.shuffle(&mut values[sorted..sorted + next.len()]);
        sorted += next.len();
    }
}
