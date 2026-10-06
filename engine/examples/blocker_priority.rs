//! Development tool: rank the native-mechanics gaps by the number of training
//! teams each one still blocks, including the greedy cascade (fixing one gap
//! can unlock teams that were blocked by several).
//!
//! This reads the frozen 1,136-team pool and the exported regulation tables; it
//! never restricts engine scope to that pool and never changes team data.
use pa3_engine::assets::Dex;
use pa3_engine::effects::{Item, MoveBehavior};
use pa3_engine::state::Team;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

type Gap = String;

fn team_gaps(dex: &Dex, team: &Team) -> BTreeSet<Gap> {
    let mut gaps = BTreeSet::new();
    for member in &team.members {
        if !dex.effects.abilities[member.ability as usize].is_ported() {
            gaps.insert(format!(
                "ability {}",
                dex.names["abilities"][member.ability as usize]
            ));
        }
        if dex.effects.items[member.item as usize] == Item::Unimplemented
            && dex.effects.mega_stones[member.item as usize].is_empty()
        {
            gaps.insert(format!(
                "item {}",
                dex.names["items"][member.item as usize]
            ));
        }
        for &move_id in &member.moves {
            if dex.effects.moves[move_id as usize] == MoveBehavior::Unimplemented {
                gaps.insert(format!("move {}", dex.names["moves"][move_id as usize]));
            }
        }
    }
    gaps
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let dex = Dex::load(&dir).unwrap();
    let teams: Vec<Team> =
        serde_json::from_slice(&std::fs::read(dir.join("training-teams.json")).unwrap()).unwrap();
    let gaps: Vec<BTreeSet<Gap>> = teams.iter().map(|t| team_gaps(&dex, t)).collect();
    let mut remaining: BTreeSet<usize> = (0..teams.len()).collect();
    let mut fixed: BTreeSet<Gap> = BTreeSet::new();

    println!("{} teams; {} initially runnable", teams.len(), gaps.iter().filter(|g| g.is_empty()).count());
    let mut rank = 0;
    while !remaining.is_empty() && rank < 400 {
        let mut counts: BTreeMap<Gap, usize> = BTreeMap::new();
        for &index in &remaining {
            let blocking: BTreeSet<&Gap> = gaps[index].difference(&fixed).collect();
            if blocking.len() == 1 {
                *counts.entry((*blocking.iter().next().unwrap()).clone()).or_default() += 1;
            }
        }
        let Some((best, solved)) = counts.into_iter().max_by_key(|(_, count)| *count) else {
            // No single fix unblocks anything; report the most common first-order blocker.
            let mut first_order: BTreeMap<Gap, usize> = BTreeMap::new();
            for &index in &remaining {
                for gap in gaps[index].difference(&fixed) {
                    *first_order.entry(gap.clone()).or_default() += 1;
                }
            }
            let mut top: Vec<(Gap, usize)> = first_order.into_iter().collect();
            top.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            println!("-- no single fix unlocks a team; largest first-order blockers:");
            for (gap, count) in top.into_iter().take(20) {
                println!("   {count:5}  {gap}");
            }
            break;
        };
        println!("{:2}. {solved:5} teams  <- fix {best}", rank + 1);
        fixed.insert(best);
        rank += 1;
        remaining.retain(|&index| gaps[index].difference(&fixed).next().is_some());
        println!(
            "    runnable now: {} / {} (fixed {} gaps)",
            teams.len() - remaining.len(),
            teams.len(),
            fixed.len()
        );
    }
    let blocked: Vec<&usize> = remaining.iter().collect();
    println!("remaining blocked teams after greedy pass: {}", remaining.len());
    if let Some(&first) = blocked.first() {
        let sample: Vec<Gap> = gaps[*first].difference(&fixed).cloned().collect();
        println!("example remaining gap set: {sample:?}");
    }
}
