//! Development tool: list ported abilities that have no interaction witness in
//! `engine/data/ability-interactions.json`, ranked by how many frozen-pool team
//! slots use them. Cold path only; never part of a battle transition.
//!
//! A ported ability without a witness is not a proof of a bug, but it is the
//! highest-risk uncovered surface: the differential corpus cannot catch a
//! wrong port until a fixture exists. Abilities already covered by the older
//! corpora (move hooks, contact family, flash fire, weather/terrain turn
//! fixtures) still show up here unless this corpus itself carries a fixture.
use pa3_engine::assets::Dex;
use pa3_engine::state::Team;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize)]
struct Corpus {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    ability: String,
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let dex = Dex::load(&dir).unwrap();
    let teams: Vec<Team> =
        serde_json::from_slice(&std::fs::read(dir.join("training-teams.json")).unwrap()).unwrap();
    let corpus: Corpus =
        serde_json::from_slice(&std::fs::read(dir.join("ability-interactions.json")).unwrap()).unwrap();
    let witnessed: std::collections::BTreeSet<String> =
        corpus.fixtures.iter().map(|f| f.ability.clone()).collect();

    let mut slots: BTreeMap<u16, usize> = BTreeMap::new();
    for team in &teams {
        for member in &team.members {
            *slots.entry(member.ability).or_default() += 1;
        }
    }
    let mut unwitnessed: Vec<(usize, String)> = slots
        .iter()
        .filter(|(id, _)| {
            dex.effects.abilities[**id as usize].is_ported()
                && !witnessed.contains(&dex.names["abilities"][**id as usize])
                // Unused catalogue entries (empty ability) never appear in a set.
                && **id != 0
        })
        .map(|(id, count)| (*count, dex.names["abilities"][*id as usize].clone()))
        .collect();
    unwitnessed.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let ported = dex
        .effects
        .abilities
        .iter()
        .filter(|a| a.is_ported())
        .count();
    println!(
        "interaction corpus: {} fixtures covering {} abilities; {ported} abilities ported",
        corpus.fixtures.len(),
        witnessed.len()
    );
    println!("ported pool abilities without an interaction witness (team slots):");
    for (count, name) in &unwitnessed {
        println!("  {count:5}  {name}");
    }
    println!("{} ported pool abilities still unwitnessed by this corpus", unwitnessed.len());
}
