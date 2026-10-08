//! Development probe: replay corpus fixtures and snapshot after every step so
//! the first state shape the snapshot validator rejects is reported with its
//! fixture, decision and step. Never used by training.
use pa3_engine::{assets::Dex, state::BattleState};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Corpus {
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    name: String,
    seed: [u16; 4],
    teams: [pa3_engine::state::Team; 2],
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    side: pa3_engine::state::SideId,
    actions: Vec<pa3_engine::actions::AtomicAction>,
}

fn main() {
    let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
    let dex = Dex::load(Path::new(&dir)).unwrap();
    let corpus: Corpus =
        serde_json::from_str(&corpus::corpus_json(&corpus::data_dir())).unwrap();
    let filter = std::env::args().nth(1);
    let mut failures = 0usize;
    for fixture in &corpus.fixtures {
        if let Some(needle) = &filter
            && !fixture.name.contains(needle.as_str())
        {
            continue;
        }
        let mut state =
            BattleState::reset(&dex, [&fixture.teams[0], &fixture.teams[1]], fixture.seed, [0, 1])
                .unwrap();
        if let Err(error) = roundtrip(&dex, &state) {
            println!("{} initial: {error}", fixture.name);
            failures += 1;
            continue;
        }
        for (index, step) in fixture.steps.iter().enumerate() {
            state.step(&dex, step.side, &step.actions).unwrap();
            if std::env::var("SNAPSHOT_PROBE_ITEMS").is_ok() {
                let world: serde_json::Value =
                    serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
                let world: serde_json::Value =
                    serde_json::from_str(world["payload"].as_str().unwrap()).unwrap();
                let names = ["P1", "P2"]
                    .iter()
                    .enumerate()
                    .map(|(side, label)| {
                        let mons = world["sides"][side]["pokemon"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .take(2)
                            .map(|m| {
                                format!(
                                    "{} item={} prev={} hp={}",
                                    m["species"], m["item"], m["previous_item"], m["hp"]
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(" | ");
                        format!("{label}: {mons}")
                    })
                    .collect::<Vec<_>>()
                    .join("  ||  ");
                println!("{} step {index}: {names}", fixture.name);
            }
            if let Err(error) = roundtrip(&dex, &state) {
                println!(
                    "{} decision {index} {:?}: {error}",
                    fixture.name, step.side
                );
                failures += 1;
                break;
            }
        }
    }
    println!("snapshot probe failures: {failures}");
}

fn roundtrip(dex: &Dex, state: &BattleState) -> Result<(), pa3_engine::EngineError> {
    BattleState::restore(dex, &state.snapshot()?).map(|_| ())
}

#[path = "../test_support/corpus.rs"]
mod corpus;
