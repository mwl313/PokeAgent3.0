//! Development probe: replay one turn-fixtures corpus entry and print the
//! native boundary state after every step. Never used by training.
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file_arg = args
        .iter()
        .position(|a| a == "--file")
        .map(|index| args.get(index + 1).cloned().expect("--file needs a path"));
    let mut positional = Vec::new();
    let mut skip_next = false;
    for arg in &args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--file" {
            skip_next = true;
            continue;
        }
        if !arg.starts_with("--") {
            positional.push(arg.clone());
        }
    }
    let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
    let dex = Dex::load(Path::new(&dir)).unwrap();
    // The ability-interaction corpus is a separate artifact but shares the
    // fixture shape, so the probe accepts it through an explicit flag.
    let ledger = std::env::args().any(|a| a == "--ledger");
    let corpus: Corpus = if let Some(path) = &file_arg {
        // Scratch corpora (a filtered generator run, a held-out scene) stay on
        // disk; the checked-in artifacts remain compiled in.
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    } else if std::env::args().any(|a| a == "--artifact") {
        // Generator artifacts hold the current fixture shape even while a
        // ledger entry still carries the snapshot it was diagnosed from.
        let artifacts: Vec<serde_json::Value> = [
            include_str!("../data/more_interactions.json"),
            include_str!("../data/more_move_coverage.json"),
            include_str!("../data/more_roost_yawn.json"),
            include_str!("../data/more_delayed_status.json"),
            include_str!("../data/more_metronome_item.json"),
            include_str!("../data/more_magicbounce.json"),
            include_str!("../data/more_healer_curious.json"),
        ]
        .into_iter()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .unwrap();
        Corpus {
            fixtures: artifacts
                .into_iter()
                .flat_map(|artifact| artifact["fixtures"].as_array().unwrap().to_vec())
                .map(|fixture| serde_json::from_value(fixture).unwrap())
                .collect(),
        }
    } else if ledger {
        // Held-out fixtures stay in the mismatch ledger until they pass;
        // replay them here so a diagnosis can continue without merging.
        #[derive(Deserialize)]
        struct Ledger {
            mismatches: Vec<LedgerEntry>,
        }
        #[derive(Deserialize)]
        struct LedgerEntry {
            name: String,
            fixture: serde_json::Value,
        }
        let raw: Ledger =
            serde_json::from_str(include_str!("../data/known-mismatches.json")).unwrap();
        // A ledger entry may point at a fixture by name when the fixture lives
        // in a generator artifact; resolve those from the artifact that
        // contains them. Fixed entries whose fixture has since been merged (or
        // superseded) are skipped instead of aborting the probe, so the open
        // entries stay replayable.
        let merged: serde_json::Value =
            serde_json::from_str(&corpus::corpus_json(&corpus::data_dir())).unwrap();
        let mut artifacts: Vec<serde_json::Value> = [
            include_str!("../data/more_interactions.json"),
            include_str!("../data/more_move_coverage.json"),
            include_str!("../data/more_roost_yawn.json"),
            include_str!("../data/more_delayed_status.json"),
            include_str!("../data/more_heal_block.json"),
            include_str!("../data/more_sleep_talk.json"),
            include_str!("../data/more_frisk_pressure.json"),
            include_str!("../data/more_curse.json"),
            include_str!("../data/more_megasol.json"),
        ]
        .into_iter()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .unwrap();
        artifacts.push(merged);
        let mut fixtures = Vec::new();
        for entry in raw.mismatches {
            let value = if entry.fixture.is_string() {
                let Some(found) = artifacts
                    .iter()
                    .flat_map(|a| a["fixtures"].as_array().unwrap())
                    .find(|f| f["name"] == entry.fixture)
                else {
                    eprintln!("{}: fixture no longer generated; skipped", entry.name);
                    continue;
                };
                found.clone()
            } else {
                entry.fixture
            };
            fixtures.push(
                serde_json::from_value(value).unwrap_or_else(|e| panic!("{}: {e}", entry.name)),
            );
        }
        Corpus { fixtures }
    } else if std::env::args().any(|a| a == "--ability-corpus") {
        serde_json::from_str(include_str!("../data/ability-interactions.json")).unwrap()
    } else {
        serde_json::from_str(&corpus::corpus_json(&corpus::data_dir())).unwrap()
    };
    let needle = positional.first().expect("fixture name");
    let limits: usize = positional
        .get(1)
        .map(|v| v.parse().unwrap())
        .unwrap_or(usize::MAX);
    let fixture = corpus
        .fixtures
        .iter()
        .find(|f| &f.name == needle)
        .expect("unknown fixture");
    let mut state =
        BattleState::reset(&dex, [&fixture.teams[0], &fixture.teams[1]], fixture.seed, [0, 1])
            .unwrap();
    state.enable_trace().unwrap();
    for (index, step) in fixture.steps.iter().enumerate() {
        if index > limits {
            break;
        }
        if let Err(error) = state.step(&dex, step.side, &step.actions) {
            println!("step {index}: ERROR {error:?}");
            break;
        }
        let view = state.observe(pa3_engine::state::SideId::P1);
        println!(
            "step {index} side {:?} turn {} request {:?} seed {:?} draws {}",
            step.side,
            view.turn,
            state.observe(step.side).request.kind,
            state.rng_seed(),
            state.rng_draws(),
        );
        if std::env::var("PA3_DUMP_SNAPSHOT").is_ok() {
            let envelope: serde_json::Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
            let world: serde_json::Value = serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
            for (side, label) in ["P1", "P2"].iter().enumerate() {
                let active = &world["sides"][side]["active"];
                let mons = world["sides"][side]["pokemon"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m["fainted"] == true || m["active_slot"].is_u64())
                    .map(|(i, m)| format!("{i}: faint={} active_slot={}", m["fainted"], m["active_slot"]))
                    .collect::<Vec<_>>()
                    .join(" | ");
                println!(
                    "    dump {label} request={:?} active={active} mons[{mons}] pending={}",
                    world["requests"][side]["kind"],
                    world["pending"][side]["is_null"],
                );
            }
        }
        if std::env::var("PA3_REQ_DBG").is_ok() {
            for sd in [pa3_engine::state::SideId::P1, pa3_engine::state::SideId::P2] {
                let v = state.observe(sd);
                println!(
                    "    req {:?} kind {:?} slots {:?}",
                    sd,
                    v.request.kind,
                    v.request
                        .slots
                        .iter()
                        .map(|s| (
                            s.present,
                            s.can_mega,
                            s.moves
                                .iter()
                                .map(|m| (m.id, m.pp, m.disabled))
                                .collect::<Vec<_>>()
                        ))
                        .collect::<Vec<_>>()
                );
            }
        }
        for (side, id) in [pa3_engine::state::SideId::P1, pa3_engine::state::SideId::P2]
            .into_iter()
            .enumerate()
        {
            let side_view = state.observe(id);
            let mons: Vec<String> = side_view
                .own
                .pokemon
                .iter()
                .enumerate()
                .map(|(roster, p)| {
                    format!(
                        "{roster}:hp{}st{}fa{}{}",
                        p.hp,
                        p.status,
                        u8::from(p.fainted),
                        p.active_slot.map(|s| format!("/a{s}")).unwrap_or_default()
                    )
                })
                .collect();
            println!("  side {side} [{}]", mons.join(" "));
        }
        if let Some(trace) = state.export_trace() {
            for side in [pa3_engine::state::SideId::P1, pa3_engine::state::SideId::P2] {
                let limit: usize = std::env::var("PA3_EVENT_LIMIT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(12);
                for event in trace.events[side.index()].iter().rev().take(limit).rev() {
                    println!(
                        "    [{side:?}] turn {} kind={:?} subject={:?} target={:?} effect={:?} value={}",
                        event.turn,
                        event.event.kind,
                        event.event.subject,
                        event.event.target,
                        event.event.effect,
                        event.event.value
                    );
                }
            }
        }
    }
}

#[path = "../test_support/corpus.rs"]
mod corpus;
