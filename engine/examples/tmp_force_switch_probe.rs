//! Development probe: replay a force-switch corpus with the family ungated in
//! memory, so the native RNG accounting can be diffed against the reference
//! boundary seeds. Never used by training; delete when the gate opens.
//!
//! Usage: cargo run --release --example tmp_force_switch_probe -- CORPUS FIXTURE [MAX_STEP]
use pa3_engine::{assets::Dex, effects::MoveBehavior, state::BattleState};
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
    command: String,
    expected: Expected,
}
#[derive(Deserialize)]
struct Expected {
    turn: u16,
    rng_seed: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus_path = args.next().expect("corpus path");
    let needle = args.next().expect("fixture name");
    let max_step: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
    let mut dex = Dex::load(Path::new(&dir)).unwrap();
    // Open the gate in memory for the four phazing moves only.
    for (name, behavior) in [
        ("roar", MoveBehavior::Effect),
        ("whirlwind", MoveBehavior::Effect),
        ("dragontail", MoveBehavior::Damage),
        ("circlethrow", MoveBehavior::Damage),
    ] {
        let id = dex.id("moves", name).unwrap();
        dex.effects.moves[id as usize] = behavior;
    }
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(&corpus_path).unwrap()).unwrap();
    let fixture = corpus
        .fixtures
        .iter()
        .find(|f| f.name == needle)
        .expect("unknown fixture");
    let mut state =
        BattleState::reset(&dex, [&fixture.teams[0], &fixture.teams[1]], fixture.seed, [0, 1])
            .unwrap();
    state.enable_trace().unwrap();
    let event_dbg = std::env::var("PA3_EVENT_DBG").is_ok();
    let mut seen_events = [0usize, 0usize];
    println!(
        "fixture {} seed {:?} steps {}",
        fixture.name,
        fixture.seed,
        fixture.steps.len()
    );
    for (index, step) in fixture.steps.iter().enumerate() {
        if index > max_step {
            break;
        }
        let before = state.rng_draws();
        match state.step(&dex, step.side, &step.actions) {
            Ok(result) => {
                let seed = state
                    .rng_seed()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let ok = seed == step.expected.rng_seed;
                println!(
                    "step {index:2} {:?} `{}` turn {} draws +{} = {} seed {seed}{}",
                    step.side,
                    step.command,
                    step.expected.turn,
                    state.rng_draws() - before,
                    state.rng_draws(),
                    if ok { "" } else { "  <-- MISMATCH" }
                );
                if !ok {
                    println!("     reference seed {}", step.expected.rng_seed);
                    if result.outcome.operational_error.is_some() {
                        println!("     operational error: {:?}", result.outcome.operational_error);
                    }
                }
                if state.observe(pa3_engine::state::SideId::P1).outcome.terminated {
                    println!("ended at step {index}");
                    break;
                }
                if event_dbg {
                    let trace = state.export_trace().unwrap();
                    for audience in 0..2 {
                        let events = &trace.events[audience];
                        for event in &events[seen_events[audience]..] {
                            println!("    ev{audience} {:?}", event);
                        }
                        seen_events[audience] = events.len();
                    }
                }
                if std::env::var("PA3_REQ_DBG").is_ok() {
                    for side in [
                        pa3_engine::state::SideId::P1,
                        pa3_engine::state::SideId::P2,
                    ] {
                        let view = state.observe(side);
                        println!(
                            "    req {side:?} {:?} slots {:?} bench {:?}",
                            view.request.kind,
                            view.request
                                .slots
                                .iter()
                                .map(|s| (s.present, s.requires_replacement))
                                .collect::<Vec<_>>(),
                            view.request.bench
                        );
                    }
                    if std::env::var("PA3_STATE_DBG").is_ok() {
                        for side in [
                            pa3_engine::state::SideId::P1,
                            pa3_engine::state::SideId::P2,
                        ] {
                            let view = state.observe(side);
                            let mons: Vec<(u16, u16, bool, Option<u8>)> = view
                                .own
                                .pokemon
                                .iter()
                                .map(|p| (p.species, p.hp, p.fainted, p.active_slot))
                                .collect();
                            println!("    mons {side:?} {mons:?}");
                        }
                    }
                }
            }
            Err(error) => {
                println!("step {index} ERROR {error:?}");
                break;
            }
        }
    }
}
