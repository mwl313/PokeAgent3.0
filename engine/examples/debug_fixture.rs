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
    let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
    let dex = Dex::load(Path::new(&dir)).unwrap();
    // The ability-interaction corpus is a separate artifact but shares the
    // fixture shape, so the probe accepts it through an explicit flag.
    let source = if std::env::args().any(|a| a == "--ability-corpus") {
        include_str!("../data/ability-interactions.json")
    } else {
        include_str!("../data/turn-fixtures.json")
    };
    let corpus: Corpus = serde_json::from_str(source).unwrap();
    let needle = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .expect("fixture name");
    let limits: usize = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .map(|v| v.parse().unwrap())
        .unwrap_or(usize::MAX);
    let fixture = corpus
        .fixtures
        .iter()
        .find(|f| f.name == needle)
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
                        "{roster}:hp{}fa{}{}",
                        p.hp,
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
