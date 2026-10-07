//! Differential and information-boundary verification for Poltergeist.
//!
//! The corpus (`generate_more_poltergeist.mjs`, merged into turn-fixtures.json)
//! pins the move-owned `onTry` item gate, the RNG accounting of a gate failure
//! and the damage of a connecting hit at every decision boundary. This test
//! covers the part the boundary comparison cannot see: the public item reveal
//! of `onTryHit`, which must reach both players exactly when the move connects
//! and never when the gate fails, the target protects or the type immunity
//! applies.
use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    knowledge::{EventKind, Known},
    state::{BattleState, SideId, Team},
};
use serde::Deserialize;
use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

#[derive(Deserialize)]
struct Corpus {
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    name: String,
    seed: [u16; 4],
    teams: [Team; 2],
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    side: SideId,
    actions: Vec<AtomicAction>,
}
fn dex() -> Arc<Dex> {
    static DEX: OnceLock<Arc<Dex>> = OnceLock::new();
    DEX.get_or_init(|| {
        Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap())
    })
    .clone()
}
fn fixture(name: &str) -> Fixture {
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    corpus
        .fixtures
        .into_iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("required legal Poltergeist fixture {name}"))
}
fn start(dex: &Dex, f: &Fixture) -> BattleState {
    for team in &f.teams {
        team.validate_structure(dex).unwrap();
    }
    let mut state = BattleState::reset(dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
    state.enable_trace().unwrap();
    state
}
fn step(dex: &Dex, state: &mut BattleState, s: &Step) {
    let result = state.step(dex, s.side, &s.actions).unwrap();
    assert_eq!(result.outcome.operational_error, None);
}
/// Every reveal of the item held by the opponent's first preview slot.
fn reveals(state: &BattleState, item: u16) -> (usize, usize) {
    let trace = state.export_trace().unwrap();
    let first = trace.events[0]
        .iter()
        .filter(|e| {
            e.event.kind == EventKind::Item && e.event.subject == 6 && e.event.effect == item
        })
        .count();
    let second = trace.events[1]
        .iter()
        .filter(|e| {
            e.event.kind == EventKind::Item && e.event.subject == 0 && e.event.effect == item
        })
        .count();
    (first, second)
}

#[test]
fn poltergeist_hit_publicly_reveals_the_item_to_both_players() {
    let dex = dex();
    let item = dex.id("items", "expertbelt").unwrap();
    let f = fixture("poltergeist_reveals_item_on_hit_101");
    let mut state = start(&dex, &f);
    assert!(!state.observe(SideId::P1).knowledge.pokemon[6].item.known);
    let mut checked = false;
    for s in &f.steps {
        step(&dex, &mut state, s);
        let (first, second) = reveals(&state, item);
        if first > 0 {
            // The reveal is public: both audience streams carry it.
            assert_eq!(second, first, "the reveal must reach both players");
            let p1 = state.observe(SideId::P1);
            assert_eq!(p1.knowledge.pokemon[6].item, Known::new(item));
            let p2 = state.observe(SideId::P2);
            assert_eq!(p2.knowledge.pokemon[0].item, Known::new(item));
            checked = true;
            break;
        }
    }
    assert!(
        checked,
        "a connecting Poltergeist must publicly reveal the target's item"
    );
}

#[test]
fn poltergeist_gate_protection_and_immunity_never_reveal_the_item() {
    let dex = dex();
    let item = dex.id("items", "expertbelt").unwrap();
    for name in [
        "poltergeist_fails_without_item_111",
        "poltergeist_blocked_by_protect_121",
        "poltergeist_type_immunity_keeps_item_hidden_131",
    ] {
        let f = fixture(name);
        let mut state = start(&dex, &f);
        for s in &f.steps {
            step(&dex, &mut state, s);
        }
        let (first, second) = reveals(&state, item);
        assert_eq!(
            (first, second),
            (0, 0),
            "{name} must not reveal an item it never touched"
        );
        assert!(!state.observe(SideId::P1).knowledge.pokemon[6].item.known);
    }
}
