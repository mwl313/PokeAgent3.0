//! Frisk changes player *knowledge*, not world state, so the decision-boundary
//! corpus cannot see it. This focused test pins the player-safe reveal on the
//! reference-generated Frisk scenes: active foe items become known to both
//! players when the holder enters, and nothing is revealed without an item.
use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    knowledge::Known,
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
        .unwrap_or_else(|| panic!("required legal Frisk fixture {name}"))
}

fn start(dex: &Dex, f: &Fixture) -> BattleState {
    for team in &f.teams {
        team.validate_structure(dex).unwrap();
    }
    BattleState::reset(dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap()
}

/// Commit both preview picks: the leads (and their entry abilities) run when
/// the second side submits its team, not during `reset`.
fn leads_in(dex: &Dex, f: &Fixture) -> BattleState {
    let mut state = start(dex, f);
    for step in f.steps.iter().take(2) {
        let result = state.step(dex, step.side, &step.actions).unwrap();
        assert_eq!(result.outcome.operational_error, None);
    }
    state
}

#[test]
fn frisk_announces_every_active_foe_item_to_both_players() {
    let dex = dex();
    let f = fixture("frisk_announces_foe_items_3000");
    let state = leads_in(&dex, &f);
    let leftovers = dex.id("items", "leftovers").unwrap();
    let sitrus = dex.id("items", "sitrusberry").unwrap();
    let frisk = dex.id("abilities", "frisk").unwrap();

    let holder_view = state.observe(SideId::P1);
    // The holder's side learns each active foe's item in field order.
    assert_eq!(holder_view.knowledge.pokemon[6].item, Known::new(leftovers));
    assert_eq!(holder_view.knowledge.pokemon[7].item, Known::new(sitrus));
    // A benched foe's item is not announced.
    assert!(!holder_view.knowledge.pokemon[8].item.known);

    // The reference `-item ... [from] ability: Frisk` message is public, so the
    // opposing player also learns the ability (opponent rosters start at 6 in
    // every viewer's own-then-opponent knowledge layout).
    let foe_view = state.observe(SideId::P2);
    assert_eq!(foe_view.knowledge.pokemon[6].ability, Known::new(frisk));
}

#[test]
fn frisk_announces_only_item_holders() {
    let dex = dex();
    let f = fixture("frisk_skips_itemless_foes_3000");
    let state = leads_in(&dex, &f);
    let leftovers = dex.id("items", "leftovers").unwrap();

    let holder_view = state.observe(SideId::P1);
    assert_eq!(holder_view.knowledge.pokemon[6].item, Known::new(leftovers));
    // The itemless active foe stays unknown.
    assert!(!holder_view.knowledge.pokemon[7].item.known);
}

#[test]
fn frisk_stays_hidden_without_any_foe_item() {
    let dex = dex();
    let f = fixture("frisk_silent_without_items_3000");
    let state = leads_in(&dex, &f);

    let holder_view = state.observe(SideId::P1);
    assert!(!holder_view.knowledge.pokemon[6].item.known);
    let foe_view = state.observe(SideId::P2);
    assert!(!foe_view.knowledge.pokemon[6].ability.known);
}
