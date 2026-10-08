//! Differential witness for Illusion's public-identity surface.
//!
//! The main corpus compares world state, which Illusion deliberately leaves
//! untouched (the real species, stats and damage are the real Pokémon's). The
//! `more_illusion.json` generator records, at every decision boundary, the
//! identity the reference *protocol* displays for each active slot
//! (`switch`/`drag`/`replace` lines). This test replays those fixtures and
//! checks:
//!
//! * the opponent's knowledge carries the displayed (masked) species until the
//!   disguise is broken, and the real species afterwards;
//! * the owner's own knowledge always carries the true species.
use pa3_engine::actions::AtomicAction;
use pa3_engine::assets::Dex;
use pa3_engine::state::{BattleState, SideId, Team};
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
    teams: [Team; 2],
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    side: SideId,
    actions: Vec<AtomicAction>,
    #[serde(default)]
    observed: Option<Observed>,
}

#[derive(Deserialize)]
struct Observed {
    p1: [Option<String>; 2],
    p2: [Option<String>; 2],
}

#[test]
fn illusion_masks_the_observed_identity_until_reveal() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus = serde_json::from_str(include_str!("../data/more_illusion.json")).unwrap();
    let mut checked = 0usize;
    for fixture in corpus.fixtures {
        let mut state = BattleState::reset(
            &dex,
            [&fixture.teams[0], &fixture.teams[1]],
            fixture.seed,
            [0, 1],
        )
        .unwrap();
        for step in &fixture.steps {
            state.step(&dex, step.side, &step.actions).unwrap();
            let Some(observed) = &step.observed else {
                continue;
            };
            for (owner, viewer, displayed) in [
                (SideId::P1, SideId::P2, &observed.p1),
                (SideId::P2, SideId::P1, &observed.p2),
            ] {
                let owner_view = state.observe(owner);
                let viewer_view = state.observe(viewer);
                for (slot, name) in displayed.iter().enumerate() {
                    let Some(name) = name else {
                        continue;
                    };
                    let roster = (0..6usize)
                        .find(|roster| {
                            owner_view.own.pokemon[*roster].active_slot == Some(slot as u8)
                        })
                        .unwrap_or_else(|| panic!("{}: no active roster in slot", fixture.name));
                    let index = roster + if owner == viewer { 0 } else { 6 };
                    let believed = viewer_view.knowledge.pokemon[index].species;
                    // Reference log names are display names ("Zoroark-Hisui");
                    // the dex tables key ids ("zoroarkhisui").
                    let expected: String = name
                        .chars()
                        .filter(|c| c.is_ascii_alphanumeric())
                        .flat_map(char::to_lowercase)
                        .collect();
                    assert_eq!(
                        dex.names["species"][believed as usize], expected,
                        "{} checklist {checked}: displayed identity for {owner:?} slot {slot}",
                        fixture.name
                    );
                    // The owner keeps the true identity in its own knowledge.
                    let truth = owner_view.own.pokemon[roster].species;
                    assert_eq!(
                        owner_view.knowledge.pokemon[roster].species, truth,
                        "{} owner knowledge must stay true",
                        fixture.name
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked >= 20,
        "expected a meaningful identity surface, saw {checked}"
    );
}
