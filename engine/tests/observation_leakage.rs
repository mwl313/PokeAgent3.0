//! Adversarial observation-leakage gate.
//!
//! Two worlds that are identical from P1's legitimate viewpoint but differ in
//! P2's hidden state must produce byte-identical P1 observations and identical
//! P1 candidate masks. This is asserted on the *numeric encoder output*, not
//! only on the structured view, so a numeric ID that is not legally known
//! cannot slip into the training tensor.
use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    observation::{Encoder, ObservationBuffers},
    state::{BattleState, SideId, Team},
};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Corpus {
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    seed: [u16; 4],
    teams: [Team; 2],
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    side: SideId,
    actions: Vec<AtomicAction>,
}

/// A hidden-world mutation applied to a snapshot payload before restore.
type Mutation<'a> = &'a dyn Fn(&mut serde_json::Value, &Dex);

fn encode(dex: &Dex, state: &BattleState) -> ObservationBuffers {
    let encoder = Encoder::new(dex).unwrap();
    let mut buffers = ObservationBuffers::default();
    encoder
        .encode_into(&state.observe(SideId::P1), &mut buffers)
        .unwrap();
    buffers
}

/// Everything the encoder emits, as one comparable value.
fn content(buffers: &ObservationBuffers) -> String {
    format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
        buffers.tokens,
        buffers.token_mask,
        buffers.effects,
        buffers.effect_ranges,
        buffers.repertoire,
        buffers.repertoire_ranges,
        buffers.types,
        buffers.type_ranges,
        buffers.base_moves,
        buffers.base_move_ranges,
        buffers.move_effects,
        buffers.move_effect_ranges,
    )
}

fn p1_mask(state: &BattleState) -> Vec<AtomicAction> {
    let view = state.observe(SideId::P1);
    view.request.candidates(&[]).unwrap_or_default()
}

#[test]
fn hidden_opponent_state_cannot_change_p1_observation_or_mask() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus =
        serde_json::from_str(&corpus::corpus_json(&corpus::data_dir())).unwrap();
    let fixture = &corpus.fixtures[0];
    let boundary = 6usize.min(fixture.steps.len());

    let replay = |mutate: Option<Mutation>| {
        let mut state =
            BattleState::reset(&dex, [&fixture.teams[0], &fixture.teams[1]], fixture.seed, [0, 1])
                .unwrap();
        for step in fixture.steps.iter().take(boundary) {
            state.step(&dex, step.side, &step.actions).unwrap();
        }
        if let Some(mutate) = mutate {
            use sha2::{Digest, Sha256};
            let envelope: serde_json::Value =
                serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
            let mut world: serde_json::Value =
                serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
            mutate(&mut world, &dex);
            let payload = serde_json::to_string(&world).unwrap();
            // Re-sign the envelope exactly as the engine's own snapshot writer
            // does; the integrity check is content hashing, not authentication,
            // and this test deliberately constructs an alternate hidden world.
            let signed = serde_json::to_vec(&serde_json::json!({
                "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())),
                "payload": payload,
            }))
            .unwrap();
            state = BattleState::restore(&dex, &signed).unwrap();
        }
        let observation = content(&encode(&dex, &state));
        let mask = p1_mask(&state);
        (observation, mask)
    };

    let (baseline, baseline_mask) = replay(None);

    // Hidden moves on a reserve P2 Pokemon (never revealed to P1).
    let hidden_moves = |world: &mut serde_json::Value, dex: &Dex| {
        let species = world["sides"][1]["pokemon"][4]["species"].as_u64().unwrap() as u16;
        let legal = &dex.legal_moves_by_species[species as usize];
        let candidate = legal
            .iter()
            .copied()
            .find(|id| {
                !world["sides"][1]["pokemon"][4]["moves"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|m| m["id"].as_u64() == Some(u64::from(*id)))
            })
            .expect("a distinct legal move");
        world["sides"][1]["pokemon"][4]["moves"][0]["id"] = serde_json::json!(candidate);
        world["sides"][1]["pokemon"][4]["base_moves"][0]["id"] = serde_json::json!(candidate);
    };
    // Hidden held item on a reserve.
    let hidden_item = |world: &mut serde_json::Value, dex: &Dex| {
        let candidate = (2..dex.legal_items.len())
            .find(|id| dex.legal_items[*id])
            .map(|id| id as u16)
            .expect("a legal item");
        world["sides"][1]["pokemon"][5]["item"] = serde_json::json!(candidate);
    };
    // Hidden ability on a reserve.
    let hidden_ability = |world: &mut serde_json::Value, dex: &Dex| {
        // Pick any reserve that has more than one legal ability; species with a
        // single legal ability cannot express this mutation.
        let mut changed = None;
        for roster in 2..6usize {
            let species = world["sides"][1]["pokemon"][roster]["species"].as_u64().unwrap() as u16;
            let legal = &dex.legal_abilities_by_species[species as usize];
            let current = world["sides"][1]["pokemon"][roster]["ability"].as_u64().unwrap() as u16;
            if let Some(candidate) = legal.iter().copied().find(|id| *id != current) {
                changed = Some((roster, candidate));
                break;
            }
        }
        if let Some((roster, candidate)) = changed {
            world["sides"][1]["pokemon"][roster]["ability"] = serde_json::json!(candidate);
        }
    };
    // Hidden stat allocation and nature on a reserve.
    let hidden_stats = |world: &mut serde_json::Value, dex: &Dex| {
        world["sides"][1]["pokemon"][4]["nature"] =
            serde_json::json!(dex.id("natures", "Modest").unwrap());
        world["sides"][1]["pokemon"][4]["stats"][1] = serde_json::json!(77);
        world["sides"][1]["pokemon"][4]["stats"][3] = serde_json::json!(191);
        world["sides"][1]["pokemon"][5]["stats"][1] = serde_json::json!(120);
    };

    for (label, mutate) in [
        ("hidden moves", &hidden_moves as Mutation),
        ("hidden item", &hidden_item),
        ("hidden ability", &hidden_ability),
        ("hidden stats", &hidden_stats),
    ] {
        let (observation, mask) = replay(Some(mutate));
        assert_eq!(
            observation, baseline,
            "{label}: hidden opponent state changed the P1 observation tensor"
        );
        assert_eq!(
            mask, baseline_mask,
            "{label}: hidden opponent state changed the P1 candidate mask"
        );
    }
}

#[path = "../test_support/corpus.rs"]
mod corpus;
