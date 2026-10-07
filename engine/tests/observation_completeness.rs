//! Observation-completeness gate.
//!
//! The encoder must be sensitive to every legally visible state a player can
//! act on. Each case perturbs exactly one *player-known* fact and requires the
//! encoded tensor to change; a silently dropped known field is a readiness
//! failure that no other test would catch.
//!
//! This is the complement of `observation_leakage.rs`, which proves that
//! *hidden* state cannot change the same tensor.
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

type Mutation<'a> = &'a dyn Fn(&mut serde_json::Value);

fn encode(dex: &Dex, state: &BattleState) -> String {
    let encoder = Encoder::new(dex).unwrap();
    let mut buffers = ObservationBuffers::default();
    encoder
        .encode_into(&state.observe(SideId::P1), &mut buffers)
        .unwrap();
    format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
        buffers.tokens,
        buffers.token_mask,
        buffers.effects,
        buffers.repertoire,
        buffers.types,
        buffers.base_moves,
        buffers.move_effects,
        buffers.effect_ranges,
    )
}

fn replay(fixture: &Fixture, dex: &Dex, boundary: usize, mutate: Option<Mutation>) -> String {
    let mut state =
        BattleState::reset(dex, [&fixture.teams[0], &fixture.teams[1]], fixture.seed, [0, 1])
            .unwrap();
    for step in fixture.steps.iter().take(boundary) {
        state.step(dex, step.side, &step.actions).unwrap();
    }
    if let Some(mutate) = mutate {
        use sha2::{Digest, Sha256};
        let envelope: serde_json::Value =
            serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
        let mut world: serde_json::Value =
            serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
        mutate(&mut world);
        let payload = serde_json::to_string(&world).unwrap();
        let signed = serde_json::to_vec(&serde_json::json!({
            "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())),
            "payload": payload,
        }))
        .unwrap();
        state = BattleState::restore(dex, &signed).unwrap();
    }
    encode(dex, &state)
}

#[test]
fn every_player_known_state_field_reaches_the_observation_tensor() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus =
        serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let fixture = &corpus.fixtures[0];
    let boundary = 6usize.min(fixture.steps.len());
    let baseline = replay(fixture, &dex, boundary, None);

    // Each mutation changes exactly one fact the acting player legitimately
    // knows about its own side.
    let own_boosts: Mutation = &|world| {
        world["sides"][0]["pokemon"][0]["boosts"][0] = serde_json::json!(2);
    };
    let own_pp: Mutation = &|world| {
        let slots = world["sides"][0]["pokemon"][0]["moves"].as_array().unwrap().len();
        if slots > 0 {
            let current = world["sides"][0]["pokemon"][0]["moves"][0]["pp"]
                .as_u64()
                .unwrap_or(0);
            let next = current.saturating_sub(1);
            world["sides"][0]["pokemon"][0]["moves"][0]["pp"] = serde_json::json!(next);
            // The stored request mirrors PP, and restore recomputes the whole
            // request from the world, so the request copy must move with it.
            if let Some(slot) = world["requests"][0]["slots"]
                .as_array_mut()
                .and_then(|slots| slots.get_mut(0))
                && let Some(moves) = slot["moves"].as_array_mut()
                && let Some(first) = moves.get_mut(0)
            {
                first["pp"] = serde_json::json!(next);
            }
        }
    };
    let own_consumed_item: Mutation = &|world| {
        world["sides"][0]["pokemon"][0]["previous_item"] = serde_json::json!(1);
    };
    let own_hp: Mutation = &|world| {
        let mon = &mut world["sides"][0]["pokemon"][0];
        let hp = mon["hp"].as_u64().unwrap_or(0);
        if hp > 1 {
            mon["hp"] = serde_json::json!(hp - 1);
        }
    };
    let own_stats: Mutation = &|world| {
        let mon = &mut world["sides"][0]["pokemon"][0];
        let atk = mon["stats"][1].as_u64().unwrap_or(0);
        mon["stats"][1] = serde_json::json!(atk + 1);
    };
    let no_op: Mutation = &|_| {};

    let cases: [(&str, Mutation, bool); 6] = [
        ("own boosts", own_boosts, true),
        ("own move PP", own_pp, true),
        ("own consumed item", own_consumed_item, true),
        ("own HP", own_hp, true),
        ("own stats", own_stats, true),
        ("no-op control", no_op, false),
    ];
    let mut observed = 0;
    for (label, mutate, should_change) in cases {
        let encoded = replay(fixture, &dex, boundary, Some(mutate));
        let changed = encoded != baseline;
        assert_eq!(
            changed, should_change,
            "{label}: expected the observation tensor to {}",
            if should_change { "change" } else { "stay identical" }
        );
        if should_change {
            observed += 1;
        }
    }
    assert_eq!(observed, 5, "known-state probes must all be observable");
}
