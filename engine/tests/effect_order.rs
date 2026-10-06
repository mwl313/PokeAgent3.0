use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    state::{BattleState, SideId, Team},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;
fn world(state: &BattleState) -> Value {
    let envelope: Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap()
}
fn order(value: &Value) -> u64 {
    value.as_u64().unwrap_or(0)
}
fn pairs(map: &Value) -> Vec<(u16, u64)> {
    let mut entries: Vec<_> = map
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, s)| (id.parse().unwrap(), order(&s["effect_order"])))
        .collect();
    entries.sort_unstable();
    entries
}
fn expected_pairs(value: &Value) -> Vec<(u16, u64)> {
    serde_json::from_value(value.clone()).unwrap()
}
fn compare(state: &BattleState, expected: &Value, context: &str) {
    let w = world(state);
    assert_eq!(
        w["next_effect_order"], expected["next"],
        "{context} counter"
    );
    for side in 0..2 {
        for roster in 0..6 {
            let actual = &w["sides"][side]["pokemon"][roster];
            let expected = &expected["sides"][side]["pokemon"][roster];
            assert_eq!(
                order(&actual["ability_effect_order"]),
                order(&expected["ability"]),
                "{context} side{side} roster{roster} ability"
            );
            assert_eq!(
                order(&actual["item_effect_order"]),
                order(&expected["item"]),
                "{context} side{side} roster{roster} item"
            );
            assert_eq!(
                order(&actual["status_state"]["effect_order"]),
                order(&expected["status"]),
                "{context} status"
            );
            assert_eq!(
                pairs(&actual["volatiles"]),
                expected_pairs(&expected["volatiles"]),
                "{context} volatiles"
            );
        }
        assert_eq!(
            pairs(&w["sides"][side]["conditions"]),
            expected_pairs(&expected["sides"][side]["conditions"]),
            "{context} sideconditions"
        );
    }
    assert_eq!(
        pairs(&w["field"]),
        expected_pairs(&expected["field"]),
        "{context} field"
    );
}
#[test]
fn global_effect_orders_match_reference_at_every_boundary() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Value = serde_json::from_str(include_str!("effect-order-reference.json")).unwrap();
    assert_eq!(corpus["oracle_commit"], pa3_engine::ORACLE_COMMIT);
    assert_eq!(corpus["format"], pa3_engine::FORMAT);
    for fixture in corpus["fixtures"].as_array().unwrap() {
        let teams: [Team; 2] = serde_json::from_value(fixture["teams"].clone()).unwrap();
        let seed = serde_json::from_value(fixture["seed"].clone()).unwrap();
        let mut state = BattleState::reset(&dex, [&teams[0], &teams[1]], seed, [0, 1]).unwrap();
        compare(
            &state,
            &fixture["initial"],
            fixture["name"].as_str().unwrap(),
        );
        for (i, step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
            let side: SideId = serde_json::from_value(step["side"].clone()).unwrap();
            let actions: Vec<AtomicAction> =
                serde_json::from_value(step["actions"].clone()).unwrap();
            state.step(&dex, side, &actions).unwrap();
            let context = format!("{} boundary{i}", fixture["name"].as_str().unwrap());
            compare(&state, &step["expected"], &context);
            state = BattleState::restore(&dex, &state.snapshot().unwrap())
                .unwrap_or_else(|error| panic!("{context} snapshot restore: {error}"));
        }
    }
}
fn signed(value: &Value) -> Vec<u8> {
    let payload = serde_json::to_string(value).unwrap();
    serde_json::to_vec(
        &json!({"sha256":format!("{:x}",Sha256::digest(payload.as_bytes())),"payload":payload}),
    )
    .unwrap()
}
#[test]
fn private_order_corruption_is_rejected_and_valid_order_permutations_do_not_leak() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Value = serde_json::from_str(include_str!("effect-order-reference.json")).unwrap();
    let fixture = &corpus["fixtures"][0];
    let teams: [Team; 2] = serde_json::from_value(fixture["teams"].clone()).unwrap();
    let mut state = BattleState::reset(
        &dex,
        [&teams[0], &teams[1]],
        serde_json::from_value(fixture["seed"].clone()).unwrap(),
        [0, 1],
    )
    .unwrap();
    for step in fixture["steps"].as_array().unwrap().iter().take(2) {
        state
            .step(
                &dex,
                serde_json::from_value(step["side"].clone()).unwrap(),
                &serde_json::from_value::<Vec<AtomicAction>>(step["actions"].clone()).unwrap(),
            )
            .unwrap();
    }
    let initial = world(&state);
    for (pointer, value) in [
        ("/next_effect_order", json!(0)),
        ("/schema", json!(3)),
        ("/sides/0/pokemon/0/ability_effect_order", json!(u32::MAX)),
    ] {
        let mut invalid = initial.clone();
        *invalid.pointer_mut(pointer).unwrap() = value;
        assert!(
            BattleState::restore(&dex, &signed(&invalid)).is_err(),
            "{pointer}"
        );
    }
    let mut assigned: Vec<String> = Vec::new();
    for side in 0..2 {
        for roster in 0..6 {
            let pointer = format!("/sides/{side}/pokemon/{roster}/ability_effect_order");
            if initial.pointer(&pointer).unwrap().is_number() {
                assigned.push(pointer);
            }
        }
    }
    let mut duplicate = initial.clone();
    *duplicate.pointer_mut(&assigned[1]).unwrap() = initial.pointer(&assigned[0]).unwrap().clone();
    assert!(BattleState::restore(&dex, &signed(&duplicate)).is_err());
    let mut permuted = initial.clone();
    *permuted.pointer_mut(&assigned[0]).unwrap() = initial.pointer(&assigned[1]).unwrap().clone();
    *permuted.pointer_mut(&assigned[1]).unwrap() = initial.pointer(&assigned[0]).unwrap().clone();
    let restored = BattleState::restore(&dex, &signed(&permuted)).unwrap();
    for side in [SideId::P1, SideId::P2] {
        assert_eq!(restored.observe(side), state.observe(side));
    }
}
