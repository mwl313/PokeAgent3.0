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
fn signed(world: &Value) -> Vec<u8> {
    let payload = serde_json::to_string(world).unwrap();
    serde_json::to_vec(
        &json!({"payload":payload,"sha256":format!("{:x}",Sha256::digest(payload.as_bytes()))}),
    )
    .unwrap()
}
fn resources() -> (Dex, Value) {
    (
        Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap(),
        serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap(),
    )
}
fn initial(dex: &Dex, fixture: &Value) -> BattleState {
    let teams: [Team; 2] = serde_json::from_value(fixture["teams"].clone()).unwrap();
    BattleState::reset(
        dex,
        [&teams[0], &teams[1]],
        serde_json::from_value(fixture["seed"].clone()).unwrap(),
        [0, 1],
    )
    .unwrap()
}
fn apply(dex: &Dex, state: &mut BattleState, step: &Value) {
    let actions: Vec<AtomicAction> = serde_json::from_value(step["actions"].clone()).unwrap();
    let result = state
        .step(
            dex,
            serde_json::from_value(step["side"].clone()).unwrap(),
            &actions,
        )
        .unwrap();
    assert!(result.outcome.operational_error.is_none());
}
fn replace_and_reject(dex: &Dex, original: &Value, pointer: &str, value: Value) {
    let mut bad = original.clone();
    *bad.pointer_mut(pointer).unwrap() = value;
    assert!(
        BattleState::restore(dex, &signed(&bad)).is_err(),
        "accepted {pointer}"
    );
}
#[test]
fn real_status_and_volatile_boundaries_restore_but_counter_and_shape_corruption_do_not() {
    let (dex, corpus) = resources();
    let mut tested = std::collections::BTreeSet::new();
    for prefix in [
        "protect_stall_",
        "burn_paralysis_boost_heal_",
        "toxic_poison_user_",
        "sleep_boost_heal_",
        "freeze_flinch_",
    ] {
        let fixture = corpus["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| {
                f["name"].as_str().unwrap().starts_with(prefix)
                    && (prefix != "freeze_flinch_"
                        || f["steps"].as_array().unwrap().iter().any(|step| {
                            step["expected"]["sides"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .any(|side| {
                                    side["pokemon"].as_array().unwrap().iter().any(|mon| {
                                        mon["status"].as_u64()
                                            == Some(u64::from(dex.effects.freeze))
                                    })
                                })
                        }))
            })
            .unwrap();
        let mut state = initial(&dex, fixture);
        assert!(BattleState::restore(&dex, &state.snapshot().unwrap()).is_ok());
        for step in fixture["steps"].as_array().unwrap() {
            apply(&dex, &mut state, step);
            let w = world(&state);
            assert!(BattleState::restore(&dex, &signed(&w)).is_ok());
            for side in 0..2 {
                for roster in 0..6 {
                    let mon = &w["sides"][side]["pokemon"][roster];
                    let status = mon["status_state"]["id"].as_u64().unwrap() as u16;
                    let base = format!("/sides/{side}/pokemon/{roster}");
                    if [dex.effects.sleep, dex.effects.freeze, dex.effects.toxic].contains(&status)
                        && tested.insert(status)
                    {
                        let pointer = format!("{base}/status_state/values");
                        for bad in [
                            json!([]),
                            json!([i64::MIN]),
                            json!([i64::MAX]),
                            json!([1, 1]),
                            json!([16]),
                        ] {
                            replace_and_reject(&dex, &w, &pointer, bad);
                        }
                    }
                    if mon["volatiles"]
                        .as_object()
                        .unwrap()
                        .contains_key(&dex.effects.stall.to_string())
                        && tested.insert(dex.effects.stall)
                    {
                        let pointer = format!("{base}/volatiles/{}/values", dex.effects.stall);
                        for bad in [
                            json!([]),
                            json!([0]),
                            json!([1]),
                            json!([i64::MAX]),
                            json!([4294967296i64]),
                            json!([3, 3]),
                        ] {
                            replace_and_reject(&dex, &w, &pointer, bad);
                        }
                        replace_and_reject(
                            &dex,
                            &w,
                            &format!("{base}/volatiles/{}/duration", dex.effects.stall),
                            json!(null),
                        );
                    }
                }
            }
        }
    }
    for id in [
        dex.effects.sleep,
        dex.effects.freeze,
        dex.effects.toxic,
        dex.effects.stall,
    ] {
        assert!(tested.contains(&id), "missing real shape {id}");
    }
}
#[test]
fn invalid_effect_locations_and_turn_overflow_are_rejected() {
    let (dex, corpus) = resources();
    let fixture = &corpus["fixtures"][0];
    let mut state = initial(&dex, fixture);
    for step in fixture["steps"].as_array().unwrap().iter().take(2) {
        apply(&dex, &mut state, step);
    }
    let w = world(&state);
    replace_and_reject(&dex, &w, "/turn", json!(65535));
    replace_and_reject(&dex, &w, "/turn", json!(1001));
    replace_and_reject(&dex, &w, "/sides/0/pokemon/0/active_turns", json!(65535));
    let condition = json!({"id":dex.effects.burn,"duration":null,"source":null,"effect_order":0,"effect_order_assigned":false,"values":[]});
    for pointer in [
        "/field",
        "/sides/0/conditions",
        "/sides/0/slot_conditions/0",
    ] {
        replace_and_reject(
            &dex,
            &w,
            pointer,
            json!({dex.effects.burn.to_string():condition}),
        );
    }
    let mut bad = w.clone();
    let order = bad["next_effect_order"].as_u64().unwrap();
    bad["next_effect_order"] = json!(order + 1);
    bad["sides"][0]["pokemon"][0]["volatiles"][dex.effects.burn.to_string()] = json!({"id":dex.effects.burn,"duration":null,"source":null,"effect_order":order,"effect_order_assigned":true,"values":[]});
    assert!(BattleState::restore(&dex, &signed(&bad)).is_err());
    // Resume a genuine normal request at the last playable turn and let
    // native residual/end_turn produce the terminal boundary itself.
    let mut final_turn = w.clone();
    final_turn["turn"] = json!(1000);
    let mut ending = BattleState::restore(&dex, &signed(&final_turn)).unwrap();
    for step in fixture["steps"].as_array().unwrap().iter().skip(2).take(2) {
        apply(&dex, &mut ending, step);
    }
    let ended = world(&ending);
    assert_eq!(ended["turn"], json!(1001));
    assert_eq!(ended["outcome"]["reason"], json!("RuleTurnLimit"));
    assert!(BattleState::restore(&dex, &signed(&ended)).is_ok());
}
#[test]
fn exhausted_order_counter_is_operational_and_its_failed_world_remains_restorable() {
    let (dex, corpus) = resources();
    let fixture = &corpus["fixtures"][0];
    let state = initial(&dex, fixture);
    let mut w = world(&state);
    w["next_effect_order"] = json!(u32::MAX);
    let mut state = BattleState::restore(&dex, &signed(&w)).unwrap();
    for step in fixture["steps"].as_array().unwrap().iter().take(2) {
        let actions: Vec<AtomicAction> = serde_json::from_value(step["actions"].clone()).unwrap();
        state
            .step(
                &dex,
                serde_json::from_value::<SideId>(step["side"].clone()).unwrap(),
                &actions,
            )
            .unwrap();
    }
    let outcome = state.observe(SideId::P1).outcome;
    assert!(
        outcome
            .operational_error
            .as_deref()
            .unwrap()
            .contains("order exhausted")
    );
    assert!(!outcome.terminated && !outcome.truncated);
    assert_eq!(outcome.reward(SideId::P1), None);
    let snapshot = state.snapshot().unwrap();
    assert_eq!(
        BattleState::restore(&dex, &snapshot)
            .unwrap()
            .snapshot()
            .unwrap(),
        snapshot
    );
}
