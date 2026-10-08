use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    state::{BattleState, Team},
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
        serde_json::from_str(&corpus::corpus_json(&corpus::data_dir())).unwrap(),
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
fn genuine_phases_restore_and_inconsistent_phases_are_rejected() {
    let (dex, corpus) = resources();
    let mut saw = std::collections::BTreeSet::new();
    for fixture in corpus["fixtures"].as_array().unwrap() {
        let mut state = initial(&dex, fixture);
        for step in fixture["steps"].as_array().unwrap() {
            let w = world(&state);
            assert!(
                BattleState::restore(&dex, &signed(&w)).is_ok(),
                "{}",
                fixture["name"]
            );
            if w["pending"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| !p.is_null())
            {
                saw.insert("Pending".into());
                let mut bad = w.clone();
                bad["outcome"]["operational_error"] = json!("injected");
                assert!(BattleState::restore(&dex, &signed(&bad)).is_err());
            }
            let kind = w["requests"][0]["kind"].as_str().unwrap();
            if saw.insert(kind.to_owned()) {
                replace_and_reject(&dex, &w, "/outcome/winner", json!("P1"));
                replace_and_reject(&dex, &w, "/requests/0/kind", json!("Finished"));
                if kind == "Normal" {
                    replace_and_reject(&dex, &w, "/requests/0/slots/0/moves/0/pp", json!(255));
                    let mut bad = w.clone();
                    bad["requests"][0]["bench"] = json!([5]);
                    assert!(BattleState::restore(&dex, &signed(&bad)).is_err());
                }
            }
            apply(&dex, &mut state, step);
            let w = world(&state);
            assert!(
                BattleState::restore(&dex, &signed(&w)).is_ok(),
                "{}",
                fixture["name"]
            );
            if w["outcome"]["terminated"] == true {
                saw.insert("Terminal".into());
                replace_and_reject(&dex, &w, "/requests/0/kind", json!("Normal"));
                replace_and_reject(&dex, &w, "/outcome/reason", json!(null));
            }
        }
        if ["Preview", "Normal", "Replacement", "Terminal", "Pending"]
            .iter()
            .all(|k| saw.contains(*k))
        {
            break;
        }
    }
    for phase in ["Preview", "Normal", "Replacement", "Terminal", "Pending"] {
        assert!(saw.contains(phase), "missing real phase {phase}");
    }
}

#[path = "../test_support/corpus.rs"]
mod corpus;
