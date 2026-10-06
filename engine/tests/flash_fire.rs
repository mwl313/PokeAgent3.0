use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    knowledge::{EventKind, Known},
    observation::{Encoder, ObservationBuffers},
    state::{BattleState, SideId, Team},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
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
        .unwrap_or_else(|| panic!("required legal Flash Fire fixture {name}"))
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
fn world(state: &BattleState) -> Value {
    let envelope: Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap()
}
fn signed(world: &Value) -> Vec<u8> {
    let payload = serde_json::to_string(world).unwrap();
    serde_json::to_vec(
        &json!({"sha256":format!("{:x}",Sha256::digest(payload.as_bytes())),"payload":payload}),
    )
    .unwrap()
}

#[test]
fn charged_houndoom_mega_event_precedes_old_flash_fire_end_and_new_ability_reveal() {
    let dex = dex();
    let f = fixture("flashfire_mega_loss_1504");
    let mut state = start(&dex, &f);
    let condition = dex.effects.flash_fire;
    let mega = dex.id("species", "houndoommega").unwrap();
    let solar = dex.id("abilities", "solarpower").unwrap();
    let mut checked = false;
    for s in &f.steps {
        step(&dex, &mut state, s);
        let events = &state.export_trace().unwrap().events[0];
        if let Some(mega_index) = events
            .iter()
            .position(|e| e.event.kind == EventKind::Mega && e.event.subject == 0)
        {
            let start_index = events
                .iter()
                .position(|e| {
                    e.event.kind == EventKind::EffectStart
                        && e.event.effect == condition
                        && e.event.subject == 0
                })
                .unwrap();
            let end_index = events
                .iter()
                .position(|e| {
                    e.event.kind == EventKind::EffectEnd
                        && e.event.effect == condition
                        && e.event.subject == 0
                })
                .unwrap();
            let ability_index = events
                .iter()
                .position(|e| {
                    e.event.kind == EventKind::Ability
                        && e.event.subject == 0
                        && e.event.effect == solar
                })
                .unwrap();
            assert!(
                start_index < mega_index && mega_index < end_index && end_index < ability_index
            );
            assert_eq!(events[mega_index].turn, events[end_index].turn);
            let view = state.observe(SideId::P1);
            assert_eq!(view.own.pokemon[0].species, mega);
            assert_eq!(view.own.pokemon[0].ability, solar);
            assert!(!view.knowledge.pokemon[0].effects.contains_key(&condition));
            assert!(
                world(&state)["sides"][0]["pokemon"][0]["volatiles"]
                    .get(condition.to_string())
                    .is_none()
            );
            assert_eq!(
                BattleState::replay_trace(&dex, state.export_trace().unwrap()).unwrap(),
                state
            );
            checked = true;
            break;
        }
    }
    assert!(
        checked,
        "guarded legal fixture must execute charged Houndoom Mega"
    );
}

#[test]
fn alive_switch_ends_flash_fire_but_zero_hp_faint_does_not_emit_end() {
    let dex = dex();
    let condition = dex.effects.flash_fire;
    let f = fixture("flashfire_switch_1503");
    let mut state = start(&dex, &f);
    let mut checked = false;
    for s in &f.steps {
        step(&dex, &mut state, s);
        let events = &state.export_trace().unwrap().events[0];
        if let Some(end_index) = events.iter().position(|e| {
            e.event.kind == EventKind::EffectEnd
                && e.event.effect == condition
                && e.event.subject == 0
        }) {
            assert!(
                events[..end_index]
                    .iter()
                    .any(|e| e.event.kind == EventKind::EffectStart
                        && e.event.effect == condition
                        && e.event.subject == 0)
            );
            assert!(
                events[end_index + 1..]
                    .iter()
                    .any(|e| e.event.kind == EventKind::Switch && e.event.subject != 0)
            );
            let view = state.observe(SideId::P1);
            assert!(view.own.pokemon[0].hp > 0 && !view.own.pokemon[0].fainted);
            assert!(view.own.pokemon[0].active_slot.is_none());
            assert!(!view.knowledge.pokemon[0].effects.contains_key(&condition));
            assert_eq!(
                BattleState::restore(&dex, &state.snapshot().unwrap()).unwrap(),
                state
            );
            checked = true;
            break;
        }
    }
    assert!(
        checked,
        "guarded legal fixture must switch out a charged alive holder"
    );
    let f = fixture("flashfire_categories_1500");
    let mut state = start(&dex, &f);
    let mut checked = false;
    for s in &f.steps {
        step(&dex, &mut state, s);
        let events = &state.export_trace().unwrap().events[0];
        if events
            .iter()
            .any(|e| e.event.kind == EventKind::Faint && e.event.subject == 0)
        {
            assert!(events.iter().any(|e| e.event.kind == EventKind::EffectStart
                && e.event.effect == condition
                && e.event.subject == 0));
            assert!(!events.iter().any(|e| e.event.kind == EventKind::EffectEnd
                && e.event.effect == condition
                && e.event.subject == 0));
            let view = state.observe(SideId::P1);
            assert_eq!(view.own.pokemon[0].hp, 0);
            assert!(view.own.pokemon[0].fainted);
            assert!(!view.knowledge.pokemon[0].effects.contains_key(&condition));
            assert_eq!(
                BattleState::restore(&dex, &state.snapshot().unwrap()).unwrap(),
                state
            );
            assert_eq!(
                BattleState::replay_trace(&dex, state.export_trace().unwrap()).unwrap(),
                state
            );
            checked = true;
            break;
        }
    }
    assert!(
        checked,
        "guarded legal fixture must faint a charged holder at zero HP"
    );
}

#[test]
fn active_flash_fire_snapshot_preserves_public_source_and_rejects_resigned_invalid_shape() {
    let dex = dex();
    let f = fixture("flashfire_categories_1500");
    let mut state = start(&dex, &f);
    let key = dex.effects.flash_fire.to_string();
    let mut reached = false;
    for s in &f.steps {
        step(&dex, &mut state, s);
        if world(&state)["sides"][0]["pokemon"][0]["volatiles"]
            .get(&key)
            .is_some()
        {
            reached = true;
            break;
        }
    }
    assert!(reached, "guarded legal fixture must create the condition");
    let original = world(&state);
    let effect = &original["sides"][0]["pokemon"][0]["volatiles"][&key];
    assert!(effect["duration"].is_null());
    assert_eq!(effect["values"], json!([]));
    assert!(effect["effect_order_assigned"].as_bool().unwrap());
    let source_side = if effect["source"][0] == "P1" { 0 } else { 1 };
    let source_roster = effect["source"][1].as_u64().unwrap() as u8;
    let encoder = Encoder::new(&dex).unwrap();
    for viewer in [SideId::P1, SideId::P2] {
        let view = state.observe(viewer);
        let holder = if viewer == SideId::P1 { 0 } else { 6 };
        let source = source_roster + if viewer.index() == source_side { 0 } else { 6 };
        assert_eq!(
            view.knowledge.pokemon[holder].effects[&dex.effects.flash_fire].source,
            Known::new(source)
        );
        assert_eq!(
            view.knowledge.pokemon[holder].ability,
            Known::new(dex.id("abilities", "flashfire").unwrap())
        );
        let mut encoded = ObservationBuffers::default();
        encoder.encode_into(&view, &mut encoded).unwrap();
        let range = encoded.effect_ranges[4 + holder];
        let entry = encoded.effects[range.start..range.start + range.len]
            .iter()
            .find(|e| e.id == dex.effects.flash_fire)
            .unwrap();
        assert!(entry.source_known);
        assert_eq!(entry.source, source);
        assert!(!entry.duration_known && !entry.stacks_known);
        let event = state.export_trace().unwrap().events[viewer.index()]
            .iter()
            .find(|e| {
                e.event.kind == EventKind::EffectStart
                    && e.event.subject == holder as u8
                    && e.event.effect == dex.effects.flash_fire
            })
            .unwrap();
        assert_eq!(event.event.target, Some(source));
    }
    assert_eq!(
        BattleState::restore(&dex, &state.snapshot().unwrap()).unwrap(),
        state
    );
    for (field, value) in [
        ("duration", json!(1)),
        ("values", json!([1])),
        ("source", Value::Null),
        ("source", json!(["P1", 6])),
        ("effect_order_assigned", json!(false)),
        ("effect_order", original["next_effect_order"].clone()),
    ] {
        let mut invalid = original.clone();
        invalid["sides"][0]["pokemon"][0]["volatiles"][&key][field] = value;
        assert!(
            BattleState::restore(&dex, &signed(&invalid)).is_err(),
            "invalid Flash Fire {field}"
        );
    }
    let mut altered = original;
    altered["next_effect_order"] = json!(altered["next_effect_order"].as_u64().unwrap() + 10);
    let alternate = BattleState::restore(&dex, &signed(&altered)).unwrap();
    for viewer in [SideId::P1, SideId::P2] {
        assert_eq!(alternate.observe(viewer), state.observe(viewer));
    }
}
