//! Primitive-level public type witnesses; full move/oracle coverage is separate.
use super::*;
use crate::state::Team;
use std::path::Path;

fn setup() -> (Dex, BattleState, Entity) {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let teams: Vec<Team> = serde_json::from_str(include_str!("../../data/a10-training-teams.json")).unwrap();
    let state = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    (dex, state, Entity { side: 0, roster: 0 })
}

#[test]
fn public_set_type_updates_both_viewers_without_revealing_identity() {
    let (dex, mut state, actor) = setup();
    let water = dex.id("types", "water").unwrap();
    // A type change is publicly announced even while identity is disguised.
    state.mon_mut(actor).illusion = Some(5);
    state.mask_illusion_knowledge(&dex, actor).unwrap();
    let species = state.knowledge[1].pokemon[6].species;
    let ability = state.knowledge[1].pokemon[6].ability;
    let item = state.knowledge[1].pokemon[6].item;
    let before = state.rng.clone();
    state.mon_mut(actor).types = vec![dex.id("types", "fire").unwrap()];
    assert!(state.set_type(&dex, actor, &[water]).unwrap());
    assert_eq!(state.mon(actor).types, vec![water]);
    assert_eq!(state.knowledge[0].pokemon[0].types, vec![water]);
    assert_eq!(state.knowledge[1].pokemon[6].types, vec![water]);
    assert_eq!(state.knowledge[1].pokemon[6].species, species);
    assert_eq!(state.knowledge[1].pokemon[6].ability, ability);
    assert_eq!(state.knowledge[1].pokemon[6].item, item);
    assert_eq!(format!("{before:?}"), format!("{:?}", state.rng));
    assert!(!state.set_type(&dex, actor, &[water]).unwrap());
}

#[test]
fn unsupported_empty_type_change_is_atomic() {
    let (dex, mut state, actor) = setup();
    let before = state.snapshot().unwrap();
    assert!(state.set_type(&dex, actor, &[]).is_err());
    assert_eq!(state.snapshot().unwrap(), before);
}

#[test]
fn imprison_hidden_disable_cannot_change_served_last_active_observation() {
    let (dex, mut state, _) = setup();
    for side in [SideId::P1, SideId::P2] {
        let mut prefix = Vec::new();
        for _ in 0..4 {
            prefix.push(state.observe(side).request.candidates(&prefix).unwrap()[0]);
        }
        state.step(&dex, side, &prefix).unwrap();
    }
    let slot = 1usize; // Reference restrictData=true for the last active slot.
    let roster = state.sides[0].active[slot].unwrap() as usize;
    assert!(state.requests[0].slots[slot].last_active);
    let mut hidden = state.clone();
    hidden.sides[0].pokemon[roster].moves[0].disabled = true;
    hidden.sides[0].pokemon[roster].moves[0].hidden = true;
    hidden.requests[0].slots[slot].moves[0].disabled = true;
    hidden.requests[0].slots[slot].moves[0].hidden = true;
    assert_eq!(state.observe(SideId::P1).request.candidates(&[]).unwrap(),
        hidden.observe(SideId::P1).request.candidates(&[]).unwrap());
    let encoder = crate::observation::Encoder::new(&dex).unwrap();
    let mut left = crate::observation::ObservationBuffers::default();
    let mut right = crate::observation::ObservationBuffers::default();
    encoder.encode_into(&state.observe(SideId::P1), &mut left).unwrap();
    encoder.encode_into(&hidden.observe(SideId::P1), &mut right).unwrap();
    assert_eq!(left, right, "same public request and legal actions must have identical input");
    assert_eq!(state.observe(SideId::P1).request, hidden.observe(SideId::P1).request);
    // Public restrictions must remain visible: only hidden+restrictData gets
    // normalized away. Neither raw battle state nor execution legality changes.
    hidden.sides[0].pokemon[roster].moves[0].hidden = false;
    hidden.requests[0].slots[slot].moves[0].hidden = false;
    let visible = hidden.observe(SideId::P1);
    assert!(visible.own.pokemon[roster].moves[0].disabled);
    assert!(visible.request.slots[slot].moves[0].disabled);
    assert!(hidden.sides[0].pokemon[roster].moves[0].disabled);
}

#[test]
fn stale_bench_disabled_does_not_become_a_current_public_restriction() {
    let (dex, mut state, _) = setup();
    state.sides[0].pokemon[5].moves[0].disabled = true;
    state.sides[0].pokemon[5].moves[0].hidden = true;
    let public = state.observe(SideId::P1);
    assert!(!public.own.pokemon[5].moves[0].disabled);
    assert!(!public.own.pokemon[5].moves[0].hidden);
    assert!(state.sides[0].pokemon[5].moves[0].disabled);
    let _ = dex;
}

#[test]
fn public_type_reversion_is_visible_on_reentry() {
    let (dex, mut state, _) = setup();
    for side in [SideId::P1, SideId::P2] {
        let mut prefix = Vec::new();
        for _ in 0..4 {
            prefix.push(state.observe(side).request.candidates(&prefix).unwrap()[0]);
        }
        state.step(&dex, side, &prefix).unwrap();
    }
    let actor = Entity { side: 0, roster: state.sides[0].active[0].unwrap() };
    let original = state.mon(actor).types.clone();
    let changed = if original == vec![dex.effects.water] { dex.effects.fire } else { dex.effects.water };
    state.set_type(&dex, actor, &[changed]).unwrap();
    let reserve = state.sides[0].positions[2];
    state.switch_in(&dex, Entity { side: 0, roster: reserve }, 0).unwrap();
    state.switch_in(&dex, actor, 0).unwrap();
    assert_eq!(state.mon(actor).types, original);
    assert_eq!(state.knowledge[0].pokemon[actor.roster as usize].types, original);
    assert_eq!(state.knowledge[1].pokemon[actor.roster as usize + 6].types, original);
}

#[test]
fn hidden_imprison_choice_cannot_commit_without_reference_rerequest_protocol() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let body:serde_json::Value=serde_json::from_str(include_str!("../../data/more_interactions.json")).unwrap();
    let fixture=body["fixtures"].as_array().unwrap().iter().find(|f|f["name"]=="imprison_shared_pool").unwrap();
    let teams:[Team;2]=serde_json::from_value(fixture["teams"].clone()).unwrap();
    let mut state=BattleState::reset(&dex,[&teams[0],&teams[1]],serde_json::from_value(fixture["seed"].clone()).unwrap(),[0,1]).unwrap();
    for step in fixture["steps"].as_array().unwrap().iter().take(4) {
        state.step(&dex,serde_json::from_value(step["side"].clone()).unwrap(),
            &serde_json::from_value::<Vec<AtomicAction>>(step["actions"].clone()).unwrap()).unwrap();
    }
    let request=state.observe(SideId::P2).request;
    let first=*request.candidates(&[]).unwrap().iter().find(|a|a.kind==ActionKind::Move && a.move_slot==255).unwrap();
    let protect=*request.candidates(&[first]).unwrap().iter().find(|a|a.kind==ActionKind::Move && a.move_slot==1).unwrap();
    let before=state.snapshot().unwrap();
    assert!(matches!(state.step(&dex,SideId::P2,&[first,protect]),Err(EngineError::Unsupported(_))));
    assert_eq!(before,state.snapshot().unwrap());
}

#[test]
fn replacement_view_has_no_inherited_move_mask() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let body:serde_json::Value=serde_json::from_str(include_str!("../../data/more_witness_tail.json")).unwrap();
    let fixture=body["fixtures"].as_array().unwrap().iter().find(|f|f["name"]=="witness_emergencyexit_switches_at_half_hp_8500").unwrap();
    let teams:[Team;2]=serde_json::from_value(fixture["teams"].clone()).unwrap();
    let mut state=BattleState::reset(&dex,[&teams[0],&teams[1]],serde_json::from_value(fixture["seed"].clone()).unwrap(),[0,1]).unwrap();
    for step in fixture["steps"].as_array().unwrap().iter().take(8) {
        state.step(&dex,serde_json::from_value(step["side"].clone()).unwrap(),
            &serde_json::from_value::<Vec<AtomicAction>>(step["actions"].clone()).unwrap()).unwrap();
    }
    let before=state.snapshot().unwrap();
    let request=state.observe(SideId::P1).request;
    assert_eq!(request.kind,RequestKind::Replacement);
    assert!(request.slots.iter().all(|slot|slot.moves.is_empty()));
    assert_eq!(before,state.snapshot().unwrap());
}
