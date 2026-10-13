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
fn hidden_imprison_choice_revises_request_without_advancing_world() {
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
    let world=state.sides.clone();
    let rng=state.rng.seed(); let turn=state.turn;
    let peer=state.observe(SideId::P1);
    let result=state.step(&dex,SideId::P2,&[first,protect]).unwrap();
    assert!(!result.accepted); assert!(result.outcome.operational_error.is_none());
    assert_eq!(serde_json::to_value(world).unwrap(),serde_json::to_value(&state.sides).unwrap());
    assert_eq!(rng,state.rng.seed()); assert_eq!(turn,state.turn);
    assert_eq!(serde_json::to_value(peer).unwrap(),serde_json::to_value(state.observe(SideId::P1)).unwrap());
    assert!(state.requests[1].candidates(&[first]).unwrap().iter().all(|a|a.move_slot!=1));
    let updated=state.snapshot().unwrap();
    assert_eq!(updated,BattleState::restore(&dex,&updated).unwrap().snapshot().unwrap());
    assert!(state.step(&dex,SideId::P2,&[first,protect]).is_err());
    assert_eq!(updated,state.snapshot().unwrap());
}

#[test]
fn imprison_retry_and_struggle_match_emitted_reference_boundaries() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let data:serde_json::Value=serde_json::from_str(include_str!("../../data/imprison-retry-fixtures.json")).unwrap();
    for fixture in data["fixtures"].as_array().unwrap() {
        let teams:[Team;2]=serde_json::from_value(fixture["teams"].clone()).unwrap();
        let mut state=BattleState::reset(&dex,[&teams[0],&teams[1]],serde_json::from_value(fixture["seed"].clone()).unwrap(),[0,1]).unwrap();
        state.enable_trace().unwrap();
        for (index,step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
            let actions:Vec<AtomicAction>=serde_json::from_value(step["actions"].clone()).unwrap();
            let result=state.step(&dex,serde_json::from_value(step["side"].clone()).unwrap(),&actions).unwrap();
            assert_eq!(result.accepted,step["accepted"].as_bool().unwrap(),"{} step {index}",fixture["name"]);
            assert!(result.outcome.operational_error.is_none(),"{:?}",result.outcome);
            assert_eq!(serde_json::to_value(state.rng.seed()).unwrap(),step["rng"],"{} RNG step {index}",fixture["name"]);
            assert_eq!(serde_json::to_value(state.turn).unwrap(),step["turn"]);
            for side in 0..2 {
                for expected in step["world"][side].as_array().unwrap() {
                    let p=&state.sides[side].pokemon[expected["roster"].as_u64().unwrap() as usize];
                    assert_eq!(serde_json::to_value(p.hp).unwrap(),expected["hp"],"HP {} step {index}",fixture["name"]);
                    assert_eq!(serde_json::to_value(p.status).unwrap(),expected["status"]);
                    for m in expected["moves"].as_array().unwrap() {
                        let native=p.moves.iter().find(|n|n.id==m["id"].as_u64().unwrap() as u16).unwrap();
                        assert_eq!(serde_json::to_value(native.pp).unwrap(),m["pp"],"PP {} step {index}",fixture["name"]);
                    }
                }
            }
            let saved=state.snapshot().unwrap();
            assert_eq!(saved,BattleState::restore(&dex,&saved).unwrap().snapshot().unwrap(),"restore {} step {index}",fixture["name"]);
            if index>=2 {
                for side in [SideId::P1,SideId::P2] {
                    let view=state.observe(side);
                    for (slot,expected) in step["masks"][side.index()].as_array().unwrap().iter().enumerate() {
                        let expected=expected.as_array().unwrap();
                        let request=&view.request.slots[slot];
                        // Struggle is a pseudo-move rather than a roster move.
                        if expected.len()==1 && expected[0]["id"]==dex.effects.struggle {
                            assert!(request.moves.iter().all(|m|m.disabled||m.pp==0));
                        } else {
                            for m in expected {
                                let native=request.moves.iter().find(|n|n.id==m["id"].as_u64().unwrap() as u16).unwrap();
                                assert_eq!(native.disabled,m["disabled"].as_bool().unwrap(),"{} step {index} side {:?} slot {slot}",fixture["name"],side);
                            }
                        }
                    }
                }
            }
        }
        // Rejected attempts and request disclosures must survive trace replay.
        let replay=BattleState::replay_trace(&dex,state.export_trace().unwrap()).unwrap();
        assert_eq!(state.snapshot().unwrap(),replay.snapshot().unwrap());
    }
}

#[test]
fn batch_retry_is_reported_in_both_submission_orders_and_preserves_peer() {
    use crate::batch::{BattleBatch,ResetSpec,SideChoice,StepSpec};
    use std::sync::Arc;
    let dex=Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let data:serde_json::Value=serde_json::from_str(include_str!("../../data/imprison-retry-fixtures.json")).unwrap();
    let f=&data["fixtures"][0];
    let teams:Vec<Team>=serde_json::from_value(f["teams"].clone()).unwrap();
    let seed=serde_json::from_value(f["seed"].clone()).unwrap();
    let mut state=BattleState::reset(&dex,[&teams[0],&teams[1]],seed,[0,1]).unwrap();
    let choice=|i:usize|SideChoice{side:serde_json::from_value(f["steps"][i]["side"].clone()).unwrap(),actions:serde_json::from_value(f["steps"][i]["actions"].clone()).unwrap()};
    for i in 0..4 {let c=choice(i);state.step(&dex,c.side,&c.actions).unwrap();}
    let start=state.snapshot().unwrap();
    for order in [[4,5],[5,4]] {
        let mut batch=BattleBatch::new(dex.clone(),Arc::new(teams.clone()),2).unwrap();
        let handle=batch.reset_batch(&[ResetSpec{team_a:0,team_b:1,seed,role_map:[0,1]}]).unwrap()[0];
        batch.restore(handle,&start).unwrap();
        let result=batch.step_batch(&[StepSpec{handle,choices:order.into_iter().map(choice).collect()}]).unwrap();
        assert!(!result[0].accepted);
        assert_eq!(result[0].request_kinds,[RequestKind::Wait,RequestKind::Normal]);
        let saved=batch.snapshot(handle).unwrap();batch.restore(handle,&saved).unwrap();
        let result=batch.step_batch(&[StepSpec{handle,choices:vec![choice(6)]}]).unwrap();
        assert!(result[0].accepted);assert!(result[0].outcome.operational_error.is_none());
        let restored=BattleState::restore(&dex,&batch.snapshot(handle).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(restored.rng.seed()).unwrap(),f["steps"][6]["rng"]);
    }
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
