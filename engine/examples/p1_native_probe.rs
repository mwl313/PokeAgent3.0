//! Read-only native boundary/choice witness for the P1 oracle discrepancy.
use pa3_engine::{actions::AtomicAction, assets::Dex, state::{BattleState, SideId, Team}};
use std::path::Path;

fn main() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let args:Vec<String>=std::env::args().collect();
    let path=args.get(1).cloned().unwrap_or_else(||concat!(env!("CARGO_MANIFEST_DIR"),"/data/more_interactions.json").into());
    let fixture=args.get(2).map(String::as_str).unwrap_or("imprison_shared_pool");
    let count=args.get(3).map(|x|x.parse().unwrap()).unwrap_or(4);
    let data: serde_json::Value=serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let f=data["fixtures"].as_array().unwrap().iter().find(|f|f["name"]==fixture).unwrap();
    let teams:[Team;2]=serde_json::from_value(f["teams"].clone()).unwrap();
    let seed=serde_json::from_value(f["seed"].clone()).unwrap();
    let mut state=BattleState::reset(&dex,[&teams[0],&teams[1]],seed,[0,1]).unwrap();
    let mut boundaries=Vec::new();
    for step in f["steps"].as_array().unwrap().iter().take(count) {
        let side:SideId=serde_json::from_value(step["side"].clone()).unwrap();
        let actions:Vec<AtomicAction>=serde_json::from_value(step["actions"].clone()).unwrap();
        state.step(&dex,side,&actions).unwrap();
        let views=[state.observe(SideId::P1),state.observe(SideId::P2)];
        let snapshot:serde_json::Value=serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
        let world:serde_json::Value=serde_json::from_str(snapshot["payload"].as_str().unwrap()).unwrap();
        boundaries.push(serde_json::json!({"views":views,"world":world}));
    }
    if fixture != "imprison_shared_pool" {
        println!("{}",serde_json::json!({"fixture":f["name"],"boundaries":boundaries}));
        return;
    }
    let actions = [AtomicAction {kind:pa3_engine::actions::ActionKind::Move,own_slot:0,move_slot:255,
        target_location:0,switch_destination:255,resource:pa3_engine::actions::Resource::None},
        AtomicAction {kind:pa3_engine::actions::ActionKind::Move,own_slot:1,move_slot:1,
        target_location:0,switch_destination:255,resource:pa3_engine::actions::Resource::None}];
    let candidates=state.observe(SideId::P2).request.candidates(&[actions[0]]).unwrap();
    let result=state.step(&dex,SideId::P2,&actions);
    println!("{}",serde_json::json!({"fixture":f["name"],"boundaries":boundaries,
        "slot1_candidates":candidates,"probe_actions":actions,"probe_result":format!("{result:?}")}));
}
