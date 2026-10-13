//! Opponent typing is reconstructed from public reference logs, never World.
use pa3_engine::{assets::Dex,state::{BattleState,SideId,Team},actions::AtomicAction};
use std::path::Path;

#[test]
fn dynamic_types_and_identity_match_public_showdown_messages() {
    let dex=Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"),"/data"))).unwrap();
    let corpus:serde_json::Value=serde_json::from_str(include_str!("../data/public-type-oracle.json")).unwrap();
    let mut boundaries=0;
    for f in corpus["fixtures"].as_array().unwrap() {
        let teams:[Team;2]=serde_json::from_value(f["teams"].clone()).unwrap();
        let mut state=BattleState::reset(&dex,[&teams[0],&teams[1]],serde_json::from_value(f["seed"].clone()).unwrap(),[0,1]).unwrap();
        for (step_no,step) in f["steps"].as_array().unwrap().iter().enumerate() {
            let side:SideId=serde_json::from_value(step["side"].clone()).unwrap();
            let actions:Vec<AtomicAction>=serde_json::from_value(step["actions"].clone()).unwrap();
            let result=state.step(&dex,side,&actions).unwrap();
            assert!(result.accepted && result.outcome.operational_error.is_none());
            for owner in [SideId::P1,SideId::P2] {
                let own=state.observe(owner);let opponent=state.observe(if owner==SideId::P1{SideId::P2}else{SideId::P1});
                for slot in 0..2 {
                    let actual=&step["own"][owner.index()][slot];
                    if actual.is_null(){continue;}
                    let roster=actual["roster"].as_u64().unwrap() as usize;
                    let expected:Vec<u16>=serde_json::from_value(actual["types"].clone()).unwrap();
                    assert_eq!(own.own.pokemon[roster].types,expected,"{} step {step_no} owner {:?} slot {slot}",f["name"],owner);
                    let public=&step["observed"][owner.index()][slot];
                    if public.is_null(){continue;}
                    let expected:Vec<u16>=serde_json::from_value(public["types"].clone()).unwrap();
                    assert_eq!(opponent.knowledge.pokemon[6+roster].types,expected,"{} step {step_no} public {:?} slot {slot}",f["name"],owner);
                    let name=public["species"].as_str().unwrap().chars().filter(|c|c.is_ascii_alphanumeric()).flat_map(char::to_lowercase).collect::<String>();
                    assert_eq!(dex.names["species"][opponent.knowledge.pokemon[6+roster].species as usize],name,"{} step {step_no} visible identity",f["name"]);
                }
            }
            boundaries+=1;
        }
    }
    println!("public type oracle: {} fixtures / {boundaries} boundaries",corpus["fixtures"].as_array().unwrap().len());
}
