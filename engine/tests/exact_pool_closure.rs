//! Exact-pool closure must be bound to dataset bytes and executable native IDs.
use pa3_engine::{assets::Dex,effects::{MoveBehavior,Item}};
use sha2::{Digest,Sha256};
use std::path::Path;

#[test]
fn exact_frozen_pool_dynamic_identity_closure_has_no_native_blockers() {
    let data=Path::new(concat!(env!("CARGO_MANIFEST_DIR"),"/data"));
    let dex=Dex::load(data).unwrap();
    let scope:serde_json::Value=serde_json::from_slice(&std::fs::read(data.join("scope.json")).unwrap()).unwrap();
    let report:serde_json::Value=serde_json::from_slice(&std::fs::read(data.join("exact-pool-closure-audit.json")).unwrap()).unwrap();
    let bytes=std::fs::read(data.join("a10-training-teams.json")).unwrap();
    assert_eq!(format!("{:x}",Sha256::digest(&bytes)),report["dataset_sha256"].as_str().unwrap());
    assert_eq!(serde_json::from_slice::<Vec<serde_json::Value>>(&bytes).unwrap().len(),1220);
    assert!(report["unreviewed"].as_array().unwrap().is_empty());
    let mut blockers=vec![];
    for (kind,rows) in report["reachable"].as_object().unwrap() {
        for row in rows.as_array().unwrap() {
            let name=row.as_str().unwrap();let id=dex.id(kind,name).unwrap() as usize;
            let ready=match kind.as_str() {
                "moves"=>dex.effects.moves[id]!=MoveBehavior::Unimplemented,
                "abilities"=>dex.effects.abilities[id].is_ported(),
                "items"=>dex.effects.items[id]!=Item::Unimplemented||!dex.effects.mega_stones[id].is_empty(),
                "species"=>dex.legal_starting_species[id]||scope["format_permitted_battle_forms"].as_array().unwrap().iter().any(|f|f.as_str()==Some(name)),
                _=>panic!("unknown closure kind"),
            };
            if !ready {blockers.push(format!("{kind}:{name}"));}
        }
    }
    assert!(blockers.is_empty(),"actual-pool dynamic blockers: {blockers:?}");
    println!("exact pool closure: 1220 teams; 347 moves, 153 abilities, 137 items, 273 species/forms; no declared native blocker");
}
