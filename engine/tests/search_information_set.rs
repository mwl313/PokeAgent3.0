//! C2a player-information boundary tests. No test here passes a raw world
//! snapshot, team ID, hidden set, pending command, or RNG into the DTO.

use pa3_engine::{
    assets::Dex,
    search::information_set::{SearchInformationSet, SEARCH_INFORMATION_SET_SCHEMA},
    state::{BattleState, SideId, Team},
};
use std::{path::Path, sync::Arc};

fn assets() -> (Arc<Dex>, Arc<Vec<Team>>) {
    (
        Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap()),
        Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap()),
    )
}

#[test]
fn hidden_opponent_truth_does_not_enter_serialized_information_set() {
    let (dex, teams) = assets();
    let mut hidden = teams[1].clone();
    hidden.id = "hidden-opponent-team-id-must-not-cross-boundary".into();
    for member in &mut hidden.members {
        member.points = [1, 0, 0, 0, 0, 0];
        member.ivs = [0; 6];
        member.nature = dex.id("natures", "serious").unwrap();
    }
    let visible =
        BattleState::reset(&dex, [&teams[0], &teams[1]], [101, 102, 103, 104], [0, 1]).unwrap();
    let alternate =
        BattleState::reset(&dex, [&teams[0], &hidden], [101, 102, 103, 104], [0, 1]).unwrap();
    let visible_view = visible.observe(SideId::P1);
    let alternate_view = alternate.observe(SideId::P1);
    assert_eq!(visible_view, alternate_view);

    let first = SearchInformationSet::from_player_view(&visible_view).unwrap();
    let second = SearchInformationSet::from_player_view(&alternate_view).unwrap();
    assert_eq!(first, second);
    let first_bytes = first.to_bytes().unwrap();
    assert_eq!(first_bytes, second.to_bytes().unwrap());
    assert_eq!(
        SearchInformationSet::from_bytes(&first_bytes).unwrap(),
        first
    );
    assert!(!first_bytes
        .windows(b"hidden-opponent-team-id-must-not-cross-boundary".len())
        .any(|window| window == b"hidden-opponent-team-id-must-not-cross-boundary"));
}

#[test]
fn information_set_schema_is_versioned_and_rejects_unknown_version() {
    let (dex, teams) = assets();
    let state =
        BattleState::reset(&dex, [&teams[0], &teams[1]], [111, 112, 113, 114], [0, 1]).unwrap();
    let info = SearchInformationSet::from_player_view(&state.observe(SideId::P1)).unwrap();
    assert_eq!(info.schema_version, SEARCH_INFORMATION_SET_SCHEMA);
    let mut value: serde_json::Value = serde_json::from_slice(&info.to_bytes().unwrap()).unwrap();
    value["schema_version"] = serde_json::json!(SEARCH_INFORMATION_SET_SCHEMA + 1);
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(SearchInformationSet::from_bytes(&bytes).is_err());
}
