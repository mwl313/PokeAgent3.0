//! C1 known-world branch tests. These are engine correctness tests, not a
//! player-information-safe planner or a search-teacher evaluation.

use pa3_engine::{
    actions::ActionKind,
    assets::Dex,
    batch::{BattleBatch, ResetSpec},
    search::test_oracle::KnownWorldTestOracle,
    state::Team,
};
use std::{path::Path, sync::Arc};

fn oracle(seed: [u16; 4]) -> KnownWorldTestOracle {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let teams: Arc<Vec<Team>> = Arc::new(
        serde_json::from_str(include_str!("../data/training-teams.json")).unwrap(),
    );
    let mut batch = BattleBatch::new(dex, teams, 1).unwrap();
    let handle = batch
        .reset_batch(&[ResetSpec {
            team_a: 0,
            team_b: 1,
            seed,
            role_map: [0, 1],
        }])
        .unwrap()[0];
    KnownWorldTestOracle::new(batch, handle).unwrap()
}

#[test]
fn complete_preview_joint_actions_are_legal_for_both_sides() {
    let mut oracle = oracle([11, 12, 13, 14]);
    for side in [pa3_engine::state::SideId::P1, pa3_engine::state::SideId::P2] {
        let joints = oracle.legal_joint_actions(side).unwrap();
        assert_eq!(joints.len(), 360, "six choose four ordered preview leaves");
        assert!(joints.iter().all(|joint| {
            joint.len() == 4
                && joint.iter().enumerate().all(|(slot, action)| {
                    action.kind == ActionKind::Pick
                        && action.own_slot == slot as u8
                        && joint[..slot]
                            .iter()
                            .all(|prior| prior.switch_destination != action.switch_destination)
                })
        }));
    }
}

#[test]
fn simultaneous_joint_branch_restores_deterministically() {
    let mut oracle = oracle([21, 22, 23, 24]);
    let p1 = oracle.legal_joint_actions(pa3_engine::state::SideId::P1).unwrap()[0].clone();
    let p2 = oracle.legal_joint_actions(pa3_engine::state::SideId::P2).unwrap()[17].clone();
    let baseline = oracle.baseline_snapshot().to_vec();

    let first = oracle.branch(p1.clone(), p2.clone()).unwrap();
    assert_eq!(oracle.snapshot().unwrap(), baseline);
    let second = oracle.branch(p1, p2).unwrap();

    assert_eq!(first, second, "snapshot restores RNG and full world state");
    assert_eq!(oracle.snapshot().unwrap(), baseline);
}

#[test]
fn branches_are_isolated_and_incomplete_actions_never_mutate_baseline() {
    let mut oracle = oracle([31, 32, 33, 34]);
    let p1 = oracle.legal_joint_actions(pa3_engine::state::SideId::P1).unwrap()[1].clone();
    let p2_a = oracle.legal_joint_actions(pa3_engine::state::SideId::P2).unwrap()[2].clone();
    let p2_b = oracle.legal_joint_actions(pa3_engine::state::SideId::P2).unwrap()[3].clone();
    let baseline = oracle.baseline_snapshot().to_vec();

    let a = oracle.branch(p1.clone(), p2_a).unwrap();
    let b = oracle.branch(p1.clone(), p2_b.clone()).unwrap();
    let p2_a_again = oracle.legal_joint_actions(pa3_engine::state::SideId::P2).unwrap()[2].clone();
    let a_again = oracle.branch(p1, p2_a_again).unwrap();
    assert_eq!(a, a_again, "another branch cannot contaminate a prior branch");
    assert_eq!(oracle.snapshot().unwrap(), baseline);

    let mut incomplete = p2_b;
    incomplete.pop();
    let p1_again = oracle.legal_joint_actions(pa3_engine::state::SideId::P1).unwrap()[1].clone();
    assert!(oracle
        .branch(p1_again, incomplete)
        .is_err());
    assert_eq!(oracle.snapshot().unwrap(), baseline);
    assert_ne!(a.snapshot, b.snapshot, "distinct complete joint branches are observable in the engine");
}
