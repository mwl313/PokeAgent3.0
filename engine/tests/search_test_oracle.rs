//! C1 known-world branch tests. These are engine correctness tests, not a
//! player-information-safe planner or a search-teacher evaluation.

use pa3_engine::{
    actions::{ActionKind, AtomicAction, Resource},
    assets::Dex,
    batch::{BattleBatch, ResetSpec, SideChoice, StepSpec},
    search::test_oracle::KnownWorldTestOracle,
    state::{BattleState, SideId, Team},
};
use std::{path::Path, sync::Arc};

fn oracle(seed: [u16; 4]) -> KnownWorldTestOracle {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let teams: Arc<Vec<Team>> =
        Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap());
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

fn combat_oracle(seed: [u16; 4]) -> (KnownWorldTestOracle, Arc<Dex>) {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let teams: Arc<Vec<Team>> =
        Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap());
    let mut batch = BattleBatch::new(dex.clone(), teams, 1).unwrap();
    let handle = batch
        .reset_batch(&[ResetSpec {
            team_a: 0,
            team_b: 1,
            seed,
            role_map: [0, 1],
        }])
        .unwrap()[0];
    let preview = |side: SideId, roster: [u8; 4]| SideChoice {
        side,
        actions: roster
            .into_iter()
            .enumerate()
            .map(|(slot, destination)| {
                AtomicAction::select(ActionKind::Pick, slot as u8, destination)
            })
            .collect(),
    };
    batch
        .step_batch(&[StepSpec {
            handle,
            choices: vec![
                preview(SideId::P1, [0, 1, 4, 5]),
                preview(SideId::P2, [0, 1, 4, 5]),
            ],
        }])
        .unwrap();
    (KnownWorldTestOracle::new(batch, handle).unwrap(), dex)
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
    let p1 = oracle
        .legal_joint_actions(pa3_engine::state::SideId::P1)
        .unwrap()[0]
        .clone();
    let p2 = oracle
        .legal_joint_actions(pa3_engine::state::SideId::P2)
        .unwrap()[17]
        .clone();
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
    let p1 = oracle
        .legal_joint_actions(pa3_engine::state::SideId::P1)
        .unwrap()[1]
        .clone();
    let p2_a = oracle
        .legal_joint_actions(pa3_engine::state::SideId::P2)
        .unwrap()[2]
        .clone();
    let p2_b = oracle
        .legal_joint_actions(pa3_engine::state::SideId::P2)
        .unwrap()[3]
        .clone();
    let baseline = oracle.baseline_snapshot().to_vec();

    let a = oracle.branch(p1.clone(), p2_a).unwrap();
    let b = oracle.branch(p1.clone(), p2_b.clone()).unwrap();
    let p2_a_again = oracle
        .legal_joint_actions(pa3_engine::state::SideId::P2)
        .unwrap()[2]
        .clone();
    let a_again = oracle.branch(p1, p2_a_again).unwrap();
    assert_eq!(
        a, a_again,
        "another branch cannot contaminate a prior branch"
    );
    assert_eq!(oracle.snapshot().unwrap(), baseline);

    let mut incomplete = p2_b;
    incomplete.pop();
    let p1_again = oracle
        .legal_joint_actions(pa3_engine::state::SideId::P1)
        .unwrap()[1]
        .clone();
    assert!(oracle.branch(p1_again, incomplete).is_err());
    assert_eq!(oracle.snapshot().unwrap(), baseline);
    assert_ne!(
        a.snapshot, b.snapshot,
        "distinct complete joint branches are observable in the engine"
    );
}

#[test]
fn real_combat_turn_has_complete_prefix_dependent_move_targets() {
    let (mut oracle, _) = combat_oracle([41, 42, 43, 44]);
    let p1 = oracle.legal_joint_actions(SideId::P1).unwrap();
    let p2 = oracle.legal_joint_actions(SideId::P2).unwrap();
    assert!(!p1.is_empty() && !p2.is_empty());
    assert!(p1.iter().all(|joint| joint.len() == 2));
    assert!(p2.iter().all(|joint| joint.len() == 2));
    let first_move = p1
        .iter()
        .flat_map(|joint| joint.iter())
        .find(|action| action.kind == ActionKind::Move)
        .copied()
        .expect("combat fixture has a move action");
    let after_prefix = oracle
        .legal_actions_for_prefix(SideId::P1, &[first_move])
        .unwrap();
    assert!(
        !after_prefix.is_empty(),
        "prefix must retain legal completions"
    );
    assert!(after_prefix
        .iter()
        .all(|action| action.own_slot != first_move.own_slot));
    assert!(p1.iter().any(|joint| {
        joint.contains(&first_move)
            && joint
                .iter()
                .any(|action| action.kind == ActionKind::Move && action.target_location != 0)
    }));
    let branch = oracle.branch(p1[0].clone(), p2[0].clone()).unwrap();
    assert!(branch.result.accepted);
    assert_eq!(oracle.snapshot().unwrap(), oracle.baseline_snapshot());
}

#[test]
fn voluntary_switch_changes_both_sides_and_restores_the_baseline() {
    let (mut oracle, dex) = combat_oracle([51, 52, 53, 54]);
    let p1 = oracle.legal_joint_actions(SideId::P1).unwrap();
    let p2 = oracle.legal_joint_actions(SideId::P2).unwrap();
    let p1_switch = p1
        .iter()
        .find(|joint| joint.iter().any(|action| action.kind == ActionKind::Switch))
        .cloned()
        .expect("P1 has a voluntary switch candidate");
    let p2_switch = p2
        .iter()
        .find(|joint| joint.iter().any(|action| action.kind == ActionKind::Switch))
        .cloned()
        .expect("P2 has a voluntary switch candidate");
    let expected_p1 = p1_switch
        .iter()
        .find(|action| action.kind == ActionKind::Switch)
        .unwrap()
        .switch_destination;
    let expected_p2 = p2_switch
        .iter()
        .find(|action| action.kind == ActionKind::Switch)
        .unwrap()
        .switch_destination;
    let baseline = oracle.baseline_snapshot().to_vec();
    let branch = oracle.branch(p1_switch, p2_switch).unwrap();
    let restored = BattleState::restore(&dex, &branch.snapshot).unwrap();
    let p1_active = restored.observe(SideId::P1).own.active;
    let p2_active = restored.observe(SideId::P2).own.active;
    assert!(p1_active.contains(&Some(expected_p1)));
    assert!(p2_active.contains(&Some(expected_p2)));
    assert_eq!(oracle.snapshot().unwrap(), baseline);
}

#[test]
fn combat_branches_reproduce_chance_and_resource_choices_without_cross_contamination() {
    let (mut oracle, _) = combat_oracle([61, 62, 63, 64]);
    let p1 = oracle.legal_joint_actions(SideId::P1).unwrap();
    let p2 = oracle.legal_joint_actions(SideId::P2).unwrap();
    let p1_mega = p1
        .iter()
        .find(|joint| joint.iter().any(|a| a.resource == Resource::Mega));
    if let Some(mega) = p1_mega {
        assert_eq!(
            mega.iter().filter(|a| a.resource == Resource::Mega).count(),
            1
        );
    }
    let a = oracle.branch(p1[0].clone(), p2[0].clone()).unwrap();
    let b = oracle.branch(p1[1].clone(), p2[0].clone()).unwrap();
    let a_again = oracle.branch(p1[0].clone(), p2[0].clone()).unwrap();
    assert_eq!(
        a, a_again,
        "RNG and chance effects reproduce from the same snapshot"
    );
    assert_ne!(
        a.snapshot, b.snapshot,
        "distinct combat actions produce isolated branches"
    );
    assert_eq!(oracle.snapshot().unwrap(), oracle.baseline_snapshot());
}

#[test]
fn malformed_combat_actions_fail_without_corrupting_both_sides() {
    let (mut oracle, _) = combat_oracle([71, 72, 73, 74]);
    let p1 = oracle.legal_joint_actions(SideId::P1).unwrap()[0].clone();
    let p2 = oracle.legal_joint_actions(SideId::P2).unwrap()[0].clone();
    let baseline = oracle.baseline_snapshot().to_vec();
    let mut incomplete = p1;
    incomplete.pop();
    assert!(oracle.branch(incomplete, p2).is_err());
    assert_eq!(oracle.snapshot().unwrap(), baseline);
}
