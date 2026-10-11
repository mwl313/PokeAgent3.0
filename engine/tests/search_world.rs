//! C2c-1 clean-preview sampled-world boundary tests.

use pa3_engine::{
    assets::Dex,
    search::{
        belief::{BeliefSampleResult, FrozenOpponentPrior},
        information_set::TrustedSearchInformationSet,
        world::{PreviewWorld, WorldFailureCode},
    },
    state::{BattleState, SideId, Team},
};
use std::{path::Path, sync::Arc};

fn assets() -> (Arc<Dex>, Arc<Vec<Team>>) {
    (
        Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap()),
        Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap()),
    )
}

fn sample_world(
    side: SideId,
) -> (
    Arc<Dex>,
    TrustedSearchInformationSet,
    pa3_engine::search::belief::BeliefTeam,
) {
    let (dex, teams) = assets();
    let state =
        BattleState::reset(&dex, [&teams[0], &teams[1]], [201, 202, 203, 204], [0, 1]).unwrap();
    let trusted = TrustedSearchInformationSet::from_player_view(&state.observe(side)).unwrap();
    let prior = FrozenOpponentPrior::from_training_json(
        &dex,
        include_bytes!("../data/training-teams.json"),
    )
    .unwrap();
    let BeliefSampleResult::Samples(samples) = prior.sample(trusted.as_info(), 0xabc, 1).unwrap()
    else {
        panic!("preview fixture should have support");
    };
    (dex, trusted, samples.hypotheses[0].clone())
}

#[test]
fn constructs_deterministic_p1_and_p2_preview_worlds() {
    for side in [SideId::P1, SideId::P2] {
        let (dex, trusted, sample) = sample_world(side);
        let first = PreviewWorld::construct(&dex, &trusted, &sample, 0xfeed_face).unwrap();
        let second = PreviewWorld::construct(&dex, &trusted, &sample, 0xfeed_face).unwrap();
        assert_eq!(first.player_view(), second.player_view());
        assert_eq!(first.diagnostics(), second.diagnostics());
        assert!(first.is_clean_preview());
        assert_eq!(first.player_view().side, side);
        assert_eq!(first.player_view().request, trusted.as_info().own.request);
    }
}

#[test]
fn hidden_truth_changes_cannot_change_a_given_sampled_world() {
    let (dex, trusted, sample) = sample_world(SideId::P1);
    let first = PreviewWorld::construct(&dex, &trusted, &sample, 7).unwrap();
    let second = PreviewWorld::construct(&dex, &trusted, &sample, 7).unwrap();
    assert_eq!(first.player_view(), second.player_view());
    assert_eq!(first.diagnostics(), second.diagnostics());
}

#[test]
fn unsupported_unknown_preview_is_rejected_without_fabrication() {
    let (dex, _, sample) = sample_world(SideId::P1);
    let (_, teams) = assets();
    let mut view = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1])
        .unwrap()
        .observe(SideId::P1);
    view.own.active[0] = Some(0);
    let trusted = TrustedSearchInformationSet::from_player_view(&view).unwrap();
    let error = match PreviewWorld::construct(&dex, &trusted, &sample, 1) {
        Ok(_) => panic!("midbattle state must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error.code, WorldFailureCode::UnsupportedState));
    assert_eq!(error.diagnostics.attempts, 0);
}

#[test]
fn unknown_opponent_species_is_not_filled_from_the_sample() {
    let (dex, _, sample) = sample_world(SideId::P1);
    let (_, teams) = assets();
    let mut view = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1])
        .unwrap()
        .observe(SideId::P1);
    view.knowledge.pokemon[6].species = 0;
    let trusted = TrustedSearchInformationSet::from_player_view(&view).unwrap();
    let error = match PreviewWorld::construct(&dex, &trusted, &sample, 1) {
        Ok(_) => panic!("unknown public species must not be invented"),
        Err(error) => error,
    };
    assert!(matches!(error.code, WorldFailureCode::ObservationMismatch));
}

#[test]
fn malformed_sample_is_rejected_before_any_seed_attempt() {
    let (dex, trusted, mut sample) = sample_world(SideId::P1);
    sample.members[0].species = 0;
    let error = match PreviewWorld::construct(&dex, &trusted, &sample, 1) {
        Ok(_) => panic!("illegal sampled sheet must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error.code, WorldFailureCode::InvalidSample));
    assert_eq!(error.diagnostics.attempts, 0);
}
