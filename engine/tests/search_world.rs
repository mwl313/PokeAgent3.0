//! C2c-1 clean-preview sampled-world boundary tests.

use pa3_engine::{
    actions::{ActionKind, AtomicAction},
    assets::Dex,
    batch::SideChoice,
    search::{
        belief::{BeliefSampleResult, FrozenOpponentPrior},
        information_set::TrustedSearchInformationSet,
        world::{EpisodeFailureCode, PreviewWorld, WorldFailureCode},
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
fn genuinely_different_hidden_opponent_builds_share_the_same_root_contract() {
    let (dex, teams) = assets();
    let baseline =
        BattleState::reset(&dex, [&teams[0], &teams[1]], [211, 212, 213, 214], [0, 1]).unwrap();
    let mut hidden = teams[1].clone();
    hidden.id = "different-hidden-sheet".into();
    for member in &mut hidden.members {
        member.points = [1, 0, 0, 0, 0, 0];
        member.ivs = [0; 6];
        member.nature = dex.id("natures", "serious").unwrap();
    }
    let altered =
        BattleState::reset(&dex, [&teams[0], &hidden], [211, 212, 213, 214], [0, 1]).unwrap();
    let first =
        TrustedSearchInformationSet::from_player_view(&baseline.observe(SideId::P1)).unwrap();
    let second =
        TrustedSearchInformationSet::from_player_view(&altered.observe(SideId::P1)).unwrap();
    assert_eq!(first.as_info(), second.as_info());
    let prior = FrozenOpponentPrior::from_training_json(
        &dex,
        include_bytes!("../data/training-teams.json"),
    )
    .unwrap();
    let BeliefSampleResult::Samples(samples) =
        prior.sample(first.as_info(), 0xfeed_face, 1).unwrap()
    else {
        panic!("fixture should have support");
    };
    let a = PreviewWorld::construct(&dex, &first, &samples.hypotheses[0], 17)
        .unwrap()
        .into_episode(dex.clone());
    let b = PreviewWorld::construct(&dex, &second, &samples.hypotheses[0], 17)
        .unwrap()
        .into_episode(dex);
    assert_eq!(a.root_view(), b.root_view());
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

#[test]
fn synthetic_episode_advances_preview_and_combat_boundaries() {
    let (dex, trusted, sample) = sample_world(SideId::P1);
    let mut episode = PreviewWorld::construct(&dex, &trusted, &sample, 19)
        .unwrap()
        .into_episode(dex.clone());
    let preview = |side| SideChoice {
        side,
        actions: [0, 1, 4, 5]
            .into_iter()
            .enumerate()
            .map(|(slot, destination)| {
                AtomicAction::select(ActionKind::Pick, slot as u8, destination)
            })
            .collect(),
    };
    let boundary = episode
        .submit_joint(&[preview(SideId::P1), preview(SideId::P2)])
        .unwrap();
    assert_eq!(
        boundary.root_view.request.kind,
        pa3_engine::actions::RequestKind::Normal
    );
    assert_eq!(episode.boundaries(), 1);
    assert!(!episode.root_candidates(&[]).unwrap().is_empty());

    let first = episode.root_candidates(&[]).unwrap()[0];
    let second = episode.root_candidates(&[first]).unwrap()[0];
    let result = episode.submit_joint(&[SideChoice {
        side: SideId::P1,
        actions: vec![first, second],
    }]);
    assert!(
        result.is_ok(),
        "legal combat-prefix submission failed: {result:?}"
    );
    assert_eq!(episode.boundaries(), 2);
}

#[test]
fn invalid_submission_is_nonmutating_and_reports_only_public_error() {
    let (dex, trusted, sample) = sample_world(SideId::P1);
    let mut episode = PreviewWorld::construct(&dex, &trusted, &sample, 23)
        .unwrap()
        .into_episode(dex);
    let before = episode.root_view();
    let error = match episode.submit_joint(&[SideChoice {
        side: SideId::P1,
        actions: vec![AtomicAction::pass(0)],
    }]) {
        Ok(_) => panic!("invalid preview action must fail"),
        Err(error) => error,
    };
    assert!(matches!(error.code, EpisodeFailureCode::InvalidAction));
    assert_eq!(episode.root_view(), before);
    assert_eq!(episode.boundaries(), 0);
}
