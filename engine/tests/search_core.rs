//! C3a bounded clean-preview search-core tests.

use pa3_engine::{
    assets::Dex,
    search::{
        belief::FrozenOpponentPrior,
        core::{BoundedSearch, SearchConfig, SearchStatus},
        information_set::TrustedSearchInformationSet,
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

fn info(dex: &Dex, teams: &[Team], side: SideId) -> TrustedSearchInformationSet {
    let state =
        BattleState::reset(dex, [&teams[0], &teams[1]], [301, 302, 303, 304], [0, 1]).unwrap();
    TrustedSearchInformationSet::from_player_view(&state.observe(side)).unwrap()
}

fn config() -> SearchConfig {
    SearchConfig {
        worlds: 2,
        max_root_candidates: 2,
        max_units: 1_000,
        rollout_combat_turns: 1,
    }
}

#[test]
fn deterministic_bounded_search_evaluates_multiple_actions() {
    let (dex, teams) = assets();
    let trusted = info(&dex, &teams, SideId::P1);
    let prior = FrozenOpponentPrior::from_training_json(
        &dex,
        include_bytes!("../data/training-teams.json"),
    )
    .unwrap();
    let first =
        BoundedSearch::search(dex.clone(), &trusted, &prior, &[0x100, 0x200], config()).unwrap();
    let second = BoundedSearch::search(dex, &trusted, &prior, &[0x100, 0x200], config()).unwrap();
    assert_eq!(first, second);
    assert!(matches!(first.status, SearchStatus::TruncatedCandidates));
    assert_eq!(first.scores.len(), 2);
    assert!(first.scores.iter().all(|score| score.evaluated_worlds == 2));
    assert!(first.selected_action.is_none());
    assert!(first.diagnostics.budget.engine_transitions >= 4);
    assert!(first.diagnostics.budget.opponent_action_samples >= 4);
    assert!(first.diagnostics.opening_pair_count >= 2);
    assert!(first.diagnostics.reserve_combination_count >= 1);
    assert!(first.diagnostics.score_variance > 0.0);
    assert!(first.diagnostics.heuristic_differentiated_actions >= 2);
    assert_eq!(first.diagnostics.tied_score_pairs, 0);
    assert!(
        first
            .scores
            .iter()
            .all(|score| (0.0..=1.0).contains(&score.mean_unknown_health_fraction))
    );
}

#[test]
fn hidden_truth_change_preserves_search_output() {
    let (dex, teams) = assets();
    let baseline =
        BattleState::reset(&dex, [&teams[0], &teams[1]], [311, 312, 313, 314], [0, 1]).unwrap();
    let mut hidden = teams[1].clone();
    hidden.id = "different-hidden-truth".into();
    for member in &mut hidden.members {
        member.points = [1, 0, 0, 0, 0, 0];
        member.ivs = [0; 6];
        member.nature = dex.id("natures", "serious").unwrap();
    }
    let altered =
        BattleState::reset(&dex, [&teams[0], &hidden], [311, 312, 313, 314], [0, 1]).unwrap();
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
    let a = BoundedSearch::search(dex.clone(), &first, &prior, &[7, 9], config()).unwrap();
    let b = BoundedSearch::search(dex, &second, &prior, &[7, 9], config()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn p2_root_and_budget_exhaustion_are_structured() {
    let (dex, teams) = assets();
    let trusted = info(&dex, &teams, SideId::P2);
    let prior = FrozenOpponentPrior::from_training_json(
        &dex,
        include_bytes!("../data/training-teams.json"),
    )
    .unwrap();
    let p2 = BoundedSearch::search(dex.clone(), &trusted, &prior, &[17], config()).unwrap();
    assert_eq!(p2.scores.len(), 2);
    let exhausted = BoundedSearch::search(
        dex,
        &trusted,
        &prior,
        &[17],
        SearchConfig {
            worlds: 1,
            max_root_candidates: 2,
            max_units: 1,
            rollout_combat_turns: 0,
        },
    )
    .unwrap();
    assert!(matches!(exhausted.status, SearchStatus::Abstain));
    assert!(exhausted.diagnostics.budget.exhausted);
    assert!(exhausted.selected_action.is_none());
}

#[test]
fn midbattle_root_is_rejected_without_reconstruction() {
    let (dex, teams) = assets();
    let mut view = BattleState::reset(&dex, [&teams[0], &teams[1]], [321, 322, 323, 324], [0, 1])
        .unwrap()
        .observe(SideId::P1);
    view.own.active[0] = Some(0);
    let trusted = TrustedSearchInformationSet::from_player_view(&view).unwrap();
    let prior = FrozenOpponentPrior::from_training_json(
        &dex,
        include_bytes!("../data/training-teams.json"),
    )
    .unwrap();
    let result = BoundedSearch::search(dex, &trusted, &prior, &[1], config()).unwrap();
    assert!(matches!(result.status, SearchStatus::InsufficientCoverage));
    assert_eq!(result.scores.len(), 0);
}
