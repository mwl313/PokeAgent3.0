//! C2b bounded prior and belief-sampler tests.

use pa3_engine::{
    assets::Dex,
    knowledge::Known,
    search::{
        belief::{BeliefSampleResult, FrozenOpponentPrior},
        information_set::SearchInformationSet,
    },
    state::{BattleState, SideId, Team},
};
use std::{path::Path, sync::Arc};

const SOURCE_SHA256: &str = "48a050d007bf43e2ba4e0cbea0757bffe1565cfaab9909b7ccb3eb78736c8073";

fn assets() -> (Arc<Dex>, Arc<Vec<Team>>) {
    (
        Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap()),
        Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap()),
    )
}

fn prior() -> (Arc<Dex>, Arc<Vec<Team>>, FrozenOpponentPrior) {
    let (dex, teams) = assets();
    let prior = FrozenOpponentPrior::from_training_json(
        &dex,
        include_bytes!("../data/training-teams.json"),
    )
    .unwrap();
    (dex, teams, prior)
}

fn information(dex: &Dex, teams: &[Team]) -> SearchInformationSet {
    let state =
        BattleState::reset(dex, [&teams[0], &teams[1]], [201, 202, 203, 204], [0, 1]).unwrap();
    SearchInformationSet::from_player_view(&state.observe(SideId::P1)).unwrap()
}

#[test]
fn prior_manifest_is_frozen_legal_uniform_and_id_free() {
    let (dex, _, prior) = prior();
    assert_eq!(prior.candidate_count(), 1137);
    assert_eq!(prior.manifest().source_sha256, SOURCE_SHA256);
    assert_eq!(prior.manifest().source_schema, "pa3-team-record-v2");
    assert_eq!(prior.manifest().format, pa3_engine::FORMAT);
    assert_eq!(prior.manifest().dex_digest, dex.asset_digest);
    assert_eq!(
        prior.manifest().weighting,
        "uniform over legal complete sheets"
    );
    let bytes = prior.to_bytes().unwrap();
    assert!(!bytes
        .windows(b"pa3-0010b735d7ce90d2472439ceb0fe959bf6ca8a64c33dc30bacf0dace4f84566e".len())
        .any(|window| window
            == b"pa3-0010b735d7ce90d2472439ceb0fe959bf6ca8a64c33dc30bacf0dace4f84566e"));
    assert_eq!(bytes, prior.to_bytes().unwrap());
}

#[test]
fn seeded_sampling_and_preview_coverage_are_reproducible() {
    let (dex, teams, prior) = prior();
    let info = information(&dex, &teams);
    let first = prior.sample(&info, 0x1234_5678_9abc_def0, 8).unwrap();
    let second = prior.sample(&info, 0x1234_5678_9abc_def0, 8).unwrap();
    assert_eq!(first, second);
    let BeliefSampleResult::Samples(samples) = first else {
        panic!("initial preview should have prior support");
    };
    let expected_samples = 8.min(samples.diagnostics.candidate_count_after);
    assert_eq!(samples.hypotheses.len(), expected_samples);
    assert_eq!(samples.diagnostics.sampled_count, expected_samples);
    assert!((samples.diagnostics.normalized_weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert_eq!(
        samples.diagnostics.effective_candidate_count as usize,
        samples.diagnostics.candidate_count_after
    );

    let coverage = prior.preview_identification_report();
    assert_eq!(coverage.observations, 1137);
    assert!((0.0..=1.0).contains(&coverage.singleton_fraction));
    assert_eq!(coverage.ordered_singleton_observations, 817);
    assert_eq!(coverage.unordered_singleton_observations, 793);
    assert!((coverage.unordered_mean_candidate_count - 3.12665).abs() < 0.0001);
    assert_eq!(
        coverage.source_sheet_disjoint_evaluation,
        "not available from the frozen training pool"
    );
    assert_eq!(
        coverage.novel_preview_roster_evaluation,
        "not available from the frozen training pool"
    );
}

#[test]
fn prior_diversity_audit_reports_unique_complete_sheets_without_ids() {
    let (dex, teams, prior) = prior();
    let info = information(&dex, &teams);
    let first = prior.diversity_report(&info);
    let second = prior.diversity_report(&info);
    assert_eq!(first, second);
    assert_eq!(first.schema_version, 1);
    assert_eq!(first.source_sha256, SOURCE_SHA256);
    assert!(first.candidate_count >= first.unique_team_hypotheses);
    assert_eq!(
        first.candidate_count,
        first.unique_team_hypotheses + first.exact_duplicate_candidates
    );
    println!(
        "C3b-1.3 prior diversity candidates={} unique={} duplicates={}",
        first.candidate_count, first.unique_team_hypotheses, first.exact_duplicate_candidates
    );
}

#[test]
fn known_species_filters_unknown_values_do_not_and_out_of_support_is_explicit() {
    let (dex, teams, prior) = prior();
    let info = information(&dex, &teams);
    let known = prior.sample(&info, 1, 1).unwrap();
    let BeliefSampleResult::Samples(known) = known else {
        panic!("known preview should have support");
    };

    let mut unknown = info.clone();
    unknown.public.knowledge.pokemon[6].species = 0;
    let BeliefSampleResult::Samples(unknown_result) = prior.sample(&unknown, 1, 1).unwrap() else {
        panic!("unknown species should not remove all support");
    };
    assert!(
        unknown_result.diagnostics.candidate_count_after >= known.diagnostics.candidate_count_after
    );

    let mut impossible = info;
    impossible.public.knowledge.pokemon[6].species = u16::MAX;
    let BeliefSampleResult::OutOfSupport(diagnostics) = prior.sample(&impossible, 1, 1).unwrap()
    else {
        panic!("impossible public species must be explicit out-of-support");
    };
    assert!(diagnostics.out_of_support);
    assert_eq!(diagnostics.candidate_count_after, 0);
    assert!(diagnostics
        .rejection_reasons
        .contains_key("species_mismatch"));
}

#[test]
fn ambiguous_revealed_set_facts_are_reported_not_used_as_hidden_truth() {
    let (dex, teams, prior) = prior();
    let mut info = information(&dex, &teams);
    let baseline = prior.sample(&info, 7, 2).unwrap();
    info.public.knowledge.pokemon[6].current_moves[0] = Known::new(1);
    info.public.knowledge.pokemon[6].ability = Known::new(1);
    info.public.knowledge.pokemon[6].item = Known::new(1);
    let BeliefSampleResult::Samples(result) = prior.sample(&info, 7, 2).unwrap() else {
        panic!("unsupported set evidence should not remove species support");
    };
    let BeliefSampleResult::Samples(baseline) = baseline else {
        panic!("baseline should have support");
    };
    assert_eq!(
        result.diagnostics.candidate_count_after,
        baseline.diagnostics.candidate_count_after
    );
    assert!(result.diagnostics.unsupported_observations.len() >= 3);
}

#[test]
fn hidden_truth_metamorphic_sampling_is_identical() {
    let (dex, teams, prior) = prior();
    let mut hidden = teams[1].clone();
    hidden.id = "actual-hidden-id-never-consumed".into();
    for member in &mut hidden.members {
        member.points = [1, 0, 0, 0, 0, 0];
        member.ivs = [0; 6];
        member.nature = dex.id("natures", "serious").unwrap();
    }
    let first =
        BattleState::reset(&dex, [&teams[0], &teams[1]], [211, 212, 213, 214], [0, 1]).unwrap();
    let second =
        BattleState::reset(&dex, [&teams[0], &hidden], [211, 212, 213, 214], [0, 1]).unwrap();
    let first_info = SearchInformationSet::from_player_view(&first.observe(SideId::P1)).unwrap();
    let second_info = SearchInformationSet::from_player_view(&second.observe(SideId::P1)).unwrap();
    assert_eq!(first_info, second_info);
    assert_eq!(
        prior.sample(&first_info, 0xfeed_face, 16).unwrap(),
        prior.sample(&second_info, 0xfeed_face, 16).unwrap()
    );
}

#[test]
fn malformed_prior_source_is_rejected() {
    let (dex, _) = assets();
    assert!(FrozenOpponentPrior::from_training_json(&dex, b"{}\n").is_err());
}
