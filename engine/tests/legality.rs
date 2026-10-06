//! Differential test for the native submission validator.
//!
//! Every case in `engine/data/legality-cases.json` carries the verdict and
//! problem category the pinned Showdown `TeamValidator` produced; the native
//! validator must agree on both.
use pa3_engine::{assets::Dex, legality::Verdict, state::Team};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Corpus {
    oracle_commit: String,
    format: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    source: String,
    teams: Vec<Team>,
    reference_legal: bool,
    reference_problems: Vec<String>,
    reference_category: String,
}

/// Reference problem category -> native verdicts that prove the same rule.
fn accepts(category: &str, verdict: Verdict) -> bool {
    match category {
        "legal" => verdict == Verdict::Legal,
        "species_clause" => verdict == Verdict::SpeciesClause,
        "item_clause" => verdict == Verdict::ItemClause,
        "banned" => verdict == Verdict::SpeciesNotAllowed,
        "species" => verdict == Verdict::SpeciesNotAllowed,
        "illegal_ability" => verdict == Verdict::IllegalAbility,
        "illegal_move" => matches!(verdict, Verdict::IllegalMove | Verdict::DuplicateMove),
        "duplicate_move" => verdict == Verdict::DuplicateMove,
        "illegal_item" => verdict == Verdict::IllegalItem,
        "stat_points" => matches!(verdict, Verdict::StatPoints | Verdict::ZeroStatPoints),
        "level" => verdict == Verdict::Level,
        "ivs" => verdict == Verdict::Ivs,
        "nicknames" | "gender" | "form_or_mega" => true,
        _ => true,
    }
}

#[test]
fn native_legality_matches_the_pinned_reference() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus =
        serde_json::from_str(include_str!("../data/legality-cases.json")).unwrap();
    assert_eq!(corpus.oracle_commit, pa3_engine::ORACLE_COMMIT);
    assert_eq!(corpus.format, pa3_engine::FORMAT);
    assert!(corpus.cases.len() >= 82, "legality corpus shrank");
    let mut pool_cases = 0;
    let mut rejected = 0;
    for case in &corpus.cases {
        assert_eq!(case.teams.len(), 1, "{} fixture shape", case.name);
        let report = pa3_engine::legality::validate_team(&dex, &case.teams[0]);
        assert_eq!(
            report.legal(),
            case.reference_legal,
            "{}: native {:?} vs reference problems {:?}",
            case.name,
            report,
            case.reference_problems
        );
        assert!(
            accepts(&case.reference_category, report.verdict),
            "{}: native {:?} does not prove reference category {}",
            case.name,
            report.verdict,
            case.reference_category
        );
        if case.source == "frozen_training_pool" {
            pool_cases += 1;
            assert!(case.reference_legal, "{}: pool team must be legal", case.name);
        }
        if !case.reference_legal {
            rejected += 1;
        }
    }
    assert!(pool_cases >= 60, "expected a pool slice, saw {pool_cases}");
    assert!(rejected >= 15, "expected crafted rejections, saw {rejected}");
}
