//! Coverage gate: every regulation move the engine can execute must have a
//! differential fixture that actually ran it, or an explicit documented
//! exemption. This keeps "enabled without proof" from creeping in as mechanics
//! are ported.
use pa3_engine::assets::Dex;
use pa3_engine::effects::MoveBehavior;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Deserialize)]
struct Scope {
    allowed_moves: Vec<String>,
}

#[derive(Deserialize)]
struct Corpus {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    #[serde(default)]
    coverage: Coverage,
}

#[derive(Deserialize, Default)]
struct Coverage {
    #[serde(default, rename = "move")]
    move_id: String,
}

/// Moves that are executable but have no dedicated fixture yet, with the
/// reason. Each entry must stay justified: remove it as soon as a fixture
/// exercises the move, and never add a move here that a fixture can cover.
const EXEMPT: &[(&str, &str)] = &[
    ("bittermalice", "no legal holder with an implemented ability yet"),
    ("nightdaze", "no legal holder with an implemented ability yet"),
    ("pound", "no legal holder with an implemented ability yet"),
    ("softboiled", "no legal holder with an implemented ability yet"),
    ("snaptrap", "no legal holder with an implemented ability yet"),
    ("spore", "no legal holder with an implemented ability yet"),
    ("struggle", "engine-internal fallback move, never selectable from a request"),
];

#[test]
fn every_executable_move_has_a_differential_fixture() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let scope: Scope =
        serde_json::from_str(include_str!("../data/scope.json")).unwrap();
    let corpus: Corpus =
        serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let covered: BTreeSet<String> = corpus
        .fixtures
        .iter()
        .filter(|fixture| !fixture.coverage.move_id.is_empty())
        .map(|fixture| fixture.coverage.move_id.clone())
        .collect();
    let exempt: BTreeSet<&str> = EXEMPT.iter().map(|(id, _)| *id).collect();
    let mut uncovered = Vec::new();
    for raw in &scope.allowed_moves {
        let Ok(id) = dex.id("moves", raw) else {
            panic!("scope lists an unknown move {raw}");
        };
        if dex.effects.moves[id as usize] == MoveBehavior::Unimplemented {
            continue;
        }
        if covered.contains(raw) || exempt.contains(raw.as_str()) {
            continue;
        }
        uncovered.push(raw.clone());
    }
    assert!(
        uncovered.is_empty(),
        "executable moves without a differential fixture: {uncovered:?}"
    );
    // The exemption list must not go stale: an exempt move either becomes
    // covered (remove the entry) or stops being executable.
    for (id, reason) in EXEMPT {
        assert!(
            !covered.contains(*id),
            "move {id} now has a fixture; remove its exemption ({reason})"
        );
        let numeric = dex.id("moves", id).unwrap_or(0);
        assert_ne!(
            dex.effects.moves[numeric as usize],
            MoveBehavior::Unimplemented,
            "move {id} is no longer executable; remove its exemption ({reason})"
        );
    }
}

/// Every executable move that can be reached *dynamically* (it calls, copies,
/// transforms into or transfers another effect) must also carry a differential
/// witness; otherwise closure callers could be enabled without proof that they
/// interact with the rest of the move set correctly. Closure membership is
/// generated from the pinned reference by `scripts/dynamic_closure.mjs`.
#[test]
fn every_executable_dynamic_caller_has_a_differential_fixture() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus =
        serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let covered: BTreeSet<String> = corpus
        .fixtures
        .iter()
        .filter(|fixture| !fixture.coverage.move_id.is_empty())
        .map(|fixture| fixture.coverage.move_id.clone())
        .collect();
    let exempt: BTreeSet<&str> = EXEMPT.iter().map(|(id, _)| *id).collect();
    let closure: serde_json::Value =
        serde_json::from_str(include_str!("../data/dynamic-closure.json")).unwrap();
    let mut missing = Vec::new();
    for entry in closure["entries"].as_array().expect("closure entries") {
        if entry["kind"].as_str() != Some("move") {
            continue;
        }
        let id = entry["id"].as_str().expect("closure id");
        let numeric = dex.id("moves", id).expect("closure move in catalogue");
        if dex.effects.moves[numeric as usize] == MoveBehavior::Unimplemented {
            continue;
        }
        if !covered.contains(id) && !exempt.contains(id) {
            missing.push(id.to_string());
        }
    }
    assert!(
        missing.is_empty(),
        "executable dynamic callers without a differential fixture: {missing:?}"
    );
}
