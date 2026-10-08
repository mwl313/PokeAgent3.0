//! Shared loader for the split differential corpus.
//!
//! The corpus is committed as size-bounded parts (`turn-fixtures.json`,
//! `turn-fixtures-2.json`, ...) so that no single file can cross the 100 MiB
//! host limit as coverage grows. Every reader sees exactly the merged view the
//! single file used to provide: `{oracle_commit, format, fixtures}`.
//!
//! Development-only support code: included by test/example binaries through
//! `#[path = ".../test_support/corpus.rs"] mod corpus;`. It embeds no data, so
//! it never reaches the library binary or the training path.
#![allow(dead_code)] // each consumer uses a subset of the helpers

use std::fs;
use std::path::{Path, PathBuf};

/// `turn-fixtures.json` -> Some(1), `turn-fixtures-2.json` -> Some(2).
pub fn part_index(name: &str) -> Option<u32> {
    let rest = name
        .strip_prefix("turn-fixtures")?
        .strip_suffix(".json")?;
    if rest.is_empty() {
        return Some(1);
    }
    rest.strip_prefix('-')?.parse().ok()
}

/// Every corpus part in index order, as `(file name, raw bytes)`.
pub fn corpus_parts(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut parts: Vec<(u32, String, Vec<u8>)> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("corpus dir {dir:?}: {error}"))
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let index = part_index(&name)?;
            Some((index, name, fs::read(entry.path()).unwrap()))
        })
        .collect();
    parts.sort_by_key(|(index, _, _)| *index);
    assert!(!parts.is_empty(), "no corpus parts under {dir:?}");
    parts
        .into_iter()
        .map(|(_, name, bytes)| (name, bytes))
        .collect()
}

/// The merged corpus JSON exactly as the single-file layout used to store it.
pub fn corpus_json(dir: &Path) -> String {
    let mut fixtures: Vec<serde_json::Value> = Vec::new();
    let (mut oracle, mut format) = (String::new(), String::new());
    for (_, bytes) in corpus_parts(dir) {
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("corpus part json");
        if oracle.is_empty() {
            oracle = value["oracle_commit"].as_str().unwrap_or_default().to_string();
            format = value["format"].as_str().unwrap_or_default().to_string();
        }
        fixtures.extend(value["fixtures"].as_array().cloned().unwrap_or_default());
    }
    serde_json::json!({
        "oracle_commit": oracle,
        "format": format,
        "fixtures": fixtures,
    })
    .to_string()
}

/// `engine/data`, resolved from the crate being compiled.
pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}
