//! Deterministic player-safe opponent belief prior and sampler.
//!
//! This module filters complete team sheets using only `SearchInformationSet`.
//! It never accepts a `BattleState`, snapshot, team identifier, pending command,
//! or battle RNG. It is a prior/sampler only: no world construction, policy
//! evaluation, search teacher, distillation, or training is implemented here.

use crate::{
    assets::{Dex, Id},
    knowledge::Known,
    legality,
    search::information_set::SearchInformationSet,
    state::{Team, TeamSet},
    EngineError, Result, FORMAT,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const BELIEF_PRIOR_SCHEMA: u16 = 1;
pub const BELIEF_SOURCE_SCHEMA: &str = "pa3-team-record-v2";
const MAX_SAMPLES: usize = 4096;

/// A complete six-member sheet with the source identifier intentionally
/// removed. Correlations within a team remain intact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefTeam {
    pub members: [TeamSet; 6],
}

impl PartialEq for BeliefTeam {
    fn eq(&self, other: &Self) -> bool {
        serde_json::to_vec(&self.members).ok() == serde_json::to_vec(&other.members).ok()
    }
}

impl Eq for BeliefTeam {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefPriorManifest {
    pub schema_version: u16,
    pub source_sha256: String,
    pub source_schema: String,
    pub format: String,
    pub dex_digest: String,
    pub candidate_count: usize,
    pub weighting: String,
    pub generation_provenance: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BeliefPriorDocument {
    manifest: BeliefPriorManifest,
    teams: Vec<BeliefTeam>,
}

/// Frozen uniform prior over legal complete sheets. Runtime-facing methods do
/// not expose source team IDs or source pool indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenOpponentPrior {
    manifest: BeliefPriorManifest,
    teams: Vec<BeliefTeam>,
}

impl FrozenOpponentPrior {
    /// Build directly from the frozen JSON bytes so the manifest records the
    /// exact source hash used for generation.
    pub fn from_training_json(dex: &Dex, source: &[u8]) -> Result<Self> {
        let teams: Vec<Team> = serde_json::from_slice(source)?;
        if teams.is_empty() {
            return Err(EngineError::InvalidInput("empty opponent prior".into()));
        }
        let mut anonymized = Vec::with_capacity(teams.len());
        for (index, team) in teams.into_iter().enumerate() {
            if team.validate_structure(dex).is_err() {
                return Err(EngineError::InvalidInput(format!(
                    "prior team {index} failed structure validation"
                )));
            }
            let report = legality::validate_team(dex, &team);
            if !report.legal() {
                return Err(EngineError::InvalidInput(format!(
                    "prior team {index} failed legality: {:?}",
                    report.verdict
                )));
            }
            anonymized.push(BeliefTeam {
                members: team.members,
            });
        }
        let source_sha256 = format!("{:x}", Sha256::digest(source));
        let manifest = BeliefPriorManifest {
            schema_version: BELIEF_PRIOR_SCHEMA,
            source_sha256,
            source_schema: BELIEF_SOURCE_SCHEMA.into(),
            format: FORMAT.into(),
            dex_digest: dex.asset_digest.clone(),
            candidate_count: anonymized.len(),
            weighting: "uniform over legal complete sheets".into(),
            generation_provenance:
                "training-teams.json; source identifiers discarded at construction".into(),
        };
        Ok(Self {
            manifest,
            teams: anonymized,
        })
    }

    pub fn manifest(&self) -> &BeliefPriorManifest {
        &self.manifest
    }

    pub fn candidate_count(&self) -> usize {
        self.teams.len()
    }

    /// Root-safe aggregate diversity audit for a player-visible observation.
    /// It exposes counts and frozen-prior provenance only; source IDs and the
    /// private team sheets themselves remain unavailable to callers.
    pub fn diversity_report(&self, info: &SearchInformationSet) -> PriorDiversityReport {
        let filtered = self.candidates_for(info);
        let unique_team_hypotheses = filtered
            .candidates
            .iter()
            .filter_map(|team| serde_json::to_vec(&team.members).ok())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        PriorDiversityReport {
            schema_version: BELIEF_PRIOR_SCHEMA,
            source_sha256: self.manifest.source_sha256.clone(),
            source_schema: self.manifest.source_schema.clone(),
            candidate_count: filtered.candidates.len(),
            unique_team_hypotheses,
            exact_duplicate_candidates: filtered
                .candidates
                .len()
                .saturating_sub(unique_team_hypotheses),
            unsupported_observations: filtered.unsupported,
        }
    }

    #[cfg(test)]
    pub(crate) fn contains_complete_team_for_test(&self, team: &Team) -> bool {
        let encoded = serde_json::to_vec(&team.members).ok();
        self.teams
            .iter()
            .any(|candidate| serde_json::to_vec(&candidate.members).ok() == encoded)
    }

    /// Deterministic prior artifact. It contains complete anonymized sheets,
    /// never the source `Team.id` values.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&BeliefPriorDocument {
            manifest: self.manifest.clone(),
            teams: self.teams.clone(),
        })
        .map_err(EngineError::Json)
    }

    /// Measure closed-pool identification from six revealed species. This is
    /// a prior-coverage statistic, not a generalization or strength result.
    pub fn preview_identification_report(&self) -> PreviewIdentificationReport {
        let mut ordered_singleton = 0usize;
        let mut ordered_zero = 0usize;
        let mut ordered_total = 0usize;
        let mut unordered_singleton = 0usize;
        let mut unordered_zero = 0usize;
        let mut unordered_total = 0usize;
        for team in &self.teams {
            let species: [Id; 6] = std::array::from_fn(|i| team.members[i].species);
            let mut unordered_species = species;
            unordered_species.sort_unstable();
            let ordered_matches = self
                .teams
                .iter()
                .filter(|candidate| {
                    candidate
                        .members
                        .iter()
                        .zip(species)
                        .all(|(member, expected)| member.species == expected)
                })
                .count();
            let unordered_matches = self
                .teams
                .iter()
                .filter(|candidate| {
                    let mut candidate_species: [Id; 6] =
                        std::array::from_fn(|i| candidate.members[i].species);
                    candidate_species.sort_unstable();
                    candidate_species == unordered_species
                })
                .count();
            ordered_total += ordered_matches;
            ordered_singleton += usize::from(ordered_matches == 1);
            ordered_zero += usize::from(ordered_matches == 0);
            unordered_total += unordered_matches;
            unordered_singleton += usize::from(unordered_matches == 1);
            unordered_zero += usize::from(unordered_matches == 0);
        }
        let observations = self.teams.len();
        PreviewIdentificationReport {
            observations,
            singleton_observations: ordered_singleton,
            zero_match_observations: ordered_zero,
            singleton_fraction: ordered_singleton as f32 / observations as f32,
            mean_candidate_count: ordered_total as f32 / observations as f32,
            ordered_singleton_observations: ordered_singleton,
            ordered_zero_match_observations: ordered_zero,
            ordered_singleton_fraction: ordered_singleton as f32 / observations as f32,
            ordered_mean_candidate_count: ordered_total as f32 / observations as f32,
            unordered_singleton_observations: unordered_singleton,
            unordered_zero_match_observations: unordered_zero,
            unordered_singleton_fraction: unordered_singleton as f32 / observations as f32,
            unordered_mean_candidate_count: unordered_total as f32 / observations as f32,
            source_sheet_disjoint_evaluation: "not available from the frozen training pool".into(),
            novel_preview_roster_evaluation: "not available from the frozen training pool".into(),
        }
    }

    fn candidates_for(&self, info: &SearchInformationSet) -> FilteredCandidates {
        let mut rejection_reasons = BTreeMap::new();
        let mut unsupported = Vec::new();
        let opponent = &info.public.knowledge.pokemon[6..12];
        for (slot, mon) in opponent.iter().enumerate() {
            if mon.species == 0 {
                unsupported.push(format!("opponent species slot {slot} unknown"));
            }
            if mon.current_moves.iter().any(|value| value.known)
                || !mon.revealed_move_repertoire.is_empty()
            {
                unsupported.push(format!(
                    "opponent move provenance slot {slot} is ambiguous (copied/transformed moves unsupported)"
                ));
            }
            if mon.ability.known {
                unsupported.push(format!(
                    "opponent ability provenance slot {slot} is ambiguous (ability changes unsupported)"
                ));
            }
            if mon.item.known {
                unsupported.push(format!(
                    "opponent item provenance slot {slot} is ambiguous (item changes unsupported)"
                ));
            }
        }
        let before = self.teams.len();
        let candidates: Vec<_> = self
            .teams
            .iter()
            .filter(|team| {
                opponent.iter().enumerate().all(|(slot, observed)| {
                    let member = &team.members[slot];
                    if observed.species != 0 && member.species != observed.species {
                        increment(&mut rejection_reasons, "species_mismatch");
                        return false;
                    }
                    if let Some(gender) = explicit_gender(observed.gender) {
                        // An empty source gender is randomized at reset, so it
                        // remains compatible with either observed sex.
                        if !member.gender.is_empty() && member.gender != gender {
                            increment(&mut rejection_reasons, "gender_mismatch");
                            return false;
                        }
                    }
                    true
                })
            })
            .cloned()
            .collect();
        FilteredCandidates {
            before,
            candidates,
            rejection_reasons,
            unsupported,
        }
    }

    /// Sample without replacement using only an explicit independent seed.
    pub fn sample(
        &self,
        info: &SearchInformationSet,
        seed: u64,
        sample_count: usize,
    ) -> Result<BeliefSampleResult> {
        if sample_count == 0 || sample_count > MAX_SAMPLES {
            return Err(EngineError::InvalidInput(format!(
                "belief sample count must be 1..{MAX_SAMPLES}"
            )));
        }
        let filtered = self.candidates_for(info);
        let after = filtered.candidates.len();
        let mut diagnostics = BeliefDiagnostics {
            candidate_count_before: filtered.before,
            candidate_count_after: after,
            sampled_count: 0,
            normalized_weights: vec![],
            effective_candidate_count: after as f32,
            rejection_reasons: filtered.rejection_reasons,
            unsupported_observations: filtered.unsupported,
            out_of_support: after == 0,
        };
        if after == 0 {
            return Ok(BeliefSampleResult::OutOfSupport(diagnostics));
        }
        let take = sample_count.min(after);
        let weight = 1.0 / after as f32;
        diagnostics.normalized_weights = vec![weight; after];
        let mut indices: Vec<usize> = (0..after).collect();
        let mut rng = DeterministicRng::new(seed);
        for i in 0..take {
            let j = i + rng.bounded((after - i) as u64) as usize;
            indices.swap(i, j);
        }
        diagnostics.sampled_count = take;
        let hypotheses = indices
            .into_iter()
            .take(take)
            .map(|index| filtered.candidates[index].clone())
            .collect();
        Ok(BeliefSampleResult::Samples(BeliefSamples {
            hypotheses,
            diagnostics,
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreviewIdentificationReport {
    pub observations: usize,
    pub singleton_observations: usize,
    pub zero_match_observations: usize,
    pub singleton_fraction: f32,
    pub mean_candidate_count: f32,
    pub ordered_singleton_observations: usize,
    pub ordered_zero_match_observations: usize,
    pub ordered_singleton_fraction: f32,
    pub ordered_mean_candidate_count: f32,
    pub unordered_singleton_observations: usize,
    pub unordered_zero_match_observations: usize,
    pub unordered_singleton_fraction: f32,
    pub unordered_mean_candidate_count: f32,
    pub source_sheet_disjoint_evaluation: String,
    pub novel_preview_roster_evaluation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorDiversityReport {
    pub schema_version: u16,
    pub source_sha256: String,
    pub source_schema: String,
    pub candidate_count: usize,
    pub unique_team_hypotheses: usize,
    pub exact_duplicate_candidates: usize,
    pub unsupported_observations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefDiagnostics {
    pub candidate_count_before: usize,
    pub candidate_count_after: usize,
    pub sampled_count: usize,
    pub normalized_weights: Vec<f32>,
    pub effective_candidate_count: f32,
    pub rejection_reasons: BTreeMap<String, usize>,
    pub unsupported_observations: Vec<String>,
    pub out_of_support: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefSamples {
    pub hypotheses: Vec<BeliefTeam>,
    pub diagnostics: BeliefDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BeliefSampleResult {
    Samples(BeliefSamples),
    OutOfSupport(BeliefDiagnostics),
}

struct FilteredCandidates {
    before: usize,
    candidates: Vec<BeliefTeam>,
    rejection_reasons: BTreeMap<String, usize>,
    unsupported: Vec<String>,
}

fn increment(map: &mut BTreeMap<String, usize>, key: &str) {
    *map.entry(key.into()).or_default() += 1;
}

fn explicit_gender(gender: Known<u8>) -> Option<&'static str> {
    if !gender.known {
        return None;
    }
    match gender.value {
        1 => Some("M"),
        2 => Some("F"),
        0 => Some("N"),
        _ => None,
    }
}

struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e3779b97f4a7c15))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545f4914f6cdd1d)
    }

    fn bounded(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}
