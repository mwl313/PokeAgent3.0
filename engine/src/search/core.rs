//! Independent bounded clean-preview search coordinator.
//!
//! This module evaluates root legal actions over anonymized sampled worlds.
//! It is research-only: scores are transparent heuristics, not calibrated
//! values, teacher labels, or CTS-strength claims.

use crate::{
    EngineError, Result,
    actions::{AtomicAction, RequestKind},
    assets::Dex,
    search::{
        belief::{BeliefSampleResult, FrozenOpponentPrior},
        information_set::TrustedSearchInformationSet,
        world::{EpisodeFailureCode, PreviewWorld},
    },
    state::{PlayerView, SideId},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const SEARCH_CORE_SCHEMA: u16 = 1;
pub const HARD_MAX_WORLDS: usize = 16;
pub const HARD_MAX_ROOT_CANDIDATES: usize = 64;
pub const HARD_MAX_UNITS: u64 = 20_000;
pub const HARD_MAX_ROLLOUT_TURNS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchConfig {
    pub worlds: usize,
    pub max_root_candidates: usize,
    pub max_units: u64,
    pub rollout_combat_turns: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            worlds: 3,
            max_root_candidates: 8,
            max_units: 2_000,
            rollout_combat_turns: 1,
        }
    }
}

impl SearchConfig {
    fn validate(&self) -> Result<()> {
        if self.worlds == 0 || self.worlds > HARD_MAX_WORLDS {
            return Err(EngineError::InvalidInput(format!(
                "search worlds must be 1..{HARD_MAX_WORLDS}"
            )));
        }
        if self.max_root_candidates == 0 || self.max_root_candidates > HARD_MAX_ROOT_CANDIDATES {
            return Err(EngineError::InvalidInput(format!(
                "root candidates must be 1..{HARD_MAX_ROOT_CANDIDATES}"
            )));
        }
        if self.max_units == 0 || self.max_units > HARD_MAX_UNITS {
            return Err(EngineError::InvalidInput(format!(
                "search units must be 1..{HARD_MAX_UNITS}"
            )));
        }
        if self.rollout_combat_turns > HARD_MAX_ROLLOUT_TURNS {
            return Err(EngineError::InvalidInput(format!(
                "rollout combat turns must be 0..{HARD_MAX_ROLLOUT_TURNS}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStatus {
    Complete,
    TruncatedCandidates,
    InsufficientCoverage,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchActionScore {
    pub action: Vec<AtomicAction>,
    pub mean_score: f32,
    pub evaluated_worlds: usize,
    pub terminal_evaluations: usize,
    pub heuristic_evaluations: usize,
    pub mean_unknown_health_fraction: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchBudget {
    pub candidate_generation: usize,
    pub candidate_evaluated: usize,
    pub belief_samples: usize,
    pub world_construction_attempts: usize,
    pub engine_transitions: usize,
    pub opponent_action_samples: usize,
    pub chance_samples: usize,
    pub units: u64,
    pub exhausted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchDiagnostics {
    pub schema_version: u16,
    pub status_reason: String,
    pub total_root_candidates: usize,
    pub omitted_root_candidates: usize,
    pub sampled_worlds: usize,
    pub successful_world_evaluations: usize,
    pub failed_world_evaluations: usize,
    pub evaluated_root_actions: usize,
    pub opening_pair_count: usize,
    pub reserve_combination_count: usize,
    pub omitted_opening_pairs: usize,
    pub omitted_reserve_combinations: usize,
    pub score_variance: f32,
    pub paired_world_score_differences: Vec<PairedScoreDifference>,
    pub score_dispersion: f32,
    pub effective_world_coverage: f32,
    pub tied_score_pairs: usize,
    pub ranking_instability: f32,
    pub unequal_coverage: bool,
    pub heuristic_differentiated_actions: usize,
    pub candidate_selection: String,
    pub budget: SearchBudget,
}

/// Paired score differences for one candidate pair. Entries are indexed by
/// sampled-world index; `None` means one side lacked a valid evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedScoreDifference {
    pub candidate_a: usize,
    pub candidate_b: usize,
    pub per_world: Vec<Option<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub schema_version: u16,
    pub status: SearchStatus,
    pub selected_action: Option<Vec<AtomicAction>>,
    pub scores: Vec<SearchActionScore>,
    pub diagnostics: SearchDiagnostics,
}

pub struct BoundedSearch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvaluationMode {
    Hardened,
    C3a1Baseline,
}

impl BoundedSearch {
    /// Search only a clean preview root. Each candidate is evaluated from a
    /// newly constructed sampled world, and the synthetic opponent uses its
    /// own player view through the coordinator-only baseline hook.
    pub fn search(
        dex: Arc<Dex>,
        info: &TrustedSearchInformationSet,
        prior: &FrozenOpponentPrior,
        seeds: &[u64],
        config: SearchConfig,
    ) -> Result<SearchResult> {
        Self::search_with_mode(dex, info, prior, seeds, config, EvaluationMode::Hardened)
    }

    /// Retain the original C3a.1 first-N, candidate-dependent-seed,
    /// post-preview-only evaluator for apples-to-apples research comparisons.
    /// Its output is a baseline diagnostic, not a validated recommendation.
    pub fn search_c3a1_baseline(
        dex: Arc<Dex>,
        info: &TrustedSearchInformationSet,
        prior: &FrozenOpponentPrior,
        seeds: &[u64],
        config: SearchConfig,
    ) -> Result<SearchResult> {
        Self::search_with_mode(
            dex,
            info,
            prior,
            seeds,
            config,
            EvaluationMode::C3a1Baseline,
        )
    }

    fn search_with_mode(
        dex: Arc<Dex>,
        info: &TrustedSearchInformationSet,
        prior: &FrozenOpponentPrior,
        seeds: &[u64],
        config: SearchConfig,
        mode: EvaluationMode,
    ) -> Result<SearchResult> {
        config.validate()?;
        let mut budget = SearchBudget::default();
        let mut diagnostics = SearchDiagnostics {
            schema_version: SEARCH_CORE_SCHEMA,
            status_reason: String::new(),
            total_root_candidates: 0,
            omitted_root_candidates: 0,
            sampled_worlds: 0,
            successful_world_evaluations: 0,
            failed_world_evaluations: 0,
            evaluated_root_actions: 0,
            opening_pair_count: 0,
            reserve_combination_count: 0,
            omitted_opening_pairs: 0,
            omitted_reserve_combinations: 0,
            score_variance: 0.0,
            paired_world_score_differences: Vec::new(),
            score_dispersion: 0.0,
            effective_world_coverage: 0.0,
            tied_score_pairs: 0,
            ranking_instability: 0.0,
            unequal_coverage: false,
            heuristic_differentiated_actions: 0,
            candidate_selection: String::new(),
            budget: SearchBudget::default(),
        };
        if info.as_info().own.request.kind != RequestKind::Preview
            || info.as_info().public.turn != 0
        {
            return Ok(abstain(
                diagnostics,
                "only clean preview roots are supported",
                budget,
            ));
        }
        if seeds.is_empty() {
            return Ok(abstain(
                diagnostics,
                "at least one independent seed is required",
                budget,
            ));
        }
        let all_candidates = enumerate_complete(&info.as_info().own.request)?;
        diagnostics.total_root_candidates = all_candidates.len();
        budget.candidate_generation = all_candidates.len();
        if !charge(&mut budget, config.max_units, all_candidates.len() as u64) {
            return Ok(abstain(
                diagnostics,
                "candidate generation budget exhausted",
                budget,
            ));
        }
        let selected_candidates = match mode {
            EvaluationMode::Hardened => {
                select_preview_candidates(&all_candidates, config.max_root_candidates)
            }
            EvaluationMode::C3a1Baseline => all_candidates
                .iter()
                .take(config.max_root_candidates)
                .cloned()
                .collect(),
        };
        let candidate_count = selected_candidates.len();
        diagnostics.omitted_root_candidates = all_candidates.len().saturating_sub(candidate_count);
        diagnostics.candidate_selection = if mode == EvaluationMode::C3a1Baseline {
            "C3a.1 baseline first-N selection (comparison only)".into()
        } else if diagnostics.omitted_root_candidates == 0 {
            "exhaustive legal action set".into()
        } else {
            "stratified opening-pair/reserve selection; omitted actions recorded".into()
        };
        let all_openings = unique_family_count(&all_candidates, true);
        let all_reserves = unique_family_count(&all_candidates, false);
        diagnostics.opening_pair_count = unique_family_count(&selected_candidates, true);
        diagnostics.reserve_combination_count = unique_family_count(&selected_candidates, false);
        diagnostics.omitted_opening_pairs =
            all_openings.saturating_sub(diagnostics.opening_pair_count);
        diagnostics.omitted_reserve_combinations =
            all_reserves.saturating_sub(diagnostics.reserve_combination_count);
        if candidate_count < 2 {
            return Ok(abstain(
                diagnostics,
                "fewer than two legal root actions",
                budget,
            ));
        }
        let BeliefSampleResult::Samples(samples) =
            prior.sample(info.as_info(), seeds[0], config.worlds)?
        else {
            return Ok(abstain(
                diagnostics,
                "belief prior has no compatible sampled worlds",
                budget,
            ));
        };
        budget.belief_samples = samples.hypotheses.len();
        diagnostics.sampled_worlds = samples.hypotheses.len();
        if !charge(
            &mut budget,
            config.max_units,
            samples.hypotheses.len() as u64,
        ) {
            return Ok(abstain(
                diagnostics,
                "belief sampling budget exhausted",
                budget,
            ));
        }

        let root_side = info.as_info().own.side;
        let mut scores = Vec::with_capacity(candidate_count);
        let mut world_score_rows: Vec<Vec<Option<f32>>> = Vec::with_capacity(candidate_count);
        for (candidate_index, action) in selected_candidates.into_iter().enumerate() {
            let mut total = 0.0f32;
            let mut evaluated = 0usize;
            let mut terminal = 0usize;
            let mut heuristic = 0usize;
            let mut unknown_total = 0.0f32;
            let mut world_scores = vec![None; samples.hypotheses.len()];
            for (world_index, sample) in samples.hypotheses.iter().enumerate() {
                let world_seed = match mode {
                    EvaluationMode::Hardened => derive_seed(seeds, world_index),
                    EvaluationMode::C3a1Baseline => {
                        derive_legacy_seed(seeds, candidate_index, world_index)
                    }
                };
                let world = match PreviewWorld::construct(&dex, info, sample, world_seed) {
                    Ok(world) => world,
                    Err(error) => {
                        budget.world_construction_attempts += error.diagnostics.attempts;
                        if !charge(
                            &mut budget,
                            config.max_units,
                            error.diagnostics.attempts as u64,
                        ) {
                            budget.exhausted = true;
                            break;
                        }
                        diagnostics.failed_world_evaluations += 1;
                        continue;
                    }
                };
                budget.world_construction_attempts += world.diagnostics().attempts;
                if !charge(
                    &mut budget,
                    config.max_units,
                    world.diagnostics().attempts as u64,
                ) {
                    budget.exhausted = true;
                    break;
                }
                let mut episode = match world.into_episode(dex.clone()) {
                    Ok(episode) => episode,
                    Err(_) => {
                        diagnostics.failed_world_evaluations += 1;
                        continue;
                    }
                };
                if !charge(&mut budget, config.max_units, 4) {
                    budget.exhausted = true;
                    break;
                }
                budget.candidate_evaluated += 1;
                budget.engine_transitions += 2;
                budget.opponent_action_samples += 1;
                budget.chance_samples += 1;
                let boundary = match episode.submit_root_with_baseline(action.clone()) {
                    Ok(boundary) => boundary,
                    Err(error) if error.code == EpisodeFailureCode::OperationalError => {
                        diagnostics.failed_world_evaluations += 1;
                        continue;
                    }
                    Err(_) => {
                        diagnostics.failed_world_evaluations += 1;
                        continue;
                    }
                };
                let mut boundary = boundary;
                let mut transitions = 0usize;
                let rollout_turns = match mode {
                    EvaluationMode::Hardened => config.rollout_combat_turns,
                    EvaluationMode::C3a1Baseline => 0,
                };
                for _ in 0..rollout_turns {
                    if boundary.root_view.outcome.terminated
                        || boundary.root_view.request.kind != RequestKind::Normal
                    {
                        break;
                    }
                    if !charge(&mut budget, config.max_units, 4) {
                        budget.exhausted = true;
                        break;
                    }
                    boundary = match episode.submit_baseline_turn() {
                        Ok(next) => next,
                        Err(_) => {
                            diagnostics.failed_world_evaluations += 1;
                            break;
                        }
                    };
                    transitions += 2;
                    budget.engine_transitions += 2;
                    budget.opponent_action_samples += 1;
                    budget.chance_samples += 1;
                }
                if transitions < rollout_turns * 2
                    && !boundary.root_view.outcome.terminated
                    && rollout_turns > 0
                {
                    continue;
                }
                let scored = match mode {
                    EvaluationMode::Hardened => score_view(&boundary.root_view, root_side),
                    EvaluationMode::C3a1Baseline => {
                        Some(score_view_legacy(&boundary.root_view, root_side))
                    }
                };
                let Some((score, is_terminal, unknown_fraction)) = scored else {
                    diagnostics.failed_world_evaluations += 1;
                    continue;
                };
                total += score;
                unknown_total += unknown_fraction;
                evaluated += 1;
                world_scores[world_index] = Some(score);
                if is_terminal {
                    terminal += 1;
                } else {
                    heuristic += 1;
                }
            }
            if evaluated > 0 {
                diagnostics.successful_world_evaluations += evaluated;
                scores.push(SearchActionScore {
                    action,
                    mean_score: total / evaluated as f32,
                    evaluated_worlds: evaluated,
                    terminal_evaluations: terminal,
                    heuristic_evaluations: heuristic,
                    mean_unknown_health_fraction: unknown_total / evaluated as f32,
                });
                world_score_rows.push(world_scores);
            }
            if budget.exhausted {
                break;
            }
        }
        let truncated = diagnostics.omitted_root_candidates > 0 || budget.exhausted;
        let coverage = scores.first().map(|score| score.evaluated_worlds);
        diagnostics.unequal_coverage = scores
            .iter()
            .any(|score| Some(score.evaluated_worlds) != coverage)
            || diagnostics.failed_world_evaluations > 0;
        diagnostics.evaluated_root_actions = scores.len();
        let status = if scores.len() < 2 || diagnostics.unequal_coverage {
            SearchStatus::InsufficientCoverage
        } else if truncated {
            SearchStatus::TruncatedCandidates
        } else {
            SearchStatus::Complete
        };
        let selected_action =
            if mode == EvaluationMode::Hardened && matches!(status, SearchStatus::Complete) {
                scores
                    .iter()
                    .max_by(|a, b| a.mean_score.total_cmp(&b.mean_score))
                    .map(|s| s.action.clone())
            } else {
                None
            };
        if scores.len() > 1 {
            let mean =
                scores.iter().map(|score| score.mean_score).sum::<f32>() / scores.len() as f32;
            diagnostics.score_variance = scores
                .iter()
                .map(|score| (score.mean_score - mean).powi(2))
                .sum::<f32>()
                / scores.len() as f32;
            diagnostics.tied_score_pairs = scores
                .iter()
                .enumerate()
                .flat_map(|(i, a)| scores.iter().skip(i + 1).map(move |b| (a, b)))
                .filter(|(a, b)| (a.mean_score - b.mean_score).abs() < 1e-6)
                .count();
            let mut distinct_scores = scores
                .iter()
                .map(|score| score.mean_score)
                .collect::<Vec<_>>();
            distinct_scores.sort_by(f32::total_cmp);
            distinct_scores.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            diagnostics.heuristic_differentiated_actions = distinct_scores.len();

            for candidate_a in 0..world_score_rows.len() {
                for candidate_b in (candidate_a + 1)..world_score_rows.len() {
                    let per_world = world_score_rows[candidate_a]
                        .iter()
                        .zip(&world_score_rows[candidate_b])
                        .map(|(a, b)| a.zip(*b).map(|(left, right)| left - right))
                        .collect();
                    diagnostics
                        .paired_world_score_differences
                        .push(PairedScoreDifference {
                            candidate_a,
                            candidate_b,
                            per_world,
                        });
                }
            }
            let valid_world_scores = world_score_rows
                .iter()
                .flat_map(|row| row.iter().flatten().copied())
                .collect::<Vec<_>>();
            if valid_world_scores.len() > 1 {
                let mean = valid_world_scores.iter().sum::<f32>() / valid_world_scores.len() as f32;
                diagnostics.score_dispersion = valid_world_scores
                    .iter()
                    .map(|score| (score - mean).powi(2))
                    .sum::<f32>()
                    / valid_world_scores.len() as f32;
            }
            diagnostics.effective_world_coverage = if diagnostics.sampled_worlds == 0 {
                0.0
            } else {
                scores
                    .iter()
                    .map(|score| score.evaluated_worlds as f32)
                    .sum::<f32>()
                    / (scores.len() * diagnostics.sampled_worlds) as f32
            };
            let aggregate_winner = scores
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.mean_score.total_cmp(&right.mean_score))
                .map(|(index, _)| index);
            let mut ranked_worlds = 0usize;
            let mut unstable_worlds = 0usize;
            for world_index in 0..diagnostics.sampled_worlds {
                let Some(world_winner) = world_score_rows
                    .iter()
                    .enumerate()
                    .filter_map(|(index, row)| {
                        row.get(world_index)
                            .and_then(|score| score.map(|value| (index, value)))
                    })
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index)
                else {
                    continue;
                };
                ranked_worlds += 1;
                if Some(world_winner) != aggregate_winner {
                    unstable_worlds += 1;
                }
            }
            diagnostics.ranking_instability = if ranked_worlds == 0 {
                0.0
            } else {
                unstable_worlds as f32 / ranked_worlds as f32
            };
        }
        diagnostics.status_reason = format!(
            "evaluated {} root actions across {} sampled worlds",
            scores.len(),
            diagnostics.sampled_worlds
        );
        diagnostics.budget = budget;
        Ok(SearchResult {
            schema_version: SEARCH_CORE_SCHEMA,
            status,
            selected_action,
            scores,
            diagnostics,
        })
    }
}

fn enumerate_complete(request: &crate::actions::Request) -> Result<Vec<Vec<AtomicAction>>> {
    fn descend(
        request: &crate::actions::Request,
        prefix: &mut Vec<AtomicAction>,
        out: &mut Vec<Vec<AtomicAction>>,
    ) -> Result<()> {
        let candidates = request.candidates(prefix)?;
        if candidates.is_empty() {
            request.validate_joint(prefix)?;
            out.push(prefix.clone());
            return Ok(());
        }
        for action in candidates {
            prefix.push(action);
            descend(request, prefix, out)?;
            prefix.pop();
        }
        Ok(())
    }
    let mut out = Vec::new();
    descend(request, &mut Vec::new(), &mut out)?;
    Ok(out)
}

fn preview_family(action: &[AtomicAction], opening: bool) -> Vec<u8> {
    let range = if opening { 0..2 } else { 2..4 };
    range
        .filter_map(|index| action.get(index).map(|value| value.switch_destination))
        .collect()
}

fn unique_family_count(actions: &[Vec<AtomicAction>], opening: bool) -> usize {
    let mut keys = std::collections::BTreeSet::new();
    for action in actions {
        keys.insert(preview_family(action, opening));
    }
    keys.len()
}

fn select_preview_candidates(all: &[Vec<AtomicAction>], limit: usize) -> Vec<Vec<AtomicAction>> {
    if all.len() <= limit {
        return all.to_vec();
    }
    let mut selected = Vec::with_capacity(limit);
    let mut openings = std::collections::BTreeSet::new();
    let mut reserves = std::collections::BTreeSet::new();
    for action in all {
        let opening = preview_family(action, true);
        let reserve = preview_family(action, false);
        if !openings.contains(&opening) && !reserves.contains(&reserve) {
            selected.push(action.clone());
            openings.insert(opening);
            reserves.insert(reserve);
            if selected.len() == limit {
                return selected;
            }
        }
    }
    for action in all {
        let opening = preview_family(action, true);
        if openings.insert(opening) {
            selected.push(action.clone());
            if selected.len() == limit {
                return selected;
            }
        }
    }
    for action in all {
        let reserve = preview_family(action, false);
        if reserves.insert(reserve) {
            selected.push(action.clone());
            if selected.len() == limit {
                return selected;
            }
        }
    }
    for action in all {
        if selected.len() == limit {
            break;
        }
        if !selected.contains(action) {
            selected.push(action.clone());
        }
    }
    selected
}

#[cfg(test)]
fn legacy_select_preview_candidates(
    all: &[Vec<AtomicAction>],
    limit: usize,
) -> Vec<Vec<AtomicAction>> {
    if all.len() <= limit {
        return all.to_vec();
    }
    let mut selected = Vec::with_capacity(limit);
    let mut openings = std::collections::BTreeSet::new();
    let mut reserves = std::collections::BTreeSet::new();
    for action in all {
        let opening = preview_family(action, true);
        let reserve = preview_family(action, false);
        if openings.insert(opening) && reserves.insert(reserve) {
            selected.push(action.clone());
            if selected.len() == limit {
                return selected;
            }
        }
    }
    for action in all {
        let opening = preview_family(action, true);
        if openings.insert(opening) {
            selected.push(action.clone());
            if selected.len() == limit {
                return selected;
            }
        }
    }
    for action in all {
        let reserve = preview_family(action, false);
        if reserves.insert(reserve) {
            selected.push(action.clone());
            if selected.len() == limit {
                return selected;
            }
        }
    }
    for action in all {
        if selected.len() == limit {
            break;
        }
        if !selected.contains(action) {
            selected.push(action.clone());
        }
    }
    selected
}

fn score_view(view: &PlayerView, side: SideId) -> Option<(f32, bool, f32)> {
    if view.outcome.truncated || view.outcome.operational_error.is_some() {
        return None;
    }
    if let Some(reward) = view.outcome.reward(side) {
        return Some((f32::from(reward), true, 0.0));
    }
    let own = view
        .own
        .pokemon
        .iter()
        .map(|pokemon| pokemon.hp as f32 / pokemon.stats[0].max(1) as f32)
        .sum::<f32>()
        / 6.0;
    let known_opponents: Vec<_> = view.knowledge.pokemon[6..12]
        .iter()
        .filter_map(|pokemon| {
            pokemon
                .health
                .known
                .then_some(pokemon.health.value.numerator as f32 / 100.0)
        })
        .collect();
    if known_opponents.is_empty() {
        return None;
    }
    let opponent = known_opponents.iter().sum::<f32>() / known_opponents.len() as f32;
    let unknown_fraction = 1.0 - known_opponents.len() as f32 / 6.0;
    Some(((own - opponent).clamp(-1.0, 1.0), false, unknown_fraction))
}

fn score_view_legacy(view: &PlayerView, side: SideId) -> (f32, bool, f32) {
    if let Some(reward) = view.outcome.reward(side) {
        return (f32::from(reward), true, 0.0);
    }
    let own = view
        .own
        .pokemon
        .iter()
        .map(|pokemon| pokemon.hp as f32 / pokemon.stats[0].max(1) as f32)
        .sum::<f32>()
        / 6.0;
    let known_count = view.knowledge.pokemon[6..12]
        .iter()
        .filter(|pokemon| pokemon.health.known)
        .count();
    let opponent = view.knowledge.pokemon[6..12]
        .iter()
        .filter_map(|pokemon| {
            pokemon
                .health
                .known
                .then_some(pokemon.health.value.numerator as f32 / 100.0)
        })
        .sum::<f32>()
        / 6.0;
    (
        (own - opponent).clamp(-1.0, 1.0),
        false,
        1.0 - known_count as f32 / 6.0,
    )
}

fn charge(budget: &mut SearchBudget, limit: u64, units: u64) -> bool {
    let Some(next) = budget.units.checked_add(units) else {
        budget.exhausted = true;
        return false;
    };
    if next > limit {
        budget.exhausted = true;
        false
    } else {
        budget.units = next;
        true
    }
}

fn derive_seed(seeds: &[u64], world: usize) -> u64 {
    seeds[world % seeds.len()].wrapping_add(world as u64)
}

fn derive_legacy_seed(seeds: &[u64], candidate: usize, world: usize) -> u64 {
    seeds[(candidate + world) % seeds.len()]
        .wrapping_add(candidate as u64)
        .wrapping_add(world as u64)
}

fn abstain(mut diagnostics: SearchDiagnostics, reason: &str, budget: SearchBudget) -> SearchResult {
    diagnostics.status_reason = reason.into();
    diagnostics.budget = budget;
    SearchResult {
        schema_version: SEARCH_CORE_SCHEMA,
        status: SearchStatus::Abstain,
        selected_action: None,
        scores: vec![],
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::score_view;
    use crate::{
        assets::Dex,
        state::{BattleState, SideId},
    };
    use std::path::Path;

    #[test]
    fn terminal_reward_precedes_heuristic_state() {
        let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
        let teams: Vec<crate::state::Team> =
            serde_json::from_str(include_str!("../../data/training-teams.json")).unwrap();
        let mut view =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [401, 402, 403, 404], [0, 1])
                .unwrap()
                .observe(SideId::P1);
        view.outcome.terminated = true;
        view.outcome.winner = Some(SideId::P1);
        assert_eq!(score_view(&view, SideId::P1), Some((1.0, true, 0.0)));
    }

    #[test]
    fn unknown_opponent_health_abstains_instead_of_zero_filling() {
        let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
        let teams: Vec<crate::state::Team> =
            serde_json::from_str(include_str!("../../data/training-teams.json")).unwrap();
        let mut view =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [405, 406, 407, 408], [0, 1])
                .unwrap()
                .observe(SideId::P1);
        for pokemon in &mut view.knowledge.pokemon[6..12] {
            pokemon.health.known = false;
        }
        assert_eq!(score_view(&view, SideId::P1), None);
    }

    #[test]
    fn candidate_selector_reproduces_and_fixes_legacy_family_mutation() {
        let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
        let teams: Vec<crate::state::Team> =
            serde_json::from_str(include_str!("../../data/training-teams.json")).unwrap();
        let view = BattleState::reset(&dex, [&teams[0], &teams[1]], [409, 410, 411, 412], [0, 1])
            .unwrap()
            .observe(SideId::P1);
        let all = super::enumerate_complete(&view.request).unwrap();
        assert_eq!(all.len(), 360);
        let fixed = super::select_preview_candidates(&all, 8);
        let legacy = super::legacy_select_preview_candidates(&all, 8);
        assert_eq!(fixed.len(), 8);
        assert_eq!(legacy.len(), 8);
        let fixed_openings = super::unique_family_count(&fixed, true);
        let legacy_openings = super::unique_family_count(&legacy, true);
        let fixed_reserves = super::unique_family_count(&fixed, false);
        let legacy_reserves = super::unique_family_count(&legacy, false);
        assert!(fixed_openings >= legacy_openings);
        assert!(fixed_reserves >= legacy_reserves);
        assert!(fixed_openings != legacy_openings || fixed_reserves != legacy_reserves);
        assert!(fixed_openings > 0 && fixed_reserves > 0);
    }
}
