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
    pub tied_score_pairs: usize,
    pub unequal_coverage: bool,
    pub heuristic_differentiated_actions: usize,
    pub candidate_selection: String,
    pub budget: SearchBudget,
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
            tied_score_pairs: 0,
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
        let selected_candidates =
            select_preview_candidates(&all_candidates, config.max_root_candidates);
        let candidate_count = selected_candidates.len();
        diagnostics.omitted_root_candidates = all_candidates.len().saturating_sub(candidate_count);
        diagnostics.candidate_selection = if diagnostics.omitted_root_candidates == 0 {
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
        for action in selected_candidates {
            let mut total = 0.0f32;
            let mut evaluated = 0usize;
            let mut terminal = 0usize;
            let mut heuristic = 0usize;
            let mut unknown_total = 0.0f32;
            for (world_index, sample) in samples.hypotheses.iter().enumerate() {
                let world_seed = derive_seed(seeds, world_index);
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
                for _ in 0..config.rollout_combat_turns {
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
                if transitions < config.rollout_combat_turns * 2
                    && !boundary.root_view.outcome.terminated
                    && config.rollout_combat_turns > 0
                {
                    continue;
                }
                let Some((score, is_terminal, unknown_fraction)) =
                    score_view(&boundary.root_view, root_side)
                else {
                    diagnostics.failed_world_evaluations += 1;
                    continue;
                };
                total += score;
                unknown_total += unknown_fraction;
                evaluated += 1;
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
        let selected_action = if matches!(status, SearchStatus::Complete) {
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
            diagnostics.heuristic_differentiated_actions = scores
                .iter()
                .filter(|score| score.mean_score.abs() > 1e-6)
                .count();
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
}
