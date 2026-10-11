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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchConfig {
    pub worlds: usize,
    pub max_root_candidates: usize,
    pub max_units: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            worlds: 3,
            max_root_candidates: 8,
            max_units: 2_000,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchDiagnostics {
    pub schema_version: u16,
    pub status_reason: String,
    pub total_root_candidates: usize,
    pub omitted_root_candidates: usize,
    pub sampled_worlds: usize,
    pub successful_world_evaluations: usize,
    pub failed_world_evaluations: usize,
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
        if !charge(&mut budget, config.max_units, all_candidates.len() as u64) {
            return Ok(abstain(
                diagnostics,
                "candidate generation budget exhausted",
                budget,
            ));
        }
        budget.candidate_generation = all_candidates.len();
        let candidate_count = config.max_root_candidates.min(all_candidates.len());
        diagnostics.omitted_root_candidates = all_candidates.len().saturating_sub(candidate_count);
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
        for (candidate_index, action) in
            all_candidates.into_iter().take(candidate_count).enumerate()
        {
            let mut total = 0.0f32;
            let mut evaluated = 0usize;
            let mut terminal = 0usize;
            let mut heuristic = 0usize;
            for (world_index, sample) in samples.hypotheses.iter().enumerate() {
                let world_seed = derive_seed(seeds, candidate_index, world_index);
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
                let (score, is_terminal) = score_view(&boundary.root_view, root_side);
                total += score;
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
                });
            }
            if budget.exhausted {
                break;
            }
        }
        let truncated = diagnostics.omitted_root_candidates > 0 || budget.exhausted;
        let status = if scores.len() < 2 {
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

fn score_view(view: &PlayerView, side: SideId) -> (f32, bool) {
    if let Some(reward) = view.outcome.reward(side) {
        return (f32::from(reward), true);
    }
    let own = view
        .own
        .pokemon
        .iter()
        .map(|pokemon| pokemon.hp as f32 / pokemon.stats[0].max(1) as f32)
        .sum::<f32>()
        / 6.0;
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
    ((own - opponent).clamp(-1.0, 1.0), false)
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

fn derive_seed(seeds: &[u64], candidate: usize, world: usize) -> u64 {
    seeds[(candidate + world) % seeds.len()]
        .wrapping_add((candidate as u64).wrapping_mul(0x9e37_79b9))
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
