//! Bounded, forward-only particle feasibility prototype.
//!
//! This module deliberately starts from a clean-preview player observation.
//! Particles own their private synthetic worlds and are conditioned only on a
//! later trusted `PlayerView` projection. It is not a live-battle
//! reconstruction API, opponent policy, search teacher, or posterior model.

use crate::{
    EngineError, Result,
    actions::AtomicAction,
    assets::Dex,
    search::{
        belief::{BeliefSampleResult, FrozenOpponentPrior},
        information_set::{SearchInformationSet, TrustedSearchInformationSet},
        world::PreviewWorld,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const PARTICLE_BELIEF_SCHEMA: u16 = 1;
pub const HARD_MAX_PARTICLES: usize = 16;
pub const HARD_MAX_PARTICLE_UNITS: u64 = 2_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticleConfig {
    pub particles: usize,
    pub min_surviving_particles: usize,
    pub max_units: u64,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            particles: 4,
            min_surviving_particles: 2,
            max_units: 512,
        }
    }
}

impl ParticleConfig {
    fn validate(&self) -> Result<()> {
        if self.particles == 0 || self.particles > HARD_MAX_PARTICLES {
            return Err(EngineError::InvalidInput(format!(
                "particle count must be 1..{HARD_MAX_PARTICLES}"
            )));
        }
        if self.min_surviving_particles == 0 || self.min_surviving_particles > self.particles {
            return Err(EngineError::InvalidInput(
                "minimum surviving particles must be within the particle count".into(),
            ));
        }
        if self.max_units == 0 || self.max_units > HARD_MAX_PARTICLE_UNITS {
            return Err(EngineError::InvalidInput(format!(
                "particle units must be 1..{HARD_MAX_PARTICLE_UNITS}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticleStatus {
    Supported,
    OutOfSupport,
    ParticleCollapse,
    InsufficientCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticleDiagnostics {
    pub schema_version: u16,
    pub requested_particles: usize,
    pub sampled_particles: usize,
    pub constructed_particles: usize,
    pub advanced_particles: usize,
    pub surviving_particles: usize,
    pub rejected_observations: usize,
    pub failed_transitions: usize,
    pub independent_seed_count: usize,
    pub exact_observation_comparison: bool,
    pub units: u64,
    pub status_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticleResult {
    pub schema_version: u16,
    pub status: ParticleStatus,
    pub conditioned_observation: Option<SearchInformationSet>,
    pub diagnostics: ParticleDiagnostics,
}

/// The comparison contract is intentionally conservative: all current DTO
/// fields are compared exactly, while private world fields are excluded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationComparisonContract {
    pub schema_version: u16,
    pub exact_fields: Vec<String>,
    pub tolerated_fields: Vec<String>,
    pub excluded_private_fields: Vec<String>,
}

pub fn observation_comparison_contract() -> ObservationComparisonContract {
    ObservationComparisonContract {
        schema_version: PARTICLE_BELIEF_SCHEMA,
        exact_fields: vec![
            "schema_version".into(),
            "public.turn".into(),
            "public.knowledge".into(),
            "public.outcome".into(),
            "own.side".into(),
            "own.state".into(),
            "own.request".into(),
        ],
        tolerated_fields: vec![],
        excluded_private_fields: vec![
            "BattleState".into(),
            "opponent team identifier and private set".into(),
            "pending opponent actions".into(),
            "live RNG state".into(),
        ],
    }
}

pub struct ParticleBelief;

impl ParticleBelief {
    /// Initialize sampled clean-preview particles, resolve preview, advance
    /// one complete simultaneous combat turn, and condition on the next root
    /// observation.
    #[allow(clippy::too_many_arguments)]
    pub fn advance_preview_and_one_turn(
        dex: Arc<Dex>,
        initial: &TrustedSearchInformationSet,
        prior: &FrozenOpponentPrior,
        preview_actions: Vec<AtomicAction>,
        combat_actions: Vec<AtomicAction>,
        next_observation: &TrustedSearchInformationSet,
        seeds: &[u64],
        config: ParticleConfig,
    ) -> Result<ParticleResult> {
        config.validate()?;
        if seeds.is_empty() {
            return Err(EngineError::InvalidInput(
                "at least one particle seed is required".into(),
            ));
        }
        let initial_info = initial.as_info();
        let target = next_observation.as_info();
        if initial_info.own.request.kind != crate::actions::RequestKind::Preview
            || initial_info.public.turn != 0
        {
            return Err(EngineError::InvalidInput(
                "particle initialization requires a clean preview root".into(),
            ));
        }
        if target.own.side != initial_info.own.side {
            return Err(EngineError::InvalidInput(
                "conditioned observation must use the same root side".into(),
            ));
        }

        let mut diagnostics = ParticleDiagnostics {
            schema_version: PARTICLE_BELIEF_SCHEMA,
            requested_particles: config.particles,
            sampled_particles: 0,
            constructed_particles: 0,
            advanced_particles: 0,
            surviving_particles: 0,
            rejected_observations: 0,
            failed_transitions: 0,
            independent_seed_count: seeds.len(),
            exact_observation_comparison: true,
            units: 0,
            status_reason: String::new(),
        };
        let samples = match prior.sample(initial_info, seeds[0], config.particles)? {
            BeliefSampleResult::Samples(samples) => samples,
            BeliefSampleResult::OutOfSupport(belief) => {
                diagnostics.status_reason = "prior has no compatible particles".into();
                diagnostics.sampled_particles = belief.sampled_count;
                return Ok(ParticleResult {
                    schema_version: PARTICLE_BELIEF_SCHEMA,
                    status: ParticleStatus::OutOfSupport,
                    conditioned_observation: None,
                    diagnostics,
                });
            }
        };
        diagnostics.sampled_particles = samples.hypotheses.len();
        let mut survivors = 0usize;
        for (index, sample) in samples.hypotheses.iter().enumerate() {
            let units = 2u64;
            if diagnostics.units + units > config.max_units {
                diagnostics.status_reason = "particle computation budget exhausted".into();
                break;
            }
            diagnostics.units += units;
            let particle_seed = seeds[index % seeds.len()]
                .wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let world = match PreviewWorld::construct(&dex, initial, sample, particle_seed) {
                Ok(world) => world,
                Err(_) => {
                    diagnostics.failed_transitions += 1;
                    continue;
                }
            };
            diagnostics.constructed_particles += 1;
            let mut episode = match world.into_episode(dex.clone()) {
                Ok(episode) => episode,
                Err(_) => {
                    diagnostics.failed_transitions += 1;
                    continue;
                }
            };
            let preview_boundary = match episode.submit_root_with_baseline(preview_actions.clone())
            {
                Ok(boundary) => boundary,
                Err(_) => {
                    diagnostics.failed_transitions += 1;
                    continue;
                }
            };
            if preview_boundary.root_view.outcome.terminated
                || preview_boundary.root_view.request.kind != crate::actions::RequestKind::Normal
            {
                diagnostics.failed_transitions += 1;
                continue;
            }
            let boundary = match episode.submit_root_with_baseline(combat_actions.clone()) {
                Ok(boundary) => boundary,
                Err(_) => {
                    diagnostics.failed_transitions += 1;
                    continue;
                }
            };
            diagnostics.advanced_particles += 1;
            let observed = SearchInformationSet::from_player_view(&boundary.root_view)?;
            if observed == *target {
                survivors += 1;
            } else {
                diagnostics.rejected_observations += 1;
            }
        }
        diagnostics.surviving_particles = survivors;
        let status = if survivors == 0 {
            diagnostics.status_reason =
                "all particles rejected by exact observation conditioning".into();
            ParticleStatus::ParticleCollapse
        } else if survivors < config.min_surviving_particles {
            diagnostics.status_reason =
                "surviving particle coverage is below the configured minimum".into();
            ParticleStatus::InsufficientCoverage
        } else {
            diagnostics.status_reason =
                "exact observation conditioning retained bounded support".into();
            ParticleStatus::Supported
        };
        Ok(ParticleResult {
            schema_version: PARTICLE_BELIEF_SCHEMA,
            status,
            conditioned_observation: (survivors > 0).then(|| target.clone()),
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        actions::ActionKind,
        assets::Dex,
        state::{BattleState, SideId, Team},
    };
    use std::{path::Path, sync::Arc};

    fn assets() -> (Arc<Dex>, Arc<Vec<Team>>) {
        (
            Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap()),
            Arc::new(serde_json::from_str(include_str!("../../data/training-teams.json")).unwrap()),
        )
    }

    fn complete(request: &crate::actions::Request) -> Vec<AtomicAction> {
        fn descend(
            request: &crate::actions::Request,
            prefix: &mut Vec<AtomicAction>,
        ) -> Option<Vec<AtomicAction>> {
            let candidates = request.candidates(prefix).ok()?;
            if candidates.is_empty() {
                return request
                    .validate_joint(prefix)
                    .is_ok()
                    .then(|| prefix.clone());
            }
            for candidate in candidates {
                prefix.push(candidate);
                if let Some(result) = descend(request, prefix) {
                    return Some(result);
                }
                prefix.pop();
            }
            None
        }
        descend(request, &mut Vec::new()).expect("fixture must have a legal action")
    }

    fn preview() -> Vec<AtomicAction> {
        [0, 1, 4, 5]
            .into_iter()
            .enumerate()
            .map(|(slot, destination)| {
                AtomicAction::select(ActionKind::Pick, slot as u8, destination)
            })
            .collect()
    }

    fn run_for_side(
        side: SideId,
    ) -> (
        TrustedSearchInformationSet,
        FrozenOpponentPrior,
        Vec<AtomicAction>,
        Vec<AtomicAction>,
        TrustedSearchInformationSet,
    ) {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [701, 702, 703, 704], [0, 1]).unwrap();
        let initial = TrustedSearchInformationSet::from_player_view(&state.observe(side)).unwrap();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let seed = 0xabcde;
        let BeliefSampleResult::Samples(samples) =
            prior.sample(initial.as_info(), seed, 1).unwrap()
        else {
            panic!("fixture has support")
        };
        let world = PreviewWorld::construct(&dex, &initial, &samples.hypotheses[0], seed).unwrap();
        let mut episode = world.into_episode(dex).unwrap();
        let preview_actions = preview();
        let preview_boundary = episode
            .submit_root_with_baseline(preview_actions.clone())
            .unwrap();
        let combat_actions = complete(&preview_boundary.root_view.request);
        let target = episode
            .submit_root_with_baseline(combat_actions.clone())
            .unwrap();
        let target = TrustedSearchInformationSet::from_player_view(&target.root_view).unwrap();
        (initial, prior, preview_actions, combat_actions, target)
    }

    #[test]
    fn exact_conditioning_replays_for_both_root_perspectives() {
        for side in [SideId::P1, SideId::P2] {
            let (initial, prior, preview_actions, combat_actions, target) = run_for_side(side);
            let (dex, _) = assets();
            let result = ParticleBelief::advance_preview_and_one_turn(
                dex,
                &initial,
                &prior,
                preview_actions,
                combat_actions,
                &target,
                &[0xabcde],
                ParticleConfig {
                    particles: 1,
                    min_surviving_particles: 1,
                    max_units: 64,
                },
            )
            .unwrap();
            assert_eq!(result.status, ParticleStatus::Supported);
            assert_eq!(result.diagnostics.surviving_particles, 1);
            assert!(result.conditioned_observation.is_some());
        }
    }

    #[test]
    fn exact_conditioning_reports_particle_collapse_without_repair() {
        let (initial, prior, preview_actions, combat_actions, _) = run_for_side(SideId::P1);
        let (dex, teams) = assets();
        let mut view =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [701, 702, 703, 704], [0, 1])
                .unwrap()
                .observe(SideId::P1);
        view.turn = 99;
        let target = TrustedSearchInformationSet::from_player_view(&view).unwrap();
        let result = ParticleBelief::advance_preview_and_one_turn(
            dex,
            &initial,
            &prior,
            preview_actions,
            combat_actions,
            &target,
            &[0xabcde],
            ParticleConfig {
                particles: 1,
                min_surviving_particles: 1,
                max_units: 64,
            },
        )
        .unwrap();
        assert_eq!(result.status, ParticleStatus::ParticleCollapse);
        assert_eq!(result.diagnostics.surviving_particles, 0);
        assert!(result.conditioned_observation.is_none());
    }

    #[test]
    fn hidden_truth_metamorphic_inputs_produce_identical_particle_results() {
        let (_, prior, preview_actions, combat_actions, target) = run_for_side(SideId::P1);
        let (dex, teams) = assets();
        let baseline =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [711, 712, 713, 714], [0, 1]).unwrap();
        let mut hidden = teams[1].clone();
        hidden.id = "different-private-truth".into();
        for member in &mut hidden.members {
            member.points = [1, 0, 0, 0, 0, 0];
            member.ivs = [0; 6];
        }
        let altered =
            BattleState::reset(&dex, [&teams[0], &hidden], [711, 712, 713, 714], [0, 1]).unwrap();
        let first =
            TrustedSearchInformationSet::from_player_view(&baseline.observe(SideId::P1)).unwrap();
        let second =
            TrustedSearchInformationSet::from_player_view(&altered.observe(SideId::P1)).unwrap();
        assert_eq!(first.as_info(), second.as_info());
        let (dex_a, _) = assets();
        let a = ParticleBelief::advance_preview_and_one_turn(
            dex_a,
            &first,
            &prior,
            preview_actions.clone(),
            combat_actions.clone(),
            &target,
            &[0xabcde],
            ParticleConfig {
                particles: 1,
                min_surviving_particles: 1,
                max_units: 64,
            },
        )
        .unwrap();
        let (dex_b, _) = assets();
        let b = ParticleBelief::advance_preview_and_one_turn(
            dex_b,
            &second,
            &prior,
            preview_actions,
            combat_actions,
            &target,
            &[0xabcde],
            ParticleConfig {
                particles: 1,
                min_surviving_particles: 1,
                max_units: 64,
            },
        )
        .unwrap();
        assert_eq!(a, b);
    }
}
