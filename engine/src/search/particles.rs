//! Bounded, forward-only particle feasibility prototype.
//!
//! This module deliberately starts from a clean-preview player observation.
//! Particles own their private synthetic worlds and are conditioned only on a
//! later trusted `PlayerView` projection. It is not a live-battle
//! reconstruction API, opponent policy, search teacher, or posterior model.

use crate::{
    actions::AtomicAction,
    assets::Dex,
    search::{
        belief::{BeliefSampleResult, FrozenOpponentPrior},
        information_set::{SearchInformationSet, TrustedSearchInformationSet},
        world::PreviewWorld,
    },
    EngineError, Result,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

pub const PARTICLE_BELIEF_SCHEMA: u16 = 1;
pub const HARD_MAX_PARTICLES: usize = 16;
pub const HARD_MAX_PARTICLE_UNITS: u64 = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpponentProposal {
    FirstLegal,
    SeededSampled,
    /// Research-only posterior proposal. It conditions a bounded legal-action
    /// enumeration on already observed root-public events, then still requires
    /// full exact DTO equality before a particle may commit.
    PublicEventConditionedV1,
}

/// Root-visible reasons why an exact player-safe observation comparison did
/// not match. Categories describe only fields already present in the two
/// `SearchInformationSet` DTOs; they never include simulated hidden values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObservationMismatchCategory {
    OwnStateAndHp,
    PublicOpponentKnowledge,
    DisplayedHealth,
    PublicFieldEffects,
    EventHistoryWindow,
    LegalRequest,
    TurnAndPhaseAlignment,
    TerminalOutcome,
    OpponentActionHypothesis,
    OtherPublicKnowledge,
}

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
    BudgetExhausted,
    UnsupportedObservation,
    InvalidRootAction,
    ProposalZeroSupport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticleDiagnostics {
    pub schema_version: u16,
    pub requested_particles: usize,
    pub pre_update_particles: usize,
    pub post_update_particles: usize,
    pub sampled_particles: usize,
    pub constructed_particles: usize,
    pub advanced_particles: usize,
    pub surviving_particles: usize,
    pub rejected_observations: usize,
    pub failed_transitions: usize,
    pub invalid_root_actions: usize,
    pub independent_seed_count: usize,
    pub exact_observation_comparison: bool,
    pub units: u64,
    pub world_construction_attempts: usize,
    pub engine_transitions: usize,
    pub opponent_action_samples: usize,
    pub chance_samples: usize,
    pub observation_comparisons: usize,
    /// Aggregate counts only: no particle's private candidate observation or
    /// hidden action is retained in this root-facing diagnostic.
    pub mismatch_categories: BTreeMap<ObservationMismatchCategory, usize>,
    /// Number of independent private chance trajectories considered. Values
    /// are aggregate counts; RNG state and seeds are never exposed.
    pub distinct_chance_trajectories: usize,
    /// Number of distinct player-safe next-observation DTOs produced before
    /// exact conditioning. This includes both proposal and chance variation.
    pub distinct_public_outcomes: usize,
    pub public_event_proposal_attempts: usize,
    pub public_event_compatible_proposals: usize,
    pub public_event_zero_support_particles: usize,
    pub opponent_proposal: OpponentProposal,
    pub opponent_proposals_generated: usize,
    pub distinct_opponent_proposals: usize,
    pub distinct_hypotheses: usize,
    pub pre_update_distinct_hypotheses: usize,
    pub post_update_distinct_hypotheses: usize,
    pub trajectory_count: usize,
    pub prior_schema: u16,
    pub prior_source_sha256: String,
    pub dex_digest: String,
    pub seed_schedule: Vec<u64>,
    pub budget_exhausted: bool,
    pub status_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticleResult {
    pub schema_version: u16,
    pub status: ParticleStatus,
    pub conditioned_observation: Option<SearchInformationSet>,
    pub diagnostics: ParticleDiagnostics,
}

pub struct PersistentInitialization {
    pub tracker: Option<PersistentParticleBelief>,
    pub result: ParticleResult,
}

/// The comparison contract is intentionally conservative: all current DTO
/// fields are compared exactly, while private world fields are excluded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationComparisonContract {
    pub schema_version: u16,
    pub exact_fields: Vec<String>,
    pub tolerated_fields: Vec<String>,
    pub event_history_fields: Vec<String>,
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
        event_history_fields: vec![
            "public.knowledge.events (exact serialized window; audited, no tolerance)".into(),
        ],
        excluded_private_fields: vec![
            "BattleState".into(),
            "opponent team identifier and private set".into(),
            "pending opponent actions".into(),
            "live RNG state".into(),
        ],
    }
}

struct PrivateParticle {
    episode: crate::search::world::SyntheticEpisode,
    hypothesis_key: Vec<u8>,
    trajectory_seed: u64,
    construction_attempts: usize,
}

/// Persistent player-safe belief tracker. The synthetic episodes and
/// hypothesis keys never cross this module's root-facing result boundary.
pub struct PersistentParticleBelief {
    root_side: crate::state::SideId,
    particles: Vec<PrivateParticle>,
    prior_schema: u16,
    prior_source_sha256: String,
    dex_digest: String,
    seed_schedule: Vec<u64>,
    min_surviving_particles: usize,
    opponent_proposal: OpponentProposal,
    current_observation: SearchInformationSet,
    #[cfg(test)]
    fault_after_rejections_for_test: Option<usize>,
}

pub struct ParticleBelief;

fn diagnostics_seeded(
    requested_particles: usize,
    seeds: &[u64],
    prior: &FrozenOpponentPrior,
    dex: &Dex,
) -> ParticleDiagnostics {
    ParticleDiagnostics {
        schema_version: PARTICLE_BELIEF_SCHEMA,
        requested_particles,
        pre_update_particles: 0,
        post_update_particles: 0,
        sampled_particles: 0,
        constructed_particles: 0,
        advanced_particles: 0,
        surviving_particles: 0,
        rejected_observations: 0,
        failed_transitions: 0,
        invalid_root_actions: 0,
        independent_seed_count: seeds.len(),
        exact_observation_comparison: true,
        units: 0,
        world_construction_attempts: 0,
        engine_transitions: 0,
        opponent_action_samples: 0,
        chance_samples: 0,
        observation_comparisons: 0,
        mismatch_categories: BTreeMap::new(),
        distinct_chance_trajectories: 0,
        distinct_public_outcomes: 0,
        public_event_proposal_attempts: 0,
        public_event_compatible_proposals: 0,
        public_event_zero_support_particles: 0,
        opponent_proposal: OpponentProposal::FirstLegal,
        opponent_proposals_generated: 0,
        distinct_opponent_proposals: 0,
        distinct_hypotheses: 0,
        pre_update_distinct_hypotheses: 0,
        post_update_distinct_hypotheses: 0,
        trajectory_count: 0,
        prior_schema: prior.manifest().schema_version,
        prior_source_sha256: prior.manifest().source_sha256.clone(),
        dex_digest: dex.asset_digest.clone(),
        seed_schedule: seeds.to_vec(),
        budget_exhausted: false,
        status_reason: String::new(),
    }
}

fn restore_particles(
    particles: &mut [PrivateParticle],
    checkpoints: &[crate::search::world::EpisodeCheckpoint],
) -> Result<()> {
    if particles.len() != checkpoints.len() {
        return Err(EngineError::InvalidInput(
            "particle checkpoint cardinality changed during update".into(),
        ));
    }
    for (particle, checkpoint) in particles.iter_mut().zip(checkpoints.iter().cloned()) {
        particle
            .episode
            .restore_checkpoint(checkpoint)
            .map_err(|error| EngineError::InvalidInput(error.to_string()))?;
    }
    Ok(())
}

/// Proposal randomness belongs to the coordinator, not the private battle RNG.
/// The turn/boundary mix prevents a repeated action sample on successive
/// observations of the same trajectory without consulting actual-world state.
fn proposal_seed(trajectory_seed: u64, turn: u16, boundaries: usize) -> u64 {
    trajectory_seed
        .wrapping_add(u64::from(turn).wrapping_mul(0xd1b5_4a32_d192_ed03))
        .wrapping_add((boundaries as u64).wrapping_mul(0x94d0_49bb_1331_11eb))
        ^ 0xa5a5_5a5a_9e37_79b9
}

fn classify_observation_mismatch(
    expected: &SearchInformationSet,
    observed: &SearchInformationSet,
) -> Vec<ObservationMismatchCategory> {
    use ObservationMismatchCategory as Category;

    let mut categories = std::collections::BTreeSet::new();
    if expected.own.state != observed.own.state {
        categories.insert(Category::OwnStateAndHp);
    }
    if expected.public.turn != observed.public.turn
        || expected.own.request.kind != observed.own.request.kind
    {
        categories.insert(Category::TurnAndPhaseAlignment);
    }
    if expected.own.request != observed.own.request {
        categories.insert(Category::LegalRequest);
    }
    if expected.public.outcome != observed.public.outcome {
        categories.insert(Category::TerminalOutcome);
    }

    let expected_knowledge = &expected.public.knowledge;
    let observed_knowledge = &observed.public.knowledge;
    if expected_knowledge
        .pokemon
        .iter()
        .zip(&observed_knowledge.pokemon)
        .any(|(left, right)| left.health != right.health)
    {
        categories.insert(Category::DisplayedHealth);
    }
    let opponent_knowledge_differs = (6..12).any(|index| {
        let mut left = expected_knowledge.pokemon[index].clone();
        let mut right = observed_knowledge.pokemon[index].clone();
        left.health = Default::default();
        right.health = Default::default();
        left != right
    });
    if opponent_knowledge_differs {
        categories.insert(Category::PublicOpponentKnowledge);
    }
    if expected_knowledge.field != observed_knowledge.field
        || expected_knowledge.sides != observed_knowledge.sides
    {
        categories.insert(Category::PublicFieldEffects);
    }
    if expected_knowledge.events != observed_knowledge.events {
        categories.insert(Category::EventHistoryWindow);
        let has_public_opponent_move = expected_knowledge
            .events
            .iter()
            .chain(observed_knowledge.events.iter())
            .any(|event| event.subject >= 6 && event.kind == crate::knowledge::EventKind::Move);
        if has_public_opponent_move {
            categories.insert(Category::OpponentActionHypothesis);
        }
    }
    if expected_knowledge != observed_knowledge
        && categories.is_disjoint(&std::collections::BTreeSet::from([
            Category::DisplayedHealth,
            Category::PublicOpponentKnowledge,
            Category::PublicFieldEffects,
            Category::EventHistoryWindow,
        ]))
    {
        categories.insert(Category::OtherPublicKnowledge);
    }
    if categories.is_empty() && expected != observed {
        // Schema is currently fixed, so this is a guard for future DTO fields.
        categories.insert(Category::OtherPublicKnowledge);
    }
    categories.into_iter().collect()
}

impl PersistentParticleBelief {
    /// Construct private trajectories from a clean-preview observation. The
    /// prior sample may contain fewer distinct sheets than trajectories; any
    /// repeated sheet is tracked as an independent seeded trajectory.
    pub fn initialize(
        dex: Arc<Dex>,
        initial: &TrustedSearchInformationSet,
        prior: &FrozenOpponentPrior,
        seeds: &[u64],
        config: ParticleConfig,
    ) -> Result<PersistentInitialization> {
        Self::initialize_with_proposal(
            dex,
            initial,
            prior,
            seeds,
            config,
            OpponentProposal::FirstLegal,
        )
    }

    pub fn initialize_with_proposal(
        dex: Arc<Dex>,
        initial: &TrustedSearchInformationSet,
        prior: &FrozenOpponentPrior,
        seeds: &[u64],
        config: ParticleConfig,
        opponent_proposal: OpponentProposal,
    ) -> Result<PersistentInitialization> {
        config.validate()?;
        if seeds.is_empty() {
            return Err(EngineError::InvalidInput(
                "at least one particle seed is required".into(),
            ));
        }
        let initial_info = initial.as_info();
        if initial_info.own.request.kind != crate::actions::RequestKind::Preview
            || initial_info.public.turn != 0
        {
            return Err(EngineError::InvalidInput(
                "persistent particles require a clean preview root".into(),
            ));
        }
        let mut diagnostics = diagnostics_seeded(config.particles, seeds, prior, &dex);
        diagnostics.opponent_proposal = opponent_proposal;
        let samples = match prior.sample(initial_info, seeds[0], config.particles)? {
            BeliefSampleResult::Samples(samples) => samples,
            BeliefSampleResult::OutOfSupport(belief) => {
                diagnostics.sampled_particles = belief.sampled_count;
                diagnostics.status_reason = "prior has no compatible particles".into();
                return Ok(PersistentInitialization {
                    tracker: None,
                    result: ParticleResult {
                        schema_version: PARTICLE_BELIEF_SCHEMA,
                        status: ParticleStatus::OutOfSupport,
                        conditioned_observation: None,
                        diagnostics,
                    },
                });
            }
        };
        diagnostics.sampled_particles = samples.hypotheses.len();
        diagnostics.units = 1; // prior sampling
        if samples.hypotheses.is_empty() {
            return Err(EngineError::InvalidInput(
                "persistent particle sample is empty".into(),
            ));
        }
        let mut particles = Vec::with_capacity(config.particles);
        let mut hypothesis_keys = std::collections::BTreeSet::new();
        for index in 0..config.particles {
            let sample = &samples.hypotheses[index % samples.hypotheses.len()];
            let hypothesis_key = serde_json::to_vec(&sample.members).map_err(EngineError::Json)?;
            hypothesis_keys.insert(hypothesis_key.clone());
            let trajectory_seed = seeds[index % seeds.len()]
                .wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let world = PreviewWorld::construct(&dex, initial, sample, trajectory_seed).map_err(
                |error| EngineError::InvalidInput(format!("particle world construction: {error}")),
            )?;
            let construction_attempts = world.diagnostics().attempts;
            diagnostics.world_construction_attempts += construction_attempts;
            diagnostics.units = diagnostics
                .units
                .saturating_add(1 + construction_attempts as u64);
            if diagnostics.units > config.max_units {
                diagnostics.budget_exhausted = true;
                diagnostics.status_reason =
                    "particle initialization computation budget exhausted".into();
                return Ok(PersistentInitialization {
                    tracker: None,
                    result: ParticleResult {
                        schema_version: PARTICLE_BELIEF_SCHEMA,
                        status: ParticleStatus::BudgetExhausted,
                        conditioned_observation: None,
                        diagnostics,
                    },
                });
            }
            diagnostics.constructed_particles += 1;
            particles.push(PrivateParticle {
                episode: world.into_episode(dex.clone()).map_err(|error| {
                    EngineError::InvalidInput(format!("particle episode construction: {error}"))
                })?,
                hypothesis_key,
                trajectory_seed,
                construction_attempts,
            });
        }
        diagnostics.distinct_hypotheses = hypothesis_keys.len();
        diagnostics.pre_update_particles = particles.len();
        diagnostics.post_update_particles = particles.len();
        diagnostics.pre_update_distinct_hypotheses = hypothesis_keys.len();
        diagnostics.post_update_distinct_hypotheses = hypothesis_keys.len();
        diagnostics.trajectory_count = particles.len();
        diagnostics.status_reason = "persistent clean-preview support initialized".into();
        let tracker = Self {
            root_side: initial_info.own.side,
            particles,
            prior_schema: prior.manifest().schema_version,
            prior_source_sha256: prior.manifest().source_sha256.clone(),
            dex_digest: prior.manifest().dex_digest.clone(),
            seed_schedule: seeds.to_vec(),
            min_surviving_particles: config.min_surviving_particles,
            opponent_proposal,
            current_observation: initial_info.clone(),
            #[cfg(test)]
            fault_after_rejections_for_test: None,
        };
        let result = ParticleResult {
            schema_version: PARTICLE_BELIEF_SCHEMA,
            status: ParticleStatus::Supported,
            conditioned_observation: Some(initial_info.clone()),
            diagnostics,
        };
        Ok(PersistentInitialization {
            tracker: Some(tracker),
            result,
        })
    }

    pub fn current_observation(&self) -> &SearchInformationSet {
        &self.current_observation
    }

    #[cfg(test)]
    fn inject_observation_conversion_fault_after_rejections_for_test(
        &mut self,
        rejected_particles: usize,
    ) {
        self.fault_after_rejections_for_test = Some(rejected_particles);
    }

    #[cfg(test)]
    fn clear_observation_conversion_fault_for_test(&mut self) {
        self.fault_after_rejections_for_test = None;
    }

    /// Advance every private surviving trajectory using the actual root
    /// action and condition on the next trusted root observation.
    pub fn update(
        &mut self,
        root_actions: Vec<AtomicAction>,
        next_observation: &TrustedSearchInformationSet,
        config: ParticleConfig,
    ) -> Result<ParticleResult> {
        self.update_with_proposal(
            root_actions,
            next_observation,
            config,
            self.opponent_proposal,
        )
    }

    pub fn update_with_proposal(
        &mut self,
        root_actions: Vec<AtomicAction>,
        next_observation: &TrustedSearchInformationSet,
        config: ParticleConfig,
        opponent_proposal: OpponentProposal,
    ) -> Result<ParticleResult> {
        config.validate()?;
        if config.min_surviving_particles != self.min_surviving_particles {
            return Err(EngineError::InvalidInput(
                "persistent tracker minimum-survival configuration cannot change".into(),
            ));
        }
        let target = next_observation.as_info();
        if target.own.side != self.root_side
            || target.public.turn < self.current_observation.public.turn
        {
            let mut diagnostics = ParticleDiagnostics {
                schema_version: PARTICLE_BELIEF_SCHEMA,
                requested_particles: self.particles.len(),
                pre_update_particles: self.particles.len(),
                post_update_particles: 0,
                sampled_particles: self.particles.len(),
                constructed_particles: self.particles.len(),
                advanced_particles: 0,
                surviving_particles: 0,
                rejected_observations: 0,
                failed_transitions: 0,
                invalid_root_actions: 0,
                independent_seed_count: self.seed_schedule.len(),
                exact_observation_comparison: true,
                units: 0,
                world_construction_attempts: 0,
                engine_transitions: 0,
                opponent_action_samples: 0,
                chance_samples: 0,
                observation_comparisons: 0,
                mismatch_categories: BTreeMap::new(),
                distinct_chance_trajectories: 0,
                distinct_public_outcomes: 0,
                public_event_proposal_attempts: 0,
                public_event_compatible_proposals: 0,
                public_event_zero_support_particles: 0,
                opponent_proposal,
                opponent_proposals_generated: 0,
                distinct_opponent_proposals: 0,
                distinct_hypotheses: 0,
                pre_update_distinct_hypotheses: 0,
                post_update_distinct_hypotheses: 0,
                trajectory_count: self.particles.len(),
                prior_schema: self.prior_schema,
                prior_source_sha256: self.prior_source_sha256.clone(),
                dex_digest: self.dex_digest.clone(),
                seed_schedule: self.seed_schedule.clone(),
                budget_exhausted: false,
                status_reason: String::new(),
            };
            diagnostics.status_reason = "next observation is unsupported for this tracker".into();
            return Ok(ParticleResult {
                schema_version: PARTICLE_BELIEF_SCHEMA,
                status: ParticleStatus::UnsupportedObservation,
                conditioned_observation: None,
                diagnostics,
            });
        }
        let mut diagnostics = ParticleDiagnostics {
            schema_version: PARTICLE_BELIEF_SCHEMA,
            requested_particles: self.particles.len(),
            pre_update_particles: self.particles.len(),
            post_update_particles: 0,
            sampled_particles: self.particles.len(),
            constructed_particles: self.particles.len(),
            advanced_particles: 0,
            surviving_particles: 0,
            rejected_observations: 0,
            failed_transitions: 0,
            invalid_root_actions: 0,
            independent_seed_count: self.seed_schedule.len(),
            exact_observation_comparison: true,
            units: 0,
            world_construction_attempts: self
                .particles
                .iter()
                .map(|p| p.construction_attempts)
                .sum(),
            engine_transitions: 0,
            opponent_action_samples: 0,
            chance_samples: 0,
            observation_comparisons: 0,
            mismatch_categories: BTreeMap::new(),
            distinct_chance_trajectories: 0,
            distinct_public_outcomes: 0,
            public_event_proposal_attempts: 0,
            public_event_compatible_proposals: 0,
            public_event_zero_support_particles: 0,
            opponent_proposal,
            opponent_proposals_generated: 0,
            distinct_opponent_proposals: 0,
            distinct_hypotheses: self
                .particles
                .iter()
                .map(|p| p.hypothesis_key.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            pre_update_distinct_hypotheses: self
                .particles
                .iter()
                .map(|p| p.hypothesis_key.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            post_update_distinct_hypotheses: 0,
            trajectory_count: self.particles.len(),
            prior_schema: self.prior_schema,
            prior_source_sha256: self.prior_source_sha256.clone(),
            dex_digest: self.dex_digest.clone(),
            seed_schedule: self.seed_schedule.clone(),
            budget_exhausted: false,
            status_reason: String::new(),
        };
        let units_per_particle = match opponent_proposal {
            OpponentProposal::PublicEventConditionedV1 => {
                5 + (2 * crate::search::world::MAX_PUBLIC_EVENT_PROPOSAL_ATTEMPTS as u64)
            }
            OpponentProposal::FirstLegal | OpponentProposal::SeededSampled => 5,
        };
        let required_units = (self.particles.len() as u64).saturating_mul(units_per_particle);
        if required_units > config.max_units {
            diagnostics.budget_exhausted = true;
            diagnostics.status_reason = "particle update computation budget exhausted".into();
            return Ok(ParticleResult {
                schema_version: PARTICLE_BELIEF_SCHEMA,
                status: ParticleStatus::BudgetExhausted,
                conditioned_observation: None,
                diagnostics,
            });
        }
        for particle in &self.particles {
            if particle
                .episode
                .validate_root_actions(&root_actions)
                .is_err()
            {
                diagnostics.invalid_root_actions = 1;
                diagnostics.status_reason =
                    "root action is invalid for at least one particle".into();
                return Ok(ParticleResult {
                    schema_version: PARTICLE_BELIEF_SCHEMA,
                    status: ParticleStatus::InvalidRootAction,
                    conditioned_observation: None,
                    diagnostics,
                });
            }
        }
        let checkpoints = self
            .particles
            .iter()
            .map(|particle| {
                particle
                    .episode
                    .checkpoint()
                    .map_err(|error| EngineError::InvalidInput(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        // Do not remove a rejected trajectory until every particle completed
        // its transition and comparison. This makes unexpected internal errors
        // fully transactional, including particles rejected before the error.
        let mut retain = vec![false; self.particles.len()];
        let mut proposal_keys = std::collections::BTreeSet::new();
        let mut chance_trajectory_keys = std::collections::BTreeSet::new();
        let mut public_outcome_keys = std::collections::BTreeSet::new();
        for (particle_index, retained) in retain.iter_mut().enumerate() {
            diagnostics.units += 5;
            let proposal_seed = proposal_seed(
                self.particles[particle_index].trajectory_seed,
                self.current_observation.public.turn,
                self.particles[particle_index].episode.boundaries(),
            );
            let proposal_actions = match opponent_proposal {
                OpponentProposal::PublicEventConditionedV1 => {
                    let proposal = match self.particles[particle_index]
                        .episode
                        .public_event_conditioned_opponent_action(
                            &root_actions,
                            target,
                            proposal_seed,
                        ) {
                        Ok(proposal) => proposal,
                        Err(error) => {
                            restore_particles(&mut self.particles, &checkpoints)?;
                            return Err(EngineError::InvalidInput(format!(
                                "public-event proposal transition failed: {error}"
                            )));
                        }
                    };
                    diagnostics.units += (2 * proposal.attempts) as u64;
                    diagnostics.engine_transitions += 2 * proposal.attempts;
                    diagnostics.public_event_proposal_attempts += proposal.attempts;
                    diagnostics.public_event_compatible_proposals += proposal.compatible;
                    match proposal.actions {
                        Some(actions) => actions,
                        None => {
                            diagnostics.public_event_zero_support_particles += 1;
                            continue;
                        }
                    }
                }
                OpponentProposal::FirstLegal | OpponentProposal::SeededSampled => self.particles
                    [particle_index]
                    .episode
                    .proposed_opponent_action(opponent_proposal, proposal_seed)
                    .ok_or_else(|| {
                        EngineError::InvalidInput(
                            "particle opponent proposal has no legal completion".into(),
                        )
                    })?,
            };
            diagnostics.opponent_proposals_generated += 1;
            if let Ok(key) = serde_json::to_vec(&proposal_actions) {
                proposal_keys.insert(key);
            }
            let boundary = match self.particles[particle_index]
                .episode
                .submit_root_with_opponent_actions(root_actions.clone(), proposal_actions)
            {
                Ok(boundary) => boundary,
                Err(error) => {
                    restore_particles(&mut self.particles, &checkpoints)?;
                    return Err(EngineError::InvalidInput(format!(
                        "particle operational transition failed: {error}"
                    )));
                }
            };
            diagnostics.advanced_particles += 1;
            diagnostics.engine_transitions += 2;
            diagnostics.opponent_action_samples += 1;
            diagnostics.chance_samples += 1;
            chance_trajectory_keys.insert(self.particles[particle_index].trajectory_seed);
            let observed = match SearchInformationSet::from_player_view(&boundary.root_view) {
                Ok(observed) => observed,
                Err(error) => {
                    restore_particles(&mut self.particles, &checkpoints)?;
                    return Err(error);
                }
            };
            diagnostics.observation_comparisons += 1;
            if let Ok(key) = observed.to_bytes() {
                public_outcome_keys.insert(key);
            }
            if observed == *target {
                *retained = true;
            } else {
                diagnostics.rejected_observations += 1;
                for category in classify_observation_mismatch(target, &observed) {
                    *diagnostics.mismatch_categories.entry(category).or_default() += 1;
                }
            }
            #[cfg(test)]
            if self
                .fault_after_rejections_for_test
                .is_some_and(|required| {
                    particle_index > 0 && diagnostics.rejected_observations >= required
                })
            {
                restore_particles(&mut self.particles, &checkpoints)?;
                return Err(EngineError::InvalidInput(
                    "controlled test observation-conversion fault".into(),
                ));
            }
        }
        diagnostics.distinct_opponent_proposals = proposal_keys.len();
        diagnostics.distinct_chance_trajectories = chance_trajectory_keys.len();
        diagnostics.distinct_public_outcomes = public_outcome_keys.len();
        diagnostics.surviving_particles = retain.iter().filter(|keep| **keep).count();
        diagnostics.post_update_particles = diagnostics.surviving_particles;
        diagnostics.post_update_distinct_hypotheses = self
            .particles
            .iter()
            .zip(&retain)
            .filter(|(_, keep)| **keep)
            .map(|(particle, _)| particle.hypothesis_key.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        diagnostics.distinct_hypotheses = diagnostics.post_update_distinct_hypotheses;
        let status = if diagnostics.budget_exhausted {
            diagnostics.status_reason = "particle update computation budget exhausted".into();
            ParticleStatus::BudgetExhausted
        } else if diagnostics.public_event_zero_support_particles > 0
            && diagnostics.advanced_particles == 0
        {
            diagnostics.status_reason =
                "public-event-conditioned proposal found no compatible legal completion".into();
            ParticleStatus::ProposalZeroSupport
        } else if diagnostics.advanced_particles > 0 && diagnostics.surviving_particles == 0 {
            diagnostics.status_reason =
                "all advanced particles rejected by exact observation conditioning".into();
            ParticleStatus::ParticleCollapse
        } else if diagnostics.surviving_particles < self.min_surviving_particles {
            diagnostics.status_reason =
                "surviving particle coverage is below the configured minimum".into();
            ParticleStatus::InsufficientCoverage
        } else {
            diagnostics.status_reason = "persistent exact-support update retained particles".into();
            ParticleStatus::Supported
        };
        self.particles = std::mem::take(&mut self.particles)
            .into_iter()
            .zip(retain)
            .filter_map(|(particle, keep)| keep.then_some(particle))
            .collect();
        if !self.particles.is_empty() {
            self.current_observation = target.clone();
        }
        Ok(ParticleResult {
            schema_version: PARTICLE_BELIEF_SCHEMA,
            status,
            conditioned_observation: (!self.particles.is_empty()).then(|| target.clone()),
            diagnostics,
        })
    }
}

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

        let mut diagnostics = diagnostics_seeded(config.particles, seeds, prior, &dex);
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
                diagnostics.budget_exhausted = true;
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
        let status = if diagnostics.budget_exhausted {
            diagnostics.status_reason = "particle computation budget exhausted".into();
            ParticleStatus::BudgetExhausted
        } else if survivors == 0 {
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
    use std::{path::Path, sync::Arc, time::Instant};

    fn assets() -> (Arc<Dex>, Arc<Vec<Team>>) {
        (
            Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap()),
            Arc::new(serde_json::from_str(include_str!("../../data/training-teams.json")).unwrap()),
        )
    }

    fn legal_held_out_variation(dex: &Dex, team: &Team) -> Team {
        for member_index in 0..team.members.len() {
            for donor in 0..6 {
                for recipient in 0..6 {
                    if donor == recipient
                        || team.members[member_index].points[donor] == 0
                        || team.members[member_index].points[recipient] >= 32
                    {
                        continue;
                    }
                    let mut variant = team.clone();
                    variant.id = "held-out-research-fixture".into();
                    variant.members[member_index].points[donor] -= 1;
                    variant.members[member_index].points[recipient] += 1;
                    if crate::legality::validate_team(dex, &variant).legal() {
                        return variant;
                    }
                }
            }
        }
        panic!("fixture must admit a legal held-out point-allocation variation");
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

    fn complete_last(request: &crate::actions::Request) -> Vec<AtomicAction> {
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
            for candidate in candidates.into_iter().rev() {
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

    type TwoTurnSequence = (
        Arc<Dex>,
        TrustedSearchInformationSet,
        FrozenOpponentPrior,
        Vec<AtomicAction>,
        TrustedSearchInformationSet,
        Vec<AtomicAction>,
        TrustedSearchInformationSet,
        Vec<AtomicAction>,
        TrustedSearchInformationSet,
    );

    fn two_turn_sequence(side: SideId) -> TwoTurnSequence {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [721, 722, 723, 724], [0, 1]).unwrap();
        let initial = TrustedSearchInformationSet::from_player_view(&state.observe(side)).unwrap();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let seed = 0xdefaced;
        let BeliefSampleResult::Samples(samples) =
            prior.sample(initial.as_info(), seed, 1).unwrap()
        else {
            panic!("fixture has support")
        };
        let world = PreviewWorld::construct(&dex, &initial, &samples.hypotheses[0], seed).unwrap();
        let mut episode = world.into_episode(dex.clone()).unwrap();
        let preview_actions = preview();
        let after_preview = episode
            .submit_root_with_baseline(preview_actions.clone())
            .unwrap();
        let target_preview =
            TrustedSearchInformationSet::from_player_view(&after_preview.root_view).unwrap();
        let combat_one = complete_last(&after_preview.root_view.request);
        let after_one = episode
            .submit_root_with_baseline(combat_one.clone())
            .unwrap();
        let target_one =
            TrustedSearchInformationSet::from_player_view(&after_one.root_view).unwrap();
        let combat_two = complete_last(&after_one.root_view.request);
        let after_two = episode
            .submit_root_with_baseline(combat_two.clone())
            .unwrap();
        let target_two =
            TrustedSearchInformationSet::from_player_view(&after_two.root_view).unwrap();
        (
            dex,
            initial,
            prior,
            preview_actions,
            target_preview,
            combat_one,
            target_one,
            combat_two,
            target_two,
        )
    }

    type IndependentSequence = (
        TrustedSearchInformationSet,
        Vec<AtomicAction>,
        TrustedSearchInformationSet,
        Vec<AtomicAction>,
        TrustedSearchInformationSet,
        Vec<AtomicAction>,
        TrustedSearchInformationSet,
    );

    fn independent_actual_sequence(
        dex: &Dex,
        teams: &[Team],
        side: SideId,
        seed_base: u16,
        out_of_prior: bool,
    ) -> IndependentSequence {
        let mut actual_teams = [teams[0].clone(), teams[1].clone()];
        if out_of_prior {
            let opponent_index = match side {
                SideId::P1 => 1,
                SideId::P2 => 0,
            };
            actual_teams[opponent_index].id = format!("independent-hidden-{seed_base}");
            for member in &mut actual_teams[opponent_index].members {
                member.points = [1, 0, 0, 0, 0, 0];
                member.ivs = [0; 6];
            }
        }
        execute_actual_sequence(dex, actual_teams, side, seed_base)
    }

    /// C3b-1.3's paired study needs a legal held-out truth.  Keep this separate
    /// from the frozen C3b-1.2 diagnosis fixture above, whose historical rows
    /// intentionally use its original synthetic mutation.
    fn legal_independent_actual_sequence(
        dex: &Dex,
        teams: &[Team],
        side: SideId,
        seed_base: u16,
        out_of_prior: bool,
    ) -> IndependentSequence {
        let mut actual_teams = [teams[0].clone(), teams[1].clone()];
        if out_of_prior {
            let opponent_index = match side {
                SideId::P1 => 1,
                SideId::P2 => 0,
            };
            actual_teams[opponent_index] =
                legal_held_out_variation(dex, &actual_teams[opponent_index]);
            actual_teams[opponent_index].id = format!("independent-hidden-{seed_base}");
        }
        assert!(crate::legality::validate_team(dex, &actual_teams[0]).legal());
        assert!(crate::legality::validate_team(dex, &actual_teams[1]).legal());
        execute_actual_sequence(dex, actual_teams, side, seed_base)
    }

    fn execute_actual_sequence(
        dex: &Dex,
        actual_teams: [Team; 2],
        side: SideId,
        seed_base: u16,
    ) -> IndependentSequence {
        let mut actual = BattleState::reset(
            dex,
            [&actual_teams[0], &actual_teams[1]],
            [seed_base, seed_base + 1, seed_base + 2, seed_base + 3],
            [0, 1],
        )
        .unwrap();
        let initial = TrustedSearchInformationSet::from_player_view(&actual.observe(side)).unwrap();
        let preview_actions = preview();
        actual.step(dex, side, &preview_actions).unwrap();
        actual.step(dex, side.other(), &preview()).unwrap();
        let target_preview =
            TrustedSearchInformationSet::from_player_view(&actual.observe(side)).unwrap();
        let combat_one = complete_last(&actual.observe(side).request);
        let opponent_one = complete(&actual.observe(side.other()).request);
        actual.step(dex, side, &combat_one).unwrap();
        actual.step(dex, side.other(), &opponent_one).unwrap();
        let target_one =
            TrustedSearchInformationSet::from_player_view(&actual.observe(side)).unwrap();
        let combat_two = complete_last(&actual.observe(side).request);
        let opponent_two = complete(&actual.observe(side.other()).request);
        actual.step(dex, side, &combat_two).unwrap();
        actual.step(dex, side.other(), &opponent_two).unwrap();
        let target_two =
            TrustedSearchInformationSet::from_player_view(&actual.observe(side)).unwrap();
        (
            initial,
            preview_actions,
            target_preview,
            combat_one,
            target_one,
            combat_two,
            target_two,
        )
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

    #[test]
    fn persistent_tracker_advances_two_successive_boundaries_for_both_roots() {
        for side in [SideId::P1, SideId::P2] {
            let (
                dex,
                initial,
                prior,
                preview_actions,
                target_preview,
                combat_one,
                target_one,
                combat_two,
                target_two,
            ) = two_turn_sequence(side);
            let initialized = PersistentParticleBelief::initialize(
                dex,
                &initial,
                &prior,
                &[0xdefaced, 0x123456],
                ParticleConfig {
                    particles: 4,
                    min_surviving_particles: 1,
                    max_units: 2_000,
                },
            )
            .unwrap();
            let result = initialized.result;
            let mut tracker = initialized.tracker.expect("supported fixture");
            assert_eq!(result.status, ParticleStatus::Supported);
            assert_eq!(result.diagnostics.trajectory_count, 4);
            assert!(result.diagnostics.distinct_hypotheses >= 1);
            let first = tracker
                .update(
                    preview_actions,
                    &target_preview,
                    ParticleConfig {
                        particles: 4,
                        min_surviving_particles: 1,
                        max_units: 128,
                    },
                )
                .unwrap();
            assert!(matches!(first.status, ParticleStatus::Supported));
            let second = tracker
                .update(
                    combat_one,
                    &target_one,
                    ParticleConfig {
                        particles: 4,
                        min_surviving_particles: 1,
                        max_units: 128,
                    },
                )
                .unwrap();
            assert!(matches!(second.status, ParticleStatus::Supported));
            let third = tracker
                .update(
                    combat_two,
                    &target_two,
                    ParticleConfig {
                        particles: 4,
                        min_surviving_particles: 1,
                        max_units: 128,
                    },
                )
                .unwrap();
            assert!(matches!(third.status, ParticleStatus::Supported));
            assert!(third.diagnostics.engine_transitions >= 2);
            assert_eq!(tracker.current_observation(), target_two.as_info());
        }
    }

    #[test]
    fn persistent_budget_exhaustion_is_not_collapse() {
        let (dex, initial, prior, preview_actions, target_preview, _, _, _, _) =
            two_turn_sequence(SideId::P1);
        let initialized = PersistentParticleBelief::initialize(
            dex,
            &initial,
            &prior,
            &[0xdefaced],
            ParticleConfig {
                particles: 1,
                min_surviving_particles: 1,
                max_units: 2_000,
            },
        )
        .unwrap();
        let mut tracker = initialized.tracker.expect("supported fixture");
        let result = tracker
            .update(
                preview_actions,
                &target_preview,
                ParticleConfig {
                    particles: 1,
                    min_surviving_particles: 1,
                    max_units: 1,
                },
            )
            .unwrap();
        assert_eq!(result.status, ParticleStatus::BudgetExhausted);
        assert!(result.diagnostics.budget_exhausted);
    }

    #[test]
    fn persistent_initialization_reports_out_of_support() {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [731, 732, 733, 734], [0, 1]).unwrap();
        let mut view = state.observe(SideId::P1);
        view.knowledge.pokemon[6].species = u16::MAX;
        let initial = TrustedSearchInformationSet::from_player_view(&view).unwrap();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let result = PersistentParticleBelief::initialize(
            dex,
            &initial,
            &prior,
            &[0xface],
            ParticleConfig::default(),
        )
        .unwrap();
        assert_eq!(result.result.status, ParticleStatus::OutOfSupport);
        assert!(result.tracker.is_none());
    }

    #[test]
    fn bounded_particle_counts_report_requested_trajectories() {
        let (dex, initial, prior, _, _, _, _, _, _) = two_turn_sequence(SideId::P1);
        for particles in [2, 4, 8, 16] {
            let result = PersistentParticleBelief::initialize(
                dex.clone(),
                &initial,
                &prior,
                &[0x1001, 0x1002, 0x1003, 0x1004],
                ParticleConfig {
                    particles,
                    min_surviving_particles: 1,
                    max_units: 2_000,
                },
            )
            .unwrap();
            assert_eq!(result.result.status, ParticleStatus::Supported);
            assert_eq!(result.result.diagnostics.trajectory_count, particles);
            assert_eq!(result.result.diagnostics.constructed_particles, particles);
            assert!(result.tracker.is_some());
        }
    }

    #[test]
    fn sampled_opponent_proposal_is_legal_and_reports_diversity() {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [741, 742, 743, 744], [0, 1]).unwrap();
        let initial =
            TrustedSearchInformationSet::from_player_view(&state.observe(SideId::P1)).unwrap();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let seed = 0xabc123;
        let BeliefSampleResult::Samples(samples) =
            prior.sample(initial.as_info(), seed, 1).unwrap()
        else {
            panic!("fixture has support")
        };
        let mut target_episode =
            PreviewWorld::construct(&dex, &initial, &samples.hypotheses[0], seed)
                .unwrap()
                .into_episode(dex.clone())
                .unwrap();
        let preview_actions = preview();
        let target = target_episode
            .submit_root_with_proposal(
                preview_actions.clone(),
                OpponentProposal::SeededSampled,
                seed,
            )
            .unwrap();
        let target = TrustedSearchInformationSet::from_player_view(&target.root_view).unwrap();
        let initialized = PersistentParticleBelief::initialize_with_proposal(
            dex,
            &initial,
            &prior,
            &[seed, seed.wrapping_add(1)],
            ParticleConfig {
                particles: 4,
                min_surviving_particles: 1,
                max_units: 2_000,
            },
            OpponentProposal::SeededSampled,
        )
        .unwrap();
        let mut tracker = initialized.tracker.expect("supported fixture");
        let result = tracker
            .update(
                preview_actions,
                &target,
                ParticleConfig {
                    particles: 4,
                    min_surviving_particles: 1,
                    max_units: 128,
                },
            )
            .unwrap();
        assert_eq!(
            result.diagnostics.opponent_proposal,
            OpponentProposal::SeededSampled
        );
        assert_eq!(
            result.diagnostics.opponent_proposals_generated,
            result.diagnostics.advanced_particles
        );
        assert!(result.diagnostics.distinct_opponent_proposals >= 1);
    }

    #[test]
    fn public_event_conditioning_retains_exact_replay_without_relaxing_final_match() {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [745, 746, 747, 748], [0, 1]).unwrap();
        let initial =
            TrustedSearchInformationSet::from_player_view(&state.observe(SideId::P1)).unwrap();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let seed = 0xabc456;
        let BeliefSampleResult::Samples(samples) =
            prior.sample(initial.as_info(), seed, 1).unwrap()
        else {
            panic!("fixture has support")
        };
        let mut target_episode =
            PreviewWorld::construct(&dex, &initial, &samples.hypotheses[0], seed)
                .unwrap()
                .into_episode(dex.clone())
                .unwrap();
        let preview_actions = preview();
        let target = target_episode
            .submit_root_with_baseline(preview_actions.clone())
            .unwrap();
        let target = TrustedSearchInformationSet::from_player_view(&target.root_view).unwrap();
        let initialized = PersistentParticleBelief::initialize_with_proposal(
            dex,
            &initial,
            &prior,
            &[seed],
            ParticleConfig {
                particles: 1,
                min_surviving_particles: 1,
                max_units: 2_000,
            },
            OpponentProposal::PublicEventConditionedV1,
        )
        .unwrap();
        let mut tracker = initialized.tracker.expect("supported fixture");
        let result = tracker
            .update_with_proposal(
                preview_actions,
                &target,
                ParticleConfig {
                    particles: 1,
                    min_surviving_particles: 1,
                    max_units: 128,
                },
                OpponentProposal::PublicEventConditionedV1,
            )
            .unwrap();
        assert_eq!(result.status, ParticleStatus::Supported);
        assert!(result.diagnostics.public_event_proposal_attempts > 0);
        assert!(result.diagnostics.public_event_compatible_proposals > 0);
        assert_eq!(
            result.conditioned_observation,
            Some(target.as_info().clone())
        );
    }

    #[test]
    fn public_event_conditioning_rejects_authoritative_public_contradiction() {
        let (dex, initial, prior, preview_actions, target_preview, _, _, _, _) =
            two_turn_sequence(SideId::P1);
        let mut contradictory = target_preview.as_info().clone();
        contradictory.public.knowledge.pokemon[6].species = u16::MAX;
        let contradictory = TrustedSearchInformationSet::from_info_for_test(contradictory);
        let initialized = PersistentParticleBelief::initialize_with_proposal(
            dex,
            &initial,
            &prior,
            &[0xdefaced],
            ParticleConfig {
                particles: 1,
                min_surviving_particles: 1,
                max_units: 2_000,
            },
            OpponentProposal::PublicEventConditionedV1,
        )
        .unwrap();
        let mut tracker = initialized.tracker.expect("supported fixture");
        let before = tracker.current_observation().clone();
        let result = tracker
            .update_with_proposal(
                preview_actions,
                &contradictory,
                ParticleConfig {
                    particles: 1,
                    min_surviving_particles: 1,
                    max_units: 128,
                },
                OpponentProposal::PublicEventConditionedV1,
            )
            .unwrap();
        assert_eq!(result.status, ParticleStatus::ProposalZeroSupport);
        assert_eq!(result.diagnostics.public_event_zero_support_particles, 1);
        assert_eq!(tracker.current_observation(), &before);
    }

    #[test]
    fn public_event_conditioned_proposals_are_hidden_truth_metamorphic() {
        let (dex, teams) = assets();
        let mut hidden = legal_held_out_variation(&dex, &teams[1]);
        hidden.id = "different-hidden-world-never-consumed".into();
        let baseline =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [749, 750, 751, 752], [0, 1]).unwrap();
        let altered =
            BattleState::reset(&dex, [&teams[0], &hidden], [749, 750, 751, 752], [0, 1]).unwrap();
        let first =
            TrustedSearchInformationSet::from_player_view(&baseline.observe(SideId::P1)).unwrap();
        let second =
            TrustedSearchInformationSet::from_player_view(&altered.observe(SideId::P1)).unwrap();
        assert_eq!(first.as_info(), second.as_info());
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let seed = 0xface_1234;
        let BeliefSampleResult::Samples(samples) = prior.sample(first.as_info(), seed, 1).unwrap()
        else {
            panic!("fixture has support")
        };
        let mut target_episode =
            PreviewWorld::construct(&dex, &first, &samples.hypotheses[0], seed)
                .unwrap()
                .into_episode(dex.clone())
                .unwrap();
        let actions = preview();
        let target = target_episode
            .submit_root_with_baseline(actions.clone())
            .unwrap();
        let target = TrustedSearchInformationSet::from_player_view(&target.root_view).unwrap();
        let config = ParticleConfig {
            particles: 1,
            min_surviving_particles: 1,
            max_units: 512,
        };
        let mut left = PersistentParticleBelief::initialize_with_proposal(
            dex.clone(),
            &first,
            &prior,
            &[seed],
            config.clone(),
            OpponentProposal::PublicEventConditionedV1,
        )
        .unwrap()
        .tracker
        .expect("fixture has support");
        let mut right = PersistentParticleBelief::initialize_with_proposal(
            dex,
            &second,
            &prior,
            &[seed],
            config.clone(),
            OpponentProposal::PublicEventConditionedV1,
        )
        .unwrap()
        .tracker
        .expect("fixture has support");
        let left = left
            .update_with_proposal(
                actions.clone(),
                &target,
                config.clone(),
                OpponentProposal::PublicEventConditionedV1,
            )
            .unwrap();
        let right = right
            .update_with_proposal(
                actions,
                &target,
                config,
                OpponentProposal::PublicEventConditionedV1,
            )
            .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn held_out_hidden_build_is_legal_and_not_in_frozen_prior() {
        let (dex, teams) = assets();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let held_out = legal_held_out_variation(&dex, &teams[1]);
        assert!(crate::legality::validate_team(&dex, &held_out).legal());
        assert!(!prior.contains_complete_team_for_test(&held_out));
    }

    #[test]
    fn invalid_root_action_is_rejected_transactionally() {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [751, 752, 753, 754], [0, 1]).unwrap();
        let initial =
            TrustedSearchInformationSet::from_player_view(&state.observe(SideId::P1)).unwrap();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let initialized = PersistentParticleBelief::initialize(
            dex,
            &initial,
            &prior,
            &[0x12345],
            ParticleConfig {
                particles: 1,
                min_surviving_particles: 1,
                max_units: 2_000,
            },
        )
        .unwrap();
        let mut tracker = initialized.tracker.expect("supported fixture");
        let before = tracker.current_observation().clone();
        let result = tracker
            .update(
                vec![AtomicAction::pass(0)],
                &initial,
                ParticleConfig {
                    particles: 1,
                    min_surviving_particles: 1,
                    max_units: 128,
                },
            )
            .unwrap();
        assert_eq!(result.status, ParticleStatus::InvalidRootAction);
        assert_eq!(result.diagnostics.invalid_root_actions, 1);
        assert_eq!(tracker.current_observation(), &before);
    }

    #[test]
    fn later_internal_fault_restores_earlier_rejected_particles_and_order() {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [761, 762, 763, 764], [0, 1]).unwrap();
        let initial =
            TrustedSearchInformationSet::from_player_view(&state.observe(SideId::P1)).unwrap();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let config = ParticleConfig {
            particles: 2,
            min_surviving_particles: 1,
            max_units: 2_000,
        };
        let mut faulted = PersistentParticleBelief::initialize(
            dex.clone(),
            &initial,
            &prior,
            &[0x9001, 0x9002],
            config.clone(),
        )
        .unwrap()
        .tracker
        .expect("fixture has support");
        let mut reference = PersistentParticleBelief::initialize(
            dex,
            &initial,
            &prior,
            &[0x9001, 0x9002],
            config.clone(),
        )
        .unwrap()
        .tracker
        .expect("fixture has support");
        let before = faulted
            .particles
            .iter()
            .map(|particle| {
                (
                    particle.hypothesis_key.clone(),
                    particle.trajectory_seed,
                    particle.episode.checkpoint_bytes_for_test().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        faulted.inject_observation_conversion_fault_after_rejections_for_test(1);
        let error = faulted
            .update(preview(), &initial, config.clone())
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("controlled test observation-conversion fault"));
        let after = faulted
            .particles
            .iter()
            .map(|particle| {
                (
                    particle.hypothesis_key.clone(),
                    particle.trajectory_seed,
                    particle.episode.checkpoint_bytes_for_test().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(after, before);
        assert_eq!(faulted.current_observation(), initial.as_info());

        faulted.clear_observation_conversion_fault_for_test();
        let replay = faulted.update(preview(), &initial, config.clone()).unwrap();
        let expected = reference.update(preview(), &initial, config).unwrap();
        assert_eq!(replay, expected);
        assert!(replay.diagnostics.rejected_observations >= 1);
        assert!(!replay.diagnostics.mismatch_categories.is_empty());
    }

    #[test]
    fn mismatch_classifier_reports_only_player_safe_field_categories() {
        let (dex, teams) = assets();
        let state =
            BattleState::reset(&dex, [&teams[0], &teams[1]], [771, 772, 773, 774], [0, 1]).unwrap();
        let expected =
            TrustedSearchInformationSet::from_player_view(&state.observe(SideId::P1)).unwrap();
        let mut observed = expected.as_info().clone();
        observed.own.state.pokemon[0].hp = observed.own.state.pokemon[0].hp.saturating_sub(1);
        observed.public.knowledge.pokemon[6].health.known = true;
        observed.public.knowledge.pokemon[6].health.value.numerator = 99;
        observed.public.knowledge.pokemon[6].species =
            observed.public.knowledge.pokemon[6].species.wrapping_add(1);
        observed.own.request.kind = crate::actions::RequestKind::Wait;
        let categories = classify_observation_mismatch(expected.as_info(), &observed);
        assert!(categories.contains(&ObservationMismatchCategory::OwnStateAndHp));
        assert!(categories.contains(&ObservationMismatchCategory::DisplayedHealth));
        assert!(categories.contains(&ObservationMismatchCategory::PublicOpponentKnowledge));
        assert!(categories.contains(&ObservationMismatchCategory::LegalRequest));
        assert!(categories.contains(&ObservationMismatchCategory::TurnAndPhaseAlignment));
    }

    #[test]
    fn proposal_seed_changes_by_public_boundary_not_battle_rng() {
        let base = proposal_seed(0x1234, 1, 0);
        assert_ne!(base, proposal_seed(0x1234, 2, 0));
        assert_ne!(base, proposal_seed(0x1234, 1, 1));
        assert_eq!(base, proposal_seed(0x1234, 1, 0));
    }

    #[test]
    fn independent_actual_oracle_is_kept_separate_from_tracker_state() {
        let (dex, teams) = assets();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        for side in [SideId::P1, SideId::P2] {
            for (seed_base, out_of_prior) in [(991u16, false), (995u16, true)] {
                let mut actual_teams = [teams[0].clone(), teams[1].clone()];
                if out_of_prior {
                    let opponent_index = match side {
                        SideId::P1 => 1,
                        SideId::P2 => 0,
                    };
                    actual_teams[opponent_index].id = format!("actual-hidden-{seed_base}");
                    for member in &mut actual_teams[opponent_index].members {
                        member.points = [1, 0, 0, 0, 0, 0];
                        member.ivs = [0; 6];
                    }
                }
                let mut actual = BattleState::reset(
                    &dex,
                    [&actual_teams[0], &actual_teams[1]],
                    [seed_base, seed_base + 1, seed_base + 2, seed_base + 3],
                    [0, 1],
                )
                .unwrap();
                let initial =
                    TrustedSearchInformationSet::from_player_view(&actual.observe(side)).unwrap();
                let preview_actions = preview();
                let opponent_preview = preview();
                actual.step(&dex, SideId::P1, &preview_actions).unwrap();
                actual.step(&dex, SideId::P2, &opponent_preview).unwrap();
                let target_preview =
                    TrustedSearchInformationSet::from_player_view(&actual.observe(side)).unwrap();
                let combat_actions = complete_last(&target_preview.as_info().own.request);
                let opponent_combat = complete(&actual.observe(side.other()).request);
                actual.step(&dex, side, &combat_actions).unwrap();
                actual.step(&dex, side.other(), &opponent_combat).unwrap();
                let target_combat =
                    TrustedSearchInformationSet::from_player_view(&actual.observe(side)).unwrap();
                let initialized = PersistentParticleBelief::initialize(
                    dex.clone(),
                    &initial,
                    &prior,
                    &[0x1234, 0x5678],
                    ParticleConfig {
                        particles: 2,
                        min_surviving_particles: 1,
                        max_units: 2_000,
                    },
                )
                .unwrap();
                let mut tracker = initialized.tracker.expect("supported fixture");
                let preview_result = tracker
                    .update(
                        preview_actions,
                        &target_preview,
                        ParticleConfig {
                            particles: 2,
                            min_surviving_particles: 1,
                            max_units: 128,
                        },
                    )
                    .unwrap();
                let combat_result = tracker
                    .update(
                        combat_actions,
                        &target_combat,
                        ParticleConfig {
                            particles: 2,
                            min_surviving_particles: 1,
                            max_units: 128,
                        },
                    )
                    .unwrap();
                assert!(preview_result.diagnostics.advanced_particles <= 2);
                assert!(combat_result.diagnostics.advanced_particles <= 2);
                assert!(matches!(
                    combat_result.status,
                    ParticleStatus::Supported
                        | ParticleStatus::InsufficientCoverage
                        | ParticleStatus::ParticleCollapse
                ));
            }
        }
    }

    #[test]
    fn bounded_independent_reliability_matrix_records_multi_turn_support() {
        let started = Instant::now();
        let (dex, teams) = assets();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        let mut rows = 0usize;
        let mut preview_supported = 0usize;
        let mut combat_one_supported = 0usize;
        let mut combat_two_supported = 0usize;
        let mut collapsed = 0usize;
        let mut diagnosed_rejections = 0usize;
        let mut attempted_boundaries = 0usize;
        let mut proposal_diverse_boundaries = 0usize;
        let mut chance_diverse_boundaries = 0usize;
        let mut public_outcome_diverse_boundaries = 0usize;
        let mut mismatch_totals = BTreeMap::<ObservationMismatchCategory, usize>::new();
        let mut by_condition = BTreeMap::<(bool, OpponentProposal), [usize; 7]>::new();
        for side in [SideId::P1, SideId::P2] {
            for out_of_prior in [false, true] {
                for proposal in [
                    OpponentProposal::FirstLegal,
                    OpponentProposal::SeededSampled,
                ] {
                    for seed_base in [801u16, 811u16] {
                        for particles in [2usize, 4, 8, 16] {
                            rows += 1;
                            let (
                                initial,
                                preview_actions,
                                target_preview,
                                combat_one,
                                target_one,
                                combat_two,
                                target_two,
                            ) = independent_actual_sequence(
                                &dex,
                                &teams,
                                side,
                                seed_base,
                                out_of_prior,
                            );
                            let config = ParticleConfig {
                                particles,
                                min_surviving_particles: 1,
                                max_units: 2_000,
                            };
                            let initialized = PersistentParticleBelief::initialize_with_proposal(
                                dex.clone(),
                                &initial,
                                &prior,
                                &[0x501, 0x502, 0x503, 0x504],
                                config.clone(),
                                proposal,
                            )
                            .unwrap();
                            let mut tracker = initialized.tracker.expect("fixture has support");
                            let mut results = Vec::new();
                            let collapsed_before = collapsed;
                            let rejections_before = diagnosed_rejections;
                            for (actions, target) in [
                                (preview_actions, target_preview),
                                (combat_one, target_one),
                                (combat_two, target_two),
                            ] {
                                let result = tracker
                                    .update_with_proposal(
                                        actions,
                                        &target,
                                        ParticleConfig {
                                            particles,
                                            min_surviving_particles: 1,
                                            max_units: 128,
                                        },
                                        proposal,
                                    )
                                    .unwrap();
                                diagnosed_rejections += result.diagnostics.rejected_observations;
                                if result.status == ParticleStatus::ParticleCollapse {
                                    collapsed += 1;
                                }
                                attempted_boundaries += 1;
                                proposal_diverse_boundaries +=
                                    usize::from(result.diagnostics.distinct_opponent_proposals > 1);
                                chance_diverse_boundaries += usize::from(
                                    result.diagnostics.distinct_chance_trajectories > 1,
                                );
                                public_outcome_diverse_boundaries +=
                                    usize::from(result.diagnostics.distinct_public_outcomes > 1);
                                for (category, count) in &result.diagnostics.mismatch_categories {
                                    *mismatch_totals.entry(*category).or_default() += count;
                                }
                                results.push(result);
                                if results.last().unwrap().conditioned_observation.is_none() {
                                    break;
                                }
                            }
                            preview_supported += usize::from(matches!(
                                results.first().map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            combat_one_supported += usize::from(matches!(
                                results.get(1).map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            combat_two_supported += usize::from(matches!(
                                results.get(2).map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            let stats = by_condition.entry((out_of_prior, proposal)).or_default();
                            stats[0] += 1;
                            stats[1] += usize::from(matches!(
                                results.first().map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            stats[2] += usize::from(matches!(
                                results.get(1).map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            stats[3] += usize::from(matches!(
                                results.get(2).map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            stats[4] += collapsed - collapsed_before;
                            stats[5] += diagnosed_rejections - rejections_before;
                            stats[6] += results.len();
                        }
                    }
                }
            }
        }
        assert_eq!(rows, 64);
        assert!(
            preview_supported > 0,
            "matrix must retain some preview support"
        );
        assert!(
            collapsed > 0,
            "matrix must expose independently observed collapse"
        );
        assert!(
            diagnosed_rejections > 0,
            "collapse must have classified rejections"
        );
        assert!(
            proposal_diverse_boundaries > 0,
            "sampled-policy matrix must produce multiple legal proposals"
        );
        println!(
            "C3b-1.2 matrix rows={rows} attempted_boundaries={attempted_boundaries} preview_supported={preview_supported} \
             combat_one_supported={combat_one_supported} combat_two_supported={combat_two_supported} \
             collapsed_boundaries={collapsed} diagnosed_rejections={diagnosed_rejections} \
             proposal_diverse_boundaries={proposal_diverse_boundaries} \
             chance_diverse_boundaries={chance_diverse_boundaries} \
             public_outcome_diverse_boundaries={public_outcome_diverse_boundaries} elapsed_ms={}",
            started.elapsed().as_millis()
        );
        for ((out_of_prior, proposal), stats) in by_condition {
            println!(
                "C3b-1.2 condition out_of_prior={out_of_prior} proposal={proposal:?} \
                 rows={} preview_supported={} combat_one_supported={} combat_two_supported={} \
                 collapsed_boundaries={} diagnosed_rejections={} attempted_boundaries={}",
                stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6]
            );
        }
        for (category, count) in mismatch_totals {
            println!("C3b-1.2 mismatch {category:?} count={count}");
        }
    }

    #[test]
    fn paired_public_event_conditioning_matrix_keeps_exact_final_filter() {
        let started = Instant::now();
        let (dex, teams) = assets();
        let prior = FrozenOpponentPrior::from_training_json(
            &dex,
            include_bytes!("../../data/training-teams.json"),
        )
        .unwrap();
        // rows, boundaries, preview support, combat-1 support, combat-2
        // support, collapse, zero-support, conditioned attempts, compatible.
        let mut stats = BTreeMap::<OpponentProposal, [usize; 9]>::new();
        for side in [SideId::P1, SideId::P2] {
            for out_of_prior in [false, true] {
                for seed_base in [801u16, 811u16] {
                    for particles in [2usize, 4] {
                        let (
                            initial,
                            preview_actions,
                            target_preview,
                            combat_one,
                            target_one,
                            combat_two,
                            target_two,
                        ) = legal_independent_actual_sequence(
                            &dex,
                            &teams,
                            side,
                            seed_base,
                            out_of_prior,
                        );
                        for proposal in [
                            OpponentProposal::FirstLegal,
                            OpponentProposal::SeededSampled,
                            OpponentProposal::PublicEventConditionedV1,
                        ] {
                            let config = ParticleConfig {
                                particles,
                                min_surviving_particles: 1,
                                max_units: 512,
                            };
                            let initialized = PersistentParticleBelief::initialize_with_proposal(
                                dex.clone(),
                                &initial,
                                &prior,
                                &[0x601, 0x602, 0x603, 0x604],
                                config.clone(),
                                proposal,
                            )
                            .unwrap();
                            let mut tracker = initialized.tracker.expect("fixture has support");
                            let mut results = Vec::new();
                            for (actions, target) in [
                                (preview_actions.clone(), target_preview.clone()),
                                (combat_one.clone(), target_one.clone()),
                                (combat_two.clone(), target_two.clone()),
                            ] {
                                let result = tracker
                                    .update_with_proposal(
                                        actions,
                                        &target,
                                        config.clone(),
                                        proposal,
                                    )
                                    .unwrap();
                                results.push(result);
                                if results.last().unwrap().conditioned_observation.is_none() {
                                    break;
                                }
                            }
                            let entry = stats.entry(proposal).or_default();
                            entry[0] += 1;
                            entry[1] += results.len();
                            entry[2] += usize::from(matches!(
                                results.first().map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            entry[3] += usize::from(matches!(
                                results.get(1).map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            entry[4] += usize::from(matches!(
                                results.get(2).map(|result| &result.status),
                                Some(
                                    ParticleStatus::Supported
                                        | ParticleStatus::InsufficientCoverage
                                )
                            ));
                            entry[5] += results
                                .iter()
                                .filter(|result| result.status == ParticleStatus::ParticleCollapse)
                                .count();
                            entry[6] += results
                                .iter()
                                .filter(|result| {
                                    result.status == ParticleStatus::ProposalZeroSupport
                                })
                                .count();
                            entry[7] += results
                                .iter()
                                .map(|result| result.diagnostics.public_event_proposal_attempts)
                                .sum::<usize>();
                            entry[8] += results
                                .iter()
                                .map(|result| result.diagnostics.public_event_compatible_proposals)
                                .sum::<usize>();
                        }
                    }
                }
            }
        }
        let first = stats[&OpponentProposal::FirstLegal];
        let sampled = stats[&OpponentProposal::SeededSampled];
        let conditioned = stats[&OpponentProposal::PublicEventConditionedV1];
        assert_eq!(first[0], 16);
        assert_eq!(sampled[0], 16);
        assert_eq!(conditioned[0], 16);
        assert!(first[2] > 0, "paired baseline must retain preview support");
        assert!(
            conditioned[7] > 0,
            "conditioned mode must enumerate legal proposals"
        );
        println!(
            "C3b-1.3 paired proposal=FirstLegal rows={} boundaries={} preview={} combat1={} combat2={} collapse={} zero_support={} attempts={} compatible={}",
            first[0], first[1], first[2], first[3], first[4], first[5], first[6], first[7], first[8]
        );
        println!(
            "C3b-1.3 paired proposal=SeededSampled rows={} boundaries={} preview={} combat1={} combat2={} collapse={} zero_support={} attempts={} compatible={}",
            sampled[0], sampled[1], sampled[2], sampled[3], sampled[4], sampled[5], sampled[6], sampled[7], sampled[8]
        );
        println!(
            "C3b-1.3 paired proposal=PublicEventConditionedV1 rows={} boundaries={} preview={} combat1={} combat2={} collapse={} zero_support={} attempts={} compatible={} elapsed_ms={}",
            conditioned[0], conditioned[1], conditioned[2], conditioned[3], conditioned[4], conditioned[5], conditioned[6], conditioned[7], conditioned[8], started.elapsed().as_millis()
        );
    }
}
