//! Restricted clean-preview sampled-world construction.
//!
//! This is a C2c-1 boundary utility, not a search simulator or teacher.  It
//! accepts a trusted player-view DTO and one anonymized sampled opponent sheet,
//! then builds a fresh preview-only engine state with an independent seed.

use crate::{
    FORMAT,
    actions::RequestKind,
    assets::Dex,
    batch::SideChoice,
    legality,
    search::{belief::BeliefTeam, information_set::TrustedSearchInformationSet},
    state::{BattleState, PlayerView, SideId, Team, TeamSet},
};
use std::{collections::BTreeMap, fmt, sync::Arc};

pub const PREVIEW_WORLD_SCHEMA: u16 = 1;
pub const MAX_SEED_ATTEMPTS: usize = 256;
/// Hard stop for a synthetic episode. A caller must create a new bounded
/// world rather than silently truncating an episode.
pub const MAX_EPISODE_BOUNDARIES: usize = 256;
/// Reserved accounting budget for future C3 search work. C2d consumes no
/// search units, but the limit is part of the stable execution contract.
pub const MAX_SEARCH_COMPUTATION_UNITS: u64 = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldFailureCode {
    UnsupportedState,
    InvalidOwnTeam,
    InvalidSample,
    ObservationMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldConstructionDiagnostics {
    pub schema_version: u16,
    pub format: String,
    pub attempts: usize,
    pub rejection_reasons: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldConstructionError {
    pub code: WorldFailureCode,
    pub message: String,
    pub diagnostics: WorldConstructionDiagnostics,
}

impl fmt::Display for WorldConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for WorldConstructionError {}

/// Private synthetic state plus its player-safe projection. No snapshot or
/// raw state accessor is exposed from this type.
pub struct PreviewWorld {
    state: BattleState,
    view: PlayerView,
    diagnostics: WorldConstructionDiagnostics,
    dex_digest: String,
}

/// Player-safe diagnostics returned at each synthetic decision boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeDiagnostics {
    pub schema_version: u16,
    pub turn: u16,
    pub root_request_kind: RequestKind,
    pub accepted: bool,
    pub boundary_count: usize,
    pub max_boundaries: usize,
    pub search_computation_units_remaining: u64,
}

/// A boundary result contains only the requested root player's view and
/// nonsecret operational metadata. It never carries a snapshot or opponent
/// action/state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeBoundary {
    pub root_view: PlayerView,
    pub diagnostics: EpisodeDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeFailureCode {
    InvalidAction,
    OperationalError,
    RollbackError,
    LimitExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeError {
    pub code: EpisodeFailureCode,
    pub message: String,
}

impl fmt::Display for EpisodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for EpisodeError {}

/// Privately owned synthetic episode. This runner advances only states created
/// by `PreviewWorld`; it is not a reconstruction API for live PPO battles.
pub struct SyntheticEpisode {
    dex: Arc<Dex>,
    state: BattleState,
    root_side: SideId,
    boundaries: usize,
    search_computation_units: u64,
}

impl PreviewWorld {
    pub fn construct(
        dex: &Dex,
        info: &TrustedSearchInformationSet,
        sampled_opponent: &BeliefTeam,
        independent_seed: u64,
    ) -> Result<Self, WorldConstructionError> {
        let expected = info.as_info();
        let mut diagnostics = WorldConstructionDiagnostics {
            schema_version: PREVIEW_WORLD_SCHEMA,
            format: FORMAT.into(),
            attempts: 0,
            rejection_reasons: BTreeMap::new(),
        };
        let own = match own_team(expected) {
            Ok(team) => team,
            Err(message) => {
                return Err(failure(
                    WorldFailureCode::UnsupportedState,
                    message,
                    diagnostics,
                ));
            }
        };
        if let Err(error) = own.validate_structure(dex) {
            return Err(failure(
                WorldFailureCode::InvalidOwnTeam,
                format!("reconstructed own team failed structure validation: {error}"),
                diagnostics,
            ));
        }
        let own_report = legality::validate_team(dex, &own);
        if !own_report.legal() {
            return Err(failure(
                WorldFailureCode::InvalidOwnTeam,
                format!(
                    "reconstructed own team failed pinned legality: {:?}",
                    own_report.verdict
                ),
                diagnostics,
            ));
        }
        let opponent = Team {
            id: String::new(),
            members: sampled_opponent.members.clone(),
        };
        if let Err(error) = opponent.validate_structure(dex) {
            return Err(failure(
                WorldFailureCode::InvalidSample,
                format!("sampled opponent failed structure validation: {error}"),
                diagnostics,
            ));
        }
        let report = legality::validate_team(dex, &opponent);
        if !report.legal() {
            return Err(failure(
                WorldFailureCode::InvalidSample,
                format!(
                    "sampled opponent failed pinned legality: {:?}",
                    report.verdict
                ),
                diagnostics,
            ));
        }

        let mut teams: [&Team; 2] = [&own, &opponent];
        if expected.own.side == SideId::P2 {
            teams = [&opponent, &own];
        }
        for attempt in 0..MAX_SEED_ATTEMPTS {
            diagnostics.attempts = attempt + 1;
            let state =
                match BattleState::reset(dex, teams, seed_words(independent_seed, attempt), [0, 1])
                {
                    Ok(state) => state,
                    Err(error) => {
                        increment(&mut diagnostics.rejection_reasons, "reset_rejected");
                        return Err(failure(
                            WorldFailureCode::InvalidSample,
                            format!("synthetic preview reset rejected: {error}"),
                            diagnostics,
                        ));
                    }
                };
            let view = state.observe(expected.own.side);
            let candidate =
                crate::search::information_set::SearchInformationSet::from_player_view(&view)
                    .map_err(|error| {
                        failure(
                            WorldFailureCode::ObservationMismatch,
                            error.to_string(),
                            diagnostics.clone(),
                        )
                    })?;
            if candidate == *expected {
                return Ok(Self {
                    state,
                    view,
                    diagnostics,
                    dex_digest: dex.asset_digest.clone(),
                });
            }
            increment(&mut diagnostics.rejection_reasons, "observation_mismatch");
        }
        Err(failure(
            WorldFailureCode::ObservationMismatch,
            "no bounded independent seed reproduced the supplied clean-preview observation".into(),
            diagnostics,
        ))
    }

    pub fn player_view(&self) -> &PlayerView {
        &self.view
    }

    pub fn diagnostics(&self) -> &WorldConstructionDiagnostics {
        &self.diagnostics
    }

    /// Keep the synthetic state owned and observable only through the view;
    /// this method proves the state is retained without exposing snapshots.
    pub fn is_clean_preview(&self) -> bool {
        self.state.observe(self.view.side).request.kind == RequestKind::Preview
    }

    pub fn into_episode(self, dex: Arc<Dex>) -> Result<SyntheticEpisode, EpisodeError> {
        if self.dex_digest != dex.asset_digest {
            return Err(EpisodeError {
                code: EpisodeFailureCode::InvalidAction,
                message: "synthetic world and Dex asset digests differ".into(),
            });
        }
        Ok(SyntheticEpisode {
            root_side: self.view.side,
            dex,
            state: self.state,
            boundaries: 0,
            search_computation_units: 0,
        })
    }
}

impl SyntheticEpisode {
    pub fn root_view(&self) -> PlayerView {
        self.state.observe(self.root_side)
    }

    /// Prefix candidates are generated from the root request only. The
    /// request tree preserves the engine's conditional target/switch rules.
    pub fn root_candidates(
        &self,
        prefix: &[crate::actions::AtomicAction],
    ) -> Result<Vec<crate::actions::AtomicAction>, EpisodeError> {
        self.state
            .observe(self.root_side)
            .request
            .candidates(prefix)
            .map_err(|error| invalid_action(error.to_string()))
    }

    pub fn boundaries(&self) -> usize {
        self.boundaries
    }

    pub fn search_computation_units_remaining(&self) -> u64 {
        MAX_SEARCH_COMPUTATION_UNITS.saturating_sub(self.search_computation_units)
    }

    /// Validate every side's complete action before mutating the synthetic
    /// state. If resolution reports an operational error, restore the exact
    /// pre-submit snapshot and return a failure.
    pub fn submit_joint(
        &mut self,
        choices: &[SideChoice],
    ) -> Result<EpisodeBoundary, EpisodeError> {
        if self.boundaries >= MAX_EPISODE_BOUNDARIES {
            return Err(EpisodeError {
                code: EpisodeFailureCode::LimitExceeded,
                message: "synthetic episode boundary limit exhausted".into(),
            });
        }
        if choices.is_empty() || choices.len() > 2 {
            return Err(invalid_action("one or two side choices required"));
        }
        let mut seen = 0u8;
        for choice in choices {
            let bit = 1 << choice.side.index();
            if seen & bit != 0 {
                return Err(invalid_action("duplicate side choice"));
            }
            seen |= bit;
            self.state
                .validate_choice(choice.side, &choice.actions)
                .map_err(|error| invalid_action(error.to_string()))?;
        }
        let before = self.state.snapshot().map_err(|error| EpisodeError {
            code: EpisodeFailureCode::RollbackError,
            message: error.to_string(),
        })?;
        let mut last = None;
        for choice in choices {
            match self.state.step(&self.dex, choice.side, &choice.actions) {
                Ok(result) => last = Some(result),
                Err(error) => return self.rollback(before, error.to_string()),
            }
        }
        let result = last.expect("nonempty choices prevalidated");
        if let Some(error) = &result.outcome.operational_error {
            return self.rollback(before, error.clone());
        }
        self.boundaries += 1;
        self.search_computation_units = self.search_computation_units.saturating_add(1);
        let root_view = self.state.observe(self.root_side);
        let diagnostics = EpisodeDiagnostics {
            schema_version: PREVIEW_WORLD_SCHEMA,
            turn: root_view.turn,
            root_request_kind: root_view.request.kind,
            accepted: result.accepted,
            boundary_count: self.boundaries,
            max_boundaries: MAX_EPISODE_BOUNDARIES,
            search_computation_units_remaining: self.search_computation_units_remaining(),
        };
        Ok(EpisodeBoundary {
            root_view,
            diagnostics,
        })
    }

    /// Controlled fault hook for rollback tests only. It deliberately begins
    /// one valid mutation, then restores the exact pre-submit state. It is not
    /// a battle mechanic or a production execution path.
    #[doc(hidden)]
    pub fn submit_joint_with_controlled_failure_for_test(
        &mut self,
        choices: &[SideChoice],
    ) -> Result<EpisodeBoundary, EpisodeError> {
        if choices.is_empty() {
            return Err(invalid_action("one side choice required"));
        }
        for choice in choices {
            self.state
                .validate_choice(choice.side, &choice.actions)
                .map_err(|error| invalid_action(error.to_string()))?;
        }
        let before = self.state.snapshot().map_err(|error| EpisodeError {
            code: EpisodeFailureCode::RollbackError,
            message: error.to_string(),
        })?;
        self.state
            .step(&self.dex, choices[0].side, &choices[0].actions)
            .map_err(|error| EpisodeError {
                code: EpisodeFailureCode::OperationalError,
                message: error.to_string(),
            })?;
        self.state = BattleState::restore(&self.dex, &before).map_err(|error| EpisodeError {
            code: EpisodeFailureCode::RollbackError,
            message: error.to_string(),
        })?;
        Err(EpisodeError {
            code: EpisodeFailureCode::OperationalError,
            message: "controlled test fault after mutation; state restored".into(),
        })
    }

    fn rollback(
        &mut self,
        before: Vec<u8>,
        message: String,
    ) -> Result<EpisodeBoundary, EpisodeError> {
        self.state = BattleState::restore(&self.dex, &before).map_err(|error| EpisodeError {
            code: EpisodeFailureCode::RollbackError,
            message: error.to_string(),
        })?;
        Err(EpisodeError {
            code: EpisodeFailureCode::OperationalError,
            message,
        })
    }
}

fn invalid_action(message: impl Into<String>) -> EpisodeError {
    EpisodeError {
        code: EpisodeFailureCode::InvalidAction,
        message: message.into(),
    }
}

fn own_team(info: &crate::search::information_set::SearchInformationSet) -> Result<Team, String> {
    if info.public.turn != 0
        || info.own.request.kind != RequestKind::Preview
        || info.own.state.selected_order.is_some()
        || info.own.state.active != [None, None]
        || info.own.state.mega_used
    {
        return Err("only an untouched clean preview information set is supported".into());
    }
    if info.own.request.preview_roster != (0..6).collect::<Vec<_>>() {
        return Err("preview roster is not the pinned six-member order".into());
    }
    let members: Vec<TeamSet> = (0..6)
        .map(|index| {
            let pokemon = &info.own.state.pokemon[index];
            if pokemon.species != pokemon.base_species
                || pokemon.selected
                || pokemon.active_slot.is_some()
                || pokemon.fainted
                || pokemon.transformed
                || pokemon.status != 0
                || pokemon.boosts != [0; 7]
                || pokemon.hp != pokemon.stats[0]
                || pokemon.moves != pokemon.base_moves
                || pokemon.moves.is_empty()
                || pokemon
                    .moves
                    .iter()
                    .any(|mv| mv.disabled || mv.hidden || mv.used || mv.pp != mv.max_pp)
            {
                return Err(format!(
                    "own roster slot {index} is not clean preview state"
                ));
            }
            let gender = match pokemon.gender {
                0 => "N",
                1 => "M",
                2 => "F",
                _ => return Err(format!("own roster slot {index} has invalid gender")),
            };
            Ok::<TeamSet, String>(TeamSet {
                species: pokemon.base_species,
                ability: pokemon.ability,
                item: pokemon.item,
                nature: pokemon.nature,
                moves: pokemon.base_moves.iter().map(|mv| mv.id).collect(),
                points: pokemon.points,
                ivs: pokemon.ivs,
                gender: gender.into(),
                level: pokemon.level,
            })
        })
        .collect::<Result<_, _>>()?;
    let members: [TeamSet; 6] = members
        .try_into()
        .map_err(|_| "own roster reconstruction did not produce six members".to_string())?;
    Ok(Team {
        id: String::new(),
        members,
    })
}

fn seed_words(seed: u64, attempt: usize) -> [u16; 4] {
    let mut value = seed.wrapping_add((attempt as u64).wrapping_mul(0x9e37_79b9));
    std::array::from_fn(|_| {
        value ^= value << 7;
        value ^= value >> 9;
        value = value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        value as u16
    })
}

fn increment(map: &mut BTreeMap<String, usize>, key: &str) {
    *map.entry(key.into()).or_default() += 1;
}

fn failure(
    code: WorldFailureCode,
    message: String,
    diagnostics: WorldConstructionDiagnostics,
) -> WorldConstructionError {
    WorldConstructionError {
        code,
        message,
        diagnostics,
    }
}
