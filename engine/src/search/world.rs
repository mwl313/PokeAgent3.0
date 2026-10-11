//! Restricted clean-preview sampled-world construction.
//!
//! This is a C2c-1 boundary utility, not a search simulator or teacher.  It
//! accepts a trusted player-view DTO and one anonymized sampled opponent sheet,
//! then builds a fresh preview-only engine state with an independent seed.

use crate::{
    FORMAT,
    actions::RequestKind,
    assets::Dex,
    legality,
    search::{belief::BeliefTeam, information_set::TrustedSearchInformationSet},
    state::{BattleState, PlayerView, SideId, Team, TeamSet},
};
use std::{collections::BTreeMap, fmt};

pub const PREVIEW_WORLD_SCHEMA: u16 = 1;
pub const MAX_SEED_ATTEMPTS: usize = 256;

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
