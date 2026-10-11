//! Player-information-only input boundary for future search work.
//!
//! This DTO deliberately accepts `PlayerView`, never `BattleState`, snapshots,
//! traces, team records, pending commands, or RNG. It is a transport boundary,
//! not a belief sampler or a search policy.

use crate::{
    EngineError, Result,
    actions::Request,
    knowledge::Knowledge,
    state::{Outcome, OwnSideView, PlayerView, SideId},
};
use serde::{Deserialize, Serialize};

/// Version of the player-safe search input wire format.
pub const SEARCH_INFORMATION_SET_SCHEMA: u16 = 1;

/// Immutable fields visible to the player through public history/current view.
/// `Knowledge` contains only audience-filtered public facts; unknown values are
/// represented by their `Known::known == false` masks and are never inferred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicInformation {
    pub turn: u16,
    pub knowledge: Knowledge,
    pub outcome: Outcome,
}

/// The player's own private observation and currently served legal request.
/// This is the only place where exact own set/state fields are carried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnInformation {
    pub side: SideId,
    pub state: OwnSideView,
    pub request: Request,
}

/// Minimal typed information set accepted by future search components.
///
/// There is intentionally no constructor from `BattleState` or snapshot. A
/// caller must first obtain the engine's audience-filtered `PlayerView`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchInformationSet {
    pub schema_version: u16,
    pub public: PublicInformation,
    pub own: OwnInformation,
}

impl SearchInformationSet {
    /// Copy only fields already present in one player's `PlayerView`.
    pub fn from_player_view(view: &PlayerView) -> Result<Self> {
        view.request.validate()?;
        Ok(Self {
            schema_version: SEARCH_INFORMATION_SET_SCHEMA,
            public: PublicInformation {
                turn: view.turn,
                knowledge: view.knowledge.clone(),
                outcome: view.outcome.clone(),
            },
            own: OwnInformation {
                side: view.side,
                state: view.own.clone(),
                request: view.request.clone(),
            },
        })
    }

    /// Deterministic JSON bytes: struct fields have fixed order and all
    /// knowledge maps are `BTreeMap`s. Unknown fields are rejected on decode.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(EngineError::Json)
    }

    /// Decode only the current schema. No hidden value reconstruction occurs.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let value: Self = serde_json::from_slice(bytes)?;
        if value.schema_version != SEARCH_INFORMATION_SET_SCHEMA {
            return Err(EngineError::InvalidInput(
                "search information-set schema".into(),
            ));
        }
        value.own.request.validate()?;
        Ok(value)
    }
}
