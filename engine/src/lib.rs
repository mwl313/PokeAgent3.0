//! Native Champions doubles engine. See ENGINE_SCOPE.md for the full contract.
//! Assets and reference fixture generation run offline, never in battle transitions.
pub mod actions;
pub mod assets;
pub mod batch;
pub mod battle;
pub mod damage;
pub mod effects;
pub mod knowledge;
pub mod legality;
pub mod items;
pub mod observation;
#[cfg(feature = "python")]
pub mod python;
pub mod queue;
pub mod rng;
pub mod state;
pub mod stats;

pub const FORMAT: &str = "gen9championsvgc2026regmc";
pub const ORACLE_COMMIT: &str = "14546894d86f9589ac11130c510bbe73b6968665";

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("unsupported mechanic: {0}")]
    Unsupported(String),
    #[error("asset mismatch: {0}")]
    AssetMismatch(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, EngineError>;
