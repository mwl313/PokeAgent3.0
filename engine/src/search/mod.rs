//! Search-related code kept outside the battle mechanics.
//!
//! This module currently contains only a known-world correctness harness. It
//! is not an information-set search API and is deliberately unsuitable for
//! teacher generation.

pub mod belief;
pub mod core;
pub mod information_set;
pub mod test_oracle;
pub mod world;
