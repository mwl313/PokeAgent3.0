# Engine readiness blockers for A10

The authoritative command is `bash scripts/cargo.sh run --locked --release --example readiness_check`. On this host no `cargo`, `rustc`, `rustup`, or project-local `.tools/cargo/bin/cargo` was present, so the 16-criterion native readiness gate is **NOT VERIFIED**. The repository readiness documentation continues to record unresolved full-regulation mechanics, dynamic-closure, and differential criteria; successful PPO games only establish training-pool trajectory completion.

Before A10, restore the repository-approved pinned Rust toolchain without changing system packages, run readiness_check and established engine tests, record every criterion independently, and validate any newly approved teams against the authoritative engine. A10 full-production authorization remains blocked until this evidence exists.
