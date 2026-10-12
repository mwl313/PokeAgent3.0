# A10-P1 engine compatibility report

## Verdict

The exact 1,220-team pool is **NOT VERIFIED** for pool-scoped native Rust readiness. The P0 manifest remains unchanged (SHA recorded in the companion JSON). The project-local Rust toolchain was restored with official rustup into `.tools/` at Rust/Cargo 1.90.0; no system packages or NVIDIA/CUDA files were changed.

## Evidence

The engine compiled with `./scripts/cargo.sh check --manifest-path engine/Cargo.toml`. Selected bounded suites passed: contracts (16), fixture coverage (2), legality (1), ability contact (1), ability interactions (2), move hooks (1), and Poltergeist regressions (3). Existing native tests exercise the pinned Showdown fixtures and action legality. Static catalog comparison found all 194 species, 346 moves, 139 abilities, and 137 items present in `engine/data/dex.json`.

The 73 resource forms still require executable form/initialization evidence. Pool-specific reachable dependency closure, exact 1,220-team legal action/follow-up checks, targeted differential expansion, and all-team natural completion were not completed in this bounded milestone. The existing Rust test corpus includes v3-oriented training fixtures; it does not establish zero unsupported failures for every newly added team. Accordingly no pool-scoped PASS is claimed.

No engine source was modified, so no new mechanic defect was introduced or fixed. The remaining blocker is validation scope, not a demonstrated team-specific regression. PPO training and A10 were not started.
