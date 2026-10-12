# A9.4.2 Final Report

## Configuration integrity

The canonical profile is `configs/validated_a94_runtime.yaml`. Its semantic hash excludes evidence/documentation metadata while its artifact hash covers the complete profile. The launcher consumes the resolved profile and rejects explicit contradictions such as PPO-4, LR scale 1.0, Mode A, default FP16 reductions, or disabled recompute gate. The resolver validates dataset hash, batch accounting, precision contract, and profile version before execution.

The approved runtime remains PPO-2, LR scale 0.5, global minibatch 4096, microbatch 1024, D+D1, historical probability 0, and tolerance 1e-3.

## Evaluation verification

Evaluation identity version 3 includes both policies' precision mode, stability, contract version, observation quantization, checkpoint/model identity, engine, dataset, fixture, seed, and sampling semantics. A fresh namespace was used and reported `cache_hit=false`: 2,048 games, 1,050 wins, 998 losses, 0 draws, 80.71 games/s, zero illegal actions and operational errors. Same-fixture batch tests produced 46.35, 65.56, 80.71, and 80.80 games/s at batch sizes 32, 64, 128, and 256 respectively, with identical outcomes and zero errors.

## Engine and A10 status

No Rust toolchain was discoverable (`cargo`, `rustc`, `rustup`, and project-local Cargo were absent), so all 16 native readiness criteria are NOT VERIFIED in this environment. This is documented separately in `docs/ENGINE_READINESS_A10_BLOCKERS.md`; no unsupported mechanics are hidden. No teams were imported and A10 training was not started.
