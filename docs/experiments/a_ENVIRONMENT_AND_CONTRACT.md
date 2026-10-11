# Track A0 environment and contract snapshot

Date: 2026-10-11 (Asia/Seoul)

- Worktree: `/home/minidc/Desktop/PokeAgent3.0-A`
- Branch: `research/a-ppo-efficiency`
- Base: `8997cb4`, including v6 throughput work and the Rust Poltergeist `Item(0)` observation fix.
- Production defaults preserved: PPO 4 epochs, global minibatch 4096, microbatch 1024, gamma 1.0, GAE lambda 0.95, FP16 forward with FP32 probability/loss, synchronized DDP denominators and GradScaler.
- Historical reference: 36.563 cold / 43.477 steady games/s; PPO update about 29 s of a 47.1 s steady iteration. These are prior measurements, not an A1 claim.
- Existing instrumentation partitions collection, GAE/cache, PPO forward/backward/optimizer, digest, checkpoint, cleanup and barriers. The cached rollout path materializes once and reuses it.
- `--epochs` overrides PPO config for bounded experiments. Checkpoint publishing verifies equal rank model and optimizer digests, but launcher checkpoints do not restore collector opponent state and process RNG provenance; complete resume fidelity is incomplete.
- Poltergeist receipt: `8997cb4`; public `EventKind::Item` with item ID 0 is accepted only for the redirected empty-target reveal. Snapshot and unit tests passed before this branch was created.
- Readiness remains the previously reported 10/16 result; this snapshot does not promote it.

## First bounded A1 change

`experimental_cached_contiguous_select` is an off-by-default `PPOConfig` flag. For a contiguous cached minibatch it uses `RolloutBatch.narrow`, preserving values and masks while avoiding index-select copies. Shuffled minibatches keep the existing gather path. The objective, ordering, legality, hidden-information, DDP, checkpoint and production defaults are unchanged.

Acceptance test: `tests/agent/test_a_cached_select.py`. This is engineering evidence only; no long RL run is authorized.
