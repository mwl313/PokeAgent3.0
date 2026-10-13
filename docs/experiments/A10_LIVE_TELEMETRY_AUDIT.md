# A10 live telemetry audit

Measured live from rank logs: committed matches, completed iteration, rollout rows/time, PPO update time, actual epochs and optimizer steps, skipped updates, recompute maximum, and DDP digest equality. The live trainer also exposes GPU/disk/process telemetry through the independent status watchdog.

Policy loss, value loss, entropy, approximate KL, learning rate, gradient norm, and GradScaler state are not emitted in the active rank logs. The learner computes some diagnostics in memory, but they cannot be recovered externally without changing the trainer. An optional `agent/ppo/telemetry.py` schema/writer was added in the isolated worktree and tested; it is not imported by the active launcher and is READY only for a future validated resume.
