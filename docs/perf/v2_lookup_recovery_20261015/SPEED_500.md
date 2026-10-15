# Requested500-game two-V100 speed test

Executed source: `3b4f178fac8b6c52043c3ca0c2e66e589aa9a937`.
Both V100s,250 natural games/rank, exactly500 total, one bounded iteration.
PPO2/global4096/micro1024/workers16, envs250/rank, DDP, ModeD/reduction-off,
deterministic CUDA and CUBLAS_WORKSPACE_CONFIG=:4096:8. CPU optimizations ON;
experimental flat-row GPU lookup OFF. No microbatch experiment.

| Metric | Actual |
|---|---:|
| Iteration-only games/s | **38.5863** |
| Startup/shutdown-inclusive games/s | **25.6933** |
| Successful launcher wall | 19.4603s |
| Critical-rank complete iteration wall | 12.9580s |
| Collection + clock wait | 5.2164s |
| Preparation | 1.1533s |
| PPO update | 5.6303s |
| Other iteration stages, residual | 0.9580s |
| Learner rows | 11,393 |
| PPO epochs | 2 on both ranks |
| Synchronized optimizer steps | 6, not12 |
| Maximum recompute error | 1.90735e-6, below1e-3 |
| Operational errors / skipped optimizer steps | 0 / 0 |
| Both model/Adam rank digests agree | YES |

This uses the verified57,344-match policy weights with a **fresh diagnostic
Adam and scheduler clock**, not checkpoint continuation. The wrapper validates
rank-file checksums, the original source identity, dataset/action contract,
model config, strict tensor loading and initial model digest before inference.
It restores only model weights; it does not bypass full-resume source checks
or claim to preserve campaign optimizer/RNG. Outputs are discarded after
measurement; the campaign clock/checkpoints/ledger remain unchanged.

The existing launcher requires an experimental checkpoint prefix to allow
LR-scale0.5; a private diagnostic prefix was provided with the default100,000
interval. The500-match run never reaches it, and no checkpoint file was
created. An initial CLI-only setup attempt was rejected before any games;
its logs remain local under setup_failure and are excluded from successful
launch timing. All safeguards remained enabled.

This short, single-iteration,250-env/rank workload is not steady-state
production speed or a matched4096-game baseline comparison. Startup is large
relative to500 games; do not interpret25.69 as sustained training throughput.
No GPU-backend speedup claim: it remains OFF after the FP16 numerical finding.

Raw report, per-rank logs and exact diagnostic wrapper are in
`speed500_evidence/`; original artifacts are at
`/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_speed_500_20261015/`.
No training/evaluation service was started and no job remains scheduled.
