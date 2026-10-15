# Bounded two-V100 continuation

**READY_FOR_CUTOVER — 2026-10-15 KST. Production remains STOPPED at the user's explicit request.** Canonical `codex/pa3-v2-correctness` was ancestry-preserving fast-forwarded and pushed to the tested executable commit `0b7253cfd915ff61c88248b1979bd8e254285ef0`; subsequent changes are documentation/evidence only. No main/C-track change, destructive cleanup or production launch.

Executed clean source `0b7253cfd915ff61c88248b1979bd8e254285ef0`, both V100175W, ModeD, fp16 reduction OFF, deterministic CUDA workspace`:4096:8`, PPO2/global4096/micro1024, env1024/workers16 per rank,1220 frozen teams, secondary ON, canonical encoder1024, gather. Initial full state57344/iteration14/718 Adam steps restored exactly.

| Iteration | Diagnostic clock | Rows | Games/s | Optimizer steps | Primary max discrepancy |
|---|---:|---:|---:|---:|---:|
| 15 | 61440 | 92224 | 39.531 | 46 | 1.90734863e-06 |
| 16 | 65536 | 89932 | 38.306 | 46 | 1.43051147e-06 |
| 17 | 69632 | 89688 | 41.228 | 44 | 1.43051147e-06 |
| 18 | 73728 | 87455 | 41.570 | 44 | 1.90734863e-06 |
| 19 | 77824 | 85324 | 44.307 | 42 | 1.90734863e-06 |
| 20 | 81920 | 84405 | 43.385 | 42 | 0.000609636307 |

Main arm24576 naturally completed games, weighted **41.28453 games/s**,888.699 rows/s,0.443486 optimizer steps/s. All6 iterations2 epochs,0 skipped updates/operational errors, matching rank model/Adam digests. Per-stage timings, GPU reserved peak, CPU RSS, losses,KL,entropy,clip,gradient norm,LR/scaler and exact digests: [ACCEPTANCE.json](final_evidence/ACCEPTANCE.json). Timing uses slowest-rank measured iteration wall and includes gates/checkpoint/cleanup; not a kernel profiler. No concurrent evaluation workload ran during these measurements.

Uninterrupted iteration16 versus independent process exit/recreated-DDP/restore iteration16: **BIT_EXACT** learner(model/full Adam/scheduler/scaler/config), Python/NumPy/Torch CPU/CUDA/shuffle RNG,accounting,collector restart and rollout hashes. All team W/L/D/matchup/policy counters match after excluding distinct attempt IDs. Separate real stop-file run requested2 iterations, executed1, saved65536 boundary and both ranks exited cooperatively; saved state is BIT_EXACT to uninterrupted16.

Smoke and split/stop arms each rerun4096 games independently; they are not added to the main arm's clock. Total successful canonical test arms36864 physical games, main lineage advances24576 only. Earlier failed attempts are retained separately and never counted as production progress.

Final full diagnostic checkpoint `/home/minidc/Desktop/PokeAgent3.0-V2-integration/runs/v2_final_release/continuation_canonical/uninterrupted/rolling/generations/iter-000000081920-1792043149726370535`; manifest SHA `de1d9c366b277598fc31c702168238999368c8af9a3c6235c7e3560272e729af`. Model `1602e744a135df094f64e714170b479aace1999a6eb0cb5066be43ec0a738612`, Adam `ff5a9d42b8c069d342d17df15317ea6fc546b1a40a8619c688ee51b8df514a87`,982 cumulative optimizer steps, LR5.25136e-05. This81920 checkpoint is diagnostic; approved real campaign restart remains original57344. Production games added0.

This is bounded same-hardware/source/runtime proof, not universal numerical stability. Iteration20 still shows residual primary variation.0006096363<.001; future unexplained or excessive outliers stop both ranks before update and preserve evidence. The full periodic checkpoint remains100k; a future failure may require rollback to the last complete generation. All approved PPO hyperparameters remain unchanged.

Evaluation exact50k light/200k full snapshot cadence is not implemented in this release. The user-authorized minimal plan is checkpoint-driven CPU-only256-pair/batch32 evaluation, strict V2 archive identity and independent counters, plus health monitoring. Actual4-game native health check4W/0L/0D,0 operational errors/illegal actions, unchanged model/checkpoint. It is functionality proof, not strength evidence. GPU evaluation is not enabled. Automatic evaluator is stopped along with production; its existing resource safeguards remain unchanged. Do not claim the intended cadence is active.

Disk space is outside release-blocker scope by user instruction. Bounded diagnostic namespaces explicitly used an8GiB reserve; production15GiB protection remains unchanged. No branch, worktree, checkpoint or historical data was deleted.
