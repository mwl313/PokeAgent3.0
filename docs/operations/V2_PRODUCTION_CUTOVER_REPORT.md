# Production cutover preparation

**READY_FOR_CUTOVER — 2026-10-15 KST. Production remains STOPPED at the user's explicit request.** Canonical `codex/pa3-v2-correctness` was ancestry-preserving fast-forwarded and pushed to the tested executable commit `0b7253cfd915ff61c88248b1979bd8e254285ef0`; subsequent changes are documentation/evidence only. No main/C-track change, destructive cleanup or production launch.

**Do not start PPO or its evaluator in this task.** Latest user instruction supersedes earlier automatic-start authorization. No systemd training/evaluation service or timer was started.

Production source will be frozen at the final canonical documentation HEAD, with executable blob map identical to tested0b7253cfd915ff61c88248b1979bd8e254285ef0. A fresh exact-SHA migration receipt and absolute runtime profile are prepared outside Git under the original campaign's new release subdirectory; original immutable release files remain unchanged. The final source SHA, receipt file SHA and command paths are recorded in local cutover manifest and the final response.

Real restart checkpoint: `/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699`. Manifest b9539f3a0c426f0878375b61630cb4de9dcd2a5d4a6a59858b4d64f4287fd424. Both artifact checksums/model/Adam/scheduler/scaler/RNG/28 receipts validated. Never use original rolling/LATEST4096 or model-only73728 as if they were this full state. Do not promote diagnostic81920 to production silently.

Prepared managed launch uses systemd-run with explicit working directory, shared existing Python interpreter,OMP2 and workspace4096, `--experimental-resume` exact57344, reviewed migration, original ledger/rolling roots, PPO2/global4096/micro1024/seed20261014,100k checkpoints and24401 remaining iterations toward100003840 boundary. It is a JSON command plan, not an executed launch. Production storage guards stay enabled; normal resource preflight must be repeated at actual cutover.

Safe stop: touch only the future service's explicitly configured release-specific STOP file, wait for both ranks' completed drained boundary and verified generation publication. Rollback: preserve frozen5ccc51b runtime,57344 recovery+backup, all original generations/receipts and current source. Resume only a complete generation matching exact source/migration contract; never graft weights onto older Adam or manually rewrite LATEST. Existing old STOP request files are preserved.

This is bounded same-hardware/source/runtime proof, not universal numerical stability. Iteration20 still shows residual primary variation.0006096363<.001; future unexplained or excessive outliers stop both ranks before update and preserve evidence. The full periodic checkpoint remains100k; a future failure may require rollback to the last complete generation. All approved PPO hyperparameters remain unchanged.

Evaluation exact50k light/200k full snapshot cadence is not implemented in this release. The user-authorized minimal plan is checkpoint-driven CPU-only256-pair/batch32 evaluation, strict V2 archive identity and independent counters, plus health monitoring. Actual4-game native health check4W/0L/0D,0 operational errors/illegal actions, unchanged model/checkpoint. It is functionality proof, not strength evidence. GPU evaluation is not enabled. Automatic evaluator is stopped along with production; its existing resource safeguards remain unchanged. Do not claim the intended cadence is active.

Disk space is outside release-blocker scope by user instruction. Bounded diagnostic namespaces explicitly used an8GiB reserve; production15GiB protection remains unchanged. No branch, worktree, checkpoint or historical data was deleted.
