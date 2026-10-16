# V2 migration, FP32 PPO stability and V100 optimization

Production remains **STOPPED**. Production matches added: **0**.

**Snapshot paused by user for an FP16-versus-FP32 decision.** The ten-iteration results and reviewed migration proofs apply to tested executable `9d3d6051d6d17cc517259a3e6807bb067c717a9e`. This snapshot changes documentation/evidence only. Its final HEAD-specific migration receipt has **not been prepared**; it remains pending while work is paused. Do not use the older target-specific receipt to launch this newer documentation commit. This report supersedes the 2026-10-15 Mode-D blocked assessment; historical failures and prior evidence remain preserved in Git and local artifacts. The user explicitly approved an FP32 policy encoder on 2026-10-16.

Delivery branch: `codex/v2-secondary-precision-stability`. Tested executable: `9d3d6051d6d17cc517259a3e6807bb067c717a9e`. Final documentation commit has identical executable blobs; its exact HEAD-specific proof/receipt is generated afterward under local `runs/v2_final_release/final_delivery_20261016`. The final commit is recorded there and by Git, avoiding a self-referential hash inside this document.

## Precision repair and feature preservation

The earlier drift came from Mode-D FP16 encoder arithmetic changing with row grouping and the strict FP32 reference. Actor replay itself was close, but absolute FP16-versus-FP32 differences exceeded 0.001; selective FP32 repairs did not resolve all captured failures. Those failures remain rejected and unchanged.

Additive Mode E, identity `pa3-precision-E-v1`, executes the policy encoder, scorer and GRU in FP32 while retaining Mode-D FP16-round-then-FP32 observation quantization, existing value-head AMP, GradScaler and bounded collective AMP retries. E uses logical FP32 row batches; the canonical1024 compatibility setting remains recorded. Architecture and parameter names are unchanged. **All 96 token slots remain; observation-padding trimming is disabled.** Five ragged families, masks, roster/action mappings, legal candidates and player-visible information are preserved.

Primary 0.001 and secondary provenance/replay/grouped-versus-isolated reference rules are unchanged. Rejected secondary audit evidence now preserves the actual failing actor groups, including failures where the primary gate passed. There is no tolerance increase or acceptance of historical failed rollouts.

PPO remains2 maximum epochs, global minibatch4096, per-rank2048, microbatch1024, accumulation2, LR scale0.5, current/current, envs1024/workers16 and the frozen1220-team pool. Existing target-KL early stopping remains active. Mode-E evaluation archive/load paths now carry the correct quantization and precision identity.

## Strict full-state migration

The v3 reviewed migration adds explicit D→E policy-encoder and physical-execution transitions while retaining older receipt schemas. It pins the exact target SHA, named executable Git blobs, sealed validator AST, native binary, dataset, original checkpoint manifest and both complete rank artifacts. Actual CLI execution values must match the approved receipt. Dirty or unknown executable changes and tampered weights/Adam/RNG/rank manifests/source/precision/backend/statistics are rejected.

Independent reference code is frozen at `0b7253cfd915ff61c88248b1979bd8e254285ef0`; it independently replaces its original precision dataclass's encoder AMP boundary. Matched captured1024-row loss, every gradient, fresh diagnostic Adam and updated model are bit exact against candidate E. These diagnostic probes supplement actual PPO, which restores the original full Adam/scaler/scheduler/RNG/accounting/collector state without resetting the optimizer.

Original recovery:57,344 matches, absolute iteration14,718 steps, source `5ccc51b0cd4a49efb93018258b9750d5e804ae32`; checkpoint path `/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699`. Manifest SHA256 `b9539f3a0c426f0878375b61630cb4de9dcd2a5d4a6a59858b4d64f4287fd424`. Rank0 `242783259d942c3847764c966cac1f3e077ae062a326bb702b7866330fcf5efc`; rank1 `c8aff16cd22b0264d067872c3d8df457bba30f1008815765c2cb999a5216aaf8`. The original rolling pointer remains4,096; failed102,400 is never restored or counted. The recovered historical RNG limitation remains documented; restoring this selected recovered state is exact.

## Matched performance

All three repetitions per arm restore the same original full checkpoint, use identical seeds,2048 natural matches per repetition, learning settings and frozen sources. All18 rank results have identical final model and Adam digests; rollout rows/action counts and restored state agree across arms.

| FP32 arm | Iteration matches/sec, three repetitions | Median | Launcher all-in median |
|---|---|---:|---:|
| Gather |28.78212,29.32487,28.67772|28.78212|24.07415|
| Exact row lookup |29.75720,29.76901,29.18975|29.75720|24.35859|
| Exact + two additional Actor transfers |28.82526,29.84718,30.78076|29.84718|24.65465|

The selected exact reducer improves iteration median **3.39%** and Learner update time about8% versus the valid FP32 gather baseline. Launcher all-in includes startup/restore and improves1.18%. The two Actor changes add0.30% iteration median with overlapping variation and are **not promoted**. Existing Actor improvements remain. No further speculative optimization is included. One brief small-tensor CUDA ownership test overlapped the startup window of exact repeat3; no statistical significance is claimed. The Learner update improvement is consistent across all three exact repetitions.

Iteration throughput is2048 divided by the slower rank's complete iteration wall time; single-iteration trials do not provide a steady-state estimate. Launcher all-in includes startup. The final stability run forces expensive audits and publishes every4096 matches, so it is not a matched speed benchmark. The former FP16 numbers and historical~90games/sec used different precision/workloads/gates and are not comparable safe targets. No claim of restored90games/sec or a hardware ceiling is made.

The exact-order V100 reducer retains bitwise gradient order and device/compiler guards. Flat embedding/approximate reduction stays disabled. All 96 observation token slots remain. Detailed raw repetitions, stages and hashes are in the compact evidence.

| FP32 arm | Median collection s | Median update s | Peak CUDA reserved GiB/rank | Peak RSS GiB/rank |
|---|---:|---:|---:|---:|
|E_gather|23.093|40.985|16.881|5.597|
|E_exact|23.520|37.782|16.326|5.602|

Stage values use the slower rank per repetition; memory is the maximum observed across ranks/repetitions. CUDA reserved memory is allocator reservation, not active tensor bytes. All-in benchmark rates include startup; iteration rates do not.

## Validation and readiness

Migration: **PASS**. PPO stability: **PASS within ten bounded iterations**. Batch optimization: **IMPROVED**, with exact lookup selected and extra Actor changes not promoted. Production remains **STOPPED**; production matches added: **0**.

The selected E executable completes **10 genuine iterations /40,960 natural diagnostic matches /412 successful optimizer steps** (718→1,130), with820,173 learner rows across both ranks. Every iteration runs2 epochs, with zero AMP retries, zero skipped optimizer steps and zero operational errors. Both ranks' model and Adam digests agree at every boundary. Maximum primary recompute error is **4.76837158203e-06**, below the unchanged0.001 limit. Both forced secondary audits pass on both ranks every iteration; maximum recorded FP32 residual is **1.71661376953e-05**. The isolated final boundary is98,304 /absolute iteration24; it does not replace the57,344 production restart boundary.

The usage-limit interruption occurred after three completed iterations, before final parent/rank JSON was emitted. The third pair of complete rank envelopes survived; the unchanged checkpoint publisher verified and published69,632 before continuation. Two subsequent iterations restored that complete state. Following the user's instruction to keep going, five more iterations completed the original ten-iteration target. Original durable post-barrier telemetry supplies the interrupted segment's recorded metrics; missing rollout/communication fields are not invented. All ten checkpoint generations /20 full learner rank artifacts pass finite-tensor, identity, hash and rank-parity checks.

Restoring61,440 and replaying the next4,096 matches under a precreated cooperative STOP produces a **BIT_EXACT**65,536 boundary versus uninterrupted training: complete learner (model/Adam/scaler/scheduler), RNG, accounting, collector restart, rank digests, dataset and opponent manifests. Both ranks exit after the drained checkpoint. The interrupted segment did not persist a rollout digest; this comparison hashes every nested saved tensor byte and complete owned state rather than abbreviated repr strings.

Tests: **449 passed /36 CUDA cases skipped** in the full agent/integration suite; **90 passed** in a fresh explicit CUDA suite. Configuration/source-of-truth parity passes. A separate two-game CPU evaluation sidecar smoke passes E-v1 and observation-quantization identities with zero illegal actions/errors; it makes no strength claim and has its own namespace. All **89 protected original files remain byte-identical**. Compact [evidence](v2_fp32_stability_evidence/) includes raw benchmark repetitions, full training telemetry, checkpoint/restore verification, evaluation receipt, cleanup receipts and test logs.

The former Mode-D precision blocker is resolved in these bounded E cases. The requested migration, ten-iteration stability, restore and matched benchmark checks are complete. Bounded success does not establish indefinite stability or unseen-team strength. Production remains stopped; any separately authorized restart still requires normal live preflight and the final HEAD-specific migration receipt. The existing production campaign, checkpoint pointer, counters and evaluator remain stopped and unchanged.

Two Tesla V100 PCIe32GB at175W; driver580.178.04; torch2.14.0+cu126, CUDA12.6 runtime/nvcc12.8, deterministic workspace4096:8. Native SHA256 `8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62`. No hardware/power/driver change. Large checkpoints, original failure captures and traces remain local. Cache-only cleanup reclaimed1.715GiB; no checkpoint/source/environment/history/evidence was removed, and the user freed additional space.

Use the versioned E profile and exact HEAD-specific reviewed receipt for any separately authorized future restart from57,344. Production remains stopped. Bounded success cannot guarantee indefinite learning stability or unseen-team strength. No main merge, force push, production counter advance or historical rewrite is performed.
