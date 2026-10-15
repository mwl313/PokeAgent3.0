# V2 migration, PPO stability and batch optimization — final engineering report

Migration: **PASS**. PPO stability: **BLOCKED** by the forced secondary audit despite successful ordinary continuations. Batch optimization: **PARTIAL**: measured improvements, exact lookup production activation held back. Production remains **STOPPED**; production progress added: **0**.

Delivery branch: `codex/v2-migration-stability-batch`. Starting pushed fix: `93ebc1b5086b7a2bf247b779eb6442671bbbdd10`. Integrated experimental benchmark executable: `163a7c3e3a2af219f98de017d1fdb1b636d824e8`. Delivered conservative executable before documentation: `04781f7f0c11c3716b00df084f0d8d872e26486b`; it retains Actor C and keeps gather in the production profile. Final documentation commit is recorded by Git and by the separate local HEAD-specific `runs/v2_final_release/final_delivery/FINAL_DELIVERY.json`; embedding a commit's own hash inside its tracked report would be self-referential. The final receipt must pin delivery HEAD, not the intermediate benchmark SHA.

## Runtime and preserved checkpoint

Two Tesla V100 PCIe 32GB, both 175W; driver580.178.04; Python3.12.3; torch2.14.0+cu126, CUDA runtime12.6, nvcc12.8. Deterministic CUDA workspace4096:8; FP16 reduced-precision reduction disabled; existing NUMA/NCCL setup preserved. Native binary SHA256 `8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62`.

Read-only recovery checkpoint: `/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699`. Manifest SHA256 `b9539f3a0c426f0878375b61630cb4de9dcd2a5d4a6a59858b4d64f4287fd424`; rank0 `242783259d942c3847764c966cac1f3e077ae062a326bb702b7866330fcf5efc`; rank1 `c8aff16cd22b0264d067872c3d8df457bba30f1008815765c2cb999a5216aaf8`.

The full model/Adam/scheduler/scaler/RNG/accounting/team lineage was restored together. No model-only graft or new diagnostic optimizer was used in actual PPO runs. All experiment ledgers/checkpoints are isolated. Original campaign rolling pointer remains 4,096; the separately reconstructed verified resumable recovery is57,344. Failed102,400 and every diagnostic boundary are excluded from production progress. The protected-file inventory and final verification cover original checkpoints, statistics and pointer.

## Migration repair and state integrity

Actual preflight at the starting fix rejected five executable files: `pa3_model.py`, `precision.py`, `v2_training_status.py`, `v2_evaluation_sidecar.py`, `v2_recompute_forensics.py`. Adding names alone was insufficient: original audited blobs, stale proof targets, native identity, dispatch scope and precision metadata also required strict repair.

The new versioned contract retains historical source `5ccc51b0cd4a49efb93018258b9750d5e804ae32`, independent audit `0b7253cfd915ff61c88248b1979bd8e254285ef0`, and reviewed selective-FP32 source93ebc1b. Every executable delta is checked against exact named Git blobs; both historical→target and reviewed→target differences are audited, so reverting to historical code cannot evade review. Validator AST is sealed with only its own digest literal blanked. Dirty executable trees, unknown files and wrong target SHA fail closed.

Receipts explicitly authorize effects FP16→FP32, canonical0→1024, secondary OFF→ON, AMP retries0→3, and an experimental gather→exact-row-v100-v1 transition for the benchmark branch. Delivered production profile retains gather→gather. Architecture, dataset and all learning hyperparameters remain identical. Independent GPU proof uses historical code with a separately installed FP32 effects linear boundary, matched captured input, and bit-exact loss/full gradients/full Adam/updated model. Legacy proofs and hashes remain preserved. Actual `prepare()` and `validate()` accept both original rank artifacts. Actual copied-artifact tampering of weights, Adam, RNG, manifest and missing/altered rank files is rejected; target, precision, source map, PPO hash, backend and statistics lineage tampering also rejects. Unit tests cover unknown executable and encoder edits, rollback escape, missing precision approval and unsupported contracts. See [compact evidence](v2_migration_stability_batch_evidence/).

## Genuine PPO validation and overflow repair

Corrected gather baseline completed **10 iterations /40,960 natural diagnostic matches**, full checkpoint restoration between runs, **368 successful optimizer steps**, **0 skipped**, and max recompute **2.6226043701171875e-6**. Both rank model and Adam digests agree at every boundary. All operational errors are zero and metrics finite. Maximum PPO epochs stays2; existing target-KL.03 early stopping legitimately ends some iterations after1 epoch.

A real iteration17 tail exposed AMP overflow at scale65536 with four valid rows on rank0 and an empty peer. Two parameter gradients overflowed while forward loss stayed finite. The initial run correctly stopped without publishing that failed update. Exact captured-tail replay at scale32768 was finite. The repair retries the **same collectively synchronized minibatch** after ordinary scaler backoff, at most3 times; no reshuffle, extra samples, Adam update on bad gradients or duplicate metrics. Dropout-free model and enabled scaler are required. Persistent failure still raises. Real two-rank regression tests cover asymmetric overflow, empty/uneven peers and exhausted retries. Baseline ten-iteration run needed **one retry**.

The integrated optimized executable additionally completed **4 iterations /16,384 matches**, **144 successful steps**, **1 retries**, **0 skips**, max recompute **1.90734863281e-06**. Weighted integrated sustained throughput: **45.34534 matches/sec**. This evolving-policy measurement is separate from fixed-start A/B results.

Both baseline and integrated cooperative save/restore were **BIT_EXACT** against uninterrupted continuation for full learner state (model, Adam, scheduler, scaler/config), owned RNG streams, accounting, collector restart, dataset/opponents and rank digests. The forced real secondary audit **REJECTED BOTH RANKS BEFORE PPO**, despite primary maxima9.54e-7/1.43e-6. On rank0, FP16 replay -5.803071 differs from stable FP32 -5.800765 by0.002306; rank1 has a four-branch difference about0.003756. Original inputs and stored replay hashes verify; actor-repeat is exact; FP32 grouped/isolated differences are only a few e-6. This is FP16-versus-FP32 absolute precision drift, distinct from the fixed Actor/Learner grouping mismatch. See SECONDARY_BLOCKER.json and raw rank logs. The failed2,048-match diagnostic adds **zero accepted matches/optimizer updates/checkpoints**. Per-stratum mean/p99/max are preserved in successful run reports. Primary tolerance remains.001; secondary evidence rule remains unchanged and forced audit rejection now stops even when primary passes. Previously failing `kind1_branches2_actor1` is checked in real reports and both preserved failure reproductions. Historical failed rollouts remain rejected; source artifacts are never rewritten.

All eight optimized rank checkpoints were scanned: model/Adam/scheduler/scaler tensors are finite. At all four full-size boundaries, learner numerical state and owned RNG are bit-exact with the corrected gather baseline; see `integrated/FINITE_STATES.json` and `integrated/BASELINE_NUMERICAL_PARITY.json`. This does not turn the rejected secondary audit into a pass.

## Matched performance panel

Each B/C/D/E arm has **three independent full-state restores** from the same57,344 checkpoint, 2,048 natural games (1,024/rank), envs1024, workers16, PPO2, global4096/per-rank2048/micro1024/accum2, same teams, policy, seed and CUDA settings. All arms produce exactly the same rollout counts/action kinds, final model and Adam digests and optimizer steps. This quota differs explicitly from the4,096-game stability iterations. Publication interval100,000 avoids measuring artificial frequent checkpoint copies. Iteration rate includes prepare, recompute, digests and barriers; all-in includes launch/shutdown. Sequential arms are not randomized confidence intervals; raw ranges show observed variation.

A historical unsafe FP16-effects path is used **only for inference microtiming/reproduction**, never PPO. Its original actor/sample discrepancy reproduces; paired seven-sample CUDA-event timings are in `integrated/forensics_rank*.json`.

| Arm | Executed SHA | Raw iteration matches/sec | Median | Range |
|---|---|---|---:|---:|
| B | `0045e32f6d7e0981339e1d28fa5944cbc7e33fd9` | 37.21175, 36.75315, 37.61506 | 37.21175 | 36.75315–37.61506 |
| C | `5d9f58fa0ecf38f232eb92d36e09e2dc2edcb5de` | 39.71671, 39.77498, 39.64649 | 39.71671 | 39.64649–39.77498 |
| D | `e18b386a2a12cf00b1a204c690427fc9ed29e5c5` | 39.99471, 40.26158, 40.01392 | 40.01392 | 39.99471–40.26158 |
| E | `163a7c3e3a2af219f98de017d1fdb1b636d824e8` | 41.69974, 41.76978, 42.04183 | 41.76978 | 41.69974–42.04183 |

B: corrected baseline plus identical instrumentation and forced-audit safety repair. C: Actor provenance serialization optimization. D: Learner exact-order row lookup. E: combined. E candidate median improves **12.25%** over B, but is **not promoted**. The retained Actor-only C configuration measures **39.71671 matches/sec**, **6.73%** above B; the delivered production profile retains gather. The following optimized column describes experimental E, not a production-readiness claim.

| Metric | Corrected baseline median | Optimized median | Difference |
|---|---:|---:|---:|
| Actor useful batch utilization, rank0 | 27.63913% | 27.63913% | +0.00000% |
| Actor useful batch utilization, rank1 | 31.51042% | 31.51042% | +0.00000% |
| Actor policy rows/sec | 1754.16229 | 1895.98512 | +141.82283 |
| Collector natural matches/sec | 77.83271 | 84.12543 | +6.29272 |
| Learner epoch row visits/sec | 4240.92444 | 5056.77265 | +815.84821 |
| PPO update wall, seconds | 21.76742 | 18.25552 | -3.51191 |
| End-to-end iteration matches/sec | 37.21175 | 41.76978 | +4.55803 |
| Launcher all-in matches/sec | 30.53712 | 33.00081 | +2.46369 |
| Peak reserved GPU0 GiB | 11.29688 | 10.74219 | -0.55469 |
| Peak reserved GPU1 GiB | 11.31250 | 10.75586 | -0.55664 |
| Actor encoder kernel latency in live collection | N/A | N/A | Host timers are not CUDA kernel timing |
| Primary recompute max | 1.43051147461e-06 | 1.43051147461e-06 | unchanged |
| DDP model/Adam digest parity | exact | exact | unchanged |

Actor policy rows/sec is useful encoder rows divided by slowest-rank collection wall; learner rows/sec counts two epoch visits per row divided by slowest-rank update wall. Peak values are CUDA **reserved** VRAM, not allocated tensors or nvidia-smi occupancy. Actual batch histogram, padding counts and five ragged-family record distributions are retained. Kernel occupancy and live per-module encoder latency are N/A. Single-GPU captured-row profiles separately record synchronized CUDA-event full-step timings, forward/backward/attention operator scopes, optimizer and synchronization timing; synthetic advantages/fresh Adam are declared and are not games/sec evidence.

## Accepted and rejected candidates

Actor C preserves every provenance byte and copied-payload ownership. Per-call metadata headers are reused; trusted canonical candidate arrays use their equivalent nested integer representation. No persisted validation cache or borrowed mutable memory is introduced. Collection median improves26.31→23.43s; full C median37.21→39.72 matches/sec, above observed B variation. CPU probe and independent legacy hash tests cover noncontiguous/endian/empty arrays, mutation, NaN/Inf and custom sequence subclasses.

Experimental Learner D tests the **existing** exact-order V100 token-reference gradient reducer, previously measured only with fresh diagnostic Adam and one repetition. Stable ordering/no atomics preserves tested gather gradients bitwise. New full restored-state three-run comparisons and explicit backend migration establish bounded continuation evidence, but production activation is held back by the secondary audit blocker. Runtime/device/compiler guards remain; higher-order autograd is unsupported. Plain flat embedding remains rejected. Physical encoder1024, learning micro1024 and batch semantics remain unchanged.

Structured native-record concatenation/vectorization was locally slower (old/new speed ratio0.70–0.86) and rejected. Owned-export reuse was a small local gain insufficient for promotion. Memoryview hashing was rejected for empty-buffer/ownership behavior. Existing CUDA rollout cache, compact candidates, bundled H2D and ownership optimizations were inspected and retained rather than duplicated. Canonical padding utilization is low (~28–32% in the matched actor workload), but removing canonical1024 or merging sequential requests lacks full safe speed evidence and is not promoted. Recorded packing/hash CPU costs and deterministic gather backward were more direct opportunities; engine step is a small fraction. Existing micro2048 experiments show higher activation/cache pressure without consistent speed, so no effective batch change is made.

Preserved failures are rechecked across original actor/sample grouping,1/17/576/1024/1025 logical rows, first/last row positions and full/partial canonical chunks. Different padding fractions retain the same.001 gate. No legal candidates, tokens, valid gradients, masks or branch dependencies are removed.

Historical ~90 games/sec is **not a comparable baseline**. Local [earlier analysis](../perf/V2_PPO_BOTTLENECK_REPORT_2026-10-14.md) records A5 gate-off92.57 all-in and A9.3 ~89.45 steady with older contracts and18 steps/2,048-game iteration; V2 has corrected observation/action handling, different policy battle lengths/rows per match and more update work. No90 recovery or hardware ceiling is claimed.

## Tests, readiness and delivery

**Final verification:402 passed /35 CUDA skips in85.45s; separate isolated CUDA suite86 passed in5.37s.** CPU evaluation modules hide CUDA during full-suite collection, so all CUDA variants were explicitly rerun in a fresh process. Logs are FINAL_TESTS.log and FINAL_CUDA_TESTS.log. Source-of-truth/profile validation PASS; all89 protected production files remain byte-identical. Full relevant agent/integration suite and explicit CUDA parity tests are recorded in evidence logs. Earlier full run was382 passed/35 skipped excluding stale source-of-truth facts; final facts are synchronized only after integrated validation. One initial invalid suite loaded an accidentally mismatched native binary while that binary was replaced; its failure log is preserved, and clean frozen-native reruns supersede it. One GPU proof attempt compared different rank captures and was rejected before training; matched-input proof passes, and proof preparation now checks input hashes first.

Recommended runtime: versioned `configs/pa3_v2_production.yaml`, unchanged learning settings, canonical1024, effectsFP32, primary.001, secondary enabled, retry3, gather backend, CUDA cache/compact candidates. Exact lookup remains an isolated benchmark candidate. Do not restart production until the secondary blocker is resolved. Any future authorized restart must use verified57,344 with the **final HEAD-specific receipt**, original full state and appropriate isolated/new run lineage. Never resume the failed102,400 boundary or count diagnostics as production progress.

The ordinary primary-gated continuations pass, but **production resumption is NOT approved** while the real secondary audit fails; it does not prove indefinitely stable learning, unseen-team strength, another CUDA/Torch/device version or faster dynamic batching. A production-readiness blocker remains: the unchanged strict secondary evidence rule rejects absolute FP16/FP32 drift. It must be resolved under a separately verified precision contract before claiming all audit gates pass. Raising tolerance, ignoring forced failures, or switching the whole encoder to FP32 would violate this assignment’s retained contract and is not done. Actual production remains stopped as requested. Git delivery uses a normal dedicated-branch push, no force push or main merge. Large checkpoints, gradient captures and profiler traces remain local; compact raw reports and hash receipts are committed.

## Supplementary synchronized GPU profile

Captured-row single-GPU1024-row synthetic-loss update, three post-warmup samples per backend: gather CUDA-event milliseconds `[527.37841796875, 529.470458984375, 535.5048217773438]`, exact `[384.1679382324219, 380.3617248535156, 388.1994323730469]`; medians **529.470→384.168 ms**. These are isolated microtimings, not independent games/sec repetitions. Full operator and kernel attribution is retained in the profile summaries.

Matched learner useful-row utilization is **93.907%** of the fixed padded execution slots in each epoch; two micro1024 forwards per rank minibatch and2 epochs remain unchanged. Baseline rank0 host stage timers: forward9.487s, backward8.277s, optimizer2.947s, minibatch select0.515s; cache build3.681s. Host stage timers include synchronization/launch effects and are not standalone kernel times.
