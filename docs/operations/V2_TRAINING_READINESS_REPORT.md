# Training readiness

**2026-10-15 KST — BLOCKED; PPO not restarted.** Canonical branch remains `codex/pa3-v2-correctness @ c96c35f12d3f3aff32b072b48e32ac243a7b0425`. Integration candidate is `codex/pa3-v2-integration-release`, based on frozen throughput `9e685b701f170436f9b9dd82162c8e192e4e6904`. No canonical fast-forward is authorized by failed release gates. Main and Track C unchanged.

1. Historical iteration19 discrepancies (rank0 .0011582374572753906; rank1 .0017061233520507812) have **zero actor groups** in their saved evidence. Original-group replay/repeat/FP32 proof cannot be reconstructed from sampled rows alone. Secondary must reject; no harmlessness claim.
2. Full57,344 checkpoint records secondary=false and old source5ccc51b. Candidate requires secondary=true; no validated exact-source production migration exists. Existing bounded migration whitelist rejects protected model/observation/config changes. No bypass added.
3. Broad agent suite has unresolved failures, including candidate-padding scalar tolerance. Original-baseline reproduction is retained; passing short tests does not certify sustained release.
4. Production-shaped4096-game matched A/B, beyond-iteration19 continuation, merged-source DDP save/restore equivalence and actual evaluation schedule reconciliation are **NOT EXECUTED**. Release remains blocked.
Disk space is excluded as a release blocker per the user's later instruction. Existing resource-related test failures remain recorded honestly; no cleanup or guard disabling was performed.

Evidence: targeted initial89 passed; full agent304 passed/33 skipped/9 failed. Additional native/safety/drift test receipts are in evidence. Failed tests are not waived. Baseline subset reproduces8 failures; candidate-padding failure did not reproduce in the subset and remains order-sensitive/unclassified, not proven a candidate regression or harmless.

Historical correct-workspace clean6a2a693 same-source1500-game diagnostic: gather43.23565 vs exact47.32024 iteration games/s; all-in36.56656 vs39.41525. 33,533 rows and18 updates, yielding966.55 vs1057.86 rows/s and.519 vs.568 steps/s. Collection13.257/13.388s, preparation3.887/3.402s, update16.248/13.540s. Model/Adam/rollout digests match. These were weights-only/fresh-Adam diagnostics and experimental exact backend, not new-source full checkpoint performance. CPU/GPU memory and sustained rank skew not fully measured; do not invent them. No new production performance acceptance run occurred.

Evaluation intended cadence50k/200k differs from actual sidecar implementation: polls published checkpoints (normally100k cadence); saved command256 pairs/batch32, CPU-only. No new V2 competitive evaluation performed. Old A10 results are incompatible strength evidence. Reconcile before production.


Final clean-source receipt: `e9dc1d20141986d9e0e19234d26240a7d933d5d0`,23 focused tests passed in1.92s. Subsequent commit adds receipts only. Candidate is pushed separately; origin/codex/pa3-v2-correctness remainsc96c35f and C-track remains06dbb31. No training/evaluation process or GPU workload remains. Original rolling/LATEST is unchanged. Disk-space issue excluded per user.
