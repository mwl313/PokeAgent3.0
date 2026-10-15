# Recompute safety release

**2026-10-15 KST — BLOCKED; PPO not restarted.** Canonical branch remains `codex/pa3-v2-correctness @ c96c35f12d3f3aff32b072b48e32ac243a7b0425`. Integration candidate is `codex/pa3-v2-integration-release`, based on frozen throughput `9e685b701f170436f9b9dd82162c8e192e4e6904`. No canonical fast-forward is authorized by failed release gates. Main and Track C unchanged.

Candidate profile now explicitly requires recompute_secondary=true and primary tolerance.001. Resolver passes secondary through launcher_values; parent already propagates CLI to workers; collector already records actor groups when secondary=true; envelope already records the flag. Resume now rejects a changed secondary contract rather than silently adopting it. Profile hashes necessarily change; immutable original runtime profile is untouched.

Existing rule: actor replay/stored and replay-repeat <=1e-4; FP32 group/isolate <=1e-4; actor, batched and isolated AMP each within.001 of its independent FP32 reference. At most8 flagged rows; evidence mismatch, corruption, nonfinite, missing group or exhausted budget rejects. Primary pass does not invoke secondary unless explicitly requesting audit. No old-logprob edits, PPO loss change or tolerance increase.

Added audit input-length/budget rejection and tests. Existing tests cover replay, observation/mask/prefix corruption, missing metadata, RNG/model/stored-logprob immutability and pure evidence decisions. Native Imprison retry tests executed separately. Saved failure evidence uses existing atomic envelope publication.

Historical failures remain unexplained without provenance; short tests do not close long-run safety. Candidate does not run isolated expensive audits on every sample.

1. Historical iteration19 discrepancies (rank0 .0011582374572753906; rank1 .0017061233520507812) have **zero actor groups** in their saved evidence. Original-group replay/repeat/FP32 proof cannot be reconstructed from sampled rows alone. Secondary must reject; no harmlessness claim.
2. Full57,344 checkpoint records secondary=false and old source5ccc51b. Candidate requires secondary=true; no validated exact-source production migration exists. Existing bounded migration whitelist rejects protected model/observation/config changes. No bypass added.
3. Broad agent suite has unresolved failures, including candidate-padding scalar tolerance. Original-baseline reproduction is retained; passing short tests does not certify sustained release.
4. Production-shaped4096-game matched A/B, beyond-iteration19 continuation, merged-source DDP save/restore equivalence and actual evaluation schedule reconciliation are **NOT EXECUTED**. Release remains blocked.
Disk space is excluded as a release blocker per the user's later instruction. Existing resource-related test failures remain recorded honestly; no cleanup or guard disabling was performed.

Final focused suite includes native retry, secondary safety, profile/restore contract and SoT negative tests. See final_safety_tests.log for executed count. Original checkpoint secondary=false now explicitly fails a requested=true restore contract in a bounded unit test. New-source long-run safety is not certified.
