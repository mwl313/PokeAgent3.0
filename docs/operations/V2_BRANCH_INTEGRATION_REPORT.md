# V2 integration inventory and decision

**2026-10-15 KST — BLOCKED; PPO not restarted.** Canonical branch remains `codex/pa3-v2-correctness @ c96c35f12d3f3aff32b072b48e32ac243a7b0425`. Integration candidate is `codex/pa3-v2-integration-release`, based on frozen throughput `9e685b701f170436f9b9dd82162c8e192e4e6904`. No canonical fast-forward is authorized by failed release gates. Main and Track C unchanged.

Local/remote refreshed with inspected GitHub origin; main local089a3ab differs from origin/main50c597b. Worktree/refs/status/stash/ignored/disk inventory: evidence/INVENTORY_BEFORE.json and DESKTOP_ROOTS.json. 20 registered entries (one missing/prunable historical entry),72 local/remote refs,19 project Desktop roots. Dirty user changes preserved. No reset/clean/deletion/force-push.

V2→throughput ancestry is0 behind/16 ahead; candidate preserves all those commits. Full commit ledger: `git log --oneline c96c35f..9e685b7`. Throughput changes include5b2658e,15c125a,5752e60,2b3c93c,8a743ad,b3c189b,40e6604,2a3bdcd,1d89863,1b4be6f,3b4f178,e9618c2,c34d782,5990235,6a2a693,9e685b7. They are staged in the candidate ancestry, **not integrated into canonical**.

Candidate mapping/signed target/roster mapping/owned blocks/lazy immutable candidate views/ragged vectorization/prefix conversion/caching/compaction have targeted tests and historical parity receipts. CPU paths are candidate release features, not production-promoted. Plain flat-row lookup and exact-order V100 lookup remain EXPERIMENTAL/OFF; even newer exact diagnostic parity is not production promotion. No Track C merge or modules imported.

Canonical promotion decision: BLOCKED pending mandatory safety and continuation gates. Candidate may be pushed for review; target branch must stay atc96c35f. Research diagnosis branch was inspected through existing V2 safety code; no wholesale historical branch merge.

1. Historical iteration19 discrepancies (rank0 .0011582374572753906; rank1 .0017061233520507812) have **zero actor groups** in their saved evidence. Original-group replay/repeat/FP32 proof cannot be reconstructed from sampled rows alone. Secondary must reject; no harmlessness claim.
2. Full57,344 checkpoint records secondary=false and old source5ccc51b. Candidate requires secondary=true; no validated exact-source production migration exists. Existing bounded migration whitelist rejects protected model/observation/config changes. No bypass added.
3. Broad agent suite has unresolved failures, including candidate-padding scalar tolerance. Original-baseline reproduction is retained; passing short tests does not certify sustained release.
4. Production-shaped4096-game matched A/B, beyond-iteration19 continuation, merged-source DDP save/restore equivalence and actual evaluation schedule reconciliation are **NOT EXECUTED**. Release remains blocked.
Disk space is excluded as a release blocker per the user's later instruction. Existing resource-related test failures remain recorded honestly; no cleanup or guard disabling was performed.
