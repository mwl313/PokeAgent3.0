# Existing-model continuation and rollback

**2026-10-15 KST — BLOCKED; PPO not restarted.** Canonical branch remains `codex/pa3-v2-correctness @ c96c35f12d3f3aff32b072b48e32ac243a7b0425`. Integration candidate is `codex/pa3-v2-integration-release`, based on frozen throughput `9e685b701f170436f9b9dd82162c8e192e4e6904`. No canonical fast-forward is authorized by failed release gates. Main and Track C unchanged.

Selected complete recovered generation: `/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699`. Manifest SHA256 `b9539f3a0c426f0878375b61630cb4de9dcd2a5d4a6a59858b4d64f4287fd424`. Source `5ccc51b0cd4a49efb93018258b9750d5e804ae32`, iteration14,57,344 cumulative matches. Both rank artifact hashes verified against manifest; envelope rank/world/clock/config checks and28 receipt reconciliation are recorded in evidence/checkpoint_audit.json. Local backup rank hashes also match. These copies are on the same disk, not proven off-machine backups.

73,728 artifacts are model-only failure evidence; never combine them with older Adam/RNG. Original campaign rolling/LATEST remains4,096. Recovery RNG is historically replayed from owned initial RNG; original recent RNG is unavailable for independent direct comparison. Do not claim bit-exact original historical continuation.

Required next step: reviewed exact-source bounded migration receipt covering each changed runtime blob and secondary=false→true activation, independent objective/gradient/Adam/data proof, then isolated two-rank save/restore comparison and controlled continuation through iteration20 or later. Preserve the original full checkpoint/receipts. Source contracts must reject unsupported changes; no generic allow-source escape. DDP communicator recreation is mandatory.

After all gates and protected cutover approval: generate a release manifest, pin source/profile/binary/dataset/checkpoint hashes, reconcile receipts, publish through generation protocol, use the existing original release/systemd_command.json as the audited command template with reviewed exact replacements. Do not execute that original scratch launch command to resume. It is not a ready continuation command today.

Rollback: keep frozen5ccc51b runtime and57344 generation unchanged; stop any future approved trainer cooperatively using its configured stop file, wait for both ranks and validated publication, then select the last verified generation via the storage API. No manual LATEST overwrite or counter reset. Evaluation registry must remain independently versioned.


Additional executed verification: both rank learner payloads were loaded sequentially on CUDA0, with exact model/Adam digests, scheduler, AMP scaler and Python/NumPy/torch CPU/CUDA/owned CPU shuffle RNG restoration (evidence/state_load_verification.json). This is serialization fidelity, **not** two-rank communicator/collector continuation equivalence. Probe initially constructed an owned CUDA generator incorrectly; correcting it to the launcher's actual CPU shuffle generator allowed faithful restoration. No checkpoint contents changed.
