# Session closed at the user's bounded stop request

Do not launch more jobs automatically. The diagnostic recovery has exited
cleanly; queued tests and the resource sampler are cancelled. No production
or evaluation service was restarted. Final report:06_FINAL_THROUGHPUT_REPORT.md.

Full recovered boundary:57,344 matches,718 cumulative optimizer steps,
13 replay iterations/53,248 games after restoring4,096. Model/Adam match
original telemetry at every boundary; manifest/rank hashes,28 team receipts,
drain and additional local backup verified. It is not the full73,728 recovery.
See evidence/CLOSEOUT_RECEIPT.json and07_SAFE_CUTOVER_AND_ROLLBACK.md.

Final focused28 tests passed in2.33s on b3c189b. Corrected-workspace candidate
A/B, CUDA/DDP profiles and continuation were cancelled before launch. The
workspace-UNSET49.44% improvement remains exploratory, not production acceptance.

Next session, only when requested: corrected-workspace A/B/continuation,
long-run ModeD recompute repair, optionally finish73,728 state reconstruction.
No automatic overnight work or production cutover. Large artifacts remain
local; Git includes compact evidence and exact source identities.

## Prior in-progress handoff, retained as historical context

Original production source5ccc51b and campaign checkpoints remain unchanged.
Both production transient units were absent/not-found at task entry (no running
PPO or evaluator); do not restart them. Branch/worktree:
`optimization/pa3-v2-ppo-throughput`, `/home/minidc/Desktop/PokeAgent3.0-V2-throughput`.

Implemented canonical candidate blocks, owned observation blocks, vectorized
ragged gather, strict bounded exact-source migration and continuation probes.
Focused representation/loss/gradient/Adam tests pass. Seven broad-suite legacy
failures reproduce on original source; one near-zero padding scalar failure
is order-sensitive. No PPO/recompute tolerance was widened.

First exploratory three-by-five iteration A/B, without CUBLAS workspace env:
median28.29665→42.28747 iteration games/s (+49.44%), all-in28.01193→41.53477.
All rollouts/model/Adam/work match across arms. Source8a743ad. Bounded same-source
continuation is BIT_EXACT on learner/RNG/accounting/collector/rollouts at12,288.
These are **not production-runtime acceptance**: the original launch used
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, omitted by the first benchmark runner.

The corrected code pins/records/checks this environment and refuses conflicts.
Repeat the panel with that exact runtime before claiming production throughput.
All raw artifacts: `runs/perf/v2_recovery/`. Initial CPU parity receipt:
`CPU_PARITY.json`; new migration must match current clean committed HEAD.

GPU original exploratory trace:519.855ms leaf-kernel union/571.419ms span,
backward409.612ms; GatherBackward index-put111.523ms is the leading discrete
GPU opportunity. Its source, inputs and nonproduction workspace must be labelled.

Recovery script first probes one original-source iteration with original
ledger/workspace. Only if historical model+Adam digests match does it replay
17 iterations from4,096 to73,728 and save a separate complete boundary. It never
starts production or overwrites original generations. Additional immutable
orphan stats receipts may be appended under original ledger identity.

Previously planned, now cancelled: corrected-runtime3×5 A/B, CUDA/DDP traces, bounded continuation,
recovery outcome, evidence/CSV/final report, pinned validated runtime, commit/push.
Long-run ModeD recompute failure atiteration19 remains a production blocker;
data-path parity is not a precision repair. No cutover without full gates and
explicit approval. Preserve all local model/checkpoint files.
