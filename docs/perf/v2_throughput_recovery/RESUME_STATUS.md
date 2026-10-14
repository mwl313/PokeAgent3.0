# Active work handoff — not final acceptance

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

Remaining: corrected-runtime3×5 A/B, CUDA/DDP traces, bounded continuation,
recovery outcome, evidence/CSV/final report, pinned validated runtime, commit/push.
Long-run ModeD recompute failure atiteration19 remains a production blocker;
data-path parity is not a precision repair. No cutover without full gates and
explicit approval. Preserve all local model/checkpoint files.
