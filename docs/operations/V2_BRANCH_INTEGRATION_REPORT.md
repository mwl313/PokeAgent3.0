# Final branch integration

**READY_FOR_CUTOVER — 2026-10-15 KST. Production remains STOPPED at the user's explicit request.** Canonical `codex/pa3-v2-correctness` was ancestry-preserving fast-forwarded and pushed to the tested executable commit `0b7253cfd915ff61c88248b1979bd8e254285ef0`; subsequent changes are documentation/evidence only. No main/C-track change, destructive cleanup or production launch.

Throughput direct ancestry preserved (c96c35f→9e685b7), then strict safety/migration fixes3a6849e,d98b42c,0b7253c. No squash/rebase/force-push. Model/engine semantics and datasets unchanged; engine/native hash identical. Main andC-track unchanged; user untracked docs/audit_v2/recompute_failure_20261014 in canonical worktree preserved. CPU/data performance paths integrated; experimental row lookup and exactCUDA backend remain OFF.
