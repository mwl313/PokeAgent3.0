# AHL V2 Stage D — rollback plan (DESIGN ONLY)

Status: **PROPOSED / NOT EXECUTED**. Rollback must preserve the frozen
production lineage and all formal evaluation state.

## 1. Triggers

Stop the pilot and roll back if any of the following occurs:

- migration contract or plan hash mismatch between ranks;
- model or optimizer digest mismatch before the first optimizer step;
- unexpected LR, scaler, counter or schedule change;
- historical rows appear in PPO loss/advantages;
- rank digest divergence, skipped optimizer step or duplicated natural match;
- secondary precision gate failure;
- formal evaluation backlog appears (formal promotion always wins);
- disk reserve below 12 GiB;
- throughput falls below the approved mode-specific floor;
- any operator or owner instruction to stop.

Use a cooperative stop at a safe boundary. Never SIGKILL a running training
process.

## 2. Evaluation-side rollback

If only the diagnostic/shadow side fails:

1. Stop the isolated round-robin or shadow process (plain process, own lock).
2. Leave the formal V2 evaluation worker, schedule, receipts, alpha ledger,
   champion pointer and snapshots untouched.
3. Keep the diagnostic receipts for post-mortem; mark the run `ABORTED` in
   `status.json`.
4. Re-run harness equivalence before any future diagnostic work.

This rollback has no effect on promotion or training.

## 3. Training-side rollback

If the pilot arm is unhealthy:

1. Stop at a safe cooperative boundary; let collection drain and both ranks
   publish consistently.
2. Do not transplant optimizer state from the mixed-history arm into the
   current-only branch.
3. Return to the untouched frozen generation
   `iter-000002764800-1792212738467610757` and the frozen source `da35700`
   with the pinned profile `c816aa28...`.
4. Verify restored:
   - committed matches 2,764,800;
   - iteration 675 and optimizer steps 24,573;
   - LR `0.00014978839419625816`;
   - AMP scaler `262144`;
   - rank0/rank1 SHA256 values;
   - model and optimizer digests;
   - rank digest agreement;
   - team-stats accounting and natural-match counters.
5. Keep `STOP` in place until a separately approved restart.
6. Re-verify the power-off guard separately; do not toggle it as part of
   rollback.
7. Record an immutable rollback receipt referencing the failed contract hash,
   the stop boundary, the restored generation and the verification result.

## 4. Data and namespace rules

- Never modify the original checkpoint payloads or the paused production
  namespace.
- Never overwrite a failed pilot's receipts; write new rollback evidence.
- Do not reuse mixed-history team-stat roots for the restored current-only run.
- Do not consume or restore formal alpha as part of rollback.
- Do not delete branches, tags or remote refs.

## 5. Re-entry criteria

A new pilot may be proposed only after:

- the root cause is identified and documented;
- the migration contract is revised with a new contract hash;
- the affected tests are added or strengthened;
- a fresh preflight verifies the frozen baseline and formal evaluation priority;
- the owner approves the revised contract.

## 6. Reporting

The rollback report must state explicitly:

- what was stopped and at which safe boundary;
- which files and hashes were restored;
- whether any protected file changed (expected: none);
- whether formal evaluation was affected (expected: none);
- the final `NO_GO` or re-entry recommendation.
