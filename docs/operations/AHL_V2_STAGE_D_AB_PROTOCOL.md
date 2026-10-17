# AHL V2 Stage D — isolated A/B protocol (DESIGN ONLY)

Status: **PROPOSED / NOT RUN**. This protocol requires owner approval and the
reviewed migration contract. It does not authorize a training restart.

## 1. Question

Does a 5% current-policy learner-row history mix improve or preserve broad
policy strength without an unacceptable throughput or stability regression,
compared with the same checkpoint continued at 0% history?

The numeric mix is a pilot hypothesis, not a proven optimum.

## 2. Isolation

Both arms start from the **same verified checkpoint**:

```
generation iter-000002764800-1792212738467610757
model digest     7697797df97cf3c6d5c11466c294c8a72be1bf06d437fb1dba0c8db5109f2e6e
optimizer digest bafeee072e87845b4dc7837d485efcf1ac79de70877fe051684dcab440d0e4c6
```

- Two distinct integration worktrees and namespaces, neither of which is the
  frozen production namespace.
- Fresh team-stats roots.
- No shared checkpoint publication directory.
- No GPU concurrent with the formal evaluator; the evaluator keeps priority.
- No changes to `STOP`, the power-off guard, the formal alpha ledger, the best
  policy pointer, the formal schedule or formal receipts.

## 3. Arms and budgets

| Arm | History rows | Expected history games |
|---|---:|---:|
| A (baseline) | 0.00 | 0.00 |
| B (pilot) | 0.05 | ~0.0952 |

Equal **committed natural-match** budget and, where feasible, equal
**decision-row** budget. Report wall-clock separately. The first run is a
bounded cohort (for example 200K matches per arm); a second cohort is required
before any 10% proposal.

Both arms use the same:

- PPO-2, LR schedule, clip 0.2, GAE 0.95, entropy and KL coefficients;
- reward and team pool;
- Mode E precision path and secondary recompute gate;
- rank count, batch layout and seed policy.

## 4. Preflight

1. Verify both arms restore the exact model and optimizer digests before the
   first optimizer step.
2. Verify no LR warmup/rewarm and no scaler reset.
3. Verify the frozen model inventory and plan hash in both ranks.
4. Verify the 0% arm reproduces the frozen current-only baseline behavior.
5. Verify the 5% arm's historical rows are excluded from the loss.
6. Verify the formal evaluator has no pending/running job; otherwise defer.

## 5. Evaluation

Evaluate both arms on the same frozen fixture panels and independent
evaluation seeds. Report:

- paired score versus the common starting checkpoint;
- score versus the frozen champion and versus the broad historical anchors;
- worst-opponent score and non-transitivity coverage;
- Elo-like change per decision row, per natural match and per wall-clock hour;
- pair-cluster confidence intervals, not independent-game intervals;
- throughput: games/s, learner rows/s, model wall, VRAM;
- stability: KL, clipping, entropy, AMP scale, skipped steps, operational
  errors, secondary precision gate, rank digest agreement.

The registered V2 training teams are all used for learning. New fixture seeds
are **not** unseen team rosters; any claim of held-out-team generalization
requires separately sourced and approved rosters.

## 6. Predeclared decision rule (proposal)

Promote the 5% pilot only if all of the following hold:

1. No material regression against the common starting checkpoint on the broad
   anchor set.
2. Worst-opponent score is preserved or improved with a paired interval that
   does not support a material regression.
3. Throughput loss is within the approved mode-specific budget and the
   secondary precision gate passes throughout.
4. No rank divergence, skipped step, duplicated match or counter drift.
5. The improvement is reproduced in a second cohort or an independent seed
   set. A single noisy head-to-head win is not evidence.

If the result is inconclusive, keep 0% history. If it is negative, stop the
pilot and roll back. A positive 5% result only permits a proposal for 10%; it
does not authorize an automatic increase. The 20% ceiling remains manual and
requires a new controlled experiment.

## 7. Reporting artefacts

- `AB_REPORT.md` with arm identities, contract hashes, budgets, elapsed time,
  realized `r`/`g`, per-opponent rows, evaluation tables and intervals.
- Raw telemetry and receipts retained immutably in the isolated namespaces.
- A machine-readable summary with the same pinned hashes.
- An explicit `GO` or `NO_GO` for the 10% proposal, never for production
  activation by itself.
