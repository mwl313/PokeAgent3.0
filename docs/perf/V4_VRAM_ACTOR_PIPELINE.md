# V4 VRAM + actor-pipeline status

Current clean-HEAD measurements (raw JSON under `runs/perf/v4/`):

| quantity | measured |
|---|---|
| single-GPU actor-only rate (2k panel) | 88.3 games/s median (A0 report) |
| single-GPU all-in (2k, corrected) | 20.53 games/s median |
| dual all-in (2k total, DDP) | 24.67 games/s median |
| actor VRAM (single, micro 1024) | 9.25 GiB reserved of 28 GiB budget |
| actor H2D per minibatch | 3.0 s / 56 minibatches |

## Rolling slots (P3) and cohort pipelines (P4)

Not implemented in this pass. The decision is measurement-driven: the F0 atlas
shows the **learner update is 95.4 % GPU-busy**, so rolling-slot/cohort
overlap inside the actor loop can only pay off where the learner GPU is idle —
i.e. during the collection phase (23 % of all-in) and in the materialization
gaps. The plan's requirement for a rolling-slot change (natural match identity,
terminal/drain semantics, no double count, no dropped long match) is captured
as the acceptance test `test_rolling_slot_equivalence.py` but the
implementation is deliberately deferred until the collection/learner overlap
experiment (P4) is scheduled, because P3 alone does not remove the phase
boundary.

## Buffer materialization (P5)

Streaming materialization costs 13.3 s of 101 s all-in (13.2 %). Its measured
split is observations 62 %, candidates 31 %, columns 4 %. The neutral
experiments (one permuted copy per epoch, preallocated observation fill) were
reverted; the productive direction is columnar per-field slabs written by the
collector so a minibatch is one fancy-index gather per field, which is tracked
as the next P5 step.
