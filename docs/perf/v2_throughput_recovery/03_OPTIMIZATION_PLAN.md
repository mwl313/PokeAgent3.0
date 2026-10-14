# Implemented changes and remaining gates

1. CanonicalCandidateBlock: validate the entire native branch batch once;
   compute flat entity/move references against actual per-row active maps;
   keep owned bytes-backed readonly arrays and offsets. Actor padding and row
   storage reuse that block. Each branch prefix receives a new block; no
   cross-request/cache identity shortcut. Raw callers retain strict validation.
2. BlockObservationStore: own all fixed arrays and deep-copy all five ragged
   payloads once per producer batch. Store (block,row) references instead of
   one dictionary/array set per decision. Read paths gather independent fixed
   arrays and reconstruct owned ragged tensors. Columnar option remains opt-in;
   the ordinary native path uses the block implementation without changing
   the collector's mathematical/configuration contract.
3. Ragged select: compute gather indices in one vectorized operation, retaining
   repeats, order, empty rows and invalid-index rejection. Offsets are recomputed
   from current counts; no unsafe mutable proof cache.
4. Source migration: exact old/new revisions, allowed runtime file/blob list,
   checkpoint and parity evidence hashes. Model/loss/optimizer/schema/engine/
   dataset/config changes are prohibited. Receipt is bounded-validation-only;
   launcher refuses production profile or more than5 iterations with it.

Rollback is the untouched frozen runtime and original generation. Neither
production source nor originals are edited. No training is started from
random initialization. Micro2048, NCCL changes, find_unused_parameters removal,
model/feature removal, shuffle changes and tolerance changes are not promoted.

Validation order: ownership/action unit tests → independent captured-input
CPU representation/loss/gradient/Adam parity → clean committed source → exact
bound migration →3 independent full-stack repetitions ×5 iterations per arm
where correctness permits → CUDA trace → bounded checkpoint continuation.
Promotion requires repeatable total improvement and every mandatory gate.

Closeout: the user stopped recovery around50k replay games. A full57,344-match
checkpoint (718 cumulative optimizer steps) is now preserved, with model/Adam
matching the historical trajectory at all13 replayed boundaries and28 team
receipts reconciled. The latest73,728 model-only evidence remains preserved;
that latest boundary's full optimizer/RNG state has not been reconstructed.
The recent historical RNG was unavailable for direct comparison, so replay
does not establish universal bit-exact recovery. No production restart.

Remaining, NOT scheduled after today's stop:
1. Corrected-workspace3×5 source-pinned A/B and two-rank stop/restore test.
2. Repair the demonstrated long-run recompute failure without relaxing1e-3.
3. Complete historical recovery to73,728 only if authorized for a later session.
4. Profile/experiment with deterministic gather backward accumulation, then
   preparation/H2D; micro2048 only as a separate bounded comparison.
5. Reconcile the approved boundary/source migration and obtain cutover approval.

These are pending tasks, not an automation or permission to resume overnight.
