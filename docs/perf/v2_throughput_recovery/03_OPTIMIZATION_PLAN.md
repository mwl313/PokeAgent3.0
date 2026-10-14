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

Known pre-existing blocker: full73,728-match optimizer/RNG state was never
persisted. A safe campaign cutover requires deterministic recovery/replay of
that boundary or explicit user choice of an older exact checkpoint/model-only
continuation. Do not silently reset optimizer or counters, and do not call
a model-only artifact a resumable checkpoint.
