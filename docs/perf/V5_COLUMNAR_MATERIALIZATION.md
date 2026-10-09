# V5b — columnar materialization + candidate dtype (T2)

## 1. Observations (62 % of the materialization block) — landed behind the flag

`ColumnarObservationStore.stacked_indices` replaces the per-minibatch
`np.concatenate` over thousands of one-row arrays with one numpy fancy-index
gather per field. Measured effect inside the T1 A/B:

| metric | baseline | columnar | median delta |
|---|---:|---:|---:|
| materialization block (`minibatch_select`, 56 minibatches) | 7.60 s | 6.98 s | **−8.2 %** |
| update wall | 42.22 s | 41.38 s | −2.0 % |
| all-in | 32.502 games/s | 33.026 games/s | +1.6 % (within ±2 % noise) |

Gates: the streaming-equivalence suite and 86 tests pass; the A/B runs pass the
recompute gate and digest parity (see `V5_COLUMNAR_RECORD.md`).

## 2. Candidate u8 (31 %) — NOT ATTEMPTED, reported separately

Decision: keep dense `int64` candidates for now. Reason: the dense candidate
tensors feed `nn.Embedding` lookups in the scorer (`action_ids`, `entity_token`,
`move_token`), which require `int64`/`int32` indices; a `uint8` tensor would
force a cast back at every consumption site and change an attention/scatter
path with no measured evidence that the copy — not the gather — dominates.
Per the task's split rule this is reported as its own decision instead of being
landed blind. The next candidate step is the preallocated slab already used by
`BranchCandidatesBatch.from_rows` (which does fill a preallocated table per
minibatch); an end-to-end win there must clear the same ±2 % noise floor and
the same gates.

## 3. Decision — flag stays off (default unchanged)

The materialization win is real (−8 % on that block, −2 % update) but the net
all-in is inside noise because the record path regressed. Both columnar pieces
remain behind `--columnar-store`; the default path is unchanged. Reverting is
trivial (flag off) and no semantics changed: row SHA, stats and digests are
identical in every gate.
