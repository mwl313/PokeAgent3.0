# Track A8 — Batched Evaluation Report

Commit: `pending final A8 commit` (starting from `1b7d649`). No PPO training,
model changes, battle-mechanics changes, or Track C files were touched.

The evaluation league now has a v2 identity contract. It includes the actual
checkpoint and model digests, fixture manifest, native extension file, team
dataset, mechanics-data tree, evaluator version, backend/batch configuration,
and fixture/seat keyed sampling semantics. The old v1 receipts remain readable
under `legacy_receipts` but are never reused as v2 cache hits. JSON receipt
updates are protected by an advisory file lock in addition to atomic replace.

The scheduler rejects a newer checkpoint as the historical snapshot for an
older crossed boundary. Such a boundary is explicitly deferred with
`missing_snapshot=true`. A valid job records the exact checkpoint SHA and is
completed only after all planned panels have successful receipts with matching
job, boundary, candidate, fixture, identity, and game-count fields. The new
`process-pending` CLI exercises this lifecycle. A test override changes only
the number of games and is never the production schedule.

`agent/evaluation/batched_policy.py` is a separate native backend. It keeps
multiple handles active, uses `request_info_batch`, `observe_fixed_batch`,
`candidates_packed_walk`, and `step_batch`, groups model requests by policy and
branch count, and uses an independent fixture/seat/decision/branch keyed CDF
stream. Legal candidates are generated after each selected prefix. The
sequential evaluator remains available as the reference backend.

On MiniDC, the fresh bounded sequential reference was 32 games in 7.245 s,
or 4.42 games/s. Batched native runs of 128 complete games measured 23.59,
37.01, 43.78, and 62.38 games/s for batch sizes 16, 32, 64, and 128. The
best observed stable size was 128, about 14.1x the fresh bounded sequential
rate and above the historical 4.51 games/s baseline. All runs had zero
operational errors and zero illegal actions. The same 16-game panel at batch
sizes 16 and 128 produced identical outcomes; deterministic heuristic runs
also matched the sequential reference.

The real pending lifecycle was executed from the A5 Control rank-0/rank-1
envelopes using a 4-game smoke override: the job was scheduled, processed,
persisted, and marked completed only after its panel succeeded. The second
invocation reused the identity-keyed receipt. A deliberately changed SHA was
rejected before evaluation.

The regression report now compares a candidate with an earlier policy on the
same fixture and environment identity and reports paired score differences.
An absolute score below 50% is labelled a weak matchup; it is not called
forgetting without a comparable reference decline. Reports remain informational
and never roll back or stop PPO.

Remaining limitations are explicit: the A8 benchmark is a 128-game native
throughput sample rather than a fresh 2,048-game panel, and GPU peak-memory
capture was not wired into the receipt wrapper for this run. The outstanding
PPO recompute discrepancy (`1.0507107e-3` versus `1e-3`) remains unchanged for
Track A9.
