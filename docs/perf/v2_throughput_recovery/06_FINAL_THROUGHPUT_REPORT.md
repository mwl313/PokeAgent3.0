# V2 throughput recovery — user-requested bounded closeout

The user requested ending the test around 50,000 games. Recovery was stopped
cooperatively at the next completed iteration, and the queued corrected-runtime
experiments were cancelled before launch. **No production cutover or restart.**

Executed optimization revision: `b3c189ba9e61ffa002a3875b0848ac592f6450a0`.
Exploratory A/B revision: `8a743adb1b55d3940fa593d4ac196a91a572d3a0`.
Original/recovery runtime: `5ccc51b0cd4a49efb93018258b9750d5e804ae32`.
This report's later commit is not an executed benchmark source SHA.

## Completed original-runtime recovery test

Restored the original full 4,096-match checkpoint; replayed **13 iterations /
53,248 natural games**, ending at **57,344 cumulative committed matches**.
The request arrived after the 53,248 cumulative boundary while another iteration
was in flight. It was completed and durably saved rather than killed mid-update.
An earlier independent one-iteration probe replayed another 4,096 games; it
does not add to the recovered campaign clock.

Two Tesla V100 PCIe 32 GiB, both 175 W; PPO 2 epochs, global minibatch 4096,
microbatch 1024, 1024 environments/rank, 16 workers/rank, DDP, Mode D with FP16
reduced-precision reduction disabled, deterministic algorithms, original
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, original frozen 1,220-team dataset and engine.
Rollout/shuffle seed 20261014; actual saved model initialization seed 20261006.
The original report's initialization seed 0 was metadata, not the saved config.
Low-rate samples during the latter part of recovery recorded 175 W limits on
both cards, maximum temperatures 52/49°C, memory clocks 877 MHz and varying SM
clocks (1162–1380 / 1230–1380 MHz). They are monitoring samples, not a thermal
or clock-controlled A/B; exact sample ranges are retained in evidence.

| Measurement | Actual result |
|---|---:|
| Iteration-only throughput | 32.0347 natural games/s |
| Startup/checkpoint/report-inclusive throughput | 31.8008 natural games/s |
| Sum of critical-rank iteration wall time | 1,662.199 s |
| Full launcher wall time | 1,674.422 s |
| Mean collection + match-clock wait | 65.364 s/iteration |
| Mean preparation | 11.767 s/iteration |
| Mean PPO update | 47.996 s/iteration |
| Learner rows | 1,327,695 |
| New optimizer steps | 664 |
| Cumulative optimizer steps | 718 |
| Maximum recompute discrepancy | 0.000802279, below unchanged 0.001 |
| Operational errors / skipped steps | 0 / 0 |

These timings measure **original-code recovery**, not optimized candidate speed.
Natural games are counted once globally, not once per seat/rank. Per-iteration
work and rank evidence are retained in `evidence/recovery_report.json` and
`evidence/CLOSEOUT_RECEIPT.json`. The critical-rank decomposition uses the rank
with the larger complete iteration wall time; overlapping host substage timers
are not blindly summed. Clock collectives include rank skew, not pure transfer.

At every completed boundary, both model and Adam digests match historical
telemetry. Final rank artifacts were independently loaded on CPU and their
full model/Adam digests recomputed. Manifest/rank hashes verify; both ranks
share iteration 14, source, clock, dataset/action/opponent contracts and rank
digests. Scheduler, GradScaler, owned RNG and collector restart state are saved.
Both collector drain records show complete / zero in-flight battles. All 28
team-stat receipts pass canonical-content hash and clock/W/L/D reconciliation.
An additional local checkpoint backup is independently hash-verified.

Classification: **USER_STOPPED_VERIFIED_PARTIAL**. The fixed-target recovery
utility's raw `HISTORICAL_REPLAY_DIFFERS` label means its 73,728 target was not
reached; its actual `matched_historical_digests` fields are true. Raw output
is retained unchanged and this interpretation is recorded in CLOSEOUT_RECEIPT.
No recent historical RNG payload exists for direct comparison; replay
reconstructs the owned streams. This is not universal bit-exact recovery.

Full 73,728 recovery is unfinished. The latest 73,728 model-only evidence remains
preserved; do not combine it with an older optimizer and call that exact resume.
Original campaign rolling/LATEST still points to 4,096. Recovery saved to its
own namespace and appended immutable, uncommitted team receipts under the
original run identity; it did not overwrite existing receipts or generations.
Exact recovered path and backup are in 07_SAFE_CUTOVER_AND_ROLLBACK.md.

## Implemented optimization package and completed exploratory A/B

1. **Canonical candidate blocks:** decode signed targets, validate ordering
   and map actual active-roster entity/move references once per branch batch.
   Actor scoring and rollout recording share owned immutable arrays. Every
   branch/prefix gets its own block; raw callers retain validation.
2. **Owned observation blocks:** copy fixed arrays and all five ragged families
   once per producer batch. Store block/row references, avoiding per-decision
   dictionaries and duplicate fixed-field copies. Gathered results remain
   independent; source and returned-tensor mutation isolation are tested.
3. **Vectorized ragged selection:** construct CSR gather indices without a
   Python np.arange per row, preserving order, repeats, empty rows and shuffle.
   No unsafe permanent validated flag or mutable-offset cache.
4. **Bounded source migration:** exact source/blob/checkpoint/parity hashes,
   protected model/loss/engine/config/data files, maximum five iterations and
   no production-profile authorization.
5. **CUDA workspace contract:** pin/record/check the original deterministic
   cuBLAS setting before CUDA initialization; reject conflicting settings.

Independent captured-input CPU probes used 576 real observations / 4,233
prefix-conditioned candidates, five passes in each source process:

| Local CPU stage | Baseline median | Candidate median | Ratio |
|---|---:|---:|---:|
| Candidate processing | 0.369472 s | 0.008375 s | 44.12x |
| Recording | 0.156583 s | 0.028740 s | 5.45x |
| Materialization | 0.082730 s | 0.062985 s | 1.31x |

Representations, known masks, candidate order/refs, selected actions, every
tested loss term, gradients and one Adam update match bitwise across independent
source processes. The full-model CPU loss probe uses declared synthetic
advantages/returns; it is not a learning experiment. Local timings do not
establish each change's independent end-to-end contribution.

Completed full-stack exploratory panel: three independent timing processes
per arm, same seed and 4,096 checkpoint, five complete iterations per process:

| Exploratory metric | Original | Candidate |
|---|---:|---:|
| Median iteration-only games/s | 28.2967 | 42.2875 |
| Repetition range games/s | 27.7081–28.4945 | 41.9822–42.5941 |
| Median all-in games/s | 28.0119 | 41.5348 |
| Mean collection + clock wait | 76.764 s | 30.968 s |
| Mean preparation | 13.598 s | 12.698 s |
| Mean update | 52.356 s | 50.844 s |
| Peak reserved GPU GiB/rank | 15.897 | 15.897 |
| Peak RSS GiB/rank | 11.125 | 10.899 |

Exploratory package improvement: **49.44% iteration-only / 48.28% all-in**.
All compared boundary rollouts, model/Adam digests, rows and optimizer work
agree, with two executed epochs and 54 steps per iteration, cache admitted,
no errors/skipped updates, recompute gates passing. No individual full-stack
ablation was run. Repetitions are the same learning trajectory, not independent
training seeds; descriptive variation is not statistical learning confidence.
Controls ran before candidates; temporal/clock confounding remains. One
earlier control overlapping tests/CPU probes is excluded, not cherry-picked
into the three-run panel. Both arms omit campaign team-stat writes to preserve
the path-bound ledger; original first-iteration team cost was about 0.16 s.

**Important limitation:** these exploratory runs omitted the original cuBLAS
workspace environment. Both arms match each other but diverge from historical
model updates. This omission was discovered, corrected and disclosed; the
scheduled production-environment A/B, profiles and continuation were cancelled
by the user's stop request before launch. Therefore the optimized package's
speed under the corrected original runtime is **not yet measured**, and the
49.44% result is **not production acceptance**.

## GPU evidence, next opportunity and rejected shortcuts

The exploratory real-input 1024-row microprofile measured approximately
539–543 ms per unprofiled forward/loss/backward/Adam step. The profiled window
has 519.855 ms leaf-kernel union in a 571.419 ms span, about 91% kernel-active
time, **not SM occupancy or proof of a hardware ceiling**. Each leaf kernel
is assigned once by originating CPU External id and enclosing stage:

- Backward: 409.612 ms; transformer forward: 51.753 ms.
- Fixed embedding/validation: 13.152 ms; ragged embedding: 10.600 ms.
- Candidate token refs: 16.971 ms; action features: 7.645 ms.
- Key build/score: 3.064/1.646 ms; prefix GRU: 0.515 ms; value head: 0.051 ms.
- Optimizer/AMP/clipping: 2.016 ms.

The largest discrete origin is deterministic GatherBackward index-put
accumulation, 111.523 ms. An embedding-style flat token lookup is a promising
future experiment, but is **not implemented or validated**. CPU synchronization
calls overlap pending GPU work; their host duration is not an independently
removable time budget. Trace values use the exploratory workspace and declared
synthetic targets; do not extrapolate them directly into production games/s.
The actual two-rank NCCL trace was cancelled, so no new NCCL attribution exists.

Next priorities, only on a later user request: corrected-runtime full-stack
validation, long-run recompute repair, deterministic gather backward, then
preparation/H2D and separately bounded micro2048. No shuffle/objective change,
feature removal, tolerance relaxation, find_unused_parameters/NCCL change or
blind validation removal was promoted. Rust battle stepping was a small original
collector stage and is not the leading measured bottleneck.

The original-runtime update alone takes about 48 s/4,096 games; even eliminating
all other stages implies only about 85 games/s for this workload. Preparation
lowers that zero-collection theoretical ceiling further. Reaching stable 90+
requires actual learner acceleration and is not established by CPU changes.
Historical V4~25, V6~37 and A9~90 differ in semantics, policy length, rows and
optimizer work; they are not matched baselines. A9 used 18 steps/2,048 games
versus 54/4,096 in the exploratory V2 panel. Old A5 arithmetic also overstated
rows/game by two: 875177/51200=17.09, not 34.19.

## Correctness, compatibility and operational conclusion

Final focused tests on b3c189b: **28 passed in 2.33 s**. Earlier focused tests
and independent global-DDP gradient tests passed. The broader inherited run
had 261 passed / 8 failed / 1 skipped; seven failures reproduce on original
source, and one near-zero padding comparison is order-sensitive. These are
disclosed in 05; no all-green-suite claim or tolerance weakening.

The exploratory uninterrupted-two versus one/exit/restore/one test was
BIT_EXACT at 12,288 for learner/RNG/accounting/collector/dataset/opponents,
rollouts and outcomes. It proves only that tested same-source/workspace
contract. The corrected-workspace candidate continuation was not executed.

Model parameter contracts and protected runtime files are unchanged. Source
identity differs, so exact hash-bound migration remains necessary; its current
receipt is bounded-validation-only, not production authorization. The original
iteration19 ModeD recompute failure (0.00115824/0.00170612 > 0.001) remains an
unresolved production blocker. Short tests do not repair it.

Production/evaluation transient units were not-found/inactive at entry and
were not restarted. User main, frozen runtime, dataset and mechanics remain
unmodified by this task. The primary worktree currently has a user change in
`engine/test_support/corpus.rs`; it was left intact and excluded from this
branch/push. Diagnostic jobs and queue are stopped; nothing resumes overnight.
Code/compact reports are pushed on the isolated optimization branch; large
checkpoints/traces remain local, with a separate verified local state backup.

## Requested acceptance answers

1. Survived: native packed/batched collection, CUDA cache, streaming prep,
   compaction/key reuse/final-GRU removal, exact weighted DDP/no_sync,
   micro1024, finite/overflow checks and persistent telemetry.
2. Regressed costs: repeated signed decode/roster mapping and per-row ownership.
   Required shuffling makes contiguous cached selection ineffective.
3. New required V2 costs: five ragged families, known masks/public knowledge,
   actual roster mapping, semantic validation and team accounting.
4. Independently measured: captured CPU probes, exploratory complete A/B,
   CUDA events/leaf trace and original-runtime historical replay.
5–7. Implemented/local/package results are above. No per-change end-to-end
   ablation or corrected-runtime candidate result is invented.
8–9. Focused observation/action/loss/gradient/DDP gates pass; broad inherited
   failures and corrected-runtime validation gaps remain explicit.
10. Model/state contracts compatible; source migration is exact and bounded.
11. Full57,344 state is preserved, not full73,728. Production continuation
   still needs numerical repair, approved source/ledger plan and authorization.
12. Production/evaluation originals preserved; no restart or main modification.
13. Rejected/deferred: unsafe validation deletion, ineffective contiguous cache,
   changed math/shuffle/thresholds, unmeasured micro2048/NCCL, legacy semantics.
14. Remaining: backward/gather, preparation, residual collection and the
   long-run recompute correctness blocker.
15. Executed SHAs are at the top; final report SHA is its Git commit. Tracked
   compact evidence is in this directory. Large raw artifacts are at
   `/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/`.
