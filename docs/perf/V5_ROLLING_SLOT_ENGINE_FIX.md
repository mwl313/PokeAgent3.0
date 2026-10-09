# V5d — rolling-slot engine fix + A/B (P0c STEP 5a, W1)

Base `2b3c818`. Engine files `engine/src/batch.rs` / `engine/src/python.rs`
were assigned to this task.

## 1. Engine change (A6)

The blocker was structural: `reset_batch` *replaced* the whole `states` vector
and bumped one batch-wide generation, so a subset reset invalidated every
in-flight handle (`stale environment handle`). Fix:

* `BattleBatch` now keeps `generations: Vec<u32>` **per slot**; `state()`
  validates `handle.generation == generations[slot]`.
* New `reset_slots(slots, specs)` resets only the listed slots in place (all
  work validated before any slot is replaced) and bumps only their
  generations; `Handle {slot, generation}` is unchanged.
* `reset_batch` behaviour is unchanged (whole-batch replace; all per-slot
  generations set to the new value; same slots/handles returned).
* PyO3 binding `reset_slots_batch(slots, team_a, team_b, seeds, role_map)`
  added alongside `reset_batch`; the collector's rolling path uses it.

Rebuild: `bash scripts/build_python.sh` (same documented path as before) →
`built engine/python/pa3_engine/pa3_engine.so`. Engine verify suite:
`bash scripts/cargo.sh test --locked --release` → all suites pass.

Collector: refill now stops as soon as the open games can cover the remaining
quota, so the drain lands just above the target (2,083 games for a 2,048 quota
= 1.7 % overshoot, vs 50 % before that bound).

## 2. Equivalence gates — PASS

`tests/integration/test_rolling_slot_equivalence.py` un-xfailed and passing:
both seats recorded, one count per natural match, one opposite-sign terminal
reward per side on the side's last request, row-schema invariants, quota met
with overshoot, idle fraction < 0.25. Full suites: **88 python tests passed**
(86 + 2 rolling), engine `cargo test --release` green.

## 3. A/B (dual, 2,048 games total, micro 1024, checkpoint + recompute gate)

| arm | all-in runs | median | collect median | update median |
|---|---|---:|---:|---:|
| OFF | 32.502 / 31.994 / 32.502 + 32.501 | 32.502 | 12.43 s | 42.14 s |
| rolling ON | 32.502 / 32.501 / 32.501 | **32.501** | 12.52 s | 42.01 s |

Median deltas: all-in **−0.00 %**, collect +0.67 %, update −0.31 % — all inside
the ±2 % noise floor. Gates identical on every run (rows 27,330, 56 steps,
0 skipped, 0 operational errors, recompute 5.78e-06, model+optimizer digests
equal).

Drain-tail telemetry (single GPU, 2,048 games / 1,024 envs):

| mode | games | wall | idle slot fraction |
|---|---:|---:|---:|
| static cohorts | 2,048 | 26.12 s | 21.1 % |
| rolling | 2,083 | 25.74 s | **10.4 %** |

So rolling does what it is designed to do (idle slot-time halved, bounded
overshoot) but **does not move all-in throughput** in this configuration: the
dual collection wall is bound by per-round work, not by the drain tail, so the
roadmap's +4 % ceiling does not materialize. Decision: **keep rolling opt-in**
(`--rolling-slots`), default unchanged; the flag and the engine capability stay
for future configurations (e.g. longer games / larger env counts).

## 4. W2 f16 observation wire — NOT EXECUTED

Reported, not faked: the engine fix + gates + A/B consumed the budget for this
pass, so the f16 float-block transport PoC and its equivalence band were not
started. The B1 row stays open; no code was changed for it.
