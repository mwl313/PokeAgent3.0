# V4 collection-contract audit

Branch `optimization/pa3-realpolicy-throughput`, gates C0/D1. The contract is
the one `configs/train.yaml` pins (`collect_both_sides_when_current_self_play:
true`, no historical-opponent learner rows, one terminal reward per recorded
side, natural completion only).

## 1. What is enforced

* `NativeCollector._records_side` records **every** current-policy seat while
  the history pool is empty; historical-opponent rows stay excluded.
* On natural termination the terminal reward (+1/−1/0 per side, opposite signs
  for the two seats of a draw-free match) is written to the **last recorded
  row of that seat's own trajectory**; the natural-match counter increments
  exactly once.
* GAE groups rows by `(match_id, side)` and never crosses a trajectory
  boundary; advantage normalisation uses the iteration's actor rows only.
* The learner stores only current-policy rows; the observations,
  candidate/prefix tables and `old_logprob` come from the same request that
  produced the action (recompute gate: max |Δ logprob| 1.1e-4 vs the 1e-3 fp16
  tolerance in the A0 panel; fp32 C1 gates are tighter).

## 2. Evidence

`tests/integration/test_collect_both_current_sides.py` (3 tests, real Rust
engine, both seats):

* every natural match has both sides recorded, all `policy_id == "current"`,
  `opponent_policy_id == "current"`;
* exactly one `done` row per `(match_id, side)`, it is the last request, its
  reward ∈ {−1, 0, +1} and the two sides' rewards are opposite;
  `Σ learner-seat reward == wins − losses`;
* `request_index` chains are strictly increasing inside `(match_id, side)`;
  GAE returns are confined to trajectories that own a terminal reward.

Real run cross-checks (`runs/perf/v4/c1_single_gpu_oracle.json`):

| quantity | measured |
|---|---|
| collector games (overshoot preserved) | 512 |
| distinct natural matches | 512 |
| learner rows | 13,841 |
| rows per natural match | 27.03 (v3 reference 26.87) |
| wins / losses / draws | 261 / 251 / 0 |
| `reward_sum` (learner seat) | 10.0 == 261 − 251 |
| operational errors | 0 |

## 3. Boundary

The both-seat contract doubles learner rows per natural match; games/s and
rows/s are therefore never compared across the single-seat legacy contract
(see `V3_NUMERIC_PARITY.md` §3 for the labelled A/B). No sample was dropped,
truncated or re-labelled as naturally complete in any v4 run.
