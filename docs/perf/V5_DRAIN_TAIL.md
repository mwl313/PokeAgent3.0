# V5 — collection drain-tail measurement (P0 W2)

Read-only, flag-gated telemetry added to `NativeCollectorConfig.telemetry`
(default off). With telemetry off the collector takes the identical code path;
the neutrality check below proves the produced rows are byte-identical.

## 1. Method and exact commands

```
# neutrality: 256 games / 256 envs / 8 workers / seed 20261009
PYTHONPATH=engine/python:. .venv/bin/python scripts/v5_drain_tail.py \
  --games 256 --envs 256 --workers 8 --no-telemetry \
  --out runs/perf/v5/tail_neutral_off.json
PYTHONPATH=engine/python:. .venv/bin/python scripts/v5_drain_tail.py \
  --games 256 --envs 256 --workers 8 --telemetry \
  --out runs/perf/v5/tail_neutral_on.json

# standard single-GPU 2,048-game both-seat settings (F0 §1)
PYTHONPATH=engine/python:. .venv/bin/python scripts/v5_drain_tail.py \
  --games 2048 --envs 1024 --workers 16 --telemetry \
  --out runs/perf/v5/tail_2k.json
```

Recorded per round: active envs (≥1 decision), open envs, round wall; per
cohort: size, rounds, wall, games finished, wall from 50 % to 100 % complete;
per game: decision rounds. Derived: idle slot-seconds (finished slots × round
wall), slot-seconds, game-length percentiles.

## 2. Neutrality check — PASS

| run | rows | row SHA256 (16) | games | decisions | model SHA (12) | wall |
|---|---:|---|---:|---:|---|---:|
| telemetry off | 6,864 | `1ede49e8c0678aab` | 256 | 6,864 | `4b217e55edc4` | 3.60 s |
| telemetry on | 6,864 | `1ede49e8c0678aab` | 256 | 6,864 | `4b217e55edc4` | 3.71 s |

Rows, digests, counts and model manifest are identical; the 0.11 s wall
difference is per-round bookkeeping (order of run-to-run variance) and does not
alter any produced row.

## 3. Drain-tail result (2,048 games, 1,024 envs, 16 workers)

`runs/perf/v5/tail_2k.json`: 2,048 games, 54,443 rows, collection wall 26.12 s
(78.4 games/s collection-only), 106 rounds across 2 cohorts.

| metric | value |
|---|---|
| game length (decision rounds) P50 / P95 / P99 / max | 14 / 24 / 30 / 61 |
| mean game length | 15.0 rounds |
| active envs per round P50 / P05 | 24 / 1 |
| cohort 1: wall / rounds / 50 %→100 % tail | 13.38 s / 45 / 2.48 s |
| cohort 2: wall / rounds / 50 %→100 % tail | 12.69 s / 61 / 2.67 s |
| idle slot-seconds / slot-seconds | 5,628.7 / 26,680.0 |
| **idle slot fraction** | **21.1 %** |
| **50 %→100 % tail share of collection wall** | **19.7 %** (5.15 s) |

Interpretation: in the last ~20 % of each cohort wall only a handful of
environments still act (median 24, P05 1 active envs) while the rest of the
1,024 slots are finished and idle — 21 % of all slot-seconds are spent waiting
for long games to drain, and the actor GPU batch collapses with them.

## 4. Rolling-slot upper bound (optimistic ceiling, not a promise)

If rolling refills could remove the entire 50 %→100 % drain tail with zero
overhead, the collection wall for this configuration would shrink by up to
**19.7 %**. Using the F0 all-in split (collection ≈ 23 % of the single-GPU
all-in; ≈ 20 % of the dual wall), that is a ceiling of roughly **+4 % all-in**
(single 20.53 → ≈ 21.4 games/s; dual 32.50 → ≈ 33.8 games/s) before any
learner-side effect. The real P3 implementation must also preserve natural-match
identity, terminal/drain accounting and team-sampling fairness, and re-verify
with the rolling-slot equivalence test — none of that is claimed here.
