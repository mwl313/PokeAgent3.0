# AHL V2 — isolated historical diagnostics (corrected 2026-10-17)

Status: **IMPLEMENTED · FULL 435-PAIR GRID COMPLETE · DIAGNOSTIC ONLY**.
Production PPO remains **100% current-vs-current**; historical PPO training is
**disabled at probability 0.0** and Stage D is `NO_GO`.

> This document supersedes the smoke-era status in `FINAL_REPORT.md`. The
> authoritative run result is `docs/operations/ahl-v2/results/FULL_COMPLETE.md`
> and `run_report.json` (435 pairs / 77,184 games). The 45-pair / 5,760-game
> smoke is preserved as historical evidence only.

## 1. Ownership and isolation

| Item | Location |
|---|---|
| Diagnostic root | `/home/minidc/eval-league/roundrobin/v1` |
| Code | `agent/evaluation/historical_league.py`, `historical_schedule.py`, `historical_resources.py`, `historical_selector.py` |
| Runner | `scripts/v2_history_roundrobin.py` |
| Resource gate | `scripts/ahl_v2_resource_gate.py` |
| Proof snapshot | `scripts/v2_history_protected_snapshot.py` |
| Tests | `tests/agent/test_v2_historical_league.py`, `tests/agent/test_v2_historical_corrections.py` |
| Formal evaluation root | `/home/minidc/eval-league/v2` (read-only) |
| Frozen training worktree | `/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix` (read-only) |

The diagnostic root owns its registry snapshot, state, lock, status, receipts,
pair plan, matrix, ratings, schedule, resource gate and run report. It never
writes `alpha_budget.json`, `best_policy.json`, `schedule.json`, formal receipts,
checkpoint payloads or team-stats roots. No diagnostic job writes
`v2/state/active.json`.

## 2. Full run (FULL_COMPLETE)

| Field | Value |
|---|---|
| Policies | 30 (30 distinct model digests) |
| Pairs | **435/435** |
| Games | **77,184** (basic 379 × 128 + fine 56 × 512) |
| Panel / seed domain | `pa3-history-diagnostic-panel-v1` / `pa3-history-diagnostic-v1` |
| Champion at run | `v2-000002600960-da35700f` |
| Latest at run | `v2-000002764800-da35700f` |
| Evaluated window | 2026-10-17 06:47–07:37 UTC |
| Registry snapshot SHA256 | `24da9771e2f78e176cca8e04fe81b98cfebd4085b00cbdbc0af98f370177aa66` |
| Formal alpha spent | **0** |
| Formal protected-file writes | **0** (schedule.json changes are owned by the formal discovery timer) |

`run_report.json → validation` checks every pair against the frozen registry:
unique pair keys, model digests, planned pair coverage, expected game counts and
receipt identities. The current validation is `ok=true`, `complete=true`.

## 3. Bradley–Terry correction (P0)

The smoke-era implementation used:

```python
updated[i] = numerator / denominator
```

The MM update is:

```python
updated[i] = strength[i] * numerator / denominator
```

`denominator` already contains `strength[i]`, so dropping the leading factor
changes the fixed point. The corrected fit:

- uses the multiplicative MM update with geometric-mean normalisation;
- checks convergence on both maximum log-parameter change and the Bernoulli
  quasi-likelihood (draws contribute half a win);
- accumulates repeated rows instead of overwriting them (required for
  pair-cluster bootstrap resampling);
- preserves unequal game counts and real per-pair sample sizes;
- reports pair-cluster bootstrap uncertainty (the resampling unit is a whole
  pair summary, so seat-swapped games stay correlated);
- separates descriptive ranking from formal promotion.

Corrected descriptive top five (see `ratings.csv` / `ratings.json`):

| Rank | Policy | Clock | Rating | Pair-cluster 95% CI | P(above next) |
|---:|---|---:|---:|---|---:|
| 1 | `v2-000002764800-da35700f` | 2,764,800 | +0.8801 | [+0.7489, +0.9798] | 0.670 |
| 2 | `v2-000002502656-da35700f` | 2,502,656 | +0.8428 | [+0.7409, +0.9383] | 0.694 |
| 3 | `v2-000002400256-da35700f` | 2,400,256 | +0.8110 | [+0.7390, +0.9112] | 0.608 |
| 4 | `v2-000002600960-da35700f` (champion) | 2,600,960 | +0.8041 | [+0.7075, +0.8911] | 0.716 |
| 5 | `v2-000002703360-da35700f` | 2,703,360 | +0.7662 | [+0.6937, +0.8460] | 0.520 |

The adjacent-rank differences are **not statistically resolved** at the
pair-cluster 95% level (`difference_supported=false` for the top five). The old
smoke-era conclusion that 2,600,960 was first is superseded; the corrected
point estimate ranks the 2,764,800 checkpoint first and the champion fourth.
The champion-vs-latest fixture remains statistically indistinguishable
(latest score 0.5254, paired 95% CI [0.4863, 0.5664] over 512 games).

## 4. Durable recurring schedule (P1)

`agent/evaluation/historical_schedule.py` implements two tiers in
`diagnostic_state.json` and `milestone_schedule.json`.

| Tier | Clock | Work | Cap |
|---|---|---|---|
| A | every 200K from **2.8M** | newest verified milestone checkpoint vs up to 8 anchors × 256 games | 8 anchors / 2,048 games |
| B | every 1M from **3.8M** | expanded round robin | 24 distinct model digests |

Properties:

- milestones are keyed `tier:nominal_clock` and associated with the first
  verified checkpoint that reached them, so **2,802,944 satisfies 2,800,000**;
- each milestone is planned once; completed keys are never re-run;
- `planned`/`running` jobs are resumable from receipts;
- corrupt receipts fail closed (`corrupt_receipt`) and are never overwritten;
- multiple missed boundaries after downtime are emitted independently;
- Tier B at a 200K boundary supersedes Tier A (`covered_by_tier_b`) so
  compatible work is not duplicated;
- pending/running formal evaluation always wins: the scheduler returns
  `deferred_formal_priority` and creates no work;
- participant identity is deduplicated by model digest, always includes the
  latest candidate and the official champion, keeps stable anchors, and uses
  increasingly wider temporal spacing for older snapshots;
- cached pairs are reused only when model digest, fixture count, environment
  and panel identity all match.

See `docs/operations/AHL_V2_DIAGNOSTIC_SCHEDULE.md` for the operator runbook.

## 5. Retired shadow plan (P0)

The smoke-era `opponent_plan.json` was computed from 45 pair summaries and the
2,600,960 champion. It is now:

- archived byte-identically as `opponent_plan.smoke-20261017.json`
  (SHA256 `9349f6b43cce9c3363911ab3842f152a97709679f1a9cad2058b705d939b68d9`);
- replaced by an `OBSOLETE` record with `activatable=false`,
  `history_probability=0.0`, `history_rows_target=0`, `activation=FORBIDDEN`,
  and full provenance;
- guarded by `load_opponent_plan(..., for_training=True)`, which always raises.

No production PPO configuration references this file. Active production
profiles keep `historical_probability: 0.0`.

## 6. Statistics and limits

- Seat-swapped games are collapsed into one pair cluster before intervals.
- Pair-cluster bootstrap intervals are deterministic and seeded.
- Cycles are `SUSPECTED` unless every edge has a paired 95% interval strictly
  above 0.5; the full grid currently reports **0** cycles.
- Historical evaluation on the frozen team pool is not evidence of
  out-of-distribution / unseen-team generalization.
- These results cannot promote policies, spend formal alpha or move the
  official champion.

## 7. Stage D

Stage D remains `NO_GO`. See `AHL_V2_STAGE_D_*.md`. No historical rollout
mixing, PPO source migration, reward change or hyperparameter change was made.

## 8. Reproduction

```bash
cd /home/minidc/Desktop/PokeAgent3.0-AHL-fix
PY=/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python

# Derived reports only (does not rerun battles; reads immutable pair summaries)
"$PY" scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 report

# Idempotent milestone planning (plan only, never runs battles)
"$PY" scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 schedule

# Topology/resource gate
"$PY" scripts/ahl_v2_resource_gate.py --root /home/minidc/eval-league/roundrobin/v1 --ppo-stopped

# Read-only protected-state proof
"$PY" scripts/v2_history_protected_snapshot.py \
  --output /home/minidc/eval-league/roundrobin/v1/protected_snapshot/corrections_verification.json
```
