# AHL V2 stages A-C — isolated diagnostics and shadow plan

Status: **IMPLEMENTED, BOUNDED SMOKE RUN, SHADOW ONLY**. Historical PPO is not
active. Stage D is design-only and remains `NO_GO`.

## 1. Ownership and isolation

| Item | Location |
|---|---|
| Diagnostic root | `/home/minidc/eval-league/roundrobin/v1` |
| Code | `agent/evaluation/historical_league.py`, `agent/evaluation/historical_selector.py`, `scripts/v2_history_roundrobin.py` |
| Tests | `tests/agent/test_v2_historical_league.py` |
| Formal evaluation root | `/home/minidc/eval-league/v2` (read-only) |
| Frozen training worktree | `/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix` (read-only) |

The diagnostic root owns its registry snapshot, lock, status, receipts, pair
plan, matrix, ratings, cycle report, champion validation, incremental plan,
shadow plan and gate report. It never writes to `alpha_budget.json`,
`best_policy.json`, `schedule.json`, formal receipts, checkpoint payloads or
team-stats roots. No history job writes `v2/state/active.json`.

## 2. Fixture identity and harness equivalence

- Formal calibration panel: `MASTER_SEED = 20261014`, 64 fixture pairs,
  reproduced exactly against formal receipt
  `235eda3e...d256`. Identity, environment, fixture manifest, W/D/L and score
  all matched.
- Diagnostic panel: `RR_PANEL_VERSION = pa3-history-diagnostic-panel-v1`,
  seed domain `pa3-history-diagnostic-v1`, seed `20261017`. This is a separate
  seed family from formal screening and confirmation.
- Basic pair budget: 64 fixture pairs = 128 games. Fine pair budget: 256
  fixture pairs = 512 games. Fine edges are the union of adjacent-clock pairs
  and champion connections, computed from the live registry and never
  hardcoded.

## 3. Pair arithmetic (live registry)

At freeze time the verified registry had 30 policies and 30 distinct model
digests. The champion `v2-000002600960-da35700f` is interior with two
neighbours.

```
C(30, 2) = 435 unordered pairs
fine edge union = 56
broad remainder = 379
56 * 512 + 379 * 128 = 77,184 games
```

The bounded smoke selects by priority:

1. champion versus latest;
2. the full champion row;
3. adjacent-clock neighbours;
4. archival remainder.

The complete 435-pair grid remains resumable through `pair_plan.json`,
`status.json` and content-addressed receipts. It is not run in this bounded
smoke.

## 4. Statistics

- Seat-swapped games are collapsed into one pair cluster before intervals.
- Pair-cluster bootstrap intervals are deterministic (seeded from the pair key
  and receipt identity).
- Draws are explicit at half a win for both the descriptive score and the
  Bradley-Terry fit.
- Bradley-Terry ratings are descriptive only.
- Cycles are reported as `SUSPECTED` unless every edge has a paired 95%
  interval strictly above 0.5.
- Diagnostic results have zero authority over promotion and spend zero formal
  alpha.

## 5. Stage B — incremental diagnostics

`milestone_hook` looks for a completed 200K full milestone and plans up to
eight anchors at 256 games each:

- up to 3 former champions (from promotion history);
- up to 2 verified weakness candidates (point score below 0.45);
- up to 2 informative nonchampions (score closest to 0.50);
- up to 1 distant temporal/diversity anchor;
- remaining slots filled by verified coverage.

The hook is implemented and tested, but no new milestone is available while
training is paused at 2,764,800 matches. The next nominal boundary is
2,800,000. The plan is recorded but not executed.

## 6. Stage C — shadow selector

The tested reference formula is preserved exactly:

```
s_i = (W_i + 0.5 D_i + 64) / (G_i + 128)
hard_i = (1 - s_i)^2
close_i = 4 s_i (1 - s_i)
q_i = 0.60 * hard_i/sum(hard) + 0.30 * close_i/sum(close) + 0.10 / K
```

The single-policy cap is 40% when K >= 3, with a deterministic water-fill.
Unknown or stale opponents receive exploration mass only; stale means a clock
gap greater than 400K matches. The plan is pinned by model digest with stable
IDs (`md-<digest>`), records the evidence receipt identities, fixture and
environment identity, pool hash and selection-distribution hash.

`opponent_plan.json` sets:

```
shadow_only = true
history_rows_target = 0
history_probability = 0.0
activation = FORBIDDEN_WITHOUT_OWNER_APPROVAL_AND_REVIEWED_MIGRATION
```

No training code path was modified, no PPO rollout ran and no migration receipt
was created.

## 7. Stage D

Stage D is `NO_GO`. See:

- `AHL_V2_STAGE_D_MIGRATION_CONTRACT.md`
- `AHL_V2_STAGE_D_COLLECTOR_WEIGHTING.md`
- `AHL_V2_STAGE_D_AB_PROTOCOL.md`
- `AHL_V2_STAGE_D_ROLLBACK.md`

## 8. Operator commands

```bash
cd /home/minidc/Desktop/PokeAgent3.0-AHL-v2
CUDA_VISIBLE_DEVICES= nice -n 19 \
  /home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py freeze

CUDA_VISIBLE_DEVICES= nice -n 19 \
  /home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py harness

CUDA_VISIBLE_DEVICES= nice -n 19 \
  /home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py --max-games 5800 --fixture-pairs 64 run

/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py report
/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py incremental
/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py shadow
/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py gate
```

The full grid can be resumed later with a larger `--max-games`; completed
receipts are reused by content identity.
