# PPO per-team performance statistics — implementation-ready specification

## 1. Product requirement

At every natural match termination in the new scratch PPO, record training performance for every frozen team from the exact 1,220-team manifest. No team identifier may enter the model observation. This is **policy-conditioned training telemetry**, not an unbiased measure of intrinsic team strength.

The system shall provide team W/L/D, seat-wise splits, opponent-team and opponent-policy splits, time-window trends, match length, coverage, a CLI and CSV/JSON. Must support long-running dual-rank PPO with checkpoint rewind/replay without overcounting.

## 2. Stable identifiers

- `dataset_id`, `dataset_manifest_sha256`, `team_file_sha256`, `team_count=1220` and deterministic canonical `team_id` mapping, including team names and six species.
- `run_id` and `schema_version=pa3-ppo-team-stats-v1`.
- `rank`, `iteration_id`, `commit_generation`, `local_match_id`, `rank_match_uid=(run_id, rank, iteration_id, local_match_id)`.
- `policy_id` per seat, opponent policy ID, seed identity if available, team A and B stable IDs, seat 0 and seat 1, terminal winner/draw, natural completion marker, request or turn count.

Internal numeric team indices are not stable across a different dataset order unless manifest identity matches; never join a team by index alone.

## 3. Counter model

One actual natural match produces **one** completed-match record, **two** team appearance records. For same-team mirror, that team receives **two appearances**, with separate seat 0 and 1 counters.

Per-team counters:

```
games, wins, losses, draws, score_points = wins + 0.5*draws,
score_rate = score_points / games,
win_rate = wins / games,
seat0{games,wins,losses,draws}, seat1{...},
match_length_sum, opponent_policy_splits, window_splits
```

For all fully committed natural matches M:

- Sum of all teams' `games` = `2 * M`.
- Sum of all teams' (`wins`+`losses`+`draws`) = `2 * M`.
- Overall `sum(wins) == sum(losses) == M - D`, `sum(draws) == 2*D`, where D is count of drawn matches.
- For each team: `seat0.games + seat1.games == games`.
- `matches` is **not** `games` of both sides; match counter increments once.
- Refused submissions, operational error, timeout/abort/truncation and resimulation are never natural draws and never enter committed counters.

Record `teamA vs teamB` *oriented seat-specific* matchup and optionally aggregate to unordered pairing. Do not prematurely symmetrize when seat matters.

## 4. Crash and rollback semantics: commit-generation protocol

Do **not** append a permanent global total on each battle. Instead:

1. Collector keeps a compact in-memory rank-local aggregation, keyed by current candidate `iteration_id`; individual match receipt optional for forensic sampling.
2. When an iteration drains, **stage** local stats immutable file:
   `team_stats/staging/iteration_000XYZ.rank0.json` and rank1 counterpart, each SHA256, totals, source/data identity, natural matches count, and deterministic counter digest.
3. Both ranks validate collector and optimizer state. Global coordinator stages a merged stats receipt, asserts appearances and WLD invariants, then binds the receipt ID to checkpoint-generation staging metadata.
4. Only **after** the source/model/optimizer/checkpoint/generation is atomically published as committed may stats become eligible in cumulative global reports. Treat a stats receipt ahead of committed generation as orphan/tentative.
5. On restart, select the latest verified checkpoint generation and rebuild totals from immutable rank-iteration receipts up to its exact committed watermark. Exclude later staged/orphan records, deduplicate by `(run_id, iteration_id, rank)` and verify SHA/content hashes.
6. If checkpoint policy publishes every K iterations, iteration stats may remain tentative until the next recoverable watermark. Never advertise them as durable committed lifetime performance until covered by recoverable state. Optionally separate `latest_observed` vs `last_durable_checkpoint` in CLI.
7. Rerun a rolled-back iteration produces a **new attempt_id**; only the attempt attached to the committed checkpoint generation is counted. A retry cannot double count by colliding with an old local match ID.

Avoid large per-match synchronous disk writes. Stage batch-local memory aggregates, flush only at verified boundaries; write temp file + `fsync` as appropriate + atomic rename, include hashes.

## 5. JSON record examples

`rank_stats`:

```json
{
  "schema_version":"pa3-ppo-team-stats-v1",
  "run_id":"pa3-v2-scratch-YYYYMMDD",
  "iteration_id":1,"attempt_id":"attempt-a", "rank":0,
  "dataset_manifest_sha256":"<actual SHA>",
  "natural_matches":4096,
  "drawn_matches":8,
  "team_appearances":8192,
  "team_counts":{"team:000001":{"games":11,"wins":5,"losses":6,"draws":0,"seat0_games":6,"seat1_games":5}},
  "receipt_sha256":"<compute over canonicalized content excluding this field>"
}
```

`committed_global_stats`:

```json
{
  "schema_version":"pa3-ppo-team-stats-v1",
  "run_id":"pa3-v2-scratch-YYYYMMDD",
  "checkpoint_generation":"generation-0001",
  "through_iteration":1,
  "committed_natural_matches":8192,
  "rank_receipts":["sha256-rank0","sha256-rank1"],
  "team_appearances":16384,
  "verified":true
}
```

Amounts here are **illustrative only**, not a requirement to commit 8,192 matches each iteration. Runtime actual counts and shard sizes must be used.

## 6. Queries and output

Read-only CLI example (name may differ):

```
python -m agent.analysis.team_stats --run <new_run> summary
python -m agent.analysis.team_stats --run <new_run> leaderboard --min-games 100 --top 20
python -m agent.analysis.team_stats --run <new_run> matchup --team 123
python -m agent.analysis.team_stats --run <new_run> trends --window-matches 100000
python -m agent.analysis.team_stats --run <new_run> integrity
```

Machine-readable outputs:

- `team_totals.csv`: `team_id, pokemon_1..6, games, wins, losses, draws, win_rate, score_rate, confidence_low, confidence_high, seat0_* , seat1_*, mean_length`.
- `team_matchups.csv`: oriented pair ID, games, wins, losses, draws and seat.
- `team_windows.csv`: each frozen clock window/committed checkpoint.
- `pokemon_usage.csv`: number of team appearances containing species and associated team-level outcome; do not label as causal contribution.
- `integrity.json`: 1,220 ID coverage, 2M equation, results by rank, commit watermark and non-natural errors excluded.
- Markdown summaries including top and bottom 20 **with sample size, uncertainty and caveats**.

Use Wilson or another named confidence-interval method appropriate for binary wins, and clarify treatment of draws; score-rate intervals require an appropriate separate estimator or bootstrap. A raw 100% team on 2 games is not the strongest team.

## 7. Test fixtures (mandatory)

- One side0 win A vs B and a seat-swapped side1 win; verify oriented count.
- One draw A vs B and same-team mirror A vs A; verify two appearances and scores.
- One aborted/error match excluded.
- Two ranks with deterministic fixture counters: merged counts satisfy all equations.
- Stage duplicate rank receipt same iteration: no second count.
- Stage different attempt of same rolled-back iteration: only final checkpoint-bound attempt counted.
- Crash after stage but before global generation publish: orphan ignored.
- Resume from older checkpoint with later attempted stats present: totals reflect exact recovered checkpoint only.
- All 1,220 teams appear as rows, including zero-game teams.
- Real smoke benchmark demonstrates overhead and per-rank parity.

## 8. Integration details

Collector already has `teams`, seat/role, `winner`, `terminated`, `truncated` at `_collect_round` and `team_ids` on `RolloutRow`. **Do not rebuild a team roster by parsing model tokens.** Capture provenance at the native natural terminal event. Current-training selfplay policy ID may be same on both sides. Keep old learner's `stats.wins` (learner seat) separate from full team appearance totals.
