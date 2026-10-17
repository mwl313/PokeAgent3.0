# AHL V2 Stage D — collector weighting plan (DESIGN ONLY)

Status: **PROPOSED / NOT IMPLEMENTED IN TRAINING CODE**. The frozen training
worktree is untouched. This plan describes the reviewed change that would be
required after the migration contract is approved.

## 1. Target quantity: learner rows, not games

The pilot target is **5% of current-policy learner rows** from
current-vs-history games. It is not 5% of games.

With equal decision counts per seat, the game assignment fraction is

```
g = 2r / (1 + r)
```

so `r = 0.05` gives `g = 0.09523809523809523` (about 9.52% of games). This is an
estimate only. The implementation must measure the realized learner-row mix by
opponent and report it; it must never claim that 5% of games equals 5% of
learner experience.

The historical seat never contributes PPO rows, advantages, losses or team
statistics. Only the current-policy seat in a current-vs-history battle is a
learner row.

## 2. Existing code paths to preserve

- `agent/train/native_collector.py` already supports historical model loading,
  one-side-only current learner rows, balanced sides and uniform history
  selection via `randrange` in `_make_assignments`.
- `scripts/run_ddp_ppo.py` already has historical CLI flags and envelope
  loading.
- Current/current games still count once and still supply both learner seats.
- The 0% path must remain byte-identical: same rows, same RNG stream, same
  model/optimizer digests and the same old log probabilities as the frozen
  baseline.

## 3. Weighted assignment design

Add a versioned, deterministic weighted choice in `_make_assignments`:

1. Inputs: ordered list of frozen policy identities, weights from the accepted
   `opponent_plan.json`, `history_rows_target`, segment clock and a dedicated
   assignment RNG.
2. If `history_probability == 0.0` and the legacy uniform path is selected,
   return exactly the existing uniform/current-only behavior. Do not consume
   RNG from the new stream.
3. Otherwise, draw the per-game opponent with a deterministic weighted sampler
   seeded from `plan_hash`, rank-independent segment clock and a declared
   stream id.
4. Seat assignment remains balanced. The current policy always occupies exactly
   one seat; the historical policy occupies the other.
5. The historical model is loaded once per policy group where possible,
   `eval()`, `requires_grad=False`, `torch.inference_mode()` and identity
   verified before use.
6. Every battle's opponent is frozen for the entire episode, including all
   decision levels and hidden-information retries.
7. The distribution is frozen for the whole 200K-match segment. It never
   changes mid-iteration.

## 4. Manifest and restart contract

Both ranks persist and verify the same opponent manifest:

```json
{
  "schema": "pa3-v2-opponent-manifest-v2",
  "plan_hash": "<sha256>",
  "pool_hash": "<sha256>",
  "selection_distribution_hash": "<sha256>",
  "history_rows_target": 0.05,
  "history_game_target_estimate": 0.09523809523809523,
  "assignment_rng": {"stream": "history-assignment-v2", "seed": "<int>", "draws": 0},
  "segment_clock_start": 2764800,
  "segment_clock_end": 2964800,
  "policies": [
    {"stable_id": "md-<digest>", "model_digest": "<digest>",
     "snapshot_sha256": "<sha256>", "weight": 0.0}
  ]
}
```

Restart rules:

- Reject a conflicting plan hash between ranks.
- Reject a missing policy snapshot; fall back to the last accepted plan or
  current-only, never silently substitute another policy.
- Reproduce the exact assignment stream from the persisted RNG state.
- Do not duplicate a natural match, a step or a team-stat record after a crash.
- Persist the manifest in both-rank full-state envelopes and in the reviewed
  migration receipt.

## 5. Measurement and telemetry

Per segment, record:

- current-policy learner rows by opponent (count and fraction);
- historical games by opponent and seat;
- realized `r` and realized `g`;
- model wall time, games/s and learner rows/s;
- KL, clipping fraction, entropy, value loss, gradient norm, AMP scale,
  secondary precision gate result, VRAM and rank digest agreement;
- number of skipped optimizer steps and operational errors.

The existing formal gate baseline (`~381.9` learner rows/s) is a
current-only measurement. A history mode changes rows per match and adds
inference cost, so a new **mode-specific** baseline must be measured and
explicitly approved. It must not be adopted silently to hide a regression.

## 6. Required tests

1. 0% exact parity: identical rows, RNG, log probabilities, model and optimizer
   digests versus the frozen current-only baseline.
2. Deterministic weighted sampler with a fixed seed and a fixed plan.
3. Distribution normalization, 40% cap when K >= 3, K = 1 and K = 2
   feasibility, no NaN.
4. Historical rows excluded from PPO loss and advantages.
5. Balanced seats and per-battle frozen opponent.
6. Two-rank plan agreement and restart uniqueness.
7. Missing/stale policy fallback.
8. No duplicated natural matches or team-stat rows.
9. No LR rewarm, no scaler reset, no counter reset.
10. Secondary recompute and rank digest agreement remain enabled.

## 7. Non-goals

- No automatic 10% or 20% increase.
- No in-iteration PFSP update.
- No training against the current exact digest.
- No use of formal alpha, formal fixtures or formal confirmation outcomes as
  selector feedback.
