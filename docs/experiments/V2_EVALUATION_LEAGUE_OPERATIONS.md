# PokeAgent 3.0 V2 Evaluation League — Operations

Independent, CPU-only, read-only-of-production evaluation league for the running
V2 Mode E PPO campaign. It never loads optimizer state into a learner, never
writes a training counter, and never promotes anything into training.

## Layout

- Code: `/home/minidc/Desktop/PokeAgent3.0-V2-evaluation-league` (branch `evaluation/v2-league`)
- Evaluation-owned state: `/home/minidc/eval-league/v2`
  - `registry.json` — V2-only verified policy archive (content identities pinned)
  - `snapshots/` — content-addressed model-only frozen policy artefacts
  - `schedule.json` — milestone ledger and league jobs
  - `receipts/` — immutable per-identity matchup receipts
  - `receipts/jobs/` — per-job evaluation receipts (spec §6 outputs)
  - `best_policy.json`, `best_history/`, `decisions/` — incumbent registry
  - `state/` — discovery/worker/resource state
  - `heldout/` — Track B held-out generalization status
  - `PAUSE` — presence pauses evaluation processing

## Services

- `v2-eval-discover.timer` → `v2-eval-discover.service` (every 5 min): watches the
  production rolling store read-only, verifies fully published two-rank
  generations, derives frozen snapshots, refreshes the milestone schedule.
- `v2-eval-worker.service` (persistent): single-writer, CPU-only, nice/ionice,
  resource-gated. Runs scheduled matchups and applies conservative promotion.

## Scheduling

Absolute global committed-match milestones: light every 50,000, full every
200,000. A checkpoint serves the earliest unserved milestone at or after it;
milestones crossed before that checkpoint are coalesced into one job (a full
milestone makes the job full). Milestones before the recovered campaign start
(57,344) are `skipped_prestart`; milestones without a checkpoint are `deferred`.

## Commands

```
systemctl --user status v2-eval-worker.service
journalctl --user -u v2-eval-worker.service -f
python scripts/v2_eval_report.py status
python scripts/v2_eval_report.py jobs
python scripts/v2_eval_report.py receipts
python scripts/v2_eval_report.py best
python scripts/v2_eval_report.py trend
python scripts/v2_eval_report.py pause
python scripts/v2_eval_report.py resume
python scripts/v2_eval_report.py report
```

`trend` ratings are internal relative league diagnostics only — not Pokémon
Showdown ladder Elo and not promotion evidence.

## Promotion rules

A candidate becomes incumbent only from a valid **full** evaluation with no
operational/illegal failures, a pair-cluster Hoeffding lower bound above 0.5
(global alpha spending across the archive), no statistically supported
regression against the frozen baseline, and confirmation on an independent
held-back fixture seed. Lightweight evaluations record trajectory only and
never promote. Inconclusive evidence retains the incumbent.
