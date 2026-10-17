# AHL v2 — 30-policy round-robin results (2026-10-17)

Immutable diagnostic evidence from the first complete historical round robin of the PA3 V2
campaign. **Diagnostic only**: it has no authority over promotion, alpha spending, the
champion pointer, the formal schedule or the frozen training state.

## Provenance

| Field | Value |
|---|---|
| Run time (KST) | 2026-10-17 15:47 → 16:37 |
| Host | miniDC (80 threads, CPU-only, no GPU) |
| Training state at run | paused at 2,764,800 matches / iteration 675, source `da35700`, STOP present |
| Policies | 30 registered snapshots (dedup by model digest: 30 → 30, 0 duplicates) |
| Champion at run | `v2-000002600960-da35700f` |
| Latest at run | `v2-000002764800-da35700f` |
| Pairs / games | **435 pairs / 77,184 games** (basic 379 × 128 + fine 56 × 512) |
| Panel / seed domain | `pa3-history-diagnostic-panel-v1` / `pa3-history-diagnostic-v1` |
| Runner | `scripts/v2_history_roundrobin.py` + `agent/evaluation/historical_league.py` |
| Parallel runner | `scripts/rr_pool.sh` (16 workers, 2 threads each, distinct physical cores) |
| Formal files touched | **none** — `gate_report.json → protected_state.critical_mismatches` is empty |
| Alpha spent | **0** |

## Headline results

- Bradley–Terry (descriptive) top: **2,600,960 (champion) +0.2477**, 2,764,800 (latest) +0.2471,
  2,502,656 +0.2379; bottom: bootstrap −1.0964.
- Champion vs latest: **0.4746, paired 95% CI [0.4336, 0.5137]** over 512 games → statistically
  indistinguishable; the latest checkpoint is **not** demonstrably stronger.
- Champion row: 29/29 opponents, 14,848 games, game-weighted score **0.6733**.
- Non-transitivity cycles: **0**.

## Precision caveat (important when reusing these numbers)

The first pass ran the champion row / adjacent (fine) pairs at 128 games because a bounded smoke
had cached them at `--fixture-pairs 64`. Those 45 pair summaries were then re-run at the designed
fine tier (256 fixture pairs = 512 games) and the grid re-aggregated. **The 128-game version ranked
the latest checkpoint first; the 512-game version ranks the champion first.** Always check the
per-pair `games` column in `matrix.csv` before drawing conclusions, and treat 128-game cells as
wide-interval estimates.

## Files

| File | Contents |
|---|---|
| `matrix.csv` | 435 rows, one per unordered pair: ids, clocks, digests, tier, fixture pairs, games, W/D/L, score, paired CI, status |
| `ratings.csv` | Bradley–Terry ratings (descriptive only), 30 policies |
| `cycles.md` | non-transitivity report (suspected vs statistically supported) |
| `champion_validation.md` | champion row: every opponent, games, score, CI |
| `gate_report.json` | disk gate, formal-eval priority, protected-file before/after hashes, Stage D GO/NO-GO |
| `opponent_plan.json` | SHADOW-ONLY opponent selection plan (`history_rows_target=0`, not active) |
| `incremental_plan.json` | milestone hook plan (8 anchors × 256 games) |
| `pair_plan.json` | full pair plan with tiers, priorities and fixture counts |
| `registry.snapshot.json` / `registry.meta.json` | frozen registry snapshot + hashes used for the run |
| `equivalence.json` + `equivalence_receipts/` | harness-equivalence check against a formal receipt (identity, environment, fixture manifest, W/D/L, score) |
| `receipts/pairs/*.json` | 435 immutable per-pair summaries (the evidence behind `matrix.csv`) |
| `protected_snapshot/` | before/after hashes proving the training/eval/alpha/checkpoint state was untouched |

**Not included:** the 480 content-addressed battle receipts (28 MB) and the raw run logs. They stay
on miniDC at `/home/minidc/eval-league/roundrobin/v1/receipts/*.json`; every number above is
reproducible from the pair summaries plus the frozen registry snapshot, and the harness is
cache-safe to re-run.

## Reproducing / extending

```bash
# from the repository root, with the agent venv that has torch
python scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 freeze
python scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 plan
python scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 harness
# one worker (sequential) or many disjoint slices:
python scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 run
python scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 \
    --pair-slice 3/16 --lock-name worker-3.lock --no-reports run
bash scripts/rr_pool.sh 16          # pool + coordinator report/gate
```

Pair summaries are skipped on re-run, so deleting a single `receipts/pairs/*.json` re-runs only that
pair. The runner yields when a formal evaluation job is pending or running, and stops on the disk
reserve (12 GiB) or an operator `PAUSE` file.

## Related design documents

- `docs/operations/AHL_V2_ROUND_ROBIN.md` — round-robin design and operating rules
- `docs/operations/AHL_V2_STAGE_D_*.md` — history-migration contract, collector weighting, A/B
  protocol, rollback (design only; Stage D is **NO_GO** without owner approval)
- `FINAL_REPORT.md` (repo root) — the implementing agent's Korean stage A–C report
