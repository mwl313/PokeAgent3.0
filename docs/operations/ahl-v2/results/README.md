# AHL v2 — 30-policy round-robin results (FULL_COMPLETE)

Authoritative index for the completed historical diagnostic grid (2026-10-17).
**Diagnostic only**: no authority over promotion, alpha spending, the champion
pointer, the formal schedule or the frozen training state.

> **Correction notice.** Earlier revisions of this README and `FINAL_REPORT.md`
> described a 45-pair / 5,760-game smoke as the latest result. That smoke is
> preserved as historical evidence, but the latest authoritative result is the
> **435-pair / 77,184-game** run described here. The smoke-era Bradley–Terry
> `ratings.csv` used `updated = numerator / denominator`, which is not the MM
> update and therefore converged to a different fixed point. Those rating
> conclusions are superseded by the corrected fit.

## Provenance

| Field | Value |
|---|---|
| Run window (UTC) | 2026-10-17 06:47:56 → 07:37:18 |
| Host | miniDC (80 threads, CPU-only, no GPU) |
| Training state at run | paused at 2,764,800 matches / iteration 675, source `da35700`, STOP present |
| Policies | 30 registered snapshots (dedup by model digest: 30 → 30) |
| Champion at run | `v2-000002600960-da35700f` |
| Latest at run | `v2-000002764800-da35700f` |
| Pairs / games | **435 / 77,184** (basic 379 × 128 + fine 56 × 512) |
| Panel / seed domain | `pa3-history-diagnostic-panel-v1` / `pa3-history-diagnostic-v1` |
| Runner | `scripts/v2_history_roundrobin.py` + `agent/evaluation/historical_league.py` |
| Rating | `pa3-v2-bradley-terry-mm-v2` (pair-cluster bootstrap, 500 draws) |
| Registry snapshot SHA256 | `24da9771e2f78e176cca8e04fe81b98cfebd4085b00cbdbc0af98f370177aa66` |
| Matrix SHA256 | `14afed32c4d2c65caeb4f5ba132d3ce3092ca1e973a26acaadd925ed9e8066c3` |
| Formal alpha spent | **0** |
| Formal protected-file writes | **0** |

## Corrected headline results

Bradley–Terry descriptive top five (corrected MM fit):

| Rank | Policy | Clock | Rating | Pair-cluster 95% CI | P(above next) |
|---:|---|---:|---:|---|---:|
| 1 | `v2-000002764800-da35700f` | 2,764,800 | +0.8801 | [+0.7489, +0.9798] | 0.670 |
| 2 | `v2-000002502656-da35700f` | 2,502,656 | +0.8428 | [+0.7409, +0.9383] | 0.694 |
| 3 | `v2-000002400256-da35700f` | 2,400,256 | +0.8110 | [+0.7390, +0.9112] | 0.608 |
| 4 | `v2-000002600960-da35700f` (champion) | 2,600,960 | +0.8041 | [+0.7075, +0.8911] | 0.716 |
| 5 | `v2-000002703360-da35700f` | 2,703,360 | +0.7662 | [+0.6937, +0.8460] | 0.520 |

- The corrected fit converges in 62 MM iterations; log-likelihood
  `-46,376.91`; max log-parameter change `7.3e-11`.
- Adjacent-rank differences in the top five are **not statistically resolved**
  (`difference_supported=false`); the point ranking is descriptive.
- Champion vs latest: latest score **0.5254**, paired 95% CI
  **[0.4863, 0.5664]** over 512 games → statistically indistinguishable.
- Champion row: 29 opponents, 14,848 games, weighted score **0.6733**.
- Non-transitivity cycles: **0**.

## Run validation

`run_report.json → validation` is `ok=true`, `complete=true`:

- 435 unique pair keys, 435 successful summaries;
- 0 duplicate / missing / digest / game-count errors;
- 77,184/77,184 games; tiers basic 379 / fine 56.

## Files

| File | Contents |
|---|---|
| `FULL_COMPLETE.md` | authoritative completed-run report |
| `run_report.json` | machine-readable run metadata + validation + hashes |
| `matrix.csv` | 435 rows: ids, clocks, digests, tier, games, W/D/L, score, paired CI |
| `ratings.csv` / `ratings.json` | corrected BT ratings, CIs, rank CIs, adjacent confidence |
| `ranking_confidence.csv` | adjacent-rank pair-cluster bootstrap confidence |
| `cycles.md` | suspected vs statistically supported non-transitivity |
| `champion_validation.md` | champion row against every opponent |
| `gate_report.json` | stage status, formal priority, protected hashes, Stage D NO_GO |
| `milestone_schedule.json` | current durable schedule plan (next A 2.8M, first B 3.8M) |
| `diagnostic_state.json` | persisted milestone/job state |
| `resource_gate.json` | topology-derived CPU masks and resource gate |
| `opponent_plan.json` | `OBSOLETE`, non-activatable retirement record |
| `opponent_plan.smoke-20261017.json` | preserved smoke-era plan (byte-identical) |
| `pair_plan.json` | full pair plan with tiers and priorities |
| `registry.snapshot.json` / `registry.meta.json` | frozen registry snapshot + hashes |
| `equivalence.json` + `equivalence_receipts/` | harness-equivalence against a formal receipt |
| `receipts/pairs/*.json` | 435 immutable per-pair summaries (not rewritten) |
| `protected_snapshot/corrections_verification.json` | read-only proof that protected files are unchanged |
| `status.json` | reconciled live status (`FULL_COMPLETE`) |

`receipts/pairs/*.json` are the immutable evidence. Derived reports are
regenerated from them; the 480 content-addressed battle receipts stay on
miniDC at `/home/minidc/eval-league/roundrobin/v1/receipts/*.json`.

## Reproducing derived results

```bash
cd /home/minidc/Desktop/PokeAgent3.0-AHL-fix
PY=/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python
RR=/home/minidc/eval-league/roundrobin/v1

# Recompute matrices/ratings/reports from existing pair summaries (no battles)
"$PY" scripts/v2_history_roundrobin.py --root "$RR" report

# Plan recurring milestones (idempotent, plan only)
"$PY" scripts/v2_history_roundrobin.py --root "$RR" schedule

# Resource gate / protected proof
"$PY" scripts/ahl_v2_resource_gate.py --root "$RR" --ppo-stopped
"$PY" scripts/v2_history_protected_snapshot.py --root "$RR" \
  --output "$RR/protected_snapshot/corrections_verification.json"
```

Do **not** activate `opponent_plan.json`. Historical PPO training remains
disabled at probability 0.0.

## Related documents

- `docs/operations/AHL_V2_ROUND_ROBIN.md` — design, correction details and results
- `docs/operations/AHL_V2_DIAGNOSTIC_SCHEDULE.md` — recurring schedule runbook
- `docs/operations/AHL_V2_CPU_TOPOLOGY.md` — affinity/topology analysis
- `FINAL_REPORT_CORRECTIONS.md` — Korean correction report
- `FINAL_REPORT.md` — preserved smoke-era stage A–C report (superseded)
