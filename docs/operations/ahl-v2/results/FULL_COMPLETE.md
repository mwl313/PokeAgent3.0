# AHL v2 — FULL_COMPLETE historical round-robin report

State: **FULL_COMPLETE**  
Generated: 2026-10-17T17:42:39+0900

> Diagnostic only. This report has no promotion authority and spent zero
> formal alpha. The official champion pointer and formal evaluation
> schedule are owned by the V2 evaluation league.

| Item | Value |
|---|---|
| Policies | 30 (30 distinct digests) |
| Pairs | 435/435 |
| Games | 77,184/77,184 |
| Basic pairs / games | 379 / 48,512 |
| Fine pairs / games | 56 / 28,672 |
| Champion at run | `v2-000002600960-da35700f` |
| Latest at run | `v2-000002764800-da35700f` |
| Evaluated window | 2026-10-17T06:47:56.022115+00:00 → 2026-10-17T07:37:18.256212+00:00 |
| Registry snapshot SHA256 | `24da9771e2f78e176cca8e04fe81b98cfebd4085b00cbdbc0af98f370177aa66` |
| Matrix SHA256 | `14afed32c4d2c65caeb4f5ba132d3ce3092ca1e973a26acaadd925ed9e8066c3` |
| Formal alpha spent | 0 |

## Validation

- Matrix/registry validation: **PASS**
- Completeness: **PASS**
- Unique pair summaries: 435
- Duplicate/missing/identity errors: 0

## Bradley-Terry correction

The smoke-era `ratings.csv` used `updated = numerator / denominator`,
which drops the previous-strength factor required by the MM update.
The corrected fit uses `updated = strength * numerator / denominator`,
normalises the geometric mean to one, and reports pair-cluster bootstrap
uncertainty. Smoke-era rating conclusions are superseded.

## Evidence

- `matrix.csv` — one row per immutable pair summary (435 rows for the complete grid)
- `ratings.csv` / `ratings.json` — corrected descriptive ratings and uncertainty
- `ranking_confidence.csv` — adjacent-rank pair-cluster bootstrap confidence
- `receipts/pairs/*.json` — immutable per-pair evidence (not rewritten)

