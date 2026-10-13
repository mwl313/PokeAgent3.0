# Evidence and report templates (agents fill, do not premark PASS)

## 1. `PHASE_EXIT_REPORT.md`

```markdown
# Phase <ID> Exit Report
- Branch / source SHA:
- Host / Python / Rust / CUDA / PyTorch versions:
- Dataset manifest SHA / NativeEngine team_count:
- Old production protected? Evidence location:
- Test commands executed:
- New/modified source files:
- Relevant defect IDs changed:
- Confirmed initial cause vs disproven suspicions:
- `09_TEST_MATRIX.csv` test IDs and actual results:
- Failing tests / blockers / reasons:
- Behavior/typed-packed/FP16/hidden-state parity evidence:
- Memory / games/sec impact, controlled comparison methodology:
- Artifact hashes and paths:
- Commit SHA:
- Authorized next phase:
```

## 2. `DEFECT_RESOLUTION_REPORT.md`

```markdown
| ID | Audit status | Local reproduction | Root cause | Code change | Exact regression test | Verdict |
|---|---|---|---|---|---|---|
| OBS-01 | CODE-CONFIRMED | <evidence> | <root cause> | <file/line> | T-OBS-01 <log> | FIXED/BLOCKED |
| KNOW-03 | RISK-REPRO | <paired states> | <if reproduced> | <if needed> | T-KNOW-03 <log> | FIXED/DISPROVED/BLOCKED |
```

**Never** assign `FIXED` from code review alone. Require a reproducer that failed before and passes after, or a documented oracle-based equivalent. If disproved, retain minimal evidence preventing regressions.

## 3. `CONTRACT_HASH_REPORT.json`

```json
{
  "source_sha":"<actual>",
  "engine_binary_build_id":"<actual>",
  "dataset_manifest_sha256":"<actual>",
  "team_count":1220,
  "observation_schema":"<version+hash>",
  "action_schema":"<version+hash>",
  "ragged_wire_schema":"<version+hash>",
  "precision_schema":"<version+hash>",
  "architecture_hash":"<hash>",
  "candidate_contract":"<version+hash>",
  "rank0_identity":"<actual>",
  "rank1_identity":"<actual>",
  "identities_equal":false
}
```

Value `identities_equal` must be calculated from actual receipts, not left as the sample false.

## 4. `NEW_PRODUCTION_LAUNCH_RECEIPT.md`

```markdown
# PA3 V2 scratch production activation
- Run ID and path:
- Prior A10 archive and immutable SHA evidence:
- New source commit and schema hashes:
- Config and dataset 1220 fingerprint:
- Random model initialization proof (old weights not used):
- Optimizer/scheduler fresh initial state proof:
- Rank IDs and GPU identity:
- Two-rank real smoke results:
- Actual committed natural matches:
- Latest iteration model/Adam digest parity:
- Policy Loss / Value Loss / Approx KL / Entropy / Clip Fraction / Grad Norm:
- Observed games/sec and run window:
- Team appearances and committed-match reconciliation:
- Team-stat committed generation receipt SHA:
- Evaluation League V2 identity:
- How to stop/resume safely:
- Remaining blockers or residual risks:
```

## 5. Open questions the local agent must answer explicitly

1. Which actual source/binary ran the original A10 1.33M matches? Does it differ from GitHub branch inspected?
2. Did original 1,220-team production use a profile override? Record loaded team file and SHA.
3. Which major effect families are reachable by the exact new training corpus, and are any demonstrably unsupported?
4. Did hidden-information noninterference tests confirm or disprove Imprison leakage?
5. Did temporary type/form switch reversion remain accurate without private leakage?
6. What exact percentage of model inputs are represented, known, unknown, dropped or saturated under V2?
7. Do all five ragged observation families survive Model/Buffer and reach the neural encoder?
8. Does every signed target map uniquely in Typed and Packed scoring?
9. Do real DDP rank weights and optimizer states match across multiple updates?
10. Are committed team totals reconstructible from generation receipts after crash/replay?

No missing answer should be filled by speculation.
