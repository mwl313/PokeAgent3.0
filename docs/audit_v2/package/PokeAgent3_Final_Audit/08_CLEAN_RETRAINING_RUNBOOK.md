# Clean retraining operating runbook

## Preconditions

Only the locally authorized agent on MiniDC can execute this runbook. No remote GitHub audit is proof that MiniDC's production build shares the same source. Preserve all old data; do not issue broad `pkill python`, `killall` or remove old run directories. The user's prior independent Evaluation League must remain untouched unless explicitly retired.

## A. Inventory (read-only)

Record actual working directory, branch, HEAD, dirty tree, `git diff`, `git status`, `git remote -v`, resolved launcher arguments, Python and Torch versions, Rust engine wheel/build SHA, NVIDIA driver/CUDA environment, two V100 temperatures and GPU memory, scheduler active state, active process commandlines and PIDs, dataset file+manifest digests, `NativeEngine.team_count()`.

Compare existing local commits `266bb65` and `3df5abd` against the remote audited `d10ac692...`; identify local-only telemetry/recovery improvements and their portability. Do not cherry-pick blindly.

## B. Old production lifecycle

- Determine if original PPO is running. If yes, prefer application's cooperative safe stop or bounded SIGTERM with verified DDP rank coordination and checkpoint generation; never SIGKILL during update/checkpoint unless there is a documented safety emergency.
- Avoid terminating or modifying independent Evaluation League's original PID and checkpoint inputs. If resources conflict, report exact competing GPU usage and get explicit scope approval before changing the league.
- Record last fully verified committed generation. Hash old checkpoint and telemetry receipts; archive by reference and mark training model V1 obsolete/incompatible.
- Do not delete 1.33M history. It can inform a post-mortem but may not seed V2 model weights.

## C. V2 implementation and gating

Use documents 01–07 and `09_TEST_MATRIX.csv`. Freeze signed action and observation schema first. Follow phased plan; always log local source SHA and tests. Any P0 failure blocks production; document unavailable Showdown reference as a blocker for the affected mechanics rather than assuming parity.

## D. New run naming and identity

Suggested directory distinct from historical path:

```
runs/pa3_v2_scratch_<UTC_DATE>/
  release/manifest.json
  checkpoints/
  telemetry/
  team_stats/staging/
  team_stats/committed/
  team_stats/reports/
  evaluation/league_v2/
  validation/
  reports/
```

`release/manifest.json` should include full source SHA, action/observation/precision/wire hashes, model architectural param count, dataset SHA 1,220 actual team count, seed, immutable config, GPU/world size, CLI argv, build/runtime versions, previous legacy run reference (read-only), and initial model/optimizer digests.

## E. PPO configuration

Start from validated A10 profile, re-resolve after V2 architecture changes:

- global minibatch 4096, microbatch 1024, DDP world_size 2, PPO epochs 2;
- LR scale 0.5 / effective peak LR 1.5e-4, initial Adam/Scheduler/GradScaler reset;
- historical opponent probability 0; 1,220 new frozen teams;
- Mode D with encoder autocast and FP16 observation roundtrip only after revised numeric contract test; FP16 reductions off;
- initial 100M natural matches horizon, checkpoints and gated recompute;
- same valid-sample/actor weighting and KL settings if confirmed by actual production config, with no silent tolerance widening.

If a severe correctness fix alters memory/precision requirements, rerun a bounded A/B and document deviations; never silently inherit inappropriate old performance expectations.

## F. Two-V100 smoke before production

1. Verify runnable native engine/fixture tests and typed/packed parity.
2. Run small full natural matches with both seats, no skipped actions, no operational errors or illegal actions.
3. Run 2-rank PPO-2 updates with model and optimizer digests equal after each verified iteration; inspect real Policy Loss, Value Loss, Approx KL, Entropy, Clip Fraction, Grad Norm, LR and GradScaler.
4. Prove semantic fixture integrity, not just recompute matching.
5. Confirm rank-local team counters and 2M reconciliation, crash/replay idempotence.
6. Measure games/sec and memory, compare controlled V1 baseline with V2 caveats. Don't set a performance threshold that forces correctness to be discarded.

## G. Launch and confirmation

Initialize from random state and fresh optimizer only. Confirm `model_init=fresh`, `optimizer_steps=0`, `committed_matches=0`; separate V2 schema and run path; do not load a prior `.pt` weights file under a misleading flag. Record initial model hash. Run first production iteration, validate rank parity, team stats receipt, effective checkpoint generation and current committed matches. Keep separate V2 Evaluation League on version-compatible inputs.

## H. Safe stop/restart in new run

At an iteration boundary, halt both ranks coherently; write and verify checkpoint and stats-generation receipt, then exit. On resume select latest fully verified generation, restore **model+Adam+scheduler+GradScaler+RNG+stats watermark**, and discard later tentative stats/replayed updates. Do not treat first-row restored stats as a replacement for detailed match accounting.

## I. Blockers and rollback

Blocking examples:

- hidden-information paired-state counterexample;
- ragged family produced but dropped from model;
- target/actor candidate mismatches;
- engine unsupported mechanics used in training without detection;
- DDP rank/model/optimizer digests disagree;
- old checkpoint or wrong team pool accepted;
- stats counters not reconcilable after resume;
- unexpected numerical divergences.

If blocked, do not launch or declare completion. Commit fixes separately and publish precise evidence, failing tests and remediation. Preserve previous verified release and the old A10 archive.

## J. Final user handoff

Show actual source SHA, number of corrected P0/P1 issues, proof tests, confirmed/false positive list, architecture and parameter count, actual 1,220 team hash, two-rank games/sec, committed matches, tracked teams coverage and first stats summary, V2 evaluation status, safe resume path and next unresolved risks.
