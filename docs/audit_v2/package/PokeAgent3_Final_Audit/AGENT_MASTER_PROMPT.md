# MASTER AGENT PROMPT — PokeAgent 3.0 Full Correctness Repair and Scratch PPO V2

## 0. Role and nonnegotiable objective

You are the lead engineer and correctness auditor for the Ubuntu MiniDC PokeAgent 3.0 dual-V100 PPO project.

The user has authorized abandoning the original approximately 1.33M-match A10-trained weights **as a source for future model initialization** and wants a new, clean PPO training campaign after comprehensive correction. The prior model, optimizer, logs and independent evaluation results are historical evidence that must NOT be deleted or silently modified.

Deliver production-quality code, automated regression tests, a full evidence dossier, per-team training statistics and a verified new scratch PPO campaign. **Do not respond with a plan alone. Implement the fixes.**

This prompt is the dispatcher for the bundled documentation package. Treat its documents as task specifications to be checked against the **actual current local code**, not unquestionable assertions. The local source may contain corrections absent from the audited GitHub revision.

## 1. First action: ingest and reconcile the package

Read every supplied document in order:

1. `00_START_HERE_KO.md`
2. `01_AUDIT_VERDICT_EVIDENCE.md`
3. `02_DEFECT_REGISTER.csv`
4. `03_PHASED_EXECUTION_PLAN.md`
5. `04_OBSERVATION_V2_CONTRACT.md`
6. `05_ACTION_V2_CONTRACT.md`
7. `06_PPO_DDP_ENGINE_VALIDATION.md`
8. `07_TEAM_STATISTICS_SPEC.md`
9. `08_CLEAN_RETRAINING_RUNBOOK.md`
10. `09_TEST_MATRIX.csv`
11. `10_REPORT_TEMPLATES.md`

Package baseline reviewed: `mwl313/PokeAgent3.0` branch `codex/a10-ppo-training-telemetry` @ `d10ac692009566686fd32285c4456309a5f383a7`.

Reported post-recovery local commits: `266bb65` and `3df5abd`. Determine which is in the actual local production source and reconcile all other local branches, hotfixes, telemetry improvements and executable Rust wheel SHA.

Do NOT assume the remote GitHub branch is the actual production source.

Initial operational paths:

- Workspace example: `/home/minidc/Desktop/PokeAgent3.0-A`.
- Original run root: `/home/minidc/Desktop/PokeAgent3.0-A/runs/a10_production_20261013_v2`.
- Original independent Evaluation League may be running; preserve it.

Read-only inventory first. Avoid destructive or broad process-kill commands. If the old PPO is active, stop that PPO only using its supported safe coordinated path. Preserve all legacy checkpoint generations, hashes, logs and League evidence. Record the previous model as V1/legacy and incompatible with the new observation/action contract. The user authorizes scratch retraining; they do NOT authorize deleting old evidence.

## 2. Diagnosis rules

The package has three classes of findings:

- **CODE-CONFIRMED**: static producer-consumer mismatch in audited revision. Prove the mismatch in current local code; fix if still present. Do not infer the magnitude of training damage solely from static evidence.
- **RISK-REPRO**: credible but not proven defect. Build a minimal deterministic failing fixture FIRST. If disproved, retain counter-evidence and regression test, do not make a speculative fix.
- **DESIGN / NOT-A-BUG**: optional enhancements and earlier claims withdrawn after review. Preserve existing correct logic (notably duration upkeeps and Mega/Mimicry type updates).

Do not claim that all 1–5 audit findings are equally confirmed. Update the register with `REPRODUCED / DISPROVED / FIXED / BLOCKED` and links to concrete artifacts.

## 3. Mandatory correction domains

### A. Engine and public information

- Complete paired-state tests for hidden information and Imprison-specific `MoveDisabled` leakage.
- Correct legitimately observable Soak, Magic Powder, Double Shock, Burn Up, Protean/Libero typing propagation to player-safe Knowledge, without revealing concealed identity.
- Test temporary type/form reversal on switching; only change public Knowledge where appropriate.
- Verify pending vs public request semantics and stale benched disabled flags.
- Trace Transform, Illusion, Mega, Mimicry, volatile effects, event ordering, turn lifecycle and form identity.
- Compare reachable actual 1,220-team mechanics to the pinned Showdown reference and classify unsupported paths without pretending the engine supports all Regulation M-C.

### B. Observation V2

- Version a corrected 96-token observation with five neural-consumed ragged groups (`effects`, `repertoire`, `types`, `base_moves`, `move_effects`).
- Preserve per-player legality/publicity; ensure no hidden or future data enter the policy.
- Differentiate known zero/false from unknown for categories, continuous values and booleans.
- Fix the FP16-destructive event change normalization (±1 and ±6 must remain distinct) and inappropriate small-value scaling for weight; audit every numeric feature's range.
- Validate category ID ranges and reject impossible known IDs rather than silently collapsing them with clamp.
- Propagate the contract through Rust wire, Python typed view, SoA/rollout buffers, model, DDP, evaluator, checkpoints and all source/dataset digests.
- Add deliberate malformed/mismatched schema tests and fail closed.

### C. Action V2

- Decode signed target `i8` in the model scorer path, not merely when submitting an action. Distinguish all valid native target locations `[-2,-1,0,1,2]`.
- Resolve actor active battlefield position to actual roster member for MOVE, SWITCH, PASS, PICK and replacement semantics; the two active Pokémon are NOT generally roster indices 0 and 1.
- Make Typed, Packed, Batched Evaluation, actor collection, stored PPO rows and learner recompute fully equivalent in candidate ordering, identity, legal masks, and token references.
- Verify conditional four-branch constraints, Mega usage, duplicate switches, Struggle, locked moves, force replacements and candidate capacity.
- Explicitly decide whether direct target-token and switch-destination token improvements are worth integrating in scratch architecture; do not misclassify them as proven correctness bugs.

### D. PPO and DDP verification

- Preserve known-good PPO mathematics unless independent tests show a real issue; reference-check terminal rewards, both current-policy sides, GAE by `(match_id, side)`, advantages, PPO objective/regularizers and masking.
- Require semantic ground-truth observation checks **in addition to** the old behavior-logprob recompute gate. Matching wrong actor and learner inputs is not proof of semantic correctness.
- Verify two V100 DDP ranks: matching model and Adam digests, exact rank gradient weighting, same schemas/configs, no illegal actions, no unexpected skipped steps.
- Preserve telemetry: Policy Loss, Value Loss, Approx KL, Entropy, Uniform KL, Clip Fraction, Gradient Norm, LR, GradScaler, errors, real throughput.
- Keep recompute tolerance at validated value unless independent mathematical evidence justifies a change. Do not remove correctness gates to increase throughput.

### E. PPO team statistics (required functionality)

Implement `07_TEAM_STATISTICS_SPEC.md` completely. Aggregate each **natural** completed match into exactly one match event and two team appearances. Track 1,220 canonical IDs, W/L/D, seats, opponent team and policy, checkpoint/window, and mean length. Store rank-local immutable iteration summaries, checkpoint-generation-bound commits, deduplication/replay-safe receipts, global reconciliation, CLI and machine-readable reports. Enforce the `2 * committed_matches` appearance invariant. Errors, aborted games and rolled-back iterations never count toward durable totals. Team IDs never enter neural input.

### F. Dataset and evaluation contract

CRITICAL: The older validated A9.4 runtime config points to `mb-mc-v3-userteam-all-train`, 1,137 teams, while `data/teams/a10-p0-proposed-20261013/manifest.json` states 1,220 train teams and 0 dev/final teams. Identify the **actual local authoritative 1,220-team file, content fingerprint, native loader indices and rank manifests** and publish explicit overrides/new V2 profile. Do not silently run the old 1,137 pool. Do not assume old production used the wrong pool without proof.

Keep a separate version-compatible V2 Evaluation League. Explicitly report that the all-train pool does not measure unseen-team generalization; arrange a future independent unseen-roster test panel if possible without mutating the 1,220 training teams.

## 4. Exact execution sequence and gates

Implement phases from `03_PHASED_EXECUTION_PLAN.md`:

- **P0:** source reconciliation, process inventory, old-run evidence backup, exact dataset identity, safe isolation, integration branch.
- **P1:** engine public Knowledge/hidden information/Showdown parity and reproducible fixtures.
- **P2:** model-consuming Observation V2, precision and strict wire/rollout/checkpoint identity.
- **P3:** canonical signed Action V2 and active roster mapping, typed/packed parity, optional rolling slot correctness.
- **P4:** independent PPO/DDP validation, crash-safe per-team stats, separate eval, 2-V100 smoke.
- **P5:** only if all P0 release gates pass, start new scratch production and produce actual verified results.

At each phase, create a short artifact-based handoff using `10_REPORT_TEMPLATES.md`; update `02_DEFECT_REGISTER.csv` and `09_TEST_MATRIX.csv` with results, source SHA, exact test command, log and known blockers. Read and honor previous phase evidence rather than overwriting it.

If a P0 cannot be fixed and tested, do not launch V2 production. Report blocking evidence rather than claiming an ambiguous PASS.

## 5. Scratch-training configuration and safeguards

On successful release gates, create a **new run ID and directory**, new random-init V2 model and fresh optimizer, scheduler, RNG and GradScaler. Strictly prohibit loading V1/A10 checkpoint weights for the new training model.

Use the validated A10 PPO starting settings, subject to explicit correctness revalidation:

- 1,220 actual frozen training teams; historical opponent probability 0.
- Two Tesla V100 32GB PCIe, distributed world size 2, PPO epochs 2.
- Effective peak LR `1.5e-4` (LR scale 0.5), global minibatch 4096, microbatch 1024.
- Mode D and FP16 reduction disabled only after corrected numeric V2 passes; no silent precision or entropy/KL coefficient change.
- 100M natural-match target and full generation-based restore receipts.

Commit initial configuration identity and record at least one verified production iteration with detailed telemetry + per-team committed stats; distinguish a two-V100 smoke PASS from long-term stability.

Do not kill or corrupt the old Evaluation League. Separate V1 and V2 evaluation cohorts and model compatibility.

## 6. Final required output

After implementation, supply:

1. Exact local authoritative source SHA and dataset SHA, active engine wheel/build SHA, comparison against remote audited baseline.
2. Updated defect register with confirmed, disproved and fixed items and evidence artifacts.
3. Source files changed, separate commits and test suite invocation/results.
4. New Observation V2 and Action V2 schemas and strict compatibility version hashes.
5. Evidence for hidden-information isolation, signed targets, actor identity and five ragged feature groups reaching neural network.
6. Native Showdown mechanics coverage status and explicitly remaining unsupported cases.
7. Independent PPO/GAE/DDP correctness, recompute parity and two-rank model/optimizer digests.
8. Team statistics reports and proof of `2 * committed matches` reconciliation and rollback/replay idempotence.
9. Distinct new scratch run: initial random weights verified, optimizer step 0 receipt, current committed matches, real throughput, losses, Entropy, KL, GradScaler, last checkpoint, stats and new Evaluation League paths.
10. Any unresolved blockers and the exact safe restore state. Do not state completion if production could not be safely launched.

## 7. Safety and engineering discipline

- Never delete historical runs, rewrite frozen 1,220 sets or lower gates to force PASS.
- Fail closed on invalid native/typed/policy metadata, misaligned candidate order, wrong model schema or hidden-data leak.
- Measure controlled V2 overhead and optimize only after correctness.
- Prefer minimal proven fixes over global rewrites of already correct mechanics.
- At every release decision prioritize **policy observation fidelity, legal action identity and reproducible scientific evidence** above total matches/sec.

Begin immediately with the Phase 0 inventory and implement through the highest phase whose gates pass. Do not ask the user to reiterate information already contained in this package. If the run is blocked, report a precise, reproducible blocker with completed work rather than initiating a misleading training campaign.
