# PokeAgent3.0 V2 — Training Source of Truth

## Current owner decision — 2026-10-17 17:45 KST (authoritative)

The owner decision below overrides the earlier "proposed change" paragraph.
Historical training is **not** authorized. The historical league is a
recurring independent diagnostic only.

- Production PPO remains **100% current-vs-current** self-play. No reward,
  model, hyperparameter, source or Stage D migration change is authorized.
- `training.historical_probability` remains **0.0** in every active production
  configuration (`configs/pa3_v2_production.yaml`,
  `configs/validated_a94_runtime.yaml`, `configs/a10_production.yaml`).
  `configs/train.yaml` is non-normative historical material; it is not an
  active production profile and is not used to launch training.
- No Stage D historical PPO rollout migration is authorized. Stage D remains
  `NO_GO` pending owner approval, a reviewed source/profile/opponent migration,
  an isolated A/B pilot and rollback proof.
- The full **30-policy / 435-pair / 77,184-game** historical round robin has
  completed. Evidence: `docs/operations/ahl-v2/results/FULL_COMPLETE.md`,
  `run_report.json`, `matrix.csv`, and 435 immutable pair summaries.
- The Bradley–Terry calculation was corrected to the MM update
  `strength_i * wins_i / sum_j n_ij * strength_i/(strength_i+strength_j)`, with
  geometric-mean normalisation, convergence checks and pair-cluster bootstrap
  uncertainty. Derived ratings/reports were regenerated from the immutable pair
  summaries. Smoke-era rating conclusions are superseded.
- Historical evaluation is now established as a **recurring diagnostic
  service**: Tier A every 200K committed matches starting at 2.8M (≤ 8 anchors,
  256 games each), Tier B every 1M starting at 3.8M (≤ 24 distinct model
  digests). The schedule is persisted in `diagnostic_state.json` /
  `milestone_schedule.json` and is idempotent, overshoot-aware and resumable.
- The expanded Tier B population is capped at **24** distinct model digests by
  default. Participants are deduplicated by digest, always include the latest
  candidate and official champion, retain stable long-term anchors, and
  progressively widen temporal spacing for older snapshots.
- Diagnostic evaluation **cannot promote policies, cannot spend formal alpha,
  cannot move the champion pointer, and cannot rewrite formal receipts**. It is
  CPU-only, low priority, keeps a 12 GiB disk reserve, yields to pending formal
  evaluation and cannot schedule ahead of it.
- The smoke-era `opponent_plan.json` is **OBSOLETE and non-activatable**
  (`activatable=false`, `history_probability=0.0`, `activation=FORBIDDEN`).
  Its bytes are preserved as `opponent_plan.smoke-20261017.json`.
- Production PPO remains protected under source SHA
  `da35700f75c95833b0e8250d7b4b04e11320f8b7`, stopped at 2,764,800 matches /
  iteration 675, with `STOP` present. The 2,764,800 checkpoint and its manifest
  are unchanged.
- Any future training-source migration requires independent review and a new
  verified restore/migration receipt. No restart is authorized here.
- Historical PPO training may be reconsidered **only** after validated,
  reproducible historical regressions and explicit owner approval. A diagnostic
  alert is not activation.

### Verification snapshot (2026-10-17)

- `docs/operations/ahl-v2/results/protected_snapshot/corrections_verification.json`
  records `unchanged=true`, zero critical mismatches, training HEAD `da35700`,
  evaluation HEAD `16cc0bf`, STOP present, and alpha SHA256 `6fdf62a3…`.
- Live formal evaluation: 29 completed job receipts, worker+timer active and
  idle, no pending/running formal jobs at inspection time.
- `schedule.json` is rewritten by the formal `v2-eval-discover` timer; it is
  reported separately and is never written by historical diagnostics.
- Live-PPO coexistence of the diagnostic pool is **UNVERIFIED** because
  production training is stopped. The resource gate, topology-derived CPU
  masks and failure-safe thresholds are implemented and unit-tested, not
  measured under live PPO.

---

## SUPERSEDED PROPOSAL — Historical League (2026-10-17 earlier)

> Kept for provenance only. The authoritative current policy is the owner
> decision above.

### Earlier 2026-10-17 status text (superseded)

> **Priority dated correction.** The historical 2026-10-16 `PRODUCTION STOPPED at 57,344 / release pending` paragraphs below describe an *earlier migration review*. They are **not the current operational state**. Do not rewrite or delete their original evidence. For the paused campaign, authoritative current operational records are `docs/ops/PRODUCTION_TRAINING_SPEC_20261017.md`, `docs/ops/production_training_spec_20261017.json`, and `docs/ops/verify/VERIFY_20261017.json`. This addendum is a documentation proposal until reviewed and merged.

### Verified paused campaign (not new training)

- Frozen training source: `codex/v2-p00b-profile-propagation` @ `da35700f75c95833b0e8250d7b4b04e11320f8b7`. Immutable training worktree `/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix`; SHA-pinned production profile `configs/pa3_v2_production.yaml` (`c816aa28f695ce57ac2319db37cd9edeab308f369c2972606abc38a11fa27941`). PPO-2, 2× V100 DDP, Mode E, `historical_probability=0.0`, 1,220 frozen training teams.
- Cooperative stop at **2,764,800 matches**, iteration **675**, optimizer steps **24,573**; LR `0.00014978839419625816`, AMP scaler `262144`, rank digest equality true, no skipped optimizer steps or operational errors. Full verified restore generation: `iter-000002764800-1792212738467610757`; rank0 SHA256 `fd3b5655a2b7d602043531cf31addcbca53c4c41523dce4abc430a8ea52fe02b`, rank1 SHA256 `6cd1ae7b84a12bbc8e12bc0e30f538ee59d03e4dd602f68eb47ed0f079b39607`.
- Evaluation V2 patched in `810ee3a` to use `learner-rows-per-second-v1` for resource gating, not matches/s. Observed current-only baseline ~381.9 learner rows/s, stale training log skip, min two completed iterations before defer, warnings reduced, disk reserve enforced; 44/44 focused resource gate tests recorded as passing.
- At pause/report: **30 archived policies and snapshots, 29 completed evaluation receipts, 11 promotions**, champion `v2-000002600960-da35700f`. Lightweight diagnostics for 2,703,360 and 2,764,800 gave 0.4765 and 0.4824 versus incumbent over 170 games, both `DIAGNOSTIC_ONLY`; no pending evaluation jobs reported. Next nominal 200K full milestone at 2,800,000 (actual generation clock can overshoot). Existing alpha ledger and the independent confirmation procedure remain authoritative.
- Local cleanup: 35 to 4 worktrees, ~73GB disk free; **remote branches were not pruned**. Main, training and evaluation refs need to be distinguished; branch names alone do not establish the deployed source.
- The **power-off guard is paused** while training is stopped; it requires a separate check on approved restart.

### Proposed change — NOT YET ACTIVE

- Introduce an isolated historical diagnostic store and an initial full round robin over the exact compatible frozen registry (30 entries before digest dedup, up to 435 pairs), followed by incremental 8-anchor matchups every 200K full milestone, independently of official promotions.
- Keep historical diagnostics read-only relative to promotion league and training; separate seeds, snapshots, locks, receipts, risk budget (no formal alpha consumption), and evaluation priority. Use pair-cluster uncertainty rather than independent seat-swap game CI.
- Shadow selector only at first: up to eight historical opponents, experimental PFSP mix 60% hard / 30% close / 10% exploratory. The active production training probability stays **0.0** during shadow work.
- Proposed pilot after explicit new source/profile/opponent migration and verified checkpoint restore: 5% *current-policy learner rows* from current-vs-history games (roughly 9.52% of games given equal seat lengths); consider 10% only after controlled comparison. No autonomous 20% increase or resume.
- `scripts/run_ddp_ppo.py` currently supports historical envelope loading, but reviewed V2 migration pins 0% history and the opponent-manifest shape must be examined. Preserve all DDP, recompute, secondary-audit, stop/resume, team statistics, and team pool invariants. A new migration protocol and new release receipt are required before production use.
- Implementation spec: `PA3_Adaptive_Historical_League_v2_20261017/DESIGN.md` in the agent handoff package. Once implementation and deployment are **actually verified**, replace this PROPOSED paragraph with the real versioned artifact, commit, ratios, migration receipt, and acceptance results.

---

2026-10-16 KST: **PRODUCTION STOPPED**. The user approved the versioned FP32 policy encoder route. Final validation status, benchmark evidence and restore results are in [the migration/stability/batch report](operations/V2_MIGRATION_STABILITY_BATCH_OPTIMIZATION_FINAL.md). Delivery branch: `codex/v2-secondary-precision-stability`; tested executable: `9d3d6051d6d17cc517259a3e6807bb067c717a9e`. Work is paused for the user's precision decision; final HEAD-specific receipt remains pending after the documentation commit in local `runs/v2_final_release/final_delivery_20261016/FINAL_DELIVERY.json`.

Authority: immutable checkpoint and receipts → exact frozen source → versioned profile → frozen observation/action contract → this index → historical research. Config semantic hash `11e69c9940199692ba9b2d4bc1848469843dd294893bb2cd5c53ce75bc8790d2`. Diagnostics never advance production progress. No production restart, main merge, history rewrite or original checkpoint modification occurred.

Goal: Pokémon Champions Regulation M-C VGC doubles agent, native Rust/PyO3 engine, V2 Entity Transformer with9,889,755 parameters, synchronous current/current scratch-origin PPO, target100M naturally completed matches. Both seats learn; one natural battle is counted once. Track C remains deferred indefinitely and protected.

## Candidate executable settings

| Setting | Value |
|---|---|
| training.algorithm | `"ppo"` |
| training.ppo_epochs | `2` |
| training.lr_scale | `0.5` |
| training.global_minibatch | `4096` |
| training.microbatch | `1024` |
| training.per_rank_minibatch | `2048` |
| training.grad_accumulation | `2` |
| training.envs | `1024` |
| training.workers | `16` |
| training.historical_probability | `0.0` |
| training.optimizer | `"adam"` |
| training.clip_epsilon | `0.2` |
| training.gamma | `1.0` |
| training.gae_lambda | `0.95` |
| training.entropy_coefficient | `0.01` |
| training.uniform_kl_coefficient | `0.001` |
| training.target_approx_kl | `0.03` |
| training.value_coefficient | `0.5` |
| training.max_grad_norm | `0.5` |
| training.action_temperature | `1.0` |
| training.warmup_matches | `250000` |
| training.horizon_matches | `100000000` |
| training.peak_lr | `0.0003` |
| training.effective_peak_lr | `0.00015` |
| training.checkpoint_interval_matches | `100000` |
| precision.mode | `"E"` |
| precision.stability | `"fp16-reduction-off"` |
| precision.observation | `"fp16-round-continuous-then-fp32"` |
| precision.encoder_amp | `false` |
| precision.ragged_effects_fp32 | `true` |
| precision.scorer_fp32 | `true` |
| precision.gru_fp32 | `true` |
| precision.recompute_gate | `true` |
| precision.tolerance | `0.001` |
| precision.contract_version | `"pa3-precision-E-v1"` |
| precision.recompute_secondary | `true` |
| precision.canonical_encoder_batch_rows | `1024` |
| precision.amp_overflow_retries | `3` |
| precision.policy_encoder_fp32 | `true` |
| precision.encoder_physical_batching | `"logical-fp32-v1"` |
| collector.batch_cache | `"cuda"` |
| collector.compact_candidates | `true` |
| collector.trim_observation_padding | `false` |
| collector.columnar_store | `false` |
| collector.rolling_slots | `false` |
| collector.executor | `"ddp"` |
| collector.world_size | `2` |
| collector.deterministic_diagnostic | `true` |
| collector.token_lookup_backend | `"exact-row-v100-v1"` |
| dataset.id | `"a10-p0-proposed-20261013"` |
| dataset.manifest | `"data/teams/a10-p0-proposed-20261013/manifest.json"` |
| dataset.manifest_sha256 | `"1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb"` |
| dataset.train_teams | `1220` |
| dataset.sampling | `"uniform_unique_team"` |
| dataset.frozen | `true` |
| dataset.teams | `"engine/data/a10-training-teams.json"` |
| dataset.data | `"engine/data"` |
| dataset.teams_sha256 | `"ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f"` |
| evaluation.lightweight_every_matches | `50000` |
| evaluation.full_every_matches | `200000` |
| evaluation.backend | `"batched"` |
| evaluation.batch_size | `128` |
| evaluation.fixture_pairs | `1024` |
| evaluation.namespace | `"pa3-v2-only"` |

Mode E (`pa3-precision-E-v1`) retains all96 token slots and Mode-D observation quantization. Policy encoder, scorer and GRU execute in FP32; the value head retains existing learner AMP and GradScaler. E uses logical FP32 encoder row batches. The canonical1024 setting remains recorded for compatibility; AMP canonical padding is not used by E. Architecture, masks, five ragged families, candidate legality, signed targets, player-visible information and PPO hyperparameters are unchanged. Observation-padding trimming is disabled.

Exact-order V100 token lookup is enabled in this candidate after matched restored-state benchmarks and independent full-gradient/Adam/model equivalence. Device/compiler guards remain. Existing Actor optimizations remain; two additional transfer experiments were not promoted because their marginal gain was within observed variation.

Primary tolerance remains0.001. Secondary provenance/replay and grouped-versus-isolated FP32 checks stay enabled and are forced during final sustained validation. Collective AMP overflow retry retains the identical minibatch, normal scaler backoff, bounded three retries and no Adam step with nonfinite gradients. The additive v3 migration receipt explicitly authorizes D→E policy precision, encoder execution, effects FP32, secondary safety, retry and lookup transitions; it pins exact executable blobs, native/data hashes and full original rank artifacts. Actual runtime CLI values must match the approved receipt.

## Progress and continuation

**Production is STOPPED. Production matches added:0.** Existing resumable recovery remains57,344 matches, absolute iteration14 and718 optimizer steps, source `5ccc51b0cd4a49efb93018258b9750d5e804ae32`. Original rolling pointer remains4,096; failed102,400 is excluded. The separately reconstructed recent RNG has its previously documented historical limitation; restoring the selected recovered checkpoint is exact. Isolated E checkpoints are validation evidence and are not silently promoted to production.

Any future production launch requires the final HEAD-specific reviewed receipt, verified57,344 full-state checkpoint, fresh live preflight and separate run/statistics lineage. Preserve original files and counters. Cooperative STOP must drain collection/update and publish both ranks before resume. Keep evaluation isolated from training accounting and use the versioned E archive identity. Storage reserve and validation gates remain unchanged.

Historical V1/PPO4/V5/V6 documents and `configs/train.yaml` are non-normative. PPOConfig's default4 epochs remains unchanged; the selected profile explicitly uses2. Historical forced Mode-D audit failures and rejected optimization branches remain preserved as evidence.
