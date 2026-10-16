# PokeAgent3.0 V2 — Training Source of Truth

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
