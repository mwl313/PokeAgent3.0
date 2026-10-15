# PokeAgent3.0 V2 — Training Source of Truth

**2026-10-15 KST — BLOCKED; PPO not restarted.** Canonical branch remains `codex/pa3-v2-correctness @ c96c35f12d3f3aff32b072b48e32ac243a7b0425`. Integration candidate is `codex/pa3-v2-integration-release`, based on frozen throughput `9e685b701f170436f9b9dd82162c8e192e4e6904`. No canonical fast-forward is authorized by failed release gates. Main and Track C unchanged.

This index describes actual state and the proposed executable release separately. Authority: immutable run manifest → verified checkpoint metadata → versioned production profile → frozen observation/action contract → this index → historical research. Proposed profile config hash: `6677f035c3ac855233af018116a81c455ba32fde4beab927496d3bd11f110884`. Original runtime JSON remains under `/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/release/`; do not rewrite its history.

Mission: train a strong Pokémon Champions Regulation M-C VGC doubles policy using native Rust/PyO3 engine, player-visible Observation V2/Action V2,9,889,755-parameter Entity Transformer, synchronous dual-V100 DDP PPO, per-team ledgers and an independent evaluation league. Scratch-origin training has already occurred; continuation must preserve existing weights and full state. Target100,000,000 naturally completed matches.

Approved strategy: PPO2 provisional from Track A throughput/learning evidence; frozen-policy current/current on-policy collection, both seats learn, one completed natural battle counts once. No historical mixture, search teacher, LLM battle loop, offline RL or behavior cloning. Production Full Spec historical defaults are not silently changed.

## Proposed executable settings (not current live training)

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
| precision.mode | `"D"` |
| precision.stability | `"fp16-reduction-off"` |
| precision.observation | `"fp16-round-continuous-then-fp32"` |
| precision.encoder_amp | `true` |
| precision.scorer_fp32 | `true` |
| precision.gru_fp32 | `true` |
| precision.recompute_gate | `true` |
| precision.tolerance | `0.001` |
| precision.contract_version | `"pa3-precision-D-v2"` |
| precision.recompute_secondary | `true` |
| collector.batch_cache | `"cuda"` |
| collector.compact_candidates | `true` |
| collector.trim_observation_padding | `false` |
| collector.columnar_store | `false` |
| collector.rolling_slots | `false` |
| collector.executor | `"ddp"` |
| collector.world_size | `2` |
| collector.deterministic_diagnostic | `true` |
| evaluation.lightweight_every_matches | `50000` |
| evaluation.full_every_matches | `200000` |
| evaluation.backend | `"batched"` |
| evaluation.batch_size | `128` |
| evaluation.fixture_pairs | `1024` |
| evaluation.namespace | `"pa3-v2-only"` |
| dataset.id | `"a10-p0-proposed-20261013"` |
| dataset.manifest | `"data/teams/a10-p0-proposed-20261013/manifest.json"` |
| dataset.manifest_sha256 | `"1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb"` |
| dataset.train_teams | `1220` |
| dataset.sampling | `"uniform_unique_team"` |
| dataset.frozen | `true` |
| dataset.teams | `"engine/data/a10-training-teams.json"` |
| dataset.data | `"engine/data"` |
| dataset.teams_sha256 | `"ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f"` |

Dual Tesla V100 PCIe32GiB,175W each; deterministic CUDA and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; ModeD encoder AMP, scorer/GRU FP32, FP16 reductions disabled. No microbatch change. Both GPU lookup experiments OFF in production. Five ragged families, known masks, signed targets, active-roster mapping and joint action logprobs remain mandatory. Semantic hash `dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35`.

Frozen1,220 teams, manifest SHA `1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb`, team SHA `ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f`. Uniform unique-team sampling. No official held-out split; do not claim unseen-team generalization from this panel. Team IDs are accounting metadata, not neural opponent-team features.

Primary stratified recompute tolerance stays.001. Secondary activates on primary outliers, requires original actor-group content hashes/replay/repeat, stable FP32 group/isolate reference and bounded AMP residuals; budget8, fail closed. Old campaign/checkpoint secondary=false; proposed profile=true. No verified migration yet. Never edit stored logprobs to pass a gate.

## Actual progress and gates

Recovered complete57,344 checkpoint, source5ccc51b, iteration14; manifest/model/Adam/receipt identities in [checkpoint audit](operations/evidence/checkpoint_audit.json). Current historical campaign LATEST remains4,096.73,728 is model-only diagnostic evidence. Replayed RNG is not independently proven equal to the unavailable original recent RNG. Latest new-source full continuation: NOT EXECUTED. PPO is stopped; no current live progress or new evaluation strength result.

Historical benchmark47.32 games/s uses experimental exact lookup on1500 games/fresh Adam; production default gather43.24 in same diagnostic. Neither is a sustained production release receipt. [Full evidence](perf/v2_all_optimizations_20261015/REPORT.md).

Evaluation intended light50k/full200k, paired fixtures/batched backend, independent accounting. Actual saved CPU sidecar256 pairs/batch32 runs when new checkpoint appears, rather than implementing both intended cadence tiers. Strict V2 identity required; V1 weights rejected. Fix/validate schedule before cutover.

1. Historical iteration19 discrepancies (rank0 .0011582374572753906; rank1 .0017061233520507812) have **zero actor groups** in their saved evidence. Original-group replay/repeat/FP32 proof cannot be reconstructed from sampled rows alone. Secondary must reject; no harmlessness claim.
2. Full57,344 checkpoint records secondary=false and old source5ccc51b. Candidate requires secondary=true; no validated exact-source production migration exists. Existing bounded migration whitelist rejects protected model/observation/config changes. No bypass added.
3. Broad agent suite has unresolved failures, including candidate-padding scalar tolerance. Original-baseline reproduction is retained; passing short tests does not certify sustained release.
4. Production-shaped4096-game matched A/B, beyond-iteration19 continuation, merged-source DDP save/restore equivalence and actual evaluation schedule reconciliation are **NOT EXECUTED**. Release remains blocked.
Disk space is excluded as a release blocker per the user's later instruction. Existing resource-related test failures remain recorded honestly; no cleanup or guard disabling was performed.

## Verified inspection and validation commands

Run from the candidate worktree with `PY=/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python` and `PYTHONPATH=engine/python:.`.

```bash
$PY scripts/run_ddp_ppo.py --profile configs/pa3_v2_production.yaml --print-effective-config
$PY scripts/validate_v2_source_of_truth.py
$PY -m pytest tests/agent/test_a10_recompute_safety.py tests/agent/test_v2_operational_contract.py tests/agent/test_v2_source_of_truth.py tests/integration/test_imprison_retry.py -q
$PY -m agent.analysis.team_stats --help
$PY scripts/v2_evaluation_sidecar.py --help
ps -eo pid,args
nvidia-smi
```

Starting/resuming/rollback is deliberately **not a runnable approved command yet**; [continuation plan](operations/V2_CHECKPOINT_CONTINUATION_PLAN.md) specifies exact missing release/migration evidence. Safe-stop uses the actual launcher's `--stop-file`, requires checkpoint publication and completed drained ranks; never kill a trainer as an ordinary pause. Evaluation invocation writes its own namespace; do not invoke against the historical campaign during this blocked audit.

## Research and history

Track C: **DEFERRED_INDEFINITELY**, protected06dbb31, no merge/import/teacher activation. Historical self-play,PFSP,populations,BC,offline RL/search,batch/epoch changes require separate research authorization. Older1,137-team/V1/PPO4/V5/V6 documents and configs/train.yaml are historical, non-normative; preserve their evidence. [Readiness](operations/V2_TRAINING_READINESS_REPORT.md), [integration](operations/V2_BRANCH_INTEGRATION_REPORT.md), [cleanup proposal](operations/BRANCH_AND_WORKTREE_CLEANUP_PLAN.md).

