# PokeAgent3.0 V2 — Training Source of Truth

**READY_FOR_CUTOVER — 2026-10-15 KST. Production remains STOPPED at the user's explicit request.** Canonical `codex/pa3-v2-correctness` was ancestry-preserving fast-forwarded and pushed to the tested executable commit `0b7253cfd915ff61c88248b1979bd8e254285ef0`; subsequent changes are documentation/evidence only. No main/C-track change, destructive cleanup or production launch.

Authority: immutable run/cutover manifest → verified checkpoint → versioned production profile → frozen observation/action contract → this index → historical research. Validated executable SHA `0b7253cfd915ff61c88248b1979bd8e254285ef0`. Final canonical documentation SHA and exact resume receipt are in the final local cutover manifest. Config semantic hash `7eec659435f3eacf6f9f0396c925f181eb107fb924b16c3da103420e3b46e5ec` (versioned relative-path profile; absolute executed profile hash recorded separately).

Goal: strong Pokémon Champions Regulation M-C VGC doubles agent, native Rust/PyO3 battle engine, V2 Entity Transformer9,889,755 parameters, synchronous current/current scratch-origin PPO, target100M naturally completed matches. Both current-policy seats learn; one natural battle counted once. No search/LLM/teacher/historical mixture. Track C **DEFERRED_INDEFINITELY**, protected06dbb31 unchanged.

## Approved executable settings

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
| precision.canonical_encoder_batch_rows | `1024` |
| collector.batch_cache | `"cuda"` |
| collector.compact_candidates | `true` |
| collector.trim_observation_padding | `false` |
| collector.columnar_store | `false` |
| collector.rolling_slots | `false` |
| collector.executor | `"ddp"` |
| collector.world_size | `2` |
| collector.deterministic_diagnostic | `true` |
| dataset.id | `"a10-p0-proposed-20261013"` |
| dataset.manifest | `"data/teams/a10-p0-proposed-20261013/manifest.json"` |
| dataset.manifest_sha256 | `"1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb"` |
| dataset.train_teams | `1220` |
| dataset.sampling | `"uniform_unique_team"` |
| dataset.frozen | `true` |
| dataset.teams | `"engine/data/a10-training-teams.json"` |
| dataset.data | `"engine/data"` |
| dataset.teams_sha256 | `"ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f"` |

Dual V100 PCIe32GiB at175W; deterministic CUDA workspace4096:8. Fixed physical1024-row AMP encoder groups stabilize batch arithmetic; internal padding never reaches action scoring or match counts. Model architecture/PPO math unchanged; actual1024-row gradients/Adam bitwise match baseline. Default gather active; both experimental lookup backends OFF. Primary.001 unchanged; secondary bounded provenance/replay/stable-FP32 proof mandatory on outliers. FP32 references remain independent, grouped versus truly isolated.

Frozen1220-team pool ID a10-p0-proposed-20261013, manifest SHA1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb, team SHAecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f. Observation/Action V2 semantic hashdd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35. Five ragged families,known masks,signed targets/roster mapping,player-visible boundaries. No official held-out split or unseen-team strength claim.

## Actual progress and continuation

**Production is STOPPED.** Approved existing full-state restart57344 matches/iteration14/718 optimizer steps, source5ccc51b. Original campaign rolling/LATEST still4096; no pointer/counter was changed.73,728 historical model-only evidence remains nonresumable alone.57344 complete checkpoint and local backup are verified; original recent RNG was historically reconstructed and cannot be independently compared to unavailable original recent RNG. Restoring the selected recovered checkpoint itself is exact.

The isolated validation reaches81920/iteration20/982 optimizer steps, model1602e744a135df094f64e714170b479aace1999a6eb0cb5066be43ec0a738612, Adamff5a9d42b8c069d342d17df15317ea6fc546b1a40a8619c688ee51b8df514a87. **Diagnostic games are not production progress and that checkpoint is not silently promoted.** Real resume remains57344 with exact-source migration0→1024 encoder execution and secondary OFF→ON for future groups. All original weights/Adam/scheduler/scaler/RNG/ledger state preserved.

Two-rank validation15–20:2 epochs each,0 errors/skips, matching rank digests, weighted41.285 games/s,888.70 rows/s. Independent process restore and real stop-file save are BIT_EXACT to uninterrupted16, including full learner/RNG/rollouts and team W/L/D. This is bounded same-hardware proof, not universal stability; future unexplained outliers still stop safely. Historical iteration19 stays insufficient-provenance; prospective reproduced failure was correctly rejected and repaired through stable encoder execution.

## Evaluation: intended versus actual

Intended target: light50k/full200k,1024 fixture pairs,batch128. **Not active in this release.** Explicit minimal approved implementation: checkpoint-driven CPU-only256-pair/batch32 sidecar (normally100k checkpoint cadence), paired fixtures over actual1220 pool, independent V2 archive/counters and restart-safe tested identities. No A10/V1 results relabeled.4-game health check4W/0L/0D,0 errors/illegal actions, policy and checkpoint immutable; not a strength estimate. Automatic evaluator remains stopped; exact cadence awaits independent immutable snapshot work. CPU evaluation is the supported release plan; GPU evaluation is not enabled.

## Operations

Use the shared existing Python interpreter `/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python`, `PYTHONPATH=engine/python:.`, `OMP_NUM_THREADS=2`, and CUDA workspace`:4096:8` from a clean source checkout.

```bash
$PY scripts/run_ddp_ppo.py --profile configs/pa3_v2_production.yaml --print-effective-config
$PY scripts/validate_v2_source_of_truth.py
$PY -m pytest tests/agent tests/integration -q
$PY -m agent.analysis.team_stats --help
$PY scripts/v2_evaluation_sidecar.py --help
ps -eo pid,args
nvidia-smi
```

Actual launch command, absolute runtime profile, exact target-SHA migration and source/native/data hashes are prepared in a NEW release subdirectory of the original campaign; original release files unchanged. **Do not execute training in this task.** Latest user explicitly requests stopping before launch. Future start must repeat normal live preflight and use `--experimental-resume` exact57344, never scratch or old rolling/LATEST. Touch the configured release-specific STOP file for a cooperative boundary save; inspect both rank manifests/ledgers before resuming. [Cutover](operations/V2_PRODUCTION_CUTOVER_REPORT.md),[continuation](operations/V2_LONG_RUN_CONTINUATION_REPORT.md),[gates](integration/FINAL_RELEASE_GATES.json).

Historical1,137-team/V1/PPO4/V5/V6 documents and configs/train.yaml are non-normative and preserved. PPOConfig's original4-epoch default is unchanged; production profile explicitly selects2. No branch/worktree/checkpoint cleanup was executed. Production storage protections unchanged; diagnostic8GiB reserve is isolated and bounded only. Future research changes require separate authorization.
