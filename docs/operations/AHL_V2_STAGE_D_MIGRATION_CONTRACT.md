# AHL V2 Stage D — reviewed history-activation migration contract (DESIGN ONLY)

Status: **PROPOSED / NOT AUTHORIZED / NOT DEPLOYED**. This document does not
authorize a training restart, a profile change, a checkpoint mutation or a
migration receipt. It defines the contract that must be reviewed and approved
before any historical PPO rollout may run.

## 1. Starting point

| Item | Pinned value |
|---|---|
| Frozen training source | `codex/v2-p00b-profile-propagation` @ `da35700f75c95833b0e8250d7b4b04e11320f8b7` |
| Worktree | `/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix` (read-only for this work) |
| Profile | `configs/pa3_v2_production.yaml`, sha256 `c816aa28f695ce57ac2319db37cd9edeab308f369c2972606abc38a11fa27941` |
| Pause clock | 2,764,800 committed natural matches, iteration 675, optimizer steps 24,573 |
| Generation | `iter-000002764800-1792212738467610757` |
| Model digest | `7697797df97cf3c6d5c11466c294c8a72be1bf06d437fb1dba0c8db5109f2e6e` |
| Optimizer digest | `bafeee072e87845b4dc7837d485efcf1ac79de70877fe051684dcab440d0e4c6` |
| Rank0 / rank1 SHA256 | `fd3b5655...fe02b` / `6cd1ae7b...39607` |
| History probability | `0.0` |
| Precision | Mode E, `pa3-precision-E-v1`, secondary recompute enabled |

The stopped generation remains the rollback source. It must not be modified,
replaced or used as a scratch area.

## 2. Why the existing migration is insufficient

1. `validate_migration_execution` (reviewed v3) explicitly pins
   `historical_probability = 0.0`. It authorizes a precision/source transition,
   not history activation.
2. `run_ddp_ppo.py` serializes an `opponent_manifest` that includes
   `historical_probability`, while the experimental migration comparison
   expects exactly `{current: "current", historical: []}`. The serialized shape
   can therefore fail its own comparison.
3. Historical evaluation snapshots are content-addressed evaluation artefacts,
   not training envelopes. They must not be assumed interchangeable with the
   `--historical-checkpoint` experimental envelope.

The activation must therefore introduce a **new, explicitly reviewed V2
migration** that recognizes the actual serialized old manifest and permits
only the declared opponent-sampling change.

## 3. Contract schema (proposed)

A migration contract is an immutable JSON document with this shape:

```json
{
  "schema": "pa3-v2-history-activation-contract-v1",
  "contract_id": "<sha256 of canonical body>",
  "created_at_kst": "YYYY-MM-DDTHH:MM:SS+0900",
  "status": "PROPOSED",
  "owner_approval": {"required": true, "record": null},
  "source": {
    "sha": "da35700f75c95833b0e8250d7b4b04e11320f8b7",
    "worktree": "/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix",
    "profile_sha256": "c816aa28...27941",
    "engine_sha256": "8f3643a9...d1c62",
    "venv_python": "/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python"
  },
  "generation": {
    "name": "iter-000002764800-1792212738467610757",
    "manifest_sha256": "<manifest sha256>",
    "rank0_sha256": "fd3b5655...fe02b",
    "rank1_sha256": "6cd1ae7b...39607",
    "model_digest": "7697797d...2e6e",
    "optimizer_digest": "bafeee07...e4c6",
    "committed_matches": 2764800,
    "iteration": 675,
    "optimizer_steps": 24573
  },
  "old_opponent_manifest": {
    "accepted_serialized_shapes": [
      {"current": "current", "historical": []},
      {"current": "current", "historical": [], "historical_probability": 0.0}
    ],
    "sha256": "<old manifest sha256>"
  },
  "new_opponent_manifest": {
    "schema": "pa3-v2-opponent-manifest-v2",
    "assignment": "historical",
    "history_rows_target": 0.05,
    "history_game_target_estimate": 0.09523809523809523,
    "pool_sha256": "<opponent_plan pool hash>",
    "selection_distribution_sha256": "<selector hash>",
    "pinned_policies": [
      {
        "stable_id": "md-<model digest>",
        "snapshot_sha256": "<file sha256>",
        "model_digest": "<model digest>",
        "committed_matches": 2600960
      }
    ],
    "assignment_rng_seed": "<derived seed>",
    "segment_clock_start": 2764800,
    "segment_clock_end": 2964800
  },
  "preserved": {
    "ppo_epochs": 2,
    "clip_epsilon": 0.2,
    "gae_lambda": 0.95,
    "lr_schedule": "unchanged, no rewarm",
    "amp_scaler": "unchanged",
    "secondary_recompute": true,
    "rank_digest_agreement": true,
    "team_pool": "frozen 1220 teams",
    "dataset_manifest_hash": "1aa64507...09bb",
    "history_rows_excluded_from_loss": true
  },
  "receipts": {
    "migration_receipt": null,
    "preflight_receipt": null,
    "rollback_pointer": null
  }
}
```

The contract body is hashed with canonical JSON. Any change to a pinned field
creates a different `contract_id` and invalidates prior approval.

## 4. Required validation before writing a receipt

1. Verify both rank payload SHA256 values and the generation manifest against
   `docs/ops/verify/VERIFY_20261017.json`.
2. Verify the source worktree HEAD is exactly `da35700` and the profile hash is
   `c816aa28...`.
3. Verify `STOP` is present and no training process or unit is active.
4. Verify the old serialized opponent manifest matches one of the accepted
   shapes byte-for-byte; reject arbitrary manifests.
5. Verify every pinned historical snapshot:
   - file SHA256 matches the plan;
   - model digest matches the plan;
   - architecture/observation/action/precision contract is V2 Mode E compatible;
   - no optimizer state is loaded;
   - load path is local and `weights_only` is not disabled for untrusted files.
6. Verify the pool is deduplicated by model digest and excludes the current
   instruction policy digest.
7. Verify the selector distribution is finite, normalized and cap-feasible.
8. Verify the new contract changes only the opponent-sampling declaration.
   Optimizer, LR schedule, AMP scaler, global match clock, team statistics,
   dataset hash, PPO hyperparameters and precision behavior must be identical.
9. Verify both ranks receive the same plan hash and the same assignment RNG
   seed, and that a restart reproduces the exact assignment stream without
   duplicating a step or a natural match.

Any failed check is a fail-closed condition. No partial migration receipt is
written.

## 5. Activation and receipts

Activation requires, in order:

1. Owner approval recorded in `owner_approval.record`.
2. A new integration worktree and branch derived from `da35700`; the frozen
   worktree is never edited.
3. An isolated training namespace and output root, not the paused production
   namespace.
4. A reviewed executable that contains the migration validator and the
   collector weighting change.
5. A preflight receipt containing all hashes above.
6. A migration receipt containing the approved contract hash, source/profile
   hashes, plan hash, rank digests before and after the first optimizer step,
   and the rollback pointer.
7. A rollback pointer to the untouched 2,764,800 generation.

The first optimizer step must not change the model or optimizer relative to the
pre-step digest check beyond the declared first update; the check exists to
prove the restore was exact and that no LR rewarm occurred.

## 6. Explicitly forbidden in this stage

- Editing `/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix`.
- Starting, stopping or restarting any training/eval systemd unit.
- Resuming PPO, clearing or moving `STOP`, or changing the power-off guard.
- Writing `alpha_budget.json`, `best_policy.json`, `schedule.json`, formal
  receipts, checkpoint payloads or team-stats roots.
- Creating a migration receipt from this document alone.
- Using GPU or the formal evaluation worker for this work.

## 7. Open questions requiring owner review

- Whether the pilot namespace may share the frozen team-stats root or must use
  a fresh root (current recommendation: fresh root).
- Whether historical inference runs on the learner GPUs or on a separate
  process; VRAM and deterministic-parity evidence are required either way.
- The exact approval artefact for a mode-specific `rows/s` rebaseline.
- Whether 10% is considered only after a repeated 5% cohort, as recommended.
