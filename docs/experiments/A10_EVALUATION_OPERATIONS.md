# A10 Evaluation Operations

`a10-evaluation.timer` is enabled as a user-level timer every ten minutes. Its service runs with `Nice=10`, `CPUQuota=25%`, one sidecar worker, and queue-only behavior. It discovers the latest complete rolling generation, writes a model-only immutable snapshot under `runs/a10_production_20261013_v2/evaluation/policies/`, updates `a10_registry.json`, and records schedule state. It never starts a CUDA process, signals PPO, writes `rolling/generations`, or modifies production checkpoints.

Inspect status with:

```bash
python scripts/a10_evaluation_status.py --run-root /home/minidc/Desktop/PokeAgent3.0-A/runs/a10_production_20261013_v2
systemctl --user status a10-evaluation.timer
```

Actual evaluation execution requires an explicit bounded CPU window and `--run-cpu`; it is not enabled automatically because a controlled before/after PPO throughput comparison has not been measured. The verified 8-game CPU smoke loaded the 102,400-match A10 snapshot, completed real engine battles, produced 5W/3L, zero illegal actions and zero operational errors in 3.50 seconds. This is execution evidence, not a strength claim.

Historical A3/A5 archive files remain untouched. Incompatible historical policies are not silently inserted into the A10 registry. Evaluation failure is isolated to sidecar state and cannot stop or restart PPO. Keep at least 15 GiB free and retain only bounded inference snapshots plus compact receipts.
