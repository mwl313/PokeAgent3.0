# Safe continuation and rollback — production cutover NOT APPROVED

The production service and evaluation timer were already inactive at task
entry; both V100s were idle. This work did not stop them and does not restart
either service. The frozen production worktree remains5ccc51b, unchanged.
No code is replaced underneath a live process.

Preserved campaign:
`/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z`.
Exact-resume generation:
`rolling/generations/iter-000000004096-1791958101732995099`.
Model-only73,728-match evidence: `reports/continuation.json.rank{0,1}.iteration19.recompute.pt`.
Untouched local duplicate backup: `backups/safe-preservation-73728-20261014/`.

The latter artifacts contain latest model weights, not Adam/scheduler/scaler/
RNG. They cannot continue the exact campaign. Copying weights into an older
optimizer or resetting optimizer would be a different experiment and is not
authorized by this cutover guide. The original team ledger and evaluation
archive remain untouched.

For exact future recovery, replay from the verified4,096 generation with the
original source/runtime and owned RNG, checkpoint every successful iteration
in an isolated recovery namespace, and compare the recovered73,728 model and
optimizer digests against saved telemetry. Only if both match may the new
full state be treated as recovered progress. This deterministic replay is
not claimed performed by the short throughput panel. The observed trajectory
can change across runtime conditions, so failure to match blocks that claim.

Executable bounded A/B workflow (not a production restart):

```bash
cd /home/minidc/Desktop/PokeAgent3.0-V2-throughput
PY=/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python
GEN=/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/rolling/generations/iter-000000004096-1791958101732995099
PROFILE=/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/release/runtime_profile.json
PYTHONPATH=engine/python:. "$PY" -m agent.ppo.v2_source_migration \
  --checkpoint "$GEN" --parity-report runs/perf/v2_recovery/CPU_PARITY.json \
  --output runs/perf/v2_recovery/migration.json
"$PY" scripts/v2_throughput_ab.py --source "$PWD" --output runs/perf/v2_recovery/new-bounded-arm \
  --checkpoint "$GEN" --profile "$PROFILE" --migration runs/perf/v2_recovery/migration.json
```

The receipt is bound to its exact clean target commit, artifact hashes and
runtime blobs. Regenerate and revalidate after any runtime change. The loader
still enforces model/PPO/dataset/action/collector/rank/world/executor/precision/
optimizer/RNG contracts. It refuses a production-profile launch under this
validation receipt. Never pass the legacy generic allow-source flag for V2.

Rollback measurement is the same runner with
`--source /home/minidc/.codex/worktrees/pa3-v2-runtime/PokeAgent3.0`, a fresh
output directory and no migration argument. All source/original files remain
available; rollback does not mean resetting committed matches silently.

Before a real cutover: resolve long-run recompute failure, recover/choose a
fully resumable boundary, pass DDP and ledger reconciliation plus uninterrupted
versus restored tests, record authorized exact source migration, preserve rollback,
and obtain the requested production-cutover approval. A systemctl restart
command is intentionally not offered as a safe solution while these prerequisites
are unmet.
