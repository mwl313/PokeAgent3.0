# Safe continuation and rollback — production cutover NOT APPROVED

## Verified partial recovery and today's stop

The user requested wrapping up around50,000 test games. At request arrival
53,248 total-clock matches were already complete and the next iteration was
in flight. A cooperative STOP completed that iteration, saved both ranks and
exited: **57,344 total-clock matches**,13 replay iterations/53,248 replay games
after restoring4,096,718 cumulative optimizer steps. No more jobs are queued.

Full checkpoint (manifest plus both ranks):

`/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699`

Additional independently hash-verified LOCAL backup:

`/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/backups/throughput-stop-57344-20261014`

The backup's manifest references28 original-ledger immutable receipt files;
those are preserved and validated, not embedded in the rank files. Retain
the campaign team_stats directory as well. This is a local-disk duplicate,
not an off-machine backup. Model weights/checkpoints are not uploaded to Git.

Source is original5ccc51b; model and full Adam hashes agree with historical
telemetry at every completed boundary. Scheduler/scaler/RNG and restart state
are present, both ranks show complete drain with0 in-flight battles,28 team
receipts reconcile to57,344, and all three checkpoint file hashes verify.
See evidence/CLOSEOUT_RECEIPT.json. The original73,728 model-only evidence
remains preserved but its full optimizer/RNG state has NOT been recovered.
No recent historical RNG payload exists for direct comparison; owned streams
were reconstructed by replay. Do not claim universal bit-exact restoration.

Recovery appended unique immutable uncommitted team-stat receipts under the
original run identity. It did not replace previous receipts or publish into
original campaign rolling/LATEST, which remains4,096. Selecting the recovered
generation for continuation is a separate future action, not today's cutover.

The original workspace is `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It was missing
from the first exploratory A/B; the launcher now pins and checks it. The
corrected candidate A/B and continuation were cancelled before launch. Source
migration receipts remain bounded-only and cannot authorize production.
Retain executedb3c189b/8a743ad identities; the final documentation SHA is not
a substitute. To reproduce an old receipt, use a clean checkout of its exact
target SHA and the same native binary/data, not the newer report checkout.

No executable production restart is authorized. Next-session requirements:
corrected-workspace candidate A/B/continuation, long-run recompute repair,
approved full-state/source/ledger plan and explicit cutover approval. There
is no background continuation scheduled after this closeout.

## Original preservation and bounded-validation reference

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
authorized by this cutover guide. The original committed ledger and evaluation
archive remain preserved; recovery only appends the immutable receipts above.

For exact future recovery, replay from the verified4,096 generation with the
original source/runtime and owned RNG, checkpoint every successful iteration
in an isolated recovery namespace, and compare the recovered73,728 model and
optimizer digests against saved telemetry. Only if both match may the new
full state be treated as recovered progress. Today's replay stopped at57,344
and has not recovered the full73,728 boundary. The observed trajectory
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
