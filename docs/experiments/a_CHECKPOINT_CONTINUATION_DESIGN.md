# Track A experimental checkpoint continuation design

This is an experiment-only design. It does not change the production launcher
or the Full Spec defaults.

## Boundary and envelope

The only safe fork point is after a complete iteration has committed its
natural matches, finished PPO, passed the rank digest gate, and written a
verified checkpoint. The envelope should contain:

| Component | Required state | Current status |
|---|---|---|
| policy | model tensors, model config, action vocabulary/schema | Restored by `PPOLearner.state_dict()`; config must be hash-checked |
| optimizer | Adam tensors, steps and hyperparameters | Restored by `PPOLearner.state_dict()` |
| scaler | scale, growth tracker and backoff/growth factors | Restored by `PPOLearner.state_dict()` |
| LR clock | absolute committed natural match count and current LR | Restored by scheduler; launcher metadata must equal it |
| PPO branch | epochs 4 or 2, all other PPO fields unchanged | Must be an explicit branch override; never inferred from checkpoint |
| RNG | Python, torch CPU/CUDA, collector-owned and shuffle generators | Experimental helpers in `agent/ppo/experimental_checkpoint.py`; not in production checkpoint yet |
| DDP | world size, rank, local GPU, executor and collective protocol | Communicator/reducer are recreated, not serialized; rank/world-size/config must match |
| collector | engine state, active handles, row cursor, match counter, slot counter, collector RNG/config | In-flight state is not currently serializable; checkpoint only at iteration boundary and restart the next collector from recorded seed/counter |
| opponents | current policy ID/weights and historical policy manifest | Current weights are in policy state; historical pool is currently empty and must be rejected if non-empty but absent |
| data | dataset manifest hash, team count, team file hash, engine/data hash, reference format | Must be recorded and compared before loading |
| accounting | iteration, committed matches, natural games, learner rows, optimizer steps, skipped steps | Launcher records only part of this; continuation must reject mismatches |

## 4-versus-2 continuation protocol

1. Start from one verified iteration-boundary envelope. Hash the model,
   optimizer, scheduler, scaler, dataset and action schema before forking.
2. Clone it into two independent branch directories. Restore learner state,
   all owned RNG streams and the absolute committed-match clock on each rank.
3. Keep the same model, optimizer, scaler, schedule, dataset, collector seed
   family, team pool and opponent manifest. Set only `ppo_epochs=4` or
   `ppo_epochs=2`; validate every other `PPOConfig` field against the frozen
   contract.
4. Derive branch-specific shuffle RNG streams from the recorded fork seed so
   branch divergence is intentional and reproducible. Do not reuse a mutable
   generator object across branches.
5. Recreate DDP with the same world size, rank mapping, executor and collective
   ordering. Compare initial digests before collecting the next iteration.
6. At every subsequent boundary record natural matches, learner rows, actor and
   value rows, optimizer steps/skips, LR, RNG digest and model/optimizer digest.

The current code can restore learner tensors, Adam, scaler and LR clock and can
restore explicitly owned RNG streams with the experimental helper. It cannot
restore an in-flight native battle/engine handle, collector observation row
cursor, historical opponent pool, DDP communicator, or process-level external
state. Therefore `--iterations` is not a complete exact resume mechanism. A
future implementation must either serialize native engine snapshots and the
collector state at an iteration boundary or explicitly restart the collector
from a recorded deterministic seed/counter and label the continuation as
boundary-equivalent rather than bit-exact.
