# A2-R clean-HEAD validation receipt

Date: 2026-10-11. Branch: `research/a-ppo-efficiency`. Executed clean commit: `f8d9a00bd6a2d83cffda10023ceb374a5d16bc6d` (`git_dirty=false`).

The bounded gate used two Tesla V100 ranks, seed `20261009`, 4 games/rank, 4 static environments/rank, one native worker/rank, global minibatch 8, microbatch 4, PPO epochs 2, `--no-recompute`, and `--batch-cache none`.

```text
PYTHONPATH=engine/python:. /home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/run_ddp_ppo.py --games 4 --envs 4 --workers 1 --minibatch 8 \
  --microbatch 4 --epochs 2 --iterations 2 --no-recompute \
  --batch-cache none --report runs/experiments/a/a2r_final_20261011/uninterrupted.json
```

The resumed arm used the same command with `--iterations 1` and `--experimental-checkpoint`, then a fresh process with `--experimental-resume` and `--iterations 1`. Environment: Python 3.12.3, PyTorch 2.14.0+cu126, CUDA 12.6, two V100s, NUMA binding enabled, static collector slots, current opponent only, and the repository dataset manifest.

Uninterrupted and resumed second boundaries both produced 8 games, 207 rows, 16 committed matches, 55 second-iteration optimizer steps, equal model tensors, equal learner reports, equal rank digests within each process, and zero operational errors. The owned shuffle RNG and checkpoint contracts were restored. The optimizer digest differed after the fresh CUDA process restart; the measured difference is sub-ULP (`1.3878e-17` in the Adam second moment), so this remains `BOUNDARY_EQUIVALENT`, not `BIT_EXACT`.

Both rank envelopes were independently inspected. They agree on source SHA, iteration 1, committed-match clock 8, current-only opponent manifest, the collector contract hash, and generation. Each contains the complete observed drain proof: `complete=true`, `in_flight_battles=0`, emitted by `NativeCollector.collect` only after all static handles reported termination. The launcher now fails closed if that runtime proof is absent.

This receipt does not authorize the 4-versus-2 PPO experiment or long RL training.
