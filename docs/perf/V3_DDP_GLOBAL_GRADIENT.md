# V3 dual-GPU DDP status (G1)

Branch `optimization/pa3-realpolicy-throughput`. This documents what is
**verified**, what is **implemented but not yet validated end-to-end**, and the
exact remaining blocker with its evidence. No speed claim is made for DDP.

## Verified: the global-denominator objective

`agent/ppo/ddp.py` implements the plan's §6.2 equations:

```
L_rank = world_size * ( sum_actor_local / A  +  value_coef * sum_value_local / V )
A = sum_r A_r (global valid actor rows),  V = sum_r V_r (global valid rows)
```

so that DDP's post-backward gradient averaging reproduces the single-process
global objective. `tests/agent/test_ddp_global_gradient.py` proves it on a
single process with uneven shards (6 vs 3 rows, the second rank holding **zero**
actor rows): the averaged shard gradients match the global reference gradient
(atol 1e-6), the loss helper matches the sharded sums, and a dedicated test
guards against double weighting (a local mean multiplied by `world_size` must
equal the global mean over the shards).

## Verified: the two-process NCCL transport

A minimal 2-process NCCL all-reduce over the two V100s completes in ~1.2 s
(init 0.31 s, all-reduce 0.35 s, destroy 0.50 s) with
`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0` **and
`NCCL_SOCKET_IFNAME=lo`**. Without the explicit interface the transport
selection stalled for many minutes on this host; the launcher sets it for its
own child processes only (no system change).

## Implemented

* `PA3Model.forward(observation, candidates, selected)` — a DDP-visible entry
  that owns the whole learner graph (encoder + value + branch evaluation).
  Calling `encode`/`evaluate` directly on the inner module made DDP's reducer
  mark parameters ready twice, so the learner routes the forward through the
  wrapper when DDP is attached.
* `PPOLearner.attach_ddp` / `update_ddp` with `DDPCommunication` (the `no_sync`
  context wraps the forward **and** backward of every non-final microbatch; only
  the final microbatch all-reduces), one batched all-reduce of the local
  actor/value sums and counts per minibatch, global gradient clipping after
  synchronisation, and rank-aligned minibatch counts (an all-reduce MAX of the
  local minibatch count before the epoch loop).
* `scripts/run_ddp_ppo.py` — two-process launcher, one GPU/NUMA node per rank
  (`numactl --cpunodebind/--membind`), `NCCL_*` and thread env set per child,
  per-rank metrics JSON, rank digest comparison, checkpoint on rank 0.

## Remaining blocker (diagnosed, not yet fixed)

The first dual smoke (32 games/rank, 64 envs, minibatch 256, microbatch 64)
reached the update and failed with an exact PyTorch error:

```
Detected mismatch between collectives on ranks. Rank 0 is running
all_reduce(TensorShape=[4]) [the global actor/value count reduction] but Rank 1
is running all_reduce(TensorShape=[1853171]) [a DDP gradient reduction].
```

Root cause: when a rank's local row count is exhausted, its aligned
padding-only minibatch has no micro with valid rows, so `update_ddp` skips the
backward and therefore the DDP gradient reduction for that minibatch while the
other rank still performs it. Ranks have different local row counts, so the
collective sequence diverges; NCCL then spins (both ranks 100% CPU/GPU, no
progress), which is exactly the failure mode the plan's §16.4 flags
(`DDP deadlock/gradient mismatch` → do not report as a speed result).

Next step (one of the following, all of which keep the objective exact):

1. truncate the update to the minimum local row count agreed by an all-reduce,
   keeping the leftover rows in the buffer for the next iteration (loses a small
   fraction of rows, no double counting), or
2. keep the padded minibatches but replace the skipped step with an agreed
   protocol: all ranks run the same backward (a graph-connected zero loss when
   a rank has no real rows) and all ranks skip the optimizer step together when
   the all-reduced real-row count for that minibatch is zero — this avoids
   Adam momentum drift from rank-asymmetric steps, or
3. pad every rank's plan to the same row count with masked rows placed so each
   rank always has real rows in every minibatch (requires a per-row validity
   mask in the plan).

After the fix: the plan's §6.3 sequence (synthetic uneven parities already
passing, then real PA3-8M one-step optimizer parity between single-GPU global
4096 and 2x2048 DDP, then the 2k x3 and 10k+ dual all-in benchmarks with
checkpoint/resume). Until then the dual-GPU **all-in** number is unmeasured and
the DDP path is not used by any default configuration.
