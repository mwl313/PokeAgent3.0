# AHL V2 — CPU topology, affinity and safe coexistence

Read-only inspection performed 2026-10-17 KST on `miniDC`. No production
launcher, service unit or CPU setting was changed.

## 1. Verified host topology

`lscpu` / `numactl --hardware` / sysfs:

| Property | Observed |
|---|---|
| Logical CPUs | 80 (0–79), online 0–79 |
| Physical cores | 40 (2 sockets × 20) |
| SMT | 2 threads/core; CPU `n` pairs with `n+40` (0–39 primary, 40–79 sibling) |
| NUMA nodes | 2 |
| node0 | 0–19 + 40–59, 32,035 MB |
| node1 | 20–39 + 60–79, 32,236 MB |
| cgroup cpuset | `0-79` (no partitioning) |
| GPUs | 2× Tesla V100-PCIE-32GB, PCI 05:00.0 and 84:00.0 |

Examples: `cpu0 → siblings 0,40`; `cpu4 → 4,44`; `cpu19 → 19,59`;
`cpu20 → 20,60`; `cpu39 → 39,79`.

## 2. Actual effective affinity (not the comments)

### PPO launcher `scripts/run_ddp_ppo.py`

```python
RANKS = [
    {"rank": 0, "numa": 0, "cpus": "0-15", "gpu": 0},
    {"rank": 1, "numa": 1, "cpus": "20-35", "gpu": 1},
]
```

The `cpus` values are **not consumed anywhere**. The parent only runs:

```text
numactl --cpunodebind=<numa> --membind=<numa> -- python ... --worker
```

Therefore the effective affinity is:

- rank0: **all node0 CPUs (0–19 and 40–59)**, memory bound to node0;
- rank1: **all node1 CPUs (20–39 and 60–79)**, memory bound to node1;
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` for the children;
- the documented `0-15` / `20-35` ranges are **not** proven CPU isolation.

`--cpunodebind` binds to the whole NUMA node, including SMT siblings. Logical
CPUs 16–19 and 36–39 are **not** free under live PPO, and neither are 40–79.
Any claim of “non-overlapping masks” relative to PPO is false unless the
launcher is changed.

### Formal V2 evaluation

- `systemd-user/v2-eval-worker.service` and `v2-eval-discover.service`:
  `CPUAffinity=4 5 6 7`, `Nice=15`, `MemoryMax=4 GiB`.
- Live worker PID 297090 confirms `Cpus_allowed_list: 4-7`, `Mems_allowed: 0-1`,
  ~698 MB RSS, 5 threads, **idle** at inspection time.
- CPUs 4–7 are primary threads of cores 4–7; their siblings are 44–47.

### Old `scripts/rr_pool.sh`

The old script pinned `N` workers to CPUs `0..(2N-1)`. With `N=16` that is
**0–31**, which:

- overlaps formal evaluation CPUs **4–7** directly;
- occupies part of node0 and part of node1 under PPO's node-wide binding;
- assumes the first 32 CPUs are free, which is not true.

This was corrected without touching PPO: affinity is now derived from sysfs by
`agent/evaluation/historical_resources.py` / `scripts/ahl_v2_resource_gate.py`.

## 3. Corrected diagnostic allocation

Reserved by construction: formal CPUs **4,5,6,7** and their SMT siblings
**44,45,46,47** (and the whole co-located cores).

| Mode | Condition | Workers | Threads | CPUs (topology-derived) |
|---|---|---:|---:|---|
| A | PPO running | 2 | 2 | first two usable cores per worker, e.g. 0–1 and 2–3 |
| A reduced | sustained degradation 3–5% | 1 | 1 | first usable core |
| A paused | sustained degradation > 5% | 0 | — | — |
| B | PPO stopped | up to 16 (configurable) | 2 | 0–3, 8–19, 20–39 (one primary thread per core) |

The gate also pauses on: formal backlog, free disk < 12 GiB, available memory
< 8 GiB, swap used > 2 GiB, or any CPU-mask conflict. Mask conflicts fail safe
with zero workers.

The current `resource_gate.json` (PPO stopped) selected 16 workers with masks
`0-1, 2-3, 8-9, 10-11, …, 34-35`; reserved CPUs 4–7 and 44–47 are absent and
there are no conflicts.

## 4. Contention map

| Resource | PPO (live) | Formal eval | Historical diagnostics |
|---|---|---|---|
| node0 CPUs | all 0–19,40–59 | 4–7 | Mode A: 0–1/2–3; Mode B: many |
| node1 CPUs | all 20–39,60–79 | — | Mode B only |
| memory | node0/node1 membind | ≤ 4 GiB cgroup | gate ≥ 8 GiB available |
| disk | checkpoint generations | receipts | 12 GiB reserve |
| GPU | 2× V100 | none | none (`CUDA_VISIBLE_DEVICES=`) |

**No true physical-core isolation from PPO is possible with the current
`numactl --cpunodebind` launcher.** Changing PPO to explicit per-rank cpusets is
a separate reviewed production migration, not part of this patch.

## 5. Throughput gate (hypothesis, not validated)

- Baseline retained: **381.9 learner rows/sec** (do not overwrite).
- Needs at least 3 completed PPO iterations; uses the median of the last 3.
- ≤ 3% sustained degradation: accept.
- 3–5%: reduce historical concurrency.
- > 5%: pause historical evaluation.

These thresholds are operational hypotheses. **Live-PPO coexistence is
UNVERIFIED** because production training is stopped; no benchmark was invented.

## 6. Proposal (not applied)

If strict isolation is later required, a separate reviewed change should:

1. allocate explicit cpusets per rank, e.g. rank0 primaries 0–15 + selected
   siblings, rank1 primaries 20–35 + selected siblings;
2. leave CPUs 4–7 and their siblings outside both PPO ranks for formal eval;
3. verify NVIDIA/PCIe locality (GPU 05:00.0 with node0, 84:00.0 with node1);
4. run an A/B throughput comparison and record a new source/launch receipt.

Until then, historical diagnostics must treat PPO nodes as shared and rely on
low priority, CPU-only execution, low thread counts and the resource gate.
