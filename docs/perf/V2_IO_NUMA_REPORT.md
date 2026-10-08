# V2 storage / memory / NUMA / PCIe report (P6 of the v2.0 plan)

Read-only diagnostics; no package, driver, power, service or kernel setting was
changed. All numbers were taken on the miniDC during a real-policy bounded PPO
run unless labelled "idle".

## Storage: not a bottleneck (`not observed`)

| observation | value | source |
|---|---|---|
| Learning-process real disk reads (`read_bytes`) | **0** during the run | `/proc/<pid>/io` |
| Learning-process writes (`write_bytes`) | 12 KB (checkpoint path only) | `/proc/<pid>/io` |
| `rchar` (source/config reads) | 55 MB, page-cache served | `/proc/<pid>/io` |
| Swap present / used | 2.0 GiB / **0 B** | `/proc/swaps`, `free -h` |
| `vmstat` si/so during the run | 0 / 0 | `vmstat 1 3` |
| Major page faults | 0/s (28k minor/s from the Python heap) | `pidstat -ru` |
| iowait during the run | 0% | `vmstat` |
| Root filesystem free | 50 GB of 218 GB (77% used) | `df -h` |

Conclusion: simulations, observations and rollouts live in RAM and move over
PCIe to the GPU; the SATA device only serves cold source reads and the
checkpoint write. **An NVMe upgrade is not justified by any measured training
bottleneck**, per the plan's rule that storage only becomes a priority when
swap, major faults, spill or fsync latency are observed.

## GPU during the learner phase (`nvidia-smi dmon`)

| device | SM util | memory-controller | VRAM | power | temperature |
|---|---:|---:|---:|---:|---:|
| GPU0 (learner) | 69–80% | 11–44% | 3.3 GiB | 150–175 W (at the 175 W cap) | 47–48 °C |
| GPU1 (idle) | 0% | 0% | 4 MiB | 25 W | 38–40 °C |

The learner genuinely saturates GPU0's SMs during forward/backward; the earlier
"GPU utilization 0" readings were idle snapshots taken before the run, and the
plan's warning about confusing them with training utilization applies.

## Host memory and NUMA

| metric | value |
|---|---:|
| Learner process RSS (2k match iteration) | 5.23 GiB |
| Anonymous/private pages | 3.21 GiB |
| Minor faults | 28,036/s |
| Major faults | 0/s |
| Swap in use | 0 |
| 10,240-match iteration peak RSS (v1 report) | 18.0 GiB (24 GiB budget respected) |

NUMA reading from `numastat -p`: the process pages were spread across both
nodes (Node0 238 MB, Node1 2,972 MB) because this benchmark was launched without
`numactl`. The production contract still requires each rank to bind to its
GPU-local node before allocating its engine groups; the new
`scripts/run_actor_pair_real.py` does this with
`numactl --cpunodebind=N --membind=N`, and the dual-rank run measured 94.15
(NUMA0/GPU0) and 100.26 (NUMA1/GPU1) games/s with no swap and 2.4 GiB RSS per
rank.

## PCIe / H2D-D2H

* Pageable H2D copies are visible in the profiler as
  `Memcpy HtoD (Pageable -> Device)` at 1.00 s for a 1,024-match learner
  update — the third-largest CUDA-side entry, which makes pinned staging a
  justified P2 experiment (bounded, ≤1 GiB total, only if the overlap is
  demonstrated on the profiler timeline).
* GPU1 at 150 W is not the slower rank in the dual-actor measurement, so no
  power or clock change is warranted; the difference sits inside run variance.

## Explicitly unchanged

Driver 580.178.04, CUDA toolkit 12.8.2, torch 2.14.0+cu126, GPU power limits
175/150 W, `dsh-web`/`llama-swap` services, and the `/usr/local/cuda` symlink
were all left untouched. No BIOS, governor, hugepage or kernel tuning was
performed.
