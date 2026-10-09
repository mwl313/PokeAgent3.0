# V4 full-stack bottleneck atlas (F0)

Measured on minidc, branch `optimization/pa3-realpolicy-throughput`, after the
C0/C1/D0/D1 fixes (`83a7383` + the P5 instrumentation commit). All numbers are
productions of the commands recorded next to them; nothing is extrapolated.

## 1. All-in budget (single GPU, corrected contract, 2,048 games)

`scripts/bench_pa3_end_to_end.py --mode full --games 2048 --envs 1024
--workers 16 --observations fixed --candidate-wire packed --precision fp16
--inference-mode --device cuda:0 --microbatch 1024 --streaming-minibatch
--repeats 3` → `runs/perf/v4/a0_single_corrected_2k.json`

| stage | median | share of all-in |
|---|---:|---:|
| collection | 23.2 s | 23.0 % |
| PPO update (incl. GAE + 4 epochs + 56 steps) | 78.2 s | 77.4 % |
| checkpoint write | 0.51 s | 0.5 % |
| all-in wall | 100.8–106.6 s | — |
| wall-clock rate | 20.11 games/s (median) | — |

Rows: 55,039 (26.87/match); optimizer steps 56; 0 operational errors; peak
RSS 4.51 GiB; peak VRAM reserved 9.25 GiB; recompute gate PASS (max diff
1.1e-4 ≤ 1e-3 fp16 tolerance).

## 2. Collection stage shares (single GPU, median of 3)

| stage | seconds | share of collection |
|---|---:|---:|
| buffer record (Python row construction) | 8.18 | 35.4 % |
| actor model inference (GPU) | 5.20 | 22.4 % |
| native observation crossing | 3.07 | 13.3 % |
| H2D | 2.03 | 8.7 % |
| unaccounted | 2.79 | 12.1 % |
| parse/convert | 0.89 | 3.8 % |
| Rust `step_batch` | 0.36 | 1.6 % |
| candidates + candidate table | 0.44 | 1.9 % |
| readback / request / reset / prefix / assemble | 0.14 | 0.6 % |

The Rust engine core is 1.6 % of collection; the cost is Python object
construction (record), GPU actor inference and the observation crossing.

## 3. Learner stage shares (single GPU, CPU timers)

| stage | median | share of PPO |
|---|---:|---:|
| forward (encode + score + value, enqueue) | 42.8 s | 54.7 % |
| minibatch materialization (`to_batch`) | 13.3 s | 17.0 % |
| optimizer block (unscale/clip/scaler/Adam) | 9.1 s | 11.7 % |
| backward (enqueue) | 7.9 s | 10.0 % |
| H2D per-microbatch | 3.0 s | 3.8 % |
| GAE / advantage prep | 2.7 s | 3.4 % |
| metrics | 0.04 s | 0.1 % |

Streaming materialization split (probe, 4,096-row minibatch):
`materialize_observations` 62 %, `materialize_candidates` 31 %, columns 4 %.

**Async-attribution caveat:** the forward/backward/optimizer split above is
CPU enqueue time, not GPU active time. The profiler pass (§5) shows the window
is 95 % GPU-busy, so the CPU timers mostly measure where the queue drains.

## 4. Dual-GPU path (1,024 games/rank, 2,048 total, 3 repeats)

Per-rank medians: collection 12.9 s, update 58.7 s, rows 27,330/27,279,
56 optimizer steps, 0 skipped, 56 synchronized + 392 no-sync micro steps.

| executor | games/s (median of 3) | update wall | flat all-reduce |
|---|---:|---:|---:|
| DDP (reducer overlap) | 25.60 (25.60–25.92) | 58.5–58.9 s | — |
| manual FP32 SUM | 25.28 (24.38–25.60) | 59.5–62.4 s | 10 ms / 35.0 MB per step |

NCCL is not the bottleneck: one flat 35 MB all-reduce costs ~10 ms inside a
~1 s update step. DDP stays the default (marginally faster, simpler, reducer
overlap), the manual executor remains available and parity-verified.

Equal-total comparison: single GPU 20.11 games/s vs dual DDP 25.60 games/s
(+27.3 % at 2,048 total natural games; same both-seat contract, same rows,
same 56 steps).

## 5. Kernel-level profile (single GPU, 4,096-row minibatch, micro 1,024)

`scripts/v4_f0_profile.py` → `runs/perf/v4/f0_learner_kernels.json`
(warmup, then one profiled 2-minibatch pass).

* profiled window 3.23 s; **leaf-kernel GPU busy 3.08 s = 95.4 %**; idle 0.15 s;
  peak VRAM 9.30 GiB.
* top kernels: `void at::native::` elementwise/copy family 406.7 ms
  (1,761 calls), `volta_fp16_s884gemm` 184.3 + 130.7 ms,
  `fmha_cutlassB_f16_aligned_64x64_k64_sm70` (SDPA) 137.3 ms,
  `cutlass_70_tensorop_f16_s884gemm` 104.5 ms, vectorized elementwise
  88.7 + 68.6 + 59.0 ms, `vectorized_gather_kernel` 60.2 ms.
* `key_averages().device_time_total` double counts aggregated ops and their
  kernels (8.1 s vs 3.2 s wall); only trace `cat=kernel` events are used for
  busy time.

Conclusion: the learner update is **GPU-bound with ~5 % idle**. CPU-side
learner optimizations have an Amdahl ceiling of ~5 % of the update (≈3.8 % of
all-in). The largest remaining measured envelope is the **collection phase**
(23 % of all-in) during which the learner GPU is idle, plus the materialization
block (17 % of update CPU time, overlappable).

## 6. Optimization experiments in this pass (recorded, not promoted)

| change | measured effect | verdict |
|---|---|---|
| zero-copy `narrow` for micro slices / `iter_microbatches` | within repeat noise (20.11 → 20.30 → 20.21 games/s across intermediate states, ranges overlap) | **neutral**, kept (strictly fewer copies, all tests green) |
| one permuted copy per epoch instead of per minibatch | within noise | reverted |
| preallocated observation fill instead of `np.concatenate` | 0.35 s vs 0.34 s per 2 minibatches | reverted |
| `Adam(foreach/fused)` micro-benchmark | plain 1.95 ms, foreach 1.69 ms, fused 0.73 ms per step vs a 19 ms scaler/clip block | fused is ~1 ms/step (≈0.06 % of all-in at 56 steps); not worth a parity-risk change |

Declared noise floor for this panel: ±2 % on games/s (min/max spread of the
3 repeats at identical configuration).

## 7. Priority list (Amdahl-bounded, measured shares)

1. **Collection/learner overlap** — collection is 23 % of all-in with an idle
   learner GPU; a bounded overlap (rolling slots P3 or independent cohorts P4)
   has a ceiling of ~23 % of all-in. Needs exact collection-contract tests
   (rolling-slot equivalence) before promotion.
2. **Materialization (P5.1 columnar slabs)** — 13.3 s / 101 s = 13.2 % of
   all-in of CPU time, overlappable with GPU work; its own ceiling as a speedup
   is the idle window (≈5 %), but it feeds §6.1 overlap directly.
3. **Buffer record (P5/P2)** — 8.2 s / 101 s = 8.1 % of all-in, CPU-side
   Python row construction during collection (same overlap argument).
4. **Actor model kernel/inference (P1/P2)** — 5.2 s / 101 s = 5.2 %;
CUDA-graph/graph-private-pool work would need a separate PoC.
5. **Optimizer block** — 9.1 s CPU-timer but ~19 ms of real per-step work
   (measured); the timer is drain time, not removable compute.

## 8. Telemetry boundary

* `nvidia-smi` inventory: GPU0 175 W / GPU1 150 W, no throttling observed
  during the runs; no driver, NCCL, BIOS, power-cap or service change was made.
* Disk read, swap and major faults were not on the critical path (host RAM peak
  4.6 GiB against the 48 GiB soft budget; VRAM peak 9.3 GiB against 28 GiB).
* Traces are written under `runs/perf/v4/` (git-ignored); only compact kernel
  summaries are committed in this document.
