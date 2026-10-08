# V2 learner breakdown (M0 of the v2.0 plan)

> **Correction (v3, 2026-10-09):** the "2,112 per-microbatch GPU syncs per 2k
> iteration" figure in this document was a scale mix-up. The measured counts are
> 448 micro calls / device syncs for the 2,048-match iteration (28 minibatches ×
> 16 micros) and 2,176 for the 10,240-match iteration (136 × 16). The
> optimizer-step counts (28 / 136) in the raw JSON confirm this; the old figure
> is not reused as evidence. See `V3_NUMERIC_PARITY.md` §4.

Measured on the miniDC with the optimized actor path (fixed observation batch,
packed candidate wire, packed rollout rows), GPU0, 1,024 environments, 16
workers, fp16 autocast + `inference_mode`, PA3-8M, frozen 1,137-team pool.
Instrument: `scripts/bench_pa3_end_to_end.py --mode full` plus the opt-in
learner stage timers (`PPOLearner.profile`, `prepare_profile`,
`RolloutBuffer.profile`) and an optional `torch.profiler` CUDA trace.

## 2,048-match iteration (27,490 rows, 28 optimizer steps)

Wall: collect 21.5 s, PPO 58.3 s, checkpoint 0.49 s, all-in 80.3 s
(25.5 committed games/s). Stage split of the 58.3 s learner phase:

| stage | seconds | share of learner |
|---|---:|---:|
| backward (GPU) | 18.17 | 31% |
| forward: encoder + scorer (GPU) | 15.42 | 26% |
| minibatch selection / tensor slicing (CPU) | 5.40 | 9% |
| GAE (2,749 side trajectories) | 1.26 | 2% |
| observation materialization (`stacked()` + `from_compact_numpy`) | 1.63 | 3% |
| candidate table build (`from_rows`) | 0.58 | 1% |
| H2D (minibatch → GPU) | 2.53 | 4% |
| optimizer + scaler + clipping | 1.00 | 2% |
| metric accumulation (GPU scalars) | 0.10 | 0.2% |
| epoch metric D2H | <0.01 | <0.1% |
| unaccounted Python/loss-graph overhead | ~12.2 | 21% |

`prepare_batch` is only 3.53 s (6%): GAE 1.26, observation stack 1.63,
candidates 0.58, columns 0.06, advantage normalization ~0. The 290.7 s learner
phase of the 10,240-match run is therefore ~90% inside `update()`, split
roughly evenly between GPU forward/backward and CPU-side selection/Python
overhead.

## CUDA kernel evidence (`runs/perf/traces/v2_p0_1k.json`)

Top CUDA kernels by device time during the learner update of a 1,024-match
iteration (1,375 optimizer... see the raw report for exact counts):

| kernel | device time |
|---|---:|
| `sum_and_scatter` (backward scatter for the embedding) | 1.45 s |
| `fmha_cutlassB_f16_aligned_64x64_k64_sm70` (SDPA math) | 1.25 s |
| `Memcpy HtoD (Pageable -> Device)` | 1.00 s |
| `vectorized_elementwise_kernel` (activations/normalization) | 0.91 s |
| `cutlass_70_tensorop_f16_s884gemm_relu` (FFN GEMM) | 0.86 s |

The embedding backward scatter and the pageable H2D copies are the two
clearly addressable kernels; SDPA and the FFN GEMMs are the actual model work.

## Per-microbatch device synchronization removal

`int(micro.row_valid.sum())` inside the micro loop forced one device sync per
microbatch (2,112 per 2,048-match iteration). Replaced with a CPU-side padding
filter plus one minibatch H2D and device-side micro slicing:

| variant | PPO wall s | all-in games/s | note |
|---|---:|---:|---|
| original (per-micro GPU sync + minibatch H2D) | 58.3 | 25.5 | baseline |
| per-micro H2D | 59.9 | 25.0 | rejected (2,112 small transfers) |
| CPU filter + single H2D + device slice | 58.8 | 22.8 (different collect) | **kept** |

The kept variant is throughput-neutral within run-to-run variation; it is kept
because it removes 2,112 device synchronizations per iteration (a latency-tail
and correctness-of-accounting improvement), not as a speed claim. Per the plan,
a neutral change is documented rather than sold as an optimization.

## Reading and next steps (P1/P2 priority)

1. **Streaming minibatch materialization (P1.3).** `minibatch_select` (5.4 s)
   plus the unaccounted Python cost and the 1.63 s observation stack all come
   from materializing the whole iteration and then slicing it. Selecting rows
   from the compact store per minibatch removes most of that and the 5.2 GiB
   per-2k-match host footprint.
2. **Candidate table dtype/size (P1.3).** `from_rows` builds `[B,4,64,6] int64`
   (12 KB/row) although the wire is `u8`. Keeping the packed bytes and casting
   once per minibatch cuts H2D bytes and the 1.0 s of pageable memcpy.
3. **Microbatch size (P2.3).** 2,112 forward calls of 256 rows each; the global
   4,096 contract is preserved by accumulation, and the new
   `exact_row_weighted_accumulation` makes a different micro split
   mathematically equivalent, so 512/1024 are now safe A/B candidates.
4. GPU utilization during the learner phase measured at 69–80% SM with
   150–175 W on GPU0 (GPU1 idle), so the learner is genuinely GPU-fed; the
   remaining CPU share is Python dispatch, not GPU starvation.

## P2.3 microbatch size (measured, adopted as a launcher option)

The global minibatch stays 4,096 and the optimizer-step count is unchanged
(28 steps for the 2,048-match iteration) for every split; the exact
row-weighted accumulation (`exact_row_weighted_accumulation`) makes a different
micro split mathematically equivalent, which the new regression tests prove.
Measured on the same 2,048-match workload, one repeat each:

| microbatch | PPO wall s | all-in games/s | backward s | forward s | GPU reserved |
|---:|---:|---:|---:|---:|---:|
| 128 | 79.3 | 20.46 | 33.2 | 34.0 | 2.84 GiB |
| 256 (spec default) | 60.0 | 24.66 | 19.1 | 29.0 | 2.86 GiB |
| 512 | 48.2 | 29.14 | 10.3 | 24.3 | 4.88 GiB |
| **1024** | **42.9** | **31.57** | 5.3 | 21.7 | 9.25 GiB |

Policy statistics are identical across splits (ratio 1.0004, clip ≈0.046,
entropy 0.9987, value loss 0.2759), which is the equivalence evidence the plan
requires. Three-repeat confirmation at 1024: actor median 102.19 games/s,
**all-in median 32.80 games/s** (min 31.68), PPO 41.8–42.8 s, checkpoint
0.54–0.57 s inside the window, 28 optimizer steps, 0 skipped, KL 0.0044–0.0050,
recompute within the fp16 gate, GPU reserved 9.27 GiB (soft budget 28 GiB).

Adoption note: `PPOConfig.microbatch_size` still defaults to the pinned 256.
The 1024 setting is a measured launcher option
(`bench_pa3_end_to_end.py --microbatch 1024`); the two-rank DDP path
(`256 × 8 × 2`) must re-validate its own split before any change there.

## P1.3 streaming minibatch materialization (opt-in, memory win)

`RolloutBuffer.to_batch` can now materialize only the selected rows
(`InlineObservationStore.stacked_indices`), and
`PPOLearner.prepare_streaming` / `update_streaming` keep the iteration as
compact rows plus one FP32 advantage vector, expanding one global minibatch at
a time. The objective, the micro split, the padded final minibatch and every
reported statistic are identical to `update()` (same helpers).

Measured, 2,048-match iteration with microbatch 1024:

| variant | PPO wall s | all-in games/s | peak RSS | GPU reserved |
|---|---:|---:|---:|---:|
| whole-iteration materialization | 42.9 | 31.57 | 5.13 GiB | 9.25 GiB |
| streaming minibatch | 43.3 | 30.24 | **3.59 GiB** | 9.25 GiB |

Speed is neutral (within repeat noise); the win is memory, and the 10,240-match
confirmation shows the intended slope: collect 100.2 s, PPO 206.0 s,
checkpoint 0.63 s, **all-in 33.37 games/s, peak RSS 7.34 GiB** (versus 18.0 GiB
for the whole-iteration path at the same scale). 136,259 rows, 136 optimizer
steps, 0 skipped, KL 0.0051 (epochs 0.0070/0.0046/0.0041/0.0047), ratio 1.0,
recompute 6.8e-5 within the fp16 gate, 0 operational errors.

The streaming path stays opt-in (`--streaming-minibatch`) until the two-rank
path is validated; it is the recommended single-process setting because it
removes the whole-iteration expansion that pushed the old run to 18–25 GiB.
