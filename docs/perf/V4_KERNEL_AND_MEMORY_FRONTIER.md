# V4 kernel and memory frontier

Compact record of the GPU-kernel and memory findings for the current clean
HEAD. Full numbers and commands: `V4_FULLSTACK_BOTTLENECK_ATLAS.md`.

## Kernel mix (single GPU, 4,096-row minibatch, micro 1,024)

Profiled window 3.23 s, **leaf-kernel GPU busy 3.08 s (95.4 %)**, idle 0.15 s,
peak VRAM 9.30 GiB (`runs/perf/v4/f0_learner_kernels.json`):

| kernel family | device time |
|---|---:|
| `void at::native::` elementwise/copy family (1,761 calls) | 406.7 ms |
| `volta_fp16_s884gemm` (tensor-core GEMM) | 184.3 + 130.7 ms |
| `fmha_cutlassB_f16_aligned_64x64_k64_sm70` (SDPA) | 137.3 ms |
| `cutlass_70_tensorop_f16_s884gemm` | 104.5 ms |
| vectorized elementwise kernels | 88.7 + 68.6 + 59.0 ms |
| `vectorized_gather_kernel` | 60.2 ms |

GPU-busy is taken from trace `cat=kernel` events only;
`key_averages().device_time_total` double counts aggregated ops and their
kernels (8.1 s vs 3.2 s wall) and is never used as a busy metric.

## Memory

| quantity | measured |
|---|---|
| peak VRAM reserved (single, fp16, micro 1024) | 9.25 GiB |
| peak VRAM reserved (single, fp16, micro 2048) | 18.5 GiB (C1) |
| VRAM soft budget per GPU | 28 GiB |
| peak host RSS (single 2k, streaming) | 4.5–4.6 GiB |
| host RAM soft budget | 48 GiB |
| manual flat gradient buffer (8.76 M params, fp32) | 35.0 MB |

The prior 2048-microbatch sweep stays rejected (no improvement, 2× VRAM);
micro 1024 remains the verified default.

## Bounded conclusions

1. The learner update is GPU-bound at ~95 % kernel busy: CPU-side learner
   optimizations have a ~5 % ceiling inside the update.
2. The remaining envelope is collection (23 % of all-in) where the learner GPU
   is idle, plus materialization (17 % of update CPU time) that can overlap.
3. `torch.compile`/CUDA-graph/Triton PoCs (plan P1.4/P7) are not attempted in
   this pass: no profiler evidence yet shows a kernel-count or launch-latency
   bottleneck beneath the 95 % busy window at this microbatch size.
