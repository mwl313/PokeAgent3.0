# V2 exact PPO throughput optimizations — 2026-10-15

Executed clean source: `6a2a6930ed65586ff79186d60187a4a942253550`. Branch: `optimization/pa3-v2-ppo-throughput`. Documentation commits follow execution. User changed the bounded speed test to **1,500 total natural games**, 750/rank; envs250/rank and microbatch1024 retained.

## Matched 1,500-game result

| Metric | Gather control | Exact lookup |
|---|---:|---:|
| Iteration games/s | 43.23565 | 47.32024 |
| Including launcher startup, games/s | 36.56656 | 39.41525 |
| Iteration wall, seconds | 34.69359 | 31.69891 |
| Collection + rank wait, seconds | 13.25726 | 13.38830 |
| Preparation, seconds | 3.88742 | 3.40206 |
| PPO update, seconds | 16.24778 | 13.53995 |

Iteration throughput improved **9.45%**, all-in throughput **7.79%**, update cost fell **16.67%**. These are one sequential run per arm, optimized first; no confidence claim or sustained production-speed claim. Stage totals exclude other iteration overhead and stages use rank maxima. Both produced 33,533 learner rows, two epochs/rank, 18 synchronized optimizer steps, zero errors/skips, and maximum recompute discrepancy 1.90735e-6. Both ranks' rollout, final model, optimizer, row/game and step digests agree across arms exactly (RECEIPT.json).

Two V100 PCIe32GiB at175W, Torch2.14.0+cu126, CUDA nvcc12.8, NUMA pinning, two OMP threads, NCCL P2P disabled, SHM enabled, loopback sockets. Full commands and wrapper hashes are in each SUMMARY.json; runner copies accompany them. Same seed20261014, ModeD, fp16-reduction-off, PPO2, minibatch4096. Frozen1,220-team pool unchanged.

These wrappers load verified **57,344-match policy weights only** and create fresh diagnostic Adam/scheduler clocks. This is not full checkpoint continuation. Private checkpoint prefixes never become due; campaign clocks/ledgers and checkpoints are untouched. The prior500-game measurements were38.515→41.353 games/s (different executed source revisions c34d782/5990235); they are secondary evidence, not the matched final comparison.

## Implemented changes

- Optional exact-order V100 CUDA token-reference gradient reduction. Stable row sorting and a deterministic warp/tail reduction reproduce the tested gather accumulation order without atomics. Normal gather remains default. Plain flat embedding remains excluded from exact claims because earlier FP16 gradients differed.
- CPU-to-CUDA rollout transfers enqueue fields together and synchronize once before returning. Public ownership remains blocking; no pinned-memory budget increase.
- Immutable trusted candidate views avoid per-candidate Python tuples. Indexing/order/equality remain compatible; pickle emits ordinary legacy tuples. Untrusted inputs retain validation.
- Private observation export owns newly serialized arrays and copies borrowed fields once, avoiding duplicate deep copies. No caller-owned cache mutation.
- Per-call CUDA semantic checks share synchronization while retaining category-before-index validation and CSR, namespace, finite, reserved-field and legality checks. No persistent “validated” bypass.

Changed runtime/test files are enumerated by `git diff --name-only e9618c2 6a2a693`; implementation spans rollout buffer, model embeddings/ragged/scorer/exact lookup CUDA and Python, native collector, observation/action/ragged/request types, launcher, benchmark, and three test files.

The custom reducer is opt-in through `--experimental-exact-token-lookup`, restricted to deterministic nonproduction ModeD/fp16-reduction-off diagnostics, at most five iterations. Callers must still bound games/envs explicitly. Checkpoint metadata records the backend and rejects incompatible restore. First-order gradients only; version/device guard requires the tested Torch/CUDA/V100 combination, and nvcc is needed for initial lazy PTX compilation. No compiler invocation on the normal path. Algorithm inspection reference: [PyTorch CUDA indexing source](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/cuda/Indexing.cu); installed-version equivalence was tested, not inferred from upstream main.

## Verification and isolated costs

126 relevant tests passed in18.32s, including full objective, global DDP gradient, real-gradient parity, checkpoint envelope, masked sampling, observation negative cases, duplicate row boundaries, nondefault CUDA streams and mutation isolation. See FINAL_ALL_TESTS.log. This is not a claim that every inherited repository test passes.

Full single-GPU ModeD1024-row captured-data update: original reference535.237ms, final387.871ms (**27.53% lower**). Loss scalars, gradients, updated model and full Adam state match bitwise. Synthetic advantages/returns and fresh Adam are declared; this microbenchmark is not games/s.

Five post-warmup local samples: whole1024-row H2D transfer15.855→13.590ms;576-row export+ownership21.255→16.865ms. All transferred tensors/output hashes match. CPU_FINAL records clean-source representation/loss/gradient/Adam/model digests. Candidate/record/materialization timings are local probes, not independent games/s gains. Collector record_seconds now includes private export previously outside that sub-timer; compare total collection rather than attributing its raw substage change.

A separate profiled500-game run at5990235 sampled one post-warmup optimizer-step boundary/rank. NCCL kernel nonoverlap with compute was1.80ms/rank0 and18.77ms/rank1, versus roughly830ms kernel spans. Host-sync unions342.7/362.6ms overlap GPU execution and cannot be added as removable time. Trace was before final grouped validation; no claimed postchange sync count. Profiling overhead is excluded from throughput. Evidence does not justify changing NCCL configuration or DDP gradient semantics.

## Limits and next decisions

We did not reach or prove a hardware ceiling. Collection remains about13.4s and update13.5s in this short case. Bigger speed claims need controlled repeated sustained measurements; the user skipped the old full3×5 panel. No microbatch change, PPO-objective change, shuffle change, main/dataset/mechanics edit, long training or production restart occurred.

Production deployment remains blocked by the earlier iteration19 actor/recompute discrepancy (rank0 .00115824, rank1 .00170612 versus unchanged .001 tolerance), full73,728-match optimizer/RNG recovery, and an authorized source/backend continuation contract. Exact lookup fixes an experimentally observed backward accumulation difference; it does **not** fix the independent forward batch-shape discrepancy. Keep the experimental backend disabled in production until those blockers are resolved. Additional high-impact work is collection scheduling/export and encoder forward/backward execution, with exactness checks; NCCL tuning is lower priority based on this trace.

Large gradients, optimizer tensors, PTX and profiler traces remain local under runs/perf/v2_all_20261015 and the user cache. Git contains compact evidence only. Original rolling generation remains iteration4,096; full57,344 recovery and73,728 model-only evidence remain preserved separately.
