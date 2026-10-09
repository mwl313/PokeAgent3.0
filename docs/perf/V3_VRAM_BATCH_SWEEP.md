# V3 VRAM / microbatch sweep (G2)

Context: the v3 collection contract records both current-policy seats, so each
2,048-match iteration now yields ~55k learner rows (26.9 rows/match) instead of
~27k. Every number below is measured inside that contract, GPU0, 1,024
environments, 16 workers, fp16 + `inference_mode`, packed wire, streaming
minibatch, real checkpoint write inside the measured window, 3 repeats of
2,048 natural matches.

| microbatch | all-in games/s median (min) | PPO wall s | GPU reserved | host RSS | optimizer steps | KL | ratio | recompute gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | (v2 basis, single-seat) 29.14 | 48.2 | 4.88 GiB | 5.1 GiB | 28 | 0.00443 | 1.0004 | pass |
| 1024 | **19.37** (19.03) | 82.2 | 9.25 GiB | 4.58 GiB | 56 | 0.00470 | 1.0002 | pass |
| 2048 | 19.32 (18.09) | 83.4 | **17.98 GiB** | 4.51 GiB | 56 | 0.00467 | 1.0002 | pass |
| 4096 | not run | — | — | — | — | — | — | — |

Reading:

* **Microbatch 2048 is rejected.** It doubles the reserved VRAM (9.25 → 17.98
  GiB) and adds ~1.2 s of PPO wall for a statistically identical all-in rate
  (19.32 vs 19.37, well inside the repeat spread). The plan's rule — do not keep
  a setting that only consumes more memory — applies directly.
* 4096 was not attempted because 2048 already showed no compute benefit while
  approaching the 28 GiB soft budget with fragmentation headroom shrinking; the
  plan permits skipping it in that case.
* All configurations preserve the global minibatch 4,096 (the optimizer-step
  count is 56 for the both-sides contract, i.e. one step per 4,096-row
  minibatch over 4 epochs) and the exact row-weighted objective; the parity
  tests in `V3_NUMERIC_PARITY.md` are the equivalence evidence.
* The GPU reserved figure for 1024 stays at 9.25 GiB in the both-sides contract
  because the micro shape, not the row count, drives the activation peak.

Remaining VRAM headroom: at microbatch 1024 the learner uses ~9.3 GiB of the
28 GiB per-card budget, leaving room for the planned DDP gradient buckets and
double-buffered staging; the 2048 trial shows where that headroom goes if the
batch is enlarged without a compute reason.
