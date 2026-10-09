# V5 — dual microbatch 1024 A/B (P0 W1)

Branch `optimization/pa3-realpolicy-throughput`, code base `b6da590` plus this
task's changes (launcher recompute gate, promoted default, collector telemetry).
Measured 2026-10-09 on minidc, both V100s idle before the panels, power caps
unchanged (GPU0 175 W / GPU1 150 W), no driver/service changes.

## 1. Exact protocol (identical for both arms except `--microbatch`)

```
# arm 256 (default at the time)
PYTHONPATH=engine/python:. .venv/bin/python scripts/run_ddp_ppo.py \
  --games 1024 --envs 1024 --workers 16 --microbatch 256 --minibatch 4096 \
  --timeout 1800 --checkpoint runs/perf/v5/a256_run<N>_ckpt.pt \
  --report runs/perf/v5/a256_run<N>.json

# arm 1024
... --microbatch 1024 ... --checkpoint runs/perf/v5/a1024_run<N>_ckpt.pt \
  --report runs/perf/v5/a1024_run<N>.json

# optional 512 probe
... --microbatch 512 ...
```

Each run: 2,048 games total (1,024/rank), both-seat contract, fixed
observations, packed candidate wire, fp16 autocast + GradScaler, inference
mode, DDP fixed-step protocol, epoch sums all-reduced, checkpoint write by rank 0
inside the all-in window, stratified sampled-vs-recomputed logprob gate before
the update, per-rank nvidia-smi sampling every 2 s
(`runs/perf/v5/telemetry_*.csv`, kept on the miniDC disk).

## 2. Results

| arm | games/s runs | median | mean | min–max spread | wall |
|---|---|---:|---:|---:|---:|
| 256 (3 runs) | 25.920 / 26.252 / 26.252 | 26.252 | 26.141 | 1.3 % | 78–79 s |
| **1024 (3 runs)** | 32.502 / 31.994 / 32.502 | **32.502** | 32.333 | 1.6 % | 63–64 s |
| 512 (1 run, probe) | 30.112 | — | — | — | 68 s |

Median improvement 256 → 1024: **+23.8 %** (mean +23.7 %), far beyond the ±2 %
noise floor used by F1; intra-arm spread is 1.3–1.6 %.

Stage detail (per rank, identical on both ranks within each run):

| arm | collect | update | no-sync micro steps | sync steps | VRAM peak reserved | RSS peak |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 12.0–12.5 s | 57.2–57.9 s | 392 | 56 | 2.88 GiB | 3.25 GiB |
| 1024 | 12.3–12.6 s | **42.1–42.3 s** | 56 | 56 | 9.06 GiB | 3.26 GiB |
| 512 | 12.3 s | 47.3 s | 168 | 56 | 4.77 GiB | 3.27 GiB |

The update wall drop (−27 %) reproduces the single-GPU sweep's 256→1024
behaviour (−28 %) and is monotonic at 512.

## 3. Gates (all runs, both arms) — PASS

| gate | 256 arm | 1024 arm |
|---|---|---|
| model + optimizer digests equal across ranks | True | True |
| rows per rank (27,330 ±1 %) | 27,330 / 27,279 | 27,330 / 27,279 |
| optimizer steps / skipped | 56 / 0 | 56 / 0 |
| epochs | 4 / 4 | 4 / 4 |
| operational errors | 0 | 0 |
| recompute gate (max abs diff vs tol 1e-3) | 5.78e-06 / 1.55e-05 | 5.78e-06 / 1.55e-05 |
| LR (rank-equal) | 1.238e-05 | 1.238e-05 |
| checkpoint wall inside window (rank 0) | 0.42–0.44 s | 0.43–0.53 s |

Epoch KL is rank-identical within each arm; it differs slightly *between* arms
(e.g. run 3: 256 → 0.00437/0.00443/0.00419/0.00423 vs 1024 →
0.00439/0.00445/0.00416/0.00432) because the micro partition changes the
accumulation order — expected for this A/B, not a gate failure.

nvidia-smi telemetry peaks: 256 → util max 78–94 %, power ≤186 W, temp ≤62 °C;
1024 → util max 100 %, power ≤222 W (transient instantaneous reading while the
175 W *cap* is unchanged), temp ≤69 °C; 512 → util max 100 %, temp ≤70 °C.
No throttle-related intervention was made.

## 4. Decision — PROMOTED

The 1024 arm improves by +23.8 % median with all gates green and 1.6 % spread,
so per the task rule the dual default is promoted:

* `scripts/run_ddp_ppo.py --microbatch` default is now **1024** (with a trailing
  note that `PA3_TRAINING_CONFIG.yaml`'s `microbatch_per_rank: 256` should be
  updated at the next spec revision);
* confirmation run with the promoted default (no `--microbatch` flag):
  `runs/perf/v5/confirm_default1024.json` → **32.501 games/s**, same gates PASS.

Reference: the F1 panel (earlier session, no checkpoint/recompute gate, F0
thermal state) measured 24.67 (DDP) / 25.60 (manual). Today's within-session
arms are used for the decision, as required.

The roadmap (`docs/PokeAgent3_Optimization_Roadmap_2026-10-09.md`) listed this
A/B as its item #1 lever; it is now completed and superseded by this result.
