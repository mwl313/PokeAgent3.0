# Requested priorities 2 → 3 → 4: bounded optimization results

User scope: skip the previous package's production-runtime A/B validation;
do not change/experiment with microbatch; start token-gather backward,
rollout preparation and remaining collector CPU overhead, in that order.
No long training, production restart, main modification or native mechanics change.

Implemented on `optimization/pa3-v2-ppo-throughput`. Benchmark execution sources:
ModeD experiments `1d89863` (row backend before gating); final gated code and
FP32/CPU tests `1b4be6f`. Reports follow these sources; do not substitute a later
documentation commit into checkpoint source provenance.

## 2. Token lookup backward

`ConditionalPrefixScorer.token_references` now has an experimental flat-row
embedding lookup. It removes the `[batch,candidates,320]` expanded gather index
from deterministic backward while preserving reference order, null masks,
validation and separate entity/move contributions. No parameter names/shapes,
PPO terms, DDP weighting or action legality change.

The final default is still the original gather. The new backend is explicitly
enabled only by `model.scorer.experimental_row_lookup=True` in the diagnostic
script/tests. It is not a serialized model parameter or launcher option, and
must not be used for checkpoint continuation without a new backend/runtime
contract. Production and current checkpoint source checks are not bypassed.

Two independent GPU processes, same real captured model/observations/candidates,
1024 rows, ModeD, deterministic CUDA, CUBLAS_WORKSPACE_CONFIG=:4096:8,
FP16 reduced-precision reduction disabled. Declared synthetic advantages/returns;
fresh diagnostic Adam, not the campaign optimizer. First-step proof, two warmups,
then five CUDA-event-timed complete forward/loss/backward/clip/Adam steps:

| Scope | Original gather median | Flat-row median | Cost reduction |
|---|---:|---:|---:|
| ModeD full model step | 533.320 ms | 381.744 ms | 28.42% |
| FP32 full model step | 924.793 ms | 771.592 ms | 16.57% |

This is a **microbenchmark**, not a games/s or full DDP training result. Timing
runs are sequential, five samples each; no statistical/long-run claim. The
microbatch remains1024; the prior package A/B was not rerun.

Every compared loss scalar is exactly equal. Full-model FP32 gradients differ
by at most1.863e-8, relative L2 1.642e-7; first Adam-updated weights differ by
at most1.490e-8. Thus the mathematical lookup is equivalent within tight FP32
roundoff, but not bitwise identical on the full GPU graph.

**ModeD limitation:** gradient max delta3.052e-5, relative L2 1.571e-4;
updated model max delta3.707e-7. Gradients fail the diagnostic per-element
rtol1e-4/atol1e-6 comparison. The changed accumulation order is amplified by
FP16 encoder backward rounding. Small isolated lookup/null/repeated-index
tests pass, but those do not override the real full-graph difference. No gate
was widened; the backend is **EXPERIMENTAL / DEFAULT OFF**, not promoted for
production or bit-exact checkpoint replay. Adam moment values also differ.

## 3. Ragged reconstruction / preparation

`RaggedBatch.from_rows` no longer copies every row immediately before
concatenating it into another owned allocation. It checks row shapes, concatenates
once, then checks all resulting values for finiteness in one vectorized pass.
Counts and output values remain independently owned, including single/empty
rows. All five families, known masks, ordering and validation remain intact.

Captured576 observations, five post-warmup samples per family:

| Family | Original median | New median |
|---|---:|---:|
| Effects | 1.801 ms | 0.514 ms |
| Repertoire | 1.657 ms | 0.507 ms |
| Types | 1.625 ms | 0.514 ms |
| Base moves | 2.419 ms | 0.615 ms |
| Move effects | 2.223 ms | 0.677 ms |

Full captured576-row materialization median was50.012ms in the new CPU probe.
The earlier package probe was62.985ms, but these are different sessions and
this is **not a controlled end-to-end improvement claim**. H2D was measured
once by the harness, not separately optimized or claimed accelerated.

An independent saved-input CPU probe with the normal gather backend matches
the earlier baseline bitwise in observation/candidate/row representations,
every loss term, full gradient digest, Adam digest and updated model digest.
These safe data-path changes are enabled in this isolated branch.

## 4. Collector CPU conversions

`NativeCollector._prefix_actions` converts the wire slice in one NumPy tolist
call, then decodes signed target bytes without six scalar conversions per
action. It preserves -1/-2 targets and immutable Python engine action tuples.
Pick readback retains its owned NumPy tensor view instead of copying int64
and rebuilding a Python list; scalar lists from tensor.tolist are no longer
converted to Python float a second time. No inference grouping, branch prefix,
RNG, shuffle, action selection or engine rule change.

10,000 two-branch prefix conversions, five post-warmup samples:
**46.727 → 10.379ms**, about4.50x locally. This small function's ratio is not
the collector's or complete PPO's speedup. No new self-play games were run;
live collector throughput and complete trajectory parity remain unmeasured.
Signed-target/source-ownership tests pass; the Rust engine is unchanged.

## Validation and deployment boundary

- 31 focused tests pass in3.75s, covering experimental lookup output/input and
 parameter gradients on CPU/CUDA, null/repeated/noncontiguous cases, ownership,
 action contract and existing global-DDP/real-gradient tests. One PyTorch
 first-use CUDA-context warning; no test failure.
- 33 additional observation, sequential sampling, legality/masking and full
 PPO objective regression tests pass in10.65s.
- All five ragged families have identical values/count hashes across independent
 benchmark processes. Default-path captured loss/gradient/Adam proof is bitwise.
- The diagnostic full FP16 gradient comparison fails, and is retained as evidence.
 No optimizer state reset, checkpoint publication, match-counter increment or
 production training occurred. Large diagnostic tensors remain local.
- No new two-rank GPU or full-stack games/s comparison; the user skipped the
 prior validation panel. No microbatch experiment or change.

Source migration to this changed scorer is intentionally blocked by the prior
bounded data-path whitelist. No permissive receipt was generated. Existing
campaign checkpoints, recovered57,344 boundary and original73,728 model-only
evidence remain preserved. The earlier long-run recompute failure is unresolved.

Next focused action is to address the flat lookup's mixed-precision gradient
delta before enabling it for PPO continuation. Preparation and collector
changes have exact captured CPU parity, but a live end-to-end speed claim
requires a later authorized benchmark. No jobs are scheduled automatically.

## Artifacts and reproduction

Compact evidence: `evidence/` in this directory. Full local gradient/model/
Adam tensors: `/home/minidc/Desktop/PokeAgent3.0-V2-throughput/runs/perf/v2_lookup_20261015/`.
JSON summaries record exact source, runtime-file hashes, environment, raw timing
samples and synthetic-target scope. `COMPARISON.json` records numerical deltas.

```bash
cd /home/minidc/Desktop/PokeAgent3.0-V2-throughput
PY=/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python
ART=/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/reports/continuation.json.rank0.iteration19.recompute.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=engine/python:. \
 "$PY" scripts/benchmark_v2_lookup_recovery.py --artifact "$ART" --output /tmp/new-lookup-probe
# Add --reference for exact2a3bdcd scorer/ragged/prefix methods;
# add --fp32 for the diagnostic FP32 comparison. Use fresh output paths.
```
