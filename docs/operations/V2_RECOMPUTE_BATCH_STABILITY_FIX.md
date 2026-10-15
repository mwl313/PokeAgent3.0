# V2 recompute batch-composition stability fix

## Scope and safety status

This change is an execution-precision repair for future Mode-D collection and
learner evaluation.  It does not change model parameters, model architecture,
PPO loss terms, action masks, sampling, the `0.001` gate, the secondary audit,
or DDP ordering.  The primary and secondary recompute checks remain fail
closed.

The historical failed rollout is deliberately **not** accepted by this change.
Its stored behavior log probabilities were produced by the old FP16 operation;
re-evaluating them through the corrected operation changes some values by more
than `0.001`.  The failed rollout must remain rejected and must not be used for
an update.  A future rollout samples and recomputes with the same corrected
contract.

## Root cause

The preserved iteration-25 rank-1 envelope was replayed with deterministic
algorithms and FP16 reduced-precision reduction disabled. The decisive row is
the documented normal two-branch request at absolute row 26,896:

| Path | Selected-action log probability |
|---|---:|
| Original actor group (1,320 rows) | -3.0076684952 |
| Sampled recompute (576 rows, padded to 1,024) | -3.0106306076 |
| Difference | 0.0029621124 |

Both paths use 1,024 physical encoder rows, but their ragged feature record
counts differ.  The first target-row divergence is the FP16 linear operation
`RaggedEmbedding.numeric["effects"]`, before token embedding LayerNorm,
attention, Q/K/V, SDPA, or the scorer:

| Target tensor delta: actor-group versus sampled batch | Max absolute |
|---|---:|
| `effects` numeric projection | 0.000244140625 |
| Token embedding | 0.0012737513 |
| Final encoder tokens | 0.0020189881 |

The projection's GEMM is selected from its packed ragged-record dimension, not
only the nominal encoder batch dimension.  Therefore canonicalizing encoder
rows to 1,024 did not canonicalize this GEMM's shape.  There is no intentional
cross-observation attention or sample leakage: the difference is an FP16
cuBLAS kernel/rounding result that is amplified by LayerNorm and later encoder
blocks.

## Fix

Mode D now runs only `RaggedEmbedding.numeric["effects"]` in FP32.  All other
ragged projections and the dense Transformer encoder remain under existing
FP16 autocast; scorer and GRU remain FP32.  This operation has no checkpoint
parameters and does not alter a model state dict.  The profile requires
`ragged_effects_fp32: true`, and the reviewed V2 source-migration receipt must
explicitly record the `fp16 -> fp32` transition before a historical checkpoint
is resumed under this source.

The full FP32 encoder (`precision mode A`) remains the conservative fallback.

## Preserved-artifact verification

`scripts/v2_recompute_forensics.py` replayed both saved rank-1 and rank-0
iteration-25 artifacts.  With the selective FP32 projection, the decisive rows
produced bitwise-identical encoder tokens across actor-group and sampled batch
composition.  All six secondary-audit outliers passed actor-versus-actual
learner recomputation with a maximum difference of `4.7683716e-7`:

| Rank | Historical actor/sample maximum tested | Corrected actor/sample maximum tested |
|---|---:|---:|
| 0 | 0.0024974346 | 0.0000004768 |
| 1 | 0.0029621124 | 0.0000002384 |

The historical stored values are not replaced or relaxed.  Their largest
corrected-path difference in this six-row set is `0.0016052723`, which is why
the envelope still fails the unchanged gate.

## Performance evidence and limits

The forensic replay measured three independent CUDA-event repetitions on the
rank-1 V100 using the preserved 1,024-row physical chunk.  This is a narrow
encoder microbenchmark, not an end-to-end PPO claim:

| Median encoder latency | Historical FP16 effects | Selective FP32 effects |
|---|---:|---:|
| Actor-group chunk | 87.409 ms | 86.948 ms |
| Sampled/padded chunk | 86.839 ms | 87.143 ms |

The change is within the observed timing noise (about -0.5% and +0.35%,
respectively).  No dual-GPU sustained collection or PPO throughput promotion
is claimed here.  Existing V2 candidate-packing, owned-observation, CUDA-cache
and compact-candidate work remains unchanged; it must be benchmarked again
with this corrected numerical contract before any production recommendation.

## Commands

```bash
PYTHONPATH=engine/python:. /home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_recompute_forensics.py \
  --artifact /path/to/continuation.json.rank1.iteration25.recompute.pt \
  --rank 1 --device cuda:0 --output /tmp/rank1-forensics.json
```

Focused validation after the change:

```text
74 passed: recompute, forward/backward, gradient/DDP, precision, canonical
encoder, migration, operational-contract, and secondary-audit tests.
```
