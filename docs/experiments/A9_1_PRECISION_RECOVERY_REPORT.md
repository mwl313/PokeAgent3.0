# Track A9.1 — FP16 precision recovery

This bounded experiment starts from the A9 implementation at `b1733cc` and
uses the trusted A5 Control rank envelopes.  No battle mechanics, model
architecture, PPO objective, Track C files, or A5 artifacts were changed.
The A9 FP32 behavior-policy path remains the default.

## Contracts and implementation

Four explicit, reversible contracts are available through
`run_ddp_ppo.py --precision-mode`:

| Mode | Observation contract | Encoder | Scorer / GRU | Status |
| --- | --- | --- | --- | --- |
| A | lossless FP32 | FP32 | FP32 | A9 reference/default |
| B | lossless FP32 | CUDA AMP FP16 | CUDA AMP FP16 | experimental |
| C | deterministic FP16-rounded values, retained losslessly after rounding | CUDA AMP FP16 | CUDA AMP FP16 | experimental |
| D | deterministic FP16-rounded values, retained losslessly after rounding | CUDA AMP FP16 | FP32 | experimental fallback |

Mode C quantizes only continuous policy inputs immediately after native decode.
The exact rounded values are used by the actor and rollout store. Categorical
IDs, masks, candidate tables, selected prefixes, and engine state are never
quantized. Probability, ratio, KL, entropy, PPO loss, optimizer, and master
weights remain FP32. The versioned identity is `pa3-precision-{A|B|C|D}-v1`.
New experimental envelopes record the selected mode and reject an accidental
mode mismatch on restore. Legacy A5/A9 envelopes are accepted only through the
existing explicit source/collector migration path.

## Numerical results

All four arms used the recompute gate at `1e-3`, PPO-2, LR scale 0.5, global
minibatch 4096, microbatch 1024, 16 native workers, 1024 environments per
rank, two V100s, and the same A5 checkpoint. Each one-iteration arm completed
2,048 natural games. A and C were repeated for two complete iterations (4,096
games) to check stability.

The one-iteration trained-checkpoint results were identical across modes for
the sampled/recomputed gate because the behavior and learner used the same
declared input contract. Per-rank maxima were rank 0 `1.4305e-5` and rank 1
`1.6391e-5`; all gates passed. In the two-iteration repeats, A maxima were
`1.4305e-5`, `9.4175e-6` and `1.6391e-5`, `1.0729e-5` by rank; C maxima were
`1.4305e-5`, `1.0729e-5` and `1.6391e-5`, `8.3447e-6`. There were zero
operational errors, zero illegal actions, finite gradients, equal DDP model
digests, and 18 optimizer steps per iteration.

On one real A5 observation, C's FP16 input rounding had maximum feature error
`9.7632e-5`, encoder-token divergence `1.3313e-3`, and value divergence
`2.3738e-5` versus A. This is a changed policy-input contract, not a
recompute inconsistency. A full action-disagreement panel was not run in this
bounded throughput screen, so C is not claimed policy-equivalent to A.

## Throughput

| Mode | All-in games/s | Speedup vs A | Recompute max error | Peak GPU reserved | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| A, one iteration | 37.2288 | 1.000x | 1.6391e-5 | 14.992 GiB | pass |
| B, one iteration | 37.9185 | 1.019x | 1.6391e-5 | 14.992 GiB | pass, marginal |
| C, one iteration | 37.2292 | 1.000x | 1.6391e-5 | 14.992 GiB | pass, no gain |
| D, one iteration | 37.9187 | 1.019x | 1.6391e-5 | 14.992 GiB | pass, marginal |
| A, two-iteration repeat | 39.7596 | — | max 1.6391e-5 | 14.992 GiB | stable |
| C, two-iteration repeat | 39.3776 | 0.990x vs A repeat | max 1.6391e-5 | 14.992 GiB | stable, slower |

The A repeat's steady-state rate was 42.8683 games/s and C's was 41.9106
games/s. Collector, recompute, PPO update, digest, and checkpoint stages were
included. Rank-0 A update time was 24.33 s and collection 10.77 s; C was
24.49 s and 11.48 s respectively. The added C conversion increased
observation/H2D/record work and erased any AMP benefit at this workload.

The historical A5 result of approximately 93.2 games/s remains context only;
it used a different numerical contract and disabled the failing gate, so it is
not an A9.1 comparison.

## Decision and limits

No mode is promoted as the new production default. B and D show only a 1.9%
single-run difference, within the observed startup and workload noise, and
were not repeated enough to justify a promotion. C is numerically valid but
did not recover throughput and changes policy inputs measurably. A remains the
correct default and the reference for future comparisons.

The A9.1 target of 90+ all-in games/s was not recovered. The measured result
does not justify A10 or a claim of Tensor Core recovery. A future approved
experiment should first repeat B and D with the same two-iteration protocol
and run a fixed action-disagreement/evaluation panel before considering either
as an opt-in default.
