# PA3-8M + PPO learner scaffolding — status (agent/pa3-ppo)

**Everything below is mock-tested scaffolding.** No real training has started,
no real engine rollout has been used as a training datum, and the Rust engine
was not modified. The Rust engine in this tree is still incomplete and is *not*
training-ready (`engine/TRAINING_READINESS.md`); the learner is written against
a typed contract so the eventual `NativeEngine` binds to it without redesign.

## What exists

| Component | File | Content |
|---|---|---|
| Typed contract | `agent/types/actions.py` | Request kinds (`0..4`), action kinds (`pick/move/switch/pass`), packed 6-field atomic action, branch-slot rules per request type, singleton detection |
| Typed contract | `agent/types/observation.py` | `ObservationBatch`: 96 padded token rows (88 active) with 32 category features, 50 continuous features, 40 flags and separate known masks; native payload adapter; compact per-row storage |
| Typed contract | `agent/types/requests.py` | `RequestRow` (observation + branch candidate tree) and `BranchCandidatesBatch`: `[B, 4, 64]` padded candidate tables with masks, entity/move token references and the preserved selected prefix |
| Model | `agent/model/pa3_model.py` | PA3-8M: one encoder pass per request, branch scorer loop, value head, `evaluate` (recompute) and `sample` (categorical, temperature 1.0) |
| Model | `agent/model/encoder.py` | 6-layer pre-LayerNorm **non-causal** Transformer, `d_model` 320, 5 heads × 64, FFN 1280, GELU, dropout 0, SDPA-math attention with an explicit matmul/softmax fallback |
| Model | `agent/model/scorer.py` | Conditional prefix scorer: 320-unit `GRUCell` over within-request choices only (no match-level recurrence), pointer logits from the referenced Pokémon/move tokens plus the six structured action fields; masked log-softmax with exact zero probability for illegal candidates |
| Model | `agent/model/config.py` | Frozen architecture constants (random init, embedding std 0.02, Xavier linears, policy output gain 0.01, seed 20261006) |
| PPO | `agent/ppo/learner.py` | Adam (3e-4 peak, betas 0.9/0.999, eps 1e-5, wd 0), 4 epochs, global minibatch 4096 from 256-row microbatches with gradient accumulation, clip 0.2, value `0.5*MSE` (no clipping), grad-norm 0.5, FP32 probability/loss math, optional FP16 autocast + dynamic GradScaler, target approx-KL 0.03 early stop for the iteration |
| PPO | `agent/ppo/losses.py` | Policy/ratio objective, value loss, approx-KL estimator, advantage normalization over iteration actor rows (std floor 1e-8) |
| PPO | `agent/ppo/gae.py` | GAE γ=1.0, λ=0.95 along one side's request sequence; terminal-only reward; optional bootstrap for non-terminal ends |
| PPO | `agent/ppo/schedule.py` | LR schedule clocked by **committed natural training matches**: 1e-5 → 3e-4 warmup over 250k, cosine to 3e-5 at 100M |
| Buffer | `agent/buffer/rollout_buffer.py` | Compact row schema (observation or reference, request kind, candidates + masks, selected prefix, old logprob, value, reward, done, side, policy id, opponent policy id, team ids, match id, turn/request index, optional seed ref); overshoot rows kept; padded/masked final minibatch; both sides of current self-play; historical-opponent rows excluded from current-policy rows; no encoder activations stored |
| Mock | `agent/mock_engine/mock_engine.py` | `MockNativeEngine` mirroring the future `NativeEngine` surface (`team_count/reset_batch/request_info_batch/candidates_batch/observe_payload_batch/step_batch`) plus `MockRolloutCollector` producing fake rollouts with masks, singleton requests and terminal ±1/0 rewards |

## Branch policy semantics

* teampreview → four picks (lead A, lead B, reserve 1, reserve 2)
* normal → slot A, then slot B conditioned on A
* replacement → only the slots that require replacement
* singleton/pass → probability 1 for the only legal completion

For a selected request the log-probability is the **sum of the selected branch
log-probabilities**, the selected prefix is preserved for recomputation, masks
are applied before sampling and log-probability math, and entropy/uniform-KL are
computed per valid branch, normalized by `log K` (K ≥ 2) and averaged over the
request's valid branches. A request whose every branch is a singleton is
excluded from the actor loss and from advantage normalization while still
learning a value.

Uniform regularization uses `KL(U || π)/log K = (-log K − mean log π)/log K`,
matching `training.uniform_kl_direction: KL_uniform_to_policy`.

## Tests

`python -m pytest tests/agent` (48 tests, all passing on CPU):

| Test file | Covers |
|---|---|
| `test_contract.py` | request/action vocabulary, branch slots per request type, singleton detection, layout indices, native payload adapter, compact round trip, junk-float flushing |
| `test_model_shapes.py` | 8–10M parameter band (measured 8.76M), architecture constants, forward shapes, **one encoder call per request regardless of branch count**, value head ignores the action prefix, FP32 probability dtypes |
| `test_masked_sampling.py` | illegal action probability exactly 0, 200-sample mask respect, singleton probability 1 with logprob 0, deterministic selection restricted to legal candidates |
| `test_forward_backward.py` | gradients reach every parameter, finite gradients for masked/padded branches |
| `test_logprob_recompute.py` | recomputation equals the sampled log-probability under frozen weights, survives a `state_dict` round trip, candidate order/mask sensitivity |
| `test_entropy_kl.py` | normalized entropy == 1 and KL == 0 for a uniform policy, singleton branches contribute 0, sharpening lowers entropy and raises KL, `H/log K ≤ 1` |
| `test_singleton_exclusion.py` | actor-mask exclusion of all-singleton rows, advantage normalization over actor rows only, value loss still covers those rows, reported actor rows |
| `test_gae.py` | terminal-only reward, λ decay, γ=λ=1 Monte-Carlo equality, single payment per match, bootstrap, per-(match, side) grouping |
| `test_rollout_buffer.py` | overshoot retention, padded/masked final minibatch, both sides of current self-play, historical rows excluded, no encoder activations stored, candidate tensor shapes |
| `test_ppo_update.py` | ratio ≈ 1 and KL ≈ 0 under identical weights, loss definitions, gradient accumulation == full-minibatch gradient, mock rollout → PPO update, target-KL early stop, frozen optimizer hyperparameters |
| `test_schedule_and_checkpoint.py` | warmup/cosine LR values and match clock, learner `state_dict` round trip restoring identical outputs, reproducible update from identical weights |

## Local reproduction

```bash
python -m pytest tests/agent
```

The tests run on CPU. This work used an agent-local `.venv` (gitignored) with
the pinned `torch==2.14.0+cu126` from the project's pinned PyTorch index plus
numpy and pytest; no host package, driver, CUDA toolkit, service or power
setting was changed.

## Still blocked on final NativeEngine readiness

1. **Collection is not wired to the engine.** `MockRolloutCollector` is the only
   producer; the real collector needs `reset_batch` / `request_info_batch` /
   `candidates_batch` / `observe_encoded_batch` / `step_batch` from the compiled
   PyO3 module (`engine/python/pa3_engine`), including the packed observation
   bytes that `ObservationBatch.from_native_payload` already accepts.
2. **No real engine rollout may be used yet.** The readiness gate
   (`engine/examples/readiness_check.rs`) still fails on move/ability/item
   coverage, dynamic closure and full-scope battles, so any rollout would be an
   operational error rather than a natural match.
3. **Observation tensor export.** The Python-visible observation still needs the
   engine's packed-form adapter wired to the typed contract, including the
   player-visible target-location encoding and the remaining effect metadata.
4. **Model-side preprocessing.** Species/move/item/ability categorical
   vocabularies are placeholders (`category_vocab = 8192`) until the compiled
   Dex vocabularies are exported; the checkpoint must record the final
   vocabulary.
5. **Training harness.** Episode scheduling, 2,048-environment actors, opponent
   pool snapshots, checkpoints, metrics and evaluation are not implemented here;
   this task is deliberately limited to the model, PPO machinery, buffer and
   mock-tested tests.
6. **DDP path.** The learner runs single-process; the documented 2×V100
   `no_sync` accumulation path (per-rank minibatch 2,048, microbatch 256) is not
   exercised.
