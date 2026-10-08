"""G0: real PA3-8M gradient parity for gradient accumulation.

The v2 tests proved the accumulation math on a stub network. These tests use
the real PA3-8M architecture: a single-pass forward/backward over the whole
global minibatch is compared against the accumulated microbatches, per
parameter group (encoder / scorer / value head). The property is independent of
the absolute row count, so the test uses a small but architecturally identical
batch; the plan's 4,096-row instance is the same computation scaled up.
"""

from __future__ import annotations

import torch

from agent.buffer.rollout_buffer import RolloutBuffer
from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import PA3Config, build_model
from agent.ppo import PPOLearner, PPOConfig


def _learner(model, minibatch, microbatch):
    config = PPOConfig(
        global_minibatch_size=minibatch,
        microbatch_size=microbatch,
        ppo_epochs=1,
        per_rank_minibatch_size=minibatch,
        grad_accumulation_per_rank=max(1, minibatch // microbatch),
        sample_weighted_ddp_reduction=False,
        # Compare raw gradients; `clip_grad_norm_` scales grads in place and is
        # verified separately, so it must not distort the parity comparison.
        max_grad_norm=1.0e9,
    )
    return PPOLearner(model, config, device="cpu", amp=False)


def _gradient_norms(model):
    groups = {"encoder": [], "scorer": [], "value": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("encoder") or name.startswith("embedding"):
            groups["encoder"].append(parameter.grad.detach().clone())
        elif name.startswith("scorer"):
            groups["scorer"].append(parameter.grad.detach().clone())
        elif name.startswith("value_head"):
            groups["value"].append(parameter.grad.detach().clone())
    return {
        group: torch.cat([value.flatten() for value in tensors]) if tensors else torch.zeros(1)
        for group, tensors in groups.items()
    }


def _run(model, buffer, minibatch, microbatch, seed):
    learner = _learner(model, minibatch=minibatch, microbatch=microbatch)
    batch = learner.prepare_batch(buffer)
    learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(seed))
    return learner


def _accumulated_loss(model, buffer, minibatch, microbatch, seed):
    """Sum of the scaled per-microbatches losses of one minibatch/epoch.

    With exact row weighting this must equal the single-pass loss of the same
    rows, because each micro contributes its own share of the minibatch's
    valid actor/value rows.
    """
    learner = _learner(model, minibatch=minibatch, microbatch=microbatch)
    batch = learner.prepare_batch(buffer)
    generator = torch.Generator().manual_seed(seed)
    total = torch.zeros((), dtype=torch.float32)
    exact = learner.config.exact_row_weighted_accumulation
    for minibatch_batch in batch.iter_minibatches(
        minibatch, shuffle=True, generator=generator, drop_last=learner.config.drop_last_minibatch
    ):
        actor_total = float((minibatch_batch.actor_mask & minibatch_batch.row_valid).sum().item())
        valid_total = float(minibatch_batch.row_valid.sum().item())
        for begin in range(0, len(minibatch_batch), microbatch):
            end = min(begin + microbatch, len(minibatch_batch))
            if not bool(minibatch_batch.row_valid[begin:end].any().item()):
                continue
            micro = minibatch_batch.select(torch.arange(begin, end, dtype=torch.long))
            terms = learner._forward_terms(micro)
            if exact:
                actor_scale = float(terms["actor_count"]) / max(actor_total, 1.0)
                value_scale = float(terms["value_count"]) / max(valid_total, 1.0)
                total = total + (
                    terms["loss_unscaled"] * actor_scale
                    + learner.config.value_coefficient * terms["value_mean"] * value_scale
                ).detach()
            else:
                scale = micro.sample_weight / max(1, (len(minibatch_batch) + microbatch - 1) // microbatch)
                total = total + (
                    terms["loss_unscaled"] + learner.config.value_coefficient * terms["value_mean"]
                ).detach() * scale
    return float(total.item())


def _reference_loss(model, buffer, minibatch, microbatch, seed):
    learner = _learner(model, minibatch=minibatch, microbatch=microbatch)
    batch = learner.prepare_batch(buffer)
    generator = torch.Generator().manual_seed(seed)
    total = torch.zeros((), dtype=torch.float32)
    for minibatch_batch in batch.iter_minibatches(
        minibatch, shuffle=True, generator=generator, drop_last=learner.config.drop_last_minibatch
    ):
        terms = learner._forward_terms(minibatch_batch)
        total = total + (
            terms["loss_unscaled"] + learner.config.value_coefficient * terms["value_mean"]
        ).detach()
    return float(total.item())


def _buffer(rows_hint=16):
    config = PA3Config(encoder_layers=2, d_model=64, attention_heads=2, head_dim=32,
                       ffn_dim=128, category_vocab=256, category_embed_dim=8,
                       prefix_hidden=64, critic_hidden=64)
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    reference_model = build_model(config)
    buffer = collect_mock_rollout(engine, reference_model, envs=3, target_matches=3)
    return config, buffer


def _align_behavior_policy(model, buffer, rows=None):
    """Set old_logprob to the current weights' logprob (ratio 1 reference).

    The plan's parity gate is declared for identical weights/observations; the
    mock rollout otherwise carries far-off behavior log-probabilities whose
    gradients cancel heavily and amplify FP32 round-off (recorded separately in
    V3_NUMERIC_PARITY.md).
    """
    rows = list(buffer.rows if rows is None else rows)
    batch = buffer.to_batch(rows, device="cpu")
    with torch.no_grad():
        encoded = model.encode(batch.observation)
        evaluation = model.evaluate_encoded(
            encoded, batch.candidates, selected=batch.candidates.selected
        )
        logprob = evaluation.request_logprob.float().tolist()
    for row, value in zip(rows, logprob):
        row.old_logprob = float(value)


def test_real_model_accumulated_gradient_matches_single_pass():
    config, buffer = _buffer()
    total_rows = len(buffer.rows)
    assert total_rows >= 8, total_rows

    reference_model = build_model(config)
    # state_dict() returns references; clone so the reference update cannot
    # mutate the weights the accumulated runs load.
    reference_state = {key: value.detach().clone() for key, value in reference_model.state_dict().items()}
    _align_behavior_policy(reference_model, buffer)
    # 1. Loss parity: the accumulated scaled losses must equal the single-pass
    #    loss of the same rows; this isolates the weighting math from backward
    #    summation-order round-off.
    single_loss = _reference_loss(reference_model, buffer, total_rows, total_rows, seed=101)
    _run(reference_model, buffer, minibatch=total_rows, microbatch=total_rows, seed=101)
    reference = _gradient_norms(reference_model)

    for microbatch in (max(1, total_rows // 4), max(1, total_rows // 2)):
        model = build_model(config)
        model.load_state_dict(reference_state)
        accumulated_loss = _accumulated_loss(model, buffer, total_rows, microbatch, seed=101)
        assert abs(accumulated_loss - single_loss) <= 1e-5 * max(abs(single_loss), 1.0), (
            microbatch, accumulated_loss, single_loss,
        )
        _run(model, buffer, minibatch=total_rows, microbatch=microbatch, seed=101)
        candidate = _gradient_norms(model)
        for group in ("encoder", "scorer", "value"):
            reference_gradient = reference[group]
            scale = float(reference_gradient.abs().max())
            difference = float((reference_gradient - candidate[group]).abs().max())
            norm_relative = abs(float(reference_gradient.norm()) - float(candidate[group].norm())) / max(
                float(reference_gradient.norm()), 1e-8
            )
            # FP32 summation order over microbatches differs from the single
            # pass; the declared bound is 1e-3 of the per-tensor gradient scale
            # plus an absolute floor, and the global norm must match tightly.
            assert difference <= 1e-4 * scale + 1e-7, (
                microbatch, group, difference, scale, norm_relative,
            )
            assert norm_relative < 1e-4, (microbatch, group, norm_relative)


def test_real_model_handles_singleton_only_and_padded_microbatches():
    config, buffer = _buffer()
    total_rows = len(buffer.rows)
    reference_model = build_model(config)
    _align_behavior_policy(reference_model, buffer)
    # Force every actor row to a singleton (no actor signal) for the middle
    # third of the rows: those microbatches must contribute no actor gradient
    # and must not produce NaN or a spurious optimizer step.
    for index, row in enumerate(buffer.rows):
        if total_rows // 3 <= index < 2 * total_rows // 3:
            row.actor_active = False

    state = {key: value.detach().clone() for key, value in reference_model.state_dict().items()}
    _run(reference_model, buffer, minibatch=total_rows, microbatch=total_rows, seed=7)
    reference = _gradient_norms(reference_model)

    model = build_model(config)
    model.load_state_dict(state)
    learner = _run(model, buffer, minibatch=total_rows, microbatch=max(1, total_rows // 5), seed=7)
    candidate = _gradient_norms(model)
    for group in ("encoder", "scorer", "value"):
        scale = float(reference[group].abs().max())
        difference = float((reference[group] - candidate[group]).abs().max())
        assert difference <= 1e-4 * scale + 1e-7, (group, difference, scale)
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
    assert learner.optimizer_steps >= 1
