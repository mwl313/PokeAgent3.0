"""C1: single-device PPO objective oracle on the real PA3 architecture.

The reference implementation below recomputes the full PPO objective from the
model's raw outputs (log-probabilities, entropy, uniform-KL, values) without
calling any of the learner's loss/aggregation helpers. It pins the sign,
coefficient and mask conventions of ``PPOLearner._forward_terms`` and of the
Adam/checkpoint continuation behaviour that the distributed work will be
compared against.
"""

from __future__ import annotations

import copy
import math

import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import PA3Config, build_model
from agent.ppo import PPOLearner, PPOConfig


def _config(microbatch: int, **overrides) -> PPOConfig:
    base = dict(
        global_minibatch_size=16,
        microbatch_size=microbatch,
        per_rank_minibatch_size=16,
        grad_accumulation_per_rank=max(1, 16 // microbatch),
        ppo_epochs=1,
        sample_weighted_ddp_reduction=False,
    )
    base.update(overrides)
    return PPOConfig(**base)


def _architecture() -> PA3Config:
    return PA3Config(
        encoder_layers=2,
        d_model=64,
        attention_heads=2,
        head_dim=32,
        ffn_dim=128,
        category_vocab=256,
        category_embed_dim=8,
        role_vocab=12,
        prefix_hidden=64,
        critic_hidden=64,
    )


def _fixture():
    config = _architecture()
    reference = build_model(config)
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, reference, envs=3, target_matches=4)
    return config, buffer


def _align_behavior_policy(model, buffer) -> None:
    """Set old_logprob to the current weights' logprob (ratio 1 reference)."""
    rows = list(buffer.rows)
    batch = buffer.to_batch(rows, device="cpu")
    with torch.no_grad():
        encoded = model.encode(batch.observation)
        evaluation = model.evaluate_encoded(
            encoded, batch.candidates, selected=batch.candidates.selected
        )
        logprobs = evaluation.request_logprob.float().tolist()
    for row, value in zip(rows, logprobs):
        row.old_logprob = float(value)


def _reference_objective(learner: PPOLearner, batch, backward: bool):
    """Independent FP32 reimplementation of the learner objective."""
    model = learner.model
    config = learner.config
    encoded = model.encode(batch.observation)
    values = model.value(encoded).float()
    evaluation = model.evaluate_encoded(
        encoded, batch.candidates, selected=batch.candidates.selected
    )
    logprob = evaluation.request_logprob.float()
    advantages = (
        batch.advantages.float()
        if batch.advantages is not None
        else batch.raw_advantages.float()
    )
    valid = batch.row_valid
    actor = batch.actor_mask & valid
    ratio = torch.exp(logprob - batch.old_logprob.float())
    objective = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages,
    )
    actor_denom = actor.to(objective.dtype).sum().clamp_min(1.0)
    policy = -(objective * actor.to(objective.dtype)).sum() / actor_denom
    value = 0.5 * torch.nn.functional.mse_loss(values[valid], batch.returns[valid])
    entropy = (evaluation.request_entropy.float() * actor.to(objective.dtype)).sum() / actor_denom
    uniform_kl = (
        evaluation.request_uniform_kl.float() * actor.to(objective.dtype)
    ).sum() / actor_denom
    loss = (
        policy
        - config.entropy_coefficient * entropy
        + config.uniform_kl_coefficient * uniform_kl
        + config.value_coefficient * value
    )
    if backward:
        loss.backward()
    return {
        "loss": loss,
        "policy": policy,
        "value": value,
        "entropy": entropy,
        "uniform_kl": uniform_kl,
    }


def _named_gradients(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_full_objective_matches_independent_reference():
    config, buffer = _fixture()
    model = build_model(config)
    _align_behavior_policy(model, buffer)
    learner = PPOLearner(model, _config(microbatch=16), device="cpu", amp=False)
    prepared = learner.prepare_batch(buffer)
    minibatch = next(iter(prepared.iter_minibatches(len(prepared), shuffle=False)))

    learner.optimizer.zero_grad(set_to_none=True)
    terms = learner._forward_terms(minibatch)
    learner_loss = terms["loss_unscaled"] + learner.config.value_coefficient * terms["value_mean"]
    learner_loss.backward()
    learner_grads = _named_gradients(model)
    learner_values = {
        "loss": float(learner_loss.detach()),
        "policy": float(terms["policy_mean"]),
        "value": float(terms["value_mean_detached"]),
        "entropy": float(terms["entropy_mean"]),
        "uniform_kl": float(terms["uniform_kl_mean"]),
    }

    learner.optimizer.zero_grad(set_to_none=True)
    reference = _reference_objective(learner, minibatch, backward=True)
    reference_grads = _named_gradients(model)

    for key, value in learner_values.items():
        assert math.isclose(value, float(reference[key].detach()), rel_tol=1e-5, abs_tol=1e-6), key

    assert learner_grads.keys() == reference_grads.keys()
    worst = 0.0
    for name, grad in learner_grads.items():
        reference_grad = reference_grads[name]
        assert grad.shape == reference_grad.shape
        scale = max(float(reference_grad.abs().max()), 1e-6)
        difference = float((grad - reference_grad).abs().max())
        worst = max(worst, difference / scale)
        assert difference <= 1e-5 * scale + 1e-7, (name, difference, scale)
    assert worst < 1e-4


def test_regularizer_signs_and_coefficients_are_exact():
    config, buffer = _fixture()
    model = build_model(config)
    _align_behavior_policy(model, buffer)
    learner = PPOLearner(model, _config(microbatch=16), device="cpu", amp=False)
    prepared = learner.prepare_batch(buffer)
    minibatch = next(iter(prepared.iter_minibatches(len(prepared), shuffle=False)))
    terms = learner._forward_terms(minibatch)
    expected = (
        terms["policy_mean"]
        - learner.config.entropy_coefficient * terms["entropy_mean"]
        + learner.config.uniform_kl_coefficient * terms["uniform_kl_mean"]
    )
    assert torch.allclose(terms["loss_unscaled"].detach(), expected.detach(), atol=1e-7)
    # The value term stays outside `loss_unscaled` and is added once by the
    # caller with `value_coefficient`.
    full = learner._forward(minibatch)
    assert torch.allclose(
        full["loss"].detach(),
        terms["loss_unscaled"].detach() + learner.config.value_coefficient * terms["value_mean"].detach(),
        atol=1e-7,
    )
    assert float(terms["entropy_mean"]) > 0.0
    assert float(terms["uniform_kl_mean"]) >= 0.0


def test_actor_free_rows_do_not_touch_the_policy_or_encoder():
    config, buffer = _fixture()
    model = build_model(config)
    _align_behavior_policy(model, buffer)
    learner = PPOLearner(model, _config(microbatch=16), device="cpu", amp=False)
    prepared = learner.prepare_batch(buffer)
    minibatch = next(iter(prepared.iter_minibatches(len(prepared), shuffle=False)))
    actor_free = torch.zeros_like(minibatch.actor_mask)
    from dataclasses import replace

    masked = replace(minibatch, actor_mask=actor_free)
    learner.optimizer.zero_grad(set_to_none=True)
    terms = learner._forward_terms(masked)
    loss = terms["loss_unscaled"] + learner.config.value_coefficient * terms["value_mean"]
    loss.backward()
    scorer_norm = 0.0
    encoder_norm = 0.0
    value_norm = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.abs().sum())
        if name.startswith("scorer"):
            scorer_norm += norm
        elif name.startswith("encoder"):
            encoder_norm += norm
        elif name.startswith("value_head"):
            value_norm += norm
    assert float(terms["actor_count"]) == 0.0
    assert scorer_norm == 0.0, "actor-free rows must not create policy gradients"
    assert value_norm > 0.0, "value rows still train the critic"

    # The shared encoder legitimately keeps the critic's gradient: with no
    # actor rows the objective must collapse exactly onto the value loss.
    learner.optimizer.zero_grad(set_to_none=True)
    encoded = model.encode(masked.observation)
    values = model.value(encoded).float()
    # The configured objective is value_coefficient * half-MSE.
    value_only = (
        learner.config.value_coefficient
        * 0.5
        * torch.nn.functional.mse_loss(
            values[masked.row_valid], masked.returns[masked.row_valid]
        )
    )
    value_only.backward()
    value_only_encoder_norm = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith("encoder") and parameter.grad is not None:
            value_only_encoder_norm += float(parameter.grad.abs().sum())
    assert math.isclose(encoder_norm, value_only_encoder_norm, rel_tol=1e-5, abs_tol=1e-6)


def _run_consecutive_updates(model, buffer, microbatch: int, steps: int) -> PPOLearner:
    learner = PPOLearner(model, _config(microbatch=microbatch), device="cpu", amp=False)
    prepared = learner.prepare_batch(buffer)
    for step in range(steps):
        learner.update(
            prepared,
            committed_matches=step * 128,
            generator=torch.Generator().manual_seed(10_000 + step),
        )
    return learner


def test_micro_split_adam_state_parity_across_consecutive_updates():
    config, buffer = _fixture()
    reference_model = build_model(config)
    _align_behavior_policy(reference_model, buffer)

    full_model = copy.deepcopy(reference_model)
    micro_model = copy.deepcopy(reference_model)
    initial = {k: v.detach().clone() for k, v in reference_model.state_dict().items()}
    for model in (full_model, micro_model):
        for name, value in model.state_dict().items():
            assert torch.equal(value, initial[name]), "fixture models must start identical"

    full = _run_consecutive_updates(full_model, buffer, microbatch=16, steps=3)
    micro = _run_consecutive_updates(micro_model, buffer, microbatch=4, steps=3)

    assert full.optimizer_steps == micro.optimizer_steps == 3
    assert full.scheduler.matches == micro.scheduler.matches
    full_state = full.model.state_dict()
    micro_state = micro.model.state_dict()
    for name, value in full_state.items():
        assert torch.allclose(value, micro_state[name], atol=1e-6, rtol=1e-5), name

    full_opt = full.optimizer.state_dict()
    micro_opt = micro.optimizer.state_dict()
    assert full_opt["param_groups"][0]["lr"] == micro_opt["param_groups"][0]["lr"]
    for full_param, micro_param in zip(
        full_opt["param_groups"][0]["params"], micro_opt["param_groups"][0]["params"]
    ):
        full_state_entry = full_opt["state"][full_param]
        micro_state_entry = micro_opt["state"][micro_param]
        assert int(full_state_entry["step"]) == int(micro_state_entry["step"]) == 3
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(
                full_state_entry[key], micro_state_entry[key], atol=1e-7, rtol=1e-5
            ), key


def test_checkpoint_resume_continues_identical_updates():
    config, buffer = _fixture()
    model = build_model(config)
    _align_behavior_policy(model, buffer)
    learner = PPOLearner(model, _config(microbatch=4), device="cpu", amp=False)
    prepared = learner.prepare_batch(buffer)
    learner.update(prepared, committed_matches=0, generator=torch.Generator().manual_seed(1))
    checkpoint = copy.deepcopy(learner.state_dict())

    for step in range(2):
        learner.update(
            prepared,
            committed_matches=(step + 1) * 128,
            generator=torch.Generator().manual_seed(20 + step),
        )
    reference_state = copy.deepcopy(learner.state_dict())

    resumed_model = build_model(config)
    resumed = PPOLearner(resumed_model, _config(microbatch=4), device="cpu", amp=False)
    resumed.load_state_dict(checkpoint)
    resumed_prepared = resumed.prepare_batch(buffer)
    for step in range(2):
        resumed.update(
            resumed_prepared,
            committed_matches=(step + 1) * 128,
            generator=torch.Generator().manual_seed(20 + step),
        )
    resumed_state = resumed.state_dict()

    assert resumed.optimizer_steps == learner.optimizer_steps
    assert resumed.scheduler.matches == learner.scheduler.matches
    for name, value in reference_state["model"].items():
        assert torch.allclose(value, resumed_state["model"][name], atol=1e-6, rtol=1e-5), name
    for left, right in zip(
        reference_state["optimizer"]["param_groups"], resumed_state["optimizer"]["param_groups"]
    ):
        assert left["lr"] == right["lr"]
    for key in reference_state["optimizer"]["state"]:
        left = reference_state["optimizer"]["state"][key]
        right = resumed_state["optimizer"]["state"][key]
        assert int(left["step"]) == int(right["step"])
        for moment in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(left[moment], right[moment], atol=1e-7, rtol=1e-5), moment
