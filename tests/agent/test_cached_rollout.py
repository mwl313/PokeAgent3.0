"""The rollout cache changes materialization, never PPO's training inputs."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest
import torch

from agent.buffer.rollout_buffer import ColumnarObservationStore
from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import build_model
from agent.ppo import PPOLearner, PPOConfig


def _fixture(model_factory):
    model = model_factory(seed=20261009)
    buffer = collect_mock_rollout(
        MockNativeEngine(num_teams=4, requests_per_match=3), model,
        envs=3, target_matches=3,
    )
    # Non-actor rows still train the critic. A non-divisible batch also makes
    # the tail contain masked, repeated rows and an entirely padded micro.
    for index, row in enumerate(buffer.rows):
        row.actor_active = index % 3 != 0
    assert len(buffer.rows) % 8 != 0
    return model, buffer


def _config():
    return PPOConfig(
        global_minibatch_size=8, microbatch_size=4,
        per_rank_minibatch_size=8, ppo_epochs=2,
        target_approx_kl=100.0,
        sample_weighted_ddp_reduction=False,
    )


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-6, abs=1e-8)
    else:
        assert left == right


@pytest.mark.parametrize("executor", ["streaming", "ddp", "manual_allreduce"])
def test_cpu_cache_preserves_shuffles_masks_gradients_and_adam(
    model_factory, monkeypatch, executor,
):
    initial_model, buffer = _fixture(model_factory)
    config = _config()
    expected = build_model(initial_model.config)
    expected.load_state_dict(initial_model.state_dict())
    cached = build_model(initial_model.config)
    cached.load_state_dict(initial_model.state_dict())
    learners = [
        PPOLearner(model, config, device="cpu", amp=False)
        for model in (expected, cached)
    ]
    materializations = []
    original_to_batch = buffer.to_batch

    def record_materialization(rows=None, device=None):
        materializations.append(len(buffer.rows if rows is None else rows))
        return original_to_batch(rows, device=device)

    monkeypatch.setattr(buffer, "to_batch", record_materialization)
    reports = []
    counts = []
    for learner, cache_device in zip(learners, (None, "cpu")):
        materializations.clear()
        prepare = (
            learner.prepare_streaming if executor == "streaming"
            else learner.prepare_streaming_ddp
        )
        plan = prepare(buffer, cache_device=cache_device)
        assert (plan.cached_batch is not None) == (cache_device == "cpu")
        update = getattr(learner, "update_" + executor)
        reports.append(update(
            plan, committed_matches=125_000,
            generator=torch.Generator().manual_seed(173),
        ))
        counts.append(list(materializations))

    assert len(counts[0]) == config.ppo_epochs * math.ceil(len(buffer) / 8)
    assert counts[1] == [1, len(buffer)]
    assert reports[0].epochs_run == config.ppo_epochs
    _assert_same(reports[0].as_dict(), reports[1].as_dict())
    _assert_same(expected.state_dict(), cached.state_dict())
    _assert_same(learners[0].optimizer.state_dict(), learners[1].optimizer.state_dict())
    for left, right in zip(expected.parameters(), cached.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            _assert_same(left.grad, right.grad)


@pytest.mark.parametrize("prepare_name", ["prepare_streaming", "prepare_streaming_ddp"])
def test_cache_memory_bound_falls_back_to_streaming(model_factory, prepare_name):
    model, buffer = _fixture(model_factory)
    learner = PPOLearner(model, _config(), device="cpu", amp=False)
    plan = getattr(learner, prepare_name)(
        buffer, cache_device="cpu", cache_max_bytes=1,
    )
    assert plan.cached_batch is None
    assert plan.rows == buffer.rows
    assert plan.advantages.shape == (len(buffer),)


def test_columnar_store_supports_full_iteration_materialization(model_factory):
    model, buffer = _fixture(model_factory)
    reference = buffer.to_batch()
    columnar = ColumnarObservationStore()
    for index in range(len(buffer.observation_store)):
        columnar.add(buffer.observation_store.get(index))
    columnar_buffer = replace(buffer, observation_store=columnar, _stacked=None)
    actual = columnar_buffer.to_batch()
    for field in fields(reference.observation):
        left = getattr(reference.observation, field.name)
        right = getattr(actual.observation, field.name)
        if isinstance(left, torch.Tensor):
            _assert_same(left, right)
        else:
            assert type(left) is type(right)
    learner = PPOLearner(model, _config(), device="cpu", amp=False)
    assert learner.prepare_streaming(columnar_buffer, cache_device="cpu").cached_batch is not None


def test_batch_selection_preserves_optional_policy_provenance(model_factory):
    _, buffer = _fixture(model_factory)
    batch = buffer.to_batch()
    indices = torch.tensor([3, 0, 3], dtype=torch.long)
    batch = replace(batch, policy_ids=[str(index) for index in range(len(batch))])
    assert batch.select(indices).policy_ids == ["3", "0", "3"]
    assert replace(batch, policy_ids=[]).select(indices).policy_ids == []


@pytest.mark.parametrize("mask_kind", ["all", "mixed", "padding_only"])
def test_fixed_shape_value_reduction_matches_indexed_loss_and_gradients(
    model_factory, mask_kind,
):
    model, buffer = _fixture(model_factory)
    learner = PPOLearner(model, _config(), device="cpu", amp=False)
    batch = learner.prepare_batch(buffer)
    valid = torch.ones(len(batch), dtype=torch.bool)
    if mask_kind == "mixed":
        valid[::2] = False
    elif mask_kind == "padding_only":
        valid[:] = False
    returns = torch.linspace(-2.0, 3.0, len(batch))
    batch = replace(batch, row_valid=valid, returns=returns)
    terms = learner._forward_terms(batch)
    actual_grads = torch.autograd.grad(
        terms["value_mean"], tuple(model.parameters()), allow_unused=True,
    )
    values = model.value(model.encode(batch.observation)).float()
    expected_loss = (
        0.5 * torch.nn.functional.mse_loss(values[valid], returns[valid])
        if valid.any() else values.sum() * 0.0
    )
    expected_grads = torch.autograd.grad(
        expected_loss, tuple(model.parameters()), allow_unused=True,
    )
    _assert_same(terms["value_mean"], expected_loss)
    for actual, expected in zip(actual_grads, expected_grads):
        assert (actual is None) == (expected is None)
        if actual is not None:
            _assert_same(actual, expected)
    assert int(terms["value_count"]) == int(valid.sum())


def test_manual_executor_peer_overflow_resets_every_rank_scaler(model_factory, monkeypatch):
    """A finite local shard must back off when its peer overflows before SUM."""
    model, buffer = _fixture(model_factory)
    config = replace(_config(), global_minibatch_size=32,
                     per_rank_minibatch_size=32, microbatch_size=16, ppo_epochs=1)
    learner = PPOLearner(model, config, device="cpu", amp=False)
    learner.scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=8.0)
    scaler_state = learner.scaler.state_dict()
    scaler_state["_growth_tracker"] = 2
    learner.scaler.load_state_dict(scaler_state)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    monkeypatch.setattr("agent.ppo.ddp.all_reduce_flag", lambda flag: flag.zero_())
    report = learner.update_manual_allreduce(learner.prepare_streaming(buffer))
    assert report.optimizer_steps == 0
    assert report.optimizer_steps_skipped == 1
    assert learner.optimizer.state_dict()["state"] == {}
    _assert_same(before, model.state_dict())
    assert learner.scaler.get_scale() == 4.0
    assert learner.scaler.state_dict()["_growth_tracker"] == 0


def test_padding_nonfinite_targets_do_not_poison_critic_gradient(model_factory):
    model, buffer = _fixture(model_factory)
    learner = PPOLearner(model, _config(), device="cpu", amp=False)
    batch = learner.prepare_batch(buffer)
    valid = torch.ones_like(batch.row_valid)
    valid[::2] = False
    returns = batch.returns.clone()
    returns[~valid] = float("nan")
    terms = learner._forward_terms(replace(batch, returns=returns, row_valid=valid))
    terms["value_mean"].backward()
    assert torch.isfinite(terms["value_mean"])
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
