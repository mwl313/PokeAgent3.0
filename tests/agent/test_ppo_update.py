"""PPO scaffolding: ratios, accumulation, mock rollout updates and KL stops."""

from __future__ import annotations

import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.ppo import PPOLearner, PPOConfig
from agent.ppo.losses import policy_loss, value_loss


def _config(**overrides) -> PPOConfig:
    base = dict(global_minibatch_size=32, microbatch_size=16)
    base.update(overrides)
    return PPOConfig(**base)


def test_ratio_is_one_and_kl_is_zero_under_identical_weights(model_factory):
    model = model_factory()
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, model, envs=2, target_matches=2)
    learner = PPOLearner(model, _config(), amp=False)
    batch = learner.prepare_batch(buffer)
    stats = learner._forward(batch)
    assert abs(float(stats["ratio_mean"]) - 1.0) < 1e-4
    assert float(stats["approx_kl"]) < 1e-6
    assert float(stats["clip_fraction"]) == 0.0
    assert torch.isfinite(stats["loss"])


def test_policy_and_value_loss_definitions():
    ratio = torch.tensor([1.0, 1.3, 0.5])
    advantages = torch.tensor([1.0, 1.0, -1.0])
    loss = policy_loss(ratio, advantages, clip_epsilon=0.2)
    expected = -torch.tensor(
        [
            min(1.0 * 1.0, 1.2 * 1.0),
            min(1.3 * 1.0, 1.2 * 1.0),
            min(0.5 * -1.0, 0.8 * -1.0),
        ]
    ).mean()
    assert torch.allclose(loss, expected)
    values = torch.tensor([0.0, 2.0])
    returns = torch.tensor([1.0, 0.0])
    # MSE = ((0-1)^2 + (2-0)^2) / 2 = 2.5, and L_value = 0.5 * MSE.
    assert torch.allclose(value_loss(values, returns), torch.tensor(0.5 * 2.5))
    assert torch.allclose(
        value_loss(torch.zeros(0), torch.zeros(0)), torch.zeros(())
    )


def test_gradient_accumulation_matches_the_full_minibatch(model_factory):
    cfg = _config(global_minibatch_size=16, microbatch_size=8)
    assert cfg.effective_accumulation == 2
    model = model_factory()
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, model, envs=2, target_matches=2)
    learner = PPOLearner(model, cfg, amp=False)
    batch = learner.prepare_batch(buffer)
    minibatch = next(iter(batch.iter_minibatches(len(batch), shuffle=False)))

    learner.optimizer.zero_grad(set_to_none=True)
    full_stats = learner._forward(minibatch)
    full_stats["loss"].backward()
    full_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in learner.model.named_parameters()
    }

    learner.optimizer.zero_grad(set_to_none=True)
    microbatches = list(minibatch.iter_microbatches(cfg.microbatch_size))
    for micro in microbatches:
        stats = learner._forward(micro)
        (stats["loss"] * (micro.sample_weight / len(microbatches))).backward()
    for name, parameter in learner.model.named_parameters():
        assert torch.allclose(
            parameter.grad, full_grads[name], atol=1e-5, rtol=1e-4
        ), name


def test_mock_rollout_reaches_a_ppo_update(model_factory):
    torch.manual_seed(20261006)
    model = model_factory(seed=20261006)
    engine = MockNativeEngine(num_teams=8, requests_per_match=4)
    buffer = collect_mock_rollout(
        engine, model, envs=4, target_matches=4, policy_id="current"
    )
    assert len(buffer) > 0
    assert buffer.natural_match_count() == 4

    learner = PPOLearner(model, _config(global_minibatch_size=16, microbatch_size=16), amp=False)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    report = learner.prepare_batch(buffer)
    report = learner.update(report, committed_matches=125_000)
    assert 1 <= report.epochs_run <= 4
    assert report.rows == len(buffer)
    assert report.optimizer_steps >= 1
    assert report.actor_rows > 0
    for key in ("policy_loss", "value_loss", "entropy", "uniform_kl", "approx_kl"):
        assert torch.isfinite(torch.tensor(getattr(report, key))), key
    # Natural-match clock: 125k of 250k warmup -> LR strictly between bounds.
    assert 1.0e-5 < report.learning_rate < 3.0e-4
    changed = sum(
        1 for k, v in model.state_dict().items() if not torch.equal(v, before[k])
    )
    assert changed > 0


def test_target_kl_skips_remaining_epochs_for_the_iteration(model_factory):
    model = model_factory()
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, model, envs=2, target_matches=2)
    cfg = _config(
        global_minibatch_size=64,
        microbatch_size=64,
        target_approx_kl=1e-12,
        ppo_epochs=4,
    )
    learner = PPOLearner(model, cfg, amp=False)
    batch = learner.prepare_batch(buffer)
    report = learner.update(batch, committed_matches=0)
    assert report.epochs_run == 1
    assert report.stopped_early is True
    assert len(report.epoch_approx_kl) == 1


def test_optimizer_matches_the_frozen_spec(model_factory):
    learner = PPOLearner(model_factory(), PPOConfig(), amp=False)
    group = learner.optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-5
    assert group["weight_decay"] == 0.0
    assert learner.config.ppo_epochs == 4
    assert learner.config.global_minibatch_size == 4096
    assert learner.config.microbatch_size == 256
    assert learner.config.effective_accumulation == 16
    assert learner.config.clip_epsilon == 0.2
    assert learner.config.gamma == 1.0
    assert learner.config.gae_lambda == 0.95
    assert learner.config.value_coefficient == 0.5
    assert learner.config.value_clip is False
    assert learner.config.max_grad_norm == 0.5
    assert learner.config.entropy_coefficient == 0.01
    assert learner.config.uniform_kl_coefficient == 0.001
    assert learner.config.target_approx_kl == 0.03
    assert learner.config.action_temperature == 1.0
    assert group["lr"] == 1e-5  # warmup start, not the peak
