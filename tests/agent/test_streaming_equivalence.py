"""G0: streaming minibatch path vs whole-iteration path equivalence.

`prepare_batch/update` and `prepare_streaming/update_streaming` must produce the
same objective, gradients, optimizer state and report on identical rows. The
one structural difference is padding placement: `iter_minibatches` spreads the
padded rows over the minibatch while the streaming path appends them. With the
exact row-weighted accumulation the objective is invariant to that placement,
which is what this test verifies.
"""

from __future__ import annotations

import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import PA3Config, build_model
from agent.ppo import PPOLearner, PPOConfig


def _config(minibatch, microbatch):
    # minibatch/k have a remainder so the padded final minibatch path runs.
    return PPOConfig(
        global_minibatch_size=minibatch,
        microbatch_size=microbatch,
        ppo_epochs=2,
        per_rank_minibatch_size=minibatch,
        grad_accumulation_per_rank=max(1, minibatch // microbatch),
        sample_weighted_ddp_reduction=False,
        max_grad_norm=1.0e9,
    )


def _fixture():
    config = PA3Config(encoder_layers=2, d_model=64, attention_heads=2, head_dim=32,
                       ffn_dim=128, category_vocab=256, category_embed_dim=8,
                       prefix_hidden=64, critic_hidden=64)
    model = build_model(config)
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, model, envs=3, target_matches=3)
    return config, model, buffer


def _gradients(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_streaming_update_matches_whole_iteration_update():
    config, model, buffer = _fixture()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    minibatch, microbatch = max(5, len(buffer.rows) // 2), max(2, len(buffer.rows) // 6)

    standard_model = build_model(config)
    standard_model.load_state_dict(state)
    standard = PPOLearner(standard_model, _config(minibatch, microbatch), device="cpu", amp=False)
    standard_batch = standard.prepare_batch(buffer)
    standard_report = standard.update(
        standard_batch, committed_matches=0, generator=torch.Generator().manual_seed(31)
    )

    streaming_model = build_model(config)
    streaming_model.load_state_dict(state)
    streaming = PPOLearner(streaming_model, _config(minibatch, microbatch), device="cpu", amp=False)
    plan = streaming.prepare_streaming(buffer)
    streaming_report = streaming.update_streaming(
        plan, committed_matches=0, generator=torch.Generator().manual_seed(31)
    )

    # Report parity (row-weighted statistics over the same rows).
    for field in ("policy_loss", "value_loss", "entropy", "uniform_kl", "approx_kl",
                  "ratio_mean", "clip_fraction", "grad_norm"):
        a = getattr(standard_report, field)
        b = getattr(streaming_report, field)
        assert abs(a - b) <= 1e-5 * max(abs(a), abs(b), 1.0), (field, a, b)
    assert standard_report.epochs_run == streaming_report.epochs_run
    assert standard_report.optimizer_steps == streaming_report.optimizer_steps
    assert standard_report.optimizer_steps_skipped == streaming_report.optimizer_steps_skipped
    assert standard_report.actor_rows == streaming_report.actor_rows
    assert len(standard_report.epoch_approx_kl) == len(streaming_report.epoch_approx_kl)
    for a, b in zip(standard_report.epoch_approx_kl, streaming_report.epoch_approx_kl):
        assert abs(a - b) <= 1e-5 * max(abs(a), abs(b), 1.0), (a, b)

    # Gradient parity.
    left, right = _gradients(standard_model), _gradients(streaming_model)
    assert set(left) == set(right)
    scale = max(float(value.abs().max()) for value in left.values())
    worst = max(float((left[name] - right[name]).abs().max()) for name in left)
    assert worst <= 1e-5 * scale + 1e-8, (worst, scale)

    # One optimizer step produced the same weights.
    left_state = standard_model.state_dict()
    right_state = streaming_model.state_dict()
    for name in left_state:
        if not left_state[name].is_floating_point():
            assert torch.equal(left_state[name], right_state[name]), name
            continue
        difference = float((left_state[name] - right_state[name]).abs().max())
        assert difference <= 1e-6 * max(float(left_state[name].abs().max()), 1.0), name


def test_streaming_equivalence_with_actor_free_rows():
    config, model, buffer = _fixture()
    total = len(buffer.rows)
    for index, row in enumerate(buffer.rows):
        if index % 3 == 0:
            row.actor_active = False
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    minibatch, microbatch = max(5, total // 2), max(2, total // 5)

    model_a = build_model(config)
    model_a.load_state_dict(state)
    learner_a = PPOLearner(model_a, _config(minibatch, microbatch), device="cpu", amp=False)
    report_a = learner_a.update(
        learner_a.prepare_batch(buffer), committed_matches=0,
        generator=torch.Generator().manual_seed(37),
    )

    model_b = build_model(config)
    model_b.load_state_dict(state)
    learner_b = PPOLearner(model_b, _config(minibatch, microbatch), device="cpu", amp=False)
    report_b = learner_b.update_streaming(
        learner_b.prepare_streaming(buffer), committed_matches=0,
        generator=torch.Generator().manual_seed(37),
    )

    assert abs(report_a.approx_kl - report_b.approx_kl) <= 1e-6
    assert report_a.optimizer_steps == report_b.optimizer_steps
    left, right = _gradients(model_a), _gradients(model_b)
    scale = max(float(value.abs().max()) for value in left.values())
    worst = max(float((left[name] - right[name]).abs().max()) for name in left)
    assert worst <= 1e-5 * scale + 1e-8, (worst, scale)
    for name, parameter in model_b.named_parameters():
        assert torch.isfinite(parameter).all(), name
