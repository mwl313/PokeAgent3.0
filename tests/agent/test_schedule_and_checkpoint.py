"""Match-clocked LR schedule and checkpoint-like state_dict round trips."""

from __future__ import annotations

import math

import torch

from agent.ppo import MatchClockScheduler, PPOLearner, PPOConfig
from agent.types.observation import ObservationBatch

from pa3_test_util import make_candidate_batch, make_request, preview_branches


def test_warmup_and_cosine_schedule_are_clocked_by_matches(model_factory):
    learner = PPOLearner(model_factory(), PPOConfig(), amp=False)
    scheduler = learner.scheduler
    cfg = learner.config
    assert abs(scheduler.lr_for_matches(0) - 1e-5) < 1e-12
    assert abs(scheduler.lr_for_matches(125_000) - 1.55e-4) < 1e-9
    assert abs(scheduler.lr_for_matches(250_000) - 3e-4) < 1e-9
    mid = (250_000 + 100_000_000) / 2
    expected_mid = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * 0.5))
    assert abs(scheduler.lr_for_matches(mid) - expected_mid) < 1e-12
    assert abs(scheduler.lr_for_matches(100_000_000) - 3e-5) < 1e-12
    assert abs(scheduler.lr_for_matches(150_000_000) - 3e-5) < 1e-12

    lr = scheduler.step_to(500_000)
    assert abs(lr - scheduler.learning_rate) < 1e-15
    assert learner.optimizer.param_groups[0]["lr"] == lr
    state = scheduler.state_dict()
    scheduler.step_to(0)
    scheduler.load_state_dict(state)
    assert scheduler.matches == 500_000

    # The clock only advances on committed natural matches.
    scheduler.step_to(0)
    assert abs(scheduler.advance(1_000) - cfg.warmup_start_lr - 1.16e-6) < 1e-9
    assert scheduler.matches == 1_000


def test_learner_state_dict_round_trip_restores_identical_outputs(model_factory):
    model = model_factory(seed=5)
    learner = PPOLearner(model, PPOConfig(global_minibatch_size=4, microbatch_size=4), amp=False)
    engine_batch = make_candidate_batch(
        [make_request(ObservationBatch.dummy(1, seed=1), preview_branches())]
    )
    observation = ObservationBatch.dummy(1, seed=1)
    _, before = model.evaluate(observation, engine_batch)
    value_before = model.value(model.encode(observation))
    learner.scheduler.step_to(1_000_000)
    state = learner.state_dict()

    clone_model = model_factory(seed=77)
    clone = PPOLearner(
        clone_model, PPOConfig(global_minibatch_size=4, microbatch_size=4), amp=False
    )
    clone.load_state_dict(state)
    clone_model.eval()
    _, after = clone_model.evaluate(observation, engine_batch)
    value_after = clone_model.value(clone_model.encode(observation))

    assert torch.allclose(before.request_logprob, after.request_logprob, atol=1e-6)
    assert torch.allclose(value_before, value_after, atol=1e-6)
    assert clone.scheduler.matches == 1_000_000
    assert clone.optimizer_steps == learner.optimizer_steps
    assert clone.optimizer.param_groups[0]["lr"] == learner.optimizer.param_groups[0]["lr"]


def test_update_is_reproducible_from_the_same_state_dict(model_factory):
    from agent.mock_engine import MockNativeEngine, collect_mock_rollout

    model = model_factory(seed=21)
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, model, envs=2, target_matches=2)
    learner = PPOLearner(
        model, PPOConfig(global_minibatch_size=8, microbatch_size=8), amp=False
    )
    batch = learner.prepare_batch(buffer)
    generator = torch.Generator().manual_seed(0)
    first = learner.update(batch, committed_matches=0, generator=generator)
    first_weights = {k: v.clone() for k, v in model.state_dict().items()}

    clone_model = model_factory(seed=21)
    clone = PPOLearner(
        clone_model, PPOConfig(global_minibatch_size=8, microbatch_size=8), amp=False
    )
    clone_batch = clone.prepare_batch(buffer)
    second = clone.update(
        clone_batch, committed_matches=0, generator=torch.Generator().manual_seed(0)
    )
    assert abs(first.policy_loss - second.policy_loss) < 1e-6
    for key, value in first_weights.items():
        assert torch.allclose(value, clone_model.state_dict()[key], atol=1e-6), key
