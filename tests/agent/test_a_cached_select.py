from dataclasses import replace

import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import build_model
from agent.ppo import PPOConfig, PPOLearner


def test_a1_contiguous_cached_select_is_view_and_preserves_values(model_factory):
    model = model_factory(seed=20261011)
    buffer = collect_mock_rollout(
        MockNativeEngine(num_teams=4, requests_per_match=3), model,
        envs=3, target_matches=3,
    )
    config = PPOConfig(global_minibatch_size=8, microbatch_size=4,
                       per_rank_minibatch_size=8,
                       experimental_cached_contiguous_select=True)
    learner = PPOLearner(model, config, device="cpu", amp=False)
    plan = learner.prepare_streaming(buffer, cache_device="cpu")
    indices = torch.arange(2, 6)
    valid = torch.ones(4, dtype=torch.bool)
    selected = learner._plan_minibatch(plan, indices, valid)
    expected = plan.cached_batch.select(indices, row_valid=valid)
    assert selected.old_logprob.data_ptr() == plan.cached_batch.old_logprob[2:].data_ptr()
    assert torch.equal(selected.old_logprob, expected.old_logprob)
    assert torch.equal(selected.row_valid, valid)


def test_a1_flag_is_off_by_default():
    assert PPOConfig().experimental_cached_contiguous_select is False


def test_a1_real_shuffle_does_not_force_contiguous_path(model_factory):
    model = model_factory(seed=20261011)
    buffer = collect_mock_rollout(
        MockNativeEngine(num_teams=4, requests_per_match=3), model,
        envs=3, target_matches=3,
    )
    config = PPOConfig(
        global_minibatch_size=8, microbatch_size=4, per_rank_minibatch_size=8,
        ppo_epochs=2, experimental_cached_contiguous_select=True,
        target_approx_kl=100.0,
    )
    learner = PPOLearner(model, config, device="cpu", amp=False)
    plan = learner.prepare_streaming(buffer, cache_device="cpu")
    before = plan.cached_batch.old_logprob.clone()
    report = learner.update_streaming(
        plan, committed_matches=0, generator=torch.Generator().manual_seed(20261011)
    )
    assert learner.profile["cached_select_calls"] > 0
    assert learner.profile["cached_contiguous_hits"] == 0
    assert learner.profile["cached_contiguous_fraction"] == 0.0
    assert torch.equal(plan.cached_batch.old_logprob, before)
