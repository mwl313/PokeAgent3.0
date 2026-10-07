"""All-singleton requests teach the value head but never the actor."""

from __future__ import annotations

import torch

from agent.ppo import PPOLearner, PPOConfig
from agent.ppo.losses import normalize_advantages

from dataclasses import replace

from pa3_test_util import synthetic_rows


def _buffer():
    return synthetic_rows(
        count=4,
        match_ids=[0, 0, 1, 1],
        sides=[0, 1, 0, 1],
        singleton=[True, False, True, False],
        values=[0.0, 0.0, 0.0, 0.0],
        rewards=[1.0, 0.0, -1.0, 0.0],
        dones=[True, False, True, False],
    )


def test_singleton_rows_are_excluded_from_the_actor_mask_but_keep_values():
    buffer = _buffer()
    assert len(buffer) == 4
    assert len(buffer.value_rows()) == 4
    assert len(buffer.actor_rows()) == 2

    batch = buffer.to_batch()
    assert batch.actor_mask.tolist() == [False, True, False, True]
    assert batch.row_valid.all()


def test_advantage_normalization_uses_only_actor_rows():
    advantages = torch.tensor([100.0, 1.0, -100.0, -1.0])
    actor_mask = torch.tensor([False, True, False, True])
    normalized = normalize_advantages(advantages, actor_mask=actor_mask)
    actor_rows = normalized[actor_mask]
    assert abs(float(actor_rows.mean())) < 1e-5
    assert abs(float(actor_rows.std(unbiased=False)) - 1.0) < 1e-4
    # Singleton rows are not folded into the actor statistics.
    assert float(normalized[0]) == 100.0


def test_value_loss_still_covers_singleton_rows(model_factory):
    learner = PPOLearner(
        model_factory(),
        PPOConfig(global_minibatch_size=4, microbatch_size=4),
        amp=False,
    )
    buffer = _buffer()
    buffer.compute_gae(gamma=1.0, gae_lambda=0.95)
    batch = buffer.to_batch()
    # Force a known return so the value loss is nonzero for every row.
    batch = replace(batch, returns=torch.tensor([2.0, 2.0, -2.0, -2.0]))
    stats = learner._forward(batch)
    model_values = learner.model.value(learner.model.encode(batch.observation)).detach()
    manual = 0.5 * torch.nn.functional.mse_loss(model_values, batch.returns.float())
    assert torch.allclose(stats["value_loss"], manual, atol=1e-6)
    assert float(stats["value_loss"]) > 0.0
    # Actor rows exclude the two all-singleton requests.
    assert float(stats["actor_rows"]) == 2.0


def test_reported_actor_rows_match_the_actor_mask(model_factory):
    learner = PPOLearner(
        model_factory(),
        PPOConfig(global_minibatch_size=4, microbatch_size=4),
        amp=False,
    )
    learner.model.eval()
    buffer = _buffer()
    batch = learner.prepare_batch(buffer)
    report = learner.update(batch, committed_matches=0)
    assert report.rows == 4
    assert report.actor_rows == 2
