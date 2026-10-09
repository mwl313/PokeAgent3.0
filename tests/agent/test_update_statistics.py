"""P0 regressions for the PPO update statistics and accumulation contract.

These exercise the learner with a stub network whose per-row log-probabilities
and values are prescribed, so the expected row-weighted statistics can be
computed in closed form. The last-microbatch-only aggregation in the pre-v2
``update()`` returns a different number for every case here.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from agent.buffer.rollout_buffer import RolloutBatch
from agent.model.config import PA3Config
from agent.ppo.config import PPOConfig
from agent.ppo.learner import PPOLearner
from agent.types.actions import BRANCH_CAPACITY
from agent.types.observation import ObservationBatch
from agent.types.requests import BranchCandidatesBatch


class StubModel(nn.Module):
    """Row-independent stand-in: logprob = base + p, value = base + 0*p."""

    def __init__(self, logprobs, values):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self._logprobs = torch.tensor(logprobs, dtype=torch.float32)
        self._values = torch.tensor(values, dtype=torch.float32)
        self.config = PA3Config()

    def encode(self, observation):
        return SimpleNamespace(batch=int(observation.token_mask.shape[0]))

    def value(self, encoded):
        return self._values[: encoded.batch] + 0.0 * self.p.sum()

    def evaluate_encoded(self, encoded, candidates, selected=None):
        count = encoded.batch
        current = self._logprobs[:count] + self.p.sum()
        return SimpleNamespace(
            request_logprob=current,
            request_entropy=torch.zeros(count, dtype=torch.float32),
            request_uniform_kl=torch.zeros(count, dtype=torch.float32),
        )


def make_batch(ratios, advantages=None, values=None, actor_mask=None, row_valid=None):
    count = len(ratios)
    logprobs = torch.zeros(count, dtype=torch.float32)
    # ratio = exp(logprob - old_logprob) at p = 0
    old = logprobs - torch.log(torch.tensor(ratios, dtype=torch.float32))
    advantages = torch.tensor(advantages if advantages is not None else [1.0] * count, dtype=torch.float32)
    values = torch.tensor(values if values is not None else [0.5] * count, dtype=torch.float32)
    dummy_obs = ObservationBatch.dummy(batch_size=count)
    candidates = BranchCandidatesBatch(
        action_ids=torch.zeros((count, BRANCH_CAPACITY, 1, 6), dtype=torch.long),
        mask=torch.ones((count, BRANCH_CAPACITY, 1), dtype=torch.bool),
        entity_token=torch.zeros((count, BRANCH_CAPACITY, 1), dtype=torch.long),
        move_token=torch.full((count, BRANCH_CAPACITY, 1), -1, dtype=torch.long),
        branch_valid=torch.ones((count, BRANCH_CAPACITY), dtype=torch.bool),
        selected=torch.zeros((count, BRANCH_CAPACITY), dtype=torch.long),
    )
    return RolloutBatch(
        observation=dummy_obs,
        candidates=candidates,
        old_logprob=old,
        values=values,
        raw_advantages=advantages,
        returns=values + advantages,
        rewards=torch.zeros(count, dtype=torch.float32),
        dones=torch.ones(count, dtype=torch.bool),
        actor_mask=torch.tensor(actor_mask if actor_mask is not None else [True] * count, dtype=torch.bool),
        row_valid=torch.tensor(row_valid if row_valid is not None else [True] * count, dtype=torch.bool),
        match_ids=torch.arange(count, dtype=torch.long),
        sides=torch.zeros(count, dtype=torch.long),
        request_index=torch.zeros(count, dtype=torch.long),
        turns=torch.zeros(count, dtype=torch.long),
        request_kind=torch.ones(count, dtype=torch.long),
        policy_ids=["current"] * count,
        advantages=advantages,
    )


def expected_kl(ratios, actor_mask=None):
    total = 0.0
    count = 0
    for index, ratio in enumerate(ratios):
        if actor_mask is not None and not actor_mask[index]:
            continue
        total += (ratio - 1.0) - math.log(ratio)
        count += 1
    return total / max(count, 1)


def build_learner(model, minibatch=2, microbatch=1, epochs=1):
    config = PPOConfig(
        global_minibatch_size=minibatch,
        microbatch_size=microbatch,
        ppo_epochs=epochs,
        per_rank_minibatch_size=minibatch,
        grad_accumulation_per_rank=max(1, minibatch // microbatch),
        # Isolate the accumulation math from the cross-rank sample weighting.
        sample_weighted_ddp_reduction=False,
    )
    return PPOLearner(model, config, device="cpu", amp=False)


def test_epoch_kl_is_the_row_weighted_mean_not_the_last_microbatch():
    # Three near-identical rows and one row whose policy moved far away. The
    # pre-v2 aggregation reports the last microbatch (0.0 or 1.66), never 0.415.
    ratios = [1.0, 1.0, 1.0, 1.5]
    model = StubModel([0.0] * 4, [0.5] * 4)
    learner = build_learner(model, minibatch=2, microbatch=1, epochs=1)
    batch = make_batch(ratios)
    report = learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(3))
    assert report.epoch_approx_kl[0] == pytest.approx(expected_kl(ratios), rel=1e-4)
    assert report.approx_kl == pytest.approx(expected_kl(ratios), rel=1e-4)
    assert report.epoch_approx_kl[0] not in (0.0, pytest.approx(expected_kl([1.5] * 4)))


def test_epoch_kl_uses_actor_rows_only_and_weights_by_actor_count():
    ratios = [1.0, 1.0, 2.0, 2.0]
    actor_mask = [False, True, True, True]
    model = StubModel([0.0] * 4, [0.5] * 4)
    learner = build_learner(model, minibatch=2, microbatch=1, epochs=1)
    batch = make_batch(ratios, actor_mask=actor_mask)
    report = learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(5))
    assert report.epoch_approx_kl[0] == pytest.approx(expected_kl(ratios, actor_mask), rel=1e-4)


def test_ratio_and_clip_fractions_are_row_weighted():
    ratios = [1.0, 1.25, 1.25, 1.25]
    model = StubModel([0.0] * 4, [0.5] * 4)
    learner = build_learner(model, minibatch=2, microbatch=1, epochs=1)
    batch = make_batch(ratios)
    report = learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(7))
    assert report.ratio_mean == pytest.approx(sum(ratios) / len(ratios), rel=1e-4)
    assert report.clip_fraction == pytest.approx(3 / 4, rel=1e-4)


def test_exact_accumulation_matches_single_full_minibatch_gradient():
    count = 5
    ratios = [1.4, 0.8, 1.1, 1.3, 0.9]
    advantages = [1.0, -1.0, 0.5, -0.5, 1.0]
    values = [0.4, -0.2, 0.1, 0.3, -0.4]

    def gradients(exact, microbatch):
        model = StubModel([0.0] * count, values)
        learner = build_learner(model, minibatch=count, microbatch=microbatch, epochs=1)
        learner.config = PPOConfig(
            global_minibatch_size=count, microbatch_size=microbatch, ppo_epochs=1,
            per_rank_minibatch_size=count, grad_accumulation_per_rank=max(1, count // microbatch),
            exact_row_weighted_accumulation=exact,
            sample_weighted_ddp_reduction=False,
        )
        batch = make_batch(ratios, advantages=advantages, values=values)
        learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(11))
        return model.p.grad.detach().clone()

    reference = gradients(True, count)          # one microbatch = full minibatch
    accumulated = gradients(True, 1)            # 5 microbatches of one row
    assert torch.allclose(reference, accumulated, atol=1e-6)

    # With uneven actor counts the legacy weighting is provably different: the
    # single actor row sits in a micro whose padded sample weight halves it.
    uneven_ratios = [1.4, 1.0, 1.0, 0.9]
    uneven_actor = [True, False, False, True]

    def uneven_gradient(exact):
        model = StubModel([0.0] * 4, [0.4] * 4)
        learner = build_learner(model, minibatch=4, microbatch=2, epochs=1)
        learner.config = PPOConfig(
            global_minibatch_size=4, microbatch_size=2, ppo_epochs=1,
            per_rank_minibatch_size=4, grad_accumulation_per_rank=2,
            exact_row_weighted_accumulation=exact,
            sample_weighted_ddp_reduction=False,
        )
        batch = make_batch(uneven_ratios, actor_mask=uneven_actor)
        learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(19))
        return model.p.grad.detach().clone()

    assert not torch.allclose(uneven_gradient(True), uneven_gradient(False), atol=1e-6), (
        "the legacy weighting is expected to differ on uneven actor rows; "
        "this guards the A/B flag"
    )


def test_exact_accumulation_handles_the_padded_final_minibatch():
    count = 3
    ratios = [1.2, 1.0, 0.8]
    advantages = [1.0, -1.0, 0.5]
    values = [0.2, -0.1, 0.4]

    def gradients(exact):
        model = StubModel([0.0] * count, values)
        learner = build_learner(model, minibatch=4, microbatch=2, epochs=1)
        learner.config = PPOConfig(
            global_minibatch_size=4, microbatch_size=2, ppo_epochs=1,
            per_rank_minibatch_size=4, grad_accumulation_per_rank=2,
            exact_row_weighted_accumulation=exact,
            sample_weighted_ddp_reduction=False,
        )
        batch = make_batch(ratios, advantages=advantages, values=values)
        learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(13))
        return model.p.grad.detach().clone()

    reference_model = StubModel([0.0] * count, values)
    reference_learner = build_learner(reference_model, minibatch=count, microbatch=count, epochs=1)
    reference_learner.config = PPOConfig(
        global_minibatch_size=count, microbatch_size=count, ppo_epochs=1,
        per_rank_minibatch_size=count, grad_accumulation_per_rank=1,
        exact_row_weighted_accumulation=True,
        sample_weighted_ddp_reduction=False,
    )
    reference_batch = make_batch(ratios, advantages=advantages, values=values)
    reference_learner.update(reference_batch, committed_matches=0)
    reference = reference_model.p.grad.detach().clone()

    padded = gradients(True)
    assert torch.allclose(reference, padded, atol=1e-6)


class SkippingScaler:
    """Minimal GradScaler stand-in that reports one skipped optimizer step."""

    def __init__(self):
        self._scale = 1024.0
        self.steps = 0
        self.skips = 0

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        self.skips += 1  # simulate an inf/nan skip

    def update(self):
        self._scale = self._scale / 2


def test_skipped_optimizer_steps_are_not_counted_as_committed_steps():
    model = StubModel([0.0] * 2, [0.5] * 2)
    learner = build_learner(model, minibatch=2, microbatch=1, epochs=1)
    learner.scaler = SkippingScaler()
    batch = make_batch([1.0, 1.0])
    report = learner.update(batch, committed_matches=0, generator=torch.Generator().manual_seed(17))
    assert learner.optimizer_steps == 0
    assert report.optimizer_steps == 0
    assert report.optimizer_steps_skipped >= 1
