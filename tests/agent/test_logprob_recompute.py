"""Frozen-weight log-probability recomputation and state_dict round trips."""

from __future__ import annotations

import copy

import torch

from agent.types.observation import ObservationBatch

from pa3_test_util import make_candidate_batch, make_request, preview_branches


def _sampled(model, observation, candidates, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return model.sample(observation, candidates, generator=generator)


def test_logprob_recomputation_matches_sampling_under_frozen_weights(model_factory):
    model = model_factory()
    model.eval()
    observation = ObservationBatch.dummy(3)
    candidates = make_candidate_batch(
        [make_request(observation.select([i]), preview_branches()) for i in range(3)]
    )
    sampled = _sampled(model, observation, candidates)
    _, recomputed = model.evaluate(observation, candidates, selected=sampled.selected)
    assert torch.allclose(
        sampled.request_logprob, recomputed.request_logprob, atol=1e-5, rtol=1e-5
    )
    # Request log-probability is the sum of the selected branch log-probs.
    assert torch.allclose(
        recomputed.request_logprob,
        recomputed.logprob_selected.sum(dim=-1),
        atol=1e-6,
    )


def test_logprob_recomputation_survives_a_state_dict_round_trip(model_factory):
    model = model_factory(seed=11)
    model.eval()
    observation = ObservationBatch.dummy(2)
    candidates = make_candidate_batch(
        [make_request(observation.select([i]), preview_branches()) for i in range(2)]
    )
    sampled = _sampled(model, observation, candidates, seed=3)
    saved = copy.deepcopy(model.state_dict())

    clone = model_factory(seed=999)
    clone.load_state_dict(saved)
    clone.eval()
    _, recomputed = clone.evaluate(observation, candidates, selected=sampled.selected)
    assert torch.allclose(
        sampled.request_logprob, recomputed.request_logprob, atol=1e-5, rtol=1e-5
    )
    # Weights are untouched by a frozen evaluation.
    for key, value in saved.items():
        assert torch.equal(value, clone.state_dict()[key])


def test_candidate_order_and_mask_are_part_of_the_recompute_contract(model_factory):
    model = model_factory()
    model.eval()
    observation = ObservationBatch.dummy(1)
    candidates = make_candidate_batch([make_request(observation, preview_branches())])
    sampled = _sampled(model, observation, candidates, seed=5)
    shifted = sampled.selected.clone()
    shifted[:, 0] = torch.where(shifted[:, 0] > 0, shifted[:, 0] - 1, shifted[:, 0] + 1)
    _, other = model.evaluate(observation, candidates, selected=shifted)
    assert not torch.allclose(
        sampled.request_logprob, other.request_logprob, atol=1e-6
    )
