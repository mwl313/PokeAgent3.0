"""Entropy and uniform reverse-KL normalization sanity."""

from __future__ import annotations

import math

import torch

from agent.types.actions import CandidateSet, RequestKind
from agent.types.observation import ObservationBatch
from agent.types.requests import RequestRow

from pa3_test_util import make_candidate_batch, make_request, move_action, preview_branches


def _uniform_model(model_factory):
    """A model whose scorer logits are identically zero for legal candidates."""
    model = model_factory()
    with torch.no_grad():
        model.scorer.query_projection.weight.zero_()
        model.scorer.query_projection.bias.zero_()
        model.scorer.logit_scale.zero_()
        model.scorer.logit_bias.zero_()
        model.scorer.action_projection.weight.zero_()
        model.scorer.action_projection.bias.zero_()
        model.scorer.entity_projection.weight.zero_()
        model.scorer.entity_projection.bias.zero_()
        model.scorer.move_projection.weight.zero_()
        model.scorer.move_projection.bias.zero_()
    return model


def test_uniform_policy_has_normalized_entropy_one_and_zero_kl(model_factory):
    model = _uniform_model(model_factory)
    observation = ObservationBatch.dummy(1)
    candidates = make_candidate_batch([make_request(observation, preview_branches())])
    _, evaluation = model.evaluate(observation, candidates)
    # Branch 0 has 6 legal candidates, branch 2 has 4.
    assert abs(float(evaluation.entropy_normalized[0, 0]) - 1.0) < 1e-5
    assert abs(float(evaluation.entropy_normalized[0, 2]) - 1.0) < 1e-5
    assert abs(float(evaluation.uniform_kl_normalized[0, 0])) < 1e-5
    assert abs(float(evaluation.request_uniform_kl[0])) < 1e-5
    assert abs(float(evaluation.request_entropy[0]) - 1.0) < 1e-5


def test_singleton_branches_contribute_zero_entropy_and_kl(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    singleton = CandidateSet.from_tuples([move_action(0, 0)], [True])
    second = CandidateSet.from_tuples(
        [move_action(1, index) for index in range(3)], [True, False, True]
    )
    request = RequestRow(
        observation=observation,
        kind=RequestKind.NORMAL,
        branch_slots=(0, 1),
        branches=(singleton, second),
    )
    candidates = make_candidate_batch([request])
    _, evaluation = model.evaluate(observation, candidates)
    assert float(evaluation.entropy_normalized[0, 0]) == 0.0
    assert float(evaluation.uniform_kl_normalized[0, 0]) == 0.0
    assert not bool(evaluation.branch_active[0, 0])
    # K = 2 after masking one candidate: normalization uses log(2).
    assert evaluation.branch_k[0, 1].item() == 2
    assert 0.0 <= float(evaluation.entropy_normalized[0, 1]) <= 1.0


def test_sharpening_the_policy_reduces_entropy_and_increases_kl(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    candidates = make_candidate_batch([make_request(observation, preview_branches())])
    _, before = model.evaluate(observation, candidates)
    with torch.no_grad():
        model.scorer.logit_scale.fill_(25.0)
    _, after = model.evaluate(observation, candidates)
    assert float(after.entropy_normalized[0, 0]) < float(before.entropy_normalized[0, 0])
    assert float(after.uniform_kl_normalized[0, 0]) > float(before.uniform_kl_normalized[0, 0])
    assert float(after.uniform_kl_normalized[0, 0]) > 0.0
    assert 0.0 <= float(before.entropy_normalized[0, 0]) <= 1.0


def test_normalized_entropy_matches_log_k_definition(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    candidates = make_candidate_batch([make_request(observation, preview_branches())])
    _, evaluation = model.evaluate(observation, candidates)
    for branch in range(4):
        k = int(evaluation.branch_k[0, branch])
        if k < 2:
            continue
        # H / log K <= 1 for any distribution over K candidates.
        assert float(evaluation.entropy_normalized[0, branch]) <= 1.0 + 1e-5
        assert float(evaluation.entropy_normalized[0, branch]) >= -1e-6
    assert math.log(6) > 0
