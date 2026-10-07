"""Masking is applied before sampling and before any log-probability math."""

from __future__ import annotations

import torch

from agent.types.actions import CandidateSet, RequestKind
from agent.types.observation import ObservationBatch
from agent.types.requests import BranchCandidatesBatch, RequestRow

from pa3_test_util import make_candidate_batch, make_request, move_action, preview_branches


def test_invalid_action_probability_is_exactly_zero(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    branch = CandidateSet.from_tuples(
        [move_action(0, index, index % 3) for index in range(4)],
        [True, False, True, False],
    )
    second = CandidateSet.from_tuples([move_action(1, 0), move_action(1, 1)], [True, True])
    request = RequestRow(
        observation=observation,
        kind=RequestKind.NORMAL,
        branch_slots=(0, 1),
        branches=(branch, second),
    )
    candidates = make_candidate_batch([request])
    result = model.sample(observation, candidates, generator=torch.Generator().manual_seed(0))
    probabilities = result.probabilities[0, 0, :4]
    assert probabilities[1].item() == 0.0
    assert probabilities[3].item() == 0.0
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6
    _, evaluation = model.evaluate(observation, candidates)
    assert evaluation.logprob_selected[0, 0].item() > -100.0


def test_sampling_never_selects_an_illegal_candidate(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    branches = preview_branches()
    # Hide everything except indices 0 and 4 in the first branch.
    mask = [True, False, False, False, True, False]
    branches = (CandidateSet.from_tuples(branches[0].tuples(), mask),) + branches[1:]
    request = make_request(observation, branches, kind=RequestKind.PREVIEW)
    candidates = make_candidate_batch([request])
    generator = torch.Generator().manual_seed(20261006)
    for _ in range(200):
        result = model.sample(observation, candidates, generator=generator)
        assert result.selected[0, 0].item() in (0, 4)


def test_singleton_branch_has_probability_one(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    singleton = CandidateSet.from_tuples([move_action(0, 3)], [True])
    second = CandidateSet.from_tuples([move_action(1, 0), move_action(1, 1)], [True, True])
    request = RequestRow(
        observation=observation,
        kind=RequestKind.NORMAL,
        branch_slots=(0, 1),
        branches=(singleton, second),
    )
    candidates = make_candidate_batch([request])
    result = model.sample(observation, candidates)
    assert result.selected[0, 0].item() == 0
    assert result.probabilities[0, 0, 0].item() == 1.0
    assert result.logprob_selected[0, 0].item() == 0.0
    assert result.branch_k[0, 0].item() == 1


def test_deterministic_selection_uses_only_legal_candidates(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    branch = CandidateSet.from_tuples(
        [move_action(0, index) for index in range(4)], [False, False, True, False]
    )
    request = RequestRow(
        observation=observation,
        kind=RequestKind.NORMAL,
        branch_slots=(0, 1),
        branches=(branch, CandidateSet.from_tuples([move_action(1, 0)], [True])),
    )
    candidates = make_candidate_batch([request])
    result = model.sample(observation, candidates, deterministic=True)
    assert result.selected[0, 0].item() == 2
