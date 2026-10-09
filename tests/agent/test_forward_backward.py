"""Forward/backward through the branch-structured objective."""

from __future__ import annotations

import torch

from agent.types.observation import ObservationBatch

from pa3_test_util import make_candidate_batch, make_request, preview_branches


def test_forward_backward_reaches_every_parameter(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(2)
    candidates = make_candidate_batch(
        [make_request(observation.select([i]), preview_branches()) for i in range(2)]
    )
    selected = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 2]])
    encoded, evaluation = model.evaluate(observation, candidates, selected=selected)
    values = model.value(encoded)
    loss = -(evaluation.request_logprob).mean() + 0.5 * (values**2).mean()
    loss.backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradients: {missing[:5]}"
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name
    nonzero = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert nonzero > 0.5 * sum(1 for _ in model.parameters())
    assert torch.isfinite(loss)


def test_backward_is_stable_for_masked_and_padded_branches(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    request = make_request(observation, preview_branches())
    candidates = make_candidate_batch([request])
    selected = torch.tensor([[0, 0, 0, 0]])
    encoded, evaluation = model.evaluate(observation, candidates, selected=selected)
    loss = -evaluation.request_logprob.mean() + model.value(encoded).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
