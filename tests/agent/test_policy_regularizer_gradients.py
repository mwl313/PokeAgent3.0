"""C0 regressions: the PPO regularizers must reach the scorer's logits.

Before the fix, ``PA3Model._branch_stats`` built both the entropy and the
uniform-KL terms from ``log_prob.detach()`` (and ``probs.detach()``), so the
``-entropy_coefficient * entropy + uniform_kl_coefficient * uniform_kl`` part
of the learner loss had no gradient path to the policy. These tests fail on the
pre-fix code (the loss either has no ``grad_fn`` or a zero logit gradient) and
pass once both terms are differentiable.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from agent.types.actions import CandidateSet, RequestKind
from agent.types.observation import ObservationBatch
from agent.types.requests import RequestRow

from pa3_test_util import (
    make_candidate_batch,
    make_request,
    move_action,
    preview_branches,
)


def _biased_logits(rows: int = 2, vocab: int = 5) -> torch.Tensor:
    base = torch.tensor(
        [
            [2.0, 0.5, -0.5, 1.0, -2.0],
            [0.1, 1.7, -1.3, 0.4, 0.9],
        ]
    )
    logits = base[:rows]
    if rows > base.shape[0]:
        logits = torch.cat([logits, logits[-1:].repeat(rows - base.shape[0], 1)], dim=0)
    return logits[:, :vocab]


def _reference_stats(logits: torch.Tensor, mask: torch.Tensor):
    """Independent float64 reference for normalized entropy / uniform-KL."""
    values = logits.double()
    legal = mask.bool()
    masked = values.masked_fill(~legal, float("-inf"))
    safe = torch.where(legal.any(dim=-1, keepdim=True), masked, torch.zeros_like(masked))
    log_prob = torch.log_softmax(safe, dim=-1)
    prob = torch.where(legal, log_prob.exp(), torch.zeros_like(log_prob))
    k = legal.sum(dim=-1).to(values.dtype)
    log_k = torch.log(k.clamp_min(1.0))
    entropy = -(prob * torch.where(legal, log_prob, torch.zeros_like(log_prob))).sum(dim=-1)
    mean_log = (
        torch.where(legal, log_prob, torch.zeros_like(log_prob)) * legal.to(values.dtype)
    ).sum(dim=-1) / k.clamp_min(1.0)
    entropy_norm = torch.where(k >= 2, entropy / log_k, torch.zeros_like(entropy))
    kl = (-log_k - mean_log) / torch.where(k >= 2, log_k, torch.ones_like(log_k))
    kl = torch.where(k >= 2, kl, torch.zeros_like(kl))
    return log_prob, prob, entropy_norm, kl


def test_entropy_loss_produces_nonzero_logit_gradient(model_factory):
    model = model_factory()
    logits = _biased_logits().requires_grad_(True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    _, _, entropy, _ = model._branch_stats(logits, mask)
    assert entropy.requires_grad, "entropy must stay attached to the scorer graph"
    loss = entropy.sum()
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 1e-6


def test_uniform_kl_loss_produces_nonzero_logit_gradient(model_factory):
    model = model_factory()
    logits = _biased_logits().requires_grad_(True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    _, _, _, uniform_kl = model._branch_stats(logits, mask)
    assert uniform_kl.requires_grad, "uniform-KL must stay attached to the scorer graph"
    loss = uniform_kl.sum()
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 1e-6


def test_entropy_kl_gradient_matches_analytic_and_finite_difference(model_factory):
    model = model_factory()
    logits = _biased_logits().requires_grad_(True)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, True, False],
        ]
    )
    _, prob, entropy, uniform_kl = model._branch_stats(logits, mask)
    loss = (entropy + 0.5 * uniform_kl).sum()
    loss.backward()
    autograd_grad = logits.grad.clone()

    # Analytic derivative for d/dz [H / log K] and d/dz [KL / log K].
    k = mask.sum(dim=-1).to(prob.dtype)
    log_k = torch.log(k.clamp_min(1.0))
    legal_prob = prob.detach()
    legal_logp = torch.log(legal_prob.clamp_min(torch.finfo(prob.dtype).tiny))
    row_entropy = entropy.detach() * log_k
    active = (k >= 2).to(prob.dtype).unsqueeze(-1)
    d_entropy = active * (-(legal_prob * (legal_logp + row_entropy.unsqueeze(-1)))) / log_k.unsqueeze(-1)
    d_kl = active * (legal_prob - 1.0 / k.clamp_min(2.0).unsqueeze(-1)) / log_k.unsqueeze(-1)
    expected = (d_entropy + 0.5 * d_kl) * mask.to(prob.dtype)
    assert torch.allclose(autograd_grad.double(), expected.double(), atol=1e-5, rtol=1e-5), (
        autograd_grad - expected
    )

    # Central finite differences on the model's own forward values.
    eps = 1e-2
    flat = logits.detach().clone().reshape(-1)
    fd = torch.zeros_like(flat)
    with torch.no_grad():
        for index in range(flat.numel()):
            plus = flat.clone()
            plus[index] += eps
            minus = flat.clone()
            minus[index] -= eps
            _, _, entropy_plus, kl_plus = model._branch_stats(plus.view_as(logits), mask)
            _, _, entropy_minus, kl_minus = model._branch_stats(minus.view_as(logits), mask)
            fd[index] = (
                (entropy_plus.sum() + 0.5 * kl_plus.sum())
                - (entropy_minus.sum() + 0.5 * kl_minus.sum())
            ) / (2 * eps)
    assert torch.allclose(
        autograd_grad.reshape(-1), fd, atol=2e-3, rtol=2e-2
    ), (autograd_grad.reshape(-1) - fd)


def test_singleton_and_padded_branch_zero_gradient(model_factory):
    model = model_factory()
    logits = torch.tensor(
        [
            [1.0, -1.0, 0.5, 0.25],  # singleton
            [0.5, 0.25, -0.5, 1.0],  # all masked (padded branch)
            [0.2, 1.4, -0.7, 0.1],  # two legal, biased
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [True, False, False, False],
            [False, False, False, False],
            [True, True, False, False],
        ]
    )
    _, probs, entropy, uniform_kl = model._branch_stats(logits, mask)
    assert torch.isfinite(probs).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(uniform_kl).all()
    assert probs[1].abs().sum().item() == 0.0
    assert entropy[0].item() == 0.0 and entropy[1].item() == 0.0
    assert uniform_kl[0].item() == 0.0 and uniform_kl[1].item() == 0.0

    (entropy.sum() + uniform_kl.sum()).backward()
    assert logits.grad is not None
    assert logits.grad[0].abs().sum().item() == 0.0, "singleton branches must not learn"
    assert logits.grad[1].abs().sum().item() == 0.0, "padded branches must not learn"
    assert logits.grad[2].abs().sum().item() > 1e-6


def test_all_illegal_mask_is_never_sampled_or_used(model_factory):
    model = model_factory()
    logits = _biased_logits(rows=2).requires_grad_(True)
    mask = torch.tensor([[False, False, False, False, False], [True, True, True, True, True]])
    _, probs, entropy, uniform_kl = model._branch_stats(logits, mask)
    assert torch.isfinite(probs).all()
    assert probs[0].abs().sum().item() == 0.0
    assert entropy[0].item() == 0.0
    assert uniform_kl[0].item() == 0.0
    (entropy.sum() + uniform_kl.sum()).backward()
    assert logits.grad is not None
    assert logits.grad[0].abs().sum().item() == 0.0

    # Sampling path: a branch that is marked invalid may carry an all-masked
    # candidate row without producing NaN or consuming an illegal action.
    observation = ObservationBatch.dummy(1)
    branches = preview_branches()
    first_size = len(branches[0].tuples())
    illegal = CandidateSet.from_tuples(
        [move_action(0, index) for index in range(first_size)],
        [False] * first_size,
    )
    request = make_request(observation, (illegal,) + tuple(branches[1:]), kind=RequestKind.PREVIEW)
    candidates = make_candidate_batch([request])
    result = model.sample(observation, candidates, generator=torch.Generator().manual_seed(7))
    assert torch.isfinite(result.request_logprob).all()
    assert result.probabilities[0, 0].abs().sum().item() == 0.0
    assert result.selected[0, 0].item() == 0
    for branch in range(1, result.selected.shape[1]):
        if int(result.branch_k[0, branch]) >= 2:
            assert 0 <= int(result.selected[0, branch]) < result.probabilities.shape[-1]


def test_model_regularizers_reach_scorer_and_keep_scalar_parity(model_factory):
    model = model_factory(seed=11)
    observation = ObservationBatch.dummy(1)
    first = CandidateSet.from_tuples(
        [move_action(0, index, index % 3) for index in range(5)]
    )
    second = CandidateSet.from_tuples(
        [move_action(1, index, index) for index in range(3)]
    )
    request = make_request(observation, (first, second))
    candidates = make_candidate_batch([request])
    selected = torch.zeros_like(candidates.selected)

    encoded, evaluation = model.evaluate(observation, candidates, selected=selected)
    assert evaluation.request_entropy.requires_grad
    assert evaluation.request_uniform_kl.requires_grad

    # Independent float64 reference for the scalar values: the fix must not
    # change any reported number, only the gradient path.
    logits = evaluation.logits[0, 0].detach()
    mask = candidates.mask[0, 0]
    _, _, entropy_ref, kl_ref = _reference_stats(logits, mask)
    assert math.isclose(
        float(evaluation.entropy_normalized[0, 0].detach()),
        float(entropy_ref),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(evaluation.uniform_kl_normalized[0, 0].detach()),
        float(kl_ref),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )

    # The learner-style objective must produce a nonzero scorer gradient.
    regularizer = (
        evaluation.request_entropy.sum() + evaluation.request_uniform_kl.sum()
    )
    regularizer.backward()
    scorer_grad = 0.0
    for parameter in model.scorer.parameters():
        if parameter.grad is not None:
            scorer_grad += float(parameter.grad.abs().sum())
    assert scorer_grad > 1e-6, "entropy/KL regularizers must reach the scorer"

    # The value head is untouched by the regularizer-only backward pass.
    assert model.value_head[0].weight.grad is None or float(
        model.value_head[0].weight.grad.abs().sum()
    ) == 0.0


def test_ppo_learner_loss_keeps_regularizer_graph(model_factory):
    """`_forward_terms` must keep the entropy/KL terms attached to the graph."""
    import dataclasses

    from agent.ppo.learner import PPOConfig, PPOLearner

    model = model_factory()
    config = dataclasses.replace(
        PPOConfig(), entropy_coefficient=1.0, uniform_kl_coefficient=1.0
    )
    learner = PPOLearner(model, config, device=torch.device("cpu"))
    observation = ObservationBatch.dummy(1)
    first = CandidateSet.from_tuples(
        [move_action(0, index, index % 3) for index in range(4)]
    )
    second = CandidateSet.from_tuples([move_action(1, 0), move_action(1, 1)])
    request = make_request(observation, (first, second))
    candidates = make_candidate_batch([request])

    from agent.buffer.rollout_buffer import RolloutBatch

    batch = RolloutBatch(
        observation=observation,
        candidates=candidates,
        old_logprob=torch.zeros(1),
        values=torch.zeros(1),
        raw_advantages=torch.zeros(1),
        returns=torch.zeros(1),
        rewards=torch.zeros(1),
        dones=torch.ones(1, dtype=torch.bool),
        row_valid=torch.ones(1, dtype=torch.bool),
        actor_mask=torch.ones(1, dtype=torch.bool),
        match_ids=torch.zeros(1, dtype=torch.long),
        sides=torch.zeros(1, dtype=torch.long),
        request_index=torch.zeros(1, dtype=torch.long),
        turns=torch.zeros(1, dtype=torch.long),
        request_kind=torch.zeros(1, dtype=torch.long),
        policy_ids=["current"],
        advantages=torch.zeros(1),
    )
    terms = learner._forward_terms(batch)
    assert terms["loss_unscaled"].requires_grad
    terms["loss_unscaled"].backward()
    scorer_grad = 0.0
    for parameter in model.scorer.parameters():
        if parameter.grad is not None:
            scorer_grad += float(parameter.grad.abs().sum())
    assert scorer_grad > 1e-6
