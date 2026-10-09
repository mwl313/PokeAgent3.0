"""Cached prefix keys and device-side guards preserve the legacy model math."""

from __future__ import annotations

import copy

import pytest
import torch

from agent.model.encoder import _sdpa_math
from agent.types.observation import ObservationBatch

from pa3_test_util import make_candidate_batch, make_request, normal_branches, preview_branches


def _legacy_branches(model, encoded, candidates, selected=None, generator=None):
    """Independent pre-optimization loop: rebuild each selected key for GRU."""
    scorer = model.scorer
    tokens = encoded.tokens
    hidden = scorer.initial_state(tokens.shape[0], tokens.device, tokens.dtype)
    logits_out, logs, entropies, kls, picks, probabilities = [], [], [], [], [], []
    for level in range(candidates.action_ids.shape[1]):
        action = candidates.action_ids[:, level]
        mask = candidates.mask[:, level]
        entity = candidates.entity_token[:, level]
        move = candidates.move_token[:, level]
        valid = candidates.branch_valid[:, level]
        logits = scorer.score(tokens, hidden, action, mask, entity, move)
        log_prob, probs, entropy, kl = model._branch_stats(logits, mask)
        if selected is None:
            any_legal = mask.any(-1, keepdim=True)
            safe_probs = torch.where(any_legal, probs, torch.zeros_like(probs))
            no_legal = ~any_legal.squeeze(-1)
            if bool(no_legal.any()):
                safe_probs = safe_probs.clone()
                safe_probs[no_legal, 0] = 1.0
            pick = torch.multinomial(safe_probs, 1, generator=generator).squeeze(-1)
            pick = torch.where(valid, pick, torch.zeros_like(pick))
        else:
            pick = selected[:, level].clamp(0, mask.shape[-1] - 1)
        log = log_prob.gather(1, pick[:, None]).squeeze(-1)
        logs.append(torch.where(valid, log, 0.0))
        entropies.append(torch.where(valid, entropy, 0.0))
        kls.append(torch.where(valid, kl, 0.0))
        logits_out.append(logits)
        picks.append(pick)
        probabilities.append(probs)
        chosen_action = action.gather(1, pick[:, None, None].expand(-1, 1, 6)).squeeze(1)
        chosen_entity = entity.gather(1, pick[:, None]).squeeze(-1)
        chosen_move = move.gather(1, pick[:, None]).squeeze(-1)
        updated = scorer.advance(tokens, hidden, chosen_action, chosen_entity, chosen_move)
        hidden = torch.where(valid[:, None], updated, hidden)
    logs = torch.stack(logs, 1)
    entropies = torch.stack(entropies, 1)
    kls = torch.stack(kls, 1)
    active = candidates.branch_valid & (candidates.branch_k >= 2)
    denominator = active.sum(-1).clamp_min(1)
    return {
        "logits": torch.stack(logits_out, 1),
        "logprob_selected": logs,
        "entropy_normalized": entropies,
        "uniform_kl_normalized": kls,
        "request_logprob": logs.sum(-1),
        "request_entropy": (entropies * active).sum(-1) / denominator,
        "request_uniform_kl": (kls * active).sum(-1) / denominator,
        "hidden": hidden,
        "selected": torch.stack(picks, 1),
        "probabilities": torch.stack(probabilities, 1),
    }


def _mixed_batch():
    observation = ObservationBatch.dummy(2, seed=41)
    candidates = make_candidate_batch([
        make_request(observation.select([0]), preview_branches()),
        make_request(observation.select([1]), normal_branches(first_mask=[True, False, True])),
    ])
    selected = torch.tensor([[5, 3, 2, 1], [2, 1, -1, -1]])
    return observation, candidates, selected


def test_cached_prefix_keys_match_legacy_outputs_and_all_gradients(model_factory):
    optimized = model_factory(seed=17)
    reference = copy.deepcopy(optimized)
    observation, candidates, selected = _mixed_batch()
    encoded = optimized.encode(observation)
    evaluation, hidden, probs, picks = optimized._run_branches(encoded, candidates, selected)
    reference_encoded = reference.encode(observation)
    legacy = _legacy_branches(reference, reference_encoded, candidates, selected)
    for name in ("logits", "logprob_selected", "entropy_normalized", "uniform_kl_normalized",
                 "request_logprob", "request_entropy", "request_uniform_kl"):
        torch.testing.assert_close(getattr(evaluation, name), legacy[name], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(hidden, legacy["hidden"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(probs, legacy["probabilities"], atol=2e-6, rtol=2e-5)
    assert torch.equal(picks, legacy["selected"])
    loss = (-evaluation.request_logprob + 0.2 * evaluation.request_uniform_kl
            - 0.3 * evaluation.request_entropy + optimized.value(encoded).square()).mean()
    reference_loss = (-legacy["request_logprob"] + 0.2 * legacy["request_uniform_kl"]
                      - 0.3 * legacy["request_entropy"]
                      + reference.value(reference_encoded).square()).mean()
    loss.backward()
    reference_loss.backward()
    for (name, parameter), (other_name, other) in zip(
        optimized.named_parameters(), reference.named_parameters()
    ):
        assert name == other_name
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=5e-4, msg=name)


def test_sampling_cached_keys_preserves_seeded_actions_and_padded_branches(model_factory):
    model = model_factory(seed=29).eval()
    observation, candidates, _ = _mixed_batch()
    with torch.no_grad():
        legacy = _legacy_branches(model, model.encode(observation), candidates,
                                  generator=torch.Generator().manual_seed(7))
        sampled = model.sample(observation, candidates, generator=torch.Generator().manual_seed(7))
    assert torch.equal(sampled.selected, legacy["selected"])
    torch.testing.assert_close(sampled.request_logprob, legacy["request_logprob"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(sampled.probabilities, legacy["probabilities"], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("global_mask", [[True, True], [False, True], [False, False]])
def test_device_global_guard_preserves_batchwide_fallback(model_factory, global_mask):
    model = model_factory(seed=43)
    observation = ObservationBatch.dummy(2, seed=8)
    observation.token_mask[:, observation.layout.GLOBAL] = torch.tensor(global_mask)
    encoded = model.encode(observation)
    if any(global_mask):
        expected = encoded.tokens[:, observation.layout.GLOBAL]
    else:
        weights = observation.token_mask[..., None].to(encoded.tokens.dtype)
        expected = (encoded.tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
    torch.testing.assert_close(encoded.global_repr, expected, atol=0, rtol=0)


def test_math_attention_device_guard_keeps_empty_rows_finite():
    torch.manual_seed(5)
    query, key, value = [torch.randn(2, 2, 3, 4) for _ in range(3)]
    valid = torch.tensor([[True, False, True], [False, False, False]])
    actual = _sdpa_math(query, key, value, valid)
    scores = query @ key.masked_fill(~valid[:, None, :, None], 0.0).transpose(-2, -1) / 2
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    scores[1] = 0
    expected = scores.softmax(-1) @ value
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
