"""Removing empty candidate slots preserves the complete PPO objective."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.ppo import PPOLearner, PPOConfig


def _fixture(model_factory):
    model = model_factory(seed=20261009)
    buffer = collect_mock_rollout(
        MockNativeEngine(num_teams=4, requests_per_match=3), model,
        envs=3, target_matches=3,
    )
    learner = PPOLearner(
        model, PPOConfig(global_minibatch_size=16, microbatch_size=8),
        device="cpu", amp=False,
    )
    return learner, buffer, learner.prepare_batch(buffer)


@pytest.mark.parametrize("width", [6, 8, 16, 64])
@pytest.mark.parametrize("mixed_masks", [False, True])
def test_candidate_padding_preserves_objective_and_all_gradients(
    model_factory, width, mixed_masks,
):
    learner, buffer, batch = _fixture(model_factory)
    required = max(len(branch) for row in buffer.rows for branch in row.action_ids)
    assert required == 6
    if mixed_masks:
        valid = torch.ones_like(batch.row_valid)
        valid[::3] = False
        actor = batch.actor_mask.clone()
        actor[1::3] = False
        batch = replace(batch, row_valid=valid, actor_mask=actor)
    full_terms = learner._forward_terms(batch)
    full_loss = full_terms["loss_unscaled"] + learner.config.value_coefficient * full_terms["value_mean"]
    full_grads = torch.autograd.grad(full_loss, tuple(learner.model.parameters()), allow_unused=True)
    compact = replace(batch, candidates=batch.candidates.trim_padding(width))
    compact_terms = learner._forward_terms(compact)
    compact_loss = (
        compact_terms["loss_unscaled"]
        + learner.config.value_coefficient * compact_terms["value_mean"]
    )
    compact_grads = torch.autograd.grad(
        compact_loss, tuple(learner.model.parameters()), allow_unused=True,
    )
    assert full_terms.keys() == compact_terms.keys()
    for key in full_terms:
        torch.testing.assert_close(full_terms[key], compact_terms[key], rtol=2e-5, atol=2e-7)
    for (name, _), full, compact in zip(
        learner.model.named_parameters(), full_grads, compact_grads,
    ):
        assert (full is None) == (compact is None), name
        if full is not None:
            difference = float((full - compact).abs().max())
            scale = float(full.abs().max())
            assert difference <= 2e-5 * scale + 2e-7, (name, difference, scale)


def test_trim_is_a_view_preserving_branch_metadata_and_selected_prefix(model_factory):
    _, _, batch = _fixture(model_factory)
    full = batch.candidates
    trimmed = full.trim_padding(6)
    assert trimmed.shape == (len(batch), full.shape[1], 6)
    assert trimmed.branch_valid is full.branch_valid
    assert trimmed.selected is full.selected
    for name in ("action_ids", "mask", "entity_token", "move_token"):
        left, right = getattr(full, name), getattr(trimmed, name)
        assert right.data_ptr() == left.data_ptr()
        assert torch.equal(right, left[:, :, :6])
    assert torch.equal(trimmed.branch_k, full.branch_k)
    assert torch.equal(trimmed.actor_active, full.actor_active)


@pytest.mark.parametrize("invalid_width", [0, -1, 65, True, 6.0])
def test_trim_rejects_invalid_widths(model_factory, invalid_width):
    _, _, batch = _fixture(model_factory)
    with pytest.raises(ValueError, match="integer"):
        batch.candidates.trim_padding(invalid_width)


@pytest.mark.parametrize("field", ["mask", "action_ids", "entity_token", "move_token", "selected"])
def test_trim_refuses_legal_stored_or_selected_entries(model_factory, field):
    _, _, batch = _fixture(model_factory)
    candidates = batch.candidates
    tensor = getattr(candidates, field).clone()
    if field == "selected":
        tensor[0, 0] = 7
    elif field == "action_ids":
        tensor[0, 0, 7, 0] = 1
    else:
        tensor[0, 0, 7] = True if field == "mask" else 9
    candidates = replace(candidates, **{field: tensor})
    with pytest.raises(ValueError, match="discard"):
        candidates.trim_padding(6)


def test_trim_refuses_to_drop_existing_legal_candidates(model_factory):
    _, _, batch = _fixture(model_factory)
    with pytest.raises(ValueError, match="discard"):
        batch.candidates.trim_padding(5)
