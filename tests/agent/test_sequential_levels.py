"""The sequential-level sampler must agree with the learner's recomputation.

The native engine's branch mask is prefix-dependent (a preview cannot pick the
same member twice; a used Mega flag disappears), so the collector samples level
by level with the table the engine returned *after* the previous selection.
The learner later recomputes the same request from the stored per-level tables,
and the PPO ratio must be exactly 1 before the first update.
"""

import torch

from agent.model.config import PA3Config
from agent.model.pa3_model import PA3Model
from agent.types.observation import ObservationBatch
from agent.types.requests import BranchCandidatesBatch


def level_table(batch, candidates, device="cpu"):
    """One level's table: `candidates` is [B][P] of 6-field action tuples."""
    capacity = max(len(row) for row in candidates)
    action_ids = torch.zeros((batch, 1, capacity, 6), dtype=torch.long)
    mask = torch.zeros((batch, 1, capacity), dtype=torch.bool)
    entity = torch.zeros((batch, 1, capacity), dtype=torch.long)
    move = torch.full((batch, 1, capacity), -1, dtype=torch.long)
    for index, row in enumerate(candidates):
        for slot, action in enumerate(row):
            action_ids[index, 0, slot] = torch.tensor(action, dtype=torch.long)
            mask[index, 0, slot] = True
            entity[index, 0, slot] = 0
    return BranchCandidatesBatch(
        action_ids=action_ids.to(device),
        mask=mask.to(device),
        entity_token=entity.to(device),
        move_token=move.to(device),
        branch_valid=torch.ones((batch, 1), dtype=torch.bool, device=device),
        selected=torch.full((batch, 1), -1, dtype=torch.long, device=device),
    )


def stack_levels(levels):
    """Concatenate single-branch tables along the branch dimension with padding."""
    capacity = max(level.action_ids.shape[2] for level in levels)

    def pad(tensor, fill):
        shape = (tensor.shape[0], tensor.shape[1], capacity, *tensor.shape[3:])
        out = tensor.new_full(shape, fill)
        out[:, :, : tensor.shape[2]] = tensor
        return out

    return BranchCandidatesBatch(
        action_ids=torch.cat([pad(level.action_ids, 0) for level in levels], dim=1),
        mask=torch.cat([pad(level.mask, False) for level in levels], dim=1),
        entity_token=torch.cat([pad(level.entity_token, 0) for level in levels], dim=1),
        move_token=torch.cat([pad(level.move_token, -1) for level in levels], dim=1),
        branch_valid=torch.cat([level.branch_valid for level in levels], dim=1),
        selected=torch.cat([level.selected for level in levels], dim=1),
    )


def test_prefix_dependent_levels_recompute_to_the_sampled_logprob():
    torch.manual_seed(7)
    config = PA3Config()
    model = PA3Model(config).eval()
    observations = ObservationBatch.dummy(batch_size=3, layout=config.observation_layout())
    with torch.no_grad():
        encoded = model.encode(observations)
    # Level 0 offers two picks per row; level 1's legal set is a *different*
    # set per row (as if the first pick removed itself from the pool).
    level0 = level_table(3, [
        [(0, 0, 255, 0, 0, 0), (0, 0, 255, 0, 1, 0)],
        [(0, 0, 255, 0, 2, 0), (0, 0, 255, 0, 3, 0)],
        [(0, 0, 255, 0, 4, 0), (0, 0, 255, 0, 5, 0)],
    ])
    level1 = level_table(3, [
        [(0, 1, 255, 0, 1, 0), (0, 1, 255, 0, 2, 0), (0, 1, 255, 0, 3, 0)],
        [(0, 1, 255, 0, 0, 0), (0, 1, 255, 0, 2, 0)],
        [(0, 1, 255, 0, 0, 0), (0, 1, 255, 0, 1, 0), (0, 1, 255, 0, 2, 0), (0, 1, 255, 0, 3, 0)],
    ])
    generator = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        sampled = model.sample_levels(encoded, [level0, level1], generator=generator)
        combined = stack_levels([level0, level1])
        recomputed = model.evaluate_encoded(
            encoded, combined, selected=sampled.selected
        )
    assert torch.allclose(
        recomputed.request_logprob, sampled.request_logprob, atol=1e-5
    )
    # Every sampled index must be inside its own level's legal mask.
    for level, table in enumerate([level0, level1]):
        picks = sampled.selected[:, level]
        legal = table.mask[:, 0].gather(1, picks.unsqueeze(-1)).squeeze(-1)
        assert bool(legal.all())
    # A request whose levels are all singletons is not an actor row.
    assert sampled.branch_k.shape == (3, 2)


def test_illegal_candidates_get_zero_probability_at_every_level():
    torch.manual_seed(13)
    config = PA3Config()
    model = PA3Model(config).eval()
    observations = ObservationBatch.dummy(batch_size=2, layout=config.observation_layout())
    with torch.no_grad():
        encoded = model.encode(observations)
    table = level_table(2, [
        [(0, 0, 255, 0, 0, 0), (0, 0, 255, 0, 1, 0), (0, 0, 255, 0, 2, 0)],
        [(0, 0, 255, 0, 3, 0), (0, 0, 255, 0, 4, 0)],
    ])
    table.mask[0, 0, 2] = False
    table.mask[1, 0, 1] = False
    generator = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():
        sampled = model.sample_levels(encoded, [table], generator=generator)
        combined = table
        evaluation = model.evaluate_encoded(encoded, combined, selected=sampled.selected)
    probabilities = torch.exp(evaluation.logits.new_zeros(evaluation.logits.shape))
    # The sampled pick can never be the masked candidate.
    assert int(sampled.selected[0, 0]) != 2
    assert int(sampled.selected[1, 0]) != 1
    recomputed = model.evaluate_encoded(encoded, combined, selected=sampled.selected)
    assert torch.allclose(recomputed.request_logprob, sampled.request_logprob, atol=1e-5)
