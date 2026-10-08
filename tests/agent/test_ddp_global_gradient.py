"""G1: DDP global-denominator objective parity (single-process simulation).

The two-rank NCCL path is exercised by the launcher smoke; this test proves the
*math* first: splitting a global batch into two uneven shards and applying the
per-rank loss from ``agent.ppo.ddp`` must reproduce the single-process global
objective and its gradient exactly (FP32 tolerance), including the case where
one rank holds zero actor rows.
"""

from __future__ import annotations

import torch

from agent.ppo.ddp import ddp_rank_loss, global_reference_loss


def _rank_terms(micro_terms):
    actor_sum = torch.stack([term[0] for term in micro_terms]).sum()
    value_sum = torch.stack([term[1] for term in micro_terms]).sum()
    actor_count = float(sum(term[2] for term in micro_terms))
    value_count = float(sum(term[3] for term in micro_terms))
    return actor_sum, value_sum, actor_count, value_count


def _shard(rows, actor_counts):
    class _Term:
        def __init__(self, row, actor):
            self.actor = torch.tensor(row, dtype=torch.float32, requires_grad=True)
            self.value = torch.tensor(row * 0.5, dtype=torch.float32, requires_grad=True)
            self.actor_count = actor
            self.value_count = 1.0

    return [_Term(row, actor) for row, actor in zip(rows, actor_counts)]


def test_uneven_shards_match_the_single_process_global_objective():
    torch.manual_seed(3)
    rows = 9
    weights = torch.randn(rows, 3, dtype=torch.float32)
    targets = torch.randn(rows, 3, dtype=torch.float32)

    def loss_fn(weight, target):
        return ((weight - target) ** 2).mean(dim=1)

    # Shard actor masks: rank A holds 5 actor rows of 6, rank B holds none
    # (the plan's actor=0 rank edge case). The global reference must use exactly
    # the same mask.
    shard_masks = [torch.tensor([True] * 5 + [False]), torch.tensor([False] * 3)]
    actor_mask = torch.cat(shard_masks)
    weight = weights.clone().requires_grad_(True)
    per_row = loss_fn(weight, targets)
    reference = (per_row[actor_mask].sum() / actor_mask.sum()) + 0.5 * per_row.mean()
    reference_grad = torch.autograd.grad(reference, weight)[0]

    # Two uneven ranks: 6 rows and 3 rows, with the second rank holding no
    # actor rows at all (the plan's actor=0 rank edge case).
    shard_rows = [(0, 6), (6, 9)]
    shard_actors = [(0, 6, shard_masks[0]), (6, 9, shard_masks[1])]
    rank_actor_sums, rank_value_sums, actor_counts, value_counts = [], [], [], []
    for start, stop, mask in shard_actors:
        shard_weight = weights[start:stop].clone().requires_grad_(True)
        shard_rows_loss = loss_fn(shard_weight, targets[start:stop])
        actor_sum = (shard_rows_loss * mask.float()).sum()
        value_sum = shard_rows_loss.sum()
        rank_actor_sums.append(actor_sum)
        rank_value_sums.append(value_sum)
        actor_counts.append(float(mask.sum()))
        value_counts.append(float(stop - start))

    actor_total = sum(actor_counts)
    value_total = sum(value_counts)
    assert actor_counts[1] == 0.0

    grads = []
    for index, (start, stop, mask) in enumerate(shard_actors):
        shard_weight = weights[start:stop].clone().requires_grad_(True)
        shard_loss = loss_fn(shard_weight, targets[start:stop])
        actor_sum = (shard_loss * mask.float()).sum()
        value_sum = shard_loss.sum()
        rank_loss = ddp_rank_loss(
            actor_sum, value_sum, actor_total, value_total, world_size=2, value_coefficient=0.5
        )
        grads.append(torch.autograd.grad(rank_loss, shard_weight)[0])
    averaged = torch.cat(grads, dim=0) / 2.0
    assert torch.allclose(averaged, reference_grad, atol=1e-6), (
        float((averaged - reference_grad).abs().max()),
    )

    # The reference helper agrees with the sharded sums.
    simulated = global_reference_loss(
        rank_actor_sums, rank_value_sums, actor_counts, value_counts, 0.5
    )
    assert torch.allclose(simulated, reference.detach(), atol=1e-6)


def test_double_weighting_is_not_applied():
    """A local mean times world_size must equal the global mean over shards."""
    actor_sum_a, value_sum_a = torch.tensor(2.0), torch.tensor(1.0)
    actor_sum_b, value_sum_b = torch.tensor(4.0), torch.tensor(3.0)
    loss_a = ddp_rank_loss(actor_sum_a, value_sum_a, 6.0, 200.0, 2, 0.5)
    loss_b = ddp_rank_loss(actor_sum_b, value_sum_b, 6.0, 200.0, 2, 0.5)
    averaged = (loss_a + loss_b) / 2.0
    global_loss = global_reference_loss(
        [actor_sum_a, actor_sum_b], [value_sum_a, value_sum_b], [2.0, 4.0], [80.0, 120.0], 0.5
    )
    assert torch.allclose(averaged, global_loss, atol=1e-6), (averaged, global_loss)
