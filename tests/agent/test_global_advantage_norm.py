"""D0: advantages are normalised over the global actor rows, not per rank.

The cross-rank reducer is injected, so the exact production code path
(``normalize_advantages_global``) is exercised against a single-process
reference on the concatenation of uneven shards, including an actor-free rank.
"""

from __future__ import annotations

import torch

from agent.ppo.ddp import normalize_advantages_global
from agent.ppo.losses import normalize_advantages


def test_global_normalization_matches_single_process_reference():
    torch.manual_seed(7)
    raw = torch.randn(23)
    actor = torch.rand(23) > 0.3
    reference = normalize_advantages(raw, actor_mask=actor)

    # Uneven shards: 10 / 0 / 13 rows, the middle rank holding no rows at all.
    shards = [slice(0, 10), slice(10, 10), slice(10, 23)]

    # all-reduce(SUM) returns the same global aggregate on every rank.
    global_count = float(actor.sum().item())
    global_sum = float(raw[actor].double().sum().item())
    global_mean = global_sum / global_count
    global_variance_sum = float(((raw[actor].double() - global_mean) ** 2).sum().item())

    def reduce(value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 2:
            return torch.tensor([global_count, global_sum], dtype=torch.float64)
        return torch.tensor([global_variance_sum], dtype=torch.float64)

    combined = torch.zeros_like(raw)
    for shard in shards:
        combined[shard] = normalize_advantages_global(
            raw[shard], actor[shard], reduce=reduce
        )
    assert torch.allclose(combined, reference, atol=1e-5, rtol=1e-5)


def test_actor_free_rank_participates_and_returns_empty():
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
    actor = torch.ones(4, dtype=torch.bool)
    calls = []

    def reduce(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.numel())
        if value.numel() == 2:
            return torch.tensor([4.0, float(raw.double().sum())], dtype=torch.float64)
        return torch.tensor([2.5], dtype=torch.float64)

    result = normalize_advantages_global(raw[:0], actor[:0], reduce=reduce)
    assert result.numel() == 0
    assert calls == [2, 1], "the empty rank must still join both all-reduces"


def test_single_actor_row_is_left_unnormalized():
    raw = torch.tensor([5.0])
    actor = torch.tensor([True])
    assert torch.equal(normalize_advantages_global(raw, actor), raw)


def test_zero_variance_returns_centered_values():
    raw = torch.tensor([3.0, 3.0, 3.0])
    actor = torch.ones(3, dtype=torch.bool)
    centered = normalize_advantages_global(raw, actor, std_floor=1e-8)
    assert torch.allclose(centered, torch.zeros(3), atol=1e-6)
