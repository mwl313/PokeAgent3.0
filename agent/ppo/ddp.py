"""DDP objective adapter for the PPO learner (v3 G1).

The plan's §6.2 requirement: the policy/entropy/uniform-KL terms are normalised
by the **global valid actor rows** ``A = sum_r A_r`` and the value term by the
**global valid value rows** ``V = sum_r V_r``. PyTorch DDP averages gradients
over the world size after the backward pass, so each rank must scale its local
*sums* by ``world_size / denominator`` to make that average equal the
single-process global objective.

Keeping the math in this module (no torch.distributed import at module import
time) lets the parity tests exercise the exact equations on a single process.
"""

from __future__ import annotations

from typing import Optional

import torch


def ddp_rank_loss(
    actor_sum: torch.Tensor,
    value_sum: torch.Tensor,
    actor_total: float,
    value_total: float,
    world_size: int,
    value_coefficient: float,
) -> torch.Tensor:
    """Rank-local loss whose DDP gradient average equals the global objective.

    ``actor_sum`` is the rank's sum of ``policy - c_ent*H + c_kl*KL`` over its
    valid actor rows (each already a row mean times that row's weight), and
    ``value_sum`` the rank's sum of ``0.5*MSE`` over its valid value rows.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    actor = actor_sum / max(float(actor_total), 1.0)
    value = value_sum / max(float(value_total), 1.0)
    return float(world_size) * (actor + value_coefficient * value)


def all_reduce_sums(
    actor_sum: torch.Tensor,
    value_sum: torch.Tensor,
    actor_count: torch.Tensor,
    value_count: torch.Tensor,
    world_size: int,
    group=None,
) -> tuple[float, float]:
    """All-reduce the four local sums and return ``(A, V)`` as Python floats.

    One collective batch instead of four round trips: the four scalars are
    stacked and reduced together. With ``world_size == 1`` (or distributed not
    initialised) this is a plain host-side read.
    """
    stacked = torch.stack([
        actor_sum.detach().float(),
        value_sum.detach().float(),
        actor_count.detach().float(),
        value_count.detach().float(),
    ])
    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stacked, op=torch.distributed.ReduceOp.SUM, group=group)
    return float(stacked[2].item()), float(stacked[3].item())


def global_reference_loss(
    actor_sums: list[torch.Tensor],
    value_sums: list[torch.Tensor],
    actor_counts: list[float],
    value_counts: list[float],
    value_coefficient: float,
) -> torch.Tensor:
    """Single-process reference for the same shards (used by tests)."""
    actor = torch.stack([value.float() for value in actor_sums]).sum()
    value = torch.stack([value.float() for value in value_sums]).sum()
    actor_total = max(sum(actor_counts), 1.0)
    value_total = max(sum(value_counts), 1.0)
    return actor / actor_total + value_coefficient * value / value_total


class DDPCommunication:
    """no_sync scheduling helper: only the final microbatch synchronises.

    PyTorch's ``DDP.no_sync`` must wrap **both** the forward and the backward of
    every non-final microbatch; wrapping only the backward leaves the forward
    doing an unnecessary all-reduce (plan §6.1).
    """

    def __init__(self, model, synchronise_every: int) -> None:
        if synchronise_every < 1:
            raise ValueError("synchronise_every must be >= 1")
        self.model = model
        self.synchronise_every = int(synchronise_every)
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def context(self):
        """Context manager for the next microbatch forward+backward."""
        from contextlib import nullcontext

        self._count += 1
        if hasattr(self.model, "no_sync") and self._count < self.synchronise_every:
            return self.model.no_sync()
        return nullcontext()
