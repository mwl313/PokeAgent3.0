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

import math
from typing import Callable, Optional

import torch


def normalize_advantages_global(
    raw: torch.Tensor,
    actor_mask: torch.Tensor,
    reduce: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    std_floor: float = 1.0e-8,
) -> torch.Tensor:
    """Two-pass advantage normalization over the *global* actor rows.

    ``reduce`` performs the cross-rank all-reduce of a small float64 tensor
    (injectable so the exact code path is unit-testable without a process
    group). The result matches ``normalize_advantages`` (unbiased=False std)
    on the concatenation of every rank's actor rows.
    """
    raw = raw.float()
    actor = actor_mask.to(torch.bool)
    stats = torch.tensor(
        [float(actor.sum().item()), float(raw[actor].sum().item())], dtype=torch.float64
    )
    if reduce is not None:
        stats = reduce(stats).cpu().double()
    count, total = float(stats[0]), float(stats[1])
    if count <= 1:
        return raw
    mean = total / count
    variance = torch.tensor(
        [float(((raw[actor].double() - mean) ** 2).sum().item())], dtype=torch.float64
    )
    if reduce is not None:
        variance = reduce(variance).cpu().double()
    std = math.sqrt(max(float(variance[0]) / count, 0.0))
    if std < std_floor:
        return raw - mean
    return (raw - mean) / std


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


def all_reduce_tensor(value: torch.Tensor, op=None, group=None) -> torch.Tensor:
    """All-reduce a tensor in place when a process group is live, else identity."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            value, op=op or torch.distributed.ReduceOp.SUM, group=group
        )
    return value


def all_reduce_flag(value: torch.Tensor, group=None) -> torch.Tensor:
    """Global AND of a 0/1 flag tensor (ReduceOp.MIN)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            value, op=torch.distributed.ReduceOp.MIN, group=group
        )
    return value


def flatten_gradients(model) -> torch.Tensor:
    """Fixed name-order FP32 flat gradient buffer.

    Parameters without a gradient contribute exact zeros, so every parameter
    participates in the manual all-reduce (the plan's §7.1 requirement).
    """
    chunks = []
    for parameter in model.parameters():
        if parameter.grad is None:
            chunks.append(
                torch.zeros(parameter.numel(), dtype=torch.float32, device=parameter.device)
            )
        else:
            chunks.append(parameter.grad.detach().float().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def assign_flat_gradients(model, flat: torch.Tensor) -> None:
    """Write a flat gradient buffer back to ``parameter.grad`` in name order."""
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        chunk = flat[offset:offset + count].view_as(parameter).to(parameter.dtype)
        if parameter.grad is None:
            parameter.grad = chunk.clone()
        else:
            parameter.grad.copy_(chunk)
        offset += count
    if offset != flat.numel():
        raise ValueError(f"flat buffer has {flat.numel()} values, expected {offset}")


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

    The caller passes an explicit ``synchronize`` flag for the final micro step.
    The older counter form (``synchronize_every``) is kept only as a fallback:
    it silently skipped the last synchronisation whenever a rank ran fewer
    micro steps than the configured accumulation count, which is exactly the
    v3 collective-mismatch failure.
    """

    def __init__(self, model, synchronise_every: int = 0) -> None:
        if synchronise_every < 0:
            raise ValueError("synchronise_every must be >= 0")
        self.model = model
        self.synchronise_every = int(synchronise_every)
        self._count = 0
        self.sync_calls = 0
        self.no_sync_calls = 0

    def reset(self) -> None:
        self._count = 0

    def context(self, synchronize: Optional[bool] = None):
        """Context manager for the next microbatch forward+backward."""
        from contextlib import nullcontext

        self._count += 1
        if not hasattr(self.model, "no_sync"):
            self.sync_calls += 1
            return nullcontext()
        if synchronize is None:
            synchronize = self.synchronise_every > 0 and self._count >= self.synchronise_every
        if synchronize:
            self.sync_calls += 1
            return nullcontext()
        self.no_sync_calls += 1
        return self.model.no_sync()
