"""PPO objective terms.

All probability and loss arithmetic stays in FP32 even when the encoder runs
under FP16 autocast.
"""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_VALUE_COEFFICIENT = 0.5


def policy_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    actor_mask: Optional[torch.Tensor] = None,
    clip_epsilon: float = DEFAULT_CLIP_EPSILON,
) -> torch.Tensor:
    """``-mean(min(ratio*A, clip(ratio, 1±eps)*A))`` over actor rows."""
    ratio = ratio.float()
    advantages = advantages.float()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    objective = torch.minimum(unclipped, clipped)
    if actor_mask is not None:
        mask = actor_mask.to(objective.dtype)
        denominator = mask.sum().clamp_min(1.0)
        return -(objective * mask).sum() / denominator
    return -objective.mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """``0.5 * MSE(value, return)`` with no value clipping."""
    values = values.float()
    returns = returns.float()
    if values.numel() == 0:  # padding-only microbatch: no value contribution
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return 0.5 * torch.nn.functional.mse_loss(values, returns)


def approx_kl(ratio: torch.Tensor) -> torch.Tensor:
    """Schulman's approximately-KL estimator ``mean((r - 1) - log r)``."""
    ratio = ratio.float()
    return ((ratio - 1.0) - torch.log(ratio.clamp_min(1e-12))).mean()


def normalize_advantages(
    advantages: torch.Tensor,
    actor_mask: Optional[torch.Tensor] = None,
    std_floor: float = 1.0e-8,
) -> torch.Tensor:
    """Normalize over the iteration's valid actor rows (std floor 1e-8)."""
    advantages = advantages.float()
    if actor_mask is None:
        rows = advantages
    else:
        rows = advantages[actor_mask.to(torch.bool)]
    if rows.numel() <= 1:
        return advantages
    std = rows.std(unbiased=False)
    if float(std) < std_floor:
        return advantages - rows.mean()
    return (advantages - rows.mean()) / std
