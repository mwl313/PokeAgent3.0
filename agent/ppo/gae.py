"""Generalized advantage estimation for request-level, terminal-only rewards.

One reward is placed at the end of each side's decision trajectory inside a
match (win +1 / loss -1 / draw 0).  Nothing is paid per slot, per target or per
opposing decision.  GAE follows that same side's request sequence with
``gamma = 1.0`` and ``gae_lambda = 0.95``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    bootstrap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for one ordered side-trajectory.

    ``rewards``/``values``/``dones`` are 1-D tensors in temporal order.
    ``dones[t]`` marks the last request of the trajectory.  A non-terminal
    trailing row is bootstrapped with ``bootstrap``.
    """
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values and dones must share a shape")
    if rewards.dim() != 1:
        raise ValueError("compute_gae expects 1-D trajectories")
    length = rewards.shape[0]
    advantages = torch.zeros_like(values)
    next_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
    next_value = torch.as_tensor(bootstrap, dtype=values.dtype, device=values.device)
    for index in range(length - 1, -1, -1):
        terminal = bool(dones[index])
        if terminal:
            next_value_t = torch.zeros((), dtype=values.dtype, device=values.device)
        else:
            next_value_t = next_value
        delta = rewards[index] + gamma * next_value_t - values[index]
        next_advantage = delta + gamma * gae_lambda * (0.0 if terminal else 1.0) * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    returns = advantages + values
    return advantages, returns


def compute_gae_for_rows(
    rows: Sequence,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    bootstrap_by_sequence: Optional[dict] = None,
) -> None:
    """Fill ``advantage`` and ``return_`` on rollout rows, grouped by trajectory.

    Each trajectory is ``(match_id, side)``: a GAE chain never crosses sides,
    and an opponent decision never contributes a reward or a bootstrap to the
    learner's chain.
    """
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row.match_id, row.side), []).append(row)

    bootstrap_by_sequence = bootstrap_by_sequence or {}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda r: r.request_index)
        rewards = torch.tensor([float(r.reward) for r in ordered], dtype=torch.float32)
        values = torch.tensor([float(r.value) for r in ordered], dtype=torch.float32)
        dones = torch.tensor([bool(r.done) for r in ordered], dtype=torch.bool)
        bootstrap = float(bootstrap_by_sequence.get(key, 0.0))
        advantages, returns = compute_gae(
            rewards, values, dones, gamma=gamma, gae_lambda=gae_lambda, bootstrap=bootstrap
        )
        for row, advantage, ret in zip(ordered, advantages.tolist(), returns.tolist()):
            row.advantage = float(advantage)
            row.return_ = float(ret)


def iterate_side_sequences(rows: Iterable) -> dict[tuple, list]:
    """Group rows into ``(match_id, side)`` trajectories (debug/test helper)."""
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row.match_id, row.side), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: r.request_index)
    return groups
