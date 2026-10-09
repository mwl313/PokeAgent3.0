"""GAE for terminal-only, per-side rewards."""

from __future__ import annotations

import torch

from agent.ppo.gae import compute_gae

from pa3_test_util import synthetic_rows


def test_terminal_only_reward_decays_backwards_with_lambda():
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values = torch.zeros(4)
    dones = torch.tensor([False, False, False, True])
    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=0.95)
    expected = torch.tensor([0.95**3, 0.95**2, 0.95, 1.0])
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.allclose(returns, expected, atol=1e-6)


def test_gamma_one_lambda_one_gives_monte_carlo_return():
    rewards = torch.tensor([0.0, 0.0, -1.0])
    values = torch.zeros(3)
    dones = torch.tensor([False, False, True])
    advantages, _ = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([-1.0, -1.0, -1.0]), atol=1e-6)


def test_no_reward_before_terminal_and_no_double_payment():
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    values = torch.zeros(5)
    dones = torch.tensor([False, False, False, False, True])
    advantages, _ = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=0.95)
    # A one-time terminal reward is not multiplied by the number of requests.
    assert float(advantages.max()) == 1.0
    assert all(float(a) <= 1.0 + 1e-6 for a in advantages)
    assert float(advantages[-1]) == 1.0


def test_bootstrap_value_is_used_for_unfinished_trajectory():
    rewards = torch.tensor([0.0, 0.0])
    values = torch.tensor([0.0, 0.0])
    dones = torch.tensor([False, False])
    advantages, _ = compute_gae(
        rewards, values, dones, gamma=1.0, gae_lambda=0.95, bootstrap=0.5
    )
    assert abs(float(advantages[-1]) - 0.5) < 1e-6
    # GAE lambda discounts the bootstrap one step further back.
    assert abs(float(advantages[0]) - 0.95 * 0.5) < 1e-6


def test_buffer_gae_groups_by_match_and_side():
    buffer = synthetic_rows(
        count=4,
        match_ids=[0, 0, 1, 1],
        sides=[0, 1, 0, 1],
        values=[0.0, 0.0, 0.0, 0.0],
        rewards=[0.0, 1.0, -1.0, 0.0],
        dones=[False, True, True, False],
    )
    buffer.compute_gae(gamma=1.0, gae_lambda=0.95)
    # Each side's chain is independent: the P2 win does not pay the P1 rows.
    assert abs(float(buffer.rows[0].advantage)) < 1e-6
    assert abs(float(buffer.rows[1].advantage) - 1.0) < 1e-6
    assert abs(float(buffer.rows[2].advantage) + 1.0) < 1e-6
    assert abs(float(buffer.rows[3].advantage)) < 1e-6
