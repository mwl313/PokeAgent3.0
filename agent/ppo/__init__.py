"""PPO learner scaffolding (no real training is started here)."""

from agent.ppo.config import PPOConfig
from agent.ppo.gae import compute_gae, compute_gae_for_rows
from agent.ppo.learner import PPOLearner, UpdateReport
from agent.ppo.losses import (
    approx_kl,
    normalize_advantages,
    policy_loss,
    value_loss,
)
from agent.ppo.schedule import MatchClockScheduler

__all__ = [
    "PPOConfig",
    "PPOLearner",
    "UpdateReport",
    "MatchClockScheduler",
    "compute_gae",
    "compute_gae_for_rows",
    "approx_kl",
    "normalize_advantages",
    "policy_loss",
    "value_loss",
]
