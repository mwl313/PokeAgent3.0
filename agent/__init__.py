"""PokeAgent 3.0 learner package.

This package contains the PA3-8M policy/value model, the branch-structured
action scorer, the rollout buffer and the PPO learner scaffolding.  It is
deliberately engine independent: every entry point consumes the typed contract
in :mod:`agent.types` and never imports the Rust engine, Showdown, or any
external service.

Scaffolding only.  No real RL training is started from this package; the
`agent.mock_engine` module provides fake observations, fake action masks and
fake rollouts so that the learner can be exercised without a completed native
engine.
"""

from agent.model.config import PA3Config
from agent.ppo.config import PPOConfig

__all__ = ["PA3Config", "PPOConfig", "MODEL_NAME"]

MODEL_NAME = "PA3-8M"
