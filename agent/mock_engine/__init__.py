"""Fake engine/rollout data for model and PPO development.

Nothing here is a real battle engine: observations, action masks and matches
are synthetic and deterministic.  Its only purpose is to exercise the typed
contract, the branch policy, the rollout buffer and the PPO learner without a
completed ``NativeEngine``.
"""

from agent.mock_engine.mock_engine import (
    MockNativeEngine,
    MockRolloutCollector,
    collect_mock_rollout,
)

__all__ = ["MockNativeEngine", "MockRolloutCollector", "collect_mock_rollout"]
