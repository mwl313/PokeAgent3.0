"""PA3-8M model package."""

from agent.model.config import PA3Config
from agent.model.pa3_model import (
    BranchEvaluation,
    EncodedState,
    PA3Model,
    SamplingResult,
)

__all__ = [
    "PA3Config",
    "PA3Model",
    "BranchEvaluation",
    "EncodedState",
    "SamplingResult",
    "build_model",
]


def build_model(config: PA3Config | None = None, device=None) -> PA3Model:
    """Randomly initialize PA3-8M under its configured seed.

    Seeding is scoped with ``fork_rng`` so that constructing a model never
    disturbs the caller's RNG stream.
    """
    import torch

    config = config or PA3Config()
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = PA3Model(config)
    if device is not None:
        model = model.to(device)
    return model
