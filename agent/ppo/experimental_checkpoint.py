"""Opt-in runtime state helpers for bounded PPO continuation experiments.

The production launcher does not call this module. It supplies only the
process RNG portion that can be restored independently of the learner state;
engine handles and in-flight collector state still require an iteration
boundary checkpoint.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional

import torch


SCHEMA_VERSION = 1


def capture_rng_state(generators: Optional[Iterable[torch.Generator]] = None) -> dict:
    """Capture Python, CPU/CUDA torch, and explicitly owned generators."""
    state = {
        "schema_version": SCHEMA_VERSION,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "generators": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = None
    if generators is not None:
        state["generators"] = [generator.get_state().clone() for generator in generators]
    return state


def restore_rng_state(state: dict, generators: Optional[Iterable[torch.Generator]] = None) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""
    if int(state.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("unsupported experimental RNG checkpoint schema")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
    if generators is not None:
        owned = list(generators)
        saved = state.get("generators", [])
        if len(owned) != len(saved):
            raise ValueError("owned generator count does not match checkpoint")
        for generator, value in zip(owned, saved):
            generator.set_state(value)
