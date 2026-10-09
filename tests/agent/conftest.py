"""Test-local setup for the PA3 learner scaffolding.

The package lives at the repository root; ``tests/agent`` only adds the root to
``sys.path`` so ``python -m pytest tests/agent`` works from a clean checkout
without installing the project.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")  # noqa: F841 - tests are torch-only


@pytest.fixture(scope="session")
def small_config():
    """Architecture-preserving miniature used by all fast tests."""
    from agent.model import PA3Config

    return PA3Config(
        encoder_layers=2,
        d_model=64,
        attention_heads=2,
        head_dim=32,
        ffn_dim=128,
        category_vocab=256,
        category_embed_dim=8,
        role_vocab=12,
        prefix_hidden=64,
        critic_hidden=64,
    )


@pytest.fixture()
def model_factory(small_config):
    from agent.model import build_model

    def _build(seed: int | None = None, config=None):
        config = config or small_config
        if seed is None:
            return build_model(config)
        import dataclasses

        return build_model(dataclasses.replace(config, seed=seed))

    return _build
