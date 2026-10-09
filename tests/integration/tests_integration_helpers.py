"""Deterministic fixture shared by the integration tests."""

from __future__ import annotations

import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import PA3Config, build_model


def small_config() -> PA3Config:
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


def collect_rows(seed: int = 20261009):
    torch.manual_seed(seed)
    model = build_model(small_config())
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, model, envs=12, target_matches=24)
    if len(buffer.rows) < 40:
        raise RuntimeError(f"fixture too small: {len(buffer.rows)} rows")
    return buffer
