"""Token embedding for the PA3-8M entity transformer.

Each padded observation row becomes one ``d_model`` vector from

* a fixed token-role embedding (global/field/side/Pokemon/move/event),
* a side embedding (neutral/self/opponent),
* a projection of the normalized continuous features and boolean flags,
* a projection of the integer category features (shared value vocabulary).

Masked features contribute zero, and unknown category ids are replaced by 0
before the lookup so a stale payload can never index outside the vocabulary.
"""

from __future__ import annotations

import torch
from torch import nn

from agent.model.config import PA3Config
from agent.types.observation import ObservationBatch


class TokenEmbedding(nn.Module):
    def __init__(self, config: PA3Config) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.role_embedding = nn.Embedding(config.role_vocab, d)
        self.side_embedding = nn.Embedding(config.side_vocab, d)
        self.category_value_embedding = nn.Embedding(
            config.category_vocab, config.category_embed_dim
        )
        numeric_dim = config.float_slots + config.flag_slots
        self.numeric_projection = nn.Linear(numeric_dim, d)
        self.category_projection = nn.Linear(
            config.category_slots * config.category_embed_dim, d
        )
        self.normalization = nn.LayerNorm(d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        cfg = self.config
        nn.init.normal_(self.role_embedding.weight, std=cfg.embedding_init_std)
        nn.init.normal_(self.side_embedding.weight, std=cfg.embedding_init_std)
        nn.init.normal_(self.category_value_embedding.weight, std=cfg.embedding_init_std)
        for layer in (self.numeric_projection, self.category_projection):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, observation: ObservationBatch) -> torch.Tensor:
        cfg = self.config
        token_mask = observation.token_mask
        role_ids = observation.role_ids.clamp(0, cfg.role_vocab - 1)
        side_ids = observation.side_ids.clamp(0, cfg.side_vocab - 1)

        # Continuous features: mask unknown entries and flush non-finite values.
        floats = observation.floats
        float_known = observation.float_known
        floats = torch.nan_to_num(floats, nan=0.0, posinf=0.0, neginf=0.0)
        floats = floats * float_known.to(floats.dtype)

        flags = observation.flags.to(floats.dtype) * observation.flag_known.to(floats.dtype)
        numeric = torch.cat([floats, flags], dim=-1)

        categories = observation.categories.clamp(0, cfg.category_vocab - 1)
        categories = categories * observation.category_known.to(torch.long)
        category_values = self.category_value_embedding(categories)
        category_values = category_values * observation.category_known.unsqueeze(-1).to(
            category_values.dtype
        )
        category_values = category_values.flatten(-2, -1)

        hidden = (
            self.numeric_projection(numeric)
            + self.category_projection(category_values)
            + self.role_embedding(role_ids)
            + self.side_embedding(side_ids)
        )
        hidden = self.normalization(hidden)
        hidden = hidden * token_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden
