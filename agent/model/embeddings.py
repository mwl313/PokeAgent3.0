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
from agent.model.ragged_embeddings import RaggedEmbedding


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
        numeric_dim = 2 * (config.float_slots + config.flag_slots)
        self.numeric_projection = nn.Linear(numeric_dim, d)
        self.category_projection = nn.Linear(
            config.category_slots * config.category_embed_dim, d
        )
        self.normalization = nn.LayerNorm(d)
        self.ragged_embedding = RaggedEmbedding(config)
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
        observation.validate_semantics()
        token_mask = observation.token_mask
        role_ids = observation.role_ids
        side_ids = observation.side_ids
        if bool(((role_ids<0)|(role_ids>=cfg.role_vocab)).any() | ((side_ids<0)|(side_ids>=cfg.side_vocab)).any()):
            raise ValueError("invalid role/side category ID")

        # Continuous features: mask unknown entries and flush non-finite values.
        floats = observation.floats
        float_known = observation.float_known
        # validate_semantics just checked known floats on this exact batch.
        floats = torch.where(float_known,floats,0.)

        flags = observation.flags.to(floats.dtype) * observation.flag_known.to(floats.dtype)
        numeric = torch.cat([floats, flags, float_known.to(floats.dtype), observation.flag_known.to(floats.dtype)], dim=-1)

        categories = torch.where(observation.category_known,observation.categories,0)
        if bool(((categories<0)|(categories>=cfg.category_vocab)).any()):
            raise ValueError("invalid known category ID; clamping is forbidden")
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
            + self.ragged_embedding(observation)
        )
        hidden = self.normalization(hidden)
        hidden = hidden * token_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden
