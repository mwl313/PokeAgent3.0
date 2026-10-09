"""Six-layer pre-LayerNorm non-causal Transformer encoder.

``nn.MultiheadAttention`` is deliberately not used: the project pins
``attention_backend: pytorch_sdpa_math`` with an explicit
``matmul_softmax`` fallback, a fixed ``head_dim`` of 64 and no attention
dropout.  The block below implements exactly that, with ``is_causal=False``
and a boolean key mask whose padded rows can never produce NaN.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from agent.model.config import PA3Config
from agent.model.embeddings import TokenEmbedding
from agent.types.observation import ObservationBatch


def _sdpa_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention in math-decomposition mode.

    ``valid`` is a boolean ``[B, T_k]`` tensor where ``True`` means the key
    participates.  A row whose every key is masked keeps key 0 so the softmax
    denominator is never zero; the caller zeroes padded outputs afterwards.
    """
    key = key.masked_fill(~valid.unsqueeze(1).unsqueeze(-1), 0.0)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~valid.unsqueeze(1).unsqueeze(2), float("-inf"))
    # Rows with no valid key at all: leave the (zeroed) first key so that the
    # softmax is finite; padded outputs are discarded by the block.
    empty_rows = ~valid.any(dim=-1)
    if bool(empty_rows.any()):
        scores = torch.where(
            empty_rows.view(-1, 1, 1, 1),
            torch.zeros_like(scores),
            scores,
        )
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


class EncoderBlock(nn.Module):
    """Pre-LayerNorm attention + FFN block (GELU, dropout 0)."""

    def __init__(self, config: PA3Config) -> None:
        super().__init__()
        if config.attention_heads * config.head_dim != config.d_model:
            raise ValueError("head configuration does not match d_model")
        self.config = config
        self.heads = config.attention_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_projection = nn.Linear(config.d_model, config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.qkv, self.out_projection):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for layer in (self.ffn[0], self.ffn[2]):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _attention(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        query, key, value = qkv[0], qkv[1], qkv[2]
        try:  # pragma: no cover - exercised on any supported torch
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=valid[:, None, None, :],
                dropout_p=0.0,
                is_causal=False,
            )
        except TypeError:  # pragma: no cover - very old torch fallback
            attended = _sdpa_math(query, key, value, valid)
        attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
        return self.out_projection(attended)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        attended = self._attention(self.norm1(hidden), valid)
        hidden = hidden + attended
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden


class TokenEncoder(nn.Module):
    """Embedding + ``encoder_layers`` pre-LN blocks + final LayerNorm."""

    def __init__(self, config: PA3Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self._forward_calls = 0

    @property
    def forward_calls(self) -> int:
        """Diagnostic counter used to assert "encode state once per request"."""
        return self._forward_calls

    def reset_counter(self) -> None:
        self._forward_calls = 0

    def forward(self, observation: ObservationBatch) -> torch.Tensor:
        self._forward_calls += 1
        hidden = self.embedding(observation)
        valid = observation.token_mask
        # Guarantee at least one participating key per row so the softmax is
        # finite; padded query rows are re-zeroed after every block.
        if valid.shape[1] > 0:
            valid = valid.clone()
            valid[:, 0] = valid[:, 0] | ~valid.any(dim=1)
        for block in self.blocks:
            hidden = block(hidden, valid)
        hidden = self.final_norm(hidden)
        hidden = hidden * observation.token_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden
