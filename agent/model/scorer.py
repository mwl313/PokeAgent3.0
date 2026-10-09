"""Conditional action-prefix scorer.

The scorer is a small pointer network over the *encoded* observation tokens:

* a query is projected from the within-request prefix hidden state (a 320-unit
  ``GRUCell`` that only carries the selections made earlier in the same
  request, never match-level memory),
* a key is built from the referenced acting-Pokemon token, the referenced move
  token (a learned null vector for switches/picks/pass) and the six structured
  action fields,
* the candidate logit is the dot product of query and key.

Illegal candidates are masked before any sampling or log-probability
computation, so an invalid action has probability exactly zero.
"""

from __future__ import annotations

import torch
from torch import nn

from agent.model.config import PA3Config
from agent.types.actions import TARGET_LOCATION_MAX, TARGET_LOCATION_MIN

TARGET_MIN = TARGET_LOCATION_MIN
TARGET_MAX = TARGET_LOCATION_MAX

LOGPROB_FLOOR = torch.finfo(torch.float32).min
"""Log-probability recorded for masked candidates (``exp`` underflows to 0)."""


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax with exact zero probability for masked entries.

    Rows whose entire candidate set is masked (padded branches only) return a
    finite zero vector so that no NaN can leak into a loss; callers multiply
    those rows by their branch-valid mask.
    """
    masked = logits.masked_fill(~mask, float("-inf"))
    any_valid = mask.any(dim=-1, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    log_prob = torch.log_softmax(safe, dim=-1)
    return torch.where(mask, log_prob, torch.full_like(log_prob, LOGPROB_FLOOR))


def probabilities(log_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Probabilities from :func:`masked_log_softmax` (masked entries are 0)."""
    return torch.where(mask, log_prob.exp(), torch.zeros_like(log_prob))


class ConditionalPrefixScorer(nn.Module):
    def __init__(self, config: PA3Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model

        self.field_embed_dims = {
            "kind": 8,
            "own_slot": 8,
            "move_slot": 8,
            "target": 8,
            "destination": 8,
            "resource": 8,
        }
        self.kind_embedding = nn.Embedding(4, self.field_embed_dims["kind"])
        self.own_slot_embedding = nn.Embedding(256, self.field_embed_dims["own_slot"])
        self.move_slot_embedding = nn.Embedding(256, self.field_embed_dims["move_slot"])
        self.target_embedding = nn.Embedding(8, self.field_embed_dims["target"])
        self.destination_embedding = nn.Embedding(
            256, self.field_embed_dims["destination"]
        )
        self.resource_embedding = nn.Embedding(2, self.field_embed_dims["resource"])

        field_dim = sum(self.field_embed_dims.values())
        self.action_projection = nn.Linear(field_dim, d)
        self.entity_projection = nn.Linear(d, d)
        self.move_projection = nn.Linear(d, d)
        self.null_move = nn.Parameter(torch.zeros(d))
        self.key_norm = nn.LayerNorm(d)
        self.query_projection = nn.Linear(d, d)
        self.query_norm = nn.LayerNorm(d)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.zeros(1))
        self.gru = nn.GRUCell(d, d)
        self.initial_hidden = nn.Parameter(torch.zeros(d))
        self.reset_parameters()

    # -- initialization -------------------------------------------------
    def reset_parameters(self) -> None:
        cfg = self.config
        for embedding in (
            self.kind_embedding,
            self.own_slot_embedding,
            self.move_slot_embedding,
            self.target_embedding,
            self.destination_embedding,
            self.resource_embedding,
        ):
            nn.init.normal_(embedding.weight, std=cfg.embedding_init_std)
        for layer in (
            self.action_projection,
            self.entity_projection,
            self.move_projection,
            self.query_projection,
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for parameter in self.gru.parameters():
            if parameter.dim() >= 2:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        # Small policy output gain: the policy starts close to uniform.
        with torch.no_grad():
            self.query_projection.weight.mul_(cfg.policy_output_init_gain)
            self.logit_scale.fill_(1.0)
            # A non-degenerate initial prefix state: an all-zero hidden state
            # would make the first branch's logits exactly zero after the
            # query LayerNorm, so the first selection of every request would
            # start perfectly uniform with no learnable direction.
            nn.init.normal_(self.initial_hidden, std=cfg.embedding_init_std)
            self.null_move.zero_()
            self.key_norm.reset_parameters()
            self.query_norm.reset_parameters()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _field_ids(action_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        kind = action_ids[..., 0].clamp(0, 3)
        own_slot = action_ids[..., 1].clamp(0, 255)
        move_slot = action_ids[..., 2].clamp(0, 255)
        raw_target = action_ids[..., 3]
        target = (raw_target.clamp(TARGET_MIN, TARGET_MAX) + 1).clamp(0, 7)
        destination = action_ids[..., 4].clamp(0, 255)
        resource = action_ids[..., 5].clamp(0, 1)
        return kind, own_slot, move_slot, target, destination, resource

    def action_features(self, action_ids: torch.Tensor) -> torch.Tensor:
        """``[..., 6]`` integer action tuples -> ``[..., d_model]`` features."""
        kind, own_slot, move_slot, target, destination, resource = self._field_ids(
            action_ids
        )
        return self.action_projection(
            torch.cat(
                [
                    self.kind_embedding(kind),
                    self.own_slot_embedding(own_slot),
                    self.move_slot_embedding(move_slot),
                    self.target_embedding(target),
                    self.destination_embedding(destination),
                    self.resource_embedding(resource),
                ],
                dim=-1,
            )
        )

    def token_references(
        self,
        encoded_tokens: torch.Tensor,
        entity_token: torch.Tensor,
        move_token: torch.Tensor,
    ) -> torch.Tensor:
        """Gather entity/move references and project them into key space."""
        tokens, token_count, d = encoded_tokens.shape
        flat_entity = entity_token.reshape(tokens, -1).clamp(0, token_count - 1)
        entity_ref = torch.gather(
            encoded_tokens, 1, flat_entity.unsqueeze(-1).expand(-1, -1, d)
        )

        flat_move = move_token.reshape(tokens, -1)
        has_move = flat_move >= 0
        move_ref = torch.gather(
            encoded_tokens,
            1,
            flat_move.clamp(0, token_count - 1).unsqueeze(-1).expand(-1, -1, d),
        )
        move_ref = torch.where(
            has_move.unsqueeze(-1), move_ref, self.null_move.view(1, 1, d)
        )
        return self.entity_projection(entity_ref) + self.move_projection(move_ref)

    def build_keys(
        self,
        encoded_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        entity_token: torch.Tensor,
        move_token: torch.Tensor,
    ) -> torch.Tensor:
        shape = action_ids.shape[:-1]
        d = encoded_tokens.shape[-1]
        token_part = self.token_references(
            encoded_tokens, entity_token, move_token
        ).view(*shape, d)
        keys = token_part + self.action_features(action_ids)
        return self.key_norm(keys)

    def score(
        self,
        encoded_tokens: torch.Tensor,
        prefix_hidden: torch.Tensor,
        action_ids: torch.Tensor,
        mask: torch.Tensor,
        entity_token: torch.Tensor,
        move_token: torch.Tensor,
    ) -> torch.Tensor:
        """Candidate logits ``[B, P]`` for one branch (masked entries are -inf)."""
        keys = self.build_keys(encoded_tokens, action_ids, entity_token, move_token)
        return self.score_keys(keys, prefix_hidden, mask)

    def score_keys(
        self,
        keys: torch.Tensor,
        prefix_hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score prebuilt keys, which can also feed the selected-prefix GRU.

        Candidate keys depend only on the observation and action tuple, not on
        the prefix. Reusing the selected key avoids repeating its embeddings,
        token gathers, projections and LayerNorm after scoring the branch.
        """
        # The small policy output gain is applied last: normalizing the prefix
        # state first keeps the projection (and therefore the initial logits)
        # genuinely small, so the random policy starts near uniform.
        query = self.query_projection(self.query_norm(prefix_hidden))
        logits = (keys * query.unsqueeze(1)).sum(dim=-1) * self.logit_scale
        logits = logits + self.logit_bias
        return logits.masked_fill(~mask, float("-inf"))

    def advance(
        self,
        encoded_tokens: torch.Tensor,
        prefix_hidden: torch.Tensor,
        action_ids: torch.Tensor,
        entity_token: torch.Tensor,
        move_token: torch.Tensor,
    ) -> torch.Tensor:
        """GRU update from the selected candidate into the prefix hidden state."""
        keys = self.build_keys(encoded_tokens, action_ids, entity_token, move_token)
        return self.advance_keys(keys, prefix_hidden)

    def advance_keys(
        self, selected_keys: torch.Tensor, prefix_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Advance the prefix using a selected key from ``build_keys``."""
        return self.gru(selected_keys, prefix_hidden)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.initial_hidden.to(device=device, dtype=dtype).expand(batch_size, -1).contiguous()
