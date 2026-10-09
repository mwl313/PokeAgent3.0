"""PA3-8M architecture configuration.

Values are the frozen architecture constants of ``configs/train.yaml``
(``model:`` block) and the Full Spec 1.1 section 6.  They are data so that a
test can assert the architecture without importing magic numbers, and so that
the eventual checkpoint records the exact architecture it was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.types.actions import BRANCH_CAPACITY
from agent.types.observation import ObservationLayout


@dataclass(frozen=True)
class PA3Config:
    """Frozen PA3-8M architecture."""

    name: str = "PA3-8M"
    initialization: str = "random"

    # Transformer encoder
    encoder_layers: int = 6
    d_model: int = 320
    attention_heads: int = 5
    head_dim: int = 64
    ffn_dim: int = 1280
    activation: str = "gelu"
    pre_layernorm: bool = True
    causal_attention: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Observation tokens
    tokens: int = ObservationLayout.TOKENS
    active_tokens: int = ObservationLayout.ACTIVE_TOKENS
    category_slots: int = ObservationLayout.CATEGORY_SLOTS
    float_slots: int = ObservationLayout.FLOAT_SLOTS
    flag_slots: int = ObservationLayout.FLAG_SLOTS
    category_vocab: int = 8192
    category_embed_dim: int = 16
    role_vocab: int = ObservationLayout.ROLE_VOCAB
    side_vocab: int = ObservationLayout.SIDE_VOCAB

    # Conditional action prefix scorer
    prefix_decoder: str = "GRUCell"
    prefix_hidden: int = 320
    max_request_branches: int = BRANCH_CAPACITY
    candidate_padding: int = 64

    # Value head (observation/global representation only)
    critic_hidden: int = 256
    critic_output: str = "linear_scalar"
    critic_input: str = "observation_only"

    # Initialization
    embedding_init_std: float = 0.02
    linear_init: str = "xavier"
    policy_output_init_gain: float = 0.01
    seed: int = 20261006

    extras: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.attention_heads * self.head_dim != self.d_model:
            raise ValueError(
                "attention_heads * head_dim must equal d_model "
                f"({self.attention_heads} * {self.head_dim} != {self.d_model})"
            )
        if self.pre_layernorm is not True:
            raise ValueError("PA3-8M is a pre-LayerNorm encoder")
        if self.causal_attention:
            raise ValueError("PA3-8M attention is non-causal")
        if self.dropout != 0.0 or self.attention_dropout != 0.0:
            raise ValueError("PA3-8M uses dropout 0.0")
        if self.activation != "gelu":
            raise ValueError("PA3-8M uses GELU")
        if self.tokens != self.active_tokens + (self.tokens - self.active_tokens):
            raise ValueError("token padding invariant broken")
        if self.active_tokens > self.tokens:
            raise ValueError("active tokens must fit the padded token block")
        if self.max_request_branches > 4:
            raise ValueError("the spec caps within-request branches at four")
        if self.prefix_hidden != self.d_model:
            raise ValueError("prefix hidden state matches d_model (320)")

    @classmethod
    def from_dict(cls, data: dict) -> "PA3Config":
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in known}
        extras = {k: v for k, v in data.items() if k not in known}
        config = cls(**values)
        if extras:
            object.__setattr__(config, "extras", extras)
        config.validate()
        return config

    def to_dict(self) -> dict:
        out = {f: getattr(self, f) for f in self.__dataclass_fields__ if f != "extras"}
        out.update(self.extras)
        return out

    def observation_layout(self) -> ObservationLayout:
        return ObservationLayout()
