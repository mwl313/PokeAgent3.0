"""PA3-8M architecture and forward-shape contract."""

from __future__ import annotations

import torch

from agent.model import PA3Config, build_model
from agent.types.observation import ObservationBatch

from pa3_test_util import make_candidate_batch, make_request, preview_branches


def test_parameter_count_and_architecture_constants():
    config = PA3Config()
    config.validate()
    model = build_model(config)
    count = model.parameter_count()
    assert 8_000_000 <= count <= 10_000_000, count
    assert config.encoder_layers == 6
    assert config.d_model == 320
    assert config.attention_heads == 5
    assert config.head_dim == 64
    assert config.attention_heads * config.head_dim == config.d_model
    assert config.ffn_dim == 1280
    assert config.activation == "gelu"
    assert config.dropout == 0.0 and config.attention_dropout == 0.0
    assert config.pre_layernorm is True
    assert config.causal_attention is False
    assert config.prefix_hidden == 320
    assert config.critic_hidden == 256
    assert config.critic_input == "observation_only"
    assert config.tokens == 96 and config.active_tokens == 88
    assert config.candidate_padding == 64
    assert config.prefix_decoder == "GRUCell"


def test_forward_shapes_and_encode_state_once_per_request(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(2)
    branches = preview_branches()
    rows = [make_request(observation.select([i]), branches) for i in range(2)]
    candidates = make_candidate_batch(rows)

    model.encoder.reset_counter()
    encoded, evaluation = model.evaluate(observation, candidates)
    # Four branches must reuse one cached encoding: the state is encoded once.
    assert model.encoder.forward_calls == 1
    assert encoded.tokens.shape == (2, 96, model.config.d_model)
    assert encoded.global_repr.shape == (2, model.config.d_model)
    assert model.value(encoded).shape == (2,)
    assert evaluation.logits.shape == (2, 4, 64)
    assert evaluation.logprob_selected.shape == (2, 4)
    assert evaluation.request_logprob.shape == (2,)
    assert evaluation.request_entropy.shape == (2,)
    assert evaluation.request_uniform_kl.shape == (2,)
    assert evaluation.branch_k[:, 0].tolist() == [6, 6]
    assert evaluation.branch_k[:, 2].tolist() == [4, 4]
    assert evaluation.branch_valid[:, 2].all()
    # The preview fills all four branches (lead A, lead B, reserve 1, reserve 2).
    assert evaluation.branch_valid.all()
    assert evaluation.branch_k[:, 3].tolist() == [3, 3]


def test_value_head_ignores_the_selected_action_prefix(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    candidates = make_candidate_batch([make_request(observation, preview_branches())])
    first = torch.tensor([[0, 0, 0, 0]])
    second = torch.tensor([[5, 4, 3, 2]])
    encoded_a, evaluation_a = model.evaluate(observation, candidates, selected=first)
    encoded_b, evaluation_b = model.evaluate(observation, candidates, selected=second)
    assert torch.equal(encoded_a.global_repr, encoded_b.global_repr)
    assert torch.equal(model.value(encoded_a), model.value(encoded_b))
    # The policy does depend on the prefix: different selections differ.
    assert not torch.allclose(
        evaluation_a.request_logprob, evaluation_b.request_logprob
    )


def test_probability_and_logprob_math_is_float32(model_factory):
    model = model_factory()
    observation = ObservationBatch.dummy(1)
    candidates = make_candidate_batch([make_request(observation, preview_branches())])
    _, evaluation = model.evaluate(observation, candidates)
    assert evaluation.logits.dtype == torch.float32
    assert evaluation.logprob_selected.dtype == torch.float32
    assert evaluation.request_logprob.dtype == torch.float32
