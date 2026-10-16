import torch
import torch.nn.functional as F
import pytest

from agent.model import PA3Config, build_model
from agent.model.precision import get_precision_contract
from agent.types.observation import ObservationBatch


def test_precision_modes_are_explicit_and_versioned():
    for mode in ("A", "B", "C", "D"):
        assert get_precision_contract(mode).identity == f"pa3-precision-{mode}-v2"
    assert not get_precision_contract("A").quantize_observation
    assert get_precision_contract("C").quantize_observation
    assert get_precision_contract("D").scorer_amp is False
    assert get_precision_contract("D").ragged_effects_fp32 is True
    candidate = get_precision_contract("E")
    assert candidate.identity == "pa3-precision-E-v1"
    assert candidate.quantize_observation and candidate.ragged_effects_fp32
    assert not any((candidate.encoder_amp, candidate.scorer_amp, candidate.gru_amp))


def test_canonical_fp16_observation_preserves_non_float_fields():
    obs = ObservationBatch.dummy(batch_size=2, seed=17)
    contract = get_precision_contract("C")
    quantized = contract.prepare_observation(obs)
    assert torch.equal(quantized.categories, obs.categories)
    assert torch.equal(quantized.token_mask, obs.token_mask)
    assert torch.equal(quantized.flags, obs.flags)
    assert torch.equal(quantized.floats, obs.floats.half().float())
    rebuilt = ObservationBatch.from_compact_numpy(quantized.to_compact_numpy())
    assert torch.equal(rebuilt.floats, quantized.floats)


def test_model_precision_configuration_is_reversible():
    model = build_model(PA3Config())
    model.configure_precision("C")
    assert model.precision_contract.identity == "pa3-precision-C-v2"
    assert model.scorer.precision_contract.identity == "pa3-precision-C-v2"
    assert model.encoder.embedding.ragged_embedding.fp32_effects is False
    model.configure_precision("D")
    assert model.encoder.embedding.ragged_embedding.fp32_effects is True


def test_mode_e_preserves_mode_d_observation_quantization():
    import numpy as np
    from agent.types.ragged import CATEGORICAL, FIELDS, RaggedBatch

    obs = ObservationBatch.dummy(batch_size=2, seed=17)
    for name, fields in FIELDS.items():
        counts = np.zeros((2, obs.layout.TOKENS), dtype=np.int64)
        counts[0, 0] = 1
        values = np.full((1, len(fields)), 0.1234567, dtype=np.float32)
        values[:, CATEGORICAL[name]] = 1
        obs.ragged[name] = RaggedBatch.from_rows(
            [values, np.zeros((0, len(fields)), dtype=np.float32)], counts, len(fields))
    original_floats = obs.floats.clone()
    original_ragged = {name: section.values.clone() for name, section in obs.ragged.items()}
    expected = get_precision_contract("D").prepare_observation(obs)
    actual = get_precision_contract("E").prepare_observation(obs)
    for field in ("token_mask", "categories", "category_known", "floats", "float_known",
                  "flags", "flag_known", "role_ids", "side_ids"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))
    assert actual.validated_token_limit == expected.validated_token_limit
    for name, section in actual.ragged.items():
        assert torch.equal(section.values, expected.ragged[name].values)
        assert torch.equal(section.counts, expected.ragged[name].counts)
        np.testing.assert_array_equal(section.host_counts, expected.ragged[name].host_counts)
        assert torch.equal(obs.ragged[name].values, original_ragged[name])
    assert torch.equal(obs.floats, original_floats)


def test_mode_e_configuration_preserves_weights_and_is_reversible(model_factory):
    model = model_factory(seed=7)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for mode in ("E", "D", "A", "E", "C", "E"):
        model.configure_precision(mode)
        contract = get_precision_contract(mode)
        assert model.precision_contract == contract == model.scorer.precision_contract
        assert model.encoder.embedding.ragged_embedding.fp32_effects == contract.ragged_effects_fp32
        assert set(model.state_dict()) == set(before)
        assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_mode_e_cpu_policy_matches_a_on_same_quantized_observations(model_factory):
    from pa3_test_util import make_candidate_batch, make_request

    model = model_factory(seed=7)
    observation = get_precision_contract("E").prepare_observation(
        ObservationBatch.dummy(batch_size=3, seed=17))
    candidates = make_candidate_batch([make_request(observation.select([index]))
                                       for index in range(len(observation))])
    selected = torch.zeros(candidates.action_ids.shape[:2], dtype=torch.long)
    model.configure_precision("E")
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        actual_encoded, actual = model.evaluate(observation, candidates, selected=selected)
    model.configure_precision("A")
    with torch.inference_mode():
        expected_encoded, expected = model.evaluate(observation, candidates, selected=selected)
    assert actual_encoded.tokens.dtype == actual.request_logprob.dtype == torch.float32
    assert torch.isfinite(actual.request_logprob).all()
    torch.testing.assert_close(actual_encoded.tokens, expected_encoded.tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual.request_logprob, expected.request_logprob, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode_e_encoder_overrides_external_autocast_and_matches_a(model_factory):
    model = model_factory(seed=7).cuda()
    model.encoder.canonical_batch_rows = 1024
    observation = get_precision_contract("E").prepare_observation(
        ObservationBatch.dummy(batch_size=7, seed=17)).to("cuda")
    physical_rows = []
    handle = model.encoder.embedding.register_forward_pre_hook(
        lambda _module, inputs: physical_rows.append(len(inputs[0])))
    try:
        model.configure_precision("E")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            actual = model.encode(observation)
        model.configure_precision("A")
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            expected = model.encode(observation)
    finally:
        handle.remove()
    assert physical_rows == [7, 7]  # FP32 uses logical rows despite canonical AMP setting.
    assert actual.tokens.dtype == torch.float32
    torch.testing.assert_close(actual.tokens, expected.tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual.global_repr, expected.global_repr, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode_d_effects_numeric_projection_is_explicit_fp32(model_factory):
    model = model_factory(seed=7).cuda()
    model.configure_precision("D")
    ragged = model.encoder.embedding.ragged_embedding
    layer = ragged.numeric["effects"]
    values = torch.randn(31, layer.in_features, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        actual = ragged._numeric_projection("effects", values)
        unaffected = ragged._numeric_projection("types", values[:, :ragged.numeric["types"].in_features])
    with torch.autocast("cuda", enabled=False):
        expected = F.linear(values.float(), layer.weight.float(), layer.bias.float())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert unaffected.dtype == torch.float16
    model.configure_precision("A")
    assert model.precision_contract.identity == "pa3-precision-A-v2"
