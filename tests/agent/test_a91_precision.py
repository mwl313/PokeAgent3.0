import torch
import torch.nn.functional as F
import pytest

from agent.model import PA3Config, build_model
from agent.model.precision import get_precision_contract
from agent.types.observation import ObservationBatch


def test_precision_modes_are_explicit_and_versioned():
    assert get_precision_contract("A").identity == "pa3-precision-A-v2"
    assert not get_precision_contract("A").quantize_observation
    assert get_precision_contract("C").quantize_observation
    assert get_precision_contract("D").scorer_amp is False
    assert get_precision_contract("D").ragged_effects_fp32 is True


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
