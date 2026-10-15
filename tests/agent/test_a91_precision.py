import torch

from agent.model import PA3Config, build_model
from agent.model.precision import get_precision_contract
from agent.types.observation import ObservationBatch


def test_precision_modes_are_explicit_and_versioned():
    assert get_precision_contract("A").identity == "pa3-precision-A-v2"
    assert not get_precision_contract("A").quantize_observation
    assert get_precision_contract("C").quantize_observation
    assert get_precision_contract("D").scorer_amp is False


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
    model.configure_precision("A")
    assert model.precision_contract.identity == "pa3-precision-A-v2"
