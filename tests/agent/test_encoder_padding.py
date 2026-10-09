"""Optional empty-token trimming preserves the 96-token model contract."""

from __future__ import annotations

import copy
from dataclasses import fields, replace

import pytest
import torch
import numpy as np

from agent.types.observation import ObservationBatch

from pa3_test_util import make_candidate_batch, make_request, normal_branches, preview_branches


def _block_widths(encoder, observation):
    widths = []
    hook = encoder.blocks[0].register_forward_pre_hook(
        lambda _module, args: widths.append(args[0].shape[1])
    )
    try:
        encoded = encoder(observation)
    finally:
        hook.remove()
    return encoded, widths


def test_trim_is_opt_in_and_preserves_full_output_shape(model_factory):
    model = model_factory(seed=67)
    observation = ObservationBatch.dummy(2, seed=11)
    assert model.encoder.trim_padding is False
    assert observation.validated_token_limit == 88
    full, widths = _block_widths(model.encoder, observation)
    assert widths == [96]
    model.encoder.trim_padding = True
    trimmed, widths = _block_widths(model.encoder, observation)
    assert widths == [88]
    assert trimmed.shape == full.shape == (2, 96, model.config.d_model)
    assert torch.equal(trimmed[:, 88:], torch.zeros_like(trimmed[:, 88:]))
    torch.testing.assert_close(trimmed, full, rtol=2e-5, atol=2e-6)


def test_trimmed_model_matches_all_outputs_and_parameter_gradients(model_factory):
    reference = model_factory(seed=71)
    # Exercise a trained-like model whose normalization biases are nonzero.
    torch.manual_seed(83)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(0.003 * torch.randn_like(parameter))
    optimized = copy.deepcopy(reference)
    optimized.encoder.trim_padding = True
    observation = ObservationBatch.dummy(3, seed=13)
    observation.token_mask[1, :88:3] = False
    observation.token_mask[2] = False
    observation = observation.to("cpu")  # revalidate after the mask mutations
    candidates = make_candidate_batch([
        make_request(observation.select([0]), preview_branches()),
        make_request(observation.select([1]), normal_branches()),
        make_request(observation.select([2]), normal_branches()),
    ])
    selected = torch.tensor([[3, 4, 2, 1], [1, 0, -1, -1], [0, 1, -1, -1]])
    outputs = []
    for model in (reference, optimized):
        encoded, values, evaluation = model(observation, candidates, selected=selected)
        outputs.append((encoded, values, evaluation))
        loss = (-evaluation.request_logprob - 0.3 * evaluation.request_entropy
                + 0.2 * evaluation.request_uniform_kl + values.square()).mean()
        loss.backward()
    for name in ("tokens", "global_repr"):
        torch.testing.assert_close(getattr(outputs[0][0], name), getattr(outputs[1][0], name),
                                   rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=3e-5, atol=3e-6)
    for entry in fields(outputs[0][2]):
        torch.testing.assert_close(getattr(outputs[0][2], entry.name),
                                   getattr(outputs[1][2], entry.name), rtol=3e-5, atol=3e-6)
    for (name, parameter), (other_name, other) in zip(
        reference.named_parameters(), optimized.named_parameters()
    ):
        assert name == other_name
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            difference = float((parameter.grad - other.grad).abs().max())
            scale = float(parameter.grad.abs().max())
            assert difference <= 5e-5 * scale + 3e-6, (name, difference, scale)


@pytest.mark.parametrize("mutation", ["in_place", "view", "replacement", "unknown"])
def test_mask_mutation_or_unknown_stamp_falls_back_to_all_tokens(model_factory, mutation):
    model = model_factory(seed=73)
    model.encoder.trim_padding = True
    observation = ObservationBatch.dummy(2, seed=17)
    assert observation.validated_token_limit == 88
    if mutation == "in_place":
        observation.token_mask[0, 95] = True
    elif mutation == "view":
        observation.token_mask[:, 88:][0, 7] = True
    elif mutation == "replacement":
        observation.token_mask = observation.token_mask.clone()
        observation.token_mask[0, 95] = True
    else:
        observation = replace(observation)
    assert observation.validated_token_limit is None
    result, widths = _block_widths(model.encoder, observation)
    assert widths == [96]
    model.encoder.trim_padding = False
    torch.testing.assert_close(result, model.encoder(observation), rtol=0, atol=0)
    # Unknown metadata must stay unknown through row transforms.
    assert observation.select([0]).validated_token_limit is None
    assert observation.narrow(0, 1).validated_token_limit is None
    assert observation.cat([observation]).validated_token_limit is None


def test_token_bounds_propagate_without_entering_the_wire_schema():
    observation = ObservationBatch.dummy(3, seed=19)
    for transformed in (observation.to("cpu"), observation.select([2, 0]),
                        observation.narrow(1, 3), observation.cat([observation]),
                        observation.clone()):
        assert transformed.validated_token_limit == 88
    compact = observation.to_compact_numpy()
    assert not any("limit" in key or "stamp" in key for key in compact)
    assert ObservationBatch.from_compact_numpy(compact).validated_token_limit == 88
    payload = {entry.name: getattr(observation, entry.name).numpy()
               for entry in fields(observation) if isinstance(getattr(observation, entry.name), torch.Tensor)}
    assert ObservationBatch.from_native_payload(payload).validated_token_limit == 88
    assert all("limit" not in entry.name and "stamp" not in entry.name for entry in fields(observation))


def test_active_tail_is_revalidated_at_cpu_transfer_and_factory_boundaries(model_factory):
    observation = ObservationBatch.dummy(2, seed=23)
    observation.token_mask[0, 95] = True
    checked = observation.to("cpu")
    assert checked.validated_token_limit == 96
    assert checked.select([0]).validated_token_limit == 96
    assert checked.cat([ObservationBatch.dummy(1)]).validated_token_limit == 96
    assert ObservationBatch.from_compact_numpy(checked.to_compact_numpy()).validated_token_limit == 96
    model = model_factory(seed=79)
    model.encoder.trim_padding = True
    _, widths = _block_widths(model.encoder, checked)
    assert widths == [96]


def test_shared_narrow_view_stamp_invalidates_when_parent_changes():
    parent = ObservationBatch.dummy(2)
    child = parent.narrow(0, 1)
    assert child.validated_token_limit == 88
    parent.token_mask[0, 90] = True
    assert parent.validated_token_limit is None
    assert child.validated_token_limit is None


def test_inference_tensor_without_version_counter_uses_full_path(model_factory):
    model = model_factory(seed=89)
    model.encoder.trim_padding = True
    with torch.inference_mode():
        observation = ObservationBatch.dummy(1)
        assert observation.validated_token_limit is None
        _, widths = _block_widths(model.encoder, observation)
    assert widths == [96]


def test_numpy_payload_and_compact_exports_cannot_mutate_verified_masks():
    observation = ObservationBatch.dummy(1)
    payload = observation.to_compact_numpy()
    adapted = ObservationBatch.from_native_payload(payload)
    assert adapted.validated_token_limit == 88
    payload["token_mask"][0, 95] = True
    assert not bool(observation.token_mask[0, 95])
    assert not bool(adapted.token_mask[0, 95])
    assert observation.validated_token_limit == adapted.validated_token_limit == 88
    array = np.zeros((1, 96), dtype=bool)
    generic = replace(observation, token_mask=torch.from_numpy(array))
    verified = generic.to("cpu")
    array[0, 95] = True
    assert bool(verified.token_mask[0, 95])
    assert verified.validated_token_limit == 96


def test_direct_numpy_alias_refreshes_cpu_bounds_and_transfer_proof(model_factory):
    observation = ObservationBatch.dummy(2)
    version = observation.token_mask._version
    observation.token_mask.numpy()[0, 95] = True
    assert observation.token_mask._version == version
    assert observation.validated_token_limit == 96
    for transformed in (observation.to("cpu"), observation.select([0]),
                        observation.narrow(0, 1), observation.cat([observation])):
        assert transformed.validated_token_limit == 96
    model = model_factory(seed=97)
    model.encoder.trim_padding = True
    _, widths = _block_widths(model.encoder, observation)
    assert widths == [96]
