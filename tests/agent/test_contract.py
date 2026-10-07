"""Typed contract: request kinds, branch structure and the observation adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.mock_engine import MockNativeEngine
from agent.types.actions import (
    ActionKind,
    AtomicAction,
    RequestKind,
    branch_slots_for_request,
    is_singleton_request,
)
from agent.types.observation import ObservationBatch, ObservationLayout


def test_request_kind_values_match_native_contract():
    assert int(RequestKind.PREVIEW) == 0
    assert int(RequestKind.NORMAL) == 1
    assert int(RequestKind.REPLACEMENT) == 2
    assert int(RequestKind.WAIT) == 3
    assert int(RequestKind.FINISHED) == 4
    assert RequestKind.PREVIEW.is_decision
    assert not RequestKind.WAIT.is_decision
    assert ActionKind.PICK == 0 and ActionKind.MOVE == 1
    assert ActionKind.SWITCH == 2 and ActionKind.PASS == 3


def test_branch_slots_for_each_request_type():
    assert branch_slots_for_request(RequestKind.PREVIEW) == (0, 1, 2, 3)
    assert branch_slots_for_request(RequestKind.NORMAL) == (0, 1)
    assert branch_slots_for_request(RequestKind.REPLACEMENT, (True, False)) == (0,)
    assert branch_slots_for_request(RequestKind.REPLACEMENT, (False, True)) == (1,)
    assert branch_slots_for_request(RequestKind.REPLACEMENT, (True, True)) == (0, 1)
    assert branch_slots_for_request(RequestKind.REPLACEMENT, (False, False)) == ()
    assert branch_slots_for_request(RequestKind.WAIT) == ()
    assert branch_slots_for_request(RequestKind.FINISHED) == ()


def test_singleton_detection_and_action_tuple_round_trip():
    assert is_singleton_request([1, 1])
    assert is_singleton_request([0, 1])
    assert is_singleton_request([])
    assert not is_singleton_request([2, 1])
    action = AtomicAction.from_tuple((1, 0, 2, -1, 255, 1))
    assert action.as_tuple() == (1, 0, 2, -1, 255, 1)
    assert action.kind == ActionKind.MOVE
    with pytest.raises(ValueError):
        AtomicAction.from_tuple((9, 0, 0, 0, 0, 0))


def test_layout_token_indices():
    layout = ObservationLayout()
    assert layout.self_pokemon_token(0) == 4
    assert layout.opponent_pokemon_token(0) == 10
    assert layout.move_token(0, 0) == 16
    assert layout.move_token(5, 3) == 16 + 23
    assert int(layout.default_token_mask().sum()) == layout.ACTIVE_TOKENS == 88
    assert layout.default_roles()[layout.GLOBAL] == layout.ROLE_GLOBAL


def test_native_payload_adapter_and_compact_round_trip():
    engine = MockNativeEngine(num_teams=4, seed=7)
    handles = engine.reset_batch([0], [1], [[1, 2, 3, 4]], [(0, 1)])
    payload = engine.observe_payload(handles[0], 0)
    batch = ObservationBatch.from_native_payload(payload)
    assert len(batch) == 1
    assert batch.token_mask.shape == (1, 96)
    assert batch.categories.shape == (1, 96, 32)
    assert batch.floats.shape == (1, 96, 50)
    assert batch.flags.shape == (1, 96, 40)
    assert batch.token_mask.sum().item() == 88
    assert batch.floats.dtype == torch.float32

    compact = batch.to_compact_numpy()
    rebuilt = ObservationBatch.from_compact_numpy(compact)
    assert torch.equal(rebuilt.token_mask, batch.token_mask)
    assert torch.equal(rebuilt.categories, batch.categories)
    assert torch.allclose(rebuilt.floats, batch.floats, atol=1e-3)


def test_payload_with_junk_floats_is_flushed_not_propagated():
    engine = MockNativeEngine(num_teams=4, seed=3)
    handles = engine.reset_batch([0], [1], [[1, 1, 1, 1]], [(0, 1)])
    payload = engine.observe_payload(handles[0], 0)
    payload["floats"][0, 0] = np.nan
    payload["floats"][1, 0] = np.inf
    with pytest.raises(ValueError):
        ObservationBatch.from_native_payload({"token_mask": np.zeros(4, dtype=bool)})
    with pytest.raises(ValueError):
        ObservationBatch.from_compact_numpy({"token_mask": np.zeros(96, dtype=bool)})
