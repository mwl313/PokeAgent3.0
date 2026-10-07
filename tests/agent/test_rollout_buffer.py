"""Rollout schema: overshoot, padding, self-play sides, no stored activations."""

from __future__ import annotations

import torch

from agent.buffer import RolloutBuffer

from pa3_test_util import synthetic_rows


def test_overshoot_rows_are_kept_and_matches_count_once():
    buffer = synthetic_rows(count=10, match_ids=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    assert len(buffer) == 10
    assert buffer.natural_match_count() == 3
    stats = buffer.stats()
    assert stats["rows"] == 10
    assert stats["matches"] == 3
    assert stats["value_rows"] == 10


def test_final_minibatch_is_padded_and_masked():
    buffer = synthetic_rows(count=10)
    minibatches = list(buffer.iter_minibatches(batch_size=4))
    assert [len(mb) for mb in minibatches] == [4, 4, 4]
    assert [int(mb.row_valid.sum()) for mb in minibatches] == [4, 4, 2]
    assert minibatches[-1].sample_weight == 0.5
    # Padded rows repeat real rows (spread across the chunk so no microbatch is
    # padding-only) but are never counted as experience.
    last = minibatches[-1]
    assert int(last.row_valid.sum()) == 2
    assert not bool(last.row_valid.all())
    padding_positions = (~last.row_valid).nonzero().flatten().tolist()
    assert len(padding_positions) == 2
    for position in padding_positions:
        assert any(
            torch.equal(last.old_logprob[position], last.old_logprob[other])
            for other in range(len(last))
            if other != position
        )


def test_current_policy_self_play_collects_both_sides_but_not_history():
    buffer = synthetic_rows(
        count=6,
        match_ids=[0, 0, 1, 1, 2, 2],
        sides=[0, 1, 0, 1, 0, 1],
        policy_ids=["current", "current", "current", "current", "current", "history"],
    )
    current = buffer.current_policy_rows("current")
    assert len(current) == 5
    assert {row.side for row in current} == {0, 1}
    assert all(row.policy_id == "current" for row in current)
    # The historical opponent's action is recorded but is not a PPO row.
    historical = [row for row in buffer if row.policy_id == "history"]
    assert len(historical) == 1
    assert historical[0].opponent_policy_id == "current"


def test_encoder_activations_are_not_stored():
    buffer = synthetic_rows(count=3)
    payload = buffer.observation_store._rows[0]
    assert set(payload) == {
        "token_mask",
        "categories",
        "category_known",
        "floats",
        "float_known",
        "flags",
        "flag_known",
        "role_ids",
        "side_ids",
    }
    # No 320-wide encoder activation of any kind is present.
    for name, array in payload.items():
        assert array.shape[-1] != 320, name
    stats = buffer.stats()
    assert stats["rows"] == 3
    assert stats["branches"] == 6
    assert stats["candidates"] == 3 * (3 + 2)


def test_batch_tensors_carry_candidate_masks_and_selected_prefix():
    buffer = synthetic_rows(count=2)
    batch = buffer.to_batch()
    assert batch.candidates.action_ids.shape == (2, 4, 64, 6)
    assert batch.candidates.mask.shape == (2, 4, 64)
    assert batch.candidates.branch_valid.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
    ]
    assert batch.candidates.branch_k[:, 0].tolist() == [3, 3]
    assert batch.candidates.selected[:, :2].tolist() == [[0, 0], [0, 0]]
    assert batch.candidates.selected[:, 2:].tolist() == [[-1, -1], [-1, -1]]
    assert batch.old_logprob.shape == (2,)
    assert batch.actor_mask.tolist() == [True, True]
    assert len(batch.policy_ids) == 2
    assert batch.request_kind.tolist() == [1, 1]


def test_row_records_every_required_field():
    buffer = synthetic_rows(count=1)
    row = buffer.rows[0]
    assert row.request_kind.name == "NORMAL"
    assert row.selected == (0, 0)
    assert row.selected_actions[0][0] == 1  # ActionKind.MOVE
    assert row.candidate_mask[0].count(True) == 3
    assert row.policy_id == "current"
    assert row.opponent_policy_id == "current"
    assert isinstance(row.team_ids, tuple) and len(row.team_ids) == 2
    assert row.match_id == 0
    assert row.turn >= 1
    assert row.seed_ref is None
    assert row.old_logprob < 0.0


def test_empty_and_padded_minibatch_edge_cases():
    empty = RolloutBuffer()
    try:
        list(empty.iter_minibatches(4))
    except ValueError as error:
        assert "empty" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("empty buffer must not iterate")

    buffer = synthetic_rows(count=1)
    minibatch = next(iter(buffer.iter_minibatches(batch_size=4)))
    assert int(minibatch.row_valid.sum()) == 1
    assert minibatch.sample_weight == 0.25
    assert torch.equal(minibatch.match_ids[:1] * 0 + minibatch.match_ids, minibatch.match_ids)
