"""C0 §3.3: the both-current-side collection contract on the real engine.

Natural completion, one terminal reward per recorded side with opposite signs,
match counted once, request_index chains contained inside (match_id, side), and
the absence of historical-opponent learner rows.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

pytest.importorskip("pa3_engine")

import pa3_engine  # noqa: E402
from agent.model import PA3Config, build_model  # noqa: E402
from agent.train.native_collector import (  # noqa: E402
    NativeCollector,
    NativeCollectorConfig,
)


@pytest.fixture(scope="module")
def collection(native_engine_data):
    data, teams = native_engine_data
    engine = pa3_engine.NativeEngine(data, teams, workers=2)
    model = build_model(
        PA3Config(
            encoder_layers=1,
            d_model=32,
            attention_heads=2,
            head_dim=16,
            ffn_dim=64,
            prefix_hidden=32,
            critic_hidden=32,
        )
    )
    config = NativeCollectorConfig(
        envs=16,
        workers=2,
        seed=424242,
        device="cpu",
        prefer_cuda=False,
        observation_mode="fixed",
        candidate_wire="packed",
        collect_both_sides_when_current_self_play=True,
    )
    collector = NativeCollector(engine, model, config, device="cpu")
    buffer = collector.collect(16)
    return collector, buffer


def test_every_natural_match_has_both_current_policy_sides(collection):
    collector, buffer = collection
    by_match = defaultdict(dict)
    for row in buffer.rows:
        by_match[row.match_id].setdefault(row.side, []).append(row)

    assert collector.stats.operational_errors == 0
    assert collector.stats.games == buffer.natural_match_count()
    assert collector.stats.games == len(by_match)
    assert collector.stats.games >= 16
    assert {row.policy_id for row in buffer.rows} == {"current"}
    assert {row.opponent_policy_id for row in buffer.rows} == {"current"}
    for match_id, sides in by_match.items():
        assert set(sides) == {0, 1}, match_id
        for side, rows in sides.items():
            assert rows, (match_id, side)
            indices = [row.request_index for row in rows]
            # request_index counts consumed decision rounds for the environment
            # (the other seat's rounds advance it too), so a side's own chain is
            # strictly increasing, not necessarily contiguous.
            assert indices == sorted(indices) and len(set(indices)) == len(indices)


def test_terminal_rewards_are_opposite_and_paid_once_per_side(collection):
    collector, buffer = collection
    by_side = defaultdict(list)
    for row in buffer.rows:
        by_side[(row.match_id, row.side)].append(row)

    terminal_rewards = {}
    for (match_id, side), rows in by_side.items():
        done_rows = [row for row in rows if row.done]
        assert len(done_rows) == 1, (match_id, side, len(done_rows))
        assert done_rows[0] is rows[-1], "the terminal reward must land on the last request"
        reward = done_rows[0].reward
        assert reward in (-1.0, 0.0, 1.0), (match_id, side, reward)
        terminal_rewards[(match_id, side)] = reward

    learner_revenue = 0.0
    for match_id in {key[0] for key in terminal_rewards}:
        first, second = terminal_rewards[(match_id, 0)], terminal_rewards[(match_id, 1)]
        assert first == -second, (match_id, first, second)
        assert first != 0.0 or second == 0.0
        learner_revenue += second  # learner seat is side 1 for this fixture's role layout
    assert learner_revenue == pytest.approx(collector.stats.reward_sum)
    assert collector.stats.wins + collector.stats.losses + collector.stats.draws == collector.stats.games


def test_gae_chains_stay_inside_match_and_side(collection):
    _, buffer = collection
    buffer.compute_gae(gamma=1.0, gae_lambda=0.95)
    by_side = defaultdict(list)
    for row in buffer.rows:
        by_side[(row.match_id, row.side)].append(row)
    for (match_id, side), rows in by_side.items():
        reward = rows[-1].reward
        returns = [row.return_ for row in rows]
        if reward == 0.0:
            assert all(value == 0.0 for value in returns), (match_id, side)
        else:
            assert any(value != 0.0 for value in returns), (match_id, side)
    # No trajectory may receive a return without a terminal reward of its own.
    nonzero_groups = {
        key for key, rows in by_side.items() if any(row.return_ != 0.0 for row in rows)
    }
    rewarded_groups = {key for key, rows in by_side.items() if rows[-1].reward != 0.0}
    assert nonzero_groups == rewarded_groups
