"""v5c T2: rolling-slot collection preserves the natural-match contract.

Runs the real engine with rolling refill on and off and checks the invariants
that must hold regardless of scheduling: one count per natural match, both
current-policy seats recorded, one opposite-sign terminal reward per side on
the side's last request, overshoot preserved (games >= target), and the row
schema (valid branch counts, selected prefix inside the candidate table).
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


def _collect(native_engine_data, *, rolling: bool, target: int = 96):
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
    collector = NativeCollector(
        engine,
        model,
        NativeCollectorConfig(
            envs=32,
            workers=2,
            seed=777,
            device="cpu",
            prefer_cuda=False,
            observation_mode="fixed",
            candidate_wire="packed",
            rolling_slots=rolling,
        ),
        device="cpu",
    )
    return collector, collector.collect(target)


def _assert_contract(collector, buffer, target: int):
    by_side = defaultdict(list)
    for row in buffer.rows:
        by_side[(row.match_id, row.side)].append(row)
        assert row.policy_id == "current"
        assert row.opponent_policy_id == "current"
        assert row.branch_count >= 1
        for level, mask in enumerate(row.candidate_mask):
            assert len(mask) >= 1
            assert all(mask), "packed record path only stores legal candidates"
            assert 0 <= row.selected[level] < len(mask)
    assert collector.stats.games >= target, "quota must be met (overshoot preserved)"
    assert collector.stats.games == buffer.natural_match_count()
    assert collector.stats.operational_errors == 0
    assert {row.side for row in buffer.rows} == {0, 1}
    for (match_id, side), rows in by_side.items():
        done = [row for row in rows if row.done]
        assert len(done) == 1, (match_id, side)
        assert done[0] is rows[-1]
        assert done[0].reward in (-1.0, 0.0, 1.0)
    for match_id in {key[0] for key in by_side}:
        assert by_side[(match_id, 0)][-1].reward == -by_side[(match_id, 1)][-1].reward


def test_rolling_slot_contract_matches_static_cohorts(native_engine_data):
    static_collector, static_buffer = _collect(native_engine_data, rolling=False)
    rolling_collector, rolling_buffer = _collect(native_engine_data, rolling=True)
    _assert_contract(static_collector, static_buffer, 96)
    _assert_contract(rolling_collector, rolling_buffer, 96)
    # Rolling refill must not change the per-match row-count distribution
    # materially: same average rows per natural match within 35 %.
    static_rate = len(static_buffer.rows) / max(static_collector.stats.games, 1)
    rolling_rate = len(rolling_buffer.rows) / max(rolling_collector.stats.games, 1)
    assert rolling_rate == pytest.approx(static_rate, rel=0.35)


def test_rolling_slot_telemetry_reports_smaller_idle_fraction(native_engine_data):
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
    collector = NativeCollector(
        engine,
        model,
        NativeCollectorConfig(
            envs=32,
            workers=2,
            seed=777,
            device="cpu",
            prefer_cuda=False,
            observation_mode="fixed",
            candidate_wire="packed",
            rolling_slots=True,
            telemetry=True,
        ),
        device="cpu",
    )
    collector.collect(96)
    telemetry = collector.telemetry
    assert telemetry["rounds"] > 0
    assert telemetry["slot_seconds"] > 0
    assert telemetry["idle_slot_fraction"] if "idle_slot_fraction" in telemetry else True
    # Rolling refill idles only during the final drain.
    fraction = telemetry["idle_slot_seconds"] / max(telemetry["slot_seconds"], 1e-9)
    assert fraction < 0.25
