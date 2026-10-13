"""Bounded native retry witnesses and consumer protocol smoke tests (no PPO)."""
import json
from pathlib import Path

import pytest
import torch

pa3_engine = pytest.importorskip("pa3_engine")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "engine/data"


def action(row):
    kinds = {"Pick": 0, "Move": 1, "Switch": 2, "Pass": 3}
    return (kinds[row["kind"]], row["own_slot"], row["move_slot"],
            row["target_location"], row["switch_destination"],
            1 if row["resource"] == "Mega" else 0)


@pytest.mark.parametrize("fixture_index", [0, 1])
def test_native_python_retry_roundtrip(tmp_path, fixture_index):
    fixture = json.loads((DATA / "imprison-retry-fixtures.json").read_text())["fixtures"][fixture_index]
    teams = tmp_path / "teams.json"
    teams.write_text(json.dumps(fixture["teams"]))
    engine = pa3_engine.NativeEngine(str(DATA), str(teams), workers=2)
    handle = engine.reset_batch([0], [1], [tuple(fixture["seed"])], [(0, 1)])[0]
    for step in fixture["steps"]:
        side = 0 if step["side"] == "P1" else 1
        result = engine.step_batch([(*handle, [(side, [action(a) for a in step["actions"]])])])[0]
        assert result[0] == step["accepted"]
        assert result[4] is None
        saved = engine.snapshot(*handle)
        engine.restore(*handle, saved)
        assert engine.snapshot(*handle) == saved
        if not result[0]:
            assert result[5:] == (3, 1)  # committed peer waits, chooser retries
            assert engine.request_kind(*handle, 0) == 3
            assert list(engine.request_branches(*handle, 0)) == []
            info=engine.request_info_batch([(*handle, 0), (*handle, 1)])
            assert [(kind,list(slots)) for kind,slots in info] == [(3, []), (1, [0, 1])]
            first = action(step["actions"][0])
            candidates = engine.candidates(*handle, side, [first])
            # Both Protect and Rest were disclosed by the first rejection.
            assert all(candidate[0] != 1 or candidate[2] not in (1, 2) for candidate in candidates)


class RetryOnce:
    """Inject only the non-error retry signal to isolate consumer control flow.

    Actual engine mask disclosure is checked separately against Showdown above.
    """
    def __init__(self, engine):
        self.engine, self.retries = engine, 0

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def step_batch(self, specs):
        if not self.retries:
            self.retries += 1
            return [(False, False, False, None, None, 0, 0) for _ in specs]
        return self.engine.step_batch(specs)


@pytest.mark.parametrize("batched", [False, True])
def test_evaluators_resample_non_error_retry(batched):
    from agent.evaluation.fixed_policy import FixedPolicyEvaluator, build_fixture_manifest
    from agent.evaluation.batched_policy import BatchedPolicyEvaluator
    engine = RetryOnce(pa3_engine.NativeEngine(str(DATA), str(DATA / "a10-training-teams.json"), workers=2))
    cls = BatchedPolicyEvaluator if batched else FixedPolicyEvaluator
    result = cls(engine, device="cpu", seed=11).run(
        build_fixture_manifest(pairs=1, team_count=2, seed=7),
        policy="random", opponent="lexicographic_legal")
    assert engine.retries == 1
    assert result["illegal_actions"] == result["operational_errors"] == 0
    assert len(result["outcomes"]) == 2


def test_collector_retains_retry_decision_and_completes_natural_match():
    from agent.model import PA3Config, build_model
    from agent.train.native_collector import NativeCollector, NativeCollectorConfig
    torch.set_num_threads(2)
    torch.manual_seed(11)
    engine = RetryOnce(pa3_engine.NativeEngine(str(DATA), str(DATA / "a10-training-teams.json"), workers=2))
    model = build_model(PA3Config(encoder_layers=1, d_model=32, attention_heads=2,
                                 head_dim=16, ffn_dim=64, prefix_hidden=32, critic_hidden=32))
    collector = NativeCollector(engine, model, NativeCollectorConfig(
        envs=1, workers=2, seed=11, device="cpu", prefer_cuda=False,
        observation_mode="fixed", candidate_wire="packed"), device="cpu")
    buffer = collector.collect(1)
    assert engine.retries == 1
    assert collector.stats.games == 1 and collector.stats.operational_errors == 0
    assert buffer.natural_match_count() == 1
    rows = [row for row in buffer.rows if row.side == 0]
    assert rows[0].request_index < rows[1].request_index
    assert rows[0].turn == rows[1].turn  # retry disclosed information, no turn
    # No optimizer exists in this bounded collection test.
