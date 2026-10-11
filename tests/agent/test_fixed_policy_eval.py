import copy

from agent.evaluation.fixed_policy import FixedPolicyEvaluator, build_fixture_manifest, fixture_manifest_hash
from agent.mock_engine import MockNativeEngine
from agent.model import build_model


def test_fixture_manifest_is_deterministic_and_seat_swapped():
    fixtures = build_fixture_manifest(pairs=2, team_count=5, seed=7)
    assert len(fixtures) == 4
    assert fixtures[0].pair_id == fixtures[1].pair_id == 0
    assert fixtures[0].policy_side == 0 and fixtures[1].policy_side == 1
    assert fixture_manifest_hash(fixtures) == fixture_manifest_hash(build_fixture_manifest(pairs=2, team_count=5, seed=7))


def test_random_and_heuristic_evaluation_is_reproducible_and_legal():
    fixtures = build_fixture_manifest(pairs=2, team_count=5, seed=7)
    first = FixedPolicyEvaluator(MockNativeEngine(num_teams=5, requests_per_match=3), seed=11).run(
        fixtures, policy="random", opponent="heuristic"
    )
    second = FixedPolicyEvaluator(MockNativeEngine(num_teams=5, requests_per_match=3), seed=11).run(
        fixtures, policy="random", opponent="heuristic"
    )
    assert first["outcomes"] == second["outcomes"]
    assert first["illegal_actions"] == 0
    assert first["operational_errors"] == 0
    assert first["fixture_manifest_hash"] == fixture_manifest_hash(fixtures)
    assert sum(item["matches"] for item in first["team_group_results"].values()) == len(fixtures)


def test_checkpoint_policy_is_frozen_and_legal():
    fixtures = build_fixture_manifest(pairs=1, team_count=5, seed=7)
    model = build_model()
    before = copy.deepcopy(model.state_dict())
    report = FixedPolicyEvaluator(MockNativeEngine(num_teams=5, requests_per_match=3), seed=11).run(
        fixtures, policy="checkpoint", model=model, opponent="heuristic"
    )
    assert report["illegal_actions"] == 0
    assert report["operational_errors"] == 0
    assert all((value == before[key]).all() for key, value in model.state_dict().items())
