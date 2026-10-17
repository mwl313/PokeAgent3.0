import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "random_baseline_eval", ROOT / "scripts" / "random_baseline_eval.py"
)
RBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RBE
SPEC.loader.exec_module(RBE)


def test_fixture_pairs_are_unique_and_stable(tmp_path):
    spec = {"offset": 500, "pairs": 12, "seed_domain": "unit:fixtures"}
    rows = RBE.build_test_fixtures(spec, list(range(1220)), "unit")
    assert len(rows) == 24
    assert len({row.fixture_id for row in rows}) == 24
    assert len({row.pair_id for row in rows}) == 12
    identities = set()
    for index in range(0, len(rows), 2):
        left, right = rows[index], rows[index + 1]
        assert left.pair_id == right.pair_id
        assert left.team_a == right.team_a and left.team_b == right.team_b
        assert left.seed == right.seed
        assert {left.policy_side, right.policy_side} == {0, 1}
        identity = (left.team_a, left.team_b, left.seed)
        assert identity not in identities
        identities.add(identity)
    again = RBE.build_test_fixtures(spec, list(range(1220)), "unit")
    assert rows == again


def test_fixture_manifest_round_trip(tmp_path):
    manifest = RBE.prepare_fixtures(str(tmp_path), RBE.DEFAULT_DATA, RBE.DEFAULT_TEAMS)
    assert manifest["team_pool_size"] == 1220
    loaded, loaded_manifest = RBE.load_fixtures(str(tmp_path))
    assert loaded_manifest["manifest_hash"] == manifest["manifest_hash"]
    assert len(loaded) == sum(row["fixtures"] for row in manifest["tests"].values())


def test_select_traces_covers_wins_losses_and_long_games():
    outcomes = []
    fixture_id = 0
    for index in range(40):
        outcome = "loss" if index < 5 else "win"
        rounds = 200 - index
        fixture_id += 1
        outcomes.append({"fixture_id": fixture_id, "pair_id": fixture_id, "outcome": outcome,
                         "rounds": rounds, "policy_side": index % 2,
                         "team_a": index, "team_b": index + 1})
    chosen = RBE.select_traces(outcomes, rng_seed=7)
    assert len(chosen) == 30
    assert len({row["fixture_id"] for row in chosen}) == 30
    assert sum(row["outcome"] == "win" for row in chosen) == 25
    assert sum(row["outcome"] == "loss" for row in chosen) == 5
    assert 1 in {row["fixture_id"] for row in chosen}  # longest game is retained


def test_paired_bootstrap_and_wilson_bounds():
    scores = [1.0] * 90 + [0.0] * 10
    low, high = RBE._bootstrap_ci(scores, seed=11, samples=2000)
    assert 0.0 <= low <= 0.9 <= high <= 1.0
    low, high = RBE._wilson(45.0, 100)
    assert 0.0 <= low <= 0.45 <= high <= 1.0


def test_csv_serializes_list_columns():
    text = RBE._csv([{"a": 1, "b": [0.1, 0.2]}], ["a", "b"])
    assert text.strip().splitlines()[1] == "1,0.100000;0.200000"
