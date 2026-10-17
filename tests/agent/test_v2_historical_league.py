import json
from pathlib import Path

import pytest

from agent.evaluation import historical_league as H
from agent.evaluation.historical_selector import Record, calculate


def _entries(count=30, step=100000, start=100000):
    rows = []
    for index in range(count):
        rows.append({
            "id": f"p{index:02d}",
            "committed_matches": start + index * step,
            "model_digest": f"digest-{index:02d}",
            "file_sha256": f"file-{index:02d}",
            "model_digest_algorithm": "pa3-model-state-sha256-sorted-v1",
            "path": f"/tmp/p{index:02d}.pt",
        })
    return rows


def _summary(left, right, *, left_score, games=128, ci=(0.4, 0.6),
             status="success", fixture_pairs=64):
    left_wins = round(left_score * games)
    draws = 0
    left_losses = games - left_wins - draws
    return {
        "schema": H.RR_PAIR_SCHEMA,
        "pair_key": H.canonical_pair(left["id"], right["id"]),
        "left_id": left["id"],
        "right_id": right["id"],
        "left_clock": left["committed_matches"],
        "right_clock": right["committed_matches"],
        "left_model_digest": left["model_digest"],
        "right_model_digest": right["model_digest"],
        "tier": "basic",
        "fixture_pairs": fixture_pairs,
        "games": games,
        "matches": games,
        "left_wins": left_wins,
        "left_losses": left_losses,
        "draws": draws,
        "left_score": left_score,
        "right_score": 1.0 - left_score,
        "paired_fixtures": fixture_pairs,
        "ci95_left": list(ci),
        "operational_errors": 0,
        "illegal_actions": 0,
        "status": status,
        "receipt_identity": f"receipt-{left['id']}-{right['id']}",
        "fixture_manifest_hash": "fixture-hash",
        "fixture_seed": 20261017,
        "panel_version": H.RR_PANEL_VERSION,
        "environment": {"backend": "batched"},
        "evaluated_at": "2026-10-17T00:00:00+00:00",
    }


def test_pair_arithmetic_n30_champion_interior():
    entries = _entries()
    arithmetic = H.pair_arithmetic(entries, "p25")
    assert arithmetic["policies"] == 30
    assert arithmetic["total_pairs"] == 435
    assert arithmetic["fine_pairs"] == 56
    assert arithmetic["broad_pairs"] == 379
    assert arithmetic["total_games"] == 56 * 512 + 379 * 128 == 77184


def test_fine_edges_champion_endpoint():
    entries = _entries()
    fine = H.fine_pair_keys(entries, "p29")
    assert len(fine) == 57
    assert H.canonical_pair("p29", "p28") in fine


def test_plan_priority_and_bounded_selection():
    entries = _entries()
    plan = H.build_pair_plan(entries, "p25", "p29")
    assert plan["pairs"][0]["pair_key"] == H.canonical_pair("p25", "p29")
    assert plan["pairs"][0]["reason"] == "champion_vs_latest"
    champion_rows = [row for row in plan["pairs"] if row["reason"] == "champion_row"]
    assert len(champion_rows) == 28
    selected = H.select_pairs(plan, max_games=1000, fixture_pairs=64)
    assert sum(row["games"] for row in selected) <= 1000
    assert selected[0]["pair_key"] == H.canonical_pair("p25", "p29")


def test_freeze_registry_dedup(tmp_path):
    formal = tmp_path / "formal"
    formal.mkdir()
    policies = {}
    for index in range(4):
        policies[f"p{index}"] = {
            "id": f"p{index}",
            "committed_matches": (index + 1) * 100000,
            "model_digest": "same" if index <= 1 else f"digest-{index}",
            "file_sha256": f"file-{index}",
            "path": f"/tmp/p{index}.pt",
        }
    registry = {
        "schema": "pa3-policy-archive-v2",
        "policies": policies,
        "evaluations": [],
        "latest": "p3",
        "incumbent": "p0",
    }
    (formal / "registry.json").write_text(json.dumps(registry))
    (formal / "best_policy.json").write_text(json.dumps({"policy_id": "p2"}))
    root = tmp_path / "rr"
    root.mkdir()
    entries, meta = H.freeze_registry(formal, root, verify=False)
    assert len(entries) == 3
    assert meta["duplicates"][0]["dropped"] == "p0"
    assert meta["champion_id"] == "p2"
    assert (root / "registry.snapshot.json").is_file()


def test_pair_scores_and_bootstrap_are_pair_clustered():
    outcomes = [
        {"pair_id": 1, "outcome": "win"}, {"pair_id": 1, "outcome": "loss"},
        {"pair_id": 2, "outcome": "draw"}, {"pair_id": 2, "outcome": "draw"},
        {"pair_id": 3, "outcome": "win"},
    ]
    scores = H.pair_scores(outcomes)
    assert scores == [0.5, 0.5]
    first = H.bootstrap_interval(scores, seed=1)
    second = H.bootstrap_interval(scores, seed=1)
    assert first == second == [0.5, 0.5]


def test_summarize_pair_rejects_partial_panel():
    left, right = _entries(2)
    row = {
        "pair_key": H.canonical_pair(left["id"], right["id"]),
        "left_id": left["id"],
        "right_id": right["id"],
        "fixture_pairs": 64,
        "tier": "basic",
    }
    receipt = {
        "status": "success", "matches": 64, "paired_fixtures": 32,
        "score_mean": 0.5, "wins": 32, "losses": 32, "draws": 0,
        "identity": "receipt", "outcomes": [],
    }
    summary = H.summarize_pair(receipt, row, left_entry=left, right_entry=right, seed=1)
    assert summary["status"] == "invalid"
    assert summary["right_score"] is None


def test_bradley_terry_and_cycles():
    entries = _entries(3)
    a, b, c = entries
    summaries = [
        _summary(a, b, left_score=0.8, ci=(0.7, 0.9)),
        _summary(b, c, left_score=0.8, ci=(0.7, 0.9)),
        _summary(c, a, left_score=0.8, ci=(0.7, 0.9)),
    ]
    ratings = H.bradley_terry(summaries, entries)
    assert set(ratings) == {entry["id"] for entry in entries}
    assert sum(1 for row in ratings.values() if row["rank"] == 1) == 1
    cycles = H.find_cycles(entries, summaries)
    assert len(cycles) == 1
    assert cycles[0]["supported"] is True


def _shadow_fixture(tmp_path):
    entries = _entries(12)
    champion = entries[5]
    summaries = [
        _summary(champion, entries[4], left_score=0.40, ci=(0.30, 0.50)),
        _summary(champion, entries[6], left_score=0.55, ci=(0.45, 0.65)),
        _summary(champion, entries[7], left_score=0.48, ci=(0.38, 0.58)),
        _summary(champion, entries[8], left_score=0.60, ci=(0.50, 0.70)),
        _summary(champion, entries[0], left_score=0.90, ci=(0.85, 0.95)),
    ]
    promotions = tmp_path / "promotions"
    promotions.mkdir()
    for index, entry in enumerate(entries[1:5]):
        (promotions / f"{index}-{entry['id']}.json").write_text(json.dumps({
            "policy_id": entry["id"],
            "committed_matches": entry["committed_matches"],
            "model_digest": entry["model_digest"],
            "updated_at": index,
        }))
    return entries, champion, summaries, promotions


def test_shadow_plan_is_shadow_only_and_digest_pinned(tmp_path):
    entries, champion, summaries, promotions = _shadow_fixture(tmp_path)
    root = tmp_path / "rr"
    root.mkdir()
    plan = H.build_shadow_plan(
        entries, summaries, reference_id=champion["id"], root=root,
        promotions_dir=promotions, fixture_identity={"panel_version": H.RR_PANEL_VERSION})
    assert plan["shadow_only"] is True
    assert plan["history_rows_target"] == 0
    assert plan["history_probability"] == 0.0
    assert plan["status"] == "SHADOW_ONLY_NOT_ACTIVE"
    assert plan["activation"].startswith("FORBIDDEN")
    assert len(plan["opponents"]) == H.MAX_ACTIVE_ANCHORS
    assert abs(sum(row["weight"] for row in plan["opponents"]) - 1.0) < 1e-9
    assert all(row["weight"] <= 0.40 + 1e-12 for row in plan["opponents"])
    assert all(row["policy_id"] == f"md-{row['model_digest']}" for row in plan["opponents"])
    assert champion["id"] not in {row["source_policy_id"] for row in plan["opponents"]}
    assert (root / "opponent_plan.json").is_file()


def test_shadow_plan_marks_stale_evidence(tmp_path):
    entries, champion, summaries, promotions = _shadow_fixture(tmp_path)
    root = tmp_path / "rr"
    root.mkdir()
    plan = H.build_shadow_plan(
        entries, summaries, reference_id=champion["id"], root=root,
        promotions_dir=promotions, fixture_identity={})
    far = next(row for row in plan["opponents"] if row["source_policy_id"] == "p00")
    assert far["verified"] is False
    assert far["evidence_status"] in {"stale", "unknown"}


def test_incremental_milestone_hook_not_triggered_and_triggered(tmp_path):
    entries = _entries(10)
    entries[-1]["committed_matches"] = 1050000
    champion = entries[5]
    summaries = [_summary(champion, entries[4], left_score=0.5)]
    no_milestone = H.milestone_hook(entries, summaries, promotions_dir=None)
    assert no_milestone["available"] is False
    assert no_milestone["executed"] is False
    assert no_milestone["anchors"] == []
    triggered_entries = [dict(row) for row in entries]
    triggered_entries[-1]["committed_matches"] = 1200000
    triggered = H.milestone_hook(triggered_entries, summaries, promotions_dir=None)
    assert triggered["available"] is True
    assert triggered["executed"] is False
    assert 0 < len(triggered["anchors"]) <= H.MAX_ACTIVE_ANCHORS
    assert sum(row["target_games"] for row in triggered["anchors"]) <= 8 * 256


def test_reference_selector_formula_unchanged():
    records = [Record("b", 0, 0, 128), Record("c", 64, 0, 64),
               Record("a", 110, 0, 18), Record("d", 60, 3, 55)]
    result = calculate(records)
    weights = [row["weight"] for row in result["opponents"]]
    assert abs(sum(weights) - 1.0) < 1e-12
    assert all(0.01 <= weight <= 0.40 + 1e-12 for weight in weights)
    assert [row["policy_id"] for row in result["opponents"]] == ["a", "b", "c", "d"]


def test_reference_selector_rejects_invalid_input():
    with pytest.raises(ValueError):
        calculate([])
    with pytest.raises(ValueError):
        calculate([Record("a", 1, 0, 1), Record("a", 1, 0, 1)])
    with pytest.raises(ValueError):
        calculate([Record("a", 1, 0, 1)], hard_mix=1.2)


def test_corrupted_pair_summary_is_ignored(tmp_path):
    root = tmp_path / "rr"
    pairs = root / "receipts" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "broken.json").write_text("{not-json")
    assert H.load_pair_summaries(root) == []


def test_source_of_truth_addendum_present():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "TRAINING_SOURCE_OF_TRUTH.md").read_text()
    assert "Current production status and proposed Historical League (2026-10-17 KST)" in text
    assert "2,764,800 matches" in text
    assert "57,344 / release pending" in text
