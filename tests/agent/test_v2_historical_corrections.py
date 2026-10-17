import json
import math
import random
from pathlib import Path

import pytest

from agent.evaluation import historical_league as H
from agent.evaluation import historical_resources as R
from agent.evaluation import historical_schedule as S


def _entries(count=30, step=100000, start=100000, distinct=True):
    rows = []
    for index in range(count):
        rows.append({
            "id": f"p{index:02d}",
            "committed_matches": start + index * step,
            "model_digest": f"digest-{index:02d}" if distinct else "same",
            "file_sha256": f"file-{index:02d}",
            "model_digest_algorithm": "pa3-model-state-sha256-sorted-v1",
            "path": f"/tmp/p{index:02d}.pt",
        })
    return rows


def _summary(left, right, *, left_score, games=128, status="success",
             fixture_pairs=64, tier="basic"):
    left_wins = round(left_score * games)
    return {
        "schema": H.RR_PAIR_SCHEMA,
        "pair_key": H.canonical_pair(left["id"], right["id"]),
        "left_id": left["id"],
        "right_id": right["id"],
        "left_clock": left["committed_matches"],
        "right_clock": right["committed_matches"],
        "left_model_digest": left["model_digest"],
        "right_model_digest": right["model_digest"],
        "tier": tier,
        "fixture_pairs": fixture_pairs,
        "games": games,
        "matches": games,
        "left_wins": left_wins,
        "left_losses": games - left_wins,
        "draws": 0,
        "left_score": left_score,
        "right_score": 1.0 - left_score,
        "paired_fixtures": fixture_pairs,
        "ci95_left": [max(0.0, left_score - 0.1), min(1.0, left_score + 0.1)],
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


def _round_robin(entries, scores):
    summaries = []
    for index, left in enumerate(entries):
        for right in entries[index + 1:]:
            summaries.append(_summary(left, right, left_score=scores((left, right))))
    return summaries


def test_bt_corrected_fixed_point_and_independent_likelihood():
    entries = _entries(6)
    rng = random.Random(7)
    summaries = _round_robin(entries, lambda pair: rng.uniform(0.25, 0.75))
    fit = H.bradley_terry_fit(summaries, entries, bootstrap_draws=0)
    ratings = fit["ratings"]
    assert fit["diagnostics"]["converged"] is True
    assert fit["diagnostics"]["iterations"] > 1
    assert fit["diagnostics"]["schema"] == H.RR_RATING_SCHEMA
    # Correct MM fixed point satisfies the score equations: observed wins equal
    # expected wins for every policy.  The old update did not.
    stats = H._bt_stats(summaries)
    adjacency = H._bt_adjacency(stats, [entry["id"] for entry in entries])
    for policy_id, row in ratings.items():
        observed = sum(wins for wins, _games in adjacency[policy_id].values())
        expected = sum(
            games * row["strength"]
            / (row["strength"] + ratings[opponent]["strength"])
            for opponent, (_wins, games) in adjacency[policy_id].items()
        )
        assert abs(observed - expected) < 1e-7
    # Reproducible.
    again = H.bradley_terry_fit(summaries, entries, bootstrap_draws=0)
    assert again["ratings"] == ratings


def test_bt_differs_from_deprecated_wrong_fixed_point():
    entries = _entries(4)
    summaries = [
        _summary(entries[0], entries[1], left_score=0.9),
        _summary(entries[1], entries[2], left_score=0.9),
        _summary(entries[2], entries[3], left_score=0.9),
        _summary(entries[3], entries[0], left_score=0.9),
        _summary(entries[0], entries[2], left_score=0.8),
        _summary(entries[1], entries[3], left_score=0.8),
    ]
    corrected = H.bradley_terry_fit(summaries, entries)["ratings"]
    # Deprecated implementation for contrast.
    ids = [entry["id"] for entry in entries]
    results = {}
    for summary in summaries:
        left, right = summary["left_id"], summary["right_id"]
        results[(left, right)] = (summary["left_score"], summary["games"])
        results[(right, left)] = (summary["right_score"], summary["games"])
    strength = {policy_id: 1.0 for policy_id in ids}
    for _ in range(200):
        updated = {}
        for policy_id in ids:
            numerator = denominator = 0.0
            for opponent in ids:
                if opponent == policy_id:
                    continue
                row = results.get((policy_id, opponent))
                if not row:
                    continue
                score, games = row
                numerator += games * score
                denominator += games * strength[policy_id] / (
                    strength[policy_id] + strength[opponent])
            updated[policy_id] = numerator / denominator
        geometric = math.exp(
            sum(math.log(value) for value in updated.values()) / len(updated))
        strength = {key: value / geometric for key, value in updated.items()}
    for policy_id in ids:
        if abs(corrected[policy_id]["rating"] - math.log(strength[policy_id])) > 1e-6:
            break
    else:
        raise AssertionError("corrected and deprecated fits should differ")


def test_bt_pair_cluster_bootstrap_is_deterministic():
    entries = _entries(8)
    rng = random.Random(11)
    summaries = _round_robin(entries, lambda pair: rng.uniform(0.2, 0.8))
    first = H.bradley_terry_fit(summaries, entries, bootstrap_draws=40, seed=5)
    second = H.bradley_terry_fit(summaries, entries, bootstrap_draws=40, seed=5)
    for policy_id in first["ratings"]:
        assert first["ratings"][policy_id]["ci95_low"] == \
            second["ratings"][policy_id]["ci95_low"]
    assert first["uncertainty"]["resampling_unit"].startswith("pair summary")
    assert all(
        row["ci95_low"] <= row["ci95_high"]
        for row in first["ratings"].values())


def _state_root(tmp_path):
    root = tmp_path / "rr"
    root.mkdir()
    return root


def test_milestone_overshoot_exact_and_missed_boundaries():
    entries = [
        _entries(1, start=2_600_000)[0],
        _entries(1, start=2_700_000)[0],
        _entries(1, start=2_802_944)[0],
    ]
    entries[0]["id"], entries[1]["id"], entries[2]["id"] = "a", "b", "c"
    assert H.latest_full_boundary(2_802_944) == 2_800_000
    assert H.latest_full_boundary(2_800_000) == 2_800_000
    assert H.latest_full_boundary(2_799_999) == 2_600_000
    assert H.milestone_satisfied(2_802_944, 2_800_000) is True
    chosen = S.checkpoint_for_milestone(entries, 2_800_000)
    assert chosen["committed_matches"] == 2_802_944
    # Downtime: one checkpoint covers several missed milestones.
    sparse = [_entries(1, start=2_800_000)[0], _entries(1, start=3_100_000)[0]]
    sparse[0]["id"] = "old"
    sparse[1]["id"] = "new"
    assert S.checkpoint_for_milestone(sparse, 2_800_000)["id"] == "old"
    assert S.checkpoint_for_milestone(sparse, 3_000_000)["id"] == "new"


def test_schedule_is_idempotent_and_formal_priority_wins(tmp_path):
    root = _state_root(tmp_path)
    entries = _entries(10, start=2_600_000, step=20_000)
    entries[-1]["committed_matches"] = 2_802_944
    summaries = [_summary(entries[0], entries[1], left_score=0.5)]
    formal = {"pending_or_running": 0, "jobs": []}
    state = S.default_state()
    first = S.plan_milestones(
        entries, state, root=root, summaries=summaries, formal=formal,
        champion_id=entries[0]["id"], reconcile=False)
    keys = {job["milestone_key"] for job in first["new_jobs"]}
    assert "a:2800000" in keys
    milestone_job = next(
        job for job in first["new_jobs"] if job["milestone_key"] == "a:2800000")
    assert milestone_job["checkpoint_clock"] == 2_802_944
    assert milestone_job["checkpoint_id"] == entries[-1]["id"]
    second = S.plan_milestones(
        entries, state, root=root, summaries=summaries, formal=formal,
        champion_id=entries[0]["id"], reconcile=False)
    assert second["new_jobs"] == []
    assert len(second["due"]) == len(first["new_jobs"])
    deferred = S.plan_milestones(
        entries, state, root=root, summaries=summaries,
        formal={"pending_or_running": 1, "jobs": ["formal"]},
        champion_id=entries[0]["id"], reconcile=False)
    assert deferred["state"] == "deferred_formal_priority"
    assert deferred["new_jobs"] == []


def test_schedule_restart_duplicate_and_corrupt_receipt(tmp_path):
    root = _state_root(tmp_path)
    (root / "receipts" / "milestones").mkdir(parents=True)
    entries = _entries(5, start=2_700_000, step=20_000)
    entries[-1]["committed_matches"] = 2_802_944
    summaries = [_summary(entries[0], entries[1], left_score=0.5)]
    state = S.default_state()
    plan = S.plan_milestones(
        entries, state, root=root, summaries=summaries,
        formal={"pending_or_running": 0}, champion_id=entries[0]["id"],
        reconcile=False)
    tier_a = next(job for job in plan["new_jobs"] if job["tier"] == "A")
    receipt_path = Path(tier_a["receipt_path"])
    # Interrupted job: no receipt, same job resumes, no duplicate is created.
    resumed = S.plan_milestones(
        entries, state, root=root, summaries=summaries,
        formal={"pending_or_running": 0}, champion_id=entries[0]["id"],
        reconcile=True)
    assert resumed["new_jobs"] == []
    assert any(row["milestone_key"] == tier_a["milestone_key"]
               for row in resumed["resumed_jobs"])
    # Corrupt receipt fails safe and never completes.
    receipt_path.write_text("{not-json")
    reconcile = S.reconcile_receipts(root, state)
    assert any(row["status"] == "corrupt_receipt" for row in reconcile)
    assert state["milestones"][tier_a["milestone_key"]]["status"] == "corrupt_receipt"
    after = S.plan_milestones(
        entries, state, root=root, summaries=summaries,
        formal={"pending_or_running": 0}, champion_id=entries[0]["id"],
        reconcile=False)
    assert all(job["milestone_key"] != tier_a["milestone_key"]
               for job in after["new_jobs"])
    # After operator review removes the corrupt file, the job resumes.
    receipt_path.unlink()
    recovered = S.plan_milestones(
        entries, state, root=root, summaries=summaries,
        formal={"pending_or_running": 0}, champion_id=entries[0]["id"],
        reconcile=False)
    assert any(row["milestone_key"] == tier_a["milestone_key"]
               for row in recovered["resumed_jobs"])
    assert state["milestones"][tier_a["milestone_key"]]["status"] == "planned"
    # A valid matching receipt completes the job exactly once.
    state2 = S.default_state()
    plan2 = S.plan_milestones(
        entries, state2, root=root, summaries=summaries,
        formal={"pending_or_running": 0}, champion_id=entries[0]["id"],
        reconcile=False)
    job2 = next(job for job in plan2["new_jobs"] if job["tier"] == "A")
    Path(job2["receipt_path"]).write_text(json.dumps({
        "schema": S.JOB_SCHEMA,
        "job_id": job2["job_id"],
        "milestone_key": job2["milestone_key"],
        "checkpoint_model_digest": job2["checkpoint_model_digest"],
        "status": "success",
        "games": 2048,
        "completed_at": 1,
    }))
    changes = S.reconcile_receipts(root, state2)
    assert state2["milestones"][job2["milestone_key"]]["status"] == "completed"
    assert any(row["status"] == "completed" for row in changes)
    assert S.reconcile_receipts(root, state2) == []


def test_tier_b_supersedes_tier_a_and_caps_distinct_digests(tmp_path):
    root = _state_root(tmp_path)
    entries = _entries(30, start=1_000_000, step=100_000)
    entries[-1]["committed_matches"] = 3_900_000
    summaries = []
    state = S.default_state()
    plan = S.plan_milestones(
        entries, state, root=root, summaries=summaries,
        formal={"pending_or_running": 0}, champion_id=entries[0]["id"],
        reconcile=False)
    keys = {job["milestone_key"] for job in plan["new_jobs"]}
    assert "b:3800000" in keys
    assert "a:3800000" not in keys
    assert state["milestones"]["a:3800000"]["status"] == "covered_by_tier_b"
    tier_b = next(job for job in plan["new_jobs"] if job["tier"] == "B")
    assert len(tier_b["participants"]) <= S.TIER_B_POLICY_CAP
    digests = [row["model_digest"] for row in tier_b["participants"]]
    assert len(digests) == len(set(digests))
    ids = {row["policy_id"] for row in tier_b["participants"]}
    assert entries[-1]["id"] in ids
    assert entries[0]["id"] in ids


def test_participant_selection_dedup_and_always_included():
    entries = _entries(40, start=1_000_000, step=100_000)
    entries.append({
        **entries[-1],
        "id": "duplicate-digest",
        "committed_matches": 9_000_000,
    })
    selected = S.select_tier_b_participants(
        entries, [], champion_id=entries[0]["id"],
        latest_id=entries[-2]["id"], promotions_dir=None, cap=24)
    rows = selected["participants"]
    assert len(rows) <= 24
    assert selected["unique_digests"] == len(rows)
    ids = {row["policy_id"] for row in rows}
    assert entries[0]["id"] in ids
    assert entries[-2]["id"] in ids
    assert "duplicate-digest" not in ids
    assert selected["always_included"] == {
        "latest": entries[-2]["id"], "champion": entries[0]["id"]}


def test_pair_cache_identity_is_strict():
    summary = _summary(_entries(2)[0], _entries(2)[1], left_score=0.5)
    left, right = _entries(2)
    assert S.pair_cache_compatible(
        summary, left_digest=left["model_digest"], right_digest=right["model_digest"],
        fixture_pairs=64, environment={"backend": "batched"})
    assert not S.pair_cache_compatible(
        summary, left_digest="wrong", right_digest=right["model_digest"],
        fixture_pairs=64, environment={"backend": "batched"})
    assert not S.pair_cache_compatible(
        summary, left_digest=left["model_digest"], right_digest=right["model_digest"],
        fixture_pairs=256, environment={"backend": "batched"})
    assert not S.pair_cache_compatible(
        summary, left_digest=left["model_digest"], right_digest=right["model_digest"],
        fixture_pairs=64, environment={"backend": "different"})


def _fake_topology(cores=40):
    cpus = {}
    core_map = {}
    nodes = {0: [], 1: []}
    for core in range(cores):
        node = 0 if core < 20 else 1
        first = core
        second = core + 40
        siblings = [first, second]
        for cpu in siblings:
            cpus[cpu] = {
                "cpu": cpu, "core_id": core, "package_id": node,
                "node": node, "siblings": siblings,
            }
        core_map[(node, core)] = {
            "key": [node, core], "core_id": core, "package_id": node,
            "node": node, "cpus": siblings,
        }
        nodes[node].extend(siblings)
    return {
        "cpus": cpus, "cores": core_map, "nodes": nodes,
        "logical_cpus": len(cpus), "physical_cores": len(core_map),
        "smt_enabled": True,
    }


def test_topology_masks_exclude_formal_cpus_and_smt_siblings():
    topology = _fake_topology()
    masks = R.derive_worker_masks(topology, workers=4, threads_per_worker=2)
    assert len(masks) == 4
    reserved = R.reserved_cpus(R.DEFAULT_FORMAL_CPUS, topology)
    assert set(range(4, 8)).issubset(reserved)
    assert set(range(44, 48)).issubset(reserved)
    assert R.mask_overlaps(masks, reserved) == []
    used = [cpu for row in masks for cpu in row["cpus"]]
    assert len(used) == len(set(used))
    assert not (set(used) & reserved)


def test_mask_conflict_detection_fails_closed():
    masks = [
        {"worker": 1, "cpus": [4, 8]},
        {"worker": 2, "cpus": [8, 9]},
    ]
    conflicts = R.mask_overlaps(masks, {4, 5, 6, 7})
    reasons = {row["reason"] for row in conflicts}
    assert "overlaps_reserved_cpus" in reasons
    assert "overlaps_other_worker" in reasons


def test_throughput_gate_thresholds_and_stale_samples():
    assert R.throughput_gate([380, 379])["action"] == "keep_conservative_default"
    assert R.throughput_gate([380, 381, 382])["action"] == "accept"
    reduced = R.throughput_gate([370, 369, 371])
    assert reduced["action"] == "reduce_concurrency"
    assert reduced["verified"] is True
    paused = R.throughput_gate([360, 358, 361])
    assert paused["action"] == "pause"


def test_resource_gate_pauses_for_formal_and_derives_mode_b(tmp_path):
    topology = _fake_topology()
    paused = R.evaluate_gate(
        ppo_running=False, formal={"pending_or_running": 1},
        topology=topology, disk_path=tmp_path, reserve_gib=0, mode_b_workers=4)
    assert paused["status"] == "paused"
    assert paused["workers"] == 0
    assert paused["coexistence_verified"] is False
    allowed = R.evaluate_gate(
        ppo_running=False, formal={"pending_or_running": 0},
        topology=topology, disk_path=tmp_path, reserve_gib=0, mode_b_workers=4)
    assert allowed["mode"] == "B_ppo_stopped"
    assert allowed["workers"] == 4
    assert allowed["conflicts"] == []
    mode_a = R.evaluate_gate(
        ppo_running=True, formal={"pending_or_running": 0},
        topology=topology, disk_path=tmp_path, reserve_gib=0,
        throughput_samples=[381, 382, 380])
    assert mode_a["mode"] == "A_ppo_running"
    assert mode_a["workers"] == 2
    assert mode_a["coexistence_verified"] is False


def test_shadow_plan_retirement_preserves_provenance(tmp_path):
    root = _state_root(tmp_path)
    original = {
        "schema": H.RR_SHADOW_SCHEMA,
        "status": "SHADOW_ONLY_NOT_ACTIVE",
        "shadow_only": True,
        "history_rows_target": 0,
        "history_probability": 0.0,
        "reference_policy": {"policy_id": "champion", "committed_matches": 2600960},
        "evidence": {"pair_summaries": 45},
        "opponents": [{"policy_id": "x"}],
        "plan_sha256": "old",
    }
    path = root / "opponent_plan.json"
    path.write_text(json.dumps(original, sort_keys=True))
    result = H.retire_shadow_plan(root)
    assert result["status"] == "OBSOLETE"
    assert result["activatable"] is False
    assert result["history_probability"] == 0.0
    archived = root / "opponent_plan.smoke-20261017.json"
    assert json.loads(archived.read_text()) == original
    assert result["provenance"]["original_pair_summaries"] == 45
    assert result["provenance"]["original_reference_policy_id"] == "champion"
    current = json.loads(path.read_text())
    assert current["status"] == "OBSOLETE"
    assert current["activation"] == "FORBIDDEN"
    assert H.retire_shadow_plan(root)["status"] == "OBSOLETE"
    with pytest.raises(PermissionError):
        H.load_opponent_plan(root, for_training=True)
    with pytest.raises(PermissionError):
        H.load_opponent_plan(root)


def test_run_validation_detects_duplicates_identity_and_game_errors():
    entries = _entries(3)
    plan = H.build_pair_plan(entries, entries[0]["id"], entries[-1]["id"])
    by_id = {entry["id"]: entry for entry in entries}
    summaries = [
        _summary(by_id[row["left_id"]], by_id[row["right_id"]], left_score=0.5,
                 games=row["games"], fixture_pairs=row["fixture_pairs"],
                 tier=row["tier"])
        for row in plan["pairs"]
    ]
    incomplete = H.validate_run(summaries[:2], entries, plan)
    assert incomplete["unique_pairs"] == 2
    assert any("planned pairs" in error for error in incomplete["errors"])
    validation = H.validate_run(summaries, entries, plan)
    assert validation["unique_pairs"] == 3
    assert validation["complete"] is True
    broken = [dict(row) for row in summaries]
    broken[0]["left_model_digest"] = "wrong"
    broken[1]["games"] = 127
    broken.append(dict(summaries[0]))
    errors = H.validate_run(broken, entries, plan)["errors"]
    assert any("model digest mismatch" in error for error in errors)
    assert any("game count mismatch" in error for error in errors)
    assert any("duplicate pair summary" in error for error in errors)


def test_published_full_run_is_complete_and_consistent():
    root = Path(__file__).resolve().parents[2] / "docs" / "operations" / "ahl-v2" / "results"
    entries, _meta = H.load_snapshot(root)
    summaries = H.load_pair_summaries(root)
    plan = H.read_json(root / "pair_plan.json")
    validation = H.validate_run(summaries, entries, plan)
    assert validation["ok"] is True
    assert validation["complete"] is True
    assert validation["unique_pairs"] == 435
    assert validation["total_games"] == 77184
    assert validation["tier_pairs"] == {"basic": 379, "fine": 56}


def test_published_ratings_match_expected_regression_positions():
    root = Path(__file__).resolve().parents[2] / "docs" / "operations" / "ahl-v2" / "results"
    entries, _meta = H.load_snapshot(root)
    summaries = H.load_pair_summaries(root)
    ratings = H.bradley_terry(summaries, entries)
    order = sorted(ratings, key=lambda policy_id: ratings[policy_id]["rank"])
    assert order[0] == "v2-000002764800-da35700f"
    assert order[3] == "v2-000002600960-da35700f"
    assert ratings["v2-000002764800-da35700f"]["rank"] == 1
    assert ratings["v2-000002600960-da35700f"]["rank"] == 4


def test_active_production_profiles_keep_history_disabled():
    root = Path(__file__).resolve().parents[2]
    for relative in (
            "configs/pa3_v2_production.yaml",
            "configs/validated_a94_runtime.yaml",
            "configs/a10_production.yaml"):
        text = (root / relative).read_text()
        assert '"historical_probability":0.0' in text.replace(" ", ""), relative
    # train.yaml is explicitly non-normative historical material.
    assert "historical_probability: 0.5" in (root / "configs/train.yaml").read_text()


def test_published_protected_verification_is_unchanged():
    root = Path(__file__).resolve().parents[2] / "docs" / "operations" / "ahl-v2" / "results"
    verification = json.loads(
        (root / "protected_snapshot" / "corrections_verification.json").read_text())
    assert verification["unchanged"] is True
    assert verification["critical_mismatches"] == []
    assert verification["training"]["head"].startswith("da35700")
    assert verification["training"]["stop_present"] is True
    assert verification["formal_root"]["alpha_sha256"].startswith("6fdf62a3")
    assert verification["writes_by_historical_diagnostics"] == 0
