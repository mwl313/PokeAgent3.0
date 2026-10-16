"""Focused verification for the V2 evaluation league (spec section 12)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from agent.contracts import execution_identity
from agent.evaluation.fixed_policy import FixedPolicyEvaluator, build_fixture_manifest
from agent.evaluation.identity import canonical_model_digest
from agent.evaluation.policy_archive import register_checkpoint, verify_archive, verify_entry
from agent.mock_engine import MockNativeEngine
from agent.model import build_model
from agent.ppo.experimental_checkpoint import hash_json


def _v2_payload(model, *, mode="E", matches=7, source="a" * 40, dataset_hash=None):
    payload = {
        "schema_version": 1,
        "kind": "v2_frozen_evaluation_snapshot",
        "learner": {
            "model": model.state_dict(),
            "contracts": execution_identity(mode),
            "config": {"model": model.config.to_dict()},
        },
        "source_sha": source,
        "model_config_hash": hash_json(model.config.to_dict()),
        "action_schema_hash": "c" * 64,
        "experiment": {"precision_mode": mode, "precision_stability": "fp16-reduction-off",
                       "precision_contract_version": execution_identity(mode)["precision_contract"]},
        "accounting": {"iteration": 1, "committed_matches": matches},
        "ddp": {"rank": 0, "world_size": 2, "communicator": "recreate"},
        "rank_digests": {str(i): {"model": "m", "optimizer": "o"} for i in (0, 1)},
    }
    if dataset_hash is not None:
        payload["dataset_manifest_hash"] = dataset_hash
    return payload


def _write_snapshot(path, model, **kwargs):
    payload = _v2_payload(model, **kwargs)
    torch.save(payload, path)
    return payload


def _small_model(seed=1):
    from agent.model import PA3Config
    torch.manual_seed(seed)
    return build_model(PA3Config(
        encoder_layers=2, d_model=64, attention_heads=2, head_dim=32, ffn_dim=128,
        category_vocab=256, category_embed_dim=8, role_vocab=12, prefix_hidden=64,
        critic_hidden=64, seed=seed))


def _native_model(seed=1):
    """Reduced-encoder model that still matches the real observation vocabulary."""
    from agent.model import PA3Config
    torch.manual_seed(seed)
    return build_model(PA3Config(
        encoder_layers=2, d_model=64, attention_heads=2, head_dim=32, ffn_dim=128,
        category_embed_dim=8, prefix_hidden=64, critic_hidden=64, seed=seed))


def test_v1_checkpoint_rejected_by_v2_archive(tmp_path):
    from agent.model import PA3Config
    model = _small_model(1)
    payload = _v2_payload(model)
    payload["learner"]["contracts"] = {
        "observation_schema": "pa3-entity-observation-v1", "action_schema": "atomic-action-v1",
        "semantic_hash": "0" * 64, "ragged_contract_hash": "0" * 64,
        "precision_contract": "pa3-precision-A-v2",
    }
    checkpoint = tmp_path / "v1.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="V2 contract identity|contract mismatch"):
        register_checkpoint(tmp_path / "archive.json", policy_id="v1", label="v1",
                            checkpoint_path=checkpoint)


def test_v2_mode_e_snapshot_registers_and_loads(tmp_path):
    from agent.evaluation.fixed_policy import load_frozen_model
    model = _small_model(2)
    checkpoint = tmp_path / "e.pt"
    _write_snapshot(checkpoint, model, mode="E")
    entry = register_checkpoint(tmp_path / "archive.json", policy_id="e", label="E",
                                checkpoint_path=checkpoint, committed_matches=7)
    assert entry["precision_mode"] == "E"
    assert entry["precision_contract_version"] == "pa3-precision-E-v1"
    assert entry["observation_schema"] == "pa3-entity-observation-v2"
    assert entry["action_schema"] == "atomic-action-v2"
    assert verify_entry(tmp_path / "archive.json", entry) == str(checkpoint)
    loaded = load_frozen_model(checkpoint, device="cpu")
    assert loaded.precision_contract.identity == "pa3-precision-E-v1"
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_v2_archive_rejects_tampered_snapshot(tmp_path):
    model = _small_model(3)
    checkpoint = tmp_path / "e.pt"
    _write_snapshot(checkpoint, model, mode="E")
    register_checkpoint(tmp_path / "archive.json", policy_id="e", label="E", checkpoint_path=checkpoint)
    entry = json.load(open(tmp_path / "archive.json"))["policies"]["e"]
    torch.save({"not": "the artifact"}, checkpoint)
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_entry(tmp_path / "archive.json", entry)


def test_frozen_evaluation_does_not_mutate_model(tmp_path):
    model = _small_model(4)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    engine = MockNativeEngine(num_teams=4, requests_per_match=2)
    fixtures = build_fixture_manifest(pairs=3, team_count=4, seed=9)
    result = FixedPolicyEvaluator(engine, device="cpu", seed=9).run(
        fixtures, policy="checkpoint", model=model, opponent="random")
    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert result["matches"] == len(fixtures)


def test_scheduling_milestones_coalescing_and_prestart():
    from agent.evaluation.v2_league import update_schedule, read_json
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "schedule.json"
        snapshots = [
            {"id": "baseline", "committed_matches": 57344},
            {"id": "c102400", "committed_matches": 102400},
            {"id": "c200704", "committed_matches": 200704},
        ]
        state = update_schedule(path, snapshots, 249856, campaign_start=57344)
        boundaries = state["boundaries"]
        assert boundaries["50000"]["status"] == "skipped_prestart"
        assert boundaries["100000"]["status"] == "approximate"
        assert boundaries["100000"]["checkpoint_id"] == "c102400"
        assert boundaries["150000"]["status"] == "approximate"
        assert boundaries["150000"]["checkpoint_id"] == "c200704"
        assert boundaries["200000"]["status"] == "coalesced"
        assert boundaries["200000"]["checkpoint_id"] == "c200704"
        jobs = state["jobs"]
        assert jobs["light:c102400"]["tier"] == "light"
        assert jobs["full:c200704"]["tier"] == "full"
        assert jobs["full:c200704"]["scheduled_boundaries"] == [150000, 200000]
        # Idempotent restart: no duplicates, no status churn.
        again = update_schedule(path, snapshots, 249856, campaign_start=57344)
        assert sorted(again["jobs"]) == sorted(jobs)
        assert len(again["jobs"]) == 2


def test_scheduling_late_and_missing_boundaries():
    from agent.evaluation.v2_league import update_schedule
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "schedule.json"
        # Milestone 400000 crossed but no checkpoint at/after it yet.
        state = update_schedule(path, [{"id": "c352256", "committed_matches": 352256}], 401408,
                                campaign_start=300000)
        assert state["boundaries"]["300000"]["status"] == "approximate"
        assert state["boundaries"]["300000"]["checkpoint_id"] == "c352256"
        assert state["boundaries"]["350000"]["status"] == "coalesced"
        assert state["boundaries"]["400000"]["status"] == "deferred"
        assert "no published checkpoint" in state["boundaries"]["400000"]["reason"]
        # A later checkpoint serves the deferred milestone, not an invented model.
        state = update_schedule(path, [
            {"id": "c352256", "committed_matches": 352256},
            {"id": "c403456", "committed_matches": 403456},
        ], 403456, campaign_start=300000)
        assert state["boundaries"]["400000"]["status"] == "approximate"
        assert state["boundaries"]["400000"]["checkpoint_id"] == "c403456"


def test_plan_panels_redistributes_budget():
    from agent.evaluation.v2_league import FULL_GAMES, LIGHT_GAMES, plan_panels
    one = plan_panels(tier="full", candidate_id="c", incumbent_id="b", baseline_id="b", anchors=[])
    assert sum(panel["games"] for panel in one) == FULL_GAMES
    assert len(one) == 1
    two = plan_panels(tier="full", candidate_id="c", incumbent_id="b", baseline_id="x", anchors=[])
    assert sum(panel["games"] for panel in two) == FULL_GAMES
    assert len(two) == 2
    three = plan_panels(tier="light", candidate_id="c", incumbent_id="i", baseline_id="b", anchors=["a"])
    assert sum(panel["games"] for panel in three) == LIGHT_GAMES
    assert len(three) == 3
    assert len({panel["opponent_id"] for panel in three}) == 3


def _outcomes(pair_scores):
    rows = []
    for index, score in enumerate(pair_scores):
        outcome = "win" if score == 1.0 else "draw" if score == 0.5 else "loss"
        for side in (0, 1):
            rows.append({"fixture_id": 2 * index + side, "pair_id": index, "outcome": outcome})
    return rows


def test_promotion_conservative_and_evidence_guards():
    from agent.evaluation.v2_league import promotion_decision
    strong = promotion_decision(_outcomes([1.0] * 256), comparison_index=1, opponent_index=1,
                                stage_is_final=True)
    assert strong["decision"] == "promote" and strong["lower"] > 0.5
    close = promotion_decision(_outcomes(([1.0, 0.0] * 128)), comparison_index=1, opponent_index=1,
                               stage_is_final=True)
    assert close["decision"] == "retain"
    invalid = _outcomes([1.0] * 64) + [{"fixture_id": 999, "pair_id": 999, "outcome": "error"}]
    assert promotion_decision(invalid, comparison_index=1, opponent_index=1)["decision"] == "invalid"
    escalating = promotion_decision(_outcomes([1.0] * 128 + [0.0] * 4), comparison_index=1,
                                    opponent_index=1, stage=0)
    assert escalating["decision"] in {"escalate", "promote"}


def test_regression_detection():
    from agent.evaluation.v2_league import evaluate_regression, paired_difference
    candidate = _outcomes([1.0] * 128)
    incumbent = _outcomes([0.5] * 128)
    result = evaluate_regression(candidate_baseline_outcomes=candidate,
                                 incumbent_baseline_outcomes=incumbent)
    assert result["status"] == "acceptable" and result["ci95"][0] > 0
    regressed = evaluate_regression(candidate_baseline_outcomes=_outcomes([0.0] * 128),
                                    incumbent_baseline_outcomes=_outcomes([1.0] * 128))
    assert regressed["status"] == "regression_warning"
    assert paired_difference([], []) is None


def test_heldout_overlap_detection(tmp_path):
    from agent.evaluation.v2_league import build_heldout_manifest, heldout_status
    training = tmp_path / "training.json"
    training.write_text(json.dumps([{"id": "team-a"}, {"id": "team-b"}]))
    overlap = tmp_path / "overlap.json"
    overlap.write_text(json.dumps([{"id": "team-b"}, {"id": "team-c"}]))
    with pytest.raises(ValueError, match="overlaps the frozen training pool"):
        build_heldout_manifest(overlap, training)
    disjoint = tmp_path / "disjoint.json"
    disjoint.write_text(json.dumps([{"id": "team-c"}, {"id": "team-d"}]))
    manifest = build_heldout_manifest(disjoint, training)
    assert manifest["overlap"] == 0 and manifest["heldout_teams"] == 2
    status = heldout_status(tmp_path / "heldout", training_teams_path=training, heldout_path=None)
    assert status["status"] == "WAITING_FOR_VALID_HELDOUT_DATA"
    # Cosmetic-only differences (renamed file) cannot fake disjointness.
    renamed = tmp_path / "renamed.json"
    renamed.write_text(json.dumps([{"members": [{"species": 1, "moves": ["a"]}]}]))
    training2 = tmp_path / "training2.json"
    training2.write_text(json.dumps([{"members": [{"species": 1, "moves": ["a"]}]}]))
    with pytest.raises(ValueError, match="overlaps"):
        build_heldout_manifest(renamed, training2)


def test_worker_restart_idempotency():
    from agent.evaluation.v2_league import mark_job, schedule_jobs, update_schedule
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "schedule.json"
        update_schedule(path, [{"id": "b", "committed_matches": 57344}], 60000, campaign_start=0)
        pending = schedule_jobs(path, statuses=("pending",))
        assert len(pending) == 1
        job_id = pending[0]["job_id"]
        mark_job(path, job_id, "completed", decided="retain")
        assert schedule_jobs(path, statuses=("pending",)) == []
        # A duplicate completion is a no-op and never regresses the job state.
        mark_job(path, job_id, "failed")
        assert schedule_jobs(path, statuses=("pending",)) == []
        assert json.load(open(path))["jobs"][job_id]["status"] == "completed"


def test_throughput_observation(tmp_path):
    from agent.evaluation.v2_league import production_throughput
    log = tmp_path / "rank0.log"
    log.write_text("\n".join(f"[rank0] iteration {i} complete in {90 + i * 0.1:.1f}s" for i in range(10)))
    gps = production_throughput(log, limit=8)
    assert gps is not None and 40 < gps < 50


def test_internal_ratings_diagnostic():
    from agent.evaluation.v2_league import internal_ratings
    receipts = [
        {"status": "success", "left_id": "a", "right_id": "b", "score_mean": 1.0, "matches": 256},
        {"status": "success", "left_id": "b", "right_id": "c", "score_mean": 0.5, "matches": 256},
        {"status": "success", "left_id": "a", "right_id": "c", "score_mean": 0.7, "matches": 256},
    ]
    result = internal_ratings(receipts)
    assert result["status"] == "internal_relative_league_rating"
    assert result["ratings"]["a"] > result["ratings"]["c"]


def _native_engine_available():
    try:
        sys_path = str(Path(__file__).resolve().parents[2] / "engine" / "python")
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        import pa3_engine  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _native_engine_available(), reason="native engine build unavailable")
def test_end_to_end_native_smoke(tmp_path):
    """Genuine CPU paired evaluation through the real native engine."""
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "engine" / "python"))
    import pa3_engine
    from agent.evaluation.v2_league import (
        evaluation_environment, master_fixtures, run_matchup_cached,
    )
    data = root / "engine" / "data"
    teams = data / "a10-training-teams.json"
    if not teams.is_file():
        pytest.skip("frozen team pool unavailable")
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    _write_snapshot(left, _native_model(5), mode="E", matches=100)
    _write_snapshot(right, _native_model(6), mode="E", matches=200)
    registry_path = tmp_path / "registry.json"
    register_checkpoint(registry_path, policy_id="left", label="left", checkpoint_path=left, committed_matches=100)
    register_checkpoint(registry_path, policy_id="right", label="right", checkpoint_path=right, committed_matches=200)
    engine = pa3_engine.NativeEngine(str(data), str(teams), workers=1)
    environment = evaluation_environment(
        str(root / "engine" / "python" / "pa3_engine" / "pa3_engine.so"), str(data), str(teams),
        backend="batched", batch_size=4, device="cpu", threads=1, engine_workers=1)
    fixtures = master_fixtures(engine, pairs=2, seed=20261014)
    receipt, cached = run_matchup_cached(registry_path, tmp_path / "receipts", engine, fixtures,
                                         "left", "right", seed=20261014, max_rounds=2000,
                                         environment=environment, batch_size=4)
    cached_receipt, cached_again = run_matchup_cached(registry_path, tmp_path / "receipts", engine,
                                                      fixtures, "left", "right", seed=20261014,
                                                      max_rounds=2000, environment=environment,
                                                      batch_size=4)
    assert receipt["matches"] == len(fixtures)
    assert receipt["wins"] + receipt["losses"] + receipt["draws"] == len(fixtures)
    assert receipt["status"] == "success"
    assert cached_again and cached_receipt["identity"] == receipt["identity"]
