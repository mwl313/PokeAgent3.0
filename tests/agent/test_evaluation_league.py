import copy
import json

import pytest
import torch

from agent.evaluation.fixed_policy import build_fixture_manifest
from agent.evaluation.league import matchup_matrix, regression_report, run_pair
from agent.evaluation.policy_archive import register_checkpoint, verify_entry
from agent.evaluation.schedule import boundary_jobs, complete_job, pending_jobs
from agent.mock_engine import MockNativeEngine
from agent.model import build_model


def _checkpoint(path, *, seed=1, matches=7):
    torch.manual_seed(seed)
    torch.save({"learner": {"model": build_model().state_dict()}, "source_sha": "a" * 40,
                "model_config_hash": "b" * 64, "action_schema_hash": "c" * 64,
                "accounting": {"committed_matches": matches},
                "rank_digests": {"0": {"model": "x", "optimizer": "y"}}}, path)


def test_schedule_all_cycles_restart_missing_and_idempotent(tmp_path):
    state = tmp_path / "schedule.json"
    assert not boundary_jobs(state, 402255)
    created = boundary_jobs(state, 802256, checkpoint={"id": "safe"})
    assert [(x["id"], x["kind"]) for x in created] == [("light:402256", "light"), ("light:452256", "light"), ("light:502256", "light"), ("full:552256", "full"), ("light:602256", "light"), ("light:652256", "light"), ("light:702256", "light"), ("full:752256", "full"), ("light:802256", "light")]
    assert not boundary_jobs(state, 802256, checkpoint={"id": "safe"})
    receipt = {"status": "success", "identity": "r"}
    assert complete_job(state, "light:402256", receipt)["status"] == "completed"
    assert complete_job(state, "light:402256", receipt)["status"] == "completed"
    with pytest.raises(ValueError): complete_job(state, "light:402256", {"status": "success", "identity": "other"})
    assert len(pending_jobs(state)) == 8  # state survives a fresh read
    missing = tmp_path / "missing.json"
    assert boundary_jobs(missing, 402256)[0]["missing_snapshot"]
    assert pending_jobs(missing)[0]["status"] == "deferred"


def test_archive_identity_latest_independent_and_league_deduplicates(tmp_path):
    first, second, registry = tmp_path / "one.pt", tmp_path / "two.pt", tmp_path / "archive.json"
    _checkpoint(first, seed=1); _checkpoint(second, seed=2)
    register_checkpoint(registry, policy_id="one", label="one", checkpoint_path=first, latest=True, incumbent=True)
    register_checkpoint(registry, policy_id="two", label="two", checkpoint_path=second)
    data = json.load(open(registry)); assert data["latest"] == data["incumbent"] == "one"
    path = verify_entry(registry, data["policies"]["one"]); assert path == str(first)
    bad = copy.deepcopy(data["policies"]["two"]); bad["model_digest"] = "0" * 64
    with pytest.raises(ValueError, match="model digest"): verify_entry(registry, bad)
    fixtures = build_fixture_manifest(pairs=2, team_count=4, seed=3)
    receipts = tmp_path / "receipts.json"; engine = MockNativeEngine(num_teams=4, requests_per_match=2)
    one, cached = run_pair(registry, receipts, engine, fixtures, "one", "two", device="cpu", seed=5)
    two, cached_again = run_pair(registry, receipts, MockNativeEngine(num_teams=4, requests_per_match=2), fixtures, "one", "two", device="cpu", seed=5)
    assert not cached and cached_again and one["identity"] == two["identity"]
    assert matchup_matrix(receipts)["one"]["two"]["matches"] == 4
    report = regression_report(receipts, candidate_id="one", incumbent_id="two")
    assert report["non_blocking"] and report["status"] in {"inconclusive", "regression_warning"}


def test_archive_sha_mismatch_rejected_without_touching_original(tmp_path):
    checkpoint, registry = tmp_path / "copy.pt", tmp_path / "archive.json"
    _checkpoint(checkpoint); register_checkpoint(registry, policy_id="copy", label="copy", checkpoint_path=checkpoint)
    data = json.load(open(registry)); torch.save({"not": "the registered artifact"}, checkpoint)
    with pytest.raises(ValueError, match="SHA mismatch"): verify_entry(registry, data["policies"]["copy"])
