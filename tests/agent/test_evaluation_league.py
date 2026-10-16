import copy
import json

import pytest
import torch

from agent.evaluation.fixed_policy import build_fixture_manifest
from agent.evaluation.league import matchup_matrix, regression_report, run_pair
from agent.evaluation.identity import canonical_model_digest
from agent.evaluation.frozen_policy import FrozenPPOPolicy
from agent.evaluation.policy_archive import register_checkpoint, verify_entry
from agent.evaluation.schedule import bind_job, boundary_jobs, complete_job, pending_jobs
from agent.mock_engine import MockNativeEngine
from agent.model import build_model


def _checkpoint(path, *, seed=1, matches=7, precision_mode=None):
    torch.manual_seed(seed)
    from agent.contracts import execution_identity
    from agent.ppo.experimental_checkpoint import hash_json
    model=build_model()
    model.configure_precision(precision_mode or 'A')
    payload={"learner": {"model": model.state_dict(), "contracts": execution_identity(precision_mode or 'A'),
                          "config": {'model':model.config.to_dict()}}, "source_sha": "a" * 40,
                "model_config_hash": hash_json(model.config.to_dict()), "action_schema_hash": "c" * 64,
                "accounting": {"committed_matches": matches},
                "ddp": {'world_size':2},
                "rank_digests": {str(i): {"model": "x", "optimizer": "y"} for i in (0,1)}}
    if precision_mode is not None:
        payload['experiment'] = {'precision_mode':precision_mode,
                                 'precision_stability':'fp16-reduction-off',
                                 'precision_contract_version':model.precision_contract.identity}
    torch.save(payload, path)


def test_schedule_all_cycles_restart_missing_and_idempotent(tmp_path):
    state = tmp_path / "schedule.json"
    assert not boundary_jobs(state, 402255)
    created = boundary_jobs(state, 802256, checkpoint={"id": "safe"})
    assert [(x["id"], x["kind"]) for x in created] == [("light:402256", "light"), ("light:452256", "light"), ("light:502256", "light"), ("full:552256", "full"), ("light:602256", "light"), ("light:652256", "light"), ("light:702256", "light"), ("full:752256", "full"), ("light:802256", "light")]
    assert not boundary_jobs(state, 802256, checkpoint={"id": "safe"})
    assert len(pending_jobs(state)) == 9
    # Catch-up assigns one safe checkpoint only to the earliest crossed
    # boundary; later boundaries remain explicit historical misses.
    jobs = pending_jobs(state)
    earliest = next(job for job in jobs if job["id"] == "light:402256")
    assert earliest["captured_matches"] == 802256 and any(job["missing_snapshot"] for job in jobs if job["id"] != earliest["id"])
    exact = tmp_path / "exact.json"
    checkpoint = {"file_sha256": "sha"}
    boundary_jobs(exact, 402256, checkpoint=checkpoint)
    plan = {"candidate_id": "c", "candidate_model_digest": "d", "panels": [{"opponent_id": "o", "games": 2}]}
    bind_job(exact, "light:402256", plan)
    receipt = {"status": "success", "job_id": "light:402256", "scheduled_boundary": 402256, "checkpoint_sha256": "sha", "candidate_id": "c", "candidate_model_digest": "d", "matchups": [{"opponent_id": "o", "matches": 2, "identity": "r", "fixture_manifest_hash": "f", "status": "success"}]}
    assert complete_job(exact, "light:402256", receipt)["status"] == "completed"
    assert complete_job(exact, "light:402256", receipt)["status"] == "completed"
    with pytest.raises(ValueError): complete_job(exact, "light:402256", {**receipt, "checkpoint_sha256": "other"})
    missing = tmp_path / "missing.json"
    assert boundary_jobs(missing, 402256)[0]["missing_snapshot"]
    assert pending_jobs(missing)[0]["status"] == "deferred"


def test_non_aligned_boundary_uses_first_safe_checkpoint_after_threshold(tmp_path):
    state = tmp_path / "nonaligned.json"
    assert boundary_jobs(state, 401408) == []
    first = boundary_jobs(state, 402256)
    assert first[0]["missing_snapshot"] and first[0]["status"] == "deferred"
    assigned = boundary_jobs(state, 403456, checkpoint={"file_sha256": "after"})
    assert assigned[0]["captured_matches"] == 403456
    assert assigned[0]["scheduled_boundary"] == 402256
    again = boundary_jobs(state, 403456, checkpoint={"file_sha256": "after"})
    assert again == []


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


def test_canonical_digest_matches_frozen_policy_and_is_order_independent():
    model = build_model()
    digest = canonical_model_digest(model.state_dict())
    reordered = {key: model.state_dict()[key] for key in reversed(list(model.state_dict()))}
    assert digest == canonical_model_digest(reordered)
    adapter = FrozenPPOPolicy(model, device="cpu")
    assert adapter.identity.model_digest == digest


def test_mode_e_archive_and_frozen_loaders_preserve_precision_identity(tmp_path):
    from agent.evaluation.fixed_policy import load_frozen_model

    checkpoint, registry = tmp_path/'e.pt', tmp_path/'archive.json'
    _checkpoint(checkpoint, precision_mode='E')
    entry = register_checkpoint(registry, policy_id='e', label='E', checkpoint_path=checkpoint)
    assert entry['precision_mode'] == 'E'
    assert entry['precision_contract_version'] == 'pa3-precision-E-v1'
    assert verify_entry(registry, entry) == str(checkpoint)
    reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    try:
        adapter = FrozenPPOPolicy.from_checkpoint(checkpoint, device='cpu')
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    model = load_frozen_model(checkpoint, device='cpu')
    assert adapter.identity.precision_contract == model.precision_contract.identity == 'pa3-precision-E-v1'
    assert model.precision_contract.quantize_observation and not model.precision_contract.encoder_amp
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert adapter.identity.model_digest == canonical_model_digest(model.state_dict())
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    payload['experiment']['precision_mode'] = 'D'
    changed = tmp_path/'mismatched.pt'
    torch.save(payload, changed)
    for loader in (FrozenPPOPolicy.from_checkpoint, load_frozen_model):
        with pytest.raises(ValueError, match='precision contract mismatch'):
            loader(changed, device='cpu')


@pytest.mark.parametrize('mode,quantized', [('A', False), ('B', False), ('C', True),
                                         ('D', True), ('E', True)])
def test_evaluation_identity_records_actual_observation_quantization(mode, quantized):
    from agent.evaluation.league import evaluation_identity
    from agent.model.precision import get_precision_contract

    fixtures = build_fixture_manifest(pairs=1, team_count=4, seed=3)
    entry = {'id':'policy', 'file_sha256':'a'*64, 'model_digest':'b'*64,
             'precision_mode':mode,
             'precision_contract_version':get_precision_contract(mode).identity}
    identity, body = evaluation_identity(entry, entry, fixtures, seed=5, max_rounds=20,
                                         environment={'backend':'batched'})
    expected = 'fp16-round-continuous-v1' if quantized else 'none'
    assert body['candidate'][3]['observation_quantization'] == expected
    assert body['opponent'][3]['observation_quantization'] == expected
    assert body['candidate'][3]['precision_contract_version'] == get_precision_contract(mode).identity
    if mode == 'E':
        historical = dict(entry, precision_mode='D', precision_contract_version='pa3-precision-D-v2')
        old_identity, _ = evaluation_identity(historical, historical, fixtures, seed=5,
                                              max_rounds=20, environment={'backend':'batched'})
        assert identity != old_identity
