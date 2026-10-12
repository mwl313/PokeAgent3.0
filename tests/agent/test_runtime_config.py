import json
from agent.runtime.config import resolve_profile

PROFILE = "configs/validated_a94_runtime.yaml"

def test_validated_profile_resolves_with_hash():
    result = resolve_profile(PROFILE)
    assert result["profile"]["training"]["ppo_epochs"] == 2
    assert result["profile"]["precision"]["stability"] == "fp16-reduction-off"
    assert len(result["config_hash"]) == 64

def test_profile_rejects_epoch_drift(tmp_path):
    data = json.load(open(PROFILE)); data["training"]["ppo_epochs"] = 4
    path = tmp_path / "bad.yaml"; path.write_text(json.dumps(data))
    try: resolve_profile(path)
    except ValueError as exc: assert "invariant" in str(exc)
    else: raise AssertionError("epoch drift was accepted")

def test_profile_rejects_dataset_hash_drift(tmp_path):
    data = json.load(open(PROFILE)); data["dataset"]["manifest_sha256"] = "0" * 64
    path = tmp_path / "bad.yaml"; path.write_text(json.dumps(data))
    try: resolve_profile(path)
    except ValueError as exc: assert "manifest hash" in str(exc)
    else: raise AssertionError("dataset drift was accepted")
