"""Strict, hashable resolver for the validated A9.4 runtime profile.

The profile is JSON-compatible YAML so it can be read without an optional YAML
dependency while remaining valid YAML for standard tooling.
"""
from __future__ import annotations
import hashlib, json, os

SCHEMA = "pa3-validated-runtime-v1"
ALLOWED = {"schema", "profile", "version", "training", "precision", "dataset", "collector", "evaluation", "evidence"}

def load_profile(path: str) -> dict:
    path = os.path.abspath(os.fspath(path))
    with open(path) as f: data = json.load(f)
    if set(data) - ALLOWED or data.get("schema") != SCHEMA:
        raise ValueError("unsupported or unknown runtime profile schema/keys")
    if data.get("version") not in {"a9.4-validated-v1", "a10-production-v1", "pa3-v2-production-v1"}:
        raise ValueError("stale runtime profile version")
    for section in ("training", "precision", "dataset", "collector", "evaluation"):
        if not isinstance(data.get(section), dict): raise ValueError(f"profile section missing: {section}")
    return data

def resolve_profile(path: str) -> dict:
    data = load_profile(path)
    t, p, c, d = data["training"], data["precision"], data["collector"], data["dataset"]
    required = {"ppo_epochs": 2, "lr_scale": 0.5, "global_minibatch": 4096, "microbatch": 1024,
                "envs": 1024, "workers": 16, "historical_probability": 0.0}
    if any(t.get(k) != v for k, v in required.items()): raise ValueError("validated training profile invariant failed")
    if p.get("mode") != "D" or p.get("stability") != "fp16-reduction-off" or not p.get("recompute_gate"): raise ValueError("precision profile invariant failed")
    if c.get("batch_cache") != "cuda" or c.get("compact_candidates") is not True: raise ValueError("collector profile invariant failed")
    if not os.path.isfile(d["manifest"]): raise ValueError("dataset manifest missing")
    digest = hashlib.sha256(open(d["manifest"], "rb").read()).hexdigest()
    if digest != d.get("manifest_sha256"): raise ValueError("dataset manifest hash mismatch")
    canonical = json.loads(json.dumps(data, sort_keys=True, separators=(",", ":")))
    artifact_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    semantic = {key: canonical[key] for key in ("version", "training", "precision", "dataset", "collector", "evaluation")}
    semantic_bytes = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    semantic_hash = hashlib.sha256(semantic_bytes).hexdigest()
    return {"profile": canonical, "profile_path": os.path.abspath(path),
            "config_hash": semantic_hash, "semantic_hash": semantic_hash,
            "artifact_hash": artifact_hash}

def launcher_values(resolved: dict) -> dict:
    t, p, c, d = (resolved["profile"]["training"], resolved["profile"]["precision"],
                   resolved["profile"]["collector"], resolved["profile"]["dataset"])
    values = {"epochs": t["ppo_epochs"], "experimental_lr_scale": t["lr_scale"], "minibatch": t["global_minibatch"],
            "microbatch": t["microbatch"], "envs": t["envs"], "workers": t["workers"],
            "historical_probability": t["historical_probability"], "precision_mode": p["mode"],
            "precision_stability": p["stability"], "recompute_gate": p["recompute_gate"],
            "batch_cache": c["batch_cache"], "compact_candidates": c["compact_candidates"]}
    if "teams" in d:
        values["teams"] = d["teams"]
    if "data" in d:
        values["data"] = d["data"]
    if "manifest" in d:
        values["dataset_manifest"] = d["manifest"]
    if "checkpoint_interval_matches" in t:
        values["checkpoint_interval_matches"] = t["checkpoint_interval_matches"]
    return values
