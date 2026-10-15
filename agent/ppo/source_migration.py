"""Exact, local A10 recovery migration; never a global source-SHA bypass."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = "0b5d930743298720e073f65264464f0d953ca7d3"
ALLOWED = {
    "scripts/run_ddp_ppo.py", "scripts/bench_pa3_end_to_end.py",
    "agent/train/native_collector.py", "agent/ppo/recompute_safety.py",
    "agent/ppo/source_migration.py", "agent/ppo/telemetry.py",
    "scripts/a10_recompute_recovery_probe.py",
    "agent/evaluation/a10_league.py",
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_recovery_migration(path, root, source, target):
    receipt = json.loads(Path(path).read_text())
    if receipt.get('schema')=='pa3-v2-reviewed-release-migration-v1':
        from agent.ppo.v2_release_migration import validate
        return validate(receipt,root,source,target)
    if receipt.get('schema')=='pa3-v2-data-path-migration-v1':
        from agent.ppo.v2_source_migration import validate_v2_migration
        return validate_v2_migration(receipt,root,source,target)
    if receipt.get("schema") != "a10-exact-recovery-migration-v1" or source != BASE:
        raise ValueError("unsupported A10 recovery source")
    if (receipt.get("source_sha"), receipt.get("target_sha")) != (source, target):
        raise ValueError("migration receipt source/target mismatch")
    if git(root, "rev-parse", "HEAD") != target or git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("migration requires exact clean committed source")
    changed = git(root, "diff", "--name-only", BASE, target, "--", "agent", "engine", "configs", "data", "scripts").splitlines()
    # Existing evaluation/monitoring code in d10ac69 was deployed separately;
    # require its exact committed content, and forbid any new engine, model,
    # learner, optimizer, DDP, schema, buffer, dataset or profile changes.
    deployed = "d10ac692009566686fd32285c4456309a5f383a7"
    for name in changed:
        if name in ALLOWED:
            continue
        if git(root, "diff", deployed, target, "--", name):
            raise ValueError(f"migration changed protected source: {name}")
        if not (name.startswith("agent/evaluation/") or name.startswith("scripts/a10_") or name == "scripts/__init__.py"):
            raise ValueError(f"migration changed protected source: {name}")
    if sorted(changed) != receipt.get("changed_runtime_files"):
        raise ValueError("migration runtime file list mismatch")
    return {"source_sha": source, "target_sha": target,
            "receipt_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "scope": "telemetry-and-recompute-verification-only"}


def prepare_recovery_receipt(root, checkpoint, output):
    """Verify actual rank payloads before recording an exact source migration."""
    import torch
    from agent.evaluation.a10_league import atomic, training_digests, validate_payload, DATASET, SOURCE
    from agent.ppo.experimental_checkpoint import hash_file
    from agent.ppo.storage_manager import CheckpointGenerationStore
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if checkpoint not in CheckpointGenerationStore(checkpoint.parent.parent).generations_ready():
        raise ValueError("checkpoint is not a verified complete generation")
    states = []
    for rank in (0, 1):
        path = checkpoint / f"rank{rank}.pt"
        if path.stat().st_size != manifest['ranks'][str(rank)]['bytes']:
            raise ValueError("checkpoint byte count mismatch")
        p = torch.load(path, map_location='cpu', weights_only=False)
        validate_payload(p, manifest, rank)
        if training_digests(p) != p['rank_digests'][str(rank)]:
            raise ValueError("actual model/optimizer digest mismatch")
        if p['accounting']['committed_matches'] != p['learner']['scheduler']['matches']:
            raise ValueError("scheduler clock mismatch")
        if p['collector_restart']['iteration'] != manifest['iteration']:
            raise ValueError("collector restart clock mismatch")
        if p['collector_restart']['seed'] != 20261009:
            raise ValueError("production collector seed mismatch")
        if not p['rng'].get('torch_cuda') or len(p['rng'].get('generators', [])) != 1:
            raise ValueError("CUDA/shuffle RNG state missing")
        if not p['learner']['scaler'].get('scale'):
            raise ValueError("GradScaler state missing")
        states.append(p)
    for field in ('rank_digests', 'source_sha', 'dataset_manifest_hash', 'opponent_manifest',
                  'model_config_hash', 'ppo_config_hash'):
        if states[0][field] != states[1][field]:
            raise ValueError(f"checkpoint peer {field} mismatch")
    for field in ('scheduler', 'scaler', 'optimizer_steps', 'config'):
        if states[0]['learner'][field] != states[1]['learner'][field]:
            raise ValueError(f"checkpoint peer learner {field} mismatch")
    if states[0]['opponent_manifest'] != {'current': 'current', 'historical': [], 'historical_probability': 0.0}:
        raise ValueError("production opponent pool mismatch")
    dataset = Path(manifest['dataset_manifest'])
    if hash_file(dataset) != DATASET:
        raise ValueError("frozen dataset manifest mismatch")
    teams = Path('/home/minidc/Desktop/PokeAgent3.0-A/engine/data/a10-training-teams.json')
    team_data = json.loads(teams.read_text())
    team_count = len(team_data['teams']) if isinstance(team_data, dict) else len(team_data)
    if team_count != 1220:
        raise ValueError("frozen training pool size mismatch")
    target = git(root, 'rev-parse', 'HEAD')
    changed = git(root, 'diff', '--name-only', BASE, target, '--', 'agent', 'engine', 'configs', 'data', 'scripts').splitlines()
    receipt = {'schema': 'a10-exact-recovery-migration-v1', 'source_sha': SOURCE,
               'target_sha': target, 'changed_runtime_files': sorted(changed),
               'checkpoint': str(checkpoint), 'checkpoint_manifest': manifest,
               'rank_artifact_hashes': {str(rank): hash_file(checkpoint/f'rank{rank}.pt') for rank in (0, 1)},
               'verified_rank_digests': states[0]['rank_digests'],
               'scheduler': states[0]['learner']['scheduler'], 'scaler': states[0]['learner']['scaler'],
               'committed_matches': manifest['committed_matches'], 'absolute_iteration': manifest['iteration'],
               'remaining_iterations': 24415-manifest['iteration'], 'team_count': team_count,
               'opponent_manifest': states[0]['opponent_manifest'],
               'rng_states_present': True, 'optimizer_steps': states[0]['learner']['optimizer_steps']}
    atomic(output, receipt)
    validate_recovery_migration(output, root, SOURCE, target)
    return receipt
