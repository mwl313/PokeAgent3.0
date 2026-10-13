"""Read-only legacy provenance capture; writes only the requested evidence file."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    desktop = pathlib.Path('/home/minidc/Desktop')
    legacy = desktop / 'PokeAgent3.0-A/runs/a10_production_20261013_v2'
    archive = desktop / 'pa3-legacy-evidence-20261013/a10_production_20261013_v2'
    generation = (legacy / 'rolling/LATEST').read_text().strip()
    # The independent league remains live. Verify immutable checkpoint/policy
    # files; its mutable queue snapshot is explicitly only point-in-time data.
    files = list((legacy / 'rolling/generations').glob('*/rank*.pt'))
    files += list((legacy / 'rolling/generations').glob('*/manifest.json'))
    files += list((legacy / 'evaluation/league_v2/policies').glob('*.pt'))
    preserved = []
    for source in files:
        copy = archive / source.relative_to(legacy)
        a, b = sha(source), sha(copy)
        if a != b:
            raise RuntimeError(f'archive mismatch: {source}')
        preserved.append({'path': str(source), 'archive': str(copy), 'sha256': a})
    teams = desktop / 'PokeAgent3.0-A/engine/data/a10-training-teams.json'
    manifest = desktop / 'PokeAgent3.0-A/data/teams/a10-p0-proposed-20261013/manifest.json'
    import sys
    sys.path.insert(0, str(desktop / 'PokeAgent3.0-A-recovery/engine/python'))
    import pa3_engine
    engines = [pa3_engine.NativeEngine(str(teams.parent), str(teams), workers=1) for _ in range(2)]
    counts = [e.team_count() for e in engines]
    if counts != [1220, 1220]:
        raise RuntimeError(f'native loader count: {counts}')
    worktrees = {}
    for name in ['PokeAgent3.0', 'PokeAgent3.0-A', 'PokeAgent3.0-A-telemetry', 'PokeAgent3.0-A-recovery', 'PokeAgent3.0-A-evaluation-league', 'PokeAgent3.0-V2']:
        root = desktop / name
        worktrees[name] = {key: command('git', '-C', str(root), *argv) for key, argv in {
            'head': ['rev-parse', 'HEAD'], 'status': ['status', '--porcelain=v1'],
            'remote': ['remote', '-v'], 'audit_diff': ['diff', '--stat', 'd10ac69', 'HEAD'],
        }.items()}
    payload = {
        'legacy_policy_classification': 'V1 historical only; prohibited as V2 initialization',
        'latest_generation': generation, 'archive_verified_files': preserved,
        'archive_scope': 'same-disk independent copy; mutable league metadata is a snapshot',
        'team_file': str(teams), 'team_file_sha256': sha(teams),
        'dataset_manifest': str(manifest), 'dataset_manifest_sha256': sha(manifest),
        'native_loader_counts': counts, 'native_loader_order': 'JSON array index, identical file bytes for both loader instances; not DDP proof',
        'engine_binary_sha256': sha(desktop / 'PokeAgent3.0-A-recovery/engine/python/pa3_engine/pa3_engine.so'),
        'worktrees': worktrees,
        'services': command('systemctl', '--user', 'show', 'a10-ppo-recovery-20261013.service', 'a10-evaluation-worker.service', '-p', 'Id', '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ExecStart'),
        'gpu': command('nvidia-smi'),
        'python': sys.version,
    }
    pathlib.Path(args.output).write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: payload[k] for k in ['latest_generation', 'team_file_sha256', 'dataset_manifest_sha256', 'native_loader_counts', 'engine_binary_sha256']}))


if __name__ == '__main__':
    main()
