"""Optional structured PPO telemetry for a future validated trainer resume.
This module is intentionally not imported by the active production launcher."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path

def append_iteration(path, *, iteration, committed_matches, rank, metrics):
    record={'schema':'pa3-ppo-telemetry-v1','iteration':int(iteration),'committed_matches':int(committed_matches),'rank':int(rank),'metrics':dict(metrics)}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.telemetry-',dir=path.parent)
    with os.fdopen(fd,'w') as f: json.dump(record,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); return record

def read_iteration(path):
    with open(path) as f: row=json.load(f)
    if row.get('schema')!='pa3-ppo-telemetry-v1': raise ValueError('unsupported telemetry schema')
    for key in ('iteration','committed_matches','rank','metrics'):
        if key not in row: raise ValueError(f'missing telemetry field: {key}')
    return row
