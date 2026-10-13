#!/usr/bin/env python3
"""A10-only, non-training evaluation adapter and durable scheduler."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile, time
from pathlib import Path
import torch

EXPECTED_DATASET='1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb'
EXPECTED_SOURCE='0b5d930'

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix='.a10-',dir=path.parent)
    with os.fdopen(fd,'w') as f: json.dump(value,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def inspect_generation(root, name=None):
    root=Path(root); latest=(root/'LATEST').read_text().strip() if name is None else name; p=root/'generations'/latest; mf=p/'manifest.json'
    if not mf.is_file(): raise ValueError('generation manifest missing')
    m=json.loads(mf.read_text())
    if m.get('complete') is not True or m.get('iteration_complete') is not True: raise ValueError('generation is incomplete')
    if m.get('world_size') != 2 or m.get('dataset_manifest_hash') != EXPECTED_DATASET: raise ValueError('A10 checkpoint contract mismatch')
    if not m.get('source_sha','').startswith(EXPECTED_SOURCE): raise ValueError('source SHA mismatch')
    if m.get('committed_matches') != 102400 and m.get('committed_matches',0) < 0: raise ValueError('invalid checkpoint clock')
    for rank in ('0','1'):
        row=m.get('ranks',{}).get(rank); f=p/(row or {}).get('path','')
        if not row or not f.is_file() or f.stat().st_size != row.get('bytes'): raise ValueError(f'rank{rank} missing or size mismatch')
    return {'generation':latest,'path':str(p),'manifest':m}

def archive_snapshot(root, archive, name=None):
    info=inspect_generation(root,name); src=Path(info['path'])/'rank0.pt'; out=Path(archive); out.mkdir(parents=True,exist_ok=True)
    # Read a complete immutable envelope and retain only model/inference metadata.
    payload=torch.load(src,map_location='cpu',weights_only=False)
    if not isinstance(payload,dict) or not isinstance(payload.get('learner',{}).get('model'),dict): raise ValueError('learner model missing')
    digest=sha(src); model_payload={'kind':'a10_inference_snapshot','source_sha':info['manifest'].get('source_sha'),'model_config_hash':info['manifest'].get('model_config_hash'), 'action_schema_hash':info['manifest'].get('action_schema_hash'),'experiment':payload.get('experiment',{}),'accounting':{'committed_matches':info['manifest']['committed_matches']},'learner':{'model':payload['learner']['model']},'checkpoint_generation':info['generation'],'source_checkpoint_sha256':digest}
    # Content-address the model-only archive by canonical source identity.
    temp=out/(f'.{digest}.tmp'); target=out/(f'{digest}.pt')
    if not target.exists(): torch.save(model_payload,temp); os.replace(temp,target); os.chmod(target,0o444)
    receipt={'schema':'a10-evaluation-snapshot-v1','policy_id':f'a10-{info["manifest"]["committed_matches"]}-{digest[:16]}','path':str(target),'source_generation':info['generation'],'committed_matches':info['manifest']['committed_matches'],'source_checkpoint_sha256':digest,'dataset_manifest_hash':info['manifest'].get('dataset_manifest_hash'),'source_sha':info['manifest'].get('source_sha'),'immutable':True,'optimizer_included':False,'created_at':time.time()}
    atomic_json(out/(receipt['policy_id']+'.json'),receipt); return receipt

def schedule(state, matches, checkpoint=None):
    state=Path(state)
    if state.exists(): s=json.loads(state.read_text())
    else: s={'schema':'a10-evaluation-schedule-v1','anchor':0,'light_interval':50000,'full_interval':200000,'jobs':{}}
    if (s.get('anchor'),s.get('light_interval'),s.get('full_interval')) != (0,50000,200000): raise ValueError('A10 schedule contract mismatch')
    created=[]; latest=checkpoint
    for boundary in range(50000,(int(matches)//50000)*50000+1,50000):
        kind='full' if boundary%200000==0 else 'light'; jid=f'{kind}:{boundary}'
        if jid in s['jobs']: continue
        job={'id':jid,'kind':kind,'scheduled_boundary':boundary,'status':'pending' if latest else 'deferred','missing_snapshot':latest is None,'checkpoint':latest,'reason':None if latest else 'no checkpoint at scheduled boundary'}
        s['jobs'][jid]=job; created.append(job)
    atomic_json(state,s); return created

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('snapshot'); q.add_argument('--rolling',required=True); q.add_argument('--archive',required=True); q.add_argument('--generation')
    q=sub.add_parser('schedule'); q.add_argument('--state',required=True); q.add_argument('--matches',type=int,required=True); q.add_argument('--checkpoint')
    q=sub.add_parser('verify'); q.add_argument('--rolling',required=True); q.add_argument('--generation')
    a=p.parse_args()
    if a.cmd=='snapshot': print(json.dumps(archive_snapshot(a.rolling,a.archive,a.generation),indent=2)); return
    if a.cmd=='verify': print(json.dumps(inspect_generation(a.rolling,a.generation),indent=2)); return
    print(json.dumps({'created':schedule(a.state,a.matches,a.checkpoint)},indent=2))
if __name__=='__main__': main()
