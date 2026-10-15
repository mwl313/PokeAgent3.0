"""CPU-only V2 league. Compare each new snapshot directly with its incumbent.

Owns only <run>/evaluation/v2. Keeps the first, best and three most recent
model snapshots; immutable outcome receipts and retired identities remain.
Never loads optimizer state into a learner or writes a PPO match counter.
"""
from __future__ import annotations
import argparse,fcntl,json,os,time,shutil,hashlib
from dataclasses import asdict
from pathlib import Path
import torch
from agent.analysis.team_stats import atomic_json
from agent.evaluation.policy_archive import register_checkpoint,verify_archive
from agent.evaluation.league import run_pair,environment_identity
from agent.evaluation.fixed_policy import build_fixture_manifest
from agent.ppo.storage_manager import CheckpointGenerationStore
from agent.contracts import require_checkpoint_identity


def retain(own,registry,best):
    entries=sorted(registry['policies'].values(),key=lambda e:e['committed_matches'])
    live=[e for e in entries if not e.get('retired')]
    keep={best,entries[0]['id'],*[e['id'] for e in live[-3:]]}
    keep_paths={Path(e['path']).resolve() for e in entries if e['id'] in keep}
    root=(own/'policies').resolve()
    for entry in live:
        if entry['id'] in keep:continue
        path=Path(entry['path']).resolve()
        if root not in path.parents:raise ValueError('refuse to retire a foreign policy artifact')
        entry.update(retired=True,retirement_reason='V2 bounded retention: first/best/recent three',retired_at=time.time())
    atomic_json(own/'registry.json',registry)
    for entry in entries:
        if entry.get('retired'):
            path=Path(entry['path']).resolve()
            if root not in path.parents:raise ValueError('refuse to delete a foreign policy artifact')
            if path.exists() and path not in keep_paths:path.unlink()


def once(run,*,pairs=256,batch_size=32,workers=2,reserve_gib=20):
    if min(pairs,batch_size,workers)<1:raise ValueError('evaluation sizes must be positive')
    if reserve_gib<0:raise ValueError('evaluation reserve must be nonnegative')
    run=Path(run).resolve();own=run/'evaluation/v2';own.mkdir(parents=True,exist_ok=True)
    with (own/'worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'already_running'}
        if shutil.disk_usage(own).free < reserve_gib*2**30:return {'status':'resource_deferred','reason':f'{reserve_gib:g}GiB evaluation reserve'}
        store=CheckpointGenerationStore(run/'rolling')
        registry_path=own/'registry.json'
        old=json.loads(registry_path.read_text()) if registry_path.exists() else None
        seen={e.get('checkpoint_generation') for e in (old or {}).get('policies',{}).values()}
        for generation in reversed(store.generations_ready()):
            if generation.name in seen:continue
            manifest=json.loads((generation/'manifest.json').read_text())
            if manifest.get('dataset_manifest_hash')!='1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb':
                raise ValueError('V2 league dataset mismatch')
            payload=torch.load(generation/'rank0.pt',map_location='cpu',weights_only=False)
            peer=torch.load(generation/'rank1.pt',map_location='cpu',weights_only=False)
            for rank,value in enumerate((payload,peer)):
                require_checkpoint_identity(value['learner'])
                if value['accounting']['committed_matches']!=manifest['committed_matches'] or value['accounting']['iteration']!=manifest['iteration']:
                    raise ValueError('V2 league checkpoint clock mismatch')
                if value['source_sha']!=manifest['source_sha'] or value['ddp']['rank']!=rank or value['ddp']['world_size']!=2:
                    raise ValueError('V2 league checkpoint source/rank mismatch')
                if value['rank_digests']!=payload['rank_digests'] or value['opponent_manifest']!=payload['opponent_manifest']:
                    raise ValueError('V2 league peer provenance mismatch')
            del peer
            snapshot={key:payload[key] for key in ['source_sha','model_config_hash','action_schema_hash',
                'experiment','accounting','ddp','rank_digests']}
            snapshot.update(kind='v2_frozen_evaluation_snapshot',learner={key:payload['learner'][key] for key in ['model','config','contracts']},
                source_generation_manifest={k:v for k,v in manifest.items() if k!='team_stats_receipts'},
                source_generation_manifest_sha256=hashlib.sha256((generation/'manifest.json').read_bytes()).hexdigest())
            policy_id=f"v2-{manifest['committed_matches']:012d}-{generation.name.split('-')[-1]}"
            path=own/'policies'/f'{policy_id}.pt';path.parent.mkdir(exist_ok=True)
            temporary=path.with_suffix('.tmp');torch.save(snapshot,temporary)
            with temporary.open('rb') as f:os.fsync(f.fileno())
            os.replace(temporary,path);path.chmod(0o444)
            register_checkpoint(registry_path,policy_id=policy_id,label=f"V2 {manifest['committed_matches']}",checkpoint_path=path)
            registry=json.loads(registry_path.read_text());registry['policies'][policy_id]['checkpoint_generation']=generation.name
            atomic_json(registry_path,registry)
        if not registry_path.exists():return {'status':'waiting_for_first_checkpoint'}
        registry=verify_archive(registry_path)
        policies=sorted((e for e in registry['policies'].values() if not e.get('retired')),key=lambda e:e['committed_matches'])
        state_path=own/'state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {'best':policies[0]['id'],'tested':[],'status':'provisional_initial_policy'}
        candidates=[e for e in policies if e['id']!=state['best'] and e['id'] not in state['tested']]
        if candidates:
            import pa3_engine
            engine=pa3_engine.NativeEngine('engine/data','engine/data/a10-training-teams.json',workers=workers)
            fixtures=build_fixture_manifest(team_count=engine.team_count(),pairs=pairs,seed=20261014)
            atomic_json(own/'fixtures.json',[asdict(f) for f in fixtures])
            candidate=candidates[-1]
            result,cached=run_pair(str(registry_path),str(own/'receipts.json'),engine,fixtures,
                candidate['id'],state['best'],device='cpu',seed=20261014,backend='batched',batch_size=batch_size,
                environment=environment_identity(engine_path=pa3_engine._native.__file__,data_path='engine/data',teams_path='engine/data/a10-training-teams.json'))
            state['tested'].append(candidate['id'])
            # The existing paired cluster-bootstrap interval is descriptive,
            # single-panel evidence, not an anytime-valid global guarantee.
            interval=result.get('paired_score_ci95',result.get('score_ci95',[0,1]))
            if result.get('operational_errors',0) or result.get('illegal_actions',0):raise ValueError('V2 evaluation operational failure')
            state['last_result']=result;state['last_candidate']=candidate['id'];state['updated_at']=time.time()
            if interval[0]>.5:state['best']=candidate['id'];state['status']='provisional_direct_panel_leader'
        atomic_json(state_path,state);retain(own,registry,state['best'])
        return {'status':state['status'],'best':state['best'],'tested':len(state['tested'])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True)
    parser.add_argument('--pairs',type=int,default=256);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--reserve-gib',type=float,default=20)
    args=parser.parse_args();torch.set_num_threads(2)
    print(json.dumps(once(args.run_root,pairs=args.pairs,batch_size=args.batch_size,reserve_gib=args.reserve_gib),indent=2))


if __name__=='__main__':main()
