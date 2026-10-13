#!/usr/bin/env python3
"""Identity-keyed, CPU-only native batched checkpoint evaluation."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'engine/python')]
from agent.evaluation.a10_league import atomic,read,verify_entry,digest

def run_job(run_root,candidate,opponent,*,pairs=128,seed=20261014,workers=2,code_root=None,batch_size=32,threads=2,fixture_start=0,registry_path=None,output_root=None):
    import torch
    torch.set_num_threads(threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    import pa3_engine
    from agent.evaluation.fixed_policy import load_frozen_model,build_fixture_manifest,fixture_manifest_hash,_fixture_json
    from agent.evaluation.batched_policy import BatchedPolicyEvaluator,BATCHED_EVALUATOR_VERSION,SAMPLING_SEMANTICS
    from agent.evaluation.league import environment_identity,evaluation_identity
    ev=Path(run_root)/'evaluation'; own=ev/'league_v2'; reg=read(registry_path or own/'registry.json',{'policies':{}})['policies']
    if candidate not in reg or opponent not in reg or candidate==opponent: raise ValueError('two distinct registered policies required')
    c,o=reg[candidate],reg[opponent]; verify_entry(c); verify_entry(o)
    data=Path('/home/minidc/Desktop/PokeAgent3.0-A/engine/data'); teams=data/'a10-training-teams.json'
    engine=pa3_engine.NativeEngine(str(data),str(teams),workers=workers)
    if engine.team_count()!=1220: raise ValueError('expected exact 1220-team pool')
    fixtures=build_fixture_manifest(pairs=1024,team_count=1220,seed=seed)[2*fixture_start:2*(fixture_start+pairs)]
    if len(fixtures)!=2*pairs: raise ValueError('fixture range exceeds frozen panel')
    env=environment_identity(engine_path=str(ROOT/'engine/python/pa3_engine/pa3_engine.so'),data_path=str(data),teams_path=str(teams))
    env.update(backend='batched',batch_size=batch_size,device='cpu',threads=threads,interop_threads=1,engine_workers=workers,battle_format='native-A10-frozen-team-pool',evaluator=BATCHED_EVALUATOR_VERSION,sampling=SAMPLING_SEMANTICS,code_sha=digest({str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [ROOT/'agent/evaluation/batched_policy.py',ROOT/'agent/evaluation/fixed_policy.py',ROOT/'scripts/a10_evaluation_runner.py']}))
    key,identity=evaluation_identity(c,o,fixtures,seed=seed,max_rounds=2000,environment=env)
    dest=Path(output_root or own/'receipts')/f'{key}.json'
    if dest.exists():
        receipt=read(dest)
        if receipt.get('status')=='success' and receipt.get('identity')==key: return dict(receipt,cached=True)
    atomic(own/'active.json',{'status':'running','candidate':candidate,'opponent':opponent,'games':len(fixtures),'completed_games':0,'started_at':time.time(),'identity':key,'pid':os.getpid()})
    panel=own/'fixtures'/f'{fixture_manifest_hash(fixtures)}.json'
    if not panel.exists(): atomic(panel,{'fixtures':_fixture_json(fixtures),'fixture_manifest_hash':fixture_manifest_hash(fixtures),'team_pool':1220,'environment':env})
    model=load_frozen_model(c['path'],device='cpu'); other=load_frozen_model(o['path'],device='cpu')
    def progress(n,total):
        atomic(own/'active.json',{'status':'running','candidate':candidate,'opponent':opponent,'games':total,'completed_games':n,'identity':key,'pid':os.getpid(),'updated_at':time.time()})
    started=time.time()
    result=BatchedPolicyEvaluator(engine,device='cpu',seed=seed,batch_size=batch_size).run(fixtures,model=model,opponent_model=other,progress_callback=progress)
    status='success' if result['illegal_actions']==0 and result['operational_errors']==0 and sum(result[x] for x in ('wins','losses','draws'))==len(fixtures) else 'invalid'
    receipt={'status':status,'identity':key,'identity_body':identity,'candidate_id':candidate,'opponent_id':opponent,'candidate_clock':c['committed_matches'],'opponent_clock':o['committed_matches'],'candidate_model_digest':c['model_digest'],'opponent_model_digest':o['model_digest'],'evaluation_at':time.time(),'games_per_second':len(fixtures)/result['runtime_seconds'],'cpu_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_affinity':sorted(os.sched_getaffinity(0)),**result}
    atomic(dest,receipt); atomic(own/'active.json',{'status':status,'identity':key,'completed_games':len(fixtures),'games':len(fixtures),'receipt':str(dest),'updated_at':time.time()})
    if status!='success': raise RuntimeError(f'evaluation invalid: {dest}')
    return dict(receipt,cached=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--candidate',required=True);p.add_argument('--opponent',required=True);p.add_argument('--pairs',type=int,default=128);p.add_argument('--seed',type=int,default=20261014);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--threads',type=int,default=2);p.add_argument('--workers',type=int,default=2);p.add_argument('--fixture-start',type=int,default=0);a=p.parse_args()
    r=run_job(a.run_root,a.candidate,a.opponent,pairs=a.pairs,seed=a.seed,batch_size=a.batch_size,threads=a.threads,workers=a.workers,fixture_start=a.fixture_start)
    print(json.dumps({k:r[k] for k in ['status','identity','candidate_clock','opponent_clock','matches','wins','losses','draws','score_mean','score_ci95','runtime_seconds','games_per_second','cpu_max_rss_kib','cached']},indent=2))
if __name__=='__main__':main()
