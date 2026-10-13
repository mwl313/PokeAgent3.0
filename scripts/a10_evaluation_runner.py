#!/usr/bin/env python3
"""Run one explicit, CPU-only A10 checkpoint-versus-checkpoint league job."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def run_job(run_root, candidate, opponent, *, pairs=4, seed=20261013, workers=1, code_root=None):
    run=Path(run_root); ev=run/'evaluation'; reg=json.loads((ev/'a10_registry.json').read_text()); policies=reg.get('policies',{})
    if candidate not in policies or opponent not in policies: raise ValueError('unknown A10 policy identity')
    if candidate==opponent: raise ValueError('candidate and opponent must differ')
    c=policies[candidate]; o=policies[opponent]; output=ev/f'results/{candidate}_vs_{opponent}_{pairs*2}.json'; output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): return {'status':'cached','output':str(output)}
    root=Path(code_root or Path(__file__).resolve().parents[1]); script=root/'scripts/evaluate_fixed_policy.py'; data=Path('/home/minidc/Desktop/PokeAgent3.0-A/engine/data'); teams=data/'a10-training-teams.json'; manifest=ev/f'fixtures_{seed}_{pairs}.json'
    cmd=[sys.executable,str(script),'--checkpoint',c['path'],'--policy','checkpoint','--opponent','checkpoint','--opponent-checkpoint',o['path'],'--pairs',str(pairs),'--seed',str(seed),'--output',str(output),'--manifest',str(manifest),'--device','cpu','--workers',str(workers),'--data',str(data),'--teams',str(teams)]
    env=dict(os.environ); env['CUDA_VISIBLE_DEVICES']=''; env['OMP_NUM_THREADS']=str(max(1,workers)); env['MKL_NUM_THREADS']=str(max(1,workers))
    started=time.time(); cp=subprocess.run(['nice','-n','10',*cmd],cwd=root,env=env,text=True,capture_output=True,timeout=900)
    if cp.returncode: raise RuntimeError(cp.stderr[-2000:] or cp.stdout[-2000:])
    report=json.loads(output.read_text()); receipt={'status':'success','candidate_id':candidate,'opponent_id':opponent,'candidate_clock':c['committed_matches'],'opponent_clock':o['committed_matches'],'result_path':str(output),'score_mean':report.get('score_mean'),'score_ci95':report.get('score_ci95'),'matches':report.get('matches'),'runtime_seconds':report.get('runtime_seconds'),'wall_seconds':time.time()-started,'fixture_manifest_hash':report.get('fixture_manifest_hash')}
    (ev/'results'/f'{candidate}_vs_{opponent}_{pairs*2}.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    best=ev/'best_policy.json'
    if receipt['score_mean'] is not None:
        old=json.loads(best.read_text()) if best.exists() else None
        if old is None or receipt['score_mean'] > old.get('score_mean',-1): best.write_text(json.dumps({'scope':f'vs:{opponent}:seed:{seed}:fixtures:{receipt["fixture_manifest_hash"]}','policy_id':candidate,'score_mean':receipt['score_mean'],'score_ci95':receipt['score_ci95'],'receipt':str(ev/'results'/f'{candidate}_vs_{opponent}_{pairs*2}.receipt.json')},indent=2,sort_keys=True)+'\n')
    return receipt

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True); p.add_argument('--candidate',required=True); p.add_argument('--opponent',required=True); p.add_argument('--pairs',type=int,default=4); p.add_argument('--seed',type=int,default=20261013); p.add_argument('--code-root'); a=p.parse_args(); print(json.dumps(run_job(a.run_root,a.candidate,a.opponent,pairs=a.pairs,seed=a.seed,code_root=a.code_root),indent=2))
if __name__=='__main__': main()
