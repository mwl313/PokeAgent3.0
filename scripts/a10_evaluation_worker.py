#!/usr/bin/env python3
"""One persistent CPU battle worker; failures never propagate to PPO."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,fcntl,json,math,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'engine/python')]
from agent.evaluation.a10_league import atomic,read,promotion
from a10_evaluation_runner import run_job

STAGES=[256,1024,2048]

def reconcile(own):
    reg=read(own/'registry.json',{'policies':{}})['policies']
    q=read(own/'jobs.json',{'schema':'a10-league-jobs-v2','jobs':{},'next_ordinal':1,'ordinals':{}})
    entries=sorted([e for e in reg.values() if not e.get('retired')],key=lambda x:(x['committed_matches'],x['id']))
    for e in entries:
        if e['id'] not in q['ordinals']:q['ordinals'][e['id']]=q['next_ordinal'];q['next_ordinal']+=1
    best=read(own/'best_policy.json')
    if not best and entries:
        first=entries[0];best={'schema':'a10-best-policy-v2','policy_id':first['id'],'path':first['path'],'file_sha256':first['file_sha256'],'model_digest':first['model_digest'],'committed_matches':first['committed_matches'],'status':'provisional_seed','reason':'earliest available verified policy; not strength evidence','promotions':[]}
        atomic(own/'best_policy.json',best)
    if 'initial_ids' not in q:q['initial_ids']=[e['id'] for e in entries]
    if best:
        for e in entries:
            if e['id']==best['policy_id']:
                if e['id'] in q['jobs']:q['jobs'][e['id']]['status']='completed'
                continue
            if e['id'] in q['jobs']:continue
            q['jobs'][e['id']]={'candidate':e['id'],'actual_matches':e['committed_matches'],'status':'pending','stage':0,'retries':0,'initial_history':e['id'] in q.get('initial_ids',[])}
        # Bound the active backlog. Preserve initial historical comparisons and all receipts.
        future=[j for j in q['jobs'].values() if j['status']=='pending' and not j['initial_history']]
        for j in sorted(future,key=lambda j:j['actual_matches'],reverse=True)[4:]:j.update(status='superseded',reason='newer candidates prioritized')
    atomic(own/'jobs.json',q);return reg,q,best

def work_once(run_root,*,batch_size=64,threads=2,workers=2):
    own=Path(run_root)/'evaluation/league_v2';reg,q,best=reconcile(own)
    candidates=[j for j in q['jobs'].values() if j['status'] in ('pending','running','retry') and j.get('retries',0)<3]
    if not candidates:return {'status':'idle'}
    # Prefer the newest relevant checkpoint while retaining initial history jobs.
    j=sorted(candidates,key=lambda j:-j['actual_matches'])[0]
    if j.get('opponent')!=best['policy_id']:
        j.update(opponent=best['policy_id'],stage=0,receipts=[],outcomes=[],retries=0)
    stage=j['stage'];previous=STAGES[stage-1] if stage else 0;games=STAGES[stage]-previous
    j.update(status='running',started_at=time.time());atomic(own/'jobs.json',q)
    try:
        # The fixed main panel is shared; alpha is allocated by outcome-independent
        # archive ordinals to every possible candidate/incumbent pair and stage.
        r=run_job(run_root,j['candidate'],j['opponent'],pairs=games//2,fixture_start=previous//2,seed=20261014,batch_size=batch_size,threads=threads,workers=workers)
        j['receipts'].append(r['identity']);j['outcomes']+=r['outcomes']
        decision=promotion(j['outcomes'],q['ordinals'][j['candidate']],stage,opponent_index=q['ordinals'][j['opponent']])
        j.update(decision=decision,finished_at=time.time(),evaluated_games=len(j['outcomes']))
        if decision['decision']=='escalate':j.update(status='pending',stage=stage+1)
        else:
            j['status']='completed'
            if decision['decision']=='promote':
                entry=reg[j['candidate']];history=best.get('promotions',[])+[{'previous_policy':best['policy_id'],'new_policy':entry['id'],'receipts':j['receipts'],'decision':decision,'at':time.time()}]
                new={'schema':'a10-best-policy-v2','policy_id':entry['id'],'path':entry['path'],'file_sha256':entry['file_sha256'],'model_digest':entry['model_digest'],'committed_matches':entry['committed_matches'],'status':'validated_direct_superiority','scope':'tested A10 league only, not ladder Elo','promotions':history}
                atomic(own/'best_history'/f'{time.time_ns()}.json',best);atomic(own/'best_policy.json',new)
            atomic(own/'decisions'/f'{j["candidate"]}.json',j)
        atomic(own/'jobs.json',q)
        atomic(own/'worker_state.json',{'status':'working','last_job':j['candidate'],'last_decision':decision,'last_receipt':r['identity'],'updated_at':time.time(),'pid':os.getpid()})
        return {'status':j['status'],'candidate':j['candidate'],'decision':decision,'games':r['matches'],'games_per_second':r['games_per_second']}
    except Exception as e:
        j.update(status='retry' if j['retries']<2 else 'failed',retries=j['retries']+1,error=str(e)[-2000:]);atomic(own/'jobs.json',q)
        atomic(own/'worker_state.json',{'status':'error','error':str(e)[-2000:],'updated_at':time.time()});return {'status':'error','error':str(e)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--threads',type=int,default=2);p.add_argument('--workers',type=int,default=2);p.add_argument('--once',action='store_true');a=p.parse_args();own=Path(a.run_root)/'evaluation/league_v2';own.mkdir(parents=True,exist_ok=True)
    with (own/'worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('another evaluation worker owns the lock')
        while True:
            if __import__('shutil').disk_usage(own).free < 20*2**30:
                result={'status':'resource_deferred','reason':'20GiB evaluator headroom'}
            else:
                import re
                log=Path(a.run_root)/'report_resume_20261013.json.rank0.log'
                times=[float(x) for x in re.findall(r'iteration \d+ complete in ([0-9.]+)s',log.read_text())][-5:] if log.exists() else []
                baseline=read(own/'benchmark.json',{}).get('baseline_gps')
                recent=4096*len(times)/sum(times) if times else None
                pressure=bool(recent and baseline and recent<baseline*.95)
                severe=bool(recent and baseline and recent<baseline*.80)
                atomic(own/'resource_state.json',{'baseline_gps':baseline,'recent_training_gps':recent,'throttled':pressure,'deferred':severe,'updated_at':time.time(),'caveat':'rolling observation includes ordinary training variability'})
                if severe:
                    result={'status':'resource_deferred','reason':'PPO rolling throughput below 80% of pre-evaluation baseline'}
                else:
                    result=work_once(a.run_root,batch_size=min(a.batch_size,16) if pressure else a.batch_size,threads=1 if pressure else a.threads,workers=1 if pressure else a.workers)
            print(json.dumps(result),flush=True)
            if a.once:break
            time.sleep(30 if result['status'] in ('idle','error','resource_deferred') else 1)
if __name__=='__main__':main()
