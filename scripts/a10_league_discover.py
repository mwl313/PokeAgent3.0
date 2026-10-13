#!/usr/bin/env python3
"""Discovery and ingestion of complete A10 artifacts; no model inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,argparse,fcntl,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'engine/python')]
from agent.evaluation.a10_league import atomic,read,capture_generation,import_legacy,retain_archive

def discover(run_root):
    import shutil
    run=Path(run_root); own=run/'evaluation/league_v2';own.mkdir(parents=True,exist_ok=True)
    with (own/'discovery.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if shutil.disk_usage(own).free < 20*2**30:raise ValueError('evaluation archive resource deferred: less than 20GiB headroom')
        registry=read(own/'registry.json',{'schema':'a10-league-registry-v2','policies':{},'sources':{},'rejected':{}})
        rolling=run/'rolling'; paths=sorted((rolling/'generations').glob('*/manifest.json'))
        for mf in paths:
            name=mf.parent.name
            if name in registry['sources']:continue
            try:
                e=capture_generation(rolling,name,own/'policies');registry['policies'][e['id']]=e;registry['sources'][name]=e['id']
            except (OSError,ValueError,KeyError) as e:registry['rejected'][name]=str(e)
        for p in sorted((run/'evaluation/policies').glob('a10-*.json')):
            if str(p) in registry['sources']: continue
            try:
                e=import_legacy(p,own/'policies');registry['policies'][e['id']]=e;registry['sources'][str(p)]=e['id']
            except (OSError,ValueError,KeyError,RuntimeError) as e: registry['rejected'][str(p)]=str(e)
        atomic(own/'registry.json',registry)
        retain_archive(own)
        registry=read(own/'registry.json')
        # Milestones are bookkeeping only. Never bind a later checkpoint to an old boundary.
        latest=max([e['committed_matches'] for e in registry['policies'].values()],default=0)
        atomic(own/'milestones.json',{'anchor':0,'light_interval':50000,'full_interval':200000,'boundaries':[{'scheduled_matches':n,'status':'exact_snapshot' if any(x['committed_matches']==n for x in registry['policies'].values()) else 'no_exact_snapshot','actual_policy_clocks':[x['committed_matches'] for x in registry['policies'].values() if x['committed_matches']==n]} for n in range(50000,latest+1,50000)]})
        return registry
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args();r=discover(a.run_root); print({'registered':len(r['policies']),'rejected':r['rejected']})
