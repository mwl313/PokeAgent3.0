#!/usr/bin/env python3
"""Read-only actual A10 league progress and canonical best policy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.evaluation.a10_league import read

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',default='/home/minidc/Desktop/PokeAgent3.0-A/runs/a10_production_20261013_v2');p.add_argument('--json',action='store_true');a=p.parse_args();own=Path(a.run_root)/'evaluation/league_v2';q=read(own/'jobs.json',{'jobs':{}});reg=read(own/'registry.json',{'policies':{}})
    result={'latest_discovered_clock':max([e['committed_matches'] for e in reg['policies'].values()],default=None),'policies':len(reg['policies']),'best':read(own/'best_policy.json'),'active':read(own/'active.json'),'worker':read(own/'worker_state.json'),'resources':read(own/'resource_state.json'),'counts':{k:sum(x['status']==k for x in q['jobs'].values()) for k in ('completed','pending','running','retry','failed','superseded')},'jobs':[{k:v for k,v in j.items() if k!='outcomes'} for j in q['jobs'].values()]}
    recent=sorted((own/'receipts').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:8]
    result['recent_results']=[{k:v for k,v in read(p).items() if k in ('candidate_id','opponent_id','candidate_clock','opponent_clock','matches','score_mean','score_ci95','games_per_second','status','fixture_manifest_hash','identity')} for p in recent]
    result['latest_evaluated_clock']=max([r.get('candidate_clock',0) for r in result['recent_results']],default=None)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
