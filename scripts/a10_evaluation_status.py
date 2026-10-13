#!/usr/bin/env python3
"""Read-only A10 evaluation league status."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True); p.add_argument('--json',action='store_true'); a=p.parse_args(); run=Path(a.run_root); ev=run/'evaluation'
 def load(name,default):
  q=ev/name
  try:return json.loads(q.read_text())
  except Exception:return default
 reg=load('a10_registry.json',{'policies':{}}); sched=load('schedule.json',{'jobs':{}}); state=load('sidecar_state.json',{})
 latest='';
 try: latest=(run/'rolling/LATEST').read_text().strip()
 except OSError: pass
 out={'latest_generation':latest,'registered_policies':sorted(reg.get('policies',{})),'pending_jobs':[j for j in sched.get('jobs',{}).values() if j.get('status')=='pending'],'failed_jobs':[j for j in sched.get('jobs',{}).values() if j.get('status')=='failed'],'completed_jobs':[j for j in sched.get('jobs',{}).values() if j.get('status')=='completed'],'sidecar':state,'smoke_result':load('cpu_smoke.json',None),'best_policy':load('best_policy.json',None)}
 if a.json: print(json.dumps(out,indent=2,sort_keys=True)); return
 print(f"latest={latest} policies={len(out['registered_policies'])} pending={len(out['pending_jobs'])} failed={len(out['failed_jobs'])} completed={len(out['completed_jobs'])}")
 print(json.dumps(out.get('sidecar',{}),sort_keys=True))
if __name__=='__main__': main()
