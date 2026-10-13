#!/usr/bin/env python3
"""Singleton, exception-only A10 watchdog. Never imports torch or restarts training."""
from __future__ import annotations
import argparse, fcntl, json, os, sys, time
from pathlib import Path
try:
    from .a10_status import DEFAULT_ROOT, snapshot
except ImportError:
    from a10_status import DEFAULT_ROOT, snapshot

RESERVE=15*2**30; WARN=25*2**30

def _load(p, default):
    try: return json.loads(p.read_text())
    except Exception: return default

def _write(p,obj): p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); tmp.replace(p)

def evaluate(s, state):
    now=time.time(); issues=[]
    if len(s['launchers'])!=1: issues.append(('critical','duplicate_launcher' if len(s['launchers'])>1 else 'launcher_missing'))
    if len(s['workers'])!=2: issues.append(('critical','rank_worker_missing'))
    for rank,r in s['ranks'].items():
        if r['error_lines']: issues.append(('critical','operational_error'))
        if r['recompute_within'] is False: issues.append(('critical','recompute_failure'))
    if len(s['ranks'])==2:
        a,b=s['ranks'][0],s['ranks'][1]
        if a['complete_iteration'] and b['complete_iteration'] and a['complete_iteration']!=b['complete_iteration']: issues.append(('critical','ddp_mismatch'))
        if a['last_update_s'] and b['last_update_s'] and abs(a['last_update_s']-b['last_update_s'])>30: issues.append(('warning','ddp_mismatch'))
    last=s.get('last_progress_epoch') or now
    if now-last>=20*60: issues.append(('critical','stale_progress'))
    elif now-last>=15*60: issues.append(('warning','stale_progress'))
    free=s['disk']['free_bytes']
    if free<RESERVE: issues.append(('critical','disk_critical'))
    elif free<WARN: issues.append(('warning','disk_warning'))
    # A checkpoint is expected at each absolute 100k boundary. Only flag after crossing it;
    # before then, absence of LATEST is normal for a fresh run.
    interval=100000
    if s['committed_matches'] >= interval and s.get('latest_checkpoint') is None:
        issues.append(('critical','checkpoint_overdue'))
    return issues

def once(root):
    op=root/'operations'; op.mkdir(parents=True,exist_ok=True); lockf=open(op/'watchdog.lock','w')
    try: fcntl.flock(lockf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return {'status':'already_running'}
    s=snapshot(root); state_path=op/'watchdog_state.json'; state=_load(state_path,{'open':{}}); incidents=op/'incidents.jsonl'; now=time.time(); current={}
    for sev,typ in evaluate(s,state):
        key=typ; current[key]={'type':typ,'severity':sev,'timestamp':now,'committed_matches':s['committed_matches'],'launchers':[x['pid'] for x in s['launchers']],'workers':[x['pid'] for x in s['workers']]}
        if key not in state.get('open',{}):
            with incidents.open('a') as f: f.write(json.dumps(current[key],sort_keys=True)+'\n')
    for key in state.get('open',{}):
        if key not in current:
            with incidents.open('a') as f: f.write(json.dumps({'type':key,'event':'resolved','timestamp':now},sort_keys=True)+'\n')
    state={'timestamp':now,'open':current,'last_snapshot':s}; _write(state_path,state)
    return {'status':'ok','open':list(current),'matches':s['committed_matches'],'launcher_count':len(s['launchers'])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=DEFAULT_ROOT); ap.add_argument('--once',action='store_true'); ap.add_argument('--interval',type=int,default=180); a=ap.parse_args()
    if a.once: print(json.dumps(once(a.root))); return
    while True: print(json.dumps(once(a.root)),flush=True); time.sleep(max(10,a.interval))
if __name__=='__main__': main()
