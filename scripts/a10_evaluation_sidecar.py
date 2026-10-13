#!/usr/bin/env python3
"""Independent A10 evaluation sidecar.

Default mode only discovers immutable checkpoints and queues durable jobs. Actual
CPU evaluation requires explicit --run-cpu and never selects a CUDA device.
"""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path
try:
 from .a10_eval_pipeline import archive_snapshot, inspect_generation, schedule, atomic_json
except ImportError:
 from a10_eval_pipeline import archive_snapshot, inspect_generation, schedule, atomic_json

def once(run_root, *, run_cpu=False):
    run=Path(run_root); ev=run/'evaluation'; lock=ev/'.sidecar.lock'; ev.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return {'status':'already_running'}
        try: info=inspect_generation(run/'rolling')
        except Exception as e: return {'status':'deferred','reason':str(e)}
        snapshot=archive_snapshot(run/'rolling',ev/'policies')
        reg_path=ev/'a10_registry.json'; reg=json.loads(reg_path.read_text()) if reg_path.exists() else {'schema':'a10-evaluation-registry-v1','policies':{}}
        reg['policies'][snapshot['policy_id']]=snapshot; atomic_json(reg_path,reg)
        created=schedule(ev/'schedule.json',info['manifest']['committed_matches'],snapshot['policy_id'])
        result={'status':'queued','policy_id':snapshot['policy_id'],'matches':snapshot['committed_matches'],'created_jobs':created,'run_cpu':False}
        if run_cpu:
            result['status']='deferred'; result['reason']='CPU evaluation execution is opt-in and requires a bounded resource window; no worker launched by sidecar'
        atomic_json(ev/'sidecar_state.json',{'timestamp':time.time(),**result})
        return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True); p.add_argument('--run-cpu',action='store_true'); a=p.parse_args(); print(json.dumps(once(a.run_root,run_cpu=a.run_cpu),indent=2))
if __name__=='__main__': main()
