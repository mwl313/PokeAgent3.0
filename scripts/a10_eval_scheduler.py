#!/usr/bin/env python3
"""CPU-only A10 evaluation scheduler: archive complete generations and queue jobs."""
from __future__ import annotations
import argparse, fcntl, json
from pathlib import Path
from a10_eval_pipeline import archive_snapshot, inspect_generation, schedule

def once(run_root):
    run=Path(run_root); lock=run/'evaluation/.scheduler.lock'; lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return {'status':'already_running'}
        try: info=inspect_generation(run/'rolling')
        except Exception as e: return {'status':'deferred','reason':str(e)}
        rec=archive_snapshot(run/'rolling',run/'evaluation/policies')
        jobs=schedule(run/'evaluation/schedule.json',info['manifest']['committed_matches'],rec['policy_id'])
        return {'status':'queued','checkpoint':rec,'created_jobs':jobs}
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True); a=p.parse_args(); print(json.dumps(once(a.run_root),indent=2))
