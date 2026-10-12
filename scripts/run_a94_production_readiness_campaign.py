"""Reproducible A9.4 bounded readiness campaign.

The GPU stages are intentionally explicit and sequential; this wrapper records
the commands and runs the CPU regression suite before any requested workload.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--skip-gpu', action='store_true')
    p.add_argument('--report', default='runs/experiments/a/a94_campaign.json')
    a=p.parse_args(); started=time.time(); results=[]
    tests=[['-m','pytest','-q','tests/agent/test_a91_precision.py','tests/agent/test_evaluation_league.py','tests/agent/test_logprob_recompute.py']]
    for args in tests:
        proc=subprocess.run([sys.executable,*args], text=True)
        results.append({'command':[sys.executable,*args],'returncode':proc.returncode})
        if proc.returncode: break
    payload={'campaign':'A9.4','started_unix':started,'cpu_stages':results,
             'gpu_stages':'skipped; invoke the recorded bounded launcher commands sequentially' if a.skip_gpu else 'not started by wrapper'}
    with open(a.report,'w') as f: json.dump(payload,f,indent=2)
    raise SystemExit(0 if all(x['returncode']==0 for x in results) else 1)
if __name__=='__main__': main()
