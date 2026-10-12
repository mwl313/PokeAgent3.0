"""Bounded A9.4.2 acceptance runner; GPU evaluation is explicitly sequential."""
from __future__ import annotations
import argparse, json, subprocess, sys
from agent.runtime.config import resolve_profile

def main():
    p=argparse.ArgumentParser(); p.add_argument('--profile',default='configs/validated_a94_runtime.yaml'); p.add_argument('--report',default='runs/experiments/a/a942_acceptance.json'); a=p.parse_args()
    resolved=resolve_profile(a.profile)
    cmd=[sys.executable,'-m','pytest','-q','tests/agent/test_runtime_config.py','tests/agent/test_a91_precision.py','tests/agent/test_evaluation_league.py','tests/agent/test_logprob_recompute.py','tests/agent/test_schedule_and_checkpoint.py']
    test=subprocess.run(cmd)
    payload={'profile':resolved,'pytest_returncode':test.returncode,'gpu_evidence':'runs/experiments/a/a942_fresh_eval_receipts.json','engine_readiness':'NOT VERIFIED if cargo is unavailable'}
    with open(a.report,'w') as f: json.dump(payload,f,indent=2,sort_keys=True)
    raise SystemExit(test.returncode)
if __name__=='__main__': main()
