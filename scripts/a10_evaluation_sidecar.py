#!/usr/bin/env python3
"""Discovery poller; the separate battle worker alone owns the evaluation queue."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a10_league_discover import discover
from agent.evaluation.a10_league import read


def once(run_root, **kwargs):
    registry = discover(run_root)
    own = Path(run_root) / 'evaluation/league_v2'
    jobs = read(own / 'jobs.json', {'jobs': {}})
    return {
        'policies': len(registry['policies']),
        'pending': sum(job['status'] == 'pending' for job in jobs['jobs'].values()),
        'best': read(own / 'best_policy.json'),
        'rejected': registry['rejected'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    print(json.dumps(once(args.run_root), indent=2))
