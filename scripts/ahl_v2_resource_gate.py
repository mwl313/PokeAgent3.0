#!/usr/bin/env python3
"""Print or persist the topology-derived historical-evaluation resource gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "engine" / "python")]

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from agent.evaluation.historical_league import (  # noqa: E402
    DEFAULT_FORMAL_ROOT,
    DEFAULT_RR_ROOT,
    formal_backlog,
)
from agent.evaluation.historical_resources import (  # noqa: E402
    DEFAULT_BASELINE_ROWS_PER_SEC,
    DEFAULT_FORMAL_CPUS,
    evaluate_gate,
    write_gate,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_RR_ROOT))
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    parser.add_argument("--ppo-running", action="store_true")
    parser.add_argument("--ppo-stopped", dest="ppo_running", action="store_false")
    parser.set_defaults(ppo_running=False)
    parser.add_argument("--mode-b-workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    parser.add_argument("--baseline-rows-per-sec", type=float,
                        default=DEFAULT_BASELINE_ROWS_PER_SEC)
    parser.add_argument("--throughput-sample", action="append", type=float, default=[],
                        help="completed PPO iteration learner rows/sec (repeatable)")
    parser.add_argument("--throughput-file", default=None,
                        help="JSON list of completed learner rows/sec samples")
    parser.add_argument("--formal-cpu", type=int, action="append", default=None)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    parser.add_argument("--output", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    samples = list(args.throughput_sample)
    if args.throughput_file:
        with open(args.throughput_file) as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("samples") or value.get("rows_per_sec") or []
        samples.extend(value)
    formal_cpus = tuple(args.formal_cpu) if args.formal_cpu else DEFAULT_FORMAL_CPUS
    payload = evaluate_gate(
        ppo_running=bool(args.ppo_running),
        formal=formal_backlog(args.formal_root),
        disk_path=args.root,
        reserve_gib=args.reserve_gib,
        mode_b_workers=args.mode_b_workers,
        formal_cpus=formal_cpus,
        throughput_samples=samples,
        baseline_rows_per_sec=args.baseline_rows_per_sec,
        threads_per_worker=args.threads,
    )
    output = Path(args.output) if args.output else Path(args.root) / "resource_gate.json"
    write_gate(output, payload)
    if args.format == "shell":
        print(f"GATE_STATUS={payload['status']}")
        print(f"GATE_MODE={payload['mode']}")
        print(f"GATE_WORKERS={payload['workers']}")
        print(f"GATE_THREADS={payload['threads_per_worker']}")
        masks = " ".join(f'"{row["mask"]}"' for row in payload["worker_masks"])
        print(f"GATE_MASKS=({masks})")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
