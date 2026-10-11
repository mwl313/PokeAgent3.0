#!/usr/bin/env python3
"""Sequential A9.2 campaign runner.

The runner deliberately launches one dual-rank job at a time.  It records the
exact command and return code for every arm; ``--plan-only`` is safe for review,
while ``--execute`` runs the bounded matrix from independent A5 Control
checkpoints without touching the historical artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", default=os.path.join(ROOT, "runs/experiments/a/a92_campaign.json"))
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args(argv)
    common = [
        args.python, os.path.join(ROOT, "scripts/run_ddp_ppo.py"),
        "--games", "4", "--envs", "1024", "--workers", "16",
        "--microbatch", "1024", "--minibatch", "4096", "--epochs", "2",
        "--experimental-lr-scale", "0.5", "--experimental-resume",
        os.path.join(ROOT, "runs/experiments/a/a5_control/boundary"),
        "--experimental-allow-source-sha", "--experimental-allow-opponent-migration",
        "--recompute-gate",
    ]
    arms = [
        ("REF", "A", 6, True),
        ("A", "A", 8, False),
        ("B", "B", 8, False),
        ("C", "C", 8, False),
        ("D", "D", 8, False),
        ("FINAL", "D", 25, False),
    ]
    receipt = {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "arms": [], "executed": bool(args.execute)}
    for name, mode, iterations, audit in arms:
        report = os.path.join(ROOT, f"runs/experiments/a/a92_campaign_{name}.json")
        cmd = common + ["--iterations", str(iterations), "--precision-mode", mode, "--report", report]
        if audit:
            cmd.append("--recompute-audit")
        entry = {"name": name, "mode": mode, "iterations": iterations, "command": cmd,
                 "report": report, "started": time.time()}
        if args.execute:
            result = subprocess.run(cmd, cwd=ROOT, text=True)
            entry["returncode"] = result.returncode
            if result.returncode:
                entry["status"] = "failed_closed"
                receipt["arms"].append(entry)
                # Unsafe precision candidates do not block independent arms.
                continue
            entry["status"] = "completed"
        else:
            entry["status"] = "planned"
        entry["finished"] = time.time()
        receipt["arms"].append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
