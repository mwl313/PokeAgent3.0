"""Bounded A9.3 precision stability campaign launcher.

Runs the requested precision arm through the existing guarded PPO launcher;
the launcher remains the single source of truth for PPO and parity semantics.
"""
from __future__ import annotations
import argparse, subprocess, sys

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--precision-mode", choices=("D",), default="D")
    p.add_argument("--precision-stability", choices=("default", "fp16-reduction-off"), required=True)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--report", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("launcher_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = [sys.executable, "scripts/run_ddp_ppo.py", "--precision-mode", a.precision_mode,
           "--precision-stability", a.precision_stability, "--iterations", str(a.iterations),
           "--report", a.report]
    if a.checkpoint:
        cmd += ["--checkpoint", a.checkpoint]
    cmd += a.launcher_args
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
