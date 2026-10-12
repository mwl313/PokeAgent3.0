#!/usr/bin/env python3
"""Report or safely clean PokeAgent-owned checkpoint storage."""
from __future__ import annotations
import argparse, json, sys, importlib.util
from pathlib import Path
_module_path = Path(__file__).resolve().parents[1] / "agent/ppo/storage_manager.py"
_spec = importlib.util.spec_from_file_location("pa3_storage_manager", _module_path)
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
CheckpointGenerationStore = _mod.CheckpointGenerationStore

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/checkpoints-rolling")
    p.add_argument("--project-root", default=".")
    p.add_argument("--project-budget-gb", type=float, default=50.0)
    p.add_argument("--projected-write-gb", type=float, default=0.0)
    p.add_argument("--cleanup-incomplete", action="store_true")
    args = p.parse_args()
    store = CheckpointGenerationStore(Path(args.project_root) / args.root, budget_bytes=int(args.project_budget_gb * 1_000_000_000))
    removed = store.cleanup_incomplete() if args.cleanup_incomplete else []
    print(json.dumps({"report": store.disk_report(projected_write_bytes=int(args.projected_write_gb * 1_000_000_000)), "removed_incomplete": removed}, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
