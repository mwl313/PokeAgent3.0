#!/usr/bin/env python3
"""Read-only terminal dashboard for the V2 PPO campaign.

The dashboard only reads telemetry, manifests, systemd state, and ``nvidia-smi``.
It never opens a checkpoint for writing, signals a process, or imports the
learner.  This makes it safe to leave running beside production training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ROOT = Path("/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z")
DEFAULT_REPORT = "v2-final-8052606/continuation.json"
HORIZON = 100_000_000


def _run(args: list[str], timeout: float = 2.0) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    # Ignore a partially written final line; the writer appends one complete
    # JSON record per iteration and fsyncs it before returning.
    for raw in path.read_bytes().splitlines():
        try:
            value = json.loads(raw)
            if value.get("schema") == "pa3-ppo-training-metrics-v1":
                rows.append(value)
        except (ValueError, UnicodeDecodeError):
            continue
    return rows


def _service(unit: str) -> str:
    if not unit:
        return "unknown"
    return _run(["systemctl", "--user", "is-active", unit]) or "inactive"


def _gpus() -> list[dict]:
    text = _run(["nvidia-smi", "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit", "--format=csv,noheader,nounits"])
    result = []
    for line in text.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 7:
            continue
        try:
            result.append(dict(index=int(values[0]), temp=float(values[1]), util=float(values[2]),
                               memory=float(values[3]), memory_total=float(values[4]),
                               power=float(values[5]), power_limit=float(values[6])))
        except ValueError:
            pass
    return result


def snapshot(root: Path, report: str = DEFAULT_REPORT) -> dict:
    report_path = root / "reports" / report
    ranks = [_read_rows(Path(f"{report_path}.rank{rank}.ppo.jsonl")) for rank in (0, 1)]
    paired = min(len(ranks[0]), len(ranks[1]))
    latest = [rows[-1] for rows in ranks if rows]
    clock = max((row["identity"]["committed_matches"] for row in latest), default=0)
    speed = [row["performance"]["training_games_per_second"] for row in latest]
    recompute = [row["safety"]["recompute"]["max_abs_diff"] for row in latest]
    errors = sum(row["collector"].get("operational_errors", 0) for row in latest)
    skips = sum(row["ppo"].get("optimizer_steps_skipped", 0) for row in latest)
    latest_generation = None
    pointer = root / "rolling" / "LATEST"
    if pointer.exists():
        name = pointer.read_text(errors="replace").strip()
        manifest = root / "rolling" / "generations" / name / "manifest.json"
        if name and manifest.exists():
            try:
                value = json.loads(manifest.read_text())
                if value.get("complete"):
                    latest_generation = dict(name=name, matches=value.get("committed_matches"), iteration=value.get("iteration"))
            except (OSError, ValueError):
                pass
    state_path = root / "evaluation" / "v2" / "state.json"
    evaluation = {}
    if state_path.exists():
        try:
            evaluation = json.loads(state_path.read_text())
        except (OSError, ValueError):
            evaluation = {"status": "unreadable"}
    usage = shutil.disk_usage(root)
    return {
        "captured_at": time.time(), "root": str(root), "report": str(report_path),
        "rank_records": [len(rows) for rows in ranks], "paired_records": paired,
        "matches": clock, "progress": clock / HORIZON, "speed": speed,
        "latest": latest, "latest_generation": latest_generation,
        "errors": errors, "skipped_updates": skips, "recompute_max": max(recompute, default=None),
        "services": {
            "training": _service("pa3-v2-ppo-final-8052606.service"),
            "evaluation": _service("pa3-v2-evaluation-final-8052606.timer"),
            "monitor": _service("pa3-v2-monitor-final-8052606.timer"),
        },
        "evaluation_state": evaluation, "disk_free_gib": usage.free / 2**30,
        "gpus": _gpus(),
    }


def _bar(fraction: float, width: int = 34) -> str:
    filled = max(0, min(width, int(fraction * width)))
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def render(data: dict) -> str:
    latest = data["latest"]
    row = latest[0] if latest else {}
    ppo = row.get("ppo", {})
    safety = row.get("safety", {})
    rank_ok = all(item.get("safety", {}).get("rank_verification", {}).get("digests_equal", False) for item in latest)
    status = "RUNNING" if data["services"]["training"] == "active" else data["services"]["training"].upper()
    lines = ["\033[H\033[J", "PokeAgent 3.0 V2  |  read-only training monitor", "═" * 76,
             f"TRAINING   {status:<10}  {data['matches']:,} / {HORIZON:,} natural matches  {data['progress']:.4%}",
             f"           {_bar(data['progress'])}  paired telemetry {data['paired_records']} iterations",
             f"RECENT     {max(data['speed'], default=0):6.2f} games/s  |  LR {ppo.get('learning_rate', 0):.8f}  |  scaler {ppo.get('scaler_scale', '—')}",
             f"           policy {ppo.get('policy_loss', '—')}  value {ppo.get('value_loss', '—')}  entropy {ppo.get('entropy', '—')}",
             f"           approx KL {ppo.get('approx_kl', '—')}  clip {ppo.get('clip_fraction', '—')}  epochs {ppo.get('epochs_run', '—')}",
             f"SAFETY     DDP digests {'OK' if rank_ok else 'CHECK'}  |  recompute max {data['recompute_max'] if data['recompute_max'] is not None else '—'}  |  errors {data['errors']}  |  skipped {data['skipped_updates']}",
             f"SERVICES   train {data['services']['training']}  |  eval timer {data['services']['evaluation']}  |  monitor {data['services']['monitor']}",
             f"EVALUATION {data['evaluation_state'].get('status', 'unknown')}  |  best {data['evaluation_state'].get('best', '—')}  |  tested {len(data['evaluation_state'].get('tested', []))}",
             f"STORAGE    {data['disk_free_gib']:.2f} GiB free  |  checkpoint {data['latest_generation'] or 'not yet published'}"]
    for gpu in data["gpus"]:
        lines.append(f"GPU {gpu['index']}       {gpu['util']:5.1f}% util  {gpu['temp']:4.1f}°C  {gpu['power']:5.1f}/{gpu['power_limit']:g}W  {gpu['memory']:.0f}/{gpu['memory_total']:.0f} MiB")
    lines.append("\nPress Ctrl-C to exit. This process only reads files and system status.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only live V2 PPO training dashboard")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", default=DEFAULT_REPORT, help="report path relative to <run-root>/reports")
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS", help="refresh interval; 0 prints one snapshot")
    parser.add_argument("--json", action="store_true", help="print one machine-readable snapshot")
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch must be nonnegative")
    try:
        while True:
            data = snapshot(args.run_root.resolve(), args.report)
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True))
                return
            print(render(data), flush=True)
            if args.watch == 0:
                return
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nmonitor stopped; training was not signaled", file=sys.stderr)


if __name__ == "__main__":
    main()
