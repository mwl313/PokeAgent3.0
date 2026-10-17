"""CPU topology discovery and safe resource gates for historical diagnostics.

Nothing in this module changes the production PPO launcher.  It derives worker
affinities from the live sysfs topology, keeps the formal evaluation CPUs and
their SMT siblings reserved, and fails safe (zero workers) on any conflict,
disk/memory pressure or formal-evaluation backlog.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from agent.evaluation.v2_league import atomic_json

DEFAULT_FORMAL_CPUS = (4, 5, 6, 7)
DEFAULT_BASELINE_ROWS_PER_SEC = 381.9
THROUGHPUT_MIN_SAMPLES = 3
THROUGHPUT_WARN_DEGRADATION = 0.03
THROUGHPUT_PAUSE_DEGRADATION = 0.05
DEFAULT_RESERVE_GIB = 12.0
MIN_AVAILABLE_MEMORY_GIB = 8.0
MAX_SWAP_USED_GIB = 2.0


def parse_cpu_list(value: str) -> list[int]:
    cpus = []
    for part in str(value).strip().split(","):
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            cpus.extend(range(int(low), int(high) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus) -> str:
    values = sorted(set(int(cpu) for cpu in cpus))
    if not values:
        return ""
    ranges = []
    start = previous = values[0]
    for cpu in values[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def read_cpu_topology(sys_root="/sys/devices/system") -> dict:
    sys_root = Path(sys_root)
    node_for_cpu = {}
    nodes = {}
    for node_dir in sorted((sys_root / "node").glob("node*")):
        try:
            node_id = int(node_dir.name.replace("node", ""))
        except ValueError:
            continue
        cpulist = (node_dir / "cpulist").read_text().strip()
        cpus = parse_cpu_list(cpulist)
        nodes[node_id] = cpus
        for cpu in cpus:
            node_for_cpu[cpu] = node_id
    cpus = {}
    cores = {}
    for cpu_dir in sorted((sys_root / "cpu").glob("cpu[0-9]*")):
        try:
            cpu = int(cpu_dir.name.replace("cpu", ""))
        except ValueError:
            continue
        topology = cpu_dir / "topology"
        if not topology.is_dir():
            continue
        try:
            core_id = int((topology / "core_id").read_text().strip())
            package_id = int((topology / "physical_package_id").read_text().strip())
            siblings = parse_cpu_list(
                (topology / "thread_siblings_list").read_text().strip())
        except (OSError, ValueError):
            continue
        node = node_for_cpu.get(cpu)
        key = (package_id, core_id)
        cpus[cpu] = {
            "cpu": cpu,
            "core_id": core_id,
            "package_id": package_id,
            "node": node,
            "siblings": siblings,
        }
        cores.setdefault(key, {
            "key": list(key),
            "core_id": core_id,
            "package_id": package_id,
            "node": node,
            "cpus": [],
        })
        cores[key]["cpus"].append(cpu)
    for core in cores.values():
        core["cpus"].sort()
    return {
        "cpus": cpus,
        "cores": cores,
        "nodes": nodes,
        "logical_cpus": len(cpus),
        "physical_cores": len(cores),
        "smt_enabled": any(len(core["cpus"]) > 1 for core in cores.values()),
    }


def reserved_cpus(formal_cpus=DEFAULT_FORMAL_CPUS, topology=None) -> set[int]:
    """Reserve the formal CPUs and every SMT sibling / co-located core CPU."""
    topology = topology or read_cpu_topology()
    reserved = set()
    for cpu in formal_cpus:
        details = topology["cpus"].get(int(cpu))
        if not details:
            reserved.add(int(cpu))
            continue
        reserved.update(details["siblings"])
        for core in topology["cores"].values():
            if int(cpu) in core["cpus"]:
                reserved.update(core["cpus"])
                break
    return reserved


def derive_worker_masks(topology, *, workers, threads_per_worker=2,
                        formal_cpus=DEFAULT_FORMAL_CPUS, use_smt=False) -> list[dict]:
    """Assign distinct physical cores, never touching formal evaluation cores."""
    workers = max(0, int(workers))
    threads_per_worker = max(1, int(threads_per_worker))
    reserved = reserved_cpus(formal_cpus, topology)
    usable = []
    for core in sorted(
            topology["cores"].values(),
            key=lambda row: (row["node"] if row["node"] is not None else 99,
                             row["core_id"], row["package_id"])):
        if any(cpu in reserved for cpu in core["cpus"]):
            continue
        # Keep one hardware thread per core; the sibling remains available to
        # the OS and other services rather than being silently consumed.
        usable.append(core["cpus"][0])
    masks = []
    cursor = 0
    for worker in range(workers):
        cpus = usable[cursor:cursor + threads_per_worker]
        cursor += threads_per_worker
        if len(cpus) < threads_per_worker:
            break
        nodes = {topology["cpus"][cpu]["node"] for cpu in cpus}
        masks.append({
            "worker": worker + 1,
            "cpus": cpus,
            "mask": format_cpu_list(cpus),
            "nodes": sorted(node for node in nodes if node is not None),
            "smt": use_smt,
        })
    return masks


def mask_overlaps(masks, reserved=()) -> list[dict]:
    reserved = set(int(cpu) for cpu in reserved)
    conflicts = []
    seen = {}
    for row in masks:
        cpus = set(int(cpu) for cpu in row.get("cpus", []))
        shared = sorted(cpus & reserved)
        if shared:
            conflicts.append({
                "worker": row.get("worker"),
                "reason": "overlaps_reserved_cpus",
                "cpus": shared,
            })
        for cpu in cpus:
            if cpu in seen:
                conflicts.append({
                    "worker": row.get("worker"),
                    "reason": "overlaps_other_worker",
                    "cpus": [cpu],
                    "other_worker": seen[cpu],
                })
            else:
                seen[cpu] = row.get("worker")
    return conflicts


def _meminfo() -> dict:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                values[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return values


def _load_average() -> list[float]:
    try:
        return list(os.getloadavg())
    except OSError:
        return []


def disk_free_bytes(path="/") -> int:
    usage = os.statvfs(str(path))
    return usage.f_bavail * usage.f_frsize


def throughput_gate(samples, *, baseline_rows_per_sec=DEFAULT_BASELINE_ROWS_PER_SEC,
                    min_samples=THROUGHPUT_MIN_SAMPLES) -> dict:
    """Classify sustained PPO throughput degradation from completed iterations."""
    values = []
    for sample in samples or []:
        try:
            value = float(sample)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if len(values) < int(min_samples):
        return {
            "status": "insufficient_samples",
            "verified": False,
            "samples": len(values),
            "min_samples": int(min_samples),
            "action": "keep_conservative_default",
            "reason": "not enough completed PPO iterations to distinguish noise",
        }
    window = values[-int(min_samples):]
    sustained = statistics.median(window)
    degradation = (float(baseline_rows_per_sec) - sustained) / float(baseline_rows_per_sec)
    if degradation > THROUGHPUT_PAUSE_DEGRADATION:
        action = "pause"
        status = "pause"
    elif degradation > THROUGHPUT_WARN_DEGRADATION:
        action = "reduce_concurrency"
        status = "degraded"
    else:
        action = "accept"
        status = "ok"
    return {
        "status": status,
        "verified": True,
        "samples": len(values),
        "window": window,
        "sustained_rows_per_sec": sustained,
        "baseline_rows_per_sec": float(baseline_rows_per_sec),
        "degradation": degradation,
        "action": action,
        "thresholds": {
            "warn": THROUGHPUT_WARN_DEGRADATION,
            "pause": THROUGHPUT_PAUSE_DEGRADATION,
        },
        "sustained_definition": f"median of the last {int(min_samples)} completed iterations",
    }


def evaluate_gate(*, ppo_running: bool, formal: dict, topology=None,
                  disk_path="/", reserve_gib=DEFAULT_RESERVE_GIB,
                  mode_b_workers=16, formal_cpus=DEFAULT_FORMAL_CPUS,
                  throughput_samples=(), baseline_rows_per_sec=DEFAULT_BASELINE_ROWS_PER_SEC,
                  threads_per_worker=2) -> dict:
    topology = topology or read_cpu_topology()
    reserve_bytes = int(float(reserve_gib) * (1 << 30))
    free = disk_free_bytes(disk_path)
    meminfo = _meminfo()
    available_memory = meminfo.get("MemAvailable", 0)
    swap_total = meminfo.get("SwapTotal", 0)
    swap_free = meminfo.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)
    reserved = sorted(reserved_cpus(formal_cpus, topology))
    reasons = []
    if formal.get("pending_or_running"):
        reasons.append("formal evaluation has pending or running jobs")
    if free < reserve_bytes:
        reasons.append("disk reserve would be breached")
    if available_memory and available_memory < int(MIN_AVAILABLE_MEMORY_GIB * (1 << 30)):
        reasons.append("available memory is below the 8 GiB guard")
    if swap_used > int(MAX_SWAP_USED_GIB * (1 << 30)):
        reasons.append("swap usage is above the 2 GiB guard")
    throttled_by = None
    if ppo_running:
        workers = 2
        threads = 2
        throughput = throughput_gate(
            throughput_samples, baseline_rows_per_sec=baseline_rows_per_sec)
        if throughput["action"] == "pause":
            workers = 0
            reasons.append("sustained PPO throughput degradation exceeds 5%")
        elif throughput["action"] == "reduce_concurrency":
            workers = 1
            threads = 1
            throttled_by = "throughput_degradation_3_to_5_percent"
        mode = "A_ppo_running"
    else:
        workers = min(int(mode_b_workers), max(1, len(topology["cores"]) // 2))
        threads = max(1, int(threads_per_worker))
        throughput = throughput_gate([]) if not throughput_samples else throughput_gate(
            throughput_samples, baseline_rows_per_sec=baseline_rows_per_sec)
        mode = "B_ppo_stopped"
    if reasons:
        workers = 0
    masks = derive_worker_masks(
        topology, workers=workers, threads_per_worker=threads,
        formal_cpus=formal_cpus) if workers else []
    conflicts = mask_overlaps(masks, reserved)
    if conflicts:
        reasons.append("worker CPU masks conflict with reserved or each other")
        masks = []
        workers = 0
    status = "paused" if workers == 0 else ("degraded" if throttled_by else "ok")
    return {
        "schema": "pa3-v2-history-resource-gate-v1",
        "generated_at": time.time(),
        "generated_at_kst": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "mode": mode,
        "workers": len(masks),
        "threads_per_worker": threads,
        "worker_masks": masks,
        "reserved_cpus": reserved,
        "formal_cpus": sorted(int(cpu) for cpu in formal_cpus),
        "conflicts": conflicts,
        "reasons": reasons,
        "ppo_running": bool(ppo_running),
        "coexistence_verified": False,
        "coexistence_note": ("UNVERIFIED UNDER LIVE PPO — no live PPO throughput "
                             "measurement exists while training is stopped"),
        "throughput": throughput,
        "throttled_by": throttled_by,
        "disk": {
            "path": str(disk_path),
            "free_bytes": free,
            "reserve_bytes": reserve_bytes,
            "ok": free >= reserve_bytes,
        },
        "memory": {
            "available_bytes": available_memory,
            "swap_used_bytes": swap_used,
            "guard_bytes": int(MIN_AVAILABLE_MEMORY_GIB * (1 << 30)),
        },
        "load_average": _load_average(),
        "formal_priority": formal,
        "topology": {
            "logical_cpus": topology["logical_cpus"],
            "physical_cores": topology["physical_cores"],
            "smt_enabled": topology["smt_enabled"],
            "nodes": topology["nodes"],
        },
        "production_launcher_modified": False,
    }


def write_gate(path, payload) -> dict:
    atomic_json(Path(path), payload)
    return payload
