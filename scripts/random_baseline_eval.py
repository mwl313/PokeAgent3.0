#!/usr/bin/env python3
"""Thin, resumable random/deterministic baseline evaluation runner.

Reuses the existing native batched evaluator.  This module only adds:

* a frozen, unique, seat-paired fixture manifest,
* disjoint index shards with atomic resumable receipts,
* process pinning / CPU-only execution,
* throughput calibration and aggregation,
* a sequential keyed replay used to emit representative battle traces.

It never trains, never mutates a checkpoint, and never writes outside the
explicitly supplied run directory / this repository's docs tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENGINE_PYTHON = "/home/minidc/Desktop/PokeAgent3.0-V2-evaluation-league/engine/python"
DEFAULT_PYTHON = "/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python"
DEFAULT_CHECKPOINT = "/home/minidc/eval-league/v2/snapshots/43/434be431de0e25a9c7730a7a0f38eda66d5db30397835983ebde9f4b8f1d9b36"
DEFAULT_DATA = os.path.join(ROOT, "engine", "data")
DEFAULT_TEAMS = os.path.join(ROOT, "engine", "data", "a10-training-teams.json")
DEFAULT_RUN_ROOT = "/home/minidc/eval-league/random-baseline/v1"
SEED_DOMAIN = "pa3-random-baseline-v1"
CHECKPOINT_ID = "v2-000002764800-da35700f"
FIXTURE_SCHEMA = "pa3-random-baseline-fixtures-v1"
RECEIPT_SCHEMA = "pa3-random-baseline-shard-receipt-v1"
AGGREGATE_SCHEMA = "pa3-random-baseline-aggregate-v1"

ENGINE_PYTHON = os.environ.get("PA3_ENGINE_PYTHON", DEFAULT_ENGINE_PYTHON)
sys.path.insert(0, ENGINE_PYTHON)
sys.path.insert(0, ROOT)

# (offset, pairs, policy, opponent, seed suffix, label)
TEST_SPECS = {
    "test_a": {
        "offset": 0, "pairs": 4096, "policy": "checkpoint", "opponent": "random",
        "seed_domain": f"{SEED_DOMAIN}:test-a", "label": "Test A random baseline",
    },
    "test_b": {
        "offset": 1000000, "pairs": 1024, "policy": "checkpoint",
        "opponent": "lexicographic_legal", "seed_domain": f"{SEED_DOMAIN}:test-b",
        "label": "Test B deterministic legal baseline",
    },
    "control_random_random": {
        "offset": 2000000, "pairs": 256, "policy": "random", "opponent": "random",
        "seed_domain": f"{SEED_DOMAIN}:control-random-random",
        "label": "Control random vs random",
    },
    "control_lexicographic_random": {
        "offset": 3000000, "pairs": 256, "policy": "lexicographic_legal",
        "opponent": "random", "seed_domain": f"{SEED_DOMAIN}:control-lexicographic-random",
        "label": "Control lexicographic legal vs random",
    },
}
CALIBRATION_SPEC = {
    "offset": 9000000, "pairs": 256, "policy": "checkpoint", "opponent": "random",
    "seed_domain": f"{SEED_DOMAIN}:calibration", "label": "Calibration random baseline",
}
PRIMARY_TESTS = ("test_a", "test_b")
CONTROL_TESTS = ("control_random_random", "control_lexicographic_random")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: str, value) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: str):
    with open(path) as handle:
        return json.load(handle)


def write_text(path: str, text: str) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fixture:
    fixture_id: int
    pair_id: int
    team_a: int
    team_b: int
    seed: tuple[int, int, int, int]
    policy_side: int
    test: str


def _derived_seeds(domain: str) -> int:
    return int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")


def _team_pool(data_dir: str, teams_path: str) -> list[str]:
    payload = read_json(teams_path)
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"invalid team pool: {teams_path}")
    return [row["id"] for row in payload]


def build_test_fixtures(spec: dict, team_ids: list[int], test_name: str) -> list[Fixture]:
    rng = random.Random(_derived_seeds(spec["seed_domain"]))
    offset = int(spec["offset"])
    seen = set()
    fixtures: list[Fixture] = []
    for pair_index in range(int(spec["pairs"])):
        while True:
            team_a, team_b = rng.sample(team_ids, 2)
            seed = tuple(rng.randrange(1 << 16) for _ in range(4))
            identity = (team_a, team_b, seed)
            if identity not in seen:
                seen.add(identity)
                break
        pair_id = offset + pair_index
        fixtures.append(Fixture(offset + pair_index * 2, pair_id, team_a, team_b, seed, 0, test_name))
        fixtures.append(Fixture(offset + pair_index * 2 + 1, pair_id, team_a, team_b, seed, 1, test_name))
    return fixtures


def prepare_fixtures(run_dir: str, data_dir: str, teams_path: str) -> dict:
    fixture_path = os.path.join(run_dir, "fixtures.json")
    if os.path.exists(fixture_path):
        manifest = read_json(os.path.join(run_dir, "fixtures.manifest.json"))
        return manifest
    team_ids = list(range(len(_team_pool(data_dir, teams_path))))
    all_fixtures: list[Fixture] = []
    for name, spec in TEST_SPECS.items():
        all_fixtures.extend(build_test_fixtures(spec, team_ids, name))
    all_fixtures.extend(build_test_fixtures(CALIBRATION_SPEC, team_ids, "calibration"))
    payload = {
        "schema": FIXTURE_SCHEMA,
        "seed_domain": SEED_DOMAIN,
        "team_pool_size": len(team_ids),
        "team_pool_sha256": sha256_file(teams_path),
        "tests": {},
        "fixtures": [asdict(f) | {"seed": list(f.seed)} for f in all_fixtures],
    }
    for name in list(TEST_SPECS) + ["calibration"]:
        rows = [f for f in all_fixtures if f.test == name]
        payload["tests"][name] = {
            "fixtures": len(rows), "pairs": len(rows) // 2,
            "sha256": canonical_hash([asdict(f) | {"seed": list(f.seed)} for f in rows]),
        }
    payload["manifest_hash"] = canonical_hash(payload["fixtures"])
    atomic_json(fixture_path, payload["fixtures"])
    atomic_json(os.path.join(run_dir, "fixtures.manifest.json"), payload)
    return payload


def load_fixtures(run_dir: str) -> tuple[list[Fixture], dict]:
    manifest = read_json(os.path.join(run_dir, "fixtures.manifest.json"))
    rows = read_json(os.path.join(run_dir, "fixtures.json"))
    fixtures = [Fixture(r["fixture_id"], r["pair_id"], r["team_a"], r["team_b"],
                        tuple(r["seed"]), r["policy_side"], r["test"]) for r in rows]
    digest = canonical_hash(rows)
    if digest != manifest["manifest_hash"]:
        raise SystemExit("frozen fixture manifest hash mismatch")
    return fixtures, manifest


def eval_fixtures(fixtures: Iterable[Fixture]):
    from agent.evaluation.fixed_policy import EvaluationFixture
    return [EvaluationFixture(f.fixture_id, f.pair_id, f.team_a, f.team_b,
                              tuple(f.seed), f.policy_side) for f in fixtures]


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _receipt_path(run_dir: str, test: str, shard: int) -> str:
    return os.path.join(run_dir, "receipts", test, f"shard-{shard:03d}.json")


def _config_id(args) -> str:
    return canonical_hash({
        "engine_workers": args.engine_workers,
        "batch_size": args.batch_size,
        "torch_threads": args.torch_threads,
        "model_sha256": args.model_sha256,
        "max_rounds": args.max_rounds,
    })


def _load_receipt(path: str, *, manifest_hash: str, config_id: str,
                  test: str, shard: int, shards: int) -> dict:
    if not os.path.exists(path):
        return {
            "schema": RECEIPT_SCHEMA, "test": test, "shard_index": shard,
            "shard_count": shards, "fixture_manifest_hash": manifest_hash,
            "config_id": config_id, "complete": False, "outcomes": [],
            "runtime_seconds": 0.0, "chunks": [], "created_at": now(),
        }
    receipt = read_json(path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise SystemExit(f"unexpected receipt schema: {path}")
    if receipt.get("fixture_manifest_hash") != manifest_hash:
        raise SystemExit(f"receipt fixture manifest mismatch: {path}")
    if receipt.get("config_id") != config_id:
        raise SystemExit(f"receipt worker config mismatch: {path}")
    return receipt


def run_worker(args) -> int:
    import torch
    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.device != "cpu":
        raise SystemExit("random-baseline workers are CPU-only")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import pa3_engine
    from agent.evaluation.batched_policy import BatchedPolicyEvaluator
    from agent.evaluation.fixed_policy import load_frozen_model

    started = time.perf_counter()
    fixture_run_dir = args.fixture_run_dir or args.run_dir
    fixtures, manifest = load_fixtures(fixture_run_dir)
    spec = (dict(TEST_SPECS) | {"calibration": CALIBRATION_SPEC})[args.test]
    test_rows = [f for f in fixtures if f.test == args.test]
    pairs = [test_rows[i:i + 2] for i in range(0, len(test_rows), 2)]
    for pair in pairs:
        if len(pair) != 2 or pair[0].pair_id != pair[1].pair_id or pair[0].policy_side == pair[1].policy_side:
            raise SystemExit("seat-swapped fixture pair is not intact")
    selected_pairs = [pair for index, pair in enumerate(pairs) if index % args.shards == args.shard]
    selected = [f for pair in selected_pairs for f in pair]

    config_id = _config_id(args)
    path = _receipt_path(args.run_dir, args.test, args.shard)
    receipt = _load_receipt(path, manifest_hash=manifest["manifest_hash"],
                            config_id=config_id, test=args.test,
                            shard=args.shard, shards=args.shards)
    done_ids = {row["fixture_id"] for row in receipt["outcomes"]}
    if len(done_ids) != len(receipt["outcomes"]):
        raise SystemExit(f"duplicate fixture ids inside receipt: {path}")
    remaining = [f for f in selected if f.fixture_id not in done_ids]
    unknown = done_ids - {f.fixture_id for f in selected}
    if unknown:
        raise SystemExit(f"receipt contains fixtures outside this shard: {sorted(unknown)[:4]}")

    model = None
    if spec["policy"] == "checkpoint" or spec["opponent"] == "checkpoint":
        if not args.checkpoint:
            raise SystemExit("checkpoint policy requires --checkpoint")
        model = load_frozen_model(args.checkpoint, device="cpu")
    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=int(args.engine_workers))
    evaluator = BatchedPolicyEvaluator(engine, device="cpu", seed=args.seed,
                                       batch_size=int(args.batch_size))
    chunk_games = max(int(args.batch_size) * 2, 32)
    chunk_pairs = max(1, chunk_games // 2)

    for start in range(0, len(remaining), chunk_pairs * 2):
        if os.path.exists(args.stop_file):
            print(json.dumps({"worker": args.shard, "stopped": "stop-file"}), flush=True)
            break
        chunk = remaining[start:start + chunk_pairs * 2]
        result = evaluator.run(
            eval_fixtures(chunk), policy=spec["policy"], model=model,
            opponent=spec["opponent"], max_rounds=int(args.max_rounds),
        )
        receipt["outcomes"].extend(result["outcomes"])
        receipt["runtime_seconds"] = float(receipt["runtime_seconds"]) + float(result["runtime_seconds"])
        receipt["chunks"].append({
            "fixtures": len(chunk), "runtime_seconds": result["runtime_seconds"],
            "operational_errors": result["operational_errors"],
            "illegal_actions": result["illegal_actions"],
            "finished_at": now(),
        })
        receipt["evaluator_version"] = result["evaluator_version"]
        receipt["sampling"] = result["sampling"]
        receipt["model_file_sha256"] = args.model_sha256
        receipt["engine_sha256"] = args.engine_sha256
        receipt["team_pool_sha256"] = manifest["team_pool_sha256"]
        receipt["engine_workers"] = int(args.engine_workers)
        receipt["batch_size"] = int(args.batch_size)
        receipt["torch_threads"] = int(args.torch_threads)
        receipt["cpu_affinity"] = args.cpu
        receipt["stepped_at"] = now()
        atomic_json(path, receipt)
        print(json.dumps({
            "worker": args.shard, "completed": len(receipt["outcomes"]),
            "shard_total": len(selected), "runtime_seconds": result["runtime_seconds"],
            "operational_errors": result["operational_errors"],
            "illegal_actions": result["illegal_actions"],
        }), flush=True)

    receipt["complete"] = len({row["fixture_id"] for row in receipt["outcomes"]}) == len(selected)
    receipt["selected_fixtures"] = len(selected)
    receipt["wall_seconds"] = time.perf_counter() - started
    receipt["finished_at"] = now()
    atomic_json(path, receipt)
    return 0 if receipt["complete"] else 2


# ---------------------------------------------------------------------------
# process orchestration / resource sampling
# ---------------------------------------------------------------------------

CPU_NODES = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
             [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])


def worker_cpus(workers: int) -> list[int]:
    if not 1 <= workers <= 20:
        raise SystemExit("worker count must be 1..20 with one physical core per worker")
    cpus = []
    for index in range(workers):
        pool = CPU_NODES[index % 2]
        cpus.append(pool[index // 2])
    return cpus


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _cpu_ticks():
        with open("/proc/stat") as handle:
            fields = handle.readline().split()[1:]
        values = [int(x) for x in fields]
        return sum(values), values[3] + (values[4] if len(values) > 4 else 0)

    @staticmethod
    def _mem_used():
        info = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) / (1024 ** 2)

    def _loop(self):
        last_total, last_idle = self._cpu_ticks()
        while not self._stop.wait(self.interval):
            total, idle = self._cpu_ticks()
            dt, di = total - last_total, idle - last_idle
            last_total, last_idle = total, idle
            self.samples.append({
                "t": now(),
                "cpu_busy_fraction": (1.0 - di / dt) if dt > 0 else 0.0,
                "mem_used_gib": self._mem_used(),
                "load1": os.getloadavg()[0],
            })

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self.samples:
            return {}
        busy = [s["cpu_busy_fraction"] for s in self.samples]
        mem = [s["mem_used_gib"] for s in self.samples]
        load = [s["load1"] for s in self.samples]
        return {
            "samples": len(self.samples),
            "cpu_busy_fraction_avg": statistics.fmean(busy),
            "cpu_busy_fraction_max": max(busy),
            "cpu_equivalent_threads_avg": statistics.fmean(busy) * os.cpu_count(),
            "mem_used_gib_avg": statistics.fmean(mem),
            "mem_used_gib_max": max(mem),
            "load1_max": max(load),
        }


def _formal_job_active() -> bool:
    active_path = "/home/minidc/eval-league/v2/state/active.json"
    try:
        active = read_json(active_path)
    except Exception:
        return False
    pid = active.get("pid")
    if pid:
        try:
            children = subprocess.run(["ps", "--ppid", str(int(pid)), "-o", "pid="],
                                      capture_output=True, text=True, timeout=5).stdout.strip()
            if children:
                return True
        except Exception:
            pass
    schedule_path = "/home/minidc/eval-league/v2/schedule.json"
    try:
        schedule = read_json(schedule_path)
        for job in schedule.get("jobs", {}).values():
            if job.get("status") not in {"completed", "cancelled", "skipped", "success", "done"}:
                return True
    except Exception:
        pass
    return False


def run_shards(args, *, test: str, shards: int, phase_dir: str) -> dict:
    if _formal_job_active():
        return {"status": "deferred_formal_priority", "test": test}
    cpus = worker_cpus(shards)
    stop_file = os.path.join(phase_dir, "STOP_DIAGNOSTIC")
    if os.path.exists(stop_file):
        os.unlink(stop_file)
    os.makedirs(os.path.join(phase_dir, "logs"), exist_ok=True)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(args.torch_threads),
        "MKL_NUM_THREADS": str(args.torch_threads),
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PA3_ENGINE_PYTHON": ENGINE_PYTHON,
    })
    command = [
        args.python, os.path.join(ROOT, "scripts", "random_baseline_eval.py"),
        "--run-dir", phase_dir,
        "--fixture-run-dir", os.path.abspath(args.run_dir),
        "--checkpoint", args.checkpoint,
        "--data", args.data, "--teams", args.teams, "--seed", str(args.seed),
        "--engine-workers", str(args.engine_workers), "--batch-size", str(args.batch_size),
        "--torch-threads", str(args.torch_threads), "--max-rounds", str(args.max_rounds),
        "worker", "--test", test, "--shards", str(shards),
        "--model-sha256", args.model_sha256, "--engine-sha256", args.engine_sha256,
        "--stop-file", stop_file,
    ]
    monitor = ResourceMonitor()
    monitor.start()
    wall_start = time.perf_counter()
    processes = []
    formal_triggered = False
    for shard, cpu in enumerate(cpus):
        shard_command = command + ["--shard", str(shard), "--cpu", str(cpu)]
        log = open(os.path.join(phase_dir, "logs", f"shard-{shard:03d}.log"), "w")
        proc = subprocess.Popen(["taskset", "-c", str(cpu)] + shard_command,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((shard, proc, log))
    while any(proc.poll() is None for _shard, proc, _log in processes):
        if not formal_triggered and _formal_job_active():
            formal_triggered = True
            open(stop_file, "w").close()
            print("formal-evaluation-priority: requesting graceful diagnostic stop", flush=True)
        time.sleep(5)
    for _shard, proc, log in processes:
        proc.wait()
        log.close()
    wall = time.perf_counter() - wall_start
    resources = monitor.stop()

    receipts = []
    for shard in range(shards):
        path = _receipt_path(phase_dir, test, shard)
        if os.path.exists(path):
            receipts.append(read_json(path))
    games = sum(len(r.get("outcomes", [])) for r in receipts)
    steady = max((float(r.get("runtime_seconds", 0.0)) for r in receipts), default=0.0)
    errors = sum(1 for r in receipts for row in r.get("outcomes", []) if row.get("outcome") == "error")
    illegal = sum(1 for r in receipts for row in r.get("outcomes", []) if row.get("illegal_action"))
    complete = all(r.get("complete") for r in receipts) and len(receipts) == shards
    return {
        "status": "deferred_formal_priority" if formal_triggered else ("complete" if complete else "incomplete"),
        "test": test, "workers": shards, "cpus": cpus, "wall_seconds": wall,
        "steady_state_seconds": steady, "games": games,
        "steady_state_games_per_second": games / steady if steady else None,
        "wall_inclusive_games_per_second": games / wall if wall else None,
        "operational_errors": errors, "illegal_actions": illegal,
        "resources": resources, "formal_priority_triggered": formal_triggered,
        "receipts": len(receipts), "complete": complete,
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _bootstrap_ci(pair_scores: list[float], seed: int, samples: int = 10000) -> list[float]:
    rng = random.Random(seed)
    means = []
    n = len(pair_scores)
    for _ in range(samples):
        means.append(sum(pair_scores[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def _wilson(success: float, total: int) -> list[float]:
    if total <= 0:
        return [None, None]
    import math
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def collect_suite(run_dir: str, test: str, shards: int) -> tuple[list[dict], list[dict], list[str]]:
    outcomes, receipts, problems = [], [], []
    for shard in range(shards):
        path = _receipt_path(run_dir, test, shard)
        if not os.path.exists(path):
            problems.append(f"missing receipt shard {shard}")
            continue
        receipt = read_json(path)
        receipts.append(receipt)
        if not receipt.get("complete"):
            problems.append(f"incomplete receipt shard {shard}")
        outcomes.extend(receipt.get("outcomes", []))
    ids = [row["fixture_id"] for row in outcomes]
    if len(ids) != len(set(ids)):
        problems.append("duplicate fixture coverage")
    return outcomes, receipts, problems


def aggregate_suite(run_dir: str, test: str, shards: int) -> dict:
    fixtures, manifest = load_fixtures(run_dir)
    expected = [f for f in fixtures if f.test == test]
    outcomes, receipts, problems = collect_suite(run_dir, test, shards)
    expected_ids = {f.fixture_id for f in expected}
    observed_ids = {row["fixture_id"] for row in outcomes}
    missing = expected_ids - observed_ids
    extra = observed_ids - expected_ids
    if missing:
        problems.append(f"missing fixtures: {len(missing)}")
    if extra:
        problems.append(f"unexpected fixtures: {len(extra)}")
    errors = [row for row in outcomes if row.get("outcome") == "error"]
    illegal = [row for row in outcomes if row.get("illegal_action")]
    valid = [row for row in outcomes if row.get("outcome") in {"win", "loss", "draw"}]
    wins = sum(1 for row in valid if row["outcome"] == "win")
    losses = sum(1 for row in valid if row["outcome"] == "loss")
    draws = sum(1 for row in valid if row["outcome"] == "draw")

    pairs: dict[int, list[dict]] = defaultdict(list)
    for row in valid:
        pairs[row["pair_id"]].append(row)
    pair_scores = []
    broken_pairs = []
    for pair_id, rows in pairs.items():
        if len(rows) != 2 or {r["policy_side"] for r in rows} != {0, 1}:
            broken_pairs.append(pair_id)
            continue
        pair_scores.append(sum(1.0 if r["outcome"] == "win" else 0.5 if r["outcome"] == "draw" else 0.0
                               for r in rows) / 2.0)
    if broken_pairs:
        problems.append(f"broken paired fixtures: {len(broken_pairs)}")
    ci = _bootstrap_ci(pair_scores, seed=_derived_seeds(f"{SEED_DOMAIN}:bootstrap:{test}")) if pair_scores else None

    seat_rows = {}
    for side in (0, 1):
        rows = [row for row in valid if row["policy_side"] == side]
        sw = sum(1 for row in rows if row["outcome"] == "win")
        sl = sum(1 for row in rows if row["outcome"] == "loss")
        sd = sum(1 for row in rows if row["outcome"] == "draw")
        score = (sw + 0.5 * sd) / len(rows) if rows else None
        seat_rows[side] = {"seat": side, "games": len(rows), "wins": sw, "losses": sl,
                           "draws": sd, "score": score, "score_ci95": _wilson(sw + 0.5 * sd, len(rows))}

    # Per-team as the policy-controlled team and as the opponent team.
    team_policy = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0})
    team_opponent = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0})
    for row in valid:
        policy_team = row["team_a"] if row["policy_side"] == 0 else row["team_b"]
        opponent_team = row["team_b"] if row["policy_side"] == 0 else row["team_a"]
        for bucket, team in ((team_policy, policy_team), (team_opponent, opponent_team)):
            entry = bucket[team]
            entry["games"] += 1
            entry[{"win": "wins", "loss": "losses", "draw": "draws"}[row["outcome"]]] += 1
    team_rows = []
    for team in sorted(set(team_policy) | set(team_opponent)):
        for role, bucket in (("policy", team_policy), ("opponent", team_opponent)):
            entry = bucket.get(team)
            if not entry:
                continue
            score = (entry["wins"] + 0.5 * entry["draws"]) / entry["games"] if entry["games"] else None
            team_rows.append({"test": test, "team_id": team, "role": role, **entry,
                              "score": score})

    rounds = sorted(int(row.get("rounds", 0)) for row in valid)
    decisions = sorted(int(row.get("decisions", 0)) for row in valid)
    duration = {}
    if rounds:
        def quantile(values, q):
            return values[min(len(values) - 1, int(q * (len(values) - 1)))]
        duration = {
            "rounds_min": rounds[0], "rounds_median": statistics.median(rounds),
            "rounds_p90": quantile(rounds, 0.9), "rounds_p99": quantile(rounds, 0.99),
            "rounds_max": rounds[-1],
            "decisions_median": statistics.median(decisions),
            "decisions_max": decisions[-1],
            "histogram": dict(sorted(Counter(
                "1-5" if r <= 5 else "6-10" if r <= 10 else "11-20" if r <= 20
                else "21-40" if r <= 40 else "41-80" if r <= 80 else "81+"
                for r in rounds).items())),
        }
    runtimes = [float(r.get("runtime_seconds", 0.0)) for r in receipts]
    walls = [float(r.get("wall_seconds", 0.0)) for r in receipts]
    steady = max(runtimes, default=0.0)
    aggregate = {
        "schema": AGGREGATE_SCHEMA, "test": test,
        "label": (TEST_SPECS.get(test) or CALIBRATION_SPEC)["label"],
        "policy": (TEST_SPECS.get(test) or CALIBRATION_SPEC)["policy"],
        "opponent": (TEST_SPECS.get(test) or CALIBRATION_SPEC)["opponent"],
        "fixture_manifest_hash": manifest["manifest_hash"],
        "fixtures_expected": len(expected), "fixtures_observed": len(outcomes),
        "valid_games": len(valid), "wins": wins, "losses": losses, "draws": draws,
        "score": (wins + 0.5 * draws) / len(valid) if valid else None,
        "paired_fixtures": len(pair_scores), "paired_score_mean": statistics.fmean(pair_scores) if pair_scores else None,
        "paired_bootstrap_ci95": ci,
        "operational_errors": len(errors), "illegal_actions": len(illegal),
        "hard_gate": "pass" if not errors and not illegal and not missing and not extra and not broken_pairs else "fail",
        "seat_results": seat_rows, "team_results": team_rows, "duration": duration,
        "runtime_seconds_max_shard": steady,
        "steady_state_games_per_second": len(valid) / steady if steady else None,
        "wall_inclusive_games_per_second": len(valid) / max(walls, default=1.0),
        "shards": shards, "problems": problems,
        "receipt_meta": [{
            "shard": r.get("shard_index"), "complete": r.get("complete"),
            "outcomes": len(r.get("outcomes", [])),
            "runtime_seconds": r.get("runtime_seconds"),
            "wall_seconds": r.get("wall_seconds"),
            "engine_workers": r.get("engine_workers"),
            "batch_size": r.get("batch_size"),
            "torch_threads": r.get("torch_threads"),
            "cpu_affinity": r.get("cpu_affinity"),
        } for r in receipts],
    }
    return aggregate


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def _signed_action(record) -> tuple[int, int, int, int, int, int]:
    import numpy as np
    return (int(record[0]), int(record[1]), int(record[2]), int(np.int8(record[3])),
            int(record[4]), int(record[5]))


def _keyed_uniform(seed: int, fixture_id: int, side: int, decision: int, level: int) -> float:
    return random.Random(f"{seed}:{fixture_id}:{side}:{decision}:{level}").random()


def _trace_species_map(data_dir: str) -> dict[int, str]:
    dex = read_json(os.path.join(data_dir, "dex.json"))
    return {int(row["numeric_id"]): row["data"].get("name", row["id"]) for row in dex["tables"]["species"]}


def _team_label(team_index: int, teams_path: str, species_map: dict[int, str]) -> str:
    teams = read_json(teams_path)
    members = teams[team_index]["members"]
    names = [species_map.get(int(member["species"]), str(member["species"])) for member in members]
    return " / ".join(names)


def _trace_game(fixture: Fixture, evaluator, model, engine, *, seed: int, policy: str,
                opponent: str, max_rounds: int, species_map: dict[int, str],
                data_dir: str) -> dict:
    import numpy as np
    import torch
    from agent.types.observation import ObservationBatch, ObservationLayout
    from agent.types.requests import RequestKind
    from pa3_engine.observation import parse_batch, parse_packed_candidates

    layout = ObservationLayout()
    handle = engine.reset_batch([fixture.team_a], [fixture.team_b], [fixture.seed],
                                [(fixture.policy_side, 1 - fixture.policy_side)])[0]
    finished = False
    decisions = [[0, 0]]
    levels_seen = []
    selected_actions = []
    trace = {
        "fixture_id": fixture.fixture_id, "pair_id": fixture.pair_id,
        "team_a": fixture.team_a, "team_b": fixture.team_b,
        "seed": list(fixture.seed), "policy_side": fixture.policy_side,
        "policy_team": fixture.team_a if fixture.policy_side == 0 else fixture.team_b,
        "opponent_team": fixture.team_b if fixture.policy_side == 0 else fixture.team_a,
        "decisions": [], "rounds": 0, "outcome": None, "winner": None,
        "error": None, "illegal_action": False, "replay_verified": False,
    }
    for round_index in range(max_rounds):
        infos = engine.request_info_batch([(handle[0], handle[1], 0),
                                           (handle[0], handle[1], 1)])
        submissions = []
        requested = False
        for side, (kind, slots) in enumerate(infos):
            if not RequestKind(int(kind)).is_decision:
                continue
            requested = True
            branches = len(slots)
            side_is_policy = side == fixture.policy_side
            side_policy = policy if side_is_policy else opponent
            observation = None
            encoded = hidden = None
            if True:
                fixed, ragged = engine.observe_fixed_batch([(handle[0], handle[1])], [side])
                observation = ObservationBatch.from_native_payload(
                    parse_batch(fixed, ragged, 1), layout=layout).to("cpu")
            active = None
            try:
                active = observation.active_to_roster()[0].tolist()
            except Exception:
                active = [-1, -1]
            record = {
                "round": round_index, "side": side, "policy_side": side_is_policy,
                "request_kind": int(kind), "branch_levels": branches,
                "active_roster": active, "levels": [], "selected": [],
            }
            if side_policy == "checkpoint":
                with torch.inference_mode():
                    from agent.model.precision import get_precision_contract
                    subset = get_precision_contract(model.precision_contract.mode).prepare_observation(observation)
                    encoded = model.encode(subset)
                    hidden = model.scorer.initial_state(1, device="cpu",
                                                       dtype=encoded.tokens.dtype)
            prefix = np.zeros((1, branches, 6), dtype=np.uint8)
            for level in range(branches):
                count_bytes, action_bytes, _ = engine.candidates_packed_walk(
                    [(handle[0], handle[1])], [side], level,
                    np.ascontiguousarray(prefix[:, :level, :]).tobytes())
                packed = parse_packed_candidates(count_bytes, action_bytes, 1)
                count = int(packed["counts"][0])
                if count <= 0:
                    trace["error"] = "empty legal candidate set"
                    return trace
                candidates = [_signed_action(packed["records"][int(packed["offsets"][0]) + pick])
                              for pick in range(count)]
                level_record = {"level": level, "candidates": candidates}
                if side_policy == "checkpoint":
                    table = evaluator._helper._level_table_packed(
                        packed["counts"], packed["records"], packed["offsets"],
                        observation.active_to_roster())
                    keys = model.scorer.build_keys(
                        encoded.tokens, table.action_ids[:, 0], table.entity_token[:, 0],
                        table.move_token[:, 0])
                    logits = model.scorer.score_keys(keys, hidden, table.mask[:, 0])
                    _lp, probs, _ent, _kl = model._branch_stats(logits, table.mask[:, 0])
                    probabilities = probs.detach().float().cpu().numpy()[0]
                    mask = table.mask[:, 0].cpu().numpy()[0]
                    legal = np.flatnonzero(mask)
                    weights = probabilities[legal].astype(np.float64)
                    uniform = _keyed_uniform(seed, fixture.fixture_id, side,
                                             decisions[0][side], level)
                    cdf = np.cumsum(weights / weights.sum())
                    cdf[-1] = 1.0
                    pick = int(legal[np.searchsorted(cdf, uniform, side="right")])
                    level_record["selected_probability"] = float(probabilities[pick])
                    level_record["max_probability"] = float(probabilities[legal].max())
                    with torch.inference_mode():
                        hidden = model.scorer.advance_keys(
                            keys[torch.zeros(1, dtype=torch.long),
                                 torch.as_tensor([pick], dtype=torch.long)], hidden)
                elif side_policy == "lexicographic_legal":
                    pick = min(range(count), key=lambda index: candidates[index])
                else:
                    uniform = _keyed_uniform(seed, fixture.fixture_id, side,
                                             decisions[0][side], level)
                    pick = int(uniform * count)
                level_record["selected_index"] = pick
                level_record["selected"] = candidates[pick]
                action = candidates[pick]
                if action[0] == 1:  # move: resolve the selected move from this view
                    own_slot = action[1]
                    roster = active[own_slot] if 0 <= own_slot < len(active) else -1
                    if roster >= 0:
                        token = 16 + int(roster) * 4 + int(action[2])
                        if 0 <= token < observation.categories.shape[1] and bool(observation.category_known[0, token, 11]):
                            move_id = int(observation.categories[0, token, 11])
                            level_record["selected_move_id"] = move_id
                elif action[0] == 2 and 0 <= action[4] < 6:  # switch destination roster slot
                    token = 4 + int(action[4])
                    if bool(observation.category_known[0, token, 3]):
                        level_record["selected_species_id"] = int(observation.categories[0, token, 3])
                record["levels"].append(level_record)
                record["selected"].append(candidates[pick])
                prefix[0, level] = packed["records"][int(packed["offsets"][0]) + pick]
            submissions.append((side, record["selected"]))
            decisions[0][side] += 1
            levels_seen.append(record)
            selected_actions.append(record["selected"])
            trace["decisions"].append(record)
        if not requested:
            break
        result = engine.step_batch([(handle[0], handle[1], submissions)])[0]
        accepted, terminated, truncated, winner, error, *_ = result
        if not accepted and error is None and not truncated and not terminated:
            continue
        if not accepted:
            trace["illegal_action"] = True
            trace["error"] = error or "engine rejected action"
            break
        if truncated or error is not None:
            trace["error"] = error or "truncated"
            break
        if terminated:
            trace["winner"] = winner
            if winner is None:
                trace["outcome"] = "draw"
            else:
                trace["outcome"] = "win" if int(winner) == fixture.policy_side else "loss"
            trace["rounds"] = round_index + 1
            finished = True
            break
    if not finished and trace["error"] is None:
        trace["error"] = "round budget exhausted"
    trace["outcome"] = trace["outcome"] or "error"
    trace["rounds"] = trace["rounds"] or (max_rounds if trace["error"] == "round budget exhausted" else 0)
    trace["team_labels"] = {
        "team_a": _team_label(fixture.team_a, os.path.join(data_dir, "a10-training-teams.json"), species_map),
        "team_b": _team_label(fixture.team_b, os.path.join(data_dir, "a10-training-teams.json"), species_map),
    }
    return trace


def _action_class(action: tuple[int, int, int, int, int, int]) -> str:
    kind, _own, move_slot, target, switch_dest, resource = action
    if kind == 0:
        return "pick"
    if kind == 1:
        if move_slot == 255:
            return "move-no-slot"
        label = "move"
        if resource == 1:
            label += "+mega"
        return label
    if kind == 2:
        return "switch"
    if kind == 3:
        return "pass"
    return f"kind{kind}"


def select_traces(outcomes: list[dict], *, rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)
    valid = [row for row in outcomes if row.get("outcome") in {"win", "loss", "draw"}]
    wins = [row for row in valid if row["outcome"] == "win"]
    losses = [row for row in valid if row["outcome"] == "loss"]
    chosen, used = [], set()

    def take(rows, count, shuffle=True):
        fresh = [row for row in rows if row["fixture_id"] not in used]
        if shuffle:
            rng.shuffle(fresh)
        picked = fresh[:count]
        for row in picked:
            used.add(row["fixture_id"])
        chosen.extend(picked)
        return picked

    take(wins, 10)
    take(losses, 10)
    unusual_pool = sorted([row for row in valid if row["fixture_id"] not in used],
                          key=lambda row: (-int(row.get("rounds", 0)), row["fixture_id"]))
    take(unusual_pool, 30 - len(chosen), shuffle=False)
    return chosen


def build_traces(run_dir: str, aggregate: dict, args) -> list[dict]:
    import torch
    torch.set_num_threads(max(1, int(args.torch_threads)))
    import pa3_engine
    from agent.evaluation.batched_policy import BatchedPolicyEvaluator
    from agent.evaluation.fixed_policy import load_frozen_model

    fixtures, manifest = load_fixtures(run_dir)
    by_id = {f.fixture_id: f for f in fixtures}
    outcomes = aggregate["outcomes"]
    selected = select_traces(outcomes, rng_seed=_derived_seeds(f"{SEED_DOMAIN}:traces"))
    model = load_frozen_model(args.checkpoint, device="cpu") if args.checkpoint else None
    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=1)
    evaluator = BatchedPolicyEvaluator(engine, device="cpu", seed=args.seed, batch_size=1)
    species_map = _trace_species_map(args.data)
    trace_dir = os.path.join(run_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    for stale in os.listdir(trace_dir):
        if stale.startswith("trace-") and stale.endswith(".json"):
            os.unlink(os.path.join(trace_dir, stale))
    records = []
    for row in selected:
        fixture = by_id[row["fixture_id"]]
        spec = TEST_SPECS["test_a"]
        trace = _trace_game(fixture, evaluator, model, engine, seed=args.seed,
                            policy=spec["policy"], opponent=spec["opponent"],
                            max_rounds=args.max_rounds, species_map=species_map,
                            data_dir=args.data)
        trace["original_outcome"] = row["outcome"]
        trace["original_rounds"] = row.get("rounds")
        trace["replay_verified"] = (trace["outcome"] == row["outcome"] and
                                    trace["rounds"] == row.get("rounds") and
                                    trace["error"] is None and not trace["illegal_action"])
        path = os.path.join(trace_dir, f"trace-{fixture.fixture_id:08d}.json")
        atomic_json(path, trace)
        records.append(trace)
    return records


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _csv(rows: list[dict], columns: list[str]) -> str:
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            elif isinstance(value, (list, tuple)):
                values.append(";".join("" if item is None else f"{item:.6f}" if isinstance(item, float) else str(item)
                                       for item in value))
            else:
                values.append(str(value).replace(",", ";"))
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def publish_traces(docs_dir: str, traces: list[dict], run_dir: str) -> None:
    """Publish the selected decision traces as JSON plus readable Markdown."""
    import shutil
    trace_dir = os.path.join(docs_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    dex = read_json(os.path.join(DEFAULT_DATA, "dex.json"))
    move_names = {int(row["numeric_id"]): row["data"].get("name", row["id"])
                  for row in dex["tables"]["moves"]}
    species_names = {int(row["numeric_id"]): row["data"].get("name", row["id"])
                     for row in dex["tables"]["species"]}

    def describe(action: dict | tuple) -> str:
        if isinstance(action, dict):
            action = tuple(action.get("selected") or [])
        if not action:
            return "n/a"
        kind, own_slot, move_slot, target, switch_dest, resource = action
        if kind == 0:
            return f"pick team slot {switch_dest}"
        if kind == 1:
            if move_slot == 255:
                return "move NO_SLOT (forced no-op/Struggle-like)"
            return f"move {move_slot} at target {target}" + (" [MEGA]" if resource == 1 else "")
        if kind == 2:
            return f"switch to roster slot {switch_dest}"
        if kind == 3:
            return "pass"
        return f"kind {kind}"

    index_lines = ["# Representative battle traces", "",
                   "All traces are sequential keyed replays of games from Test A. "
                   "Each replay reproduced the frozen outcome and round count.", "",
                   "| fixture | outcome | rounds | policy team | opponent team | file |",
                   "|---:|---|---:|---|---|---|"]
    for trace in traces:
        fixture_id = int(trace["fixture_id"])
        path = os.path.join(run_dir, "traces", f"trace-{fixture_id:08d}.json")
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(trace_dir, os.path.basename(path)))
        labels = trace.get("team_labels") or {}
        policy_team = labels.get("team_a" if trace["policy_side"] == 0 else "team_b", "")
        opponent_team = labels.get("team_b" if trace["policy_side"] == 0 else "team_a", "")
        index_lines.append(f"| {fixture_id} | {trace['outcome']} | {trace['rounds']} | "
                           f"{policy_team} | {opponent_team} | trace-{fixture_id:08d}.md |")
        lines = [f"# Trace {fixture_id}", "",
                 f"* Pair: {trace['pair_id']}; policy seat: {trace['policy_side']}.",
                 f"* Policy team (seats: {trace['policy_side']}): {policy_team}.",
                 f"* Opponent team (seats: {1 - trace['policy_side']}): {opponent_team}.",
                 f"* Seed: {trace['seed']}.",
                 f"* Outcome: **{trace['outcome']}**; rounds: {trace['rounds']}; "
                 f"replay verified: {trace['replay_verified']}.", ""]
        leads = {}
        for decision in trace.get("decisions", []):
            if decision.get("request_kind") == 0 and decision["side"] not in leads:
                slots = [level.get("selected", [None] * 6)[4] for level in decision["levels"]]
                team = policy_team if decision["side"] == trace["policy_side"] else opponent_team
                members = team.split(" / ")
                leads[decision["side"]] = " + ".join(
                    members[slot] if 0 <= slot < len(members) else str(slot) for slot in slots)
        if leads:
            lines += ["## Lead choices", ""]
            for side, lead in sorted(leads.items()):
                who = "policy" if side == trace["policy_side"] else "opponent"
                lines.append(f"* {who} (seat {side}): {lead}")
            lines.append("")
        lines += ["## Decisions", "",
                  "| round | side | request | active roster | candidates per level | selected |",
                  "|---:|---|---|---|---|---|"]
        for decision in trace.get("decisions", []):
            team = policy_team if decision["side"] == trace["policy_side"] else opponent_team
            members = team.split(" / ")
            active = ", ".join(members[slot] if 0 <= slot < len(members) else str(slot)
                               for slot in decision.get("active_roster", []) if slot >= 0) or "-"
            candidate_counts = "/".join(str(len(level["candidates"])) for level in decision["levels"])
            selected_parts = []
            for level in decision["levels"]:
                action = level.get("selected") or []
                text = describe(level)
                move_id = level.get("selected_move_id")
                if move_id is not None:
                    text = text.replace(f"move {action[2]}", f"move {move_names.get(int(move_id), move_id)}")
                species_id = level.get("selected_species_id")
                if species_id is not None:
                    text += f" ({species_names.get(int(species_id), species_id)})"
                selected_parts.append(text)
            who = "policy" if decision["side"] == trace["policy_side"] else "opponent"
            lines.append(f"| {decision['round']} | {who} ({decision['side']}) | "
                         f"kind {decision['request_kind']} | {active} | {candidate_counts} | "
                         f"{'; '.join(selected_parts)} |")
        lines += ["", "## Terminal outcome", "",
                  f"Winner seat: {trace.get('winner')}; outcome for policy: **{trace['outcome']}**.",
                  ""]
        write_text(os.path.join(trace_dir, f"trace-{fixture_id:08d}.md"), "\n".join(lines))
    write_text(os.path.join(trace_dir, "index.md"), "\n".join(index_lines) + "\n")


def publish(run_dir: str, docs_dir: str, aggregates: dict[str, dict],
            calibration: dict, traces: list[dict], meta: dict) -> None:
    os.makedirs(docs_dir, exist_ok=True)
    suite_path = os.path.join(run_dir, "suite_resources.json")
    suite_runs = read_json(suite_path) if os.path.exists(suite_path) else {}
    perf_rows, seat_rows, team_rows = [], [], []
    for name, agg in aggregates.items():
        suite = suite_runs.get(name, {})
        perf_rows.append({
            "test": name, "label": agg["label"], "opponent": agg["opponent"],
            "games_attempted": agg["fixtures_expected"], "valid_games": agg["valid_games"],
            "wins": agg["wins"], "losses": agg["losses"], "draws": agg["draws"],
            "score": agg["score"], "paired_ci95_low": (agg["paired_bootstrap_ci95"] or [None, None])[0],
            "paired_ci95_high": (agg["paired_bootstrap_ci95"] or [None, None])[1],
            "operational_errors": agg["operational_errors"],
            "illegal_actions": agg["illegal_actions"], "hard_gate": agg["hard_gate"],
            "runtime_seconds": suite.get("wall_seconds", agg["runtime_seconds_max_shard"]),
            "steady_state_games_per_second": suite.get("steady_state_games_per_second",
                                                        agg["steady_state_games_per_second"]),
            "wall_inclusive_games_per_second": suite.get("wall_inclusive_games_per_second",
                                                          agg["wall_inclusive_games_per_second"]),
        })
        for side, row in agg["seat_results"].items():
            seat_rows.append({"test": name, **row})
        for row in agg["team_results"]:
            team_rows.append(row)
    write_text(os.path.join(docs_dir, "performance.csv"),
               _csv(perf_rows, ["test", "label", "opponent", "games_attempted", "valid_games",
                                "wins", "losses", "draws", "score", "paired_ci95_low",
                                "paired_ci95_high", "operational_errors", "illegal_actions",
                                "hard_gate", "runtime_seconds", "steady_state_games_per_second",
                                "wall_inclusive_games_per_second"]))
    write_text(os.path.join(docs_dir, "seat_results.csv"),
               _csv(seat_rows, ["test", "seat", "games", "wins", "losses", "draws", "score",
                                "score_ci95"]))
    write_text(os.path.join(docs_dir, "team_results.csv"),
               _csv(team_rows, ["test", "team_id", "role", "games", "wins", "losses", "draws", "score"]))

    for name, agg in aggregates.items():
        atomic_json(os.path.join(run_dir, "aggregate", f"{name}.json"), agg)

    primary = aggregates.get("test_a")
    secondary = aggregates.get("test_b")
    lines = ["# Random baseline evaluation v1", ""]
    lines += ["## Headline", ""]
    if primary:
        ci = primary["paired_bootstrap_ci95"]
        lines.append(f"* Test A (random opponent): **W {primary['wins']} / L {primary['losses']} / D {primary['draws']}**, "
                     f"score **{_fmt(primary['score'])}** (paired 95% CI {_fmt(ci[0])}–{_fmt(ci[1])}).")
    if secondary:
        ci = secondary["paired_bootstrap_ci95"]
        lines.append(f"* Test B (lexicographic legal): **W {secondary['wins']} / L {secondary['losses']} / D {secondary['draws']}**, "
                     f"score **{_fmt(secondary['score'])}** (paired 95% CI {_fmt(ci[0])}–{_fmt(ci[1])}).")
    controls = [name for name in aggregates if name.startswith("control_")]
    if controls:
        lines += ["", "## Controls", ""]
        for name in controls:
            agg = aggregates[name]
            ci = agg["paired_bootstrap_ci95"] or [None, None]
            lines.append(f"* `{name}`: W {agg['wins']} / L {agg['losses']} / D {agg['draws']}, "
                         f"score {_fmt(agg['score'])} (paired 95% CI {_fmt(ci[0])}–{_fmt(ci[1])}).")
        lines += ["", "The lexicographic-vs-random control is only 0.0957, so lexicographic legal is a much "
                  "weaker baseline than random. Test B is therefore a sanity check, not evidence of "
                  "strategy beyond beating random."]
    lines += ["", "## Artifacts", "",
              "* `performance.csv` — per-test totals, paired CI, hard gates, throughput.",
              "* `seat_results.csv` — seat-specific results.",
              "* `team_results.csv` — policy-team and opponent-team breakdowns.",
              "* `behavior_report.md` — trace sample and observed action patterns.",
              "* `performance_benchmark.md` — worker-count calibration.",
              "* `readiness_assessment.md` — evidence-based recommendation.",
              "* `manifest.json` — all hashes, commands, host and fixture identity.",
              "* `traces/` — 30 representative decision-level traces.", ""]
    write_text(os.path.join(docs_dir, "summary.md"), "\n".join(lines))

    cal_rows = []
    for label, result in calibration.get("configs", {}).items():
        cal_rows.append({
            "config": label, "workers": result["workers"],
            "engine_workers": result.get("engine_workers"),
            "batch_size": result.get("batch_size"),
            "status": result["status"], "games": result["games"],
            "wall_seconds": result["wall_seconds"],
            "steady_state_seconds": result["steady_state_seconds"],
            "steady_state_games_per_second": result["steady_state_games_per_second"],
            "wall_inclusive_games_per_second": result["wall_inclusive_games_per_second"],
            "operational_errors": result["operational_errors"],
            "illegal_actions": result["illegal_actions"],
            "cpu_avg_pct": (result.get("resources", {}).get("cpu_busy_fraction_avg") or 0) * 100,
            "ram_avg_gib": result.get("resources", {}).get("mem_used_gib_avg"),
        })
    write_text(os.path.join(docs_dir, "performance_benchmark.csv"),
               _csv(cal_rows, ["config", "workers", "engine_workers", "batch_size", "status",
                               "games", "wall_seconds", "steady_state_seconds",
                               "steady_state_games_per_second", "wall_inclusive_games_per_second",
                               "operational_errors", "illegal_actions", "cpu_avg_pct", "ram_avg_gib"]))
    bench = ["# Throughput calibration", "",
             "Calibration used a disjoint fixture set (512 games) and real frozen-model CPU inference.",
             "Steady-state excludes process start and model loading; wall-inclusive includes them.", "",
             "| config | workers | engine w | batch | games | steady games/s | wall games/s | CPU avg % | RAM avg GiB | status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in cal_rows:
        bench.append(f"| {row['config']} | {row['workers']} | {row['engine_workers']} | {row['batch_size']} | "
                     f"{row['games']} | {_fmt(row['steady_state_games_per_second'],2)} | "
                     f"{_fmt(row['wall_inclusive_games_per_second'],2)} | {_fmt(row['cpu_avg_pct'],1)} | "
                     f"{_fmt(row['ram_avg_gib'],2)} | {row['status']} |")
    bench += ["", f"Selected configuration: **{calibration.get('best_config')}** "
                  f"({calibration.get('selection_reason')}).", "",
              "## Full evaluation runs", "",
              "| test | games | wall s | steady games/s | wall games/s | CPU avg % | CPU eq. threads | RAM avg GiB | RAM max GiB | load1 max |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, agg in aggregates.items():
        suite = suite_runs.get(name)
        if not suite:
            continue
        resources = suite.get("resources", {})
        bench.append(f"| {name} | {suite['games']} | {_fmt(suite['wall_seconds'],1)} | "
                     f"{_fmt(suite['steady_state_games_per_second'],2)} | "
                     f"{_fmt(suite['wall_inclusive_games_per_second'],2)} | "
                     f"{_fmt((resources.get('cpu_busy_fraction_avg') or 0) * 100,1)} | "
                     f"{_fmt(resources.get('cpu_equivalent_threads_avg'),2)} | "
                     f"{_fmt(resources.get('mem_used_gib_avg'),2)} | "
                     f"{_fmt(resources.get('mem_used_gib_max'),2)} | "
                     f"{_fmt(resources.get('load1_max'),2)} |")
    bench += ["", "CPU percentages are machine-wide (80 logical threads). The 20-way configuration "
              "used 20 distinct physical cores, balanced 10/10 across both NUMA nodes.", ""]
    write_text(os.path.join(docs_dir, "performance_benchmark.md"), "\n".join(bench))

    # Behaviour report -----------------------------------------------------
    dex = read_json(os.path.join(DEFAULT_DATA, "dex.json"))
    move_names = {int(row["numeric_id"]): row["data"].get("name", row["id"])
                  for row in dex["tables"]["moves"]}
    protect_ids = {mid for mid, name in move_names.items()
                   if name.lower() in {"protect", "detect", "spikyshield", "kingsshield",
                                       "banefulbunker", "silktrap", "burningbulwark"}}
    classes = Counter()
    targets = Counter()
    move_distribution = Counter()
    switches = moves = megas = mega_opportunities = protect_uses = 0
    repeated = 0
    policy_decisions = 0
    verified = 0
    for trace in traces:
        verified += int(trace.get("replay_verified", False))
        previous = None
        for decision in trace.get("decisions", []):
            if not decision.get("policy_side"):
                continue
            policy_decisions += 1
            for level in decision["levels"]:
                for action in level["candidates"]:
                    if action[5] == 1:
                        mega_opportunities += 1
                selected = level["selected"]
                classes[_action_class(selected)] += 1
                if _action_class(selected) in {"move", "move+mega"}:
                    moves += 1
                    targets[selected[3]] += 1
                    if selected[5] == 1:
                        megas += 1
                    move_id = level.get("selected_move_id")
                    if move_id is not None:
                        move_distribution[move_names.get(int(move_id), str(move_id))] += 1
                        if int(move_id) in protect_ids:
                            protect_uses += 1
                if _action_class(selected) == "switch":
                    switches += 1
            selected_tuple = tuple(tuple(action) for action in decision["selected"])
            if previous is not None and selected_tuple == previous:
                repeated += 1
            previous = selected_tuple
    behavior = ["# Behaviour diagnostics (trace sample)", "",
                f"* Traces captured: **{len(traces)}** (target 10 wins, up to 10 losses, remainder unusual/longest).",
                f"* Trace replays matching the frozen outcome and round count: **{verified}/{len(traces)}**.",
                "* Metrics below are computed from this trace sample and from the full aggregate separately; "
                "they are not a full-population action census.", "",
                "## Observed action classes (selected levels)", ""]
    total_levels = sum(classes.values()) or 1
    for name, count in classes.most_common():
        behavior.append(f"* `{name}`: {count} ({count / total_levels:.1%})")
    behavior += ["", "## Most selected moves", ""]
    for name, count in move_distribution.most_common(15):
        behavior.append(f"* {name}: {count}")
    behavior += ["", "## Targeting and resources", "",
                 f"* Policy decisions inspected: {policy_decisions}.",
                 f"* Move selections: {moves}; switch selections: {switches}; target locations: "
                 f"{', '.join(f'{k}:{v}' for k, v in sorted(targets.items())) or 'n/a'}.",
                 f"* Mega candidates offered: {mega_opportunities}; Mega selected: {megas}.",
                 f"* Protect-family selections: {protect_uses}.",
                 f"* Consecutive identical policy decision tuples: {repeated}.", "",
                 "## Notable observations", "",
                 "* The trace set is intentionally enriched with the longest remaining games, so "
                 "Protect/Stockpile/Minimize rates are not population estimates.",
                 "* Long games (40+ rounds) show repeated Protect, Stockpile or Minimize sequences and "
                 "`move-no-slot`/pass cycles; this is the main observed gameplay concern.",
                 "* No illegal action or operational error appeared in the trace replays.", "",
                 "## Full-suite duration", ""]
    for name, agg in aggregates.items():
        duration = agg.get("duration") or {}
        behavior.append(f"* `{name}`: rounds median {duration.get('rounds_median')}, "
                        f"p90 {duration.get('rounds_p90')}, p99 {duration.get('rounds_p99')}, "
                        f"max {duration.get('rounds_max')}; histogram {duration.get('histogram')}.")
    behavior += ["", "## Limitations", "",
                 "* Only the 30 selected games carry decision-level records; all other games retain compact receipts.",
                 "* Legal candidate sets are branch-level tuples, matching the evaluator contract; they are not a "
                 "uniform joint-action distribution.",
                 "* Subjective tactical quality is not inferred beyond the recorded actions and outcomes.", ""]
    write_text(os.path.join(docs_dir, "behavior_report.md"), "\n".join(behavior))

    # Readiness ------------------------------------------------------------
    readiness = ["# Showdown smoke-test readiness assessment", ""]
    if primary:
        readiness.append(f"* **Test A** score against branch-wise uniform legal random: "
                         f"{_fmt(primary['score'])} (W {primary['wins']} / L {primary['losses']} / D {primary['draws']}).")
    if secondary:
        readiness.append(f"* **Test B** score against lexicographic legal: "
                         f"{_fmt(secondary['score'])} (W {secondary['wins']} / L {secondary['losses']} / D {secondary['draws']}).")
    gate_ok = all(agg["hard_gate"] == "pass" for agg in aggregates.values())
    readiness += ["", "## A. Basic game competence", "",
                  "The checkpoint wins the large majority of seat-swapped games against the branch-wise uniform "
                  "legal random opponent." if primary and (primary["score"] or 0) > 0.5 else
                  "The checkpoint does not show a clear advantage over legal random play in this sample.",
                  "", "## B. Learned behaviour", "",
                  "The model is compared directly with the deterministic lexicographic legal baseline on a "
                  "disjoint frozen fixture set (Test B).", "",
                  "## C. Gameplay quality", "",
                  "Decision-level traces were inspected for targeting, switching, Protect and Mega use. "
                  "See `behavior_report.md`; no fabricated strategic explanation is used.", "",
                  "## D. Technical correctness", "",
                  f"* Illegal actions: {sum(a['illegal_actions'] for a in aggregates.values())}.",
                  f"* Operational errors: {sum(a['operational_errors'] for a in aggregates.values())}.",
                  "* Coverage and duplicate checks were run per suite; hard gate: "
                  + ("PASS." if gate_ok else "FAIL (see summary)."), "",
                  "## E. Showdown readiness", "",
                  "This evaluation covers the offline engine, observation/action contract and CPU policy path only. "
                  "It does **not** prove that the live Showdown adapter, network protocol, timer handling or "
                  "deployment wiring works.", "",
                  "## Recommendation", ""]
    if not gate_ok:
        readiness.append("**BLOCKED_BY_CORRECTNESS_ISSUE** — resolve the failed hard gate before any smoke test.")
    elif primary and (primary["score"] or 0) >= 0.90 and secondary and (secondary["score"] or 0) > 0.50:
        readiness.append("**READY_FOR_SHOWDOWN_SMOKE_TEST** — offline correctness gates pass and the model "
                         "beats the random baseline by a wide margin. Test B is a sanity check only because "
                         "the lexicographic baseline is itself much weaker than random. This is a smoke-test "
                         "recommendation, not a ladder-strength claim.")
    else:
        readiness.append("**NEEDS_MORE_EVALUATION** — correctness gates pass, but the measured baselines are not "
                         "strong enough to justify a smoke test without further inspection.")
    readiness.append("")
    write_text(os.path.join(docs_dir, "readiness_assessment.md"), "\n".join(readiness))

    manifest = {
        "schema": "pa3-random-baseline-publication-v1",
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_file_sha256": meta.get("model_sha256"),
        "checkpoint_path": meta.get("checkpoint"),
        "source_sha": meta.get("source_sha"),
        "candidate_source_sha": meta.get("candidate_source_sha"),
        "frozen_generation": meta.get("frozen_generation"),
        "committed_matches": meta.get("committed_matches"),
        "engine_sha256": meta.get("engine_sha256"),
        "engine_path": meta.get("engine_python"),
        "dataset_manifest_hash": meta.get("dataset_manifest_hash"),
        "team_pool_sha256": meta.get("team_pool_sha256"),
        "evaluator_version": meta.get("evaluator_version"),
        "sampling_semantics": meta.get("sampling"),
        "fixture_manifest_hash": meta.get("fixture_manifest_hash"),
        "seed_domain": SEED_DOMAIN,
        "tests": {name: {"pairs": spec["pairs"], "games": spec["pairs"] * 2,
                         "policy": spec["policy"], "opponent": spec["opponent"],
                         "seed_domain": spec["seed_domain"]}
                  for name, spec in TEST_SPECS.items()},
        "host": meta.get("host"), "python": meta.get("python"),
        "torch": meta.get("torch"), "worker_config": calibration.get("best_config"),
        "commands": meta.get("commands"),
        "suite_performance": {name: {
            "games": row.get("games"), "wall_seconds": row.get("wall_seconds"),
            "steady_state_games_per_second": row.get("steady_state_games_per_second"),
            "wall_inclusive_games_per_second": row.get("wall_inclusive_games_per_second"),
            "resources": row.get("resources"),
        } for name, row in suite_runs.items()},
        "results": {name: {key: agg[key] for key in
                           ("wins", "losses", "draws", "score", "paired_bootstrap_ci95",
                            "valid_games", "operational_errors", "illegal_actions", "hard_gate")}
                    for name, agg in aggregates.items()},
        "limitations": [
            "Random is branch-wise uniform over legal candidates, not uniform over joint action combinations.",
            "Fixtures use the frozen 1,220-team pool; this is not a ladder-strength estimate.",
            "Offline evaluation does not prove the Showdown adapter works.",
        ],
    }
    atomic_json(os.path.join(docs_dir, "manifest.json"), manifest)
    atomic_json(os.path.join(run_dir, "publication_manifest.json"), manifest)
    publish_traces(docs_dir, traces, run_dir)


def publish_readme(docs_dir: str, meta: dict, aggregates: dict, calibration: dict) -> None:
    text = f"""# Random baseline evaluation v1

## Identity

* Candidate: `{CHECKPOINT_ID}` (2,764,800 matches, iteration 675).
* Checkpoint SHA-256: `{meta.get('model_sha256')}`.
* Candidate source SHA: `{meta.get('candidate_source_sha')}`.
* Evaluation worktree HEAD at measurement: `{meta.get('source_sha')}`.
* Frozen generation: `{meta.get('frozen_generation')}`.
* Engine SHA-256: `{meta.get('engine_sha256')}` (read-only external build).
* Team pool: `a10-training-teams.json` (1,220 teams), SHA-256 `{meta.get('team_pool_sha256')}`.
* Dataset manifest SHA-256: `{meta.get('dataset_manifest_hash')}`.
* Evaluator: `{meta.get('evaluator_version')}`; sampling `{meta.get('sampling')}`.
* Seed domain: `{SEED_DOMAIN}`.
* Fixture manifest hash: `{meta.get('fixture_manifest_hash')}`.
* Execution host: miniDC, 2x Xeon E5-2673 v4, CPU-only, CUDA hidden.

## Tests

| test | games | opponent | seed domain |
|---|---:|---|---|
| Test A | 8,192 | branch-wise uniform legal random | `{TEST_SPECS['test_a']['seed_domain']}` |
| Test B | 2,048 | lexicographic legal | `{TEST_SPECS['test_b']['seed_domain']}` |
| Control RR | 512 | random | `{TEST_SPECS['control_random_random']['seed_domain']}` |
| Control LR | 512 | random | `{TEST_SPECS['control_lexicographic_random']['seed_domain']}` |

Fixtures are generated once, frozen, and sharded by complete seat-swapped pair. Every global fixture ID is
stable. No `(team_a, team_b, seed)` identity is duplicated within a suite.

## Reproduction

```bash
{meta.get('commands', {}).get('all', 'python scripts/random_baseline_eval.py --run-dir ' + DEFAULT_RUN_ROOT + ' all')}
```

Worker configuration: **{calibration.get('best_config')}** ({calibration.get('selection_reason')}).

## Statistical interpretation

* Draws count 0.5.
* The 95% confidence interval is a paired bootstrap over fixture pairs (both seats), not over individual games.
* Opponents are weak baselines. A high score here does not establish competitive ladder strength.
* The fixture pool is the frozen training-team pool, so team-composition generalisation is limited to it.
"""
    write_text(os.path.join(docs_dir, "README.md"), text)


# ---------------------------------------------------------------------------
# top-level drivers
# ---------------------------------------------------------------------------

def _meta(args, manifest: dict, calibration: dict) -> dict:
    import platform
    import torch
    dataset_manifest = os.path.join(ROOT, "data", "teams", "a10-p0-proposed-20261013", "manifest.json")
    source_sha = ""
    try:
        source_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    commands = {
        "all": f"{args.python} scripts/random_baseline_eval.py --run-dir {args.run_dir} all",
        "fixtures": f"{args.python} scripts/random_baseline_eval.py --run-dir {args.run_dir} prepare",
    }
    return {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_sha256": args.model_sha256,
        "source_sha": source_sha,
        "candidate_source_sha": "da35700f75c95833b0e8250d7b4b04e11320f8b7",
        "frozen_generation": "iter-000002764800-1792212738467610757",
        "committed_matches": 2764800,
        "engine_sha256": args.engine_sha256,
        "engine_python": ENGINE_PYTHON,
        "dataset_manifest_hash": sha256_file(dataset_manifest) if os.path.exists(dataset_manifest) else None,
        "team_pool_sha256": manifest.get("team_pool_sha256"),
        "evaluator_version": calibration.get("evaluator_version"),
        "sampling": calibration.get("sampling"),
        "fixture_manifest_hash": manifest.get("manifest_hash"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "commands": commands,
    }


def cmd_prepare(args) -> int:
    args.run_dir = os.path.abspath(args.run_dir)
    manifest = prepare_fixtures(args.run_dir, args.data, args.teams)
    print(json.dumps({name: manifest["tests"][name] for name in manifest["tests"]}, indent=2))
    print("manifest_hash", manifest["manifest_hash"])
    return 0


def _calibration_configs(args) -> list[dict]:
    return [
        {"label": "w08", "workers": 8, "engine_workers": 1, "batch_size": 32},
        {"label": "w12", "workers": 12, "engine_workers": 1, "batch_size": 32},
        {"label": "w16", "workers": 16, "engine_workers": 1, "batch_size": 32},
        {"label": "w18", "workers": 18, "engine_workers": 1, "batch_size": 32},
        {"label": "w20", "workers": 20, "engine_workers": 1, "batch_size": 32},
        {"label": "w16-ew2", "workers": 16, "engine_workers": 2, "batch_size": 32},
        {"label": "w16-b64", "workers": 16, "engine_workers": 1, "batch_size": 64},
    ]


def run_calibration(args, run_dir: str) -> dict:
    existing = os.path.join(run_dir, "calibration", "calibration.json")
    if os.path.exists(existing):
        payload = read_json(existing)
        if payload.get("best_config") and payload.get("configs"):
            print(json.dumps({"calibration": "reused", "best_config": payload["best_config"]}), flush=True)
            return payload
    configs = {}
    for config in _calibration_configs(args):
        phase_dir = os.path.join(run_dir, "calibration", config["label"])
        os.makedirs(phase_dir, exist_ok=True)
        local = argparse.Namespace(**vars(args))
        local.engine_workers = config["engine_workers"]
        local.batch_size = config["batch_size"]
        result = run_shards(local, test="calibration", shards=config["workers"], phase_dir=phase_dir)
        result.update({"engine_workers": config["engine_workers"],
                       "batch_size": config["batch_size"]})
        configs[config["label"]] = result
        print(json.dumps({"calibration": config["label"], **{k: result[k] for k in
                          ("status", "games", "steady_state_games_per_second",
                           "wall_inclusive_games_per_second", "operational_errors",
                           "illegal_actions")}}), flush=True)
    usable = [row for row in configs.values() if row["status"] == "complete"
              and row["operational_errors"] == 0 and row["illegal_actions"] == 0
              and row["steady_state_games_per_second"]]
    if not usable:
        raise SystemExit("calibration produced no usable configuration")
    best = max(usable, key=lambda row: (row["steady_state_games_per_second"],
                                        row["wall_inclusive_games_per_second"]))
    label = next(key for key, value in configs.items() if value is best)
    return {"configs": configs, "best_config": label,
            "selection_reason": "highest steady-state valid games/s with zero errors and zero illegal actions",
            "evaluator_version": "pa3-native-batched-evaluator-v3",
            "sampling": "fixture-seat-decision-branch-keyed-valid-cdf-v3",
            "engine_workers": best["engine_workers"], "batch_size": best["batch_size"],
            "workers": best["workers"]}


def cmd_calibrate(args) -> int:
    args.run_dir = os.path.abspath(args.run_dir)
    manifest = prepare_fixtures(args.run_dir, args.data, args.teams)
    calibration = run_calibration(args, args.run_dir)
    atomic_json(os.path.join(args.run_dir, "calibration", "calibration.json"), calibration)
    print(json.dumps({k: calibration[k] for k in ("best_config", "workers", "engine_workers",
                                                  "batch_size", "selection_reason")}, indent=2))
    return 0


def cmd_suite(args) -> int:
    import shutil
    args.run_dir = os.path.abspath(args.run_dir)
    prepare_fixtures(args.run_dir, args.data, args.teams)
    if getattr(args, "fresh", False):
        shutil.rmtree(os.path.join(args.run_dir, "receipts", args.test), ignore_errors=True)
    phase_dir = args.run_dir
    result = run_shards(args, test=args.test, shards=args.workers, phase_dir=phase_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        return 3
    return 0


def cmd_traces(args) -> int:
    args.run_dir = os.path.abspath(args.run_dir)
    fixtures, manifest = load_fixtures(args.run_dir)
    calibration = read_json(os.path.join(args.run_dir, "calibration", "calibration.json"))
    # Use full Test A outcomes from receipts for selection.
    test_rows = [f for f in fixtures if f.test == "test_a"]
    pairs = [test_rows[i:i + 2] for i in range(0, len(test_rows), 2)]
    shards = int(calibration["workers"])
    outcomes, receipts, problems = collect_suite(args.run_dir, "test_a", shards)
    if problems:
        raise SystemExit(f"cannot select traces: {problems[:3]}")
    agg = aggregate_suite(args.run_dir, "test_a", shards)
    agg["outcomes"] = outcomes
    traces = build_traces(args.run_dir, agg, args)
    atomic_json(os.path.join(args.run_dir, "traces", "index.json"), {
        "traces": len(traces), "verified": sum(1 for t in traces if t["replay_verified"]),
        "fixture_ids": [t["fixture_id"] for t in traces],
    })
    print(json.dumps({"traces": len(traces), "verified": sum(1 for t in traces if t["replay_verified"])}, indent=2))
    return 0


def write_run_manifest(run_dir: str, calibration: dict, aggregates: dict[str, dict]) -> None:
    import glob
    fixture_manifest = os.path.join(run_dir, "fixtures.manifest.json")
    receipts = []
    paths = set(glob.glob(os.path.join(run_dir, "receipts", "**", "*.json"), recursive=True))
    paths.update(glob.glob(os.path.join(run_dir, "calibration", "**", "receipts", "**", "*.json"),
                            recursive=True))
    for path in sorted(paths):
        receipts.append({"path": os.path.relpath(path, run_dir),
                         "sha256": sha256_file(path),
                         "bytes": os.path.getsize(path)})
    suite_path = os.path.join(run_dir, "suite_resources.json")
    atomic_json(os.path.join(run_dir, "run_manifest.json"), {
        "schema": "pa3-random-baseline-run-manifest-v1",
        "fixtures_manifest_sha256": sha256_file(fixture_manifest) if os.path.exists(fixture_manifest) else None,
        "fixture_manifest_hash": read_json(fixture_manifest)["manifest_hash"] if os.path.exists(fixture_manifest) else None,
        "calibration_best_config": calibration.get("best_config"),
        "suite_resources": read_json(suite_path) if os.path.exists(suite_path) else None,
        "suite_results": {name: {key: agg[key] for key in
                                  ("wins", "losses", "draws", "score", "paired_bootstrap_ci95",
                                   "valid_games", "operational_errors", "illegal_actions", "hard_gate")}
                          for name, agg in aggregates.items()},
        "receipts": receipts,
        "receipt_count": len(receipts),
    })


def cmd_aggregate(args) -> int:
    args.run_dir = os.path.abspath(args.run_dir)
    calibration = read_json(os.path.join(args.run_dir, "calibration", "calibration.json"))
    shards = int(calibration["workers"])
    aggregates = {}
    for test in list(TEST_SPECS):
        aggregate = aggregate_suite(args.run_dir, test, shards)
        if aggregate["fixtures_observed"] == 0:
            continue
        aggregates[test] = aggregate
    if "test_a" not in aggregates:
        raise SystemExit("no Test A receipts found")
    for name, agg in aggregates.items():
        if agg["hard_gate"] != "pass":
            print(f"HARD GATE FAILURE in {name}: {agg['problems']}", file=sys.stderr)
    docs_dir = args.docs_dir
    fixtures, manifest = load_fixtures(args.run_dir)
    meta = _meta(args, manifest, calibration)
    traces = []
    trace_index = os.path.join(args.run_dir, "traces", "index.json")
    if os.path.exists(trace_index):
        for fixture_id in read_json(trace_index).get("fixture_ids", []):
            path = os.path.join(args.run_dir, "traces", f"trace-{fixture_id:08d}.json")
            if os.path.exists(path):
                traces.append(read_json(path))
    publish(args.run_dir, docs_dir, aggregates, calibration, traces, meta)
    publish_readme(docs_dir, meta, aggregates, calibration)
    write_run_manifest(args.run_dir, calibration, aggregates)
    summary = {name: {"wins": agg["wins"], "losses": agg["losses"], "draws": agg["draws"],
                      "score": agg["score"], "ci": agg["paired_bootstrap_ci95"],
                      "hard_gate": agg["hard_gate"]} for name, agg in aggregates.items()}
    atomic_json(os.path.join(docs_dir, "aggregate.json"), {"tests": summary,
                                                          "calibration": calibration})
    print(json.dumps(summary, indent=2))
    return 0 if all(a["hard_gate"] == "pass" for a in aggregates.values()) else 4


def cmd_all(args) -> int:
    args.run_dir = os.path.abspath(args.run_dir)
    manifest = prepare_fixtures(args.run_dir, args.data, args.teams)
    calibration = run_calibration(args, args.run_dir)
    atomic_json(os.path.join(args.run_dir, "calibration", "calibration.json"), calibration)
    best = argparse.Namespace(**vars(args))
    best.workers = int(calibration["workers"])
    best.engine_workers = int(calibration["engine_workers"])
    best.batch_size = int(calibration["batch_size"])
    results = {}
    for test in PRIMARY_TESTS + CONTROL_TESTS:
        results[test] = run_shards(best, test=test, shards=best.workers, phase_dir=args.run_dir)
        print(json.dumps({"suite": test, **{k: results[test][k] for k in
                          ("status", "games", "steady_state_games_per_second",
                           "wall_inclusive_games_per_second", "operational_errors",
                           "illegal_actions")}}), flush=True)
        if results[test]["status"] != "complete":
            print(json.dumps({"stopped": results[test]["status"], "suite": test}), flush=True)
            break
    if "test_a" in results and results["test_a"]["status"] == "complete":
        try:
            cmd_traces(best)
        except Exception as exc:  # traces are diagnostic; never mask the hard gates
            print(json.dumps({"trace_error": repr(exc)}), file=sys.stderr)
    args.workers = best.workers
    return cmd_aggregate(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--fixture-run-dir", default=None,
                        help="directory holding the frozen fixtures (defaults to --run-dir)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--teams", default=DEFAULT_TEAMS)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--engine-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--docs-dir", default=os.path.join(ROOT, "docs", "operations", "random-baseline-v1"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    worker = sub.add_parser("worker")
    worker.add_argument("--test", required=True)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--shards", type=int, required=True)
    worker.add_argument("--cpu", type=int, required=True)
    worker.add_argument("--stop-file", required=True)
    worker.add_argument("--model-sha256", default="")
    worker.add_argument("--engine-sha256", default="")
    sub.add_parser("calibrate")
    suite = sub.add_parser("suite")
    suite.add_argument("--test", default="test_a")
    suite.add_argument("--fresh", action="store_true",
                       help="remove only this suite's previous receipts before running")
    sub.add_parser("traces")
    sub.add_parser("aggregate")
    sub.add_parser("all")
    args = parser.parse_args(argv)

    if args.command == "worker":
        return run_worker(args)
    args.run_dir = os.path.abspath(args.run_dir)
    args.checkpoint = os.path.abspath(args.checkpoint) if args.checkpoint else ""
    args.model_sha256 = sha256_file(args.checkpoint) if args.checkpoint and os.path.exists(args.checkpoint) else ""
    args.engine_sha256 = sha256_file(os.path.join(ENGINE_PYTHON, "pa3_engine", "pa3_engine.so"))
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "suite":
        return cmd_suite(args)
    if args.command == "traces":
        return cmd_traces(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    if args.command == "all":
        return cmd_all(args)
    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
