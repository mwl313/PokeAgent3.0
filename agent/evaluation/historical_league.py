"""Isolated historical round-robin diagnostics and shadow-plan construction.

This module implements the diagnostic side of the Adaptive Historical League.
It owns only its own root under ``/home/minidc/eval-league/roundrobin/v1`` and
never writes to the formal V2 evaluation root, the promotion alpha ledger, the
best-policy pointer, the schedule, checkpoints or training counters.

The battle rules are not forked: every matchup is executed through the
existing ``agent.evaluation.v2_league.run_matchup_cached`` path.  Statistical
helpers collapse seat-swapped games into pair clusters before any uncertainty
statement and treat draws explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from itertools import combinations
from pathlib import Path

from agent.evaluation.historical_selector import Record, SELECTOR_SCHEMA, calculate
from agent.evaluation.identity import sha256_file
from agent.evaluation.v2_league import atomic_json, read_json

RR_SCHEMA = "pa3-v2-history-roundrobin-v1"
RR_PLAN_SCHEMA = "pa3-v2-history-pair-plan-v1"
RR_STATUS_SCHEMA = "pa3-v2-history-status-v1"
RR_PAIR_SCHEMA = "pa3-v2-history-pair-v1"
RR_SHADOW_SCHEMA = "pa3-v2-history-opponent-plan-v1"
RR_GATE_SCHEMA = "pa3-v2-history-gate-report-v1"
RR_INCREMENTAL_SCHEMA = "pa3-v2-history-incremental-plan-v1"
RR_PANEL_VERSION = "pa3-history-diagnostic-panel-v1"
RR_SEED_DOMAIN = "pa3-history-diagnostic-v1"
RR_SEED = 20261017
RR_MASTER_SEED = 20261014
BASIC_FIXTURE_PAIRS = 64
FINE_FIXTURE_PAIRS = 256
MAX_ACTIVE_ANCHORS = 8
ANCHOR_GAMES = 256
STALE_CLOCK_GAP_MATCHES = 400000
DEFAULT_RESERVE_GIB = 12.0
DEFAULT_FORMAL_ROOT = Path("/home/minidc/eval-league/v2")
DEFAULT_RR_ROOT = Path("/home/minidc/eval-league/roundrobin/v1")


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:16], 16)


def clock(entry) -> int:
    return int(entry.get("committed_matches", 0))


def canonical_pair(left_id: str, right_id: str) -> str:
    return "|".join(sorted((left_id, right_id)))


def pair_game_count(tier: str) -> int:
    return 2 * (FINE_FIXTURE_PAIRS if tier == "fine" else BASIC_FIXTURE_PAIRS)


def disk_free_bytes(path) -> int:
    usage = os.statvfs(str(path))
    return usage.f_bavail * usage.f_frsize


def formal_backlog(formal_root=DEFAULT_FORMAL_ROOT) -> dict:
    formal_root = Path(formal_root)
    schedule = read_json(formal_root / "schedule.json", {}) or {}
    jobs = schedule.get("jobs", {}) or {}
    pending = sorted(
        job_id for job_id, job in jobs.items()
        if job.get("status") in {"pending", "running"}
    )
    active = read_json(formal_root / "state" / "active.json", {}) or {}
    return {
        "pending_or_running": len(pending),
        "jobs": pending,
        "active_status": active.get("status"),
        "active_job": active.get("last_job") or active.get("job"),
        "updated_at": active.get("updated_at"),
    }


def _absolute_entry_path(registry_path, entry) -> dict:
    stored = dict(entry)
    path = Path(stored["path"])
    if not path.is_absolute():
        path = Path(registry_path).resolve().parent / path
    stored["path"] = str(path.resolve())
    return stored


def freeze_registry(formal_root=DEFAULT_FORMAL_ROOT, root=DEFAULT_RR_ROOT, *,
                    verify=True) -> tuple[list[dict], dict]:
    """Copy and hash-pin the formal registry into the diagnostic root.

    Entries with an identical model digest are collapsed, newest clock first.
    No file in the formal root is modified.
    """
    from agent.evaluation.policy_archive import verify_archive

    formal_root = Path(formal_root)
    root = Path(root)
    source = formal_root / "registry.json"
    if not source.is_file():
        raise ValueError(f"formal registry is missing: {source}")
    source_sha = sha256_file(source)
    registry = read_json(source)
    if verify:
        verify_archive(str(source))
    best = read_json(formal_root / "best_policy.json", {}) or {}
    champion_id = best.get("policy_id") or registry.get("incumbent")
    latest_id = registry.get("latest")
    policies = registry.get("policies", {})
    duplicates = []
    kept = {}
    for policy_id, entry in sorted(
            policies.items(), key=lambda item: (-clock(item[1]), item[0])):
        digest = entry.get("model_digest")
        if not digest:
            raise ValueError(f"policy has no model digest: {policy_id}")
        if digest in kept:
            duplicates.append({
                "model_digest": digest,
                "kept": kept[digest]["id"],
                "dropped": policy_id,
                "dropped_clock": clock(entry),
            })
            continue
        kept[digest] = _absolute_entry_path(source, dict(entry, id=policy_id))
    entries = sorted(kept.values(), key=lambda item: (clock(item), item["id"]))
    if champion_id not in {entry["id"] for entry in entries}:
        raise ValueError("champion is not present in the verified registry")
    if latest_id not in {entry["id"] for entry in entries}:
        raise ValueError("latest policy is not present in the verified registry")
    snapshot = {
        "schema": registry.get("schema"),
        "policies": {entry["id"]: entry for entry in entries},
        "evaluations": [],
        "latest": latest_id,
        "incumbent": champion_id,
        "rr_snapshot": {
            "schema": RR_SCHEMA,
            "source_registry": str(source),
            "source_registry_sha256": source_sha,
            "captured_at": time.time(),
            "panel_version": RR_PANEL_VERSION,
            "seed_domain": RR_SEED_DOMAIN,
        },
    }
    destination = root / "registry.snapshot.json"
    atomic_json(destination, snapshot)
    meta = {
        "schema": RR_SCHEMA,
        "source_registry": str(source),
        "source_registry_sha256": source_sha,
        "snapshot_path": str(destination),
        "snapshot_sha256": sha256_file(destination),
        "captured_at": time.time(),
        "champion_id": champion_id,
        "champion_model_digest": next(
            entry["model_digest"] for entry in entries if entry["id"] == champion_id),
        "latest_id": latest_id,
        "latest_model_digest": next(
            entry["model_digest"] for entry in entries if entry["id"] == latest_id),
        "policies_before_dedup": len(policies),
        "policies_after_dedup": len(entries),
        "duplicates": duplicates,
        "policies": [
            {
                "id": entry["id"],
                "committed_matches": clock(entry),
                "model_digest": entry["model_digest"],
                "file_sha256": entry["file_sha256"],
                "source_sha": entry.get("source_sha"),
                "precision_mode": entry.get("precision_mode"),
            }
            for entry in entries
        ],
        "panel_version": RR_PANEL_VERSION,
        "seed_domain": RR_SEED_DOMAIN,
    }
    atomic_json(root / "registry.meta.json", meta)
    return entries, meta


def load_snapshot(root=DEFAULT_RR_ROOT) -> tuple[list[dict], dict]:
    root = Path(root)
    registry = read_json(root / "registry.snapshot.json")
    meta = read_json(root / "registry.meta.json")
    if not registry or not meta:
        raise ValueError("round-robin registry snapshot is missing; run freeze first")
    entries = sorted(registry["policies"].values(), key=lambda item: (clock(item), item["id"]))
    return entries, meta


def adjacent_keys(entries) -> set[str]:
    ordered = sorted(entries, key=lambda item: (clock(item), item["id"]))
    return {
        canonical_pair(left["id"], right["id"])
        for left, right in zip(ordered, ordered[1:])
    }


def fine_pair_keys(entries, champion_id: str) -> set[str]:
    keys = adjacent_keys(entries)
    for entry in entries:
        if entry["id"] != champion_id:
            keys.add(canonical_pair(champion_id, entry["id"]))
    return keys


def pair_arithmetic(entries, champion_id: str) -> dict:
    n = len(entries)
    fine = fine_pair_keys(entries, champion_id)
    total_pairs = math.comb(n, 2)
    fine_pairs = len(fine)
    broad_pairs = total_pairs - fine_pairs
    return {
        "policies": n,
        "total_pairs": total_pairs,
        "fine_pairs": fine_pairs,
        "broad_pairs": broad_pairs,
        "basic_fixture_pairs": BASIC_FIXTURE_PAIRS,
        "fine_fixture_pairs": FINE_FIXTURE_PAIRS,
        "basic_games_per_pair": 2 * BASIC_FIXTURE_PAIRS,
        "fine_games_per_pair": 2 * FINE_FIXTURE_PAIRS,
        "total_games": (fine_pairs * 2 * FINE_FIXTURE_PAIRS
                        + broad_pairs * 2 * BASIC_FIXTURE_PAIRS),
    }


def build_pair_plan(entries, champion_id: str, latest_id: str | None = None) -> dict:
    ordered = sorted(entries, key=lambda item: (clock(item), item["id"]))
    by_id = {entry["id"]: entry for entry in ordered}
    if champion_id not in by_id:
        raise ValueError("champion is not in the snapshot")
    latest_id = latest_id or champion_id
    fine = fine_pair_keys(ordered, champion_id)
    rows = []
    champion_clock = clock(by_id[champion_id])
    latest_clock = clock(by_id[latest_id])
    for left, right in combinations(ordered, 2):
        key = canonical_pair(left["id"], right["id"])
        tier = "fine" if key in fine else "basic"
        ids = {left["id"], right["id"]}
        if ids == {champion_id, latest_id} and champion_id != latest_id:
            priority, reason = 0, "champion_vs_latest"
        elif champion_id in ids:
            priority, reason = 1, "champion_row"
        elif tier == "fine":
            priority, reason = 2, "adjacent_clock"
        else:
            priority, reason = 3, "archival_remainder"
        distance = min(abs(clock(left) - champion_clock), abs(clock(right) - champion_clock),
                       abs(clock(left) - latest_clock), abs(clock(right) - latest_clock))
        rows.append({
            "pair_key": key,
            "left_id": left["id"],
            "right_id": right["id"],
            "left_clock": clock(left),
            "right_clock": clock(right),
            "tier": tier,
            "fixture_pairs": FINE_FIXTURE_PAIRS if tier == "fine" else BASIC_FIXTURE_PAIRS,
            "games": pair_game_count(tier),
            "priority": priority,
            "reason": reason,
            "distance_to_anchor": distance,
        })
    rows.sort(key=lambda row: (row["priority"], row["distance_to_anchor"],
                               -max(row["left_clock"], row["right_clock"]), row["pair_key"]))
    return {
        "schema": RR_PLAN_SCHEMA,
        "panel_version": RR_PANEL_VERSION,
        "seed_domain": RR_SEED_DOMAIN,
        "champion_id": champion_id,
        "latest_id": latest_id,
        "arithmetic": pair_arithmetic(ordered, champion_id),
        "pairs": rows,
    }


def select_pairs(plan: dict, *, max_games: int | None = None,
                 fixture_pairs: int | None = None) -> list[dict]:
    selected, games = [], 0
    for pair in plan["pairs"]:
        row = dict(pair)
        if fixture_pairs is not None:
            row["fixture_pairs"] = int(fixture_pairs)
            row["games"] = 2 * int(fixture_pairs)
        if max_games is not None and games + row["games"] > int(max_games):
            continue
        selected.append(row)
        games += row["games"]
    return selected


def _fixture_slice(master, fixture_pairs: int):
    return master[:2 * int(fixture_pairs)]


def _ordered_pair(row, fixtures) -> tuple[str, str]:
    if clock({"committed_matches": row["left_clock"]}) > clock({"committed_matches": row["right_clock"]}):
        return row["left_id"], row["right_id"]
    if row["left_clock"] == row["right_clock"]:
        return tuple(sorted((row["left_id"], row["right_id"])))
    return row["right_id"], row["left_id"]


def pair_scores(outcomes) -> list[float]:
    grouped = {}
    for row in outcomes or []:
        if row.get("outcome") not in {"win", "loss", "draw"}:
            continue
        score = 1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "draw" else 0.0
        grouped.setdefault(row.get("pair_id"), []).append(score)
    return [
        sum(values) / len(values)
        for values in grouped.values()
        if len(values) == 2
    ]


def bootstrap_interval(scores, *, seed, draws=2000, low=0.025, high=0.975):
    if not scores:
        return [None, None]
    rng = random.Random(int(seed))
    means = sorted(
        sum(scores[rng.randrange(len(scores))] for _ in scores) / len(scores)
        for _ in range(draws)
    )
    return [means[int(low * draws)], means[min(draws - 1, int(high * draws))]]


def summarize_pair(receipt, row, *, left_entry, right_entry, seed) -> dict:
    scores = pair_scores(receipt.get("outcomes", []))
    expected_games = 2 * int(row["fixture_pairs"])
    status = "success" if (
        receipt.get("status") == "success"
        and int(receipt.get("matches", -1)) == expected_games
        and int(receipt.get("paired_fixtures", -1)) == int(row["fixture_pairs"])
    ) else "invalid"
    left_score = float(receipt.get("score_mean", 0.0) or 0.0)
    ci = bootstrap_interval(
        scores,
        seed=stable_seed(row["pair_key"], receipt.get("identity", ""), seed),
    )
    return {
        "schema": RR_PAIR_SCHEMA,
        "pair_key": row["pair_key"],
        "left_id": row["left_id"],
        "right_id": row["right_id"],
        "left_clock": clock(left_entry),
        "right_clock": clock(right_entry),
        "left_model_digest": left_entry["model_digest"],
        "right_model_digest": right_entry["model_digest"],
        "tier": row["tier"],
        "fixture_pairs": int(row["fixture_pairs"]),
        "games": expected_games,
        "matches": int(receipt.get("matches", 0)),
        "left_wins": int(receipt.get("wins", 0)),
        "left_losses": int(receipt.get("losses", 0)),
        "draws": int(receipt.get("draws", 0)),
        "left_score": left_score,
        "right_score": 1.0 - left_score if status == "success" else None,
        "paired_fixtures": int(receipt.get("paired_fixtures", 0)),
        "ci95_left": ci,
        "operational_errors": int(receipt.get("operational_errors", 0)),
        "illegal_actions": int(receipt.get("illegal_actions", 0)),
        "status": status,
        "receipt_identity": receipt.get("identity"),
        "fixture_manifest_hash": receipt.get("fixture_manifest_hash"),
        "fixture_seed": receipt.get("seed"),
        "panel_version": RR_PANEL_VERSION,
        "environment": receipt.get("environment"),
        "evaluated_at": receipt.get("evaluated_at"),
    }


def summary_score_for(summary, policy_id: str):
    if summary["status"] != "success":
        return None
    if summary["left_id"] == policy_id:
        return summary["left_score"]
    if summary["right_id"] == policy_id:
        return summary["right_score"]
    return None


def summary_wdl_for(summary, policy_id: str):
    if summary["left_id"] == policy_id:
        return summary["left_wins"], summary["draws"], summary["left_losses"]
    if summary["right_id"] == policy_id:
        return summary["left_losses"], summary["draws"], summary["left_wins"]
    return 0, 0, 0


def summary_ci_for(summary, policy_id: str):
    if summary["left_id"] == policy_id:
        return list(summary["ci95_left"])
    low, high = summary["ci95_left"]
    if low is None:
        return [None, None]
    return [1.0 - high, 1.0 - low]


def bradley_terry(summaries, entries, *, iterations: int = 200) -> dict[str, dict]:
    ids = [entry["id"] for entry in entries]
    results = {}
    for summary in summaries:
        if summary["status"] != "success":
            continue
        left, right = summary["left_id"], summary["right_id"]
        results[(left, right)] = (summary["left_score"], summary["games"])
        results[(right, left)] = (summary["right_score"], summary["games"])
    strength = {policy_id: 1.0 for policy_id in ids}
    for _ in range(int(iterations)):
        updated = {}
        for policy_id in ids:
            numerator = denominator = 0.0
            for opponent in ids:
                if opponent == policy_id:
                    continue
                row = results.get((policy_id, opponent))
                if not row:
                    continue
                score, games = row
                numerator += games * score
                denominator += games * (strength[policy_id]
                                        / (strength[policy_id] + strength[opponent]))
            updated[policy_id] = numerator / denominator if denominator > 0 else strength[policy_id]
        positive = [value for value in updated.values() if value > 0]
        if not positive:
            break
        geometric = math.exp(sum(math.log(value) for value in positive) / len(positive))
        strength = {policy_id: value / geometric for policy_id, value in updated.items()}
    ordered = sorted(ids, key=lambda policy_id: (-strength[policy_id], policy_id))
    return {
        policy_id: {
            "rating": math.log(strength[policy_id]) if strength[policy_id] > 0 else 0.0,
            "strength": strength[policy_id],
            "rank": index + 1,
            "games": sum(
                row[1] for (left, _right), row in results.items() if left == policy_id
            ),
        }
        for index, policy_id in enumerate(ordered)
    }


def majority_edges(summaries) -> dict[tuple[str, str], dict]:
    edges = {}
    for summary in summaries:
        if summary["status"] != "success":
            continue
        left, right = summary["left_id"], summary["right_id"]
        if summary["left_score"] > 0.5:
            edges[(left, right)] = {
                "score": summary["left_score"],
                "ci": summary["ci95_left"],
                "supported": summary["ci95_left"][0] is not None and summary["ci95_left"][0] > 0.5,
            }
        elif summary["left_score"] < 0.5:
            low, high = summary["ci95_left"]
            edges[(right, left)] = {
                "score": 1.0 - summary["left_score"],
                "ci": [1.0 - high, 1.0 - low] if low is not None else [None, None],
                "supported": low is not None and high < 0.5,
            }
    return edges


def find_cycles(entries, summaries) -> list[dict]:
    edges = majority_edges(summaries)
    ids = sorted(entry["id"] for entry in entries)
    found = set()
    rows = []
    for a, b, c in combinations(ids, 3):
        for cycle in ((a, b, c), (a, c, b)):
            x, y, z = cycle
            if ((x, y) in edges and (y, z) in edges and (z, x) in edges):
                key = tuple(sorted((x, y, z)))
                if key in found:
                    continue
                found.add(key)
                links = [edges[(x, y)], edges[(y, z)], edges[(z, x)]]
                rows.append({
                    "policies": [x, y, z],
                    "supported": all(link["supported"] for link in links),
                    "edges": [
                        {"from": x, "to": y, **edges[(x, y)]},
                        {"from": y, "to": z, **edges[(y, z)]},
                        {"from": z, "to": x, **edges[(z, x)]},
                    ],
                })
    return sorted(rows, key=lambda row: (not row["supported"], row["policies"]))


def champion_validation(champion_id, summaries, entries) -> dict:
    by_id = {entry["id"]: entry for entry in entries}
    rows = []
    for summary in summaries:
        if champion_id not in (summary["left_id"], summary["right_id"]):
            continue
        opponent_id = summary["right_id"] if summary["left_id"] == champion_id else summary["left_id"]
        wins, draws, losses = summary_wdl_for(summary, champion_id)
        rows.append({
            "opponent_id": opponent_id,
            "opponent_clock": clock(by_id[opponent_id]),
            "opponent_model_digest": by_id[opponent_id]["model_digest"],
            "games": summary["games"],
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": summary_score_for(summary, champion_id),
            "ci95": summary_ci_for(summary, champion_id),
            "status": summary["status"],
            "receipt_identity": summary["receipt_identity"],
            "fixture_manifest_hash": summary["fixture_manifest_hash"],
        })
    rows.sort(key=lambda row: row["opponent_clock"], reverse=True)
    valid = [row for row in rows if row["status"] == "success" and row["games"] > 0]
    return {
        "champion_id": champion_id,
        "champion_clock": clock(by_id[champion_id]),
        "champion_model_digest": by_id[champion_id]["model_digest"],
        "opponents": len(rows),
        "valid_opponents": len(valid),
        "total_games": sum(row["games"] for row in valid),
        "weighted_score": (
            sum(row["score"] * row["games"] for row in valid)
            / sum(row["games"] for row in valid)
        ) if valid else None,
        "weak_opponents": [row["opponent_id"] for row in valid if row["score"] < 0.45],
        "rows": rows,
    }


def _pair_files(root) -> list[Path]:
    return sorted((Path(root) / "receipts" / "pairs").glob("*.json"))


def load_pair_summaries(root=DEFAULT_RR_ROOT) -> list[dict]:
    rows = []
    for path in _pair_files(root):
        try:
            value = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if value and value.get("schema") == RR_PAIR_SCHEMA:
            rows.append(value)
    return rows


def write_status(root, status, name="status.json") -> dict:
    root = Path(root)
    payload = dict(status)
    payload["schema"] = RR_STATUS_SCHEMA
    payload["updated_at"] = time.time()
    payload["updated_at_kst"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(root / name, payload)
    return payload


def _entry_map(entries):
    return {entry["id"]: entry for entry in entries}


def write_matrix(root, summaries, entries) -> Path:
    by_id = _entry_map(entries)
    path = Path(root) / "matrix.csv"
    fieldnames = [
        "left_id", "right_id", "left_clock", "right_clock", "left_model_digest",
        "right_model_digest", "tier", "fixture_pairs", "games", "left_wins",
        "left_losses", "draws", "left_score", "right_score", "paired_fixtures",
        "ci95_low", "ci95_high", "status", "receipt_identity",
        "fixture_manifest_hash", "fixture_seed", "panel_version",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda row: (row["right_clock"], row["left_clock"])):
            low, high = summary["ci95_left"]
            writer.writerow({
                "left_id": summary["left_id"],
                "right_id": summary["right_id"],
                "left_clock": summary["left_clock"],
                "right_clock": summary["right_clock"],
                "left_model_digest": summary["left_model_digest"],
                "right_model_digest": summary["right_model_digest"],
                "tier": summary["tier"],
                "fixture_pairs": summary["fixture_pairs"],
                "games": summary["games"],
                "left_wins": summary["left_wins"],
                "left_losses": summary["left_losses"],
                "draws": summary["draws"],
                "left_score": summary["left_score"],
                "right_score": summary["right_score"],
                "paired_fixtures": summary["paired_fixtures"],
                "ci95_low": low,
                "ci95_high": high,
                "status": summary["status"],
                "receipt_identity": summary["receipt_identity"],
                "fixture_manifest_hash": summary["fixture_manifest_hash"],
                "fixture_seed": summary["fixture_seed"],
                "panel_version": summary["panel_version"],
            })
    return path


def write_ratings(root, summaries, entries) -> Path:
    ratings = bradley_terry(summaries, entries)
    by_id = _entry_map(entries)
    path = Path(root) / "ratings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "policy_id", "committed_matches", "model_digest",
                         "rating", "strength", "games"])
        for policy_id, row in sorted(ratings.items(), key=lambda item: item[1]["rank"]):
            writer.writerow([row["rank"], policy_id, clock(by_id[policy_id]),
                             by_id[policy_id]["model_digest"], row["rating"],
                             row["strength"], row["games"]])
    return path


def write_cycles(root, entries, summaries) -> Path:
    cycles = find_cycles(entries, summaries)
    path = Path(root) / "cycles.md"
    lines = [
        "# Historical round-robin non-transitivity report",
        "",
        f"Panel: `{RR_PANEL_VERSION}`  Seed domain: `{RR_SEED_DOMAIN}`",
        "",
        "A cycle is **suspected** when three majority edges form a loop. It is",
        "**statistically supported** only when every edge in the loop has a paired",
        "95% pair-cluster interval strictly above 0.5. Seat-swapped games are",
        "collapsed into pair clusters before interval construction.",
        "",
        f"Cycles detected: **{len(cycles)}**",
        "",
    ]
    if not cycles:
        lines.append("No three-policy majority cycles were observed in the completed pairs.")
    for index, cycle in enumerate(cycles, 1):
        label = "SUPPORTED" if cycle["supported"] else "SUSPECTED"
        lines.append(f"## {index}. {label}")
        lines.append("")
        lines.append("Policies: " + " -> ".join(cycle["policies"] + [cycle["policies"][0]]))
        for edge in cycle["edges"]:
            low, high = edge["ci"]
            lines.append(
                f"- `{edge['from']}` beats `{edge['to']}`: score {edge['score']:.4f}, "
                f"paired 95% CI [{low:.4f}, {high:.4f}], supported={edge['supported']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_champion_validation(root, champion_id, summaries, entries) -> Path:
    report = champion_validation(champion_id, summaries, entries)
    path = Path(root) / "champion_validation.md"
    lines = [
        "# Champion row validation (diagnostic only)",
        "",
        f"Champion: `{report['champion_id']}` at {report['champion_clock']:,} matches",
        f"Model digest: `{report['champion_model_digest']}`",
        "",
        "This report has **zero** authority over promotion, alpha spending or the",
        "formal champion pointer. It is an isolated diagnostic.",
        "",
        f"- Opponents with evidence: {report['valid_opponents']}/{report['opponents']}",
        f"- Total games: {report['total_games']}",
        f"- Game-weighted champion score: {report['weighted_score']:.6f}"
        if report["weighted_score"] is not None else "- Game-weighted champion score: n/a",
        f"- Point-score weak opponents (<0.45): {len(report['weak_opponents'])}",
        "",
        "| opponent | clock | games | W | D | L | score | paired 95% CI | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        low, high = row["ci95"]
        ci = "n/a" if low is None else f"[{low:.4f}, {high:.4f}]"
        lines.append(
            f"| {row['opponent_id']} | {row['opponent_clock']:,} | {row['games']} | "
            f"{row['wins']} | {row['draws']} | {row['losses']} | "
            f"{row['score']:.4f} | {ci} | {row['status']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_reports(root, entries, champion_id) -> dict:
    summaries = load_pair_summaries(root)
    matrix = write_matrix(root, summaries, entries)
    ratings = write_ratings(root, summaries, entries)
    cycles = write_cycles(root, entries, summaries)
    validation = write_champion_validation(root, champion_id, summaries, entries)
    return {
        "summaries": len(summaries),
        "matrix": str(matrix),
        "ratings": str(ratings),
        "cycles": str(cycles),
        "champion_validation": str(validation),
    }


def _former_champions(promotions_dir) -> list[dict]:
    promotions_dir = Path(promotions_dir) if promotions_dir else None
    if not promotions_dir or not promotions_dir.is_dir():
        return []
    rows = {}
    for path in sorted(promotions_dir.glob("*.json")):
        value = read_json(path)
        if not value or not value.get("policy_id"):
            continue
        policy_id = value["policy_id"]
        at = float(value.get("updated_at") or value.get("committed_matches") or 0)
        previous = rows.get(policy_id)
        if previous is None or at >= previous["at"]:
            rows[policy_id] = {
                "policy_id": policy_id,
                "committed_matches": int(value.get("committed_matches", 0)),
                "model_digest": value.get("model_digest"),
                "at": at,
            }
    return sorted(rows.values(), key=lambda row: (row["committed_matches"], row["policy_id"]),
                  reverse=True)


def _evidence_for(summary, policy_id, reference_entry, opponent_entry):
    if summary is None:
        return {
            "games": 0, "wins": 0, "draws": 0, "losses": 0, "score": None,
            "ci95": [None, None], "status": "unknown", "receipt_identity": None,
            "fixture_manifest_hash": None,
        }
    wins, draws, losses = summary_wdl_for(summary, policy_id)
    status = "verified" if summary["status"] == "success" else summary["status"]
    if status == "verified" and abs(clock(opponent_entry) - clock(reference_entry)) > STALE_CLOCK_GAP_MATCHES:
        status = "stale"
    return {
        "games": summary["games"],
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": summary_score_for(summary, policy_id),
        "ci95": summary_ci_for(summary, policy_id),
        "status": status,
        "receipt_identity": summary["receipt_identity"],
        "fixture_manifest_hash": summary["fixture_manifest_hash"],
    }


def select_shadow_pool(entries, summaries, *, reference_id, promotions_dir=None,
                       max_pool: int = MAX_ACTIVE_ANCHORS) -> list[dict]:
    by_id = _entry_map(entries)
    if reference_id not in by_id:
        raise ValueError("reference policy is not in the snapshot")
    reference = by_id[reference_id]
    summary_by_opponent = {}
    for summary in summaries:
        if summary["status"] != "success":
            continue
        if summary["left_id"] == reference_id:
            summary_by_opponent[summary["right_id"]] = summary
        elif summary["right_id"] == reference_id:
            summary_by_opponent[summary["left_id"]] = summary
    candidates = []
    for entry in entries:
        if entry["id"] == reference_id or entry["model_digest"] == reference["model_digest"]:
            continue
        evidence = _evidence_for(summary_by_opponent.get(entry["id"]), reference_id,
                                 reference, entry)
        candidates.append({
            "entry": entry,
            "evidence": evidence,
            "verified": evidence["status"] == "verified",
        })
    selected, roles = [], {}

    def add(candidate, role, reason):
        policy_id = candidate["entry"]["id"]
        if len(selected) >= max_pool or policy_id in {row["entry"]["id"] for row in selected}:
            return False
        selected.append({**candidate, "role": role, "role_reason": reason})
        roles.setdefault(role, 0)
        roles[role] += 1
        return True

    former = [
        row for row in _former_champions(promotions_dir)
        if row["policy_id"] in by_id and row["policy_id"] != reference_id
    ]
    for row in former[:3]:
        candidate = next(item for item in candidates if item["entry"]["id"] == row["policy_id"])
        add(candidate, "recent_champion", "promotion history")

    weak = sorted(
        [item for item in candidates if item["verified"]
         and item["evidence"]["score"] is not None and item["evidence"]["score"] < 0.45],
        key=lambda item: (item["evidence"]["score"], item["entry"]["id"]),
    )
    for item in weak[:2]:
        add(item, "verified_weakness", "verified point score below 0.45")

    selected_ids = {row["entry"]["id"] for row in selected}
    informative = sorted(
        [item for item in candidates if item["verified"] and item["entry"]["id"] not in selected_ids],
        key=lambda item: (abs(item["evidence"]["score"] - 0.5), item["entry"]["id"]),
    )
    for item in informative[:2]:
        add(item, "informative_nonchampion", "verified score closest to 0.50")

    selected_ids = {row["entry"]["id"] for row in selected}
    anchors = sorted(
        [item for item in candidates if item["entry"]["id"] not in selected_ids],
        key=lambda item: (clock(item["entry"]), item["entry"]["id"]),
    )
    if anchors:
        add(anchors[0], "distant_anchor", "oldest available distinct digest")

    selected_ids = {row["entry"]["id"] for row in selected}
    fill = sorted(
        [item for item in candidates if item["entry"]["id"] not in selected_ids],
        key=lambda item: (not item["verified"], -clock(item["entry"]), item["entry"]["id"]),
    )
    for item in fill:
        if len(selected) >= max_pool:
            break
        add(item, "coverage_fill", "fill remaining active-pool slots")
    return selected


def build_shadow_plan(entries, summaries, *, reference_id, root=DEFAULT_RR_ROOT,
                      promotions_dir=None, fixture_identity=None, seed=RR_SEED,
                      max_pool=MAX_ACTIVE_ANCHORS) -> dict:
    by_id = _entry_map(entries)
    reference = by_id[reference_id]
    pool = select_shadow_pool(entries, summaries, reference_id=reference_id,
                              promotions_dir=promotions_dir, max_pool=max_pool)
    records = [
        Record(
            policy_id=row["entry"]["id"],
            wins=row["evidence"]["wins"],
            draws=row["evidence"]["draws"],
            losses=row["evidence"]["losses"],
            verified=row["verified"],
        )
        for row in pool
    ]
    distribution = calculate(records, prior_games=128, max_weight=0.40)
    weight_by_id = {row["policy_id"]: row for row in distribution["opponents"]}
    opponents = []
    for row in sorted(pool, key=lambda item: item["entry"]["id"]):
        entry = row["entry"]
        evidence = row["evidence"]
        selected = weight_by_id[entry["id"]]
        opponents.append({
            "policy_id": f"md-{entry['model_digest']}",
            "source_policy_id": entry["id"],
            "model_digest": entry["model_digest"],
            "model_digest_algorithm": entry.get("model_digest_algorithm"),
            "snapshot_sha256": entry["file_sha256"],
            "snapshot_path": entry["path"],
            "committed_matches": clock(entry),
            "role": row["role"],
            "role_reason": row["role_reason"],
            "verified": bool(selected["verified"]),
            "evidence_status": evidence["status"],
            "games": evidence["games"],
            "wins": evidence["wins"],
            "draws": evidence["draws"],
            "losses": evidence["losses"],
            "pair_score": evidence["score"],
            "pair_ci95": evidence["ci95"],
            "win_rate_smoothed": selected["win_rate_smoothed"],
            "weight": selected["weight"],
            "receipt_identity": evidence["receipt_identity"],
            "fixture_manifest_hash": evidence["fixture_manifest_hash"],
        })
    for index, row in enumerate(opponents, 1):
        row["slot"] = index
    pool_hash = sha256_json([
        {
            "policy_id": row["policy_id"],
            "model_digest": row["model_digest"],
            "snapshot_sha256": row["snapshot_sha256"],
            "committed_matches": row["committed_matches"],
        }
        for row in opponents
    ])
    plan = {
        "schema": RR_SHADOW_SCHEMA,
        "status": "SHADOW_ONLY_NOT_ACTIVE",
        "created_at": time.time(),
        "shadow_only": True,
        "history_rows_target": 0,
        "history_probability": 0.0,
        "reference_policy": {
            "policy_id": reference["id"],
            "model_digest": reference["model_digest"],
            "model_digest_algorithm": reference.get("model_digest_algorithm"),
            "snapshot_sha256": reference["file_sha256"],
            "committed_matches": clock(reference),
        },
        "reference_policy_id": reference["id"],
        "evidence": {
            "panel_version": RR_PANEL_VERSION,
            "seed_domain": RR_SEED_DOMAIN,
            "seed": int(seed),
            "fixture_identity": fixture_identity or {},
            "pair_summaries": len(summaries),
            "verified_opponents": sum(1 for row in opponents if row["verified"]),
            "stale_or_unknown_opponents": sum(
                1 for row in opponents if not row["verified"]),
        },
        "selector": {
            "schema": SELECTOR_SCHEMA,
            "components": distribution["components"],
            "prior_games": distribution["prior_games"],
            "max_weight": distribution["max_weight"],
            "sha256": distribution["sha256"],
        },
        "opponents": opponents,
        "pool_sha256": pool_hash,
        "selection_distribution_sha256": distribution["sha256"],
        "activation": "FORBIDDEN_WITHOUT_OWNER_APPROVAL_AND_REVIEWED_MIGRATION",
        "notes": [
            "Weights are a pilot hypothesis, not a literature-validated optimum.",
            "Active production history probability remains 0.0 during shadow work.",
            "Stale or unknown opponents receive exploration mass only.",
            "This plan cannot promote, spend alpha or mutate training state.",
        ],
    }
    plan["plan_sha256"] = sha256_json(plan)
    atomic_json(Path(root) / "opponent_plan.json", plan)
    return plan


def next_full_boundary(committed_matches: int, *, interval: int = 200000) -> int:
    committed = int(committed_matches)
    interval = int(interval)
    if committed % interval == 0:
        return committed
    return ((committed // interval) + 1) * interval


def plan_incremental_anchors(entries, summaries, *, reference_id, promotions_dir=None,
                             max_pool=MAX_ACTIVE_ANCHORS) -> list[dict]:
    pool = select_shadow_pool(entries, summaries, reference_id=reference_id,
                              promotions_dir=promotions_dir, max_pool=max_pool)
    by_id = _entry_map(entries)
    rows = []
    for item in pool:
        entry = item["entry"]
        evidence = item["evidence"]
        rows.append({
            "policy_id": entry["id"],
            "model_digest": entry["model_digest"],
            "committed_matches": clock(entry),
            "role": item["role"],
            "verified": item["verified"],
            "evidence_status": evidence["status"],
            "games": evidence["games"],
            "wins": evidence["wins"],
            "draws": evidence["draws"],
            "losses": evidence["losses"],
            "pair_score": evidence["score"],
            "pair_ci95": evidence["ci95"],
            "target_games": ANCHOR_GAMES,
            "receipt_identity": evidence["receipt_identity"],
        })
    return rows


def milestone_hook(entries, summaries, *, promotions_dir=None,
                   interval: int = 200000) -> dict:
    latest = max(entries, key=lambda entry: (clock(entry), entry["id"]))
    boundary = next_full_boundary(clock(latest), interval=interval)
    available = clock(latest) >= boundary
    planned = plan_incremental_anchors(
        entries, summaries, reference_id=latest["id"], promotions_dir=promotions_dir)
    payload = {
        "schema": RR_INCREMENTAL_SCHEMA,
        "available": available,
        "reason": "milestone_available" if available else "no_new_full_milestone",
        "latest_policy_id": latest["id"],
        "latest_clock": clock(latest),
        "next_full_boundary": boundary,
        "max_anchors": MAX_ACTIVE_ANCHORS,
        "games_per_anchor": ANCHOR_GAMES,
        "planned_games": sum(row["target_games"] for row in planned) if available else 0,
        "anchors": planned if available else [],
        "planned_anchors_if_triggered": planned,
        "executed": False,
        "activation": "NOT_TRIGGERED_WHILE_TRAINING_PAUSED",
    }
    return payload


def protected_state(root=DEFAULT_RR_ROOT, formal_root=DEFAULT_FORMAL_ROOT) -> dict:
    root = Path(root)
    formal_root = Path(formal_root)
    before_path = root / "protected_snapshot" / "key_hashes_before.txt"
    before = {}
    if before_path.is_file():
        for line in before_path.read_text().splitlines():
            digest, _, path = line.partition("  ")
            if digest and path:
                before[path] = digest
    current = {}
    for path in [
        Path("/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/configs/pa3_v2_production.yaml"),
        Path("/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/runs/production/mode_e_da35700_20261016/STOP"),
        Path("/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/runs/production/mode_e_da35700_20261016/rolling/generations/iter-000002764800-1792212738467610757/manifest.json"),
        formal_root / "alpha_budget.json",
        formal_root / "best_policy.json",
        formal_root / "schedule.json",
        formal_root / "registry.json",
    ]:
        current[str(path)] = sha256_file(path) if path.is_file() else None
    mismatches = [
        {"path": path, "before": before.get(path), "after": current[path]}
        for path in current
        if path in before and before[path] != current[path]
    ]
    schedule_path = str(formal_root / "schedule.json")
    schedule_mismatch = next(
        (row for row in mismatches if row["path"] == schedule_path), None)
    critical = [row for row in mismatches if row["path"] != schedule_path]
    schedule = formal_backlog(formal_root)
    return {
        "before_recorded": bool(before),
        "mismatches": mismatches,
        "critical_mismatches": critical,
        "unchanged": not critical,
        "schedule": {
            "path": schedule_path,
            "hash_changed": bool(schedule_mismatch),
            "formal_discovery_owner": "v2-eval-discover.service",
            "pending_or_running": schedule["pending_or_running"],
            "jobs": schedule["jobs"],
            "note": ("the formal discovery timer rewrites schedule.json even when it "
                     "registers no job; the historical worker only reads this file"),
        },
        "current": current,
    }


def write_gate_report(root, *, stage_a: dict, stage_b: dict, stage_c: dict,
                      stage_d: dict, formal_priority: dict, disk: dict,
                      protected: dict) -> dict:
    report = {
        "schema": RR_GATE_SCHEMA,
        "generated_at": time.time(),
        "generated_at_kst": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stages": {
            "A_diagnostics": stage_a,
            "B_incremental": stage_b,
            "C_shadow": stage_c,
            "D_training_migration": stage_d,
        },
        "formal_priority": formal_priority,
        "disk": disk,
        "protected_state": protected,
        "go_no_go": {
            "stage_d": stage_d.get("status", "NO_GO"),
            "reason": stage_d.get("reason"),
        },
    }
    atomic_json(Path(root) / "gate_report.json", report)
    return report
