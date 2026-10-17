"""Durable, idempotent scheduling for recurring historical diagnostics.

The scheduler is deliberately independent of the formal promotion league.  It
reads the verified registry snapshot, immutable pair summaries and the formal
schedule, then plans two diagnostic tiers:

* Tier A - every 200K committed matches, the newest verified milestone
  checkpoint versus up to eight historical anchors at 256 games each.
* Tier B - every 1M committed matches starting at 3.8M, an expanded round
  robin over at most 24 distinct model digests.

Milestones are keyed by ``tier:nominal_clock`` and associated with the actual
verified checkpoint that first reached them.  Overshoot is therefore harmless:
2,802,944 satisfies 2,800,000.  Planning is resumable and never schedules
ahead of pending formal evaluation work.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent.evaluation.historical_league import (
    ANCHOR_GAMES,
    DEFAULT_FORMAL_ROOT,
    DEFAULT_RR_ROOT,
    MAX_ACTIVE_ANCHORS,
    RR_PANEL_VERSION,
    RR_SEED,
    RR_SEED_DOMAIN,
    _former_champions,
    canonical_json,
    clock,
    find_cycles,
    formal_backlog,
    load_pair_summaries,
    load_snapshot,
    pair_arithmetic,
    plan_incremental_anchors,
    sha256_json,
)
from agent.evaluation.v2_league import atomic_json, read_json

SCHEMA = "pa3-v2-history-diagnostic-schedule-v1"
STATE_SCHEMA = "pa3-v2-history-diagnostic-state-v1"
JOB_SCHEMA = "pa3-v2-history-diagnostic-job-v1"
TIER_A_INTERVAL = 200_000
TIER_A_FIRST = 2_800_000
TIER_B_FIRST = 3_800_000
TIER_B_INTERVAL = 1_000_000
TIER_B_POLICY_CAP = 24
TIER_A_GAMES = ANCHOR_GAMES
TIER_B_ANCHOR_GAMES = 256
FILL_FRACTIONS = (0.00, 0.02, 0.05, 0.10, 0.18, 0.30, 0.46, 0.66, 0.88)
RECENT_COUNT = 6
FORMER_CHAMPION_COUNT = 5
TEMPORAL_ANCHOR_COUNT = 9
INFORMATIVE_COUNT = 4
STABLE_ANCHOR_COUNT = 4


class CorruptState(ValueError):
    pass


def _now() -> float:
    return time.time()


def _kst() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def nominal_milestones(committed_matches: int, *, interval: int = TIER_A_INTERVAL,
                       first: int | None = None) -> list[int]:
    """All nominal boundaries at or below ``committed_matches``."""
    committed = int(committed_matches)
    interval = int(interval)
    start = int(first if first is not None else interval)
    if interval <= 0 or committed < start:
        return []
    return list(range(start, committed + 1, interval))


def tier_b_milestones(committed_matches: int, *, first: int = TIER_B_FIRST,
                      interval: int = TIER_B_INTERVAL) -> list[int]:
    return nominal_milestones(committed_matches, interval=interval, first=first)


def checkpoint_for_milestone(entries, milestone: int, *,
                             interval: int = TIER_A_INTERVAL) -> dict | None:
    """Associate a milestone with the first verified checkpoint that reached it.

    The first choice is the earliest checkpoint inside the nominal window
    ``[milestone, milestone + interval)``.  If no checkpoint exists in that
    window (downtime missed several boundaries), the earliest checkpoint after
    the milestone is used so that the milestone is still satisfied exactly once.
    """
    ordered = sorted(entries, key=lambda entry: (clock(entry), entry["id"]))
    milestone = int(milestone)
    for entry in ordered:
        if milestone <= clock(entry) < milestone + int(interval):
            return entry
    for entry in ordered:
        if clock(entry) >= milestone:
            return entry
    return None


def milestone_key(tier: str, milestone: int) -> str:
    return f"{str(tier).lower()}:{int(milestone)}"


def default_state() -> dict:
    return {
        "schema": STATE_SCHEMA,
        "updated_at": _now(),
        "updated_at_kst": _kst(),
        "config": {
            "tier_a_interval": TIER_A_INTERVAL,
            "tier_a_first": TIER_A_FIRST,
            "tier_a_games_per_anchor": TIER_A_GAMES,
            "tier_a_max_anchors": MAX_ACTIVE_ANCHORS,
            "tier_b_first": TIER_B_FIRST,
            "tier_b_interval": TIER_B_INTERVAL,
            "tier_b_policy_cap": TIER_B_POLICY_CAP,
            "panel_version": RR_PANEL_VERSION,
            "seed_domain": RR_SEED_DOMAIN,
        },
        "milestones": {},
        "stable_anchor_ids": [],
    }


def load_state(root=DEFAULT_RR_ROOT) -> dict:
    path = Path(root) / "diagnostic_state.json"
    if not path.is_file():
        return default_state()
    try:
        value = read_json(path)
    except (json.JSONDecodeError, OSError) as error:
        raise CorruptState(f"diagnostic state is corrupt: {path}: {error}") from error
    if not value or value.get("schema") != STATE_SCHEMA:
        raise CorruptState(f"diagnostic state has an unexpected schema: {path}")
    value.setdefault("milestones", {})
    value.setdefault("stable_anchor_ids", [])
    value.setdefault("config", default_state()["config"])
    return value


def save_state(root, state) -> dict:
    payload = dict(state)
    payload["schema"] = STATE_SCHEMA
    payload["updated_at"] = _now()
    payload["updated_at_kst"] = _kst()
    atomic_json(Path(root) / "diagnostic_state.json", payload)
    return payload


def _receipt_path(root, job_id: str) -> Path:
    return Path(root) / "receipts" / "milestones" / f"{job_id}.json"


def reconcile_receipts(root, state) -> list[dict]:
    """Mark planned jobs completed only from a matching, uncorrupted receipt."""
    changes = []
    for key, job in state.get("milestones", {}).items():
        if job.get("status") == "completed":
            continue
        path = _receipt_path(root, job.get("job_id", ""))
        if not path.is_file():
            continue
        try:
            receipt = read_json(path)
        except (json.JSONDecodeError, OSError):
            job["status"] = "corrupt_receipt"
            job["receipt_error"] = str(path)
            changes.append({"milestone": key, "status": "corrupt_receipt"})
            continue
        if (not receipt or receipt.get("schema") != JOB_SCHEMA
                or receipt.get("status") != "success"
                or receipt.get("milestone_key") != key
                or receipt.get("checkpoint_model_digest") != job.get("checkpoint_model_digest")
                or receipt.get("job_id") != job.get("job_id")):
            job["status"] = "corrupt_receipt"
            job["receipt_error"] = str(path)
            changes.append({"milestone": key, "status": "corrupt_receipt"})
            continue
        job["status"] = "completed"
        job["completed_at"] = receipt.get("completed_at", _now())
        job["receipt"] = str(path)
        job["games_completed"] = int(receipt.get("games", 0) or 0)
        changes.append({"milestone": key, "status": "completed"})
    return changes


def _entry_summary(summary):
    return {
        "pair_key": summary.get("pair_key"),
        "status": summary.get("status"),
        "games": int(summary.get("games", 0) or 0),
        "receipt_identity": summary.get("receipt_identity"),
        "fixture_manifest_hash": summary.get("fixture_manifest_hash"),
        "environment": summary.get("environment"),
        "panel_version": summary.get("panel_version"),
        "left_model_digest": summary.get("left_model_digest"),
        "right_model_digest": summary.get("right_model_digest"),
        "fixture_pairs": int(summary.get("fixture_pairs", 0) or 0),
    }


def pair_cache_compatible(summary, *, left_digest, right_digest, fixture_pairs,
                          environment, panel_version=RR_PANEL_VERSION) -> bool:
    """A cached pair is reusable only when every identity component matches."""
    if not summary or summary.get("status") != "success":
        return False
    if int(summary.get("fixture_pairs", -1)) != int(fixture_pairs):
        return False
    if summary.get("panel_version") != panel_version:
        return False
    if environment is not None and summary.get("environment") != environment:
        return False
    digests = {summary.get("left_model_digest"), summary.get("right_model_digest")}
    return digests == {left_digest, right_digest}


def _policy_scores(summaries) -> dict[str, float]:
    totals = {}
    for summary in summaries:
        if summary.get("status") != "success" or not summary.get("games"):
            continue
        games = int(summary["games"])
        for policy_id, score in (
                (summary.get("left_id"), summary.get("left_score")),
                (summary.get("right_id"), summary.get("right_score"))):
            if policy_id is None or score is None:
                continue
            row = totals.setdefault(policy_id, [0.0, 0])
            row[0] += float(score) * games
            row[1] += games
    return {
        policy_id: wins / games
        for policy_id, (wins, games) in totals.items() if games
    }


def select_tier_b_participants(entries, summaries, *, champion_id, latest_id,
                               promotions_dir=None, cap=TIER_B_POLICY_CAP,
                               stable_anchor_ids=()) -> dict:
    """Select at most ``cap`` distinct-digest participants for Tier B."""
    cap = max(2, int(cap))
    ordered = sorted(entries, key=lambda entry: (clock(entry), entry["id"]))
    by_id = {entry["id"]: entry for entry in ordered}
    if champion_id not in by_id or latest_id not in by_id:
        raise ValueError("champion and latest must be present in the snapshot")
    selected = []
    used_ids = set()
    used_digests = set()

    def add(entry, role, reason):
        if entry is None or len(selected) >= cap:
            return False
        if entry["id"] in used_ids or entry["model_digest"] in used_digests:
            return False
        selected.append({
            "policy_id": entry["id"],
            "model_digest": entry["model_digest"],
            "committed_matches": clock(entry),
            "role": role,
            "role_reason": reason,
        })
        used_ids.add(entry["id"])
        used_digests.add(entry["model_digest"])
        return True

    # Non-negotiable participants.
    add(by_id[latest_id], "latest_candidate", "latest verified evaluation candidate")
    add(by_id[champion_id], "official_champion", "current official champion")

    for policy_id in list(stable_anchor_ids)[:STABLE_ANCHOR_COUNT]:
        entry = by_id.get(policy_id)
        add(entry, "stable_anchor", "long-term anchor carried between evaluations")

    recent = list(reversed(ordered))
    added = 0
    for entry in recent:
        if added >= RECENT_COUNT:
            break
        if add(entry, "recent_snapshot", "six most recent distinct digests"):
            added += 1

    former = [
        row for row in _former_champions(promotions_dir)
        if row.get("policy_id") in by_id
    ]
    added = 0
    for row in former:
        if added >= FORMER_CHAMPION_COUNT:
            break
        if add(by_id[row["policy_id"]], "historical_champion", "promotion history"):
            added += 1

    scores = _policy_scores(summaries)
    candidates = [entry for entry in ordered if entry["id"] not in used_ids]
    informative = []
    weak = sorted(
        (entry for entry in candidates if entry["id"] in scores),
        key=lambda entry: (scores[entry["id"]], entry["id"]))
    informative.extend((entry, "weakness", "lowest verified score") for entry in weak[:2])
    close = sorted(
        (entry for entry in candidates if entry["id"] in scores),
        key=lambda entry: (abs(scores[entry["id"]] - 0.5), entry["id"]))
    informative.extend(
        (entry, "informative", "score closest to 0.50") for entry in close[:2])
    cycle_ids = set()
    if summaries:
        for cycle in find_cycles(ordered, summaries):
            cycle_ids.update(cycle["policies"])
    for policy_id in sorted(cycle_ids):
        entry = by_id.get(policy_id)
        if entry is not None:
            informative.append((entry, "non_transitive", "observed majority cycle"))
    added = 0
    for entry, role, reason in informative:
        if added >= INFORMATIVE_COUNT:
            break
        if add(entry, role, reason):
            added += 1

    # Temporally distributed anchors: dense near the newest checkpoint and
    # progressively wider spacing for older snapshots.
    remaining = [entry for entry in ordered if entry["id"] not in used_ids]
    if remaining:
        newest = clock(ordered[-1])
        oldest = clock(ordered[0])
        span = max(1, newest - oldest)
        added = 0
        for fraction in FILL_FRACTIONS:
            if added >= TEMPORAL_ANCHOR_COUNT:
                break
            target = newest - fraction * span
            entry = min(
                (row for row in remaining if row["id"] not in used_ids),
                key=lambda row: (abs(clock(row) - target), clock(row), row["id"]),
                default=None)
            if add(entry, "temporal_anchor",
                   f"temporal target {int(target):,} ({fraction:.2f} of span)"):
                added += 1

    # Fill any unused slots with temporally representative policies.
    for entry in sorted(
            (row for row in ordered if row["id"] not in used_ids),
            key=lambda row: (abs(clock(row) - (clock(ordered[-1]) + clock(ordered[0])) / 2),
                             row["id"])):
        if len(selected) >= cap:
            break
        add(entry, "coverage_fill", "fill remaining participant slots")

    selected.sort(key=lambda row: (row["committed_matches"], row["policy_id"]))
    stable = [
        row["policy_id"] for row in selected
        if row["role"] in {"stable_anchor", "temporal_anchor"}
    ][:STABLE_ANCHOR_COUNT]
    digests = [row["model_digest"] for row in selected]
    return {
        "participants": selected,
        "stable_anchor_ids": stable,
        "deduplicated_by": "model_digest",
        "cap": cap,
        "selected": len(selected),
        "unique_digests": len(set(digests)),
        "always_included": {"latest": latest_id, "champion": champion_id},
    }


def _new_job(tier, milestone, entry, *, key, participants, games, root):
    job_id = sha256_json({
        "tier": tier,
        "milestone": int(milestone),
        "checkpoint_id": entry["id"],
        "checkpoint_model_digest": entry["model_digest"],
        "panel_version": RR_PANEL_VERSION,
        "seed_domain": RR_SEED_DOMAIN,
        "seed": RR_SEED,
        "participants": [row["model_digest"] for row in participants],
    })[:24]
    return {
        "schema": JOB_SCHEMA,
        "job_id": job_id,
        "milestone_key": key,
        "tier": tier,
        "milestone": int(milestone),
        "checkpoint_id": entry["id"],
        "checkpoint_clock": clock(entry),
        "checkpoint_model_digest": entry["model_digest"],
        "checkpoint_snapshot_sha256": entry.get("file_sha256"),
        "milestone_overshoot_matches": clock(entry) - int(milestone),
        "status": "planned",
        "created_at": _now(),
        "created_at_kst": _kst(),
        "attempts": 0,
        "planned_games": int(games),
        "participants": participants,
        "receipt_path": str(_receipt_path(root, job_id)),
        "activation": "DIAGNOSTIC_ONLY",
    }


def plan_milestones(entries, state, *, root=DEFAULT_RR_ROOT, summaries=None,
                    formal=None, promotions_dir=None, champion_id=None,
                    interval_a=TIER_A_INTERVAL, tier_a_first=TIER_A_FIRST,
                    tier_b_first=TIER_B_FIRST,
                    interval_b=TIER_B_INTERVAL, cap_b=TIER_B_POLICY_CAP,
                    games_a=TIER_A_GAMES, max_a=MAX_ACTIVE_ANCHORS,
                    reconcile=True, now=None) -> dict:
    """Plan every due milestone exactly once and persist resumable state.

    Formal evaluation always wins: if any formal job is pending or running no
    new diagnostic work is scheduled.  Completed milestones are never re-run.
    """
    root = Path(root)
    summaries = list(summaries if summaries is not None else load_pair_summaries(root))
    changes = reconcile_receipts(root, state) if reconcile else []
    formal = formal if formal is not None else formal_backlog()
    latest = max(entries, key=lambda entry: (clock(entry), entry["id"]))
    latest_clock = clock(latest)
    due_a_all = nominal_milestones(
        latest_clock, interval=interval_a, first=tier_a_first)
    due_b_all = tier_b_milestones(latest_clock, first=tier_b_first, interval=interval_b)
    new_jobs = []
    resumed_jobs = []
    if formal.get("pending_or_running"):
        payload = {
            "schema": SCHEMA,
            "state": "deferred_formal_priority",
            "action": "no_new_work",
            "reason": "formal evaluation has pending or running jobs",
            "formal_priority": formal,
            "due": [],
            "new_jobs": [],
            "resumed_jobs": [],
            "changes": changes,
        }
        atomic_json(root / "milestone_schedule.json", payload)
        save_state(root, state)
        return payload

    def ensure(tier, milestone, *, participants, games):
        key = milestone_key(tier, milestone)
        entry = checkpoint_for_milestone(
            entries, milestone,
            interval=interval_b if tier == "B" else interval_a)
        if entry is None:
            return None
        existing = state["milestones"].get(key)
        if existing:
            if existing.get("status") == "completed":
                return None
            if existing.get("status") == "corrupt_receipt":
                # Fail closed while the bad receipt exists.  If an operator has
                # reviewed and removed it, the job becomes resumable again.
                bad_path = _receipt_path(root, existing.get("job_id", ""))
                if bad_path.exists():
                    return None
                existing["status"] = "planned"
                existing["recovered_from"] = "corrupt_receipt"
                existing["recovered_at"] = _now()
            if existing.get("status") in {"planned", "running", "failed"}:
                resumed_jobs.append(dict(existing))
                return existing
        job = _new_job(
            tier, milestone, entry, key=key, participants=participants,
            games=games, root=root)
        state["milestones"][key] = job
        new_jobs.append(job)
        return job

    # A Tier B milestone that coincides with a 200K boundary supersedes the
    # lighter Tier A job so compatible work is not duplicated.
    tier_b_milestones_set = set(due_b_all)
    tier_a_superseded = {}
    for milestone in due_a_all:
        if milestone in tier_b_milestones_set:
            tier_a_superseded[milestone] = key = milestone_key("A", milestone)
            state["milestones"].setdefault(key, {
                "schema": JOB_SCHEMA,
                "milestone_key": key,
                "tier": "A",
                "milestone": int(milestone),
                "status": "covered_by_tier_b",
                "reason": "Tier B expanded round robin at the same nominal clock",
                "created_at": _now(),
                "activation": "DIAGNOSTIC_ONLY",
            })

    for milestone in due_b_all:
        key = milestone_key("B", milestone)
        if state["milestones"].get(key, {}).get("status") == "completed":
            continue
        entry = checkpoint_for_milestone(entries, milestone, interval=interval_b)
        if entry is None:
            continue
        selection = select_tier_b_participants(
            entries, summaries,
            champion_id=champion_id or state_champion_id(entries),
            latest_id=latest["id"], promotions_dir=promotions_dir, cap=cap_b,
            stable_anchor_ids=state.get("stable_anchor_ids", []))
        if selection["stable_anchor_ids"]:
            state["stable_anchor_ids"] = selection["stable_anchor_ids"]
        participants = selection["participants"]
        games = max(0, len(participants) - 1) * TIER_B_ANCHOR_GAMES
        ensure("B", milestone, participants=participants, games=games)

    for milestone in due_a_all:
        key = milestone_key("A", milestone)
        if milestone in tier_a_superseded:
            continue
        if state["milestones"].get(key, {}).get("status") == "completed":
            continue
        entry = checkpoint_for_milestone(entries, milestone, interval=interval_a)
        if entry is None:
            continue
        anchors = plan_incremental_anchors(
            entries, summaries, reference_id=entry["id"],
            promotions_dir=promotions_dir, max_pool=max_a)
        anchors = anchors[:max_a]
        participants = [{
            "policy_id": row["policy_id"],
            "model_digest": row["model_digest"],
            "committed_matches": row["committed_matches"],
            "role": row["role"],
            "role_reason": "Tier A historical anchor",
            "target_games": int(games_a),
        } for row in anchors]
        ensure("A", milestone, participants=participants,
               games=len(participants) * int(games_a))

    state = save_state(root, state)
    pending = [
        job for job in state["milestones"].values()
        if job.get("status") in {"planned", "running", "failed"}
    ]
    pending.sort(key=lambda job: (job.get("tier") == "B", job.get("milestone", 0)))
    payload = {
        "schema": SCHEMA,
        "state": "planned" if pending else "idle",
        "action": "plan_only",
        "latest_clock": latest_clock,
        "latest_policy_id": latest["id"],
        "tier_a_interval": int(interval_a),
        "tier_a_first": int(tier_a_first),
        "tier_b_first": int(tier_b_first),
        "tier_b_interval": int(interval_b),
        "tier_b_policy_cap": int(cap_b),
        "next_tier_a": next(
            (m for m in nominal_milestones(
                max(latest_clock + int(interval_a), int(tier_a_first)),
                interval=interval_a, first=tier_a_first)
             if milestone_key("A", m) not in state["milestones"]), None),
        "next_tier_b": next(
            (m for m in tier_b_milestones(
                max(latest_clock + int(interval_b), int(tier_b_first)),
                first=tier_b_first, interval=interval_b)
             if milestone_key("B", m) not in state["milestones"]), None),
        "due": pending,
        "new_jobs": new_jobs,
        "resumed_jobs": resumed_jobs,
        "changes": changes,
        "formal_priority": formal,
        "stable_anchor_ids": state.get("stable_anchor_ids", []),
        "executed": False,
        "execution_note": ("planning only; run diagnostics through the CPU-only "
                           "worker after the formal queue is empty"),
    }
    atomic_json(root / "milestone_schedule.json", payload)
    return payload


def state_champion_id(entries) -> str:
    """Fallback champion identity when no explicit champion is supplied."""
    return max(entries, key=lambda entry: (clock(entry), entry["id"]))["id"]


def schedule_for_snapshot(root=DEFAULT_RR_ROOT, formal_root=DEFAULT_FORMAL_ROOT, *,
                          promotions_dir=None, state=None, summaries=None,
                          reconcile=True) -> dict:
    entries, meta = load_snapshot(root)
    state = state if state is not None else load_state(root)
    summaries = summaries if summaries is not None else load_pair_summaries(root)
    if promotions_dir is None:
        promotions_dir = Path(formal_root) / "best_history" / "promotions"
    return plan_milestones(
        entries, state, root=root, summaries=summaries,
        promotions_dir=promotions_dir, champion_id=meta["champion_id"],
        reconcile=reconcile)


def summary_plan(payload) -> dict:
    return {
        "state": payload.get("state"),
        "action": payload.get("action"),
        "latest_clock": payload.get("latest_clock"),
        "new_jobs": len(payload.get("new_jobs", [])),
        "resumed_jobs": len(payload.get("resumed_jobs", [])),
        "due": [
            {
                "milestone_key": job.get("milestone_key"),
                "tier": job.get("tier"),
                "milestone": job.get("milestone"),
                "checkpoint_id": job.get("checkpoint_id"),
                "checkpoint_clock": job.get("checkpoint_clock"),
                "status": job.get("status"),
                "planned_games": job.get("planned_games"),
                "participants": len(job.get("participants", [])),
            }
            for job in payload.get("due", [])
        ],
        "next_tier_a": payload.get("next_tier_a"),
        "next_tier_b": payload.get("next_tier_b"),
    }
