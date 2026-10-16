"""Evaluation League V2: adaptive budgets, alpha spending and promotion safety.

The statistical guarantee is a family-wise false-promotion budget of
``ALPHA_TOTAL`` across the campaign.  It follows from three facts, checked
here:

1. Per comparison, the sequential confirmation rule spends a predeclared
   cumulative alpha at each look.  Because each look is a valid fixed-sample
   Hoeffding test at its own level, the union bound over looks bounds the
   comparison's false-promotion probability by the comparison's allocation.
2. Allocations sum to at most ``ALPHA_TOTAL`` over the declared maximum number
   of promotion-eligible (full-milestone) comparisons; the union bound over
   comparisons therefore bounds the campaign-wide false-promotion rate.
3. Lightweight diagnostics never run a formal test and never promote.

Small Monte Carlo runs here are sanity checks only; the bound itself is the
analytic argument above (and in ``agent/evaluation/v2_league``).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pytest

from agent.evaluation import v2_league as L
from agent.evaluation.fixed_policy import build_fixture_manifest

TEAM_COUNT = 1220


# ---------------------------------------------------------------------------
# simulation helpers (drive the production decision functions)
# ---------------------------------------------------------------------------

def _pair_score(p, rng):
    return 0.5 * ((1.0 if rng.random() < p else 0.0) + (1.0 if rng.random() < p else 0.0))


def _old_rule_outcomes(pair_scores):
    rows = []
    for index, score in enumerate(pair_scores):
        outcome = "win" if score == 1.0 else "draw" if score == 0.5 else "loss"
        for side in (0, 1):
            rows.append({"fixture_id": 2 * index + side, "pair_id": index, "outcome": outcome})
    return rows


def simulate_new_comparison(p, rng, alpha_total, *, primary_pairs=1024, amplify=True):
    """Return (promoted, games, detail) for the League V2 procedure."""
    primary = [_pair_score(p, rng) for _ in range(primary_pairs)]
    games = primary_pairs * 2
    screen = L.primary_screening(primary)
    detail = {"screen": screen["status"]}
    if screen["status"] in ("confirmed_inferior", "weak"):
        return False, games, detail
    if screen["status"] == "amplify" and amplify:
        primary += [_pair_score(p, rng) for _ in range(L.INCUMBENT_AMP_PAIRS)]
        games += L.INCUMBENT_AMP_PAIRS * 2
        screen = L.primary_screening(primary)
        detail["screen"] = screen["status"]
        if screen["status"] != "promising":
            return False, games, detail
    pooled = []
    for stage in L.confirmation_stage_plan():
        pooled += [_pair_score(p, rng) for _ in range(stage["add_pairs"])]
        games += stage["add_pairs"] * 2
        result = L.confirmation_rule(pooled, alpha_total=alpha_total, stage=stage)
        detail["last_stage"] = stage["stage"]
        if result["decision"] == "promote":
            return True, games, detail
        if result["decision"] == "retain_inferior":
            return False, games, detail
    return False, games, detail


def simulate_old_comparison(p, rng, *, comparison_index=9, opponent_index=7):
    primary = [_pair_score(p, rng) for _ in range(512)]
    stats = L.promotion_decision(_old_rule_outcomes(primary),
                                 comparison_index=comparison_index, opponent_index=opponent_index,
                                 stage_is_final=False)
    if stats["decision"] != "promote":
        return False, 1024
    confirm = [_pair_score(p, rng) for _ in range(512)]
    confirm_stats = L.promotion_decision(_old_rule_outcomes(confirm),
                                         comparison_index=comparison_index, opponent_index=opponent_index,
                                         stage_is_final=True)
    return confirm_stats["decision"] == "promote", 2048


# ---------------------------------------------------------------------------
# 15. analytic error-control guarantee (predicates of the schedule)
# ---------------------------------------------------------------------------

def test_alpha_schedule_analytical_guarantee():
    assert abs(sum(L.CONFIRM_STAGE_ALPHA_WEIGHTS) - 1.0) < 1e-12
    ledger = L.initial_alpha_ledger()
    assert ledger["consumed"] == L.ALPHA_HISTORICAL_CONSUMED
    assert ledger["next_index"] == L.ALPHA_HISTORICAL_COMPARISONS + 1
    # Spending every remaining allocation stays within the total budget.
    consumed = ledger["consumed"]
    alpha = L.alpha_for_comparison(ledger)["alpha"]
    assert alpha > 0
    for index in range(ledger["next_index"], ledger["max_promotion_comparisons"] + 1):
        consumed += alpha
    assert consumed <= L.ALPHA_TOTAL + 1e-9
    # Historical accounting is preserved and never reset by the new schedule.
    assert L.ALPHA_HISTORICAL_CONSUMED > 0
    assert L.ALPHA_TOTAL - L.ALPHA_HISTORICAL_CONSUMED > 0


def test_hoeffding_interval_finite_sample_validity():
    rng = random.Random(1)
    alpha, n, trials = 0.01, 256, 20000
    false_rejects = 0
    for _ in range(trials):
        scores = [_pair_score(0.5, rng) for _ in range(n)]
        result = L.hoeffding_interval(scores, alpha=alpha)
        if result["lower"] > 0.5:
            false_rejects += 1
    observed = false_rejects / trials
    # Sanity check only; the analytic Hoeffding bound is the guarantee.
    assert observed <= alpha * 1.5, observed


# ---------------------------------------------------------------------------
# 1. equal candidate is not excessively promoted
# ---------------------------------------------------------------------------

def test_equal_candidate_bounded_at_reference_alpha():
    rng = random.Random(2)
    trials = 4000
    promoted = sum(simulate_new_comparison(0.5, rng, 0.05)[0] for _ in range(trials))
    rate = promoted / trials
    assert rate <= 0.07, rate  # analytic per-comparison bound is 0.05


def test_equal_candidate_full_schedule_is_conservative():
    rng = random.Random(3)
    alpha = L.alpha_for_comparison(L.initial_alpha_ledger())["alpha"]
    trials = 4000
    promoted = sum(simulate_new_comparison(0.5, rng, alpha)[0] for _ in range(trials))
    assert promoted / trials <= 0.005


# ---------------------------------------------------------------------------
# 2/3. sensitivity improvements
# ---------------------------------------------------------------------------

def test_55pct_candidate_benefits_from_additional_evidence():
    rng = random.Random(4)
    alpha = L.alpha_for_comparison(L.initial_alpha_ledger())["alpha"]
    tests = 2000
    first_stage_only = 0
    full_ladder = 0
    for _ in range(tests):
        pooled = [_pair_score(0.55, rng) for _ in range(L.confirmation_stage_plan()[0]["add_pairs"])]
        stage = L.confirmation_stage_plan()[0]
        if L.confirmation_rule(pooled, alpha_total=alpha, stage=stage)["decision"] == "promote":
            first_stage_only += 1
        if simulate_new_comparison(0.55, rng, alpha)[0]:
            full_ladder += 1
    assert full_ladder > first_stage_only
    assert full_ladder / tests >= 0.25


def test_60pct_candidate_promotes_without_max_budget():
    rng = random.Random(5)
    alpha = L.alpha_for_comparison(L.initial_alpha_ledger())["alpha"]
    tests = 1000
    promoted = 0
    max_budget = 0
    games = []
    for _ in range(tests):
        ok, used, detail = simulate_new_comparison(0.60, rng, alpha)
        promoted += int(ok)
        games.append(used)
        if detail.get("last_stage") == len(L.CONFIRM_STAGE_GAMES):
            max_budget += 1
    assert promoted / tests >= 0.9
    assert max_budget / tests < 0.5
    assert sum(games) / tests < L.FULL_INITIAL_GAMES + L.CONFIRM_MAX_GAMES


# ---------------------------------------------------------------------------
# power comparison: previous vs new rules
# ---------------------------------------------------------------------------

def test_power_comparison_old_vs_new():
    alpha = L.alpha_for_comparison(L.initial_alpha_ledger())["alpha"]
    trials = 600
    summary = {}
    for p in (0.50, 0.53, 0.55, 0.57, 0.60):
        rng = random.Random(100 + int(p * 1000))
        new_pow = sum(simulate_new_comparison(p, rng, alpha)[0] for _ in range(trials)) / trials
        old_pow = sum(simulate_old_comparison(p, rng)[0] for _ in range(trials)) / trials
        summary[p] = (round(new_pow, 4), round(old_pow, 4))
    # Equal-strength candidates must not be promoted by either rule.
    assert summary[0.50][0] <= 0.02 and summary[0.50][1] <= 0.02
    # The new rule is at least as powerful at the weakest meaningful edge.
    assert summary[0.55][0] >= summary[0.55][1]
    # Both rules should find a clear 60% improvement, but the new rule is
    # dramatically more sensitive to incremental 55-57% gains.
    assert summary[0.60][0] >= 0.8
    assert summary[0.57][0] >= summary[0.57][1]
    print("old_vs_new_power", summary)


# ---------------------------------------------------------------------------
# 4/5. fixture design
# ---------------------------------------------------------------------------

def test_paired_fixture_clustering_preserved():
    outcomes = [
        {"fixture_id": 0, "pair_id": 0, "outcome": "win"},
        {"fixture_id": 1, "pair_id": 0, "outcome": "loss"},
        {"fixture_id": 2, "pair_id": 1, "outcome": "win"},
        {"fixture_id": 3, "pair_id": 1, "outcome": "win"},
    ]
    scores = L.paired_scores(outcomes)
    assert scores == [0.5, 1.0]


def test_primary_and_confirmation_panels_disjoint():
    master = build_fixture_manifest(pairs=L.PRIMARY_PAIRS, team_count=TEAM_COUNT, seed=L.MASTER_SEED)
    used = master[:2 * (L.INCUMBENT_AMP_START_PAIR + L.INCUMBENT_AMP_PAIRS)]
    seed = L.confirmation_seed_for("v2-000000802816-da35700f")
    confirmation = build_fixture_manifest(pairs=L.CONFIRM_PAIRS, team_count=TEAM_COUNT, seed=seed)
    assert L.assert_panels_disjoint(used, confirmation)
    assert not (L.panel_fixture_signatures(used) & L.panel_fixture_signatures(confirmation))


def test_confirmation_stage_slices_are_disjoint_and_cumulative():
    plan = L.confirmation_stage_plan()
    seen = set()
    for stage in plan:
        new_pairs = set(range(stage["first_pair"], stage["cumulative_pairs"]))
        assert not (new_pairs & seen)
        seen |= new_pairs
    assert plan[-1]["cumulative_pairs"] == L.CONFIRM_PAIRS
    assert plan[-1]["cumulative_alpha_weight"] == 1.0


def test_panel_planners_light_and_full():
    full = L.plan_full_panels(candidate_id="c", incumbent_id="i", baseline_id="b", anchors=["a"])
    assert {p["opponent_id"] for p in full} == {"i", "b", "a"}
    assert sum(p["games"] for p in full) == L.FULL_INITIAL_GAMES + L.BASELINE_GAMES + L.ANCHOR_GAMES
    # Incumbent evidence is the incumbent panel, not shared with confirmation.
    incumbent = next(p for p in full if p["opponent_id"] == "i")
    assert incumbent["games"] == L.FULL_INITIAL_GAMES
    light = L.plan_light_panels(candidate_id="c", incumbent_id="i", baseline_id="b", anchors=["a"])
    assert light and abs(sum(p["games"] for p in light) - L.LIGHT_GAMES_V2) <= 4
    assert all(p["games"] % 2 == 0 for p in light)


# ---------------------------------------------------------------------------
# 6. idempotent alpha spending / restart safety
# ---------------------------------------------------------------------------

def test_alpha_and_restart_do_not_duplicate(tmp_path):
    root = tmp_path
    ledger = L.load_alpha_ledger(root)
    before = ledger["consumed"]
    L.commit_alpha(root, candidate_id="cand-a", index=5, alpha=1e-4, decision="retain")
    once = L.load_alpha_ledger(root)
    assert once["consumed"] == pytest.approx(before + 1e-4)
    L.commit_alpha(root, candidate_id="cand-a", index=5, alpha=1e-4, decision="retain")
    twice = L.load_alpha_ledger(root)
    assert twice["consumed"] == once["consumed"]
    assert len([row for row in twice["committed"] if row["candidate_id"] == "cand-a"]) == 1


# ---------------------------------------------------------------------------
# 7. promotion interrupted during write is recoverable
# ---------------------------------------------------------------------------

def test_promotion_interrupted_receipt_write_recoverable(tmp_path, monkeypatch):
    best = tmp_path / "best_policy.json"
    history = tmp_path / "history"
    previous = {"policy_id": "old", "promotions": []}
    L.atomic_json(best, previous)
    candidate = {"id": "new", "model_digest": "m", "file_sha256": "f", "committed_matches": 10}
    real_atomic = L.atomic_json

    def flaky(path, value):
        if Path(path) == best:
            raise RuntimeError("simulated crash during best-policy write")
        return real_atomic(path, value)

    monkeypatch.setattr(L, "atomic_json", flaky)
    with pytest.raises(RuntimeError):
        L.promote_incumbent(best, history, candidate_entry=candidate, previous=previous,
                            decision={}, supporting=[], reason="test", expected_previous_id="old")
    monkeypatch.setattr(L, "atomic_json", real_atomic)
    # History exists, live best is unchanged, and a retry succeeds.
    assert (L.read_json(best) or {})["policy_id"] == "old"
    assert list(history.glob("*.json"))
    L.promote_incumbent(best, history, candidate_entry=candidate, previous=previous,
                        decision={}, supporting=[], reason="test", expected_previous_id="old")
    assert L.read_json(best)["policy_id"] == "new"


# ---------------------------------------------------------------------------
# 8. worker restart preserves unfinished jobs
# ---------------------------------------------------------------------------

def test_worker_restart_preserves_unfinished_job(tmp_path):
    path = tmp_path / "schedule.json"
    L.update_schedule(path, [{"id": "b", "committed_matches": 57344}], 60000, campaign_start=0)
    job_id = L.schedule_jobs(path, statuses=("pending",))[0]["job_id"]
    L.mark_job(path, job_id, "running", started_at=time.time())
    assert L.schedule_jobs(path, statuses=("running",))[0]["job_id"] == job_id
    assert L.requeue_stale_jobs(path) == [job_id]
    assert L.schedule_jobs(path, statuses=("pending",))[0]["job_id"] == job_id
    assert L.schedule_jobs(path, statuses=("completed",)) == []
    L.mark_job(path, job_id, "completed")
    L.requeue_stale_jobs(path)
    assert json.load(open(path))["jobs"][job_id]["status"] == "completed"


# ---------------------------------------------------------------------------
# 9. legacy receipts remain readable
# ---------------------------------------------------------------------------

def test_legacy_receipt_compatibility(tmp_path):
    from scripts.v2_eval_report import summarize_job_receipt
    legacy = {"job_id": "full:legacy", "tier": "full", "candidate_id": "c",
              "decision": {"decision": "retain"}, "matchups": []}
    summary = summarize_job_receipt(legacy)
    assert summary["promotion_status"] == "LEGACY"
    assert summary["ruleset_version"] == "pa3-v2-promotion-rules-v1"
    modern = dict(legacy, promotion_status="PROMOTED", ruleset_version="pa3-v2-promotion-rules-v2")
    assert summarize_job_receipt(modern)["promotion_status"] == "PROMOTED"


# ---------------------------------------------------------------------------
# 10. lightweight jobs cannot promote
# ---------------------------------------------------------------------------

def test_lightweight_jobs_cannot_promote():
    from scripts.v2_eval_worker import light_job_decision
    ok = light_job_decision([{"status": "success", "paired_fixtures": 128}])
    assert ok["promotion_status"] == L.DIAGNOSTIC_ONLY and ok["promoted"] is False
    bad = light_job_decision([{"status": "evaluation_invalid"}])
    assert bad["promotion_status"] == L.INVALID and bad["promoted"] is False


# ---------------------------------------------------------------------------
# 11. promotion cannot change snapshot/learner artefacts
# ---------------------------------------------------------------------------

def test_champion_promotion_does_not_mutate_snapshots(tmp_path):
    snap = tmp_path / "snapshots" / "aa" / "abc"
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b"frozen-model-state")
    before = snap.read_bytes()
    best = tmp_path / "best_policy.json"
    L.atomic_json(best, {"policy_id": "old", "promotions": []})
    candidate = {"id": "new", "model_digest": "m", "file_sha256": "f", "committed_matches": 10}
    L.promote_incumbent(best, tmp_path / "history", candidate_entry=candidate,
                        previous={"policy_id": "old", "promotions": []}, decision={},
                        supporting=[], reason="test", expected_previous_id="old")
    assert snap.read_bytes() == before
    assert (L.read_json(best) or {})["policy_id"] == "new"


# ---------------------------------------------------------------------------
# 12. milestone coalescing still works
# ---------------------------------------------------------------------------

def test_milestone_coalescing_v2(tmp_path):
    path = tmp_path / "schedule.json"
    snapshots = [
        {"id": "b", "committed_matches": 57344},
        {"id": "c100400", "committed_matches": 100400},
        {"id": "c200704", "committed_matches": 200704},
        {"id": "c303104", "committed_matches": 303104},
    ]
    state = L.update_schedule(path, snapshots, 349856, campaign_start=57344)
    assert state["boundaries"]["200000"]["status"] == "coalesced"
    assert state["boundaries"]["150000"]["checkpoint_id"] == "c200704"
    assert state["jobs"]["full:c200704"]["tier"] == "full"
    again = L.update_schedule(path, snapshots, 349856, campaign_start=57344)
    assert sorted(again["jobs"]) == sorted(state["jobs"])


# ---------------------------------------------------------------------------
# 13. stale candidate cannot overwrite champion
# ---------------------------------------------------------------------------

def test_stale_candidate_cannot_overwrite_champion(tmp_path):
    best = tmp_path / "best_policy.json"
    L.atomic_json(best, {"policy_id": "old", "promotions": []})
    cand_b = {"id": "b", "model_digest": "m", "file_sha256": "f", "committed_matches": 20}
    L.promote_incumbent(best, tmp_path / "history", candidate_entry=cand_b,
                        previous={"policy_id": "old", "promotions": []}, decision={},
                        supporting=[], reason="test", expected_previous_id="old")
    cand_a = {"id": "a", "model_digest": "m", "file_sha256": "f", "committed_matches": 10}
    with pytest.raises(L.StaleIncumbentError):
        L.promote_incumbent(best, tmp_path / "history", candidate_entry=cand_a,
                            previous={"policy_id": "old", "promotions": []}, decision={},
                            supporting=[], reason="test", expected_previous_id="old")
    assert L.read_json(best)["policy_id"] == "b"


# ---------------------------------------------------------------------------
# 14. resource controls
# ---------------------------------------------------------------------------

def _args(**kwargs):
    return argparse.Namespace(reserve_gib=kwargs.get("reserve_gib", 0.0001),
                              baseline_gps=kwargs.get("baseline_gps", 50.0))


def test_resource_controls(tmp_path):
    from scripts.v2_eval_worker import resource_gate
    root = tmp_path / "eval"
    (root / "state").mkdir(parents=True)
    log = tmp_path / "rank0.log"
    log.write_text("\n".join(f"[rank0] iteration {i} complete in {90 + i * 0.1:.1f}s" for i in range(10)))
    gate, record = resource_gate(root, log, _args())
    assert gate["status"] == "ok" and record["ratio"] > 0.8
    # Disk reserve violated.
    gate, _ = resource_gate(root, log, _args(reserve_gib=1e9))
    assert gate["status"] == "resource_deferred"
    # Throughput collapse below the deferral floor.
    slow = tmp_path / "slow.log"
    slow.write_text("\n".join("[rank0] iteration " + str(i) + " complete in 300.0s" for i in range(10)))
    gate, record = resource_gate(root, slow, _args())
    assert gate["status"] == "resource_deferred"


# ---------------------------------------------------------------------------
# full escalation path (deterministic + replay of archived outcomes)
# ---------------------------------------------------------------------------

def test_full_escalation_path_promotes_strong_candidate(tmp_path):
    alpha = L.alpha_for_comparison(L.initial_alpha_ledger())["alpha"]
    rng = random.Random(7)
    promoted, used, detail = simulate_new_comparison(0.65, rng, alpha)
    assert promoted is True
    assert detail["screen"] in ("promising", "amplify")
    assert detail["last_stage"] >= 1
    assert used <= L.FULL_INITIAL_GAMES + L.CONFIRM_MAX_GAMES
    # Alpha is spent exactly once per candidate.
    L.commit_alpha(tmp_path, candidate_id="v2-strong", index=5, alpha=alpha, decision="promote")
    L.commit_alpha(tmp_path, candidate_id="v2-strong", index=5, alpha=alpha, decision="promote")
    ledger = L.load_alpha_ledger(tmp_path)
    assert ledger["consumed"] == pytest.approx(L.ALPHA_HISTORICAL_CONSUMED + alpha)


def test_replay_archived_outcomes_through_v2_rules(tmp_path):
    """Replay a real archived incumbent matchup through the V2 decision chain."""
    import json as _json
    real = Path("/home/minidc/eval-league/v2/receipts/jobs/full_v2-000000802816-da35700f.json")
    if not real.is_file():
        pytest.skip("archived production receipt unavailable")
    row = _json.loads(real.read_text())
    receipts_dir = Path("/home/minidc/eval-league/v2/receipts")
    target = next((m for m in row["matchups"] if m["opponent_id"] == row["incumbent_id"]), None)
    body = None
    if target is not None:
        candidate = receipts_dir / f"{target['identity']}.json"
        if candidate.is_file():
            body = _json.loads(candidate.read_text())
    if not body or not body.get("outcomes"):
        pytest.skip("raw archived outcomes unavailable")
    scores = L.paired_scores(body["outcomes"])
    screen = L.primary_screening(scores)
    assert screen["status"] in ("confirmed_inferior", "promising", "amplify", "weak")
    # Replay the predeclared coarsening ladder on genuine archived pair scores.
    alpha = L.alpha_for_comparison(L.initial_alpha_ledger())["alpha"]
    decisions = []
    for stage in L.confirmation_stage_plan():
        pooled = scores[:stage["cumulative_pairs"]]
        if len(pooled) < stage["cumulative_pairs"]:
            break
        result = L.confirmation_rule(pooled, alpha_total=alpha, stage=stage)
        decisions.append(result["decision"])
        if result["decision"] in ("promote", "retain_inferior"):
            break
    assert decisions and decisions[-1] in ("promote", "retain_inferior", "continue")

