# AHL V2 — recurring historical diagnostic runbook

Status: **IMPLEMENTED, PLAN-ONLY BY DEFAULT**. Historical PPO training is
disabled. Diagnostics never touch formal promotion authority.

## 1. What runs and when

- **Tier A**: every 200,000 committed matches starting at **2,800,000**.
  The first verified checkpoint at or after the milestone is evaluated against
  up to 8 historical anchors, 256 games per opponent (≤ 2,048 games).
- **Tier B**: every 1,000,000 committed matches starting at **3,800,000**
  (then 4.8M, 5.8M, …). Expanded round robin over at most **24 distinct model
  digests**.
- Tier B at a 200K clock supersedes Tier A at the same clock.
- If several milestones were missed during downtime, each is emitted once and
  associated with the first verified checkpoint that reached it.

## 2. Files

| File | Meaning |
|---|---|
| `diagnostic_state.json` | durable milestone/job state (status, checkpoint, digest, receipt, attempts) |
| `milestone_schedule.json` | latest planning snapshot (`state`, `due`, `new_jobs`, `resumed_jobs`) |
| `resource_gate.json` | topology-derived worker masks, disk/memory/formal/throughput gate |
| `receipts/milestones/<job_id>.json` | completion receipt for a milestone job |
| `run_report.json` / `FULL_COMPLETE.md` | authoritative completed-grid report |
| `ratings.csv` / `ratings.json` | corrected BT ratings + pair-cluster uncertainty |
| `ranking_confidence.csv` | adjacent-rank bootstrap confidence |
| `opponent_plan.json` | `OBSOLETE`, non-activatable provenance record |
| `opponent_plan.smoke-20261017.json` | preserved smoke-era plan (byte-identical) |

## 3. Normal operating procedure

```bash
cd /home/minidc/Desktop/PokeAgent3.0-AHL-fix
PY=/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python
RR=/home/minidc/eval-league/roundrobin/v1

# 1. Refresh the frozen registry snapshot only when a new verified policy exists.
"$PY" scripts/v2_history_roundrobin.py --root "$RR" freeze

# 2. Plan due milestones. This is idempotent and plan-only.
"$PY" scripts/v2_history_roundrobin.py --root "$RR" schedule

# 3. Inspect the resource gate before launching any worker.
"$PY" scripts/ahl_v2_resource_gate.py --root "$RR" --ppo-stopped
#    Use --ppo-running when PPO is live. Current production training is stopped;
#    live-PPO coexistence is UNVERIFIED.

# 4. Execute the planned jobs (CPU-only, low priority, own lock) only after
#    the formal queue is empty. The existing pair runner is resumable.
AHL_RR_ROOT="$RR" bash scripts/rr_pool.sh 16

# 5. Reconcile receipts, regenerate derived reports, then gate.
"$PY" scripts/v2_history_roundrobin.py --root "$RR" report
"$PY" scripts/v2_history_roundrobin.py --root "$RR" gate
"$PY" scripts/v2_history_protected_snapshot.py --root "$RR" \
  --output "$RR/protected_snapshot/corrections_verification.json"
```

## 4. Failure and restart semantics

| Situation | Behaviour | Operator action |
|---|---|---|
| Formal job pending/running | `deferred_formal_priority`; no new work | wait for formal queue |
| Worker interrupted mid-pair | pair receipt absent; job stays `planned`; next plan lists it in `resumed_jobs` | re-run worker |
| Receipt present and matching | job becomes `completed` exactly once | none |
| Receipt corrupt / wrong identity | job becomes `corrupt_receipt`; never completed silently | inspect receipt, remove only the bad file after review |
| Disk below 12 GiB | resource gate pauses; worker stops issuing new pairs | free space |
| Available memory below 8 GiB or swap > 2 GiB | resource gate pauses | inspect host |
| CPU mask conflict with formal CPUs 4–7 / SMT siblings | gate fails closed (0 workers) | fix topology/gate inputs |
| PPO degradation > 5% (sustained) | gate pauses Mode A | review PPO throughput |
| PPO degradation 3–5% | gate reduces to 1 worker × 1 thread | observe |

## 5. Milestone statuses

`planned` → `running` → `completed`, or `failed` / `corrupt_receipt`.
`covered_by_tier_b` marks a Tier A clock folded into the expanded run.
A completed key is never re-planned. `stable_anchor_ids` persists long-term
anchors between Tier B evaluations.

## 6. Hard boundaries

- CPU-only (`CUDA_VISIBLE_DEVICES=`), `nice 19`, constrained threads.
- Never write `/home/minidc/eval-league/v2/**`, the frozen `da35700` worktree or
  the live evaluation worktree.
- Never start/stop/restart training or formal evaluation units.
- Never activate `opponent_plan.json`; it is obsolete.
- Historical diagnostics spend **zero** formal alpha and cannot promote.
- Historical training probability stays **0.0** in every active profile.
