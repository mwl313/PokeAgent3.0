#!/usr/bin/env bash
# rr_pool.sh [N] — run the AHL v2 round-robin grid with N disjoint slice workers.
#
# CPU affinity is derived from the live sysfs topology by
# scripts/ahl_v2_resource_gate.py. Workers never use the formal evaluation
# CPUs (4-7) or their SMT siblings, and the pool pauses when formal evaluation
# has pending work or the disk/memory guard is breached.
#
# Each worker: low priority, CPU-only, 2 threads, own lock, shared
# content-addressed receipt store. Safe to interrupt: successes are cached and
# workers skip pairs that already have a compatible success summary.
set -u

REQUESTED=${1:-0}
W=$(cd "$(dirname "$0")/.." && pwd)
PY=${PA3_PYTHON:-/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python}
ROOT=${AHL_RR_ROOT:-/home/minidc/eval-league/roundrobin/v1}
FORMAL_ROOT=${AHL_FORMAL_ROOT:-/home/minidc/eval-league/v2}
STAMP=$(date +%Y%m%d-%H%M%S)
LOGD=${AHL_LOG_DIR:-/home/minidc/pa3-ahl-v2/logs}/pool-$STAMP
mkdir -p "$LOGD" || exit 9
cd "$W" || exit 9

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

PPO_FLAG=--ppo-stopped
if pgrep -f 'run_ddp_ppo.py' >/dev/null 2>&1; then
  PPO_FLAG=--ppo-running
fi

GATE=$("$PY" scripts/ahl_v2_resource_gate.py \
  --root "$ROOT" --formal-root "$FORMAL_ROOT" "$PPO_FLAG" \
  --mode-b-workers "${REQUESTED:-16}" --format shell) || {
  echo "resource gate failed; refusing to start workers" | tee "$LOGD/summary.txt"
  exit 9
}
eval "$GATE"

echo "pool start $(date -Is) mode=$GATE_MODE gate=$GATE_STATUS workers=$GATE_WORKERS logdir=$LOGD" \
  | tee "$LOGD/summary.txt"
if [ "$GATE_WORKERS" -le 0 ]; then
  echo "resource gate allows no workers; see $ROOT/resource_gate.json" | tee -a "$LOGD/summary.txt"
  exit 0
fi

N=$GATE_WORKERS
if [ "${REQUESTED:-0}" -gt 0 ] && [ "${REQUESTED:-0}" -lt "$N" ]; then
  N=$REQUESTED
fi

pids=()
for k in $(seq 1 "$N"); do
  mask=${GATE_MASKS[$((k - 1))]}
  if [ -z "$mask" ]; then
    continue
  fi
  taskset -c "$mask" "$PY" scripts/v2_history_roundrobin.py \
    --root "$ROOT" --formal-root "$FORMAL_ROOT" \
    --pair-slice "$k/$N" --lock-name "worker-$k.lock" --no-reports \
    run > "$LOGD/w$k.log" 2>&1 &
  pids+=($!)
done
echo "pids=${pids[*]}" | tee -a "$LOGD/summary.txt"

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "workers finished $(date -Is) failed=$fail" | tee -a "$LOGD/summary.txt"

# Coordinator: regenerate the aggregate reports once, then the gate report.
"$PY" scripts/v2_history_roundrobin.py --root "$ROOT" --formal-root "$FORMAL_ROOT" \
  report > "$LOGD/report.log" 2>&1
rc_report=$?
"$PY" scripts/v2_history_roundrobin.py --root "$ROOT" --formal-root "$FORMAL_ROOT" \
  gate > "$LOGD/gate.log" 2>&1
rc_gate=$?
echo "report rc=$rc_report gate rc=$rc_gate" | tee -a "$LOGD/summary.txt"
mkdir -p /home/minidc/pa3-ahl-v2/logs 2>/dev/null || true
echo "$LOGD" > /home/minidc/pa3-ahl-v2/logs/last-pool-dir 2>/dev/null || true
