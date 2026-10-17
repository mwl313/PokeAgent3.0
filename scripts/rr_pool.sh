#!/usr/bin/env bash
# rr_pool.sh [N] — run the AHL v2 round-robin grid with N disjoint slice workers.
# Each worker: 2 threads, pinned to 2 distinct physical cores (CPUs 0..39 are the
# first SMT thread of each core on this box), own lock, shared content-addressed
# receipt store. Safe to interrupt: receipts are cached and workers skip pairs
# that already have a success summary.
set -u

N=${1:-16}
W=/home/minidc/Desktop/PokeAgent3.0-AHL-v2
PY=/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python
ROOT=/home/minidc/eval-league/roundrobin/v1
STAMP=$(date +%Y%m%d-%H%M%S)
LOGD=/home/minidc/pa3-ahl-v2/logs/pool-$STAMP
mkdir -p "$LOGD"
cd "$W" || exit 9

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "pool start $(date -Is) workers=$N logdir=$LOGD" | tee "$LOGD/summary.txt"

pids=()
for k in $(seq 1 "$N"); do
  a=$(( (k - 1) * 2 ))
  b=$(( a + 1 ))
  taskset -c "$a,$b" "$PY" scripts/v2_history_roundrobin.py \
    --root "$ROOT" --pair-slice "$k/$N" --lock-name "worker-$k.lock" --no-reports \
    run > "$LOGD/w$k.log" 2>&1 &
  pids+=($!)
done
echo "pids=${pids[*]}" | tee -a "$LOGD/summary.txt"

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "workers finished $(date -Is) failed=$fail" | tee -a "$LOGD/summary.txt"

# coordinator: regenerate the aggregate reports once, then the gate report
"$PY" scripts/v2_history_roundrobin.py --root "$ROOT" report > "$LOGD/report.log" 2>&1
rc_report=$?
"$PY" scripts/v2_history_roundrobin.py --root "$ROOT" gate > "$LOGD/gate.log" 2>&1
rc_gate=$?
echo "report rc=$rc_report gate rc=$rc_gate" | tee -a "$LOGD/summary.txt"
echo "$LOGD" > /home/minidc/pa3-ahl-v2/logs/last-pool-dir
