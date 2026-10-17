"""Resource-gate regression tests: work-normalised throughput + stale-log handling.

Root cause this guards: the gate used to compare matches/second against a frozen
baseline. Matches get longer as policies improve (more decisions per match), so
matches/second falls with no machine slowdown and the gate eventually deferred
every evaluation. The gate now compares learner rows/second instead.
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / 'engine' / 'python')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.evaluation.v2_league import production_throughput, production_work_throughput  # noqa: E402

_spec = importlib.util.spec_from_file_location('v2_eval_worker', ROOT / 'scripts' / 'v2_eval_worker.py')
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def _log(tmp_path, iterations, name='continuation.json.rank0.log'):
    """iterations: list of (iteration, rows, wall_s)."""
    lines = []
    for index, rows, wall in iterations:
        lines.append(f'[rank0] iteration {index}/24401: collected 2048 games / {rows} rows in {wall * 0.3:.1f}s')
        lines.append(f'[rank0] iteration {index} complete in {wall:.1f}s, digests equal')
    path = tmp_path / name
    path.write_text('\n'.join(lines) + '\n')
    return path


def _args(**over):
    base = dict(baseline_gps=None, baseline_rows_per_s=None, reserve_gib=12.0,
                stale_after_s=900.0, gate_log_interval=300.0)
    base.update(over)
    return SimpleNamespace(**base)


def _root(tmp_path):
    root = tmp_path / 'league'
    (root / 'state').mkdir(parents=True, exist_ok=True)
    return root


def test_work_throughput_is_invariant_to_game_length(tmp_path):
    """Games lengthen (rows/match up ~29%) but the machine is equally fast."""
    short = [(i, 35000, 90.0) for i in range(1, 9)]
    long = [(i, 45000, 115.7) for i in range(1, 9)]
    short_rate, short_detail = production_work_throughput(_log(tmp_path, short, 'short.log'))
    long_rate, long_detail = production_work_throughput(_log(tmp_path, long, 'long.log'))
    assert short_rate == pytest.approx(35000 / 90.0, rel=1e-6)
    assert long_rate == pytest.approx(45000 / 115.7, rel=1e-6)
    assert long_rate / short_rate == pytest.approx(1.0, abs=0.01), long_detail
    # the metric that used to gate: matches/second drops by >20% on the same machine
    old_short = production_throughput(tmp_path / 'short.log')
    old_long = production_throughput(tmp_path / 'long.log')
    assert old_long / old_short < 0.80


def test_gate_ok_when_work_rate_stable(tmp_path):
    root = _root(tmp_path)
    log = _log(tmp_path, [(i, 44000, 115.0) for i in range(1, 9)])
    gate, record = worker.resource_gate(root, log, _args(baseline_rows_per_s=44000 / 115.0))
    assert gate['status'] == 'ok'
    assert record['ratio'] == pytest.approx(1.0, abs=1e-6)
    assert record['deferred'] is False
    assert record['metric'] == 'learner-rows-per-second-v1'


def test_gate_defers_on_real_slowdown(tmp_path):
    """Same game length, machine 25% slower -> genuine degradation, defer."""
    root = _root(tmp_path)
    log = _log(tmp_path, [(i, 44000, 153.0) for i in range(1, 9)])
    gate, record = worker.resource_gate(root, log, _args(baseline_rows_per_s=44000 / 115.0))
    assert gate['status'] == 'resource_deferred'
    assert record['deferred'] is True
    assert record['ratio'] < 0.80
    assert 'rows/s' in gate['reason']


def test_gate_skips_when_production_log_is_stale(tmp_path):
    """Training stopped: evaluation cannot slow it down, so the gate must not block."""
    root = _root(tmp_path)
    log = _log(tmp_path, [(i, 44000, 153.0) for i in range(1, 9)])
    old = 1_000_000.0
    os.utime(log, (old, old))
    gate, record = worker.resource_gate(root, log, _args(baseline_rows_per_s=44000 / 115.0))
    assert gate['status'] == 'ok'
    assert record['stale'] is True
    assert record['deferred'] is False
    assert record['production_log_age_s'] and record['production_log_age_s'] > 900


def test_gate_adopts_baseline_on_first_run(tmp_path):
    root = _root(tmp_path)
    log = _log(tmp_path, [(i, 44000, 115.0) for i in range(1, 9)])
    gate, record = worker.resource_gate(root, log, _args())
    assert gate['status'] == 'ok'
    assert record['baseline_rows_per_s'] == pytest.approx(44000 / 115.0, rel=1e-6)
    assert record['ratio'] == pytest.approx(1.0, abs=1e-6)


def test_disk_reserve_still_defers(tmp_path, monkeypatch):
    root = _root(tmp_path)
    log = _log(tmp_path, [(i, 44000, 115.0) for i in range(1, 9)])
    monkeypatch.setattr(worker, 'disk_headroom_bytes', lambda _p: 1 << 20)
    gate, _record = worker.resource_gate(root, log, _args(baseline_rows_per_s=44000 / 115.0))
    assert gate['status'] == 'resource_deferred'
    assert 'disk reserve' in gate['reason']


def test_deferral_is_published_to_active_json(tmp_path):
    root = _root(tmp_path)
    (root / 'state' / 'active.json').write_text('{"status": "idle", "last_job": "full:x", "updated_at": 1}')
    log = _log(tmp_path, [(i, 44000, 153.0) for i in range(1, 9)])
    worker.resource_gate(root, log, _args(baseline_rows_per_s=44000 / 115.0))
    import json
    published = json.loads((root / 'state' / 'active.json').read_text())
    assert published['status'] == 'resource_deferred'
    assert published['last_job'] == 'full:x'          # preserved
    assert published['gate_reason']
    assert published['gate']['metric'] == 'learner-rows-per-second-v1'


def test_immature_window_does_not_defer(tmp_path):
    """One iteration is not enough evidence to throttle on."""
    root = _root(tmp_path)
    log = _log(tmp_path, [(1, 44000, 153.0)])
    gate, record = worker.resource_gate(root, log, _args(baseline_rows_per_s=44000 / 115.0))
    assert gate['status'] == 'ok'
    assert record['mature_window'] is False
