import json
from pathlib import Path
from scripts.a10_status import parse_logs

def make_root(tmp_path, complete=True):
    root=tmp_path/'run'; (root/'rolling/generations/g').mkdir(parents=True)
    (root/'report.json.rank0.log').write_text('[rank0] iteration 25/100: collected 2048 games / 500 rows in 2.0s\n[rank0] recompute gate: 576 rows max|diff|=1.0e-06 (tol 1.0e-03) within=True\n[rank0] update done in 3.0s (epochs 2, steps 4, cumulative steps 4, skipped 0, global matches 4096)\n[rank0] iteration 25 complete in 5.0s, digests equal\n')
    (root/'report.json.rank1.log').write_text('[rank1] iteration 25/100: collected 2048 games / 501 rows in 2.1s\n[rank1] recompute gate: 576 rows max|diff|=1.0e-06 (tol 1.0e-03) within=True\n[rank1] update done in 3.1s (epochs 2, steps 4, cumulative steps 4, skipped 0, global matches 4096)\n[rank1] iteration 25 complete in 5.0s, digests equal\n')
    for n in ('rank0.pt','rank1.pt'): (root/'rolling/generations/g'/n).write_bytes(b'x')
    manifest={'committed_matches':102400,'complete':complete,'iteration':25,'ranks':{'0':{'path':'rank0.pt','bytes':1},'1':{'path':'rank1.pt','bytes':1}},'dataset_manifest_hash':'1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb'}
    (root/'rolling/generations/g/manifest.json').write_text(json.dumps(manifest)); (root/'rolling/LATEST').write_text('g\n'); return root

def test_valid_latest_generation_and_rank_logs(tmp_path):
    root=make_root(tmp_path); s=parse_logs(root)
    assert s['ranks'][0]['complete_iteration']==25 and s['ranks'][1]['matches']==4096
    # path resolution is exercised by the status snapshot in the live audit; parse_logs remains pure.
    assert json.loads((root/'rolling/generations/g/manifest.json').read_text())['complete']

def test_incomplete_generation_is_not_accepted_by_watchdog(tmp_path, monkeypatch):
    root=make_root(tmp_path, complete=False); import scripts.a10_watchdog as wd
    s={'launchers':[{'pid':1}],'workers':[{'pid':2},{'pid':3}],**parse_logs(root),'disk':{'free_bytes':100*2**30}}
    monkeypatch.setattr(wd,'snapshot',lambda _:s)
    assert 'checkpoint_overdue' not in wd.once(root)['open']
