import json, time
from pathlib import Path
import scripts.a10_watchdog as wd

def healthy(tmp):
    return {'launchers':[{'pid':1}], 'workers':[{'pid':2},{'pid':3}], 'ranks':{0:{'complete_iteration':2,'last_update_s':30,'recompute_within':True,'error_lines':[]},1:{'complete_iteration':2,'last_update_s':31,'recompute_within':True,'error_lines':[]}}, 'last_progress_epoch':time.time(), 'disk':{'free_bytes':100*2**30}, 'committed_matches':8192, 'last_complete_iteration':2, 'latest_checkpoint':None}

def test_healthy_no_incident(tmp_path, monkeypatch):
    monkeypatch.setattr(wd,'snapshot',lambda root: healthy(tmp_path))
    out=wd.once(tmp_path); assert out['open']==[]
    assert not (tmp_path/'operations/incidents.jsonl').exists()

def test_duplicate_deduplicated_and_resolved(tmp_path, monkeypatch):
    s=healthy(tmp_path); s['launchers']=[{'pid':1},{'pid':4}]
    monkeypatch.setattr(wd,'snapshot',lambda root:s)
    assert 'duplicate_launcher' in wd.once(tmp_path)['open']; assert 'duplicate_launcher' in wd.once(tmp_path)['open']
    s['launchers']=s['launchers'][:1]; assert wd.once(tmp_path)['open']==[]
    lines=(tmp_path/'operations/incidents.jsonl').read_text().splitlines(); assert len(lines)==2
    assert json.loads(lines[-1])['event']=='resolved'

def test_recompute_and_stale(tmp_path, monkeypatch):
    s=healthy(tmp_path); s['ranks'][0]['recompute_within']=False; s['last_progress_epoch']=time.time()-21*60
    monkeypatch.setattr(wd,'snapshot',lambda root:s)
    open_types=wd.once(tmp_path)['open']; assert 'recompute_failure' in open_types and 'stale_progress' in open_types

def test_disk_critical(tmp_path, monkeypatch):
    s=healthy(tmp_path); s['disk']['free_bytes']=10*2**30
    monkeypatch.setattr(wd,'snapshot',lambda root:s)
    assert 'disk_critical' in wd.once(tmp_path)['open']
