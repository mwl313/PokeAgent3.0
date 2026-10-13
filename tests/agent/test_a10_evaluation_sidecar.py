import json
from scripts.a10_evaluation_sidecar import once

def test_sidecar_queues_and_deduplicates(tmp_path, monkeypatch):
 run=tmp_path/'run'; roll=run/'rolling/generations/g'; roll.mkdir(parents=True); (run/'rolling/LATEST').write_text('g\n')
 for n in ('rank0.pt','rank1.pt'): (roll/n).write_bytes(b'x')
 m={'complete':True,'iteration_complete':True,'world_size':2,'dataset_manifest_hash':'1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb','source_sha':'0b5d930','committed_matches':102400,'ranks':{'0':{'path':'rank0.pt','bytes':1},'1':{'path':'rank1.pt','bytes':1}}}; (roll/'manifest.json').write_text(json.dumps(m))
 monkeypatch.setattr('scripts.a10_evaluation_sidecar.archive_snapshot',lambda *x,**k:{'policy_id':'a10-test','committed_matches':102400})
 assert once(run)['status']=='queued'; assert once(run)['status']=='queued'
 assert len(json.loads((run/'evaluation/schedule.json').read_text())['jobs'])==2
