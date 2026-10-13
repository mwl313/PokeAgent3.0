import json
from pathlib import Path
from scripts.a10_eval_pipeline import inspect_generation, schedule

def gen(tmp, complete=True):
 r=tmp/'rolling'; p=r/'generations/g'; p.mkdir(parents=True); (r/'LATEST').write_text('g\n')
 for n in ('rank0.pt','rank1.pt'): (p/n).write_bytes(b'x')
 m={'complete':complete,'iteration_complete':complete,'world_size':2,'dataset_manifest_hash':'1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb','source_sha':'0b5d9307432','committed_matches':102400,'ranks':{'0':{'path':'rank0.pt','bytes':1},'1':{'path':'rank1.pt','bytes':1}}}; (p/'manifest.json').write_text(json.dumps(m)); return r

def test_generation_verified(tmp_path): assert inspect_generation(gen(tmp_path))['manifest']['committed_matches']==102400

def test_incomplete_rejected(tmp_path):
 import pytest
 with pytest.raises(ValueError): inspect_generation(gen(tmp_path,False))

def test_a10_schedule_anchor_and_missed_boundaries(tmp_path):
 state=tmp_path/'s.json'; jobs=schedule(state,102400,{'policy_id':'a10-102400'})
 assert [x['id'] for x in jobs]==['light:50000','light:100000']
 assert jobs[0]['scheduled_boundary']==50000 and jobs[0]['checkpoint']['policy_id']=='a10-102400'
