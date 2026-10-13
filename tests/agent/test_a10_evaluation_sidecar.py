from pathlib import Path
from agent.evaluation.a10_league import atomic,read
from scripts.a10_evaluation_sidecar import once

def test_discovery_poller_does_not_clobber_worker_queue(tmp_path,monkeypatch):
 own=tmp_path/'evaluation/league_v2';own.mkdir(parents=True)
 q={'jobs':{'candidate':{'status':'running','stage':1}}};atomic(own/'jobs.json',q)
 monkeypatch.setattr('scripts.a10_evaluation_sidecar.discover',lambda _: {'policies':{},'rejected':{}})
 assert once(tmp_path)['pending']==0
 assert read(own/'jobs.json')==q

def test_worker_initializes_actual_policy_jobs_once(tmp_path):
 from scripts.a10_evaluation_worker import reconcile
 own=tmp_path/'league';own.mkdir()
 e=lambda n:{'id':str(n),'committed_matches':n,'path':str(tmp_path/f'{n}.pt'),'file_sha256':'x','model_digest':'y'}
 atomic(own/'registry.json',{'policies':{'100':e(100),'200':e(200)}})
 _,q,b=reconcile(own);assert b['policy_id']=='100';assert len(q['jobs'])==1;assert q['jobs']['200']['actual_matches']==200
 assert len(reconcile(own)[1]['jobs'])==1
