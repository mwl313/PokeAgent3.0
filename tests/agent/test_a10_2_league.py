import copy,json
from pathlib import Path
import pytest
from agent.evaluation.a10_league import atomic,read,promotion,digest,validate_payload,SOURCE,DATASET,ACTION,MODEL

def payload():
 return {'kind':'ppo_experimental_iteration_boundary','source_sha':SOURCE,'dataset_manifest_hash':DATASET,'action_schema_hash':ACTION,'model_config_hash':MODEL,'accounting':{'iteration':5,'committed_matches':20480},'ddp':{'rank':0,'world_size':2,'communicator':'recreate'},'collector_drain':{'complete':True,'in_flight_battles':0},'experiment':{'precision_mode':'D','precision_stability':'fp16-reduction-off','precision_contract_version':'pa3-execution-contract-v1'},'rank_digests':{'0':{'model':'a','optimizer':'b'},'1':{'model':'a','optimizer':'b'}}}

def test_checkpoint_contract_rejects_mismatches():
 p=payload();m={'iteration':5,'committed_matches':20480};assert validate_payload(p,m,0)==p
 for field,val in [('source_sha','other'),('dataset_manifest_hash','other'),('action_schema_hash','other'),('model_config_hash','other')]:
  changed=copy.deepcopy(p);changed[field]=val
  with pytest.raises(ValueError):validate_payload(changed,m,0)
 changed=copy.deepcopy(p);changed['rank_digests']['1']['optimizer']='other'
 with pytest.raises(ValueError):validate_payload(changed,m,0)

def outcomes(score,n=128):
 return [{'pair_id':i,'fixture_id':2*i+j,'outcome':'win' if score==1 else 'loss' if score==0 else 'draw','policy_side':j} for i in range(n) for j in (0,1)]

def test_promotion_requires_direct_evidence_and_errors_fail():
 assert promotion(outcomes(1),2,0)['decision']=='promote'
 assert promotion(outcomes(.5),2,0)['decision']=='escalate'
 assert promotion(outcomes(.5),2,2)['decision']=='retain'
 x=outcomes(1);x[0]['outcome']='error';assert promotion(x,2,0)['decision']=='invalid'

def test_global_alpha_spending():
 total=sum(.05/(a*(a+1)*b*(b+1)*3) for a in range(1,100) for b in range(1,100) for stage in range(3))
 assert total<.05

def test_atomic_schema_and_full_identity(tmp_path):
 p=tmp_path/'x.json';atomic(p,{'seed':3});assert read(p)=={'seed':3}
 assert digest({'seed':1,'opponent':'a','fixtures':'x'})!=digest({'seed':2,'opponent':'a','fixtures':'x'})

def test_no_milestone_clock_relabel(tmp_path):
 from scripts.a10_league_discover import discover
 run=tmp_path/'run';(run/'rolling/generations').mkdir(parents=True)
 r=discover(run); assert r['policies']=={};assert read(run/'evaluation/league_v2/milestones.json')['anchor']==0

def test_snapshot_integrity_and_clock(tmp_path):
 import torch
 from agent.evaluation.a10_league import store_snapshot,verify_entry
 p={'learner':{'model':{'weight':torch.ones(2)}},'accounting':{'committed_matches':102400},'experiment':{'precision_mode':'D'},'source_sha':SOURCE,'dataset_manifest_hash':DATASET,'action_schema_hash':ACTION,'model_config_hash':MODEL,'checkpoint_generation':'g','kind':'a10_inference_snapshot'}
 e=store_snapshot(tmp_path,p);assert e['committed_matches']==102400;assert verify_entry(e)['kind']=='a10_inference_snapshot'
 Path(e['path']).chmod(0o644);Path(e['path']).write_bytes(b'corrupt')
 with pytest.raises(ValueError):verify_entry(e)

def test_incomplete_and_corrupt_generation_rejected(tmp_path):
 from agent.evaluation.a10_league import capture_generation
 r=tmp_path/'rolling';g=r/'generations/g';g.mkdir(parents=True)
 atomic(g/'manifest.json',{'complete':False})
 with pytest.raises(ValueError):capture_generation(r,'g',tmp_path/'archive')
 atomic(g/'manifest.json',{'complete':True,'iteration_complete':True,'world_size':2,'source_sha':SOURCE,'dataset_manifest_hash':DATASET,'ranks':{'0':{'path':'rank0.pt','bytes':1,'sha256':'bad'},'1':{'path':'rank1.pt','bytes':1,'sha256':'bad'}}})
 (g/'rank0.pt').write_bytes(b'x');(g/'rank1.pt').write_bytes(b'y')
 with pytest.raises(ValueError):capture_generation(r,'g',tmp_path/'archive')

def test_new_trainer_source_requires_registered_exact_migration(tmp_path):
 from agent.evaluation.a10_league import capture_generation
 r=tmp_path/'rolling';g=r/'generations/g';g.mkdir(parents=True)
 m={'complete':True,'iteration_complete':True,'world_size':2,'source_sha':'f'*40,'dataset_manifest_hash':DATASET}
 atomic(g/'manifest.json',m)
 with pytest.raises(ValueError,match='unregistered'):
  capture_generation(r,'g',tmp_path/'archive')

def test_evaluation_identity_separates_material_inputs():
 from agent.evaluation.league import evaluation_identity
 from agent.evaluation.fixed_policy import build_fixture_manifest
 e={'id':'a','file_sha256':'b','model_digest':'c'};f=build_fixture_manifest(pairs=2,team_count=1220)
 key=lambda seed,env:evaluation_identity(e,e,f,seed=seed,max_rounds=2000,environment=env)[0]
 assert key(1,{'backend':'batched','batch_size':16})!=key(2,{'backend':'batched','batch_size':16})
 assert key(1,{'backend':'batched','batch_size':16})!=key(1,{'backend':'batched','batch_size':32})
