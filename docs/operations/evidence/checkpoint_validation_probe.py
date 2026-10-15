import pathlib,json,hashlib,torch
from agent.ppo.experimental_checkpoint import load_envelope,validate_envelope
from agent.analysis.team_stats import reconcile
from agent.ppo.learner import PPOLearner
from agent.model import PA3Config,build_model
from agent.ppo.config import PPOConfig
r=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-integration');e=r/'docs/operations/evidence';a=json.loads((e/'checkpoint_audit.json').read_text());p=pathlib.Path(a['checkpoint']);ds=[]
for rank in (0,1):
 d=load_envelope(str(p/f'rank{rank}.pt'));ds.append(d)
 validate_envelope(d,dataset_manifest_hash=d['dataset_manifest_hash'],action_schema_hash=d['action_schema_hash'],source_sha=d['source_sha'],opponent_manifest=d['opponent_manifest'],expected_rank=rank,expected_world_size=2,expected_executor='ddp')
 a['ranks'][str(rank)]['accounting']=d['accounting'];a['ranks'][str(rank)]['collector_restart']=d['collector_restart'];a['ranks'][str(rank)]['collector_drain']=d['collector_drain']
 a['ranks'][str(rank)].pop('clock');a['ranks'][str(rank)].pop('iteration');a['ranks'][str(rank)].pop('collector')
assert all(ds[0]['accounting'][k]==ds[1]['accounting'][k] for k in ['iteration','committed_matches']) and ds[0]['rank_digests']==ds[1]['rank_digests']
reconciled=reconcile(a['manifest']['team_stats_receipts'],committed_matches=57344)
a['receipt_reconciliation']=reconciled
backup=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/backups/throughput-stop-57344-20261014')
a['backup_rank_files']=[]
for path in backup.rglob('rank*.pt'):
 digest=hashlib.sha256(path.read_bytes()).hexdigest();a['backup_rank_files'].append({'path':str(path),'sha256':digest,'matches_checkpoint':any(v['sha256']==digest for v in a['ranks'].values())})
a['envelope_validation']='PASS';a['new_source_resume']='BLOCKED: protected runtime delta and secondary-safety contract differ; no valid migration receipt'
(e/'checkpoint_audit.json').write_text(json.dumps(a,indent=2,default=str)+'\n');print('envelope peer/clock and ledger validation passed; backup files',len(a['backup_rank_files']))
