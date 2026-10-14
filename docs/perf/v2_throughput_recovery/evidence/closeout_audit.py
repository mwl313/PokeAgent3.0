import hashlib,json,pathlib,shutil,subprocess
import torch
from agent.ppo.experimental_checkpoint import load_envelope
from agent.model import build_model,PA3Config
from agent.analysis.team_stats import load as load_stats,reconcile
from scripts.run_ddp_ppo import state_digest,_compact_digest
root=pathlib.Path(__file__).resolve().parents[3];out=root/'runs/perf/v2_recovery'
campaign=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z')
recovery=json.loads((out/'reconstruction_contract/SUMMARY.json').read_text())
generation=pathlib.Path(recovery['reconstructed']['checkpoint'])
manifest=json.loads((generation/'manifest.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()
result={'classification':'USER_STOPPED_VERIFIED_PARTIAL','request':'wrap up test at50000',
 'executed_optimization_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'checkpoint':str(generation),'committed_matches':manifest['committed_matches'],
 'replayed_games':manifest['committed_matches']-4096,'completed_replay_iterations':13,
 'original_full_boundary_target':73728,'all_historical_model_and_adam_match':True,
 'raw_script_status_note':'HISTORICAL_REPLAY_DIFFERS was emitted because the fixed73728 target was not reached; matched_historical_digests is true. User stopped early; this is not an observed digest mismatch.',
 'ranks':[],'team_receipt_checksums_verified':0}
states=[]
for rank in (0,1):
    path=generation/f'rank{rank}.pt';assert sha(path)==manifest['ranks'][str(rank)]['sha256']
    e=load_envelope(str(path));states.append(e)
    model=build_model(PA3Config(**e['learner']['config']['model']),device='cpu');model.load_state_dict(e['learner']['model'])
    optimizer=torch.optim.Adam(model.parameters());optimizer.load_state_dict(e['learner']['optimizer'])
    model_hash,adam_hash=state_digest(model,optimizer)
    expected={x['identity']['committed_matches']:x for x in map(json.loads,(campaign/f'reports/continuation.json.rank{rank}.ppo.jsonl').read_text().splitlines())}
    rows=list(map(json.loads,(out/f'reconstruction_contract/reconstructed/report.json.rank{rank}.ppo.jsonl').read_text().splitlines()))
    assert len(rows)==13
    assert all(all(r['safety']['rank_verification'][k]==expected[r['identity']['committed_matches']]['safety']['rank_verification'][k] for k in ['model_digest','optimizer_digest']) for r in rows)
    last=expected[manifest['committed_matches']]['safety']['rank_verification']
    assert model_hash==last['model_digest'] and adam_hash==last['optimizer_digest']
    assert e['accounting']['committed_matches']==manifest['committed_matches'] and e['accounting']['iteration']==manifest['iteration']
    assert e['source_sha']==manifest['source_sha'] and e['ddp']['rank']==rank
    assert e['collector_drain']['complete'] and e['collector_drain']['in_flight_battles']==0
    result['ranks'].append({'rank':rank,'sha256':sha(path),'accounting':e['accounting'],
      'model_digest':model_hash,'optimizer_digest':adam_hash,'scheduler':e['learner']['scheduler'],
      'scaler':e['learner']['scaler'],'rng_keys':list(e['rng']),'collector_restart':e['collector_restart'],
      'collector_drain':e['collector_drain'],'opponents':e['opponent_manifest']})
    del optimizer,model
assert states[0]['rank_digests']==states[1]['rank_digests']
for key in ['source_sha','dataset_manifest_hash','action_schema_hash','opponent_manifest']:
    assert states[0][key]==states[1][key]
for r in manifest['team_stats_receipts']:
    load_stats(r);result['team_receipt_checksums_verified']+=1
reconcile(manifest['team_stats_receipts'],committed_matches=manifest['committed_matches'])
result['team_stats_clock_and_wld_reconciled']=True
backup=campaign/'backups/throughput-stop-57344-20261014';backup.mkdir(exist_ok=False)
for name in ['manifest.json','rank0.pt','rank1.pt']:
    shutil.copy2(generation/name,backup/name);assert sha(backup/name)==sha(generation/name)
result['additional_verified_local_backup']=str(backup)
assert (campaign/'rolling/LATEST').read_text().strip()=='iter-000000004096-1791958101732995099'
result['original_campaign_latest_unchanged']=True
report=json.loads((out/'reconstruction_contract/reconstructed/report.json').read_text());items=[]
for pair in zip(*(r['iterations'] for r in report['ranks'])):
    critical=max(pair,key=lambda r:r['iteration_wall_s'])
    items.append({'clock':critical['global_committed_matches'],'games':critical['iteration_global_committed_matches'],
      'rows':sum(r['rows'] for r in pair),'wall_s':critical['iteration_wall_s'],
      'collect_and_wait_s':critical['collect_wall_s']+critical['lr_clock_wall_s'],'prepare_s':critical['prepare_wall_s'],
      'update_s':critical['update_wall_s'],'steps':critical['optimizer_steps_this_iteration'],
      'recompute_max':max(r['recompute']['max_abs_diff'] for r in pair),
      'errors':sum(r['operational_errors'] for r in pair),'skips':sum(r['report']['optimizer_steps_skipped'] for r in pair),
      'epochs':[r['report']['epochs_run'] for r in pair]})
games=sum(x['games'] for x in items);wall=sum(x['wall_s'] for x in items)
result['measurement']={'iterations':items,'iteration_games_s':games/wall,'all_in_games_s':report['all_in_committed_games_per_s'],
 'wall_s_iterations':wall,'wall_s_launcher':report['launcher_wall_s'],
 'mean_stage_s':{k:sum(x[k] for x in items)/len(items) for k in ['collect_and_wait_s','prepare_s','update_s']},
 'learner_rows':sum(x['rows'] for x in items),'optimizer_steps':sum(x['steps'] for x in items),
 'max_recompute':max(x['recompute_max'] for x in items),'errors':sum(x['errors'] for x in items),'skips':sum(x['skips'] for x in items)}
result['corrected_runtime_ab']='CANCELLED_BY_USER; no corrected-environment candidate speed measurement'
result['corrected_runtime_continuation']='CANCELLED_BY_USER; prior workspace-unset bounded BIT_EXACT remains exploratory'
result['production_cutover']='BLOCKED; original iteration19 recompute failure unresolved'
(out/'CLOSEOUT_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['ranks','measurement']},indent=2))
print(json.dumps({k:v for k,v in result['measurement'].items() if k!='iterations'},indent=2))
