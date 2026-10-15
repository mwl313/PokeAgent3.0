import pathlib,json,hashlib,math
from agent.ppo.experimental_checkpoint import load_envelope,hash_file,validate_envelope
from agent.analysis.team_stats import reconcile,load
r=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-integration');out=r/'runs/v2_final_release/continuation_canonical';summary=json.loads((out/'SUMMARY.json').read_text());assert summary['classification']=='BIT_EXACT'
pair=[json.loads((out/'uninterrupted'/f'report.json.rank{k}.json').read_text()) for k in (0,1)];rows=[]
for i in range(6):
 a,b=[rank['iterations'][i] for rank in pair];wall=max(a['iteration_wall_s'],b['iteration_wall_s']);model=a['model_digest'];adam=a['optimizer_digest'];assert model==b['model_digest'] and adam==b['optimizer_digest']
 assert a['global_committed_matches']==b['global_committed_matches']==57344+(i+1)*4096
 assert all(x['report']['epochs_run']==2 and x['report']['optimizer_steps_skipped']==0 and x['operational_errors']==0 and x['recompute']['within_gate'] and x['recompute']['all_ranks_accepted'] for x in (a,b))
 assert all(math.isfinite(x['report'][key]) for x in (a,b) for key in ['policy_loss','value_loss','approx_kl','entropy','grad_norm','learning_rate'])
 rows.append({'iteration':15+i,'diagnostic_clock':a['global_committed_matches'],'games':4096,'rows':a['rows']+b['rows'],'wall_s':wall,'games_s':4096/wall,'rows_s':(a['rows']+b['rows'])/wall,'collect_s':max(x['collect_wall_s'] for x in (a,b)),'prepare_s':max(x['prepare_wall_s'] for x in (a,b)),'update_s':max(x['update_wall_s'] for x in (a,b)),'checkpoint_s':max(x['checkpoint_wall_s'] for x in (a,b)),'epochs':[x['report']['epochs_run'] for x in (a,b)],'steps':a['optimizer_steps_this_iteration'],'cumulative_steps':a['optimizer_steps_cumulative'],'max_recompute':max(x['recompute']['max_abs_diff'] for x in (a,b)),'learning_rate':a['effective_learning_rate'],'model_digest':model,'optimizer_digest':adam,'ppo':[x['report'] for x in (a,b)],'gpu_peak_reserved_gib':[x['gpu_peak_reserved_gib'] for x in (a,b)],'cpu_rss_peak_gib':[x['rss_peak_gib'] for x in (a,b)]})
checkpoint=pathlib.Path(summary['final_checkpoint']);manifest=json.loads((checkpoint/'manifest.json').read_text());states=[]
for rank in (0,1):
 path=checkpoint/f'rank{rank}.pt';assert hash_file(path)==manifest['ranks'][str(rank)]['sha256'];state=load_envelope(str(path));states.append(state);validate_envelope(state,dataset_manifest_hash=state['dataset_manifest_hash'],action_schema_hash=state['action_schema_hash'],source_sha=summary['source_sha'],expected_rank=rank,expected_world_size=2,expected_executor='ddp');assert state['experiment']['canonical_encoder_batch_rows']==1024 and state['experiment']['recompute_secondary'] and state['experiment']['token_lookup_backend']=='gather'
reconcile(manifest['team_stats_receipts'],committed_matches=81920)
# Fork receipts have distinct attempt IDs; compare every outcome counter.
reference=next(p for p in (out/'uninterrupted/rolling/generations').iterdir() if p.name.startswith('iter-000000065536-'))
restored=out/'resumed/rolling/generations'/(out/'resumed/rolling/LATEST').read_text().strip()
def ledger(path):
 refs=json.loads((path/'manifest.json').read_text())['team_stats_receipts'];reconcile(refs,committed_matches=65536);values=[load(ref) for ref in refs]
 return sorted([{k:v[k] for k in ['rank','iteration_id','start_matches','end_matches','natural_matches','drawn_matches','team_count','team_counts','matchups','policy_splits']} for v in values],key=lambda v:(v['iteration_id'],v['rank']))
ledger_equal=ledger(reference)==ledger(restored);assert ledger_equal
result={'scope':'isolated diagnostic; production matches added0','source_sha':summary['source_sha'],'iterations':rows,'weighted_games_s':24576/sum(x['wall_s'] for x in rows),'rows_s':sum(x['rows'] for x in rows)/sum(x['wall_s'] for x in rows),'steps_s':sum(x['steps'] for x in rows)/sum(x['wall_s'] for x in rows),'final_checkpoint':str(checkpoint),'manifest_sha256':hash_file(checkpoint/'manifest.json'),'rank_artifact_hashes':{str(k):hash_file(checkpoint/f'rank{k}.pt') for k in (0,1)},'final_rank_digests':states[0]['rank_digests'],'final_scheduler':states[0]['learner']['scheduler'],'final_scaler':states[0]['learner']['scaler'],'ledger_equal_on_restore':ledger_equal,'restore_classification':summary['classification'],'primary_tolerance':.001,'errors':0,'skips':0}
(r/'runs/v2_final_release/ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['iterations']},indent=2))
