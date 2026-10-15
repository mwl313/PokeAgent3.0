import copy,json,shutil,tempfile,os
from pathlib import Path
import torch
from agent.ppo.v2_release_migration import validate,BASE,git
root=Path.cwd();out=Path(os.environ.get('V2_SUITE_DIR', str(root/'runs/v2_final_release/migration_batch')));receipt=json.loads((out/'migration.json').read_text());target=git(root,'rev-parse','HEAD');results={}
validate(receipt,root,BASE,target);results['approved_actual_checkpoint']='PASS'
for key in ['target_sha','ragged_effects_precision_transition','team_stats_lineage_id','ppo_config_hash','runtime_blobs','token_lookup_backend']:
 r=copy.deepcopy(receipt);r[key]='tampered'
 try:validate(r,root,BASE,target)
 except (ValueError,KeyError,TypeError) as e:results[key]={'rejected':True,'error':str(e)}
 else:raise AssertionError(key)
for kind in ['weights','adam','rng','rank_missing','rank_altered','manifest']:
 with tempfile.TemporaryDirectory(prefix='v2-tamper-') as d:
  checkpoint=Path(d);original=Path(receipt['checkpoint'])
  for n in ['manifest.json','rank0.pt','rank1.pt']:shutil.copy2(original/n,checkpoint/n)
  if kind=='rank_missing':(checkpoint/'rank1.pt').unlink()
  elif kind in ['rank_altered','manifest']:
   path=checkpoint/('rank1.pt' if kind=='rank_altered' else 'manifest.json');path.write_bytes(path.read_bytes()+b' ')
  else:
   path=checkpoint/'rank0.pt';s=torch.load(path,map_location='cpu',weights_only=False)
   if kind=='weights':next(iter(s['learner']['model'].values())).view(-1)[0]+=1
   elif kind=='adam':
    entry=next(iter(s['learner']['optimizer']['state'].values()));entry['exp_avg'].view(-1)[0]+=1
   else:s['rng']['torch_cpu'][0]^=1
   torch.save(s,path)
  r=copy.deepcopy(receipt);r['checkpoint']=str(checkpoint)
  try:validate(r,root,BASE,target)
  except (ValueError,FileNotFoundError) as e:results[kind]={'rejected':True,'error':str(e)}
  else:raise AssertionError(kind)
(out/'actual_tamper_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
