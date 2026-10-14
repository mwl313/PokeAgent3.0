"""Bounded two-rank uninterrupted/exit/restore comparison: 256 total natural games."""
import subprocess,json,os,sys,hashlib
import torch
import numpy as np
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
root=Path.cwd();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
py=sys.executable
base=[py,'scripts/run_ddp_ppo.py','--games','32','--envs','32','--workers','8','--epochs','2','--microbatch','1024','--minibatch','4096','--teams','engine/data/a10-training-teams.json','--dataset-manifest','data/teams/a10-p0-proposed-20261013/manifest.json','--precision-mode','D','--precision-stability','fp16-reduction-off','--experimental-lr-scale','0.5','--checkpoint-interval-matches','1','--deterministic-diagnostic','--timeout','900']
env=dict(os.environ,PYTHONPATH='engine/python:.',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
commands=[]
for name,iters,resume in [('reference',2,None),('first',1,None),('resumed',1,'first')]:
 d=out/name;d.mkdir(exist_ok=True)
 cmd=base+['--iterations',str(iters),'--report',str(d/'report.json'),'--experimental-checkpoint',str(d/'boundary'),'--rolling-checkpoint-root',str(d/'rolling'),'--team-stats-root',str(out/('first' if resume else name)/'team_stats')]
 if resume:cmd+=['--resume-latest-from',str(out/resume/'rolling')]
 commands.append(cmd)
 with (d/'launcher.log').open('w') as log:
  r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:
  (out/'FAILED.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode},indent=2));sys.exit(r.returncode)
 print(name+' complete',flush=True)
def hash_tree(value):
 h=hashlib.sha256()
 def visit(v):
  if isinstance(v,torch.Tensor):
   a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   h.update(str(type(v)).encode())
   for a in v:visit(a)
  else:h.update(repr(v).encode())
 visit(value);return h.hexdigest()
report={'commands':commands,'comparisons':{}}
for rank in (0,1):
 ref=json.load(open(out/'reference'/f'report.json.rank{rank}.json'))['iterations'][1]
 resumed=json.load(open(out/'resumed'/f'report.json.rank{rank}.json'))['iterations'][0]
 pairs={k:(ref[k],resumed[k]) for k in ['model_digest','optimizer_digest','games','rows','global_committed_matches']}
 for k in ['rollout_digest','optimizer_steps']:pairs[k]=(ref['parity'][k],resumed['parity'][k])
 for k in ['wins','losses','draws','operational_errors']:pairs['collector_'+k]=(ref['collector_stats'][k],resumed['collector_stats'][k])
 states=[torch.load(out/name/f'boundary.rank{rank}',weights_only=False,map_location='cpu') for name in ['reference','resumed']]
 for k in ['scheduler','scaler']:
  pairs[k]=tuple(hash_tree(e['learner'][k]) for e in states)
 pairs['rng']=tuple(hash_tree(e['rng']) for e in states)
 report['comparisons'][str(rank)]={k:{'reference':a,'resumed':b,'equal':a==b} for k,(a,b) in pairs.items()}
report['classification']='BIT_EXACT' if all(v['equal'] for rank in report['comparisons'].values() for v in rank.values()) else 'NOT_REPRODUCIBLE'
(out/'COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n');print(report['classification'],flush=True)
