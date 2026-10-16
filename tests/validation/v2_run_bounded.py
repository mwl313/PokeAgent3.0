"""Bounded full-state native DDP validation; all paths remain isolated."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from agent.ppo.v2_release_migration import prepare
p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--name',required=True);p.add_argument('--iterations',type=int,default=1);p.add_argument('--games',type=int,default=1024);p.add_argument('--checkpoint-interval',type=int,default=100000);p.add_argument('--resume');p.add_argument('--stats');p.add_argument('--stop',action='store_true');p.add_argument('--audit',action='store_true');a=p.parse_args()
root=Path.cwd();out=Path(a.suite).resolve();master=json.loads((out/'migration.json').read_text());profile=json.loads((root/'configs/pa3_v2_production.yaml').read_text());checkpoint=a.resume or master['checkpoint'];path=out/a.name;path.mkdir(exist_ok=False);stats=Path(a.stats).resolve() if a.stats else path/'team_stats'
# Retain the complete validated fixed-step argument vector, replacing only
# declared isolated paths/quota/precision/backend and bounded publication.
cmd=json.loads((root/'docs/operations/v2_migration_stability_batch_evidence/integrated/sustained/command.json').read_text());cmd[0]=sys.executable;cmd[1]=str(root/'scripts/run_ddp_ppo.py')
for flag in ['--experimental-exact-token-lookup','--recompute-audit']:
 if flag in cmd:cmd.remove(flag)
if profile['collector'].get('token_lookup_backend')=='exact-row-v100-v1':cmd+=['--experimental-exact-token-lookup']
if a.audit:cmd+=['--recompute-audit']
for flag,value in {'--iterations':a.iterations,'--games':a.games,'--precision-mode':profile['precision']['mode'],'--checkpoint-interval-matches':a.checkpoint_interval,'--experimental-resume':checkpoint,'--experimental-checkpoint':path/'boundary','--rolling-checkpoint-root':path/'rolling','--team-stats-root':stats,'--report':path/'report.json','--experimental-source-migration':path/'migration.json'}.items():
 cmd[cmd.index(flag)+1]=str(value)
if not a.resume:prepare(root,checkpoint,out/'parity.json',stats,path/'migration.json')
else:
 i=cmd.index('--experimental-source-migration');del cmd[i:i+2]
if a.stop:(path/'STOP').touch();cmd+=['--stop-file',str(path/'STOP')]
(path/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (path/'process.log').open('x') as log:
 result=subprocess.run(cmd,env={**os.environ,'PYTHONPATH':f'{root}/engine/python:{root}','OMP_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'},stdout=log,stderr=subprocess.STDOUT)
print(a.name,result.returncode,flush=True);sys.exit(result.returncode)
