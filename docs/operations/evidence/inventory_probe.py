import subprocess,json,pathlib,datetime,os
root=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-integration'); out=root/'docs/operations/evidence'
def run(*args,cwd=root):
 p=subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);return p.stdout.strip() if not p.returncode else {'error':p.stderr.strip(),'exit':p.returncode}
raw=run('git','worktree','list','--porcelain'); trees=[]
for block in raw.split('\n\n'):
 fields=dict(line.split(' ',1) if ' ' in line else (line,'') for line in block.splitlines()); path=pathlib.Path(fields['worktree'])
 if not path.exists():trees.append({**fields,'exists':False});continue
 item={**fields,'exists':True,'status':run('git','status','--short',cwd=path),'ignored':run('git','ls-files','--others','--ignored','--exclude-standard','--directory',cwd=path),'common_dir':run('git','rev-parse','--git-common-dir',cwd=path),'disk':run('du','-sk',str(path))}
 trees.append(item)
refs=[]
for line in run('git','for-each-ref','--format=%(refname) %(objectname) %(upstream)','refs/heads','refs/remotes').splitlines():
 ref,sha,*up=line.split(); unique=run('git','rev-list','--count',ref,'^9e685b7');base=run('git','merge-base',ref,'9e685b7');name=ref.replace('refs/heads/','').replace('refs/remotes/origin/',''); cls='UNIQUE_UNMERGED_WORK' if unique!='0' else 'MERGED_HISTORICAL'
 if name in ['main','codex/pa3-v2-correctness']:cls='CANONICAL_ACTIVE'
 elif name=='research/c-rust-search-teacher':cls='DEFERRED_PROTECTED'
 elif name=='optimization/pa3-v2-ppo-throughput':cls='ACTIVE_RESEARCH'
 refs.append({'ref':ref,'sha':sha,'upstream':up,'merge_base':base,'unique_vs_throughput':unique,'classification':cls,'approval':'NOT_APPROVED','action':'KEEP; artifact reference audit required before retirement'})
result={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_base':'9e685b701f170436f9b9dd82162c8e192e4e6904','trees':trees,'refs':refs,'stashes':run('git','stash','list'),'processes':run('ps','-eo','pid,args'),'units':run('systemctl','--user','list-units','--all','--no-pager','--plain'),'disk':run('df','-h',str(root))}
# Limit processes to project consumers; never collect process environments.
result['processes']='\n'.join(x for x in result['processes'].splitlines() if any(s in x for s in ['run_ddp_ppo','v2_evaluation_sidecar','a10_league','torchrun']))
(out/'INVENTORY_BEFORE.json').write_text(json.dumps(result,indent=2)+'\n')
print('Inventoried',len(trees),'worktrees and',len(refs),'refs')
