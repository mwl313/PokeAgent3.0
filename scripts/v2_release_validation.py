"""Isolated full-state migration and continuity acceptance; never writes campaign."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from agent.ppo.v2_release_migration import prepare,BASE,blobs
from agent.ppo.experimental_checkpoint import load_envelope

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--parity-report',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    campaign=Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z')
    profile=json.loads((campaign/'release/runtime_profile.json').read_text());data=profile['dataset']
    common=['--games','2048','--seed','20261014','--epochs','2','--experimental-lr-scale','0.5','--precision-mode','D','--precision-stability','fp16-reduction-off','--deterministic-diagnostic','--recompute-secondary','--envs','1024','--workers','16','--microbatch','1024','--minibatch','4096','--data',data['data'],'--teams',data['teams'],'--dataset-manifest',data['manifest'],'--rolling-keep-generations','20','--timeout','1800']
    def run(name,count,checkpoint,migrate=False,interval=4096):
        path=out/name;path.mkdir(exist_ok=False)
        command=[sys.executable,str(root/'scripts/run_ddp_ppo.py'),*common,'--iterations',str(count),'--experimental-resume',str(checkpoint),'--checkpoint-interval-matches',str(interval),'--experimental-checkpoint',str(path/'boundary'),'--rolling-checkpoint-root',str(path/'rolling'),'--team-stats-root',str(out/('first' if name=='resumed' else name)/'team_stats'),'--report',str(path/'report.json')]
        if migrate:
            receipt=path/'migration.json';prepare(root,checkpoint,args.parity_report,path/'team_stats',receipt);command+=['--experimental-source-migration',str(receipt)]
        (path/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (path/'process.log').open('x') as log:
            result=subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':f'{root}/engine/python:{root}','OMP_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'},stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' failed; refusing cutover')
        pointer=path/'rolling/LATEST';return pointer.parent/'generations'/pointer.read_text().strip()
    # Smoke is also split-arm iteration15. Uninterrupted arm crosses15–20.
    first=run('first',1,args.checkpoint,True)
    full=run('uninterrupted',6,args.checkpoint,True,8192)
    resumed=run('resumed',1,first)
    # Compare uninterrupted iteration16 with fresh communicator resumed16.
    reference=next(p for p in (out/'uninterrupted/rolling/generations').iterdir() if p.name.startswith('iter-000000065536-'))
    from scripts.run_ddp_ppo import _compact_digest
    checks={}
    for rank in (0,1):
        a=load_envelope(str(reference/f'rank{rank}.pt'));b=load_envelope(str(resumed/f'rank{rank}.pt'))
        checks[str(rank)]={k:_compact_digest(a[k])==_compact_digest(b[k]) for k in ['learner','rng','accounting','collector_restart','rank_digests','dataset_manifest_hash','opponent_manifest']}
        reports=[json.loads((out/name/f'report.json.rank{rank}.json').read_text())['iterations'] for name in ['uninterrupted','resumed']]
        checks[str(rank)]['rollout_digest']=reports[0][1]['parity']['rollout_digest']==reports[1][0]['parity']['rollout_digest']
    summary={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'classification':'BIT_EXACT' if all(all(v.values()) for v in checks.values()) else 'NOT_REPRODUCIBLE','checks':checks,'final_checkpoint':str(full),'diagnostic_matches':24576,'final_diagnostic_clock':81920,'production_matches_added':0,'scope':'same-source,hardware/runtime,full4096-game iterations; subprocess exit/recreated DDP and restored state; not universal reproducibility'}
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
    if summary['classification']!='BIT_EXACT':raise RuntimeError('continuation differs; refusing cutover')
if __name__=='__main__':main()
