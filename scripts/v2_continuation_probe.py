"""Two iterations versus one/process exit/restore/one; isolated checkpoint roots."""
import argparse,json,os,pathlib,subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--migration',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1];out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    profile=json.loads(pathlib.Path(a.profile).read_text());data=profile['dataset']
    common=['--games','2048','--seed','20261014','--epochs','2','--experimental-lr-scale','0.5',
      '--precision-mode','D','--precision-stability','fp16-reduction-off','--deterministic-diagnostic',
      '--envs','1024','--workers','16','--microbatch','1024','--minibatch','4096',
      '--data',data['data'],'--teams',data['teams'],'--dataset-manifest',data['manifest'],
      '--checkpoint-interval-matches','4096','--timeout','1800']
    def run(name,iterations,checkpoint,migration=None):
        path=out/name;path.mkdir()
        command=[sys.executable,str(root/'scripts/run_ddp_ppo.py'),*common,
          '--iterations',str(iterations),'--experimental-resume',str(checkpoint),
          '--experimental-checkpoint',str(path/'boundary'),'--rolling-checkpoint-root',str(path/'rolling'),
          '--report',str(path/'report.json')]
        if migration:command+=['--experimental-source-migration',migration]
        (path/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (path/'process.log').open('x') as log:
            result=subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':f'{root}/engine/python:{root}'},stdout=log,stderr=subprocess.STDOUT,timeout=2400)
        if result.returncode:raise RuntimeError(f'{name} failed; refusing continuation')
        latest=(path/'rolling/LATEST').read_text().strip()
        return path/'rolling/generations'/latest
    full=run('uninterrupted',2,a.checkpoint,a.migration)
    first=run('split_first',1,a.checkpoint,a.migration)
    restored=run('split_restored',1,first)
    from agent.ppo.experimental_checkpoint import load_envelope
    from scripts.run_ddp_ppo import _compact_digest
    result={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'ranks':[]}
    for rank in (0,1):
        left=load_envelope(str(full/f'rank{rank}.pt'));right=load_envelope(str(restored/f'rank{rank}.pt'))
        checks={k:_compact_digest(left[k])==_compact_digest(right[k]) for k in
           ['learner','rng','accounting','collector_restart','rank_digests','dataset_manifest_hash','opponent_manifest']}
        continuous=json.loads((out/'uninterrupted/report.json').read_text())['ranks'][rank]['iterations'][-1]
        resumed=json.loads((out/'split_restored/report.json').read_text())['ranks'][rank]['iterations'][-1]
        checks['rollout_digest']=continuous['parity']['rollout_digest']==resumed['parity']['rollout_digest']
        checks['outcomes_rows_steps']=all(continuous[k]==resumed[k] for k in ['rows','games','optimizer_steps_cumulative']) and all(
          continuous['collector_stats'][k]==resumed['collector_stats'][k] for k in ['wins','losses','draws','operational_errors'])
        result['ranks'].append({'rank':rank,'checks':checks,'matches':left['accounting']['committed_matches']})
    result['classification']='BIT_EXACT' if all(all(r['checks'].values()) for r in result['ranks']) else 'NOT_REPRODUCIBLE'
    result['scope']='bounded same-source/same-runtime static collector; no campaign team-ledger writes; no cross-hardware guarantee'
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['classification']!='BIT_EXACT':raise RuntimeError('bounded continuation differs')

if __name__=='__main__':main()
