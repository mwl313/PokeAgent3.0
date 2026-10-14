"""Bounded source-pinned two-V100 experiments; never manages production services."""
import argparse,json,os,pathlib,subprocess,time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--migration');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--iterations',type=int,default=5)
    a=p.parse_args();root=pathlib.Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if not 1<=a.repeats<=3 or not 1<=a.iterations<=5:raise ValueError('bounded benchmark budget exceeded')
    # A benchmark must never compete with training, even if called manually.
    active=subprocess.run(['systemctl','--user','is-active','pa3-v2-ppo-20261014-060525.service'],capture_output=True,text=True)
    if active.stdout.strip()=='active':raise RuntimeError('production occupies GPUs')
    source=subprocess.check_output(['git','-C',a.source,'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',a.source,'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:raise RuntimeError('benchmark requires clean committed runtime')
    receipts=[]
    # Separate experiment namespace: no writes to campaign team receipts. The
    # production wrapper requires its original path-bound ledger. All training,
    # collector, dataset and precision values remain byte-for-byte equivalent.
    experiment_profile=json.loads(pathlib.Path(a.profile).read_text())
    profile=root/'benchmark_profile.json'
    profile.write_text(json.dumps(experiment_profile,indent=2)+'\n')
    for i in range(a.repeats):
        report=root/f'repeat{i+1}.json'
        if report.exists() or report.with_suffix('.log').exists():raise FileExistsError(report)
        command=[os.sys.executable,str(pathlib.Path(a.source)/'scripts/run_ddp_ppo.py'),
            '--games','2048','--iterations',str(a.iterations),
            '--epochs','2','--experimental-lr-scale','0.5',
            '--precision-mode','D','--precision-stability','fp16-reduction-off',
            '--deterministic-diagnostic','--envs','1024','--workers','16',
            '--microbatch','1024','--minibatch','4096',
            '--data',experiment_profile['dataset']['data'],
            '--teams',experiment_profile['dataset']['teams'],
            '--dataset-manifest',experiment_profile['dataset']['manifest'],
            '--seed','20261014','--experimental-resume',a.checkpoint,
            '--experimental-checkpoint',str(root/f'repeat{i+1}-boundary'),
            '--report',str(report),'--timeout','1800']
        if a.migration:command+=['--experimental-source-migration',a.migration]
        started=time.time()
        with report.with_suffix('.log').open('x') as log:
            result=subprocess.run(command,cwd=a.source,stdout=log,stderr=subprocess.STDOUT,
                                  env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                                       'PYTHONPATH':f'{a.source}/engine/python:{a.source}'},timeout=2400)
        receipt=dict(source_sha=source,command=command,execution_environment={'CUBLAS_WORKSPACE_CONFIG':':4096:8'},returncode=result.returncode,wall_s=time.time()-started)
        receipts.append(receipt);(root/'commands.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        if result.returncode:raise RuntimeError('benchmark failed; inspect logs before continuing')

if __name__=='__main__':main()
