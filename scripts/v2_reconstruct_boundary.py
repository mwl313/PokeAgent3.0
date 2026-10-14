"""Recover lost recent learner state by bounded, digest-verified original-source replay.

Never starts production or overwrites its generations. Appended team receipts
are immutable, uncommitted orphans until selected by a separate new manifest.
Stop after one probe if the historical model/Adam cannot be reproduced.
"""
import argparse,json,os,pathlib,subprocess,sys,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();campaign=pathlib.Path(a.campaign).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    source=pathlib.Path('/home/minidc/.codex/worktrees/pa3-v2-runtime/PokeAgent3.0')
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!='5ccc51b0cd4a49efb93018258b9750d5e804ae32':raise RuntimeError('original source changed')
    active=subprocess.run(['systemctl','--user','is-active','pa3-v2-ppo-20261014-060525.service'],capture_output=True,text=True)
    if active.stdout.strip()=='active':raise RuntimeError('production occupies GPUs')
    initial=campaign/'rolling/generations/iter-000000004096-1791958101732995099'
    expected={}
    for rank in (0,1):
        for line in (campaign/f'reports/continuation.json.rank{rank}.ppo.jsonl').read_text().splitlines():
            row=json.loads(line);expected[row['identity']['committed_matches'],rank]=row
    def execute(name,iterations,probe=False):
        path=out/name;path.mkdir();stop=path/'STOP'
        if probe:stop.touch()
        command=[sys.executable,str(source/'scripts/run_ddp_ppo.py'),
          '--profile',str(campaign/'release/runtime_profile.json'),'--games','2048','--iterations',str(iterations),
          '--seed','20261014','--experimental-resume',str(initial),'--stop-file',str(stop),
          '--experimental-checkpoint',str(path/'boundary'),'--rolling-checkpoint-root',str(path/'rolling'),
          '--team-stats-root',str(campaign/'team_stats'),'--report',str(path/'report.json'),'--timeout','1800']
        (path/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (path/'process.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,
               env={**os.environ,'PYTHONPATH':f'{source}/engine/python:{source}','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
            mismatch=False;began=time.monotonic()
            while proc.poll() is None:
                # Final update finishes at73,728; request checkpoint before its
                # post-update stop-file poll, not after process exit.
                ranklog=path/'report.json.rank0.log'
                if not probe and ranklog.exists() and f'iteration {iterations}/{iterations}: collected' in ranklog.read_text():stop.touch(exist_ok=True)
                for rank in (0,1):
                    telemetry=path/f'report.json.rank{rank}.ppo.jsonl'
                    if not telemetry.exists():continue
                    for line in telemetry.read_bytes().splitlines(keepends=True):
                        if not line.endswith(b'\n'):continue
                        value=json.loads(line);clock=value['identity']['committed_matches'];old=expected.get((clock,rank))
                        if old is None or any(value['safety']['rank_verification'][field]!=old['safety']['rank_verification'][field] for field in ['model_digest','optimizer_digest']):
                            mismatch=True;stop.touch(exist_ok=True)
                if time.monotonic()-began>3600:stop.touch(exist_ok=True);raise TimeoutError('bounded reconstruction exceeded one hour; cooperative stop requested')
                time.sleep(1)
        if proc.returncode:raise RuntimeError('original replay failed')
        latest=(path/'rolling/LATEST').read_text().strip();manifest=json.loads((path/'rolling/generations'/latest/'manifest.json').read_text())
        return dict(matched_historical_digests=not mismatch,matches=manifest['committed_matches'],checkpoint=str(path/'rolling/generations'/latest))
    result={'execution_environment':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},'probe':execute('probe',1,True)}
    if result['probe']['matched_historical_digests']:
        result['reconstructed']=execute('reconstructed',17)
        result['status']='RECOVERED_MODEL_AND_ADAM' if result['reconstructed']['matched_historical_digests'] and result['reconstructed']['matches']==73728 else 'HISTORICAL_REPLAY_DIFFERS'
    else:result['status']='HISTORICAL_REPLAY_DIFFERS'
    result['limitations']='Original recent RNG payload is unavailable for direct comparison; state is replayed from owned initial RNG. This is not a production restart or a new campaign.'
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
