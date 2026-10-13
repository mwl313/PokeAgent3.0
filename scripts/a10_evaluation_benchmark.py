#!/usr/bin/env python3
"""Bounded concurrent evaluator tuning; observes PPO, never signals it."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.evaluation.a10_league import read,atomic

def iterations(log):
    return [(int(i),float(s)) for i,s in re.findall(r'iteration (\d+) complete in ([0-9.]+)s',log.read_text())]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args();run=Path(a.run_root);own=run/'evaluation/league_v2';log=run/'report_resume_20261013.json.rank0.log';reg=read(own/'registry.json')['policies']
    entries=sorted(reg.values(),key=lambda e:e['committed_matches']);c=next(e['id'] for e in entries if e['committed_matches']==700416);o=next(e['id'] for e in entries if e['committed_matches']==602112)
    base=iterations(log); baseline=4096*len(base[-8:])/sum(t for i,t in base[-8:]); results=[]
    for games,batch,threads in [(16,8,2),(64,8,2),(64,16,2),(64,32,2),(128,64,2),(256,128,2),(256,32,1)]:
        before=iterations(log);start=time.time()
        cmd=['taskset','-c','16-19','ionice','-c','3','nice','-n','10',sys.executable,str(ROOT/'scripts/a10_evaluation_runner.py'),'--run-root',str(run),'--candidate',c,'--opponent',o,'--pairs',str(games//2),'--batch-size',str(batch),'--threads',str(threads),'--workers','2']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1')
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env);samples=[]
        while proc.poll() is None:
            status=Path(f'/proc/{proc.pid}/status'); sample={'at':time.time()}
            try: sample['VmRSS']=next((x.strip() for x in status.read_text().splitlines() if x.startswith('VmRSS:')),None)
            except OSError:pass
            samples.append(sample);time.sleep(2)
        out,err=proc.communicate();after=iterations(log);new=[x for x in after if x[0]>before[-1][0]+1];gps=4096*len(new)/sum(t for i,t in new) if new else None
        row={'games':games,'batch_size':batch,'threads':threads,'native_workers':2,'cpu_affinity':[16,17,18,19],'wall_s':time.time()-start,'returncode':proc.returncode,'result':json.loads(out) if proc.returncode==0 else None,'stderr':err[-2000:],'training_iterations_fully_inside_window':new,'training_gps':gps,'baseline_gps':baseline,'training_pct_change':100*(gps/baseline-1) if gps else None,'rss_samples':samples}
        results.append(row);atomic(own/'benchmark.json',{'baseline_iterations':base[-8:],'baseline_gps':baseline,'results':results}); print(json.dumps({k:v for k,v in row.items() if k not in ('rss_samples','stderr')},indent=2),flush=True)
        if proc.returncode:break
if __name__=='__main__':main()
