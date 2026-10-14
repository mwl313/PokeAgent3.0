import json,os,pathlib,socket,subprocess,sys
root=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-throughput');out=root/'runs/perf/v2_recovery/ddp_profile_contract';out.mkdir(exist_ok=True)
run=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z')
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
common=['--worker','--port',str(port),'--games','2048','--iterations','1','--seed','20261014',
 '--epochs','2','--experimental-lr-scale','0.5','--precision-mode','D','--precision-stability','fp16-reduction-off',
 '--deterministic-diagnostic','--envs','1024','--workers','16','--microbatch','1024','--minibatch','4096',
 '--data','/home/minidc/Desktop/PokeAgent3.0-V2/engine/data','--teams','/home/minidc/Desktop/PokeAgent3.0-V2/engine/data/a10-training-teams.json',
 '--dataset-manifest','/home/minidc/Desktop/PokeAgent3.0-V2/data/teams/a10-p0-proposed-20261013/manifest.json',
 '--experimental-resume',str(run/'rolling/generations/iter-000000004096-1791958101732995099'),
 '--experimental-source-migration',str(root/'runs/perf/v2_recovery/migration_contract.json'),
 '--experimental-checkpoint',str(out/'boundary'),'--report',str(out/'report.json'),'--timeout','1200']
processes=[];logs=[]
for rank in (0,1):
    env={**os.environ,'PYTHONPATH':f'{root}/engine/python:{root}','RANK':str(rank),'LOCAL_RANK':str(rank),'WORLD_SIZE':'2',
      'NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'0','NCCL_SOCKET_IFNAME':'lo','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    cmd=['numactl',f'--cpunodebind={rank}',f'--membind={rank}','--',sys.executable,str(root/'runs/perf/v2_recovery/ddp_profile_worker.py'),*common]
    (out/f'command.rank{rank}.json').write_text(json.dumps(cmd,indent=2)+'\n')
    log=(out/f'process.rank{rank}.log').open('x');logs.append(log)
    processes.append(subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT))
try:
    codes=[p.wait(timeout=1200) for p in processes]
    if any(codes):raise RuntimeError(f'profile worker failed: {codes}')
finally:
    for p in processes:
        if p.poll() is None:p.terminate()
    for log in logs:log.close()
def union(events):
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in events)
    if not intervals:return 0
    total=0;left,right=intervals[0]
    for a,b in intervals[1:]:
        if a>right:total+=right-left;left,right=a,b
        else:right=max(right,b)
    return (total+right-left)/1000
summaries=[]
for rank in (0,1):
    trace=json.loads((out/f'trace.rank{rank}.json').read_text())['traceEvents']
    kernels=[e for e in trace if e.get('cat')=='kernel'];nccl=[e for e in kernels if 'nccl' in e['name'].lower()]
    compute=[e for e in kernels if e not in nccl]
    summaries.append({'rank':rank,'leaf_kernel_count':len(kernels),'nccl_kernel_count':len(nccl),
      'kernel_union_ms':union(kernels),'nccl_union_ms':union(nccl),'compute_union_ms':union(compute),
      'overlap_ms':union(nccl)+union(compute)-union(kernels),
      'nccl_names':sorted(set(e['name'] for e in nccl))})
reports=[json.loads((out/f'report.json.rank{r}.json').read_text()) for r in (0,1)]
result={'scope':'one actual two-rank PPO iteration, one post-warmup optimizer step profiled; profiler overhead not throughput evidence',
 'ranks':summaries,'model_digest_equal':reports[0]['model_digest']==reports[1]['model_digest'],
 'optimizer_digest_equal':reports[0]['optimizer_digest']==reports[1]['optimizer_digest'],
 'gates':[r['recompute']['within_gate'] for r in reports],
 'note':'NCCL kernel spans include waiting/polling and overlap; not pure transfer time or bandwidth.'}
(out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
