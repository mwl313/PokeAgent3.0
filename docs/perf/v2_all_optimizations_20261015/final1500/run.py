"""500-game weights-only diagnostic: no checkpoint continuation/publication."""
import hashlib,json,os,pathlib,socket,subprocess,sys,time
ROOT=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-throughput')
OUT=pathlib.Path(__file__).resolve().parent
GEN=ROOT/'runs/perf/v2_recovery/reconstruction_contract/reconstructed/rolling/generations/iter-000000057344-1791986684864852699'
CAM=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if '--diagnostic-worker' in sys.argv:
    sys.path[:0]=[str(ROOT),str(ROOT/'engine/python')]
    import agent.model as models
    import torch
    from agent.contracts import require_checkpoint_identity,OBSERVATION_SCHEMA,ACTION_SCHEMA,CONTRACT_HASH
    from agent.ppo.experimental_checkpoint import load_envelope,hash_json
    import scripts.run_ddp_ppo as launcher
    manifest=json.loads((GEN/'manifest.json').read_text());rank=int(os.environ['RANK']);path=GEN/f'rank{rank}.pt'
    assert sha(path)==manifest['ranks'][str(rank)]['sha256']
    frozen=load_envelope(str(path));require_checkpoint_identity(frozen['learner'],'D')
    assert frozen['source_sha']=='5ccc51b0cd4a49efb93018258b9750d5e804ae32'
    assert frozen['dataset_manifest_hash']==sha('/home/minidc/Desktop/PokeAgent3.0-V2/data/teams/a10-p0-proposed-20261013/manifest.json')
    assert frozen['action_schema_hash']==hash_json({'observation':OBSERVATION_SCHEMA,'actions':ACTION_SCHEMA,'candidates':'branch-candidates-v2','semantic_hash':CONTRACT_HASH})
    original=models.build_model
    def build(config,*args,**kwargs):
        assert config.to_dict()==frozen['learner']['config']['model']
        model=original(config,*args,**kwargs);model.load_state_dict(frozen['learner']['model'],strict=True)
        assert not model.scorer.experimental_row_lookup
        digest,_=launcher.state_digest(model,torch.optim.Adam(model.parameters()))
        assert digest==frozen['rank_digests'][str(rank)]['model']
        return model
    models.build_model=build
    args=launcher.parse_args(sys.argv[sys.argv.index('--diagnostic-worker')+1:])
    assert args.games==750 and args.envs==250 and args.iterations==1 and not args.experimental_resume and not args.checkpoint and args.experimental_checkpoint==str(OUT/'diagnostic-never-due') and not args.team_stats_root
    launcher.worker(args)
else:
    assert not (OUT/'SUMMARY.json').exists()
    active=subprocess.run(['systemctl','--user','is-active','pa3-v2-ppo-20261014-060525.service'],capture_output=True,text=True)
    if active.stdout.strip()=='active':raise RuntimeError('production occupies GPUs')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('GPU workloads already active')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    common=['--worker','--games','250','--envs','250','--workers','16','--iterations','1','--epochs','2','--microbatch','1024','--minibatch','4096','--seed','20261014','--precision-mode','D','--precision-stability','fp16-reduction-off','--deterministic-diagnostic','--experimental-lr-scale','0.5','--experimental-checkpoint',str(OUT/'diagnostic-never-due'),'--port',str(port),'--data','/home/minidc/Desktop/PokeAgent3.0-V2/engine/data','--teams','/home/minidc/Desktop/PokeAgent3.0-V2/engine/data/a10-training-teams.json','--dataset-manifest','/home/minidc/Desktop/PokeAgent3.0-V2/data/teams/a10-p0-proposed-20261013/manifest.json','--report',str(OUT/'report.json'),'--timeout','600']
    common.append('--experimental-exact-token-lookup')
    common[common.index('--games')+1]='750'
    children=[];logs=[];commands=[];started=time.perf_counter()
    for rank in (0,1):
        env={**os.environ,'PYTHONPATH':f'{ROOT}/engine/python:{ROOT}','CUBLAS_WORKSPACE_CONFIG':':4096:8','RANK':str(rank),'LOCAL_RANK':str(rank),'WORLD_SIZE':'2','NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'0','NCCL_SOCKET_IFNAME':'lo','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        command=['numactl',f'--cpunodebind={rank}',f'--membind={rank}','--',sys.executable,str(pathlib.Path(__file__).resolve()),'--diagnostic-worker',*common];commands.append(command)
        log=(OUT/f'process.rank{rank}.log').open('x');logs.append(log);children.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
    try:
        codes=[p.wait(timeout=700) for p in children]
        if any(codes):raise RuntimeError(f'worker failed: {codes}; no speed acceptance')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for log in logs:log.close()
    wall=time.perf_counter()-started
    ranks=[json.loads((OUT/f'report.json.rank{r}.json').read_text()) for r in (0,1)]
    games=sum(r['games'] for r in ranks);assert games==1500
    assert ranks[0]['model_digest']==ranks[1]['model_digest'] and ranks[0]['optimizer_digest']==ranks[1]['optimizer_digest']
    assert all(r['recompute']['within_gate'] and r['operational_errors']==0 and r['report']['optimizer_steps_skipped']==0 for r in ranks)
    critical=max(ranks,key=lambda r:r['iteration_wall_s'])
    result={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'diagnostic_wrapper_sha256':sha(__file__),'commands':commands,'games':games,'rows':sum(r['rows'] for r in ranks),'launcher_wall_s':wall,'all_in_games_s':games/wall,'iteration_wall_s':critical['iteration_wall_s'],'iteration_games_s':games/critical['iteration_wall_s'],'collect_and_wait_s':critical['collect_wall_s']+critical['lr_clock_wall_s'],'prepare_s':critical['prepare_wall_s'],'update_s':critical['update_wall_s'],'optimizer_steps':critical['optimizer_steps_this_iteration'],'epochs':[r['report']['epochs_run'] for r in ranks],'max_recompute':max(r['recompute']['max_abs_diff'] for r in ranks),'model_and_optimizer_rank_parity':True,'errors':0,'skips':0,'scope':'500 natural games, two ranks, envs250/rank, weights-only frozen57344 policy, fresh diagnostic Adam/clock; not checkpoint continuation or steady-state production speed; row lookup OFF; micro1024 unchanged; no campaign writes/checkpoints'}
    result['scope']=result['scope'].replace('row lookup OFF','exact-order row lookup ON')
    result['scope']=result['scope'].replace('500 natural games','1500 natural games')
    (OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
