"""Bounded diagnostic only: original launcher and learning math are unchanged."""
import collections, functools, gzip, hashlib, importlib.util, json, os, pathlib, sys, time
from contextlib import contextmanager

ROOT = pathlib.Path('/home/minidc/Desktop/PokeAgent3.0')
sys.path[:0] = [str(ROOT),str(ROOT/'engine/python')]
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from agent.ppo.learner import PPOLearner
from agent.train.native_collector import NativeCollector
from agent.model.encoder import TokenEncoder
from agent.model.scorer import ConditionalPrefixScorer

spec = importlib.util.spec_from_file_location('original_launcher',ROOT/'scripts/run_ddp_ppo.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.__file__ = __file__  # parent launches this diagnostic worker, same original args
STATE = {'enabled':False,'prof':None,'events':[],'collect_count':0,'update_count':0}

@contextmanager
def scope(name):
    if not STATE['enabled']:
        yield; return
    prof = STATE['prof']
    active = prof is not None and prof.step_num in (2,3)
    start = torch.cuda.Event(enable_timing=True) if active else None
    end = torch.cuda.Event(enable_timing=True) if active else None
    with record_function('pa3/'+name):
        if start is not None: start.record()
        cpu = time.perf_counter()
        try: yield
        finally:
            if end is not None:
                end.record(); STATE['events'].append((name,start,end,time.perf_counter()-cpu))

def annotate(cls, method, label):
    original = getattr(cls,method)
    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        with scope(label): return original(*args,**kwargs)
    setattr(cls,method,wrapped)

def union(intervals):
    total = 0.; end = float('-inf')
    for left,right in sorted(intervals):
        total += max(0.,right-max(left,end)); end=max(end,right)
    return total

def kernel_group(name):
    s=name.lower()
    if 'nccl' in s: return 'NCCL'
    if 'fmha' in s or 'attention' in s: return 'attention'
    if 'gemm' in s or 'cutlass' in s: return 'GEMM'
    if 'embedding' in s: return 'embedding'
    if any(x in s for x in ('gather','scatter','index')): return 'gather_scatter_index'
    if any(x in s for x in ('layer_norm','layernorm','softmax','reduce','normkernel')): return 'reduce_norm_softmax'
    if any(x in s for x in ('copy','cast','convert')): return 'copy_cast'
    if any(x in s for x in ('elementwise','vectorized','unrolled','fill','gelu')): return 'elementwise'
    return 'other'

def summarize_trace(prof,kind,args):
    rank=os.environ.get('RANK','parent')
    base=pathlib.Path(str(args.report)+f'.rank{rank}.{kind}')
    trace_path=str(base)+'.trace.json'
    prof.export_chrome_trace(trace_path)
    trace=json.loads(pathlib.Path(trace_path).read_text())
    events=trace['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('dur',0)>0]
    copies=[e for e in events if e.get('cat') in ('gpu_memcpy','gpu_memset') and e.get('dur',0)>0]
    annotations=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('pa3/')]
    by_thread=collections.defaultdict(list)
    for a in annotations: by_thread[(a.get('pid'),a.get('tid'))].append(a)
    owners={}
    for e in events:
        if e.get('cat')!='cpu_op': continue
        external=e.get('args',{}).get('External id')
        if external is None: continue
        enclosing=[a for a in by_thread[(e.get('pid'),e.get('tid'))]
                   if a['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=a['ts']+a.get('dur',0)+.01]
        if enclosing: owners[external]=min(enclosing,key=lambda a:a['dur'])['name']
    groups=collections.defaultdict(list); phases=collections.defaultdict(list); names=collections.defaultdict(list)
    for e in kernels:
        interval=(e['ts'],e['ts']+e['dur'])
        groups[kernel_group(e['name'])].append(interval)
        phases[owners.get(e.get('args',{}).get('External id'),'unattributed')].append(interval)
        names[e['name']].append(interval)
    def rows(mapping):
        return sorted([{'name':name,'calls':len(v),'sum_ms':sum(b-a for a,b in v)/1000,
                        'union_ms':union(v)/1000} for name,v in mapping.items()],key=lambda r:-r['sum_ms'])
    all_intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
    compute=[(e['ts'],e['ts']+e['dur']) for e in kernels if kernel_group(e['name'])!='NCCL']
    comm=groups.get('NCCL',[])
    span=max((b for a,b in all_intervals),default=0)-min((a for a,b in all_intervals),default=0)
    total=union(all_intervals); overlap=union(compute)+union(comm)-total
    payload={'kind':kind,'rank':int(rank),'captured_steps':2,'kernel_union_ms':total/1000,
             'kernel_span_ms':span/1000,'kernel_busy_fraction':total/max(span,1e-9),
             'compute_union_ms':union(compute)/1000,'nccl_union_ms':union(comm)/1000,
             'compute_nccl_overlap_ms':max(0,overlap)/1000,
             'nccl_without_compute_ms':max(0,union(comm)-overlap)/1000,
             'kernel_groups':rows(groups),'gpu_launch_owner_groups':rows(phases),
             'top_leaf_kernels':rows(names)[:35],
             'memcpy_events':[{'name':e['name'],'duration_us':e['dur'],'args':e.get('args',{})} for e in copies],
             'top_cpu_ops':sorted([{'name':e.key,'calls':e.count,'cpu_total_ms':e.cpu_time_total/1000,
                                   'cpu_self_ms':e.self_cpu_time_total/1000}
                                  for e in prof.key_averages()],key=lambda x:-x['cpu_self_ms'])[:40],
             'notes':'GPU category unions can overlap. NCCL time includes peer wait and overlap; not pure network cost. CPU self excludes children but can include GPU blocking. Scope is two early collection rounds or two PPO minibatches, not the whole iteration.',
             'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    base.with_suffix(base.suffix+'.json').write_text(json.dumps(payload,indent=2)+'\n')
    with open(trace_path,'rb') as source,gzip.open(str(base)+'.trace.json.gz','wb') as dest:
        import shutil; shutil.copyfileobj(source,dest)
    pathlib.Path(trace_path).unlink()

def make_prof(kind,args):
    return profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        schedule=schedule(wait=1,warmup=1,active=2,repeat=1),
        on_trace_ready=lambda p:summarize_trace(p,kind,args),record_shapes=True,
        profile_memory=False,with_stack=False)

annotate(PPOLearner,'_forward_terms','forward')
annotate(PPOLearner,'_plan_minibatch','minibatch_select')
annotate(TokenEncoder,'forward','encoder')
for method in ('build_keys','score_keys','advance_keys'): annotate(ConditionalPrefixScorer,method,'scorer/'+method)
annotate(torch.optim.Adam,'step','optimizer')
original_backward=torch.autograd.backward
def backward(*args,**kwargs):
    with scope('backward'): return original_backward(*args,**kwargs)
torch.autograd.backward=backward
import agent.ppo.ddp as ddp
for method in ('all_reduce_tensor','all_reduce_flag'): annotate(ddp,method,'collective/'+method)

original_round=NativeCollector._collect_round
def collect_round(*args,**kwargs):
    result=original_round(*args,**kwargs)
    if STATE['enabled'] and STATE['prof'] is not None and STATE.get('kind')=='collect': STATE['prof'].step()
    return result
NativeCollector._collect_round=collect_round

def export_events(kind,args):
    torch.cuda.synchronize()
    groups=collections.defaultdict(lambda:{'calls':0,'cuda_event_ms':0.,'cpu_wall_s':0.})
    for name,a,b,wall in STATE['events']:
        row=groups[name]; row['calls']+=1; row['cuda_event_ms']+=a.elapsed_time(b); row['cpu_wall_s']+=wall
    pathlib.Path(str(args.report)+f'.rank{os.environ["RANK"]}.{kind}.events.json').write_text(
        json.dumps({'scopes':groups,'note':'Nested scopes overlap. Events cover current-stream elapsed time including dependencies; only active sample steps.'},indent=2)+'\n')
    STATE['events']=[]

original_collect=NativeCollector.collect
def collect(self,*a,**kw):
    STATE['collect_count']+=1
    if STATE['collect_count']!=1: return original_collect(self,*a,**kw)
    self.config.telemetry=True
    STATE.update(enabled=True,kind='collect',events=[])
    with make_prof('collect',ARGS) as prof:
        STATE['prof']=prof
        result=original_collect(self,*a,**kw)
    STATE.update(enabled=False,prof=None)
    export_events('collect',ARGS)
    pathlib.Path(str(ARGS.report)+f'.rank{os.environ["RANK"]}.collector_tail.json').write_text(json.dumps(self.telemetry,indent=2)+'\n')
    return result
NativeCollector.collect=collect

original_update=PPOLearner.update_ddp
def update(self,*a,**kw):
    STATE['update_count']+=1
    if STATE['update_count']!=1: return original_update(self,*a,**kw)
    original_step=self.optimizer.step
    def step(*sa,**sk):
        result=original_step(*sa,**sk)
        STATE['prof'].step()
        return result
    self.optimizer.step=step
    STATE.update(enabled=True,kind='update',events=[])
    try:
        with make_prof('update',ARGS) as prof:
            STATE['prof']=prof
            result=original_update(self,*a,**kw)
    finally:
        self.optimizer.step=original_step
        STATE.update(enabled=False,prof=None)
    export_events('update',ARGS)
    return result
PPOLearner.update_ddp=update

ARGS=launcher.parse_args()

# Isolate scalar-control latency and one full model-gradient-sized all-reduce.
# This is diagnostic traffic after the verified PPO iterations, not training.
if ARGS.worker:
    import torch.distributed as dist
    original_destroy=dist.destroy_process_group
    def destroy(*a,**kw):
        if not a and not kw and dist.is_initialized():
            from agent.model import PA3Config,build_model
            parameters=sum(p.numel() for p in build_model(PA3Config()).parameters())
            results=[]
            for label,count in [('one_float',1),('two_floats',2),('model_gradient',parameters)]:
                tensor=torch.zeros(count,device=torch.cuda.current_device(),dtype=torch.float32)
                for _ in range(5): dist.all_reduce(tensor)
                dist.barrier(); torch.cuda.synchronize()
                timings=[]; started=time.perf_counter()
                for _ in range(20):
                    begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    begin.record(); dist.all_reduce(tensor); end.record(); timings.append((begin,end))
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                ms=[begin.elapsed_time(end) for begin,end in timings]
                results.append({'label':label,'payload_bytes':count*4,'repeats':20,'total_wall_s':elapsed,
                                'mean_cuda_ms':sum(ms)/len(ms),'min_cuda_ms':min(ms),'max_cuda_ms':max(ms),
                                'payload_GB_per_s':count*4/(sum(ms)/len(ms)/1000)/1e9})
            pathlib.Path(str(ARGS.report)+f'.rank{os.environ["RANK"]}.nccl_micro.json').write_text(json.dumps(
                {'results':results,'transport_env':{k:os.environ.get(k) for k in ('NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_SOCKET_IFNAME')},
                 'note':'Isolated repeated SUM, not a deduction from production wall. CUDA events measure current-stream dependencies.'},indent=2)+'\n')
        return original_destroy(*a,**kw)
    dist.destroy_process_group=destroy

if __name__=='__main__':
    if ARGS.worker: launcher.worker(ARGS)
    else: launcher.parent(ARGS)
