"""Inference/loss/backward profiler on preserved real V2 inputs; no games."""
import argparse,dataclasses,json,pathlib,time
import torch
from agent.buffer.rollout_buffer import RolloutBuffer
from agent.model import build_model,PA3Config
from agent.ppo import PPOLearner,PPOConfig
from agent.ppo.experimental_checkpoint import load_envelope

def union(intervals):
    if not intervals:return 0.
    total=0.;start,end=sorted(intervals)[0]
    for left,right in sorted(intervals)[1:]:
        if left>end:total+=end-start;start,end=left,right
        else:end=max(end,right)
    return total+end-start

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rows',type=int,default=1024);a=p.parse_args()
    if not 1<=a.rows<=1024:raise ValueError('bounded GPU profile exceeds micro1024')
    root=pathlib.Path(a.output);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    e=load_envelope(str(pathlib.Path(a.artifact).resolve()));buffer=RolloutBuffer()
    for i in range(a.rows):
        j=i%len(e['sampled_rows']);ref=buffer.observation_store.add(e['sampled_observations'][j])
        buffer.rows.append(dataclasses.replace(e['sampled_rows'][j],observation_ref=ref))
    h2d_start=time.perf_counter();batch=buffer.to_batch(device='cuda:0');torch.cuda.synchronize();h2d=time.perf_counter()-h2d_start
    batch.raw_advantages=torch.linspace(-1,1,a.rows,device='cuda:0');batch.returns=torch.linspace(-.5,.5,a.rows,device='cuda:0')
    model=build_model(PA3Config(**e['model_config']),device='cuda:0');model.load_state_dict(e['model']);model.configure_precision('D')
    learner=PPOLearner(model,PPOConfig(ppo_epochs=2),device='cuda:0')
    def step():
        learner.optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function('V2_FORWARD_LOSS'):
            torch.cuda.nvtx.range_push('V2_FORWARD_LOSS');terms=learner._forward_terms(batch);torch.cuda.nvtx.range_pop()
            loss=terms['loss_unscaled']+learner.config.value_coefficient*terms['value_mean']
        with torch.profiler.record_function('V2_BACKWARD'):
            torch.cuda.nvtx.range_push('V2_BACKWARD');learner.scaler.scale(loss).backward();torch.cuda.nvtx.range_pop()
        with torch.profiler.record_function('V2_CLIP_ADAM'):
            learner.scaler.unscale_(learner.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),learner.config.max_grad_norm)
            learner.scaler.step(learner.optimizer);learner.scaler.update()
    for _ in range(2):step()
    torch.cuda.synchronize();events=[];host=[]
    for _ in range(3):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        began=time.perf_counter();start.record();step();end.record();end.synchronize()
        events.append(start.elapsed_time(end));host.append(1000*(time.perf_counter()-began))
    # Nested module CPU scopes are inclusive and must not be added together.
    handles=[];scopes={}
    names=['encoder.embedding','encoder.embedding.ragged_embedding','value_head']+[f'encoder.blocks.{i}' for i in range(6)]
    for name,module in model.named_modules():
        if name not in names:continue
        def before(module,args,key=name):
            scope=torch.profiler.record_function('V2_MODULE_'+key);scope.__enter__();scopes[key]=scope
        def after(module,args,output,key=name):scopes.pop(key).__exit__(None,None,None)
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    for name in ['build_keys','score_keys','advance_keys','token_references','action_features']:
        original=getattr(model.scorer,name)
        def wrapped(*args,_name=name,_original=original,**kwargs):
            with torch.profiler.record_function('V2_SCORER_'+_name):
                torch.cuda.nvtx.range_push('V2_SCORER_'+_name)
                try:return _original(*args,**kwargs)
                finally:torch.cuda.nvtx.range_pop()
        setattr(model.scorer,name,wrapped)
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        step();torch.cuda.synchronize()
    for handle in handles:handle.remove()
    trace=root/'trace.json';prof.export_chrome_trace(str(trace));raw=json.loads(trace.read_text())['traceEvents']
    kernels=[e for e in raw if e.get('cat')=='kernel' and e.get('dur',0)>0]
    grouped={}
    for event in kernels:grouped[event['name']]=grouped.get(event['name'],0)+event['dur']/1000
    intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
    runtime=[e for e in raw if e.get('cat')=='cuda_runtime' and e.get('dur',0)>0]
    sync=[e for e in runtime if 'Synchronize' in e.get('name','') or e.get('name')=='cudaMemcpy']
    ops=[{'name':e.key,'cpu_total_ms':e.cpu_time_total/1000,'self_cpu_ms':e.self_cpu_time_total/1000,
          'device_total_ms':getattr(e,'device_time_total',0)/1000,'count':e.count} for e in prof.key_averages()]
    result={'rows':a.rows,'scope':'single GPU captured-row synthetic-loss profile; no DDP; duplicates to requested micro size',
       'cuda_event_step_ms':events,'synchronized_host_step_ms':host,'materialize_and_h2d_s':h2d,
       'kernel_union_ms':union(intervals)/1000,'kernel_span_ms':(max(x[1] for x in intervals)-min(x[0] for x in intervals))/1000,
       'top_kernels_ms':sorted(grouped.items(),key=lambda x:x[1],reverse=True)[:25],
       'runtime_sync_calls':len(sync),'runtime_sync_host_ms':sum(e['dur'] for e in sync)/1000,
       'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
       'operations':ops,'trace':str(trace.resolve()),'note':'operator device totals overlap; CUDA event step includes host starvation; union measures kernel-active intervals, not SM occupancy'}
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='operations'},indent=2))

if __name__=='__main__':main()
