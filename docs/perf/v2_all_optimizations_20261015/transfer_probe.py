import dataclasses,hashlib,json,pathlib,statistics,subprocess,sys,types,time
import torch
from agent.buffer.rollout_buffer import RolloutBuffer,RolloutBatch
from agent.ppo.experimental_checkpoint import load_envelope
from agent.ppo.recompute_safety import compact_hash
root=pathlib.Path(__file__).resolve().parent
torch.set_num_threads(2);torch.cuda.set_device(0)
source=subprocess.check_output(['git','show','e9618c2:agent/buffer/rollout_buffer.py'],text=True)
module=types.ModuleType('_original_bundle');sys.modules[module.__name__]=module;exec(compile(source,'e9618c2:rollout_buffer.py','exec'),module.__dict__)
old=module.RolloutBatch.to
artifact=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/reports/continuation.json.rank0.iteration19.recompute.pt')
saved=load_envelope(str(artifact));buffer=RolloutBuffer()
for i in range(1024):
    j=i%len(saved['sampled_rows']);ref=buffer.observation_store.add(saved['sampled_observations'][j]);buffer.rows.append(dataclasses.replace(saved['sampled_rows'][j],observation_ref=ref))
batch=buffer.to_batch(device='cpu');results={};outputs={}
for name,fn in [('original_blocking',old),('bundled',RolloutBatch.to)]:
    samples=[]
    for _ in range(7):
        torch.cuda.synchronize();start=time.perf_counter();result=fn(batch,'cuda:0');torch.cuda.synchronize();samples.append(1000*(time.perf_counter()-start))
    outputs[name]=result;results[name]={'samples_ms':samples[2:],'median_ms':statistics.median(samples[2:])}
left,right=outputs.values();checks={}
for field in dataclasses.fields(RolloutBatch):
    a=getattr(left,field.name);b=getattr(right,field.name)
    if isinstance(a,torch.Tensor):checks[field.name]=torch.equal(a,b)
checks['observations']=compact_hash(left.observation.to_compact_numpy())==compact_hash(right.observation.to_compact_numpy())
checks['candidates']=all(torch.equal(getattr(left.candidates,f.name),getattr(right.candidates,f.name)) for f in dataclasses.fields(left.candidates))
assert all(checks.values())
results.update({'exact_values':checks,'scope':'1024 real saved rows, whole blocking public transfer wall, five post-warmup samples; no pinned allocation or games/s claim','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
(root/'TRANSFER.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
