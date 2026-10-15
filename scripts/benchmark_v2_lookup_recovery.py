"""Bounded isolated operations and real-row GPU update; never resumes training."""
import argparse,dataclasses,hashlib,json,pathlib,statistics,subprocess,sys,time,types
import numpy as np
import torch
from agent.buffer.rollout_buffer import RolloutBuffer
from agent.model import build_model,PA3Config
from agent.model.scorer import ConditionalPrefixScorer
from agent.ppo import PPOLearner,PPOConfig
from agent.ppo.experimental_checkpoint import load_envelope
from agent.types.ragged import RaggedBatch
from agent.train.native_collector import NativeCollector

BASE='2a3bdcdacc488753548cdb1b159eedf3dbbd64ff'
def install_reference(path):
    name='_lookup_reference_'+path.replace('/','_').replace('.','_')
    module=types.ModuleType(name);sys.modules[name]=module
    code=subprocess.check_output(['git','show',f'{BASE}:{path}'],text=True)
    exec(compile(code,f'{BASE}:{path}','exec'),module.__dict__)
    return module
def digest(items):
    h=hashlib.sha256()
    for name,tensor in items:
        if tensor is not None:
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',action='store_true');p.add_argument('--fp32',action='store_true');p.add_argument('--exact',action='store_true');p.add_argument('--artifact',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    from types import SimpleNamespace
    from scripts.run_ddp_ppo import pin_deterministic_workspace
    pin_deterministic_workspace(SimpleNamespace(deterministic_diagnostic=True))
    torch.set_num_threads(2);torch.manual_seed(7);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    if a.reference:
        ConditionalPrefixScorer.token_references=install_reference('agent/model/scorer.py').ConditionalPrefixScorer.token_references
        old_ragged=install_reference('agent/types/ragged.py').RaggedBatch
        RaggedBatch.from_rows=classmethod(old_ragged.__dict__['from_rows'].__func__)
        NativeCollector._prefix_actions=staticmethod(install_reference('agent/train/native_collector.py').NativeCollector._prefix_actions)
    saved=load_envelope(str(pathlib.Path(a.artifact).resolve()))
    # Actual captured family rows, repeated solely for bounded timing.
    observations=saved['sampled_observations'];compact=observations[0].cat(observations[1:]).to_compact_numpy()
    local={};ragged_hashes={}
    from agent.types.ragged import FIELDS,COUNTS
    for name,fields in FIELDS.items():
        samples=[]
        for _ in range(7):
            start=time.perf_counter();section=RaggedBatch.from_rows(compact['ragged_'+name],compact[COUNTS[name]],len(fields));samples.append(time.perf_counter()-start)
        local['ragged_'+name]=samples[2:];ragged_hashes[name]=digest([('values',section.values),('counts',section.counts)])
    wire=np.array([[[1,0,2,255,255,1],[1,1,0,254,255,0]]],np.uint8)
    samples=[]
    for _ in range(7):
        start=time.perf_counter()
        for i in range(10000):actions=NativeCollector._prefix_actions(wire,0,2)
        samples.append(time.perf_counter()-start)
    local['prefix_10000']=samples[2:]
    buffer=RolloutBuffer()
    for i in range(1024):
        j=i%len(saved['sampled_rows']);ref=buffer.observation_store.add(observations[j]);buffer.rows.append(dataclasses.replace(saved['sampled_rows'][j],observation_ref=ref))
    start=time.perf_counter();batch=buffer.to_batch(device='cuda:0');torch.cuda.synchronize();prepare=time.perf_counter()-start
    batch.raw_advantages=torch.linspace(-1,1,1024,device='cuda:0');batch.returns=torch.linspace(-.5,.5,1024,device='cuda:0')
    model=build_model(PA3Config(**saved['model_config']),device='cuda:0');model.load_state_dict(saved['model']);model.configure_precision('A' if a.fp32 else 'D')
    if a.reference and a.exact:raise ValueError('reference and exact backend are mutually exclusive')
    model.scorer.experimental_row_lookup=not a.reference and not a.exact
    model.scorer.experimental_exact_row_lookup=a.exact
    learner=PPOLearner(model,PPOConfig(ppo_epochs=2),device='cuda:0',**({'amp':False} if a.fp32 else {}))
    def step(proof=False):
        learner.optimizer.zero_grad(set_to_none=True);terms=learner._forward_terms(batch)
        loss=terms['loss_unscaled']+learner.config.value_coefficient*terms['value_mean'];learner.scaler.scale(loss).backward();learner.scaler.unscale_(learner.optimizer)
        if proof:
            gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
            torch.save(gradients,out/'gradients.pt')
            losses={k:float(v.detach()) for k,v in terms.items()}
        torch.nn.utils.clip_grad_norm_(model.parameters(),learner.config.max_grad_norm);learner.scaler.step(learner.optimizer);learner.scaler.update()
        if proof:
            torch.save(model.state_dict(),out/'updated_model.pt');torch.save(learner.optimizer.state_dict(),out/'optimizer.pt')
            return {'losses':losses,'gradient_digest':digest(gradients.items()),'updated_model_digest':digest(model.state_dict().items())}
    proof=step(True)
    for _ in range(2):step()
    times=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();step();end.record();end.synchronize();times.append(start.elapsed_time(end))
    result={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'reference_methods_sha':BASE if a.reference else None,
      'runtime_file_sha256':{name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in ['agent/model/scorer.py','agent/types/ragged.py','agent/train/native_collector.py']},
      'CUBLAS_WORKSPACE_CONFIG':__import__('os').environ.get('CUBLAS_WORKSPACE_CONFIG'),'rows':1024,'local_seconds':local,'ragged_hashes':ragged_hashes,
      'prefix_actions':actions,'prepare_with_H2D_s':prepare,'full_gpu_step_ms':times,'full_gpu_step_median_ms':statistics.median(times),**proof,
      'precision':'A/FP32' if a.fp32 else 'D/FP16','experimental_row_lookup':not a.reference and not a.exact,'experimental_exact_row_lookup':a.exact,
      'scope':'real captured observations/candidates and model, synthetic advantage/return; single-process update, not DDP/self-play or games/s; local CPU operations each five post-warmup samples'}
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
