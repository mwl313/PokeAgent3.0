"""Independent corrected-reference and candidate GPU proof, captured real rows."""
import argparse, dataclasses, hashlib, json, os, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from agent.buffer.rollout_buffer import RolloutBuffer
from agent.model import PA3Config, build_model
from agent.ppo import PPOConfig, PPOLearner
from agent.ppo.experimental_checkpoint import load_envelope
from scripts.run_ddp_ppo import _compact_digest

p=argparse.ArgumentParser();p.add_argument('--reference',action='store_true');p.add_argument('--artifact',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
torch.set_num_threads(2);torch.manual_seed(7);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
e=load_envelope(a.artifact);buffer=RolloutBuffer()
for i in range(1024):
 j=i%len(e['sampled_rows']);ref=buffer.observation_store.add(e['sampled_observations'][j]);buffer.rows.append(dataclasses.replace(e['sampled_rows'][j],observation_ref=ref))
batch=buffer.to_batch(device='cuda:0');batch.raw_advantages=torch.linspace(-1,1,1024,device='cuda:0');batch.returns=torch.linspace(-.5,.5,1024,device='cuda:0')
model=build_model(PA3Config(**e['model_config']),device='cuda:0');model.load_state_dict(e['model']);model.configure_precision('D')
if a.reference:
 # Historical executable, independent autocast boundary on the original layer.
 layer=model.encoder.embedding.ragged_embedding.numeric['effects']
 def explicit_fp32(x):
  with torch.autocast('cuda',enabled=False):return F.linear(x.float(),layer.weight.float(),None if layer.bias is None else layer.bias.float())
 layer.forward=explicit_fp32
learner=PPOLearner(model,PPOConfig(ppo_epochs=2),device='cuda:0')
learner.optimizer.zero_grad(set_to_none=True);terms=learner._forward_terms(batch)
loss=terms['loss_unscaled']+learner.config.value_coefficient*terms['value_mean'];learner.scaler.scale(loss).backward();learner.scaler.unscale_(learner.optimizer)
grads={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
assert all(torch.isfinite(g).all() for g in grads.values())
torch.nn.utils.clip_grad_norm_(model.parameters(),learner.config.max_grad_norm);learner.scaler.step(learner.optimizer);learner.scaler.update()
result={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'reference':a.reference,'rows':1024,'precision':'D/FP16-effects-FP32','workspace':os.environ['CUBLAS_WORKSPACE_CONFIG'],'model_config':model.config.to_dict(),'losses':{k:float(v.detach()) for k,v in terms.items()},'gradient_digest':_compact_digest(grads),'adam_digest':_compact_digest(learner.optimizer.state_dict()),'updated_model_digest':_compact_digest(model.state_dict()),'captured_artifact_sha256':hashlib.sha256(Path(a.artifact).read_bytes()).hexdigest(),'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
