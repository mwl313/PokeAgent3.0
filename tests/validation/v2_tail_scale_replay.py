import json,os,torch
from pathlib import Path
from agent.model import build_model,PA3Config
from agent.ppo import PPOLearner,PPOConfig
from agent.ppo.ddp import ddp_rank_loss
out=Path('runs/v2_final_release/migration_batch');e=torch.load(out/'gradient_failure/nonfinite.rank0.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
model=build_model(PA3Config(),device='cuda:0');model.load_state_dict(e['learner']['model']);model.configure_precision('D');model.encoder.canonical_batch_rows=1024
learner=PPOLearner(model,PPOConfig(**e['learner']['config']['ppo']),device='cuda:0');batches=[b.to('cuda:0') for b,t in e['last_micros']]
result=[]
for scale in [65536.,32768.,16384.,8192.,4096.,2048.,1024.]:
 learner.optimizer.zero_grad(set_to_none=True)
 terms_saved=[]
 for batch in batches:
  terms=learner._forward_terms(batch);loss=ddp_rank_loss(terms['loss_unscaled']*terms['actor_count'],terms['value_mean']*terms['value_count'],4.,4.,2,learner.config.value_coefficient)
  (loss*scale).backward();terms_saved.append({k:float(v.detach()) for k,v in terms.items() if k in ['loss_unscaled','value_mean']})
 bad={n:int((~torch.isfinite(p.grad)).sum()) for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())}
 # Post-DDP unscale and averaging, without applying Adam.
 for p in model.parameters():
  if p.grad is not None:p.grad.div_(scale*2)
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
 result.append({'scale':scale,'nonfinite_parameters':bad,'grad_norm':float(norm),'forward':terms_saved})
 print('scale',scale,'bad_parameters',len(bad),'norm',float(norm),flush=True)
(out/'tail_scale_replay.json').write_text(json.dumps(result,indent=2)+'\n')
