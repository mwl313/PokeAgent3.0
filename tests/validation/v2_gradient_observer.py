"""Observational failure capture only; arithmetic, scale and ordering unchanged."""
import os
if os.environ.get('V2_CAPTURE_NONFINITE'):
 import json,torch
 from pathlib import Path
 from agent.ppo.learner import PPOLearner
 old_forward=PPOLearner._forward_terms
 old_scale=PPOLearner._update_distributed_scaler
 def observed_forward(self,batch):
  terms=old_forward(self,batch)
  self._diagnostic_last_micros=getattr(self,'_diagnostic_last_micros',[])[-1:]+[(batch,{k:v.detach().clone() for k,v in terms.items()})]
  return terms
 def observed_scale(self,globally_finite):
  if not globally_finite:
   import torch.distributed as dist
   rank=dist.get_rank();root=Path(os.environ['V2_CAPTURE_NONFINITE']);root.mkdir(parents=True,exist_ok=True)
   p=root/f'nonfinite.rank{rank}.pt'
   if not p.exists():
    # Gradients may already have been cleared by the distributed skip path.
    payload={'rank':rank,'learner':self.state_dict(),'last_micros':self._diagnostic_last_micros,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'optimizer_steps':self.optimizer_steps,'scale':self.scaler.get_scale()}
    torch.save(payload,p)
    print('Captured nonfinite diagnostic:',p,flush=True)
  return old_scale(self,globally_finite)
 PPOLearner._forward_terms=observed_forward;PPOLearner._update_distributed_scaler=observed_scale
 import subprocess
 original_popen=subprocess.Popen
 def observed_popen(command,*args,**kwargs):
  if '--worker' in command:
   env=kwargs['env'].copy();env['PYTHONPATH']='/tmp/v2_gradient_diagnostic:'+env['PYTHONPATH'];kwargs['env']=env
  return original_popen(command,*args,**kwargs)
 subprocess.Popen=observed_popen
 original_clip=torch.nn.utils.clip_grad_norm_
 def observed_clip(parameters,*args,**kwargs):
  params=list(parameters);norm=original_clip(params,*args,**kwargs)
  if not bool(torch.isfinite(norm)):
   print('Nonfinite gradient norm:',float(norm), 'parameters:',[(i,int((~torch.isfinite(p.grad)).sum())) for i,p in enumerate(params) if p.grad is not None and not bool(torch.isfinite(p.grad).all())],flush=True)
  return norm
 torch.nn.utils.clip_grad_norm_=observed_clip
