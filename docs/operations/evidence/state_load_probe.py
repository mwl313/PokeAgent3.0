import json,pathlib,importlib.util,torch
from agent.ppo.experimental_checkpoint import load_envelope,capture_rng_state,restore_rng_state
from agent.ppo.learner import PPOLearner
from agent.ppo.config import PPOConfig
from agent.model import PA3Config,build_model
r=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-integration');e=r/'docs/operations/evidence';a=json.loads((e/'checkpoint_audit.json').read_text());p=pathlib.Path(a['checkpoint']);s=importlib.util.spec_from_file_location('launch',r/'scripts/run_ddp_ppo.py');mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
torch.set_num_threads(2);torch.cuda.set_device(0);out={}
for rank in (0,1):
 d=load_envelope(str(p/f'rank{rank}.pt'));model=build_model(PA3Config.from_dict(d['learner']['config']['model']));model.configure_precision('D');learner=PPOLearner(model,PPOConfig(**d['learner']['config']['ppo']),device='cuda:0',lr_scale=.5)
 learner.load_state_dict(d['learner']);md,od=mod.state_digest(learner.model,learner.optimizer);expected=d['rank_digests'][str(rank)];assert (md,od)==(expected['model'],expected['optimizer'])
 assert learner.scheduler.state_dict()==d['learner']['scheduler'];assert learner.scaler.state_dict()==d['learner']['scaler']
 gens=[torch.Generator(device='cpu') for _ in d['rng']['generators']];restore_rng_state(d['rng'],gens);now=capture_rng_state(gens)
 import numpy as np
 assert now['python']==d['rng']['python'] and np.array_equal(now['numpy'][1],d['rng']['numpy'][1]) and now['numpy'][2:]==d['rng']['numpy'][2:]
 assert torch.equal(now['torch_cpu'],d['rng']['torch_cpu']) and all(torch.equal(x,y) for x,y in zip(now['torch_cuda'],d['rng']['torch_cuda'])) and all(torch.equal(x,y) for x,y in zip(now['generators'],d['rng']['generators']))
 out[str(rank)]={'model_digest':md,'optimizer_digest':od,'scheduler':learner.scheduler.state_dict(),'optimizer_steps':learner.optimizer_steps,'scaler_equal':True,'rng_equal':True,'scope':'serialized state load on cuda0, sequential ranks; no DDP recreation or rollout continuation'}
 del learner,model;torch.cuda.empty_cache()
(e/'state_load_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('both rank payloads restored; digests/scheduler/scaler/RNG match')
