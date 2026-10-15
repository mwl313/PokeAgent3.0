import json,torch,types,pathlib,dataclasses
from agent.model import PA3Config,build_model
from agent.buffer.rollout_buffer import RolloutBuffer,BlockObservationStore
from agent.ppo.experimental_checkpoint import load_envelope
import torch.nn.functional as F
root=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-integration');torch.set_num_threads(2);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
out=[]
for rank in (0,1):
 path=root/f'runs/v2_final_release/continuation_fixed/uninterrupted/report.json.rank{rank}.iteration19.recompute.pt';saved=load_envelope(str(path));model=build_model(PA3Config.from_dict(saved['model_config']),device='cuda:0');model.load_state_dict(saved['model']);model.configure_precision('D')
 old=model.encoder.forward
 def canonical(obs):
  if not torch.is_autocast_enabled('cuda'):return old(obs)
  outputs=[]
  for start in range(0,len(obs),1024):
   part=obs.narrow(start,min(start+1024,len(obs)));n=len(part)
   if n<1024:part=part.cat([part.select([0]*(1024-n))])
   outputs.append(old(part)[:n])
  return torch.cat(outputs,0)
 model.encoder.forward=canonical
 for group,rows,obs in zip(saved['actor_groups'],saved['group_rows'],saved['group_observations']):
  store=BlockObservationStore();block=store.own_observation(obs);rows=[dataclasses.replace(row,observation_ref=store.add_block_row(block,j)) for j,row in enumerate(rows)];buf=RolloutBuffer(observation_store=store);buf.rows=rows;batch=buf.to_batch(rows,device='cuda:0')
  with torch.inference_mode():amp=model.evaluate(batch.observation,batch.candidates,selected=batch.candidates.selected)[1].request_logprob.float().cpu()
  for evidence in saved['result']['secondary']['rows']:
   if not group['start']<=evidence['row_index']<group['stop']:continue
   index=evidence['row_index']-group['start'];one=buf.to_batch([rows[index]],device='cuda:0')
   with torch.inference_mode():isolated=model.evaluate(one.observation,one.candidates,selected=one.candidates.selected)[1].request_logprob.float().item()
   result={'rank':rank,'row_index':evidence['row_index'],'canonical_group':amp[index].item(),'canonical_isolated':isolated,'fp32':evidence['fp32_isolated'],'group_isolated_diff':abs(amp[index].item()-isolated),'isolated_fp32_diff':abs(isolated-evidence['fp32_isolated'])};out.append(result);print(json.dumps(result),flush=True)
  del batch,buf,store;torch.cuda.empty_cache()
 del model;torch.cuda.empty_cache()
(root/'runs/v2_final_release/canonical_encoder_only_probe.json').write_text(json.dumps(out,indent=2)+'\n')
