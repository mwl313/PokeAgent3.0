"""Inference-only composition proof on preserved failures; never rewrites rows."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import torch
from scripts.v2_recompute_forensics import _buffer,_target_index,_evaluate
from agent.ppo.experimental_checkpoint import load_envelope
from agent.model import PA3Config,build_model
p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(2);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
e=load_envelope(a.artifact);meta=e['result']['secondary']['rows'][0]
target=(meta['match_id'],meta['request_index'],meta['side'],tuple(meta['selected']))
sampled=_buffer(e['sampled_rows'],e['sampled_observations']);j=_target_index(sampled.rows,target)
g=next(i for i,g in enumerate(e['actor_groups']) if g['start']<=meta['row_index']<g['stop'])
actor=_buffer(e['group_rows'][g],e['group_observations'][g]);i=meta['row_index']-e['actor_groups'][g]['start']
model=build_model(PA3Config(**e['model_config']),device='cuda:0');model.load_state_dict(e['model']);model.eval();model.configure_precision('D');model.encoder.canonical_batch_rows=1024
batch=sampled.to_batch(device='cuda:0');reference=_evaluate(model,actor.to_batch(device='cuda:0'),i)
variants={'original_actor':reference,'sampled_learner':_evaluate(model,batch,j)}
for n in (1,17,576,1024,1025):
 for position in (0,n-1):
  indices=[(k%len(sampled.rows)) for k in range(n)];indices[position]=j
  rows=[sampled.rows[k] for k in indices];obs=[e['sampled_observations'][k] for k in indices]
  b=_buffer(rows,obs).to_batch(device='cuda:0');variants[f'rows{n}_position{position}']=_evaluate(model,b,position)
diffs={k:abs(v-reference) for k,v in variants.items()};assert max(diffs.values())<=.001,diffs
# Preserve the historical policy's rejection rather than relabeling it corrected.
model.encoder.embedding.ragged_embedding.fp32_effects=False
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
 _,evaluation=model.evaluate(batch.observation,batch.candidates,selected=batch.candidates.selected)
stored=torch.tensor([r.old_logprob for r in sampled.rows],device='cuda:0') if hasattr(sampled.rows[0],'old_logprob') else batch.old_logprobs
old_max=float((evaluation.request_logprob-stored).abs().max());assert old_max>.001,old_max
result={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'artifact_sha256':hashlib.sha256(Path(a.artifact).read_bytes()).hexdigest(),'canonical_rows':1024,'tolerance':.001,'logprobs':variants,'differences':diffs,'max_difference':max(diffs.values()),'historical_stored_rollout_max':old_max,'historical_rollout_rejected':True,'optimizer_updates':0}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(result)
