import json,pathlib,statistics,subprocess,time
import torch
from agent.buffer.rollout_buffer import BlockObservationStore
from agent.ppo.experimental_checkpoint import load_envelope
from agent.ppo.recompute_safety import compact_hash
torch.set_num_threads(2);root=pathlib.Path(__file__).resolve().parent
artifact=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/reports/continuation.json.rank0.iteration19.recompute.pt')
saved=load_envelope(str(artifact));obs=saved['sampled_observations'][0].cat(saved['sampled_observations'][1:]);samples={'export_then_deepcopy':[],'private_owned_export':[]};hashes={}
for repeat in range(7):
    for name in samples:
        store=BlockObservationStore();start=time.perf_counter()
        if name=='export_then_deepcopy':block=store.own_block(obs.to_compact_numpy())
        else:block=store.own_observation(obs)
        samples[name].append(1000*(time.perf_counter()-start))
        if repeat==6:
            for i in range(len(obs)):store.add_block_row(block,i)
            hashes[name]=compact_hash(store.stacked().to_compact_numpy())
assert len(set(hashes.values()))==1
result={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'rows':len(obs),'output_hashes':hashes,'times':{name:{'samples_ms':values[2:],'median_ms':statistics.median(values[2:])} for name,values in samples.items()},'scope':'same real576-row observation; interleaved five post-warmup CPU samples; export+ownership only, not whole collection'}
(root/'EXPORT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
