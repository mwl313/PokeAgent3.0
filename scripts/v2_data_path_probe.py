"""Same captured V2 inputs across independent source processes, CPU only."""
import argparse,dataclasses,hashlib,json,pathlib,resource,time
import numpy as np
import torch
from agent.buffer.rollout_buffer import RolloutBuffer,InlineObservationStore
from agent.model import PA3Config,build_model
from agent.ppo import PPOLearner,PPOConfig
from agent.ppo.experimental_checkpoint import load_envelope
from agent.ppo.recompute_safety import compact_hash
from agent.train.native_collector import NativeCollector,NativeCollectorConfig
from agent.types.observation import ObservationLayout

def digest_tensors(items):
    h=hashlib.sha256()
    for name,value in items:
        if value is None:continue
        v=value.detach().cpu().contiguous();h.update(name.encode());h.update(str((v.shape,v.dtype)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    saved=load_envelope(str(pathlib.Path(a.artifact).resolve()))
    rows=saved['sampled_rows'];observations=saved['sampled_observations']
    combined=observations[0].cat(observations[1:]);compact=combined.to_compact_numpy()
    maps=combined.active_to_roster();h=NativeCollector.__new__(NativeCollector)
    h.layout=ObservationLayout();h.device=torch.device('cpu');h.config=NativeCollectorConfig()
    # Real branch-dependent candidate sets from the failed sampled rollout.
    groups=[]
    for branches in sorted(set(r.branch_count for r in rows)):
        indices=[i for i,r in enumerate(rows) if r.branch_count==branches]
        for level in range(branches):
            records=np.concatenate([np.asarray(rows[i].action_ids[level],np.int64) for i in indices]).astype(np.uint8)
            counts=np.array([len(rows[i].action_ids[level]) for i in indices],np.int64);offsets=np.r_[0,counts.cumsum()]
            groups.append((indices,level,counts,records,offsets))
    times=[];representation=None;buffer=None
    for repetition in range(a.repeats):
        started=time.perf_counter();tables=[];prepared={}
        for indices,level,counts,records,offsets in groups:
            # Baseline helper independently canonicalizes per request. Candidate
            # implementation independently validates the flat batch once.
            block=None
            try:
                from agent.types.action_mapping import CanonicalCandidateBlock
            except ImportError:pass
            else:block=CanonicalCandidateBlock.from_packed(counts,records,offsets,maps[indices],h.layout,64)
            table=h._level_table_packed(counts,records,offsets,maps[indices],**({'canonical':block} if block is not None else {}))
            tables.append(table)
            for j,i in enumerate(indices):
                if block is not None:
                    row=block.row(j);prepared[i,level]=(row,row.entity,row.move)
                else:
                    prepared[i,level]=h._packed_row_inputs([{'counts':counts,'records':records,'offsets':offsets}],j,1,maps[i])
                    prepared[i,level]=tuple(v[0] for v in prepared[i,level])
        candidate_s=time.perf_counter()-started
        try:
            from agent.buffer.rollout_buffer import BlockObservationStore
            store=BlockObservationStore()
        except ImportError:store=InlineObservationStore()
        buffer=RolloutBuffer(observation_store=store)
        started=time.perf_counter();block_ref=store.own_block(compact) if hasattr(store,'own_block') else None
        for i,row in enumerate(rows):
            records,entities,moves=zip(*(prepared[i,k] for k in range(row.branch_count)))
            fields={k:getattr(row,k) for k in ['selected','old_logprob','value','match_id','side','policy_id','opponent_policy_id','team_ids','request_index','turn','request_kind','branch_slots','reward','done','actor_active','seed_ref']}
            buffer.record_packed(compact,observation_index=i,branch_records=records,entity_token=entities,move_token=moves,
                                **({'observation_block_ref':block_ref} if block_ref is not None else {}),**fields)
        record_s=time.perf_counter()-started
        started=time.perf_counter();batch=buffer.to_batch();materialize_s=time.perf_counter()-started
        representation=dict(observation=compact_hash(batch.observation.to_compact_numpy()),
          candidates=digest_tensors((k,getattr(batch.candidates,k)) for k in ['action_ids','mask','entity_token','move_token','branch_valid','selected']),
          rows=hashlib.sha256(repr([(r.action_ids,r.entity_token,r.move_token,r.selected,r.selected_actions,r.team_ids) for r in buffer.rows]).encode()).hexdigest())
        times.append(dict(candidate_s=candidate_s,record_s=record_s,materialize_s=materialize_s))
    # Identical real observations/candidates with declared synthetic advantage
    # and return targets. This compares math/gradients/Adam, not a learning run.
    selected=[*range(16),*range(200,216),*range(400,416)]
    small=buffer.to_batch([buffer.rows[i] for i in selected])
    small.raw_advantages=torch.linspace(-1,1,len(selected));small.returns=torch.linspace(-0.5,0.5,len(selected))
    model=build_model(PA3Config(**saved['model_config']));model.load_state_dict(saved['model']);model.configure_precision('D')
    learner=PPOLearner(model,PPOConfig(ppo_epochs=2),device='cpu',amp=False)
    terms=learner._forward_terms(small)
    loss=terms['loss_unscaled']+learner.config.value_coefficient*terms['value_mean'];loss.backward()
    gradient=digest_tensors((n,p.grad) for n,p in model.named_parameters())
    losses={k:float(v.detach()) for k,v in terms.items()}
    learner.optimizer.step()
    adam=digest_tensors((f'{n}:{k}',value) for n,p in model.named_parameters() for k,value in sorted(learner.optimizer.state[p].items()))
    result=dict(source_sha=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
      runtime_file_hashes={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in
        ['agent/types/action_mapping.py','agent/types/ragged.py','agent/buffer/rollout_buffer.py','agent/train/native_collector.py']},
      dirty=bool(__import__('subprocess').check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()),
      rows=len(rows),candidate_records=sum(len(g[3]) for g in groups),times=times,representation=representation,
      losses=losses,gradient_digest=gradient,adam_digest=adam,
      updated_model_digest=digest_tensors(model.state_dict().items()),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      model_config=saved['model_config'])
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
