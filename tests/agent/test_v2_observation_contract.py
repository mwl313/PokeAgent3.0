import copy
import numpy as np
import pytest
import torch

from agent.contracts import CONTRACT_HASH, SPEC
from agent.types.observation import ObservationBatch
from agent.types.ragged import RaggedBatch, FIELDS
from agent.model import PA3Config
from agent.model.embeddings import TokenEmbedding
from agent.model.precision import get_precision_contract
from agent.buffer.rollout_buffer import InlineObservationStore, ColumnarObservationStore
from agent.ppo.recompute_safety import compact_hash


def config():
    return PA3Config(encoder_layers=1,d_model=32,attention_heads=2,head_dim=16,
                     ffn_dim=64,prefix_hidden=32,critic_hidden=32)


def rich(batch=3):
    observation=ObservationBatch.dummy(batch,seed=7)
    templates={'effects':[1,1,3/32768,1,1/32768,1,4,1],
               'repertoire':[1],'types':[20],'base_moves':[1,.2,.4,0,1],
               'move_effects':[1,.5,1,0,1/6,0,0,0,0,0,0,0,0]}
    for name,template in templates.items():
        counts=np.zeros((batch,96),np.int64); rows=[]
        for i in range(batch):
            counts[i,4]=i+1
            rows.append(np.tile(np.asarray(template,np.float32),(i+1,1)))
        observation.ragged[name]=RaggedBatch.from_rows(rows,counts,len(FIELDS[name]))
    return observation


def identity(observation):
    return compact_hash(observation.to_compact_numpy())


@pytest.mark.parametrize('store_type',[InlineObservationStore,ColumnarObservationStore])
def test_five_families_survive_dynamic_store_selection_and_source_mutation(store_type):
    original=rich();store=store_type()
    for i in range(len(original)):store.add(original.select([i]))
    assert identity(store.stacked())==identity(original)
    selected=store.stacked_indices([2,0,2])
    assert identity(selected)==identity(original.select([2,0,2]))
    assert identity(original.narrow(1,3))==identity(original.select([1,2]))
    assert identity(original.select([0]).cat([original.select([1,2])]))==identity(original)
    compact=original.to_compact_numpy(); owned=store_type();owned.add_compact({k:v[:1] for k,v in compact.items()})
    before=identity(owned.get(0));compact['ragged_types'][0][0,0]=1
    compact['floats'][0,0,0]=88
    assert identity(owned.get(0))==before


@pytest.mark.parametrize('name',list(FIELDS))
def test_every_family_reaches_embedding_and_gradient(name):
    torch.manual_seed(7);model=TokenEmbedding(config());original=rich(1)
    changed=original.clone(); changed.ragged[name].values[0,0]+=(-1 if name=='types' else 1)
    before=model(original);after=model(changed)
    assert not torch.equal(before,after),name
    (before*torch.linspace(-1,1,32)).sum().backward()
    projection=model.ragged_embedding.project[name]
    assert projection.weight.grad is not None and bool(projection.weight.grad.abs().sum()>0)


def test_empty_families_remain_graph_connected_for_ddp():
    model=TokenEmbedding(config());output=model(ObservationBatch.dummy(1))
    output.sum().backward()
    assert all(p.grad is not None for p in model.ragged_embedding.parameters())
    assert all(torch.count_nonzero(p.grad)==0 for p in model.ragged_embedding.parameters())


def test_known_zero_and_false_are_explicit_neural_inputs():
    model=TokenEmbedding(config());observation=ObservationBatch.dummy(1)
    observation.floats[0,4,31]=0;observation.flags[0,4,0]=False
    inputs=[]
    hook=model.numeric_projection.register_forward_pre_hook(lambda module,args:inputs.append(args[0].clone()))
    model(observation)
    unknown=observation.clone();unknown.float_known[0,4,31]=False;unknown.flag_known[0,4,0]=False
    model(unknown);hook.remove()
    assert inputs[0][0,4,31]==inputs[1][0,4,31]==0
    assert inputs[0][0,4,90+31]==1 and inputs[1][0,4,90+31]==0
    assert inputs[0][0,4,140]==1 and inputs[1][0,4,140]==0


def test_hidden_unknown_payload_values_do_not_change_embedding():
    model=TokenEmbedding(config());first=rich(1)
    first.float_known[0,4,31]=False;first.category_known[0,4,3]=False
    first.ragged['effects'].values[0,[3,5,7]]=0
    other=first.clone();other.floats[0,4,31]=float('nan');other.categories[0,4,3]=99999
    other.ragged['effects'].values[0,[2,4,6]]=torch.tensor([55.,99.,9999.])
    assert torch.equal(model(first),model(other))


@pytest.mark.parametrize('corruption',['version','hash','family','counts','category','nan','namespace'])
def test_corrupt_contracts_fail_closed(corruption):
    observation=rich(1);payload=observation.to_compact_numpy()
    if corruption=='version':payload['schema_version'][:]=1
    elif corruption=='hash':payload['contract_hash'][0,0]^=1
    elif corruption=='family':del payload['ragged_effects']
    elif corruption=='counts':payload['type_counts'][0,4]+=1
    elif corruption=='category':payload['categories'][0,4,11]=8191
    elif corruption=='nan':payload['floats'][0,4,31]=np.nan
    elif corruption=='namespace':payload['categories'][0,64,16]=0;payload['categories'][0,64,17]=1
    with pytest.raises(ValueError):
        rebuilt=ObservationBatch.from_compact_numpy(payload)
        TokenEmbedding(config())(rebuilt)


def test_ragged_count_overflow_and_fractional_counts_rejected():
    with pytest.raises(ValueError):RaggedBatch.from_rows([np.zeros((0,1))],np.full((1,96),65536),1)
    with pytest.raises(ValueError):RaggedBatch.from_rows([np.zeros((0,1))],np.zeros((1,96),np.float32),1)


def test_mode_d_preserves_event_sign_weight_and_ragged_ids():
    observation=rich(1)
    raw=torch.tensor([-6.,-1.,0.,1.,6.])
    observation.floats[0,64:69,43]=raw.asinh()/torch.tensor(2147483648.).asinh()
    observation.floats[0,4:7,46]=torch.tensor([1.,10.,100.])/10000
    observation.ragged['base_moves'].values[0,0]=947  # categorical integer, never half rounded
    quantized=get_precision_contract('D').prepare_observation(observation)
    assert len(torch.unique(quantized.floats[0,64:69,43]))==5
    assert torch.equal(quantized.ragged['base_moves'].values[:,0],observation.ragged['base_moves'].values[:,0])
    assert bool((quantized.floats[0,4:7,46]>0).all())


def test_native_wire_hash_and_nonempty_metadata_reaches_model():
    pa3=pytest.importorskip('pa3_engine')
    from pathlib import Path
    from pa3_engine.observation import parse_batch,parse_view
    data=Path(__file__).resolve().parents[2]/'engine/data'
    engine=pa3.NativeEngine(str(data),str(data/'a10-training-teams.json'),workers=2)
    handles=engine.reset_batch([0],[1],[(1,2,3,4)],[(0,1)])
    fixed,ragged=engine.observe_fixed_batch(handles,[0])
    payload=parse_batch(fixed,ragged,1);observation=ObservationBatch.from_native_payload(payload)
    single=ObservationBatch.from_native_payload(parse_view(engine.observe_encoded_batch(handles,[0])[0]))
    assert identity(observation)==identity(single)
    assert pa3.CONTRACT_HASH==CONTRACT_HASH
    for name in ['types','base_moves','move_effects']:assert len(observation.ragged[name].values)>0
    output=TokenEmbedding(config())(observation);assert torch.isfinite(output).all()
    bad=bytearray(fixed);bad[2]^=1
    with pytest.raises(ValueError):parse_batch(bytes(bad),ragged,1)


def test_checkpoint_and_evaluator_reject_v1_or_wrong_contract_before_loading(tmp_path):
    from agent.model import build_model
    from agent.ppo import PPOLearner,PPOConfig
    from agent.evaluation.fixed_policy import load_frozen_model
    from agent.evaluation.frozen_policy import FrozenPPOPolicy
    from agent.ppo.experimental_checkpoint import fork_for_epochs
    learner=PPOLearner(build_model(config()),PPOConfig(),device='cpu',amp=False)
    state=learner.state_dict();path=tmp_path/'policy.pt'
    torch.save({'learner':state},path)
    frozen=load_frozen_model(path)
    assert frozen.config.to_dict()==learner.model.config.to_dict()
    assert FrozenPPOPolicy.from_checkpoint(path).identity.observation_schema.endswith('v2')
    old=copy.deepcopy(state);del old['contracts']
    with pytest.raises(ValueError,match='V1'):learner.load_state_dict(old)
    torch.save({'learner':old},path)
    with pytest.raises(ValueError,match='V1'):load_frozen_model(path)
    with pytest.raises(ValueError,match='V1'):FrozenPPOPolicy.from_checkpoint(path)
    with pytest.raises(ValueError,match='V1'):fork_for_epochs({'learner':old},2)
    wrong=copy.deepcopy(state);wrong['contracts']['semantic_hash']='0'*64
    with pytest.raises(ValueError,match='contract'):learner.load_state_dict(wrong)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='bounded V100 precision test requires CUDA')
def test_cuda_mode_d_nonempty_ragged_forward_backward_and_regrouping():
    from agent.model import build_model
    torch.manual_seed(7)
    model=build_model(config(),device='cuda:0');model.configure_precision('D')
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    observation=model.precision_contract.prepare_observation(rich(3).to('cuda:0'))
    encoded=model.encode(observation)
    assert torch.isfinite(encoded.tokens).all()
    (encoded.tokens*torch.linspace(-1,1,32,device='cuda:0')).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.embedding.ragged_embedding.parameters())
    whole=model.encoder.embedding(observation)
    regrouped=torch.cat([model.encoder.embedding(observation.select([i])) for i in range(3)])
    torch.testing.assert_close(whole,regrouped,rtol=1e-5,atol=1e-5)
