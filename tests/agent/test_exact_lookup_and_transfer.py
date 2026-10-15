import dataclasses
import numpy as np
import pytest
import torch
from agent.types.action_mapping import CanonicalCandidateBlock,CanonicalCandidateSequence
from agent.types.observation import ObservationLayout,ObservationBatch
from agent.types.requests import BranchCandidatesBatch
from agent.types.ragged import RaggedBatch,FIELDS
from agent.buffer.rollout_buffer import RolloutBatch


@pytest.mark.parametrize('count',[0,1,31,32,33,64,65,128])
def test_exact_cuda_duplicate_accumulation_and_nondefault_stream(count):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    from agent.model.exact_row_lookup import exact_rows
    previous=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[0]),torch.cuda.stream(torch.cuda.Stream()):
            torch.manual_seed(91)
            a=torch.randn(3*11,320,device='cuda',requires_grad=True);b=a.detach().clone().requires_grad_()
            refs=torch.randint(3,(3,count),device='cuda');index=refs+torch.arange(3,device='cuda')[:,None]*11
            grad=torch.randn(3,count,320,device='cuda')
            old=a.reshape(3,11,320).gather(1,refs[...,None].expand(-1,-1,320))
            new=exact_rows(b,index)
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            old.backward(grad);new.backward(grad)
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(previous)


def test_lazy_candidates_keep_tuple_repr_numpy_and_immutable_ownership():
    wire=np.array([[1,0,0,255,255,0],[1,0,1,2,255,0]],np.uint8)
    block=CanonicalCandidateBlock.from_packed(np.array([2]),wire,np.array([0,2]),np.array([[5,3]]),ObservationLayout(),64)
    row=block.row(0)
    for field in ['records','entity','move']:
        seq=CanonicalCandidateSequence(row,field)
        expected=tuple(tuple(v) if isinstance(v,list) else int(v) for v in getattr(row,field).tolist())
        assert seq==expected and expected==seq and repr(seq)==repr(expected)
        import pickle,copy
        assert type(pickle.loads(pickle.dumps(seq))) is tuple
        assert pickle.loads(pickle.dumps(seq))==expected and copy.deepcopy(seq)==expected
        assert seq[-1]==expected[-1] and seq[:1]==expected[:1]
        np.testing.assert_array_equal(np.asarray(seq),np.asarray(expected))
        with pytest.raises(ValueError):np.asarray(seq).setflags(write=True)
    wire[:]=0
    assert CanonicalCandidateSequence(row)[0]==(1,0,0,-1,255,0)


def test_bundled_H2D_is_complete_and_isolated_when_public_call_returns():
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    obs=ObservationBatch.dummy(2)
    for name,fields in FIELDS.items():
        counts=np.zeros((2,96),np.int64);counts[:,0]=1
        obs.ragged[name]=RaggedBatch.from_rows([np.zeros((1,len(fields)),np.float32)]*2,counts,len(fields))
    candidates=BranchCandidatesBatch(torch.zeros(2,1,2,6,dtype=torch.long),torch.ones(2,1,2,dtype=torch.bool),torch.zeros(2,1,2,dtype=torch.long),torch.full((2,1,2),-1,dtype=torch.long),torch.ones(2,1,dtype=torch.bool),torch.zeros(2,1,dtype=torch.long))
    values={f.name:torch.arange(2,dtype=torch.float32) for f in dataclasses.fields(RolloutBatch) if f.name not in ['observation','candidates','policy_ids','advantages']}
    batch=RolloutBatch(observation=obs,candidates=candidates,policy_ids=['current']*2,**values)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):actual=batch.to('cuda:0')
    assert stream.query()
    batch.old_logprob.fill_(99);batch.observation.floats.fill_(99);batch.candidates.mask.fill_(False)
    torch.testing.assert_close(actual.old_logprob.cpu(),torch.arange(2,dtype=torch.float32))
    assert bool(actual.candidates.mask.all()) and not bool(actual.observation.floats.eq(99).any())
    for name in FIELDS:
        batch.observation.ragged[name].values.fill_(99);batch.observation.ragged[name].host_counts[:]=0
        assert not bool(actual.observation.ragged[name].values.eq(99).any())
        assert actual.observation.ragged[name].host_counts[:,0].tolist()==[1,1]
