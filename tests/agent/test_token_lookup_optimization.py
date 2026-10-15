"""Independent expanded-gather reference, including null and repeated indices."""
import copy
import numpy as np
import pytest
import torch
from agent.model.config import PA3Config
from agent.model.scorer import ConditionalPrefixScorer
from agent.types.ragged import RaggedBatch
from agent.train.native_collector import NativeCollector


def reference(scorer, encoded, entity, move):
    b,t,d=encoded.shape
    def lookup(index,null):
        index=index.reshape(b,-1)
        selected=encoded.gather(1,index.clamp_min(0)[...,None].expand(-1,-1,d))
        return torch.where((index>=0)[...,None],selected,null.view(1,1,d))
    return scorer.entity_projection(lookup(entity,scorer.null_entity))+scorer.move_projection(lookup(move,scorer.null_move))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('all_null',[False,True])
@pytest.mark.parametrize('noncontiguous',[False,True])
def test_lookup_output_input_and_parameter_gradients(device,all_null,noncontiguous):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    with torch.random.fork_rng(devices=[0] if device=='cuda' else []):
        torch.manual_seed(77)
        new=ConditionalPrefixScorer(PA3Config()).to(device);old=copy.deepcopy(new)
        x=torch.randn(3,11,320,device=device)
        if noncontiguous:x=x.transpose(1,2).contiguous().transpose(1,2)
        a=x.detach().requires_grad_();b=x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        entity=torch.tensor([[-1,0,0,10,4,4,4]]*3,device=device)
        move=torch.tensor([[5,-1,5,0,0,10,10]]*3,device=device)
        if all_null:entity.fill_(-1);move.fill_(-1)
        actual=new.token_references(a,entity,move);expected=reference(old,b,entity,move)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        dy=torch.randn_like(actual);actual.backward(dy);expected.backward(dy)
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-6)
        for (name,p),(_,q) in zip(new.named_parameters(),old.named_parameters()):
            assert (p.grad is None)==(q.grad is None),name
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-6)


def test_ragged_constructor_owns_inputs_and_rejects_nonfinite():
    row=np.arange(6,dtype=np.float32).reshape(2,3);counts=np.zeros((1,96),np.int64);counts[0,0]=2
    result=RaggedBatch.from_rows([row],counts,3);row[:]=99;counts[:]=0
    np.testing.assert_array_equal(result.values.numpy(),np.arange(6).reshape(2,3))
    assert result.host_counts[0,0]==2
    for value in [float('nan'),float('inf')]:
        row[0,0]=value;counts[0,0]=2
        with pytest.raises(ValueError):RaggedBatch.from_rows([row],counts,3)


def test_bulk_prefix_signed_wire_decode_and_source_ownership():
    wire=np.array([[[1,0,2,255,255,1],[1,1,0,254,255,0],[0,0,255,0,3,0]]],dtype=np.uint8)
    expected=[(1,0,2,-1,255,1),(1,1,0,-2,255,0),(0,0,255,0,3,0)]
    actual=NativeCollector._prefix_actions(wire,0,3);assert actual==expected
    wire[:]=0;assert actual==expected
