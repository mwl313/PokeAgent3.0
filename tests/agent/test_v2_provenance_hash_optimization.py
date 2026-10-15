"""Byte-level equivalence to independently loaded historical hash code."""
import copy,subprocess,types
from collections import UserList
import numpy as np
import pytest
from agent.ppo.recompute_safety import compact_hash,row_contract_hash
from agent.types.action_mapping import CanonicalCandidateSequence,CanonicalCandidateBlock
from agent.types.observation import ObservationLayout

@pytest.fixture(scope='module')
def reference():
    module=types.ModuleType('historical_hash_reference')
    source=subprocess.check_output(['git','show','93ebc1b5086b7a2bf247b779eb6442671bbbdd10:agent/ppo/recompute_safety.py'],text=True)
    exec(compile(source,'historical_hash_reference','exec'),module.__dict__)
    return module

@pytest.mark.parametrize('dtype',['<f4','>f4','f8','i4','u2'])
def test_compact_bytes_empty_repeated_noncontiguous_and_mutated(reference,dtype):
    arrays=[np.arange(n*6,dtype=dtype).reshape(n,6)[:,::2] for n in [0,1,4,1,4]]
    values=np.empty(len(arrays),dtype=object)
    for i,row in enumerate(arrays):values[i]=row
    fields={'z':np.arange(30,dtype=np.int32).reshape(5,6)[:,::2],'ragged':values,'a':np.ones((5,2),bool)}
    before=compact_hash(fields);assert before==reference.compact_hash(fields)
    arrays[2][0,0]=99
    assert compact_hash(fields)==reference.compact_hash(fields)!=before
    values[1]=values[1].astype('f8')
    assert compact_hash(fields)==reference.compact_hash(fields)


def rows():
    block=CanonicalCandidateBlock.from_packed(np.array([2]),np.array([[1,0,0,255,255,0],[1,0,1,254,255,0]],np.uint8),np.array([0,2]),np.array([[5,3]]),ObservationLayout(),64)
    view=block.row(0)
    return [types.SimpleNamespace(action_ids=(CanonicalCandidateSequence(view),),candidate_mask=((True,True),),entity_token=(CanonicalCandidateSequence(view,'entity'),),move_token=(CanonicalCandidateSequence(view,'move'),),selected=(1,),selected_actions=((1,0,1,-2,255,0),),branch_slots=(0,),old_logprob=-.75)]

@pytest.mark.parametrize('field',['action_ids','candidate_mask','entity_token','move_token','selected','selected_actions','branch_slots','old_logprob'])
def test_hash_matches_legacy_and_detects_each_contract_change(reference,field):
    r=rows();before=row_contract_hash(r);assert before==reference.row_contract_hash(r)
    setattr(r[0],field,0 if field=='old_logprob' else ((0,),))
    assert row_contract_hash(r)==reference.row_contract_hash(r)!=before

@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_logprobs_still_rejected(reference,value):
    r=rows();r[0].old_logprob=value
    for fn in [row_contract_hash,reference.row_contract_hash]:
        with pytest.raises(ValueError):fn(r)

def test_untrusted_sequences_and_candidate_subclasses_keep_legacy_iteration(reference):
    r=rows();r[0].selected=UserList([1])
    class Custom(CanonicalCandidateSequence):
        def __iter__(self):return iter([7,8])
    r[0].entity_token=(Custom(r[0].entity_token[0].row,'entity'),)
    assert row_contract_hash(r)==reference.row_contract_hash(r)
