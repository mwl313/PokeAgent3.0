"""Independent representation references and mutation isolation for V2 fast paths."""
import numpy as np
import pytest
import torch
from agent.types.action_mapping import CanonicalCandidateBlock, token_refs
from agent.types.observation import ObservationBatch, ObservationLayout
from agent.types.ragged import RaggedBatch
from agent.buffer.rollout_buffer import InlineObservationStore, ColumnarObservationStore, BlockObservationStore


def test_candidate_block_matches_per_request_reference_and_owns_wire():
    layout=ObservationLayout()
    actions=np.array([(1,0,0,-2,255,0),(1,1,2,-1,255,1),(2,0,255,0,4,0),
                      (3,1,255,0,255,0),(1,0,255,0,255,0),(0,2,255,0,3,0)],dtype=np.int64)
    counts=np.array([3,2,1]);offsets=np.array([0,3,5,6]);wire=actions.astype(np.uint8)
    active=np.array([[5,3],[2,4],[-1,-1]])
    block=CanonicalCandidateBlock.from_packed(counts,wire,offsets,active,layout,64)
    for i in range(3):
        expected=token_refs(wire[offsets[i]:offsets[i+1]],active[i],layout)
        row=block.row(i)
        for actual,wanted in zip((row.records,row.entity,row.move),expected):
            np.testing.assert_array_equal(actual,wanted)
            with pytest.raises(ValueError):actual.setflags(write=True)
    wire[:]=0;active[:]=0;offsets[:]=0
    np.testing.assert_array_equal(block.records,actions)


@pytest.mark.parametrize('corrupt', ['target','offset','empty','capacity','active','move'])
def test_candidate_block_rejects_invalid_boundary(corrupt):
    wire=np.array([(1,0,0,1,255,0)],dtype=np.uint8);counts=np.array([1]);offsets=np.array([0,1]);active=np.array([[5,3]])
    capacity=64
    if corrupt=='target':wire[0,3]=3
    if corrupt=='offset':offsets[-1]=0
    if corrupt=='empty':counts[0]=0
    if corrupt=='capacity':capacity=0
    if corrupt=='active':active[0]=[3,3]
    if corrupt=='move':wire[0,2]=4
    with pytest.raises(ValueError):CanonicalCandidateBlock.from_packed(counts,wire,offsets,active,ObservationLayout(),capacity)


@pytest.mark.parametrize('store_type',[InlineObservationStore,ColumnarObservationStore,BlockObservationStore])
def test_compact_at_owns_all_fixed_and_five_ragged_families(store_type):
    observation=ObservationBatch.dummy(3)
    for name,section in observation.ragged.items():
        width=section.values.shape[1]
        observation.ragged[name]=RaggedBatch.from_rows([np.zeros((1,width),np.float32) for _ in range(3)],
          np.tile(np.r_[1,np.zeros(95,dtype=np.int64)],(3,1)),width)
    compact=observation.to_compact_numpy();expected={k:np.copy(v[1:2]) for k,v in compact.items()}
    for k,v in expected.items():
        if v.dtype.hasobject:v[0]=v[0].copy()
    store=store_type();ref=store.add_compact_at(compact,1)
    for value in compact.values():
        if value.dtype.hasobject:
            for item in value:item[:]=42
        else:value[:]=0
    actual=store.get(ref).to_compact_numpy()
    for key in expected:
        if expected[key].dtype.hasobject:np.testing.assert_array_equal(actual[key][0],expected[key][0])
        else:np.testing.assert_array_equal(actual[key],expected[key])


def test_owned_blocks_gather_across_batches_and_isolate_returned_tensors():
    store=BlockObservationStore();reference=InlineObservationStore()
    for offset in [0.,0.2,0.4]:
        observation=ObservationBatch.dummy(3);observation.floats[:,:,0]=offset
        compact=observation.to_compact_numpy();block=store.own_block(compact)
        for i in [2,0,1]:
            store.add_block_row(block,i);reference.add_compact_at(compact,i)
        compact['floats'][:]=123
    for indices in [list(range(9)),[8,0,3,0,7],[2]]:
        a=store.stacked_indices(indices).to_compact_numpy();b=reference.stacked_indices(indices).to_compact_numpy()
        for key in a:
            if a[key].dtype.hasobject:
                for x,y in zip(a[key],b[key]):np.testing.assert_array_equal(x,y)
            else:np.testing.assert_array_equal(a[key],b[key])
    store.get(0).floats[:]=999
    torch.testing.assert_close(store.get(0).floats,reference.get(0).floats)


def test_ragged_vectorized_selection_matches_independent_loop():
    counts=np.zeros((5,96),np.int64);counts[:,0]=[2,0,3,1,0]
    rows=[np.arange(n*3,dtype=np.float32).reshape(n,3)+10*i for i,n in enumerate(counts[:,0])]
    section=RaggedBatch.from_rows(rows,counts,3)
    for index in ([3,0,2,0,1],[],[4,1]):
        actual=section.select(index)
        expected=np.concatenate([rows[i] for i in index]) if index else np.zeros((0,3),np.float32)
        np.testing.assert_array_equal(actual.values.numpy(),expected)
        np.testing.assert_array_equal(actual.host_counts,counts[index])
    # A later counts mutation must not leave stale memoized offsets.
    section.host_counts[0,0]=1;section.host_counts[1,0]=1
    torch.testing.assert_close(section.select([1]).values,section.values[1:2])
