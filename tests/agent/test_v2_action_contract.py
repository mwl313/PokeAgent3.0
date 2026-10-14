import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.model import PA3Config,build_model
from agent.model.scorer import ConditionalPrefixScorer
from agent.train.native_collector import NativeCollector,NativeCollectorConfig
from agent.types.actions import AtomicAction
from agent.types.action_mapping import canonical_records,token_refs
from agent.types.observation import ObservationBatch,ObservationLayout
from agent.types.requests import ActionRef,RequestRow
from agent.types.actions import CandidateSet,RequestKind


def helper():
    value=NativeCollector.__new__(NativeCollector)
    value.layout=ObservationLayout();value.device=torch.device('cpu');value.config=NativeCollectorConfig()
    return value


def test_all_signed_targets_are_distinct_before_scoring_and_storage():
    canonical=np.asarray([(1,0,0,t,255,0) for t in [-2,-1,0,1,2]],np.int64)
    wire=canonical.astype(np.uint8)
    assert np.array_equal(canonical_records(wire),canonical)
    for row in canonical:assert AtomicAction.from_tuple(row).as_tuple()==tuple(row)
    fields=ConditionalPrefixScorer._field_ids(torch.as_tensor(canonical))
    assert fields[3].tolist()==[0,1,2,3,4]
    with pytest.raises(ValueError):AtomicAction.from_tuple((1,0,0,255,255,0))
    with pytest.raises(ValueError):canonical_records(np.asarray([(1,0,0,3,255,0)],np.uint8))


def test_typed_packed_and_stored_refs_use_actual_roster_and_signed_values():
    h=helper();mapping=np.asarray([[5,3]])
    canonical=np.asarray([(1,0,1,-2,255,0),(1,1,2,-1,255,1),(2,0,255,0,2,0),
                          (3,1,255,0,255,0),(0,2,255,0,4,0),(1,0,255,0,255,0)],np.int64)
    raw=canonical.astype(np.uint8);counts=np.asarray([len(raw)],np.uint16);offsets=np.asarray([0,len(raw)])
    typed=h._level_table([canonical.tolist()],mapping)
    packed=h._level_table_packed(counts,raw,offsets,mapping)
    n=len(raw)
    for name in ['action_ids','entity_token','move_token','mask']:
        assert torch.equal(getattr(typed,name)[:,:,:n],getattr(packed,name))
    assert packed.entity_token[0,0].tolist()==[9,7,9,7,8,9]
    assert packed.move_token[0,0].tolist()==[37,30,-1,-1,-1,-1]
    records,entities,moves=h._packed_row_inputs([{'counts':counts,'records':raw,'offsets':offsets}],0,1,mapping[0])
    assert np.array_equal(records[0],canonical)
    assert entities[0].tolist()==packed.entity_token[0,0].tolist()
    assert moves[0].tolist()==packed.move_token[0,0].tolist()
    from agent.buffer import RolloutBuffer
    observation=ObservationBatch.dummy(1)
    observation.categories[0,4:10,22]=0
    observation.categories[0,9,22]=1;observation.categories[0,7,22]=2
    buffer=RolloutBuffer()
    row=buffer.record_packed(observation.to_compact_numpy(),branch_records=[raw],
        entity_token=entities,move_token=moves,selected=[0],old_logprob=0.,value=0.,
        match_id=0,side=0,policy_id='current',opponent_policy_id='current',team_ids=(0,1),
        request_index=0,turn=0,request_kind=RequestKind.NORMAL,branch_slots=[0])
    restored=buffer.to_batch()
    assert row.action_ids[0][0][3]==-2
    assert torch.equal(restored.candidates.action_ids[:,:1,:n],packed.action_ids)
    assert torch.equal(restored.candidates.entity_token[:,:1,:n],packed.entity_token)
    assert torch.equal(restored.candidates.move_token[:,:1,:n],packed.move_token)
    with pytest.raises(ValueError):ActionRef.resolve(h.layout,AtomicAction.from_tuple(canonical[0]))
    with pytest.raises(ValueError):h._level_table_packed(counts,raw,offsets)


def test_bad_active_mapping_and_candidate_metadata_fail_closed():
    observation=ObservationBatch.dummy(1)
    observation.categories[0,6,22]=1
    with pytest.raises(ValueError,match='duplicate'):observation.active_to_roster()
    observation=ObservationBatch.dummy(1);observation.category_known[0,4,22]=False
    with pytest.raises(ValueError,match='missing'):observation.active_to_roster()
    with pytest.raises(ValueError):token_refs([(1,0,0,1,255,0)],[-1,3],ObservationLayout())
    h=helper()
    with pytest.raises(ValueError):h._level_table_packed(np.asarray([1]),np.zeros((1,6),np.uint8),np.asarray([0,0]),[[0,1]])


def test_native_arbitrary_leads_switches_candidate_order_and_logits():
    pa3=pytest.importorskip('pa3_engine')
    from pa3_engine.observation import parse_batch,parse_packed_candidates
    data=Path(__file__).resolve().parents[2]/'engine/data'
    engine=pa3.NativeEngine(str(data),str(data/'a10-training-teams.json'),workers=2)
    handle=engine.reset_batch([0],[0],[(2026,10,14,17)],[(0,1)])[0]
    preview=lambda order:[(0,slot,255,0,roster,0) for slot,roster in enumerate(order)]
    result=engine.step_batch([(*handle,[(0,preview([5,3,2,4])),(1,preview([0,1,2,3]))])])[0]
    assert result[0] and result[4] is None
    h=helper();model=build_model(PA3Config(encoder_layers=1,d_model=32,attention_heads=2,head_dim=16,
                                         ffn_dim=64,prefix_hidden=32,critic_hidden=32))
    targets=set()
    for wanted in ([5,3],[2,4]):
        fixed,ragged=engine.observe_fixed_batch([handle],[0])
        observation=ObservationBatch.from_native_payload(parse_batch(fixed,ragged,1));mapping=observation.active_to_roster()
        world=json.loads(json.loads(bytes(engine.snapshot(*handle)))['payload'])
        assert world['sides'][0]['active']==wanted==mapping[0].tolist()
        encoded=model.encode(observation);prefix=[]
        for level in range(2):
            candidates=engine.candidates(*handle,0,prefix)
            prefix_wire=np.asarray(prefix,np.int64).reshape(1,level,6).astype(np.uint8).tobytes()
            counts,records,_=engine.candidates_packed_walk([handle],[0],level,prefix_wire)
            packed=parse_packed_candidates(counts,records,1)
            assert canonical_records(packed['records']).tolist()==[list(c) for c in candidates]
            typed_table=h._level_table([candidates],mapping)
            packed_table=h._level_table_packed(packed['counts'],packed['records'],packed['offsets'],mapping)
            n=len(candidates)
            for field in ['action_ids','entity_token','move_token']:
                assert torch.equal(getattr(typed_table,field)[:,:,:n],getattr(packed_table,field))
            hidden=model.scorer.initial_state(1, encoded.tokens.device, encoded.tokens.dtype)
            def logits(table):
                keys=model.scorer.build_keys(encoded.tokens,table.action_ids[:,0,:n],table.entity_token[:,0,:n],table.move_token[:,0,:n])
                return model.scorer.score_keys(keys,hidden,table.mask[:,0,:n])
            torch.testing.assert_close(logits(typed_table),logits(packed_table),rtol=1e-6,atol=1e-6)
            for c in candidates:targets.add(c[3])
            prefix.append(candidates[0])
        if wanted==[5,3]:
            result=engine.step_batch([(*handle,[(0,[(2,0,255,0,2,0),(2,1,255,0,4,0)]),
                                               (1,[(2,0,255,0,2,0),(2,1,255,0,3,0)])])])[0]
            assert result[0] and result[4] is None
    assert targets=={-2,-1,0,1,2}


def test_partial_refill_preserves_unrelated_policy_assignments():
    import random
    from agent.train.native_collector import MatchPolicyAssignment
    h=helper();h._slot_counter=0;h._team_count=2;h.stats=type('Stats',(),{'reset_seconds':0.,'cohorts':0,'team_ids_seen':0})()
    h._assignments=[MatchPolicyAssignment('current_current',(str(i),str(i)),None,None,i) for i in range(4)]
    original=list(h._assignments)
    class Engine:
        def reset_slots_batch(self,*args):return [(1,2)]
    h.engine=Engine();h._make_assignments=lambda count,roles:[MatchPolicyAssignment('current_current',('new','new'),None,None,99)]
    h._reset_slots(1,random.Random(7),slots=[1])
    assert len(h._assignments)==4 and h._assignments[1].seat_policy_ids==('new','new')
    assert all(h._assignments[i] is original[i] for i in [0,2,3])
