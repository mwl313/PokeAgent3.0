import numpy as np
import pytest
from pa3_engine.observation import (split_batch_ragged,split_batch_ragged_all,
    EFFECT,REPERTOIRE,TYPE,BASE_MOVE,MOVE_EFFECT)


@pytest.mark.parametrize('batch_size',[0,1,17])
def test_bulk_offsets_equal_independent_single_view_decoder(batch_size):
    sections=[('effects',EFFECT,'effect_counts'),('repertoire',REPERTOIRE,'repertoire_counts'),
              ('types',TYPE,'type_counts'),('base_moves',BASE_MOVE,'base_move_counts'),
              ('move_effects',MOVE_EFFECT,'move_effect_counts')]
    payload={key:np.zeros((batch_size,96),np.uint16) for _,_,key in sections}
    chunks=[]
    for i in range(batch_size):
        for j,(_,dtype,key) in enumerate(sections):
            count=(i+j)%4;payload[key][i,4+j]=count
            rows=np.zeros(count,dtype=dtype)
            if count:rows[dtype.names[0]]=i+j+1
            chunks.append(rows.tobytes())
    payload['ragged']=b''.join(chunks)
    all_rows=split_batch_ragged_all(payload)
    assert len(all_rows)==batch_size
    for i,bulk in enumerate(all_rows):
        single=split_batch_ragged(payload,i)
        for name,_,_ in sections:
            assert bulk[name].dtype==single[name].dtype
            assert bulk[name].tobytes()==single[name].tobytes()
    if batch_size:
        # A mutable caller cannot reuse a stale offset proof.
        payload['effect_counts'][0,0]+=1
        with pytest.raises(ValueError,match='byte/count'):split_batch_ragged_all(payload)
