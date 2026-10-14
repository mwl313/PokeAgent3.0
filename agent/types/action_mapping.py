"""Shared canonical decode/token references for typed, packed and stored rows."""
import numpy as np


def canonical_records(records):
    raw=np.asarray(records)
    if raw.ndim!=2 or raw.shape[1]!=6 or not np.issubdtype(raw.dtype,np.integer):
        raise ValueError('invalid atomic action records')
    result=raw.astype(np.int64,copy=True)
    if raw.dtype==np.uint8:
        result[:,3]=raw[:,3].astype(np.int8).astype(np.int64)
    if np.any((result[:,3]<-2)|(result[:,3]>2)):
        raise ValueError('invalid signed target; packed decode cannot clamp')
    if np.any((result[:,[0,1,2,4,5]]<0)|(result[:,[0,1,2,4,5]]>255)) or np.any(result[:,0]>3) or np.any(result[:,5]>1):
        raise ValueError('invalid canonical action fields')
    return result


def token_refs(records,active,layout):
    records=canonical_records(records)
    kind,own,move,destination=records[:,0],records[:,1],records[:,2],records[:,4]
    pick=kind==0
    if np.any((~pick)&(own>1)) or np.any(pick&(destination>=6)):
        raise ValueError('invalid action active position or preview destination')
    active=np.asarray(active,dtype=np.int64)
    if active.shape!=(2,):raise ValueError('missing active-to-roster map')
    if np.any((active < -1)|(active>5)) or active[0]>=0 and active[0]==active[1]:
        raise ValueError('invalid active roster identities')
    roster=np.where(pick,destination,active[np.clip(own,0,1)])
    if np.any((kind==1)&(roster<0)):
        raise ValueError('move has no actual active actor')
    if np.any((kind==1)&(move>=4)&(move!=255)):
        raise ValueError('invalid move slot')
    entity=np.where(roster>=0,layout.POKEMON_START+roster,-1)
    move_token=np.where((kind==1)&(move<4)&(roster>=0),layout.MOVES_START+roster*4+move,-1)
    return records,entity,move_token
