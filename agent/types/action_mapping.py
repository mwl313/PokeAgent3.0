"""Shared canonical decode/token references for typed, packed and stored rows."""
import numpy as np
from dataclasses import dataclass


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
    return _mapped_records(records, active, layout)


def _mapped_records(records, active, layout):
    kind,own,move,destination=records[:,0],records[:,1],records[:,2],records[:,4]
    pick=kind==0
    if np.any((~pick)&(own>1)) or np.any(pick&(destination>=6)):
        raise ValueError('invalid action active position or preview destination')
    active=np.asarray(active,dtype=np.int64)
    if active.shape not in ((2,), (len(records),2)):raise ValueError('missing active-to-roster map')
    if np.any((active < -1)|(active>5)) or np.any((active[...,0]>=0)&(active[...,0]==active[...,1])):
        raise ValueError('invalid active roster identities')
    acting = active[np.clip(own,0,1)] if active.ndim==1 else active[np.arange(len(records)),np.clip(own,0,1)]
    roster=np.where(pick,destination,acting)
    if np.any((kind==1)&(roster<0)):
        raise ValueError('move has no actual active actor')
    if np.any((kind==1)&(move>=4)&(move!=255)):
        raise ValueError('invalid move slot')
    entity=np.where(roster>=0,layout.POKEMON_START+roster,-1)
    move_token=np.where((kind==1)&(move<4)&(roster>=0),layout.MOVES_START+roster*4+move,-1)
    return records,entity,move_token


def _owned_readonly(array):
    """Bytes-backed arrays cannot regain writeability through setflags/aliases."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, init=False)
class CanonicalCandidateBlock:
    """One validated native branch batch, with immutable action/reference ownership.

    Construct through from_packed only. Prefix-dependent branches each receive
    a fresh block; no cache crosses an engine request or policy iteration.
    """
    records: np.ndarray
    entity: np.ndarray
    move: np.ndarray
    offsets: np.ndarray
    rows: np.ndarray
    positions: np.ndarray

    @classmethod
    def from_packed(cls, counts, records, offsets, active_maps, layout, capacity):
        counts=np.asarray(counts); offsets=np.asarray(offsets)
        if not np.issubdtype(counts.dtype,np.integer) or not np.issubdtype(offsets.dtype,np.integer):
            raise ValueError('packed counts/offsets must be integers')
        if counts.ndim!=1 or offsets.shape!=(len(counts)+1,) or int(offsets[0])!=0 or int(offsets[-1])!=len(records) or not np.array_equal(np.diff(offsets),counts):
            raise ValueError('packed candidate offsets/counts mismatch')
        if np.any(counts<=0) or np.any(counts>capacity):
            raise ValueError('empty/overflowed packed candidate capacity')
        canonical=canonical_records(records)
        if active_maps is None:
            if np.any(canonical[:,0]!=0):raise ValueError('normal action requires active-to-roster metadata')
            active_maps=np.full((len(counts),2),-1,dtype=np.int64)
        active_maps=np.asarray(active_maps,dtype=np.int64)
        if active_maps.shape!=(len(counts),2):raise ValueError('active map batch mismatch')
        rows=np.repeat(np.arange(len(counts),dtype=np.int64),counts)
        canonical,entity,move=_mapped_records(canonical,active_maps[rows],layout)
        positions=np.arange(len(records),dtype=np.int64)-offsets[rows]
        result=object.__new__(cls)
        for key,value in dict(records=canonical,entity=entity,move=move,offsets=offsets.astype(np.int64),rows=rows,positions=positions).items():
            object.__setattr__(result,key,_owned_readonly(value))
        return result

    def row(self,index):
        if not 0<=index<len(self.offsets)-1:raise IndexError(index)
        return CanonicalCandidateRow(self,index)


@dataclass(frozen=True)
class CanonicalCandidateRow:
    block: CanonicalCandidateBlock
    index: int

    def __post_init__(self):
        if not isinstance(self.block,CanonicalCandidateBlock) or not 0<=self.index<len(self.block.offsets)-1:
            raise ValueError('invalid canonical candidate row reference')

    @property
    def records(self):
        return self.block.records[int(self.block.offsets[self.index]):int(self.block.offsets[self.index+1])]

    @property
    def entity(self):
        return self.block.entity[int(self.block.offsets[self.index]):int(self.block.offsets[self.index+1])]

    @property
    def move(self):
        return self.block.move[int(self.block.offsets[self.index]):int(self.block.offsets[self.index+1])]


@dataclass(frozen=True,repr=False,eq=False)
class CanonicalCandidateSequence:
    """Tuple-compatible lazy view over trusted immutable candidate storage.

    NumPy materialization reads the typed array directly. Legacy iteration,
    indexing and repr produce the same Python integers/tuples as before.
    """
    row: CanonicalCandidateRow
    field: str = 'records'

    def __post_init__(self):
        if not isinstance(self.row,CanonicalCandidateRow) or self.field not in ('records','entity','move'):
            raise ValueError('candidate sequence requires a trusted immutable row')

    @property
    def _array(self):return getattr(self.row,self.field)

    def __len__(self):return len(self._array)

    def __getitem__(self,index):
        import operator
        if isinstance(index,slice):
            return tuple(tuple(v) if isinstance(v,list) else int(v) for v in self._array[index].tolist())
        value=self._array[operator.index(index)].tolist()
        return tuple(value) if isinstance(value,list) else int(value)

    def __iter__(self):
        for value in self._array.tolist():yield tuple(value) if isinstance(value,list) else int(value)

    def __array__(self,dtype=None,copy=None):
        array=self._array
        if dtype is not None:array=array.astype(dtype,copy=False)
        return array.copy() if copy else array

    def __repr__(self):return repr(tuple(self))

    def __eq__(self,other):
        if not hasattr(other,'__iter__'):return NotImplemented
        return tuple(self)==tuple(other)
