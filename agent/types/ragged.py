"""Uncapped per-token CSR features, including lossless compact row storage."""
from dataclasses import dataclass
import numpy as np
import torch
from agent.contracts import SPEC

FIELDS = SPEC["ragged"]
COUNTS = {"effects":"effect_counts", "repertoire":"repertoire_counts", "types":"type_counts",
          "base_moves":"base_move_counts", "move_effects":"move_effect_counts"}
CATEGORICAL = {"effects":(0,6), "repertoire":(0,), "types":(0,), "base_moves":(0,), "move_effects":(0,2,3)}


@dataclass
class RaggedBatch:
    values: torch.Tensor
    counts: torch.Tensor
    host_counts: np.ndarray

    @classmethod
    def empty(cls, batch, width, device="cpu"):
        return cls.from_rows([np.zeros((0,width),np.float32) for _ in range(batch)],
                             np.zeros((batch,96),np.int64),width).to(device)

    @classmethod
    def from_rows(cls, rows, counts, width):
        raw=np.asarray(counts)
        if not np.issubdtype(raw.dtype,np.integer):
            raise ValueError("ragged counts must be integers")
        counts=np.array(counts,dtype=np.int64,copy=True)
        if counts.ndim!=2 or counts.shape[1]!=96 or np.any(counts<0) or np.any(counts>65535) or len(rows)!=len(counts):
            raise ValueError("invalid ragged counts shape/range")
        checked=[]
        for row,length in zip(rows,counts.sum(1)):
            row=np.array(row,dtype=np.float32,copy=True)
            if row.shape!=(int(length),width) or not np.isfinite(row).all():
                raise ValueError("ragged row count/width mismatch or nonfinite data")
            checked.append(row)
        values=np.concatenate(checked,axis=0) if checked else np.zeros((0,width),np.float32)
        return cls(torch.from_numpy(values),torch.from_numpy(counts.copy()),counts)

    def to(self,device):
        return RaggedBatch(self.values.to(device),self.counts.to(device),self.host_counts.copy())

    def rows(self):
        values=self.values.detach().cpu().numpy()
        offsets=np.r_[0,np.cumsum(self.host_counts.sum(1))]
        return [values[int(a):int(b)].copy() for a,b in zip(offsets[:-1],offsets[1:])]

    def select(self,index):
        index=np.asarray(index.detach().cpu() if isinstance(index,torch.Tensor) else index,dtype=np.int64)
        if index.ndim!=1 or np.any(index<0) or np.any(index>=len(self.host_counts)):
            raise ValueError("ragged selection index out of range")
        offsets=np.r_[0,np.cumsum(self.host_counts.sum(1))]
        lengths=offsets[index+1]-offsets[index]
        starts=np.cumsum(lengths)-lengths
        # One vectorized gather, including repeats, arbitrary order and empty
        # rows. Offsets are recomputed: no mutable-CSR cache is trusted.
        gather=np.arange(int(lengths.sum()),dtype=np.int64)+np.repeat(offsets[index]-starts,lengths)
        counts=self.host_counts[index].copy()
        return RaggedBatch(self.values.index_select(0,torch.as_tensor(gather,device=self.values.device)),
                           torch.as_tensor(counts,device=self.counts.device),counts)

    def narrow(self,begin,end):
        if not 0<=begin<=end<=len(self.host_counts):
            raise ValueError("invalid ragged slice")
        offsets=np.r_[0,np.cumsum(self.host_counts.sum(1))]
        return RaggedBatch(self.values[int(offsets[begin]):int(offsets[end])],self.counts[begin:end],self.host_counts[begin:end].copy())

    @staticmethod
    def cat(parts):
        return RaggedBatch(torch.cat([p.values for p in parts]),torch.cat([p.counts for p in parts]),
                           np.concatenate([p.host_counts for p in parts]))

    def quantized(self,name):
        values=self.values.clone()
        numeric=[i for i in range(values.shape[1]) if i not in CATEGORICAL[name]]
        if numeric:
            values[:,numeric]=values[:,numeric].half().float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("ragged FP16 quantization overflow")
        return RaggedBatch(values,self.counts,self.host_counts.copy())


def _matrix(records,name):
    """Native structured rows → exact FP32 integer IDs and numeric fields."""
    if records.dtype.names is None:
        return np.asarray(records,dtype=np.float32)
    expected=tuple(FIELDS[name]) if name!='move_effects' else ('kind','chance','status','volatile','boosts','heal','heal_known')
    if records.dtype.names!=expected:
        raise ValueError(f"unexpected structured fields for {name}")
    columns=[]
    for field in records.dtype.names:
        value=records[field]
        columns.append(value.reshape(len(records),-1) if len(records) else np.zeros((0,7 if field=='boosts' else 1)))
    return np.concatenate(columns,axis=1).astype(np.float32)


def from_native(payload,batch):
    if batch==0:
        return {name:RaggedBatch.empty(0,len(fields)) for name,fields in FIELDS.items()}
    if "ragged" in payload:
        from pa3_engine.observation import split_batch_ragged_all
        rows=split_batch_ragged_all(payload)
        if len(rows)!=batch:
            raise ValueError('ragged/fixed batch size mismatch')
    else:
        if batch!=1:
            raise ValueError("multi-view ragged payload requires explicit batch sidecar")
        rows=[payload]
    result={}
    for name,fields in FIELDS.items():
        if name not in rows[0] or COUNTS[name] not in payload:
            raise ValueError(f"missing V2 ragged family {name}")
        counts=np.asarray(payload[COUNTS[name]])
        if counts.ndim==1:counts=counts[None]
        result[name]=RaggedBatch.from_rows([_matrix(r[name],name) for r in rows],counts,len(fields))
    return result


def to_compact(ragged):
    out={}
    for name,section in ragged.items():
        rows=section.rows();objects=np.empty(len(rows),dtype=object)
        for i,row in enumerate(rows):objects[i]=row
        out['ragged_'+name]=objects
        out[COUNTS[name]]=section.host_counts.copy().astype(np.uint16)
    return out


def from_compact(payload):
    result={}
    for name,fields in FIELDS.items():
        if 'ragged_'+name not in payload or COUNTS[name] not in payload:
            raise ValueError(f"missing compact V2 ragged family {name}")
        result[name]=RaggedBatch.from_rows(payload['ragged_'+name],payload[COUNTS[name]],len(fields))
    return result
