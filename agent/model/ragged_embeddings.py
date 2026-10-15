"""All five public ragged families contribute uncapped per-token features."""
import torch
import torch.nn.functional as F
from torch import nn
from agent.types.ragged import FIELDS,CATEGORICAL
from agent.contracts import SPEC


class RaggedEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.numeric=nn.ModuleDict({name:nn.Linear(len(fields),config.d_model) for name,fields in FIELDS.items()})
        self.ids=nn.ModuleDict({f'{name}_{column}':nn.Embedding(config.category_vocab,config.category_embed_dim)
                               for name,columns in CATEGORICAL.items() for column in columns})
        self.project=nn.ModuleDict({name:nn.Linear(len(CATEGORICAL[name])*config.category_embed_dim,config.d_model,bias=False) for name in FIELDS})
        self.count=nn.ModuleDict({name:nn.Linear(1,config.d_model,bias=False) for name in FIELDS})
        # Execution-only, not a model parameter or checkpoint field.  Mode D
        # enables it through ``PrecisionContract`` after the V100 forensic
        # replay located the first batch-composition-dependent operation here.
        self.fp32_effects = False
        for embedding in self.ids.values():nn.init.normal_(embedding.weight,std=config.embedding_init_std)
        for linear in [*self.numeric.values(),*self.project.values(),*self.count.values()]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:nn.init.zeros_(linear.bias)

    def forward(self,observation):
        output=observation.floats.new_zeros((*observation.token_mask.shape,self.config.d_model))
        gpu_validated=output.is_cuda
        if gpu_validated:self._validate_cuda_values(observation)
        for name,section in observation.ragged.items():
            values=section.values
            if values.shape[0]==0:
                # DDP keeps the same reducer graph even when a rank/minibatch
                # has no public rows for this family. Exact zero contribution.
                modules=[self.numeric[name],self.project[name],self.count[name]]
                modules.extend(self.ids[f'{name}_{c}'] for c in CATEGORICAL[name])
                zero=sum(p.reshape(-1)[0]*0 for module in modules for p in module.parameters())
                output=output+zero
                continue
            numeric=values.clone()
            boolean_columns={'effects':(1,3,5,7),'base_moves':(3,4),'move_effects':(12,)}.get(name,())
            if not gpu_validated and boolean_columns and bool(((values[:,boolean_columns]!=0)&(values[:,boolean_columns]!=1)).any()):
                raise ValueError(f'invalid ragged boolean {name}')
            if name=='effects':
                numeric[:,2]=torch.where(values[:,3].bool(),values[:,2],0.)
                numeric[:,4]=torch.where(values[:,5].bool(),values[:,4],0.)
                numeric[:,6]=torch.where(values[:,7].bool(),values[:,6],0.)
            elif name=='move_effects':
                numeric[:,11]=torch.where(values[:,12].bool(),values[:,11],0.)
            ids=[]
            catalogue=SPEC['catalogue_max_ids']
            limits={'effects':{0:catalogue['conditions'],6:255},'repertoire':{0:catalogue['moves']},
                    'types':{0:catalogue['types']},'base_moves':{0:catalogue['moves']},
                    'move_effects':{0:4,2:catalogue['conditions'],3:catalogue['conditions']}}
            for column in CATEGORICAL[name]:
                raw=numeric[:,column]
                if not gpu_validated and bool(((raw<0)|(raw>limits[name][column])|(raw>=self.config.category_vocab)|(raw!=raw.floor())).any()):
                    raise ValueError(f'invalid known ragged category {name}:{column}')
                ids.append(self.ids[f'{name}_{column}'](raw.long()))
                numeric[:,column]=0.
            rows=torch.nn.functional.silu(
                self._numeric_projection(name, numeric)
                + self.project[name](torch.cat(ids,dim=-1))
            )
            # CSR order makes each token reduction independent of other batch
            # rows. No padding cap, no CUDA scatter atomics, no clipped rows.
            pooled=torch.segment_reduce(rows,'sum',lengths=section.counts.reshape(-1)).reshape_as(output)
            counts=section.counts.to(output.dtype).log1p().unsqueeze(-1)
            output=output+pooled+self.count[name](counts)
        return output

    def _numeric_projection(self, name, numeric):
        """Run the known unstable Mode-D effects GEMM with FP32 accumulation.

        The normal autocast path lets cuBLAS select an FP16 Tensor Core kernel
        from the *total ragged record count*.  That count changes when the
        same observation is packed with different independent requests.  A
        V100 replay showed the first target-row difference in the ``effects``
        numeric projection itself.  This explicit FP32 linear call is limited
        to that projection; all other ragged and dense encoder operations keep
        their existing AMP behavior.
        """
        layer = self.numeric[name]
        if not (self.fp32_effects and name == "effects" and numeric.is_cuda):
            return layer(numeric)
        with torch.autocast(device_type="cuda", enabled=False):
            return F.linear(numeric.float(), layer.weight.float(),
                            None if layer.bias is None else layer.bias.float())

    def _validate_cuda_values(self,observation):
        catalogue=SPEC['catalogue_max_ids']
        limits={'effects':{0:catalogue['conditions'],6:255},'repertoire':{0:catalogue['moves']},
                'types':{0:catalogue['types']},'base_moves':{0:catalogue['moves']},
                'move_effects':{0:4,2:catalogue['conditions'],3:catalogue['conditions']}}
        checks=[]
        for name,section in observation.ragged.items():
            values=section.values
            if not len(values):continue
            columns={'effects':(1,3,5,7),'base_moves':(3,4),'move_effects':(12,)}.get(name,())
            if columns:checks.append((((values[:,columns]!=0)&(values[:,columns]!=1)).any(),f'invalid ragged boolean {name}'))
            for column in CATEGORICAL[name]:
                raw=values[:,column]
                if name=='effects' and column==6:raw=torch.where(values[:,7].bool(),raw,0.)
                checks.append((((raw<0)|(raw>limits[name][column])|(raw>=self.config.category_vocab)|(raw!=raw.floor())).any(),f'invalid known ragged category {name}:{column}'))
        if checks:
            bad=torch.stack([flag for flag,_ in checks]).cpu().tolist()
            for failed,(_,message) in zip(bad,checks):
                if failed:raise ValueError(message)
