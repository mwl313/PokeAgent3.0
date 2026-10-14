from types import SimpleNamespace
import numpy as np
import torch
from agent.evaluation.batched_policy import BatchedPolicyEvaluator


def test_cdf_roundoff_cannot_select_padding():
    evaluator=BatchedPolicyEvaluator(None,device='cpu')
    class Scorer:
        def build_keys(self,*args):return torch.zeros(1,4,2)
        def score_keys(self,*args):return torch.zeros(1,4)
        def advance_keys(self,keys,hidden):return keys
    class Model:
        scorer=Scorer()
        def _branch_stats(self,*args):
            return None,torch.tensor([[.49999994,.49999994,0.,0.]]),None,None
    table=SimpleNamespace(action_ids=torch.zeros(1,1,4,6,dtype=torch.long),
        entity_token=torch.zeros(1,1,4,dtype=torch.long),move_token=torch.zeros(1,1,4,dtype=torch.long),
        mask=torch.tensor([[[True,True,False,False]]]))
    picks,_=evaluator._sample_model(Model(),SimpleNamespace(tokens=torch.zeros(1,1,2)),None,table,[np.nextafter(1.,0.)])
    assert picks.tolist()==[1]
