"""Inference-only explicit reload contract for the canonical A10 best pointer."""
from pathlib import Path
from agent.evaluation.a10_league import read, verify_entry
from agent.evaluation.fixed_policy import load_frozen_model

class BestPolicyHandle:
    """Call refresh between requests; publish a new model only after strict loading."""
    def __init__(self, run_root, device='cpu'):
        self.root=Path(run_root)/'evaluation/league_v2';self.device=device;self.identity=None;self.model=None
    def refresh(self):
        best=read(self.root/'best_policy.json')
        if not best: raise ValueError('canonical best policy unavailable')
        if best['policy_id']==self.identity:return False
        entry=read(self.root/'registry.json')['policies'][best['policy_id']]
        verify_entry(entry);candidate=load_frozen_model(entry['path'],device=self.device)
        self.model=candidate;self.identity=best['policy_id'];return True
