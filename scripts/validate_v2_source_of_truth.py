#!/usr/bin/env python3
"""Read-only release consistency check. PASS is configuration parity, not cutover approval."""
import argparse,hashlib,json
from pathlib import Path
from agent.runtime.config import resolve_profile,launcher_values

def validate(root):
    root=Path(root);p=root/'configs/pa3_v2_production.yaml'
    resolved=resolve_profile(str(p));profile=resolved['profile']
    facts=json.loads((root/'docs/operations/V2_SOURCE_OF_TRUTH_FACTS.json').read_text())
    for section in ['training','precision','dataset','collector','evaluation']:
        if facts[section]!=profile[section]:raise ValueError('Source of Truth drift: '+section)
    text=(root/'docs/TRAINING_SOURCE_OF_TRUTH.md').read_text()
    if resolved['config_hash'] not in text:raise ValueError('Source of Truth profile hash drift')
    contract=hashlib.sha256((root/'configs/pa3_v2_contract.json').read_bytes()).hexdigest()
    if contract!=facts['contract_file_sha256']:raise ValueError('semantic contract file drift')
    teams=json.loads(Path(profile['dataset']['teams']).read_text())
    count=len(teams['teams']) if isinstance(teams,dict) and 'teams' in teams else len(teams)
    if count!=profile['dataset']['train_teams']:raise ValueError('team count drift')
    from agent.ppo.config import PPOConfig
    config=PPOConfig(ppo_epochs=profile['training']['ppo_epochs'])
    mapping={'global_minibatch':'global_minibatch_size','microbatch':'microbatch_size',
             'per_rank_minibatch':'per_rank_minibatch_size','grad_accumulation':'grad_accumulation_per_rank',
             'warmup_matches':'warmup_matches','horizon_matches':'horizon_matches','peak_lr':'peak_lr',
             'clip_epsilon':'clip_epsilon','gamma':'gamma','gae_lambda':'gae_lambda',
             'entropy_coefficient':'entropy_coefficient','uniform_kl_coefficient':'uniform_kl_coefficient',
             'target_approx_kl':'target_approx_kl','value_coefficient':'value_coefficient',
             'max_grad_norm':'max_grad_norm','action_temperature':'action_temperature','optimizer':'optimizer'}
    for key,attribute in mapping.items():
        if profile['training'][key]!=getattr(config,attribute):raise ValueError('PPO executable drift: '+key)
    if profile['training']['effective_peak_lr']!=config.peak_lr*profile['training']['lr_scale']:
        raise ValueError('effective LR drift')
    values=launcher_values(resolved)
    if not values.get('recompute_secondary') or not values['recompute_gate']:raise ValueError('secondary safety inactive')
    return {'status':'PASS','scope':'read-only proposed configuration/SoT/dataset parity; NOT production readiness',
            'config_hash':resolved['config_hash'],'teams':count,'contract_file_sha256':contract,'launcher_values':values}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));args=parser.parse_args()
    print(json.dumps(validate(args.root),indent=2))
if __name__=='__main__':main()
