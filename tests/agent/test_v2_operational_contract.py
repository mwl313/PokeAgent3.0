import importlib.util,json
from pathlib import Path
import pytest


def launcher():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('v2_operational_launcher',root/'scripts/run_ddp_ppo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return root,module


def test_v2_profile_keeps_ppo_hyperparameters_and_requires_durable_stats(tmp_path,monkeypatch):
    root,module=launcher();monkeypatch.chdir(root)
    old=json.loads((root/'configs/a10_production.yaml').read_text())
    new=json.loads((root/'configs/pa3_v2_production.yaml').read_text())
    assert old['training']==new['training']
    with pytest.raises(SystemExit):module.parse_args(['--profile','configs/pa3_v2_production.yaml'])
    args=module.parse_args(['--profile','configs/pa3_v2_production.yaml',
        '--team-stats-root',str(tmp_path/'stats'),'--rolling-checkpoint-root',str(tmp_path/'rolling'),
        '--experimental-checkpoint',str(tmp_path/'boundary'),'--stop-file',str(tmp_path/'STOP')])
    assert args.deterministic_diagnostic and args.epochs==2 and args.checkpoint_interval_matches==100000
    assert args.microbatch==1024 and args.minibatch==4096 and args.experimental_lr_scale==.5


def test_cooperative_stop_requires_checkpoint_destination():
    _,module=launcher()
    with pytest.raises(SystemExit):module.parse_args(['--stop-file','STOP'])


def test_deterministic_workspace_is_pinned_and_conflicts_rejected(monkeypatch):
    from types import SimpleNamespace
    _,module=launcher();args=SimpleNamespace(deterministic_diagnostic=True)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    module.pin_deterministic_workspace(args)
    import os
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':16:8')
    with pytest.raises(RuntimeError,match='requires'):module.pin_deterministic_workspace(args)


def test_exact_lookup_requires_bounded_explicit_contract_and_resume_parity():
    _,module=launcher()
    assert not module.parse_args([]).experimental_exact_token_lookup
    with pytest.raises(SystemExit):module.parse_args(['--experimental-exact-token-lookup'])
    flags=['--experimental-exact-token-lookup','--deterministic-diagnostic','--precision-mode','D','--precision-stability','fp16-reduction-off']
    args=module.parse_args(flags)
    module.validate_lookup_restore({'token_lookup_backend':'exact-row-v100-v1'},args)
    with pytest.raises(RuntimeError,match='backend mismatch'):module.validate_lookup_restore({},args)
    with pytest.raises(RuntimeError,match='backend mismatch'):module.validate_lookup_restore({'token_lookup_backend':'exact-row-v100-v1'},module.parse_args([]))
    with pytest.raises(SystemExit):module.parse_args(flags+['--iterations','6'])


def test_v2_secondary_is_resolved_and_configuration_drift_rejected(tmp_path, monkeypatch):
    from agent.runtime.config import resolve_profile, launcher_values
    root,module=launcher();monkeypatch.chdir(root)
    profile=json.loads((root/'configs/pa3_v2_production.yaml').read_text())
    resolved=resolve_profile(str(root/'configs/pa3_v2_production.yaml'))
    assert launcher_values(resolved)['recompute_secondary'] is True
    for field,value in [('recompute_secondary',False),('tolerance',.002),('ragged_effects_fp32',False)]:
        bad=json.loads(json.dumps(profile));bad['precision'][field]=value
        path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='secondary safety|stable ragged-effects'):resolve_profile(str(path))
    args=module.parse_args(['--profile','configs/pa3_v2_production.yaml',
        '--team-stats-root',str(tmp_path/'stats'),'--rolling-checkpoint-root',str(tmp_path/'rolling'),
        '--experimental-checkpoint',str(tmp_path/'boundary')])
    assert args.recompute_secondary and args.recompute_gate
    assert not args.experimental_exact_token_lookup


def test_secondary_restore_contract_cannot_silently_change():
    from types import SimpleNamespace
    _, module = launcher()
    for saved in (False, True):
        module.validate_secondary_restore({'recompute_secondary':saved}, SimpleNamespace(recompute_secondary=saved))
        with pytest.raises(RuntimeError,match='secondary contract mismatch'):
            module.validate_secondary_restore({'recompute_secondary':saved}, SimpleNamespace(recompute_secondary=not saved))
    with pytest.raises(RuntimeError,match='secondary contract mismatch'):
        module.validate_secondary_restore({}, SimpleNamespace(recompute_secondary=True))


def test_rank_dependent_recompute_rejection_stops_every_rank(monkeypatch):
    import torch
    _,module=launcher()
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda value,op:value.zero_())
    result=module.verify_recompute_ranks({'within_gate':True,'primary_within_gate':True},torch.device('cpu'))
    assert not result['within_gate'] and not result['all_ranks_accepted']
    assert result['primary_within_gate']  # local evidence is never relabeled


def test_parent_worker_execution_identity_includes_lookup_backend():
    _,module=launcher();args=module.parse_args([])
    expected=module.execution_options(args)
    assert expected['token_lookup_backend']=='gather'
    assert module.execution_options(args,lr_scale=args.experimental_lr_scale)==expected
    args.experimental_exact_token_lookup=True
    assert module.execution_options(args)['token_lookup_backend']=='exact-row-v100-v1'


def test_isolated_validation_storage_cannot_target_production(tmp_path):
    _,module=launcher()
    with pytest.raises(SystemExit):module.parse_args(['--isolated-validation-storage'])
    with pytest.raises(SystemExit):module.parse_args(['--isolated-validation-storage','--experimental-resume',str(tmp_path),'--rolling-checkpoint-root',str(tmp_path),'--experimental-checkpoint',str(tmp_path/'boundary'),'--team-stats-root',str(tmp_path/'stats')])


def test_lookup_profile_is_explicit_and_rejects_unapproved_override(tmp_path, monkeypatch):
    from agent.runtime.config import resolve_profile, launcher_values
    root,module=launcher();monkeypatch.chdir(root)
    profile=json.loads((root/'configs/pa3_v2_production.yaml').read_text())
    assert not launcher_values(resolve_profile(str(root/'configs/pa3_v2_production.yaml')))['experimental_exact_token_lookup']
    paths=['--team-stats-root',str(tmp_path/'stats'),'--rolling-checkpoint-root',str(tmp_path/'rolling'),'--experimental-checkpoint',str(tmp_path/'boundary')]
    assert not module.parse_args(['--profile','configs/pa3_v2_production.yaml','--iterations','10']+paths).experimental_exact_token_lookup
    profile['collector']['token_lookup_backend']='exact-row-v100-v1'
    approved=tmp_path/'approved.json';approved.write_text(json.dumps(profile))
    assert launcher_values(resolve_profile(str(approved)))['experimental_exact_token_lookup']
    profile['collector']['token_lookup_backend']='unknown'
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='unsupported token lookup'):resolve_profile(str(path))
    profile['collector']['token_lookup_backend']='gather';path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='not approved'):module.parse_args(['--profile',str(path),'--experimental-exact-token-lookup']+paths)
